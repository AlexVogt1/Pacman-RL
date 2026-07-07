import gym
from gym import spaces
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Observation layout (see wrappers/README.md):
# 34 fixed values (indices 0-33), then the flattened game grid.
GRID_OFFSET = 34
GRID_WIDTH = 26
GRID_HEIGHT = 29
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT

# Grid cell values that must be cleared.
PELLET = 1
POWER_PELLET = 2
# Grid cell values for Pacman.
PACMAN_NORMAL = 3
PACMAN_ATTACK = 4


def _reshape_grid(obs: Any) -> np.ndarray:
    """
    Extract the flattened game grid from an observation and reshape it.

    Args:
        obs: The flattened ML-Agents observation vector.

    Returns:
        The game grid as a (height, width) array, orientation (29, 26).
    """
    flat = np.asarray(obs)[GRID_OFFSET:GRID_OFFSET + GRID_SIZE]
    return flat.reshape(GRID_HEIGHT, GRID_WIDTH)


def _quadrant_slices() -> List[Tuple[slice, slice]]:
    """
    Build the (row, col) slices for the four 2x2 grid quadrants.

    Returns:
        Quadrant slices ordered: top-left, top-right, bottom-left, bottom-right.
    """
    mid_row = GRID_HEIGHT // 2
    mid_col = GRID_WIDTH // 2
    return [
        (slice(0, mid_row), slice(0, mid_col)),
        (slice(0, mid_row), slice(mid_col, GRID_WIDTH)),
        (slice(mid_row, GRID_HEIGHT), slice(0, mid_col)),
        (slice(mid_row, GRID_HEIGHT), slice(mid_col, GRID_WIDTH)),
    ]


def _quadrant_pellet_counts(grid: np.ndarray) -> List[int]:
    """
    Count remaining (power) pellets in each of the four grid quadrants.

    Args:
        grid: The (height, width) game grid.

    Returns:
        Pellet counts per quadrant, ordered TL, TR, BL, BR.
    """
    counts = []
    for row_slice, col_slice in _quadrant_slices():
        sector = grid[row_slice, col_slice]
        counts.append(int(np.count_nonzero((sector == PELLET) | (sector == POWER_PELLET))))
    return counts


def _pacman_quadrant(grid: np.ndarray) -> Optional[int]:
    """
    Locate Pacman in the grid and map the cell to a quadrant index.

    Args:
        grid: The (height, width) game grid.

    Returns:
        The quadrant index (0-3), or None if Pacman is not found.
    """
    rows, cols = np.where((grid == PACMAN_NORMAL) | (grid == PACMAN_ATTACK))
    if len(rows) == 0:
        return None

    row, col = int(rows[0]), int(cols[0])
    mid_row = GRID_HEIGHT // 2
    mid_col = GRID_WIDTH // 2
    top = row < mid_row
    left = col < mid_col
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and left:
        return 2
    return 3


class Sp1SpeedWrapper(gym.Wrapper):
    """
    Speed behaviour Sp1 - Average Cycles Per Sector.

    Rewards the agent for clearing each maze sector (a 2x2 grid quadrant) in fewer
    cycles. Applies a small per-step penalty to pressure faster play, and a bonus
    when a sector is fully cleared that is scaled inversely to the cycles spent in
    that sector (faster clear gives a larger bonus, capped at clear_bonus).

    Args:
        env: The environment to wrap
        step_penalty: Reward subtracted on every step (default: 2.0)
        clear_bonus: Maximum bonus granted when a sector is cleared (default: 200.0)
        ref_cycles: Reference cycle count for full bonus scaling (default: 50)
    """

    def __init__(self, env: gym.Env, step_penalty: float = 2.0,
                 clear_bonus: float = 200.0, ref_cycles: int = 50):
        super().__init__(env)
        self.step_penalty = abs(step_penalty)
        self.clear_bonus = clear_bonus
        self.ref_cycles = max(1, ref_cycles)

        self._quad_remaining: List[int] = [0, 0, 0, 0]
        self._quad_cycles: List[int] = [0, 0, 0, 0]
        self._cleared: List[bool] = [False, False, False, False]

    def _init_sector_state(self, obs: Any) -> None:
        """Initialise per-quadrant pellet, cycle and cleared tracking from an observation."""
        grid = _reshape_grid(obs)
        self._quad_remaining = _quadrant_pellet_counts(grid)
        self._quad_cycles = [0, 0, 0, 0]
        # A quadrant that starts with no pellets is treated as already cleared.
        self._cleared = [count == 0 for count in self._quad_remaining]

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)

        # Gymnasium reset returns (obs, info); old Gym returns obs.
        if isinstance(data, tuple) and len(data) == 2:
            obs, _info = data
            self._init_sector_state(obs)
            return data

        self._init_sector_state(data)
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        """Apply the per-step penalty and any sector-clear bonus."""
        grid = _reshape_grid(obs)

        # Count this step as a cycle for the quadrant Pacman currently occupies.
        quadrant = _pacman_quadrant(grid)
        if quadrant is not None:
            self._quad_cycles[quadrant] += 1

        reward -= self.step_penalty

        # Award a speed-scaled bonus for any quadrant that has just been cleared.
        counts = _quadrant_pellet_counts(grid)
        for i, count in enumerate(counts):
            if count == 0 and not self._cleared[i]:
                cycles = max(self._quad_cycles[i], self.ref_cycles)
                reward += self.clear_bonus * (self.ref_cycles / cycles)
                self._cleared[i] = True
        self._quad_remaining = counts

        return reward

    def step(self, action: Any):
        data = self.env.step(action)

        # 4 values: Unity / old Gym API
        if len(data) == 4:
            obs, reward, done, info = data
            reward = self._shape_reward(obs, reward)
            return obs, reward, done, info

        # 5 values: Gymnasium API
        elif len(data) == 5:
            obs, reward, terminated, truncated, info = data
            reward = self._shape_reward(obs, reward)
            return obs, reward, terminated, truncated, info


class Sp2SpeedWrapper(gym.Wrapper):
    """
    Speed behaviour Sp2 - Average States.

    Rewards the agent for clearing a level in fewer moves by applying a small
    per-step penalty on every step. Completion itself is already rewarded by the
    existing +1000 level-clear signal, so this wrapper only adds move pressure.

    Args:
        env: The environment to wrap
        step_penalty: Reward subtracted on every step (default: 2.0)
    """

    def __init__(self, env: gym.Env, step_penalty: float = 2.0):
        super().__init__(env)
        self.step_penalty = abs(step_penalty)

    def step(self, action: Any):
        data = self.env.step(action)

        # 4 values: Unity / old Gym API
        if len(data) == 4:
            obs, reward, done, info = data
            reward = reward - self.step_penalty
            return obs, reward, done, info

        # 5 values: Gymnasium API
        elif len(data) == 5:
            obs, reward, terminated, truncated, info = data
            reward = reward - self.step_penalty
            return obs, reward, terminated, truncated, info
