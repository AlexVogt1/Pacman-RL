import gym
from gym import spaces
from typing import Any, Optional, Tuple

import numpy as np

from .speed_wrapper import (
    _reshape_grid,
    PACMAN_NORMAL,
    PACMAN_ATTACK,
    PELLET,
    POWER_PELLET,
    GRID_HEIGHT,
    GRID_WIDTH,
)

# Observation layout (see wrappers/README.md):
ATTACK_STATE_INDEX = 3   # Pacman attack state (bool): 1 = on a pill / hunt mode
GHOST_DISTANCE_SLICE = slice(30, 34)  # Ghost distances array (float[4], Manhattan)

# Largest possible Manhattan distance on the 26x29 grid, used to normalise distances.
MAX_MANHATTAN = (GRID_WIDTH - 1) + (GRID_HEIGHT - 1)

# Grid cell range for ghosts (5 = home ... 9 = eaten) and the eaten value.
GHOST_MIN = 5
GHOST_MAX = 9
GHOST_EATEN = 9


def _is_attacking(obs: Any) -> bool:
    """Return True when Pacman is on a pill / in hunt mode."""
    return bool(np.asarray(obs)[ATTACK_STATE_INDEX] >= 0.5)


def _ghost_closeness(obs: Any) -> float:
    """Average closeness to ghosts in [0, 1] (1 = on top of them, 0 = far)."""
    distances = np.asarray(obs)[GHOST_DISTANCE_SLICE]
    return 1.0 - (float(np.mean(distances)) / MAX_MANHATTAN)


def _count_pellets(grid: np.ndarray) -> int:
    """Count remaining (power) pellets on the board."""
    return int(np.count_nonzero((grid == PELLET) | (grid == POWER_PELLET)))


def _ghost_house_centroid(grid: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Locate the ghost house as the centroid of all ghost cells.

    Intended for use at reset, when the ghosts are clustered in their pen.

    Args:
        grid: The (height, width) game grid.

    Returns:
        The (row, col) centroid of ghost cells, or None if no ghost is present.
    """
    rows, cols = np.where((grid >= GHOST_MIN) & (grid <= GHOST_MAX))
    if len(rows) == 0:
        return None
    return float(np.mean(rows)), float(np.mean(cols))


def _pacman_cell(grid: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return Pacman's (row, col) in the grid, or None if not found."""
    rows, cols = np.where((grid == PACMAN_NORMAL) | (grid == PACMAN_ATTACK))
    if len(rows) == 0:
        return None
    return int(rows[0]), int(cols[0])


def _count_eaten_ghosts(grid: np.ndarray) -> int:
    """Count ghosts currently in the eaten state (eyes returning home)."""
    return int(np.count_nonzero(grid == GHOST_EATEN))


def _reset_obs(data: Any) -> Any:
    """Extract the observation from a Gym (obs) or Gymnasium (obs, info) reset return."""
    if isinstance(data, tuple) and len(data) == 2:
        return data[0]
    return data


class A1AggressionWrapper(gym.Wrapper):
    """
    Aggression behaviour A1 - Hunt Close To Ghost House.

    Rewards chasing ghosts up to their house while attacking. The ghost house is
    located once at reset as the centroid of the ghost cells (the pen). Each step
    Pacman is in attack mode and within house_distance of that centroid, a bonus
    is added.

    Args:
        env: The environment to wrap
        bonus: Reward added per qualifying step (default: 5.0)
        house_distance: Manhattan distance to the house centroid that counts as
            "close" (default: 5.0)
    """

    def __init__(self, env: gym.Env, bonus: float = 5.0, house_distance: float = 5.0):
        super().__init__(env)
        self.bonus = bonus
        self.house_distance = house_distance
        self._house: Optional[Tuple[float, float]] = None

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        self._house = _ghost_house_centroid(_reshape_grid(_reset_obs(data)))
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        if self._house is None or not _is_attacking(obs):
            return reward

        cell = _pacman_cell(_reshape_grid(obs))
        if cell is None:
            return reward

        distance = abs(cell[0] - self._house[0]) + abs(cell[1] - self._house[1])
        if distance <= self.house_distance:
            reward += self.bonus
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


class A3AggressionWrapper(gym.Wrapper):
    """
    Aggression behaviour A3 - Ghost Kills.

    Rewards eating ghosts while attacking, on top of the environment's existing
    ghost-eaten reward. A kill is detected as the number of eaten ghosts (grid
    cells with value 9) rising between steps.

    Args:
        env: The environment to wrap
        kill_bonus: Reward added per newly eaten ghost (default: 40.0)
    """

    def __init__(self, env: gym.Env, kill_bonus: float = 40.0):
        super().__init__(env)
        self.kill_bonus = kill_bonus
        self._prev_eaten = 0

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        self._prev_eaten = _count_eaten_ghosts(_reshape_grid(_reset_obs(data)))
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        eaten = _count_eaten_ghosts(_reshape_grid(obs))
        kills = eaten - self._prev_eaten
        if kills > 0:
            reward += self.kill_bonus * kills
        self._prev_eaten = eaten
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


class A6AggressionWrapper(gym.Wrapper):
    """
    Aggression behaviour A6 - Chase Ghosts or Collect Dots.

    Encourages using pills to chase ghosts rather than passively collecting dots.
    Two toggleable signals, both active only while Pacman is attacking:
      - reward_chasing: bonus proportional to closeness to ghosts (closer = more).
      - penalise_dots: penalty for each dot eaten while attacking.

    Args:
        env: The environment to wrap
        chase_weight: Max per-step chase bonus while attacking (default: 5.0)
        dot_penalty: Penalty per dot eaten while attacking (default: 5.0)
        reward_chasing: Enable the chase bonus (default: True)
        penalise_dots: Enable the dot penalty (default: True)
    """

    def __init__(self, env: gym.Env, chase_weight: float = 5.0,
                 dot_penalty: float = 5.0, reward_chasing: bool = True,
                 penalise_dots: bool = True):
        super().__init__(env)
        self.chase_weight = chase_weight
        self.dot_penalty = abs(dot_penalty)
        self.reward_chasing = reward_chasing
        self.penalise_dots = penalise_dots

        self._prev_pellets = 0

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        self._prev_pellets = _count_pellets(_reshape_grid(_reset_obs(data)))
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        pellets = _count_pellets(_reshape_grid(obs))

        if _is_attacking(obs):
            if self.reward_chasing:
                reward += self.chase_weight * _ghost_closeness(obs)
            if self.penalise_dots:
                dots_eaten = self._prev_pellets - pellets
                if dots_eaten > 0:
                    reward -= self.dot_penalty * dots_eaten

        self._prev_pellets = pellets
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
