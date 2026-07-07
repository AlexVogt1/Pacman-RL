import gym
from gym import spaces
from typing import Any, Tuple

import numpy as np

from .speed_wrapper import _reshape_grid, PACMAN_NORMAL, PACMAN_ATTACK

# Observation layout (see wrappers/README.md):
ATTACK_STATE_INDEX = 3          # Pacman attack state (bool): 1 = on a pill / hunt mode
GHOST_DISTANCE_SLICE = slice(30, 34)  # Ghost distances array (float[4], Manhattan)

# Grid is 26 wide x 29 tall, so the largest possible Manhattan distance is
# (26 - 1) + (29 - 1) = 53. Used to normalise the distance reward to [0, 1].
MAX_MANHATTAN = (26 - 1) + (29 - 1)

# Grid cell values for ghosts that can actually trap / kill Pacman:
# 5 = home, 6 = scatter, 7 = chase. Frightened (8) and eaten (9) are excluded.
DANGEROUS_GHOST_MIN = 5
DANGEROUS_GHOST_MAX = 7


def _avg_ghost_distance(obs: Any) -> float:
    """
    Average normalised Manhattan distance from Pacman to the four ghosts.

    Args:
        obs: The flattened ML-Agents observation vector.

    Returns:
        The mean ghost distance scaled to [0, 1].
    """
    distances = np.asarray(obs)[GHOST_DISTANCE_SLICE]
    return float(np.mean(distances)) / MAX_MANHATTAN


def _is_attacking(obs: Any) -> bool:
    """Return True when Pacman is on a pill / in hunt mode."""
    return bool(np.asarray(obs)[ATTACK_STATE_INDEX] >= 0.5)


def _proximity_trapped(obs: Any, trap_distance: float, min_ghosts: int) -> bool:
    """
    Detect trapping via proximity: several ghosts close to Pacman at once.

    Args:
        obs: The flattened observation vector.
        trap_distance: Manhattan distance under which a ghost counts as closing in.
        min_ghosts: Number of close ghosts required to count as trapped.

    Returns:
        True if at least min_ghosts ghosts are within trap_distance.
    """
    distances = np.asarray(obs)[GHOST_DISTANCE_SLICE]
    return int(np.count_nonzero(distances <= trap_distance)) >= min_ghosts


def _directional_trapped(obs: Any) -> bool:
    """
    Detect corridor trapping via the grid: a ghost on opposite sides of Pacman.

    Trapped when a dangerous ghost sits both left and right of Pacman on the same
    row, or both above and below on the same column.

    Args:
        obs: The flattened observation vector.

    Returns:
        True if Pacman is boxed in on opposite sides.
    """
    grid = _reshape_grid(obs)

    pac_rows, pac_cols = np.where((grid == PACMAN_NORMAL) | (grid == PACMAN_ATTACK))
    if len(pac_rows) == 0:
        return False
    pr, pc = int(pac_rows[0]), int(pac_cols[0])

    ghost_rows, ghost_cols = np.where(
        (grid >= DANGEROUS_GHOST_MIN) & (grid <= DANGEROUS_GHOST_MAX)
    )

    same_row = ghost_rows == pr
    left = np.any(ghost_cols[same_row] < pc)
    right = np.any(ghost_cols[same_row] > pc)

    same_col = ghost_cols == pc
    up = np.any(ghost_rows[same_col] < pr)
    down = np.any(ghost_rows[same_col] > pr)

    return bool((left and right) or (up and down))


def _is_trapped(obs: Any, use_proximity: bool, use_directional: bool,
                trap_distance: float, min_ghosts: int) -> bool:
    """Return True if any enabled detector flags Pacman as trapped."""
    if use_proximity and _proximity_trapped(obs, trap_distance, min_ghosts):
        return True
    if use_directional and _directional_trapped(obs):
        return True
    return False


class C2aCautionWrapper(gym.Wrapper):
    """
    Caution behaviour C2.a - Average Distance to Ghosts.

    Rewards the agent for keeping its distance from ghosts while NOT on a pill.
    Adds a small per-step reward proportional to the average (normalised) ghost
    distance, applied only when Pacman is not in attack mode.

    Args:
        env: The environment to wrap
        weight: Maximum per-step distance reward (default: 5.0)
    """

    def __init__(self, env: gym.Env, weight: float = 5.0):
        super().__init__(env)
        self.weight = weight

    def _shape_reward(self, obs: Any, reward: float) -> float:
        if not _is_attacking(obs):
            reward += self.weight * _avg_ghost_distance(obs)
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


class C2bCautionWrapper(gym.Wrapper):
    """
    Caution behaviour C2.b - Average Distance During Hunt.

    Rewards the agent for keeping its distance from ghosts while in hunt mode
    (on a pill). Adds a small per-step reward proportional to the average
    (normalised) ghost distance, applied only when Pacman is in attack mode.

    Args:
        env: The environment to wrap
        weight: Maximum per-step distance reward (default: 5.0)
    """

    def __init__(self, env: gym.Env, weight: float = 5.0):
        super().__init__(env)
        self.weight = weight

    def _shape_reward(self, obs: Any, reward: float) -> float:
        if _is_attacking(obs):
            reward += self.weight * _avg_ghost_distance(obs)
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


class C1aCautionWrapper(gym.Wrapper):
    """
    Caution behaviour C1.a - Times Trapped By Ghosts.

    Penalises being trapped in a corridor by ghosts. A trap is flagged by either
    detector (toggle independently):
      - proximity: several ghosts within a small distance at once
      - directional: a dangerous ghost on opposite sides of Pacman in the grid
    Applies a per-step penalty for every step the trapped condition holds.

    Args:
        env: The environment to wrap
        trap_penalty: Reward subtracted on each trapped step (default: 10.0)
        use_proximity: Enable the proximity detector (default: True)
        use_directional: Enable the directional detector (default: True)
        trap_distance: Proximity distance threshold, Manhattan (default: 3.0)
        min_ghosts: Ghosts within range needed for a proximity trap (default: 2)
    """

    def __init__(self, env: gym.Env, trap_penalty: float = 10.0,
                 use_proximity: bool = True, use_directional: bool = True,
                 trap_distance: float = 3.0, min_ghosts: int = 2):
        super().__init__(env)
        self.trap_penalty = abs(trap_penalty)
        self.use_proximity = use_proximity
        self.use_directional = use_directional
        self.trap_distance = trap_distance
        self.min_ghosts = min_ghosts

    def _shape_reward(self, obs: Any, reward: float) -> float:
        if _is_trapped(obs, self.use_proximity, self.use_directional,
                       self.trap_distance, self.min_ghosts):
            reward -= self.trap_penalty
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


class C1bCautionWrapper(gym.Wrapper):
    """
    Caution behaviour C1.b - Times Trapped and Killed By Ghosts.

    Adds an extra death penalty when the episode ends (Pacman caught) while the
    trapped condition holds, on top of the environment's existing death reward.
    Uses the same trap detectors as C1a; the death is detected from the episode
    termination flag, so no life tracking is needed.

    Args:
        env: The environment to wrap
        death_penalty: Extra reward subtracted on a trapped death (default: 200.0)
        use_proximity: Enable the proximity detector (default: True)
        use_directional: Enable the directional detector (default: True)
        trap_distance: Proximity distance threshold, Manhattan (default: 3.0)
        min_ghosts: Ghosts within range needed for a proximity trap (default: 2)
    """

    def __init__(self, env: gym.Env, death_penalty: float = 200.0,
                 use_proximity: bool = True, use_directional: bool = True,
                 trap_distance: float = 3.0, min_ghosts: int = 2):
        super().__init__(env)
        self.death_penalty = abs(death_penalty)
        self.use_proximity = use_proximity
        self.use_directional = use_directional
        self.trap_distance = trap_distance
        self.min_ghosts = min_ghosts

    def _shape_reward(self, obs: Any, reward: float, done: bool) -> float:
        if done and _is_trapped(obs, self.use_proximity, self.use_directional,
                                self.trap_distance, self.min_ghosts):
            reward -= self.death_penalty
        return reward

    def step(self, action: Any):
        data = self.env.step(action)

        # 4 values: Unity / old Gym API
        if len(data) == 4:
            obs, reward, done, info = data
            reward = self._shape_reward(obs, reward, done)
            return obs, reward, done, info

        # 5 values: Gymnasium API
        elif len(data) == 5:
            obs, reward, terminated, truncated, info = data
            reward = self._shape_reward(obs, reward, terminated)
            return obs, reward, terminated, truncated, info
