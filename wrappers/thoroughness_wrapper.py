import gym
from gym import spaces
from typing import Any, Optional

import numpy as np

from .speed_wrapper import (
    _reshape_grid,
    _quadrant_pellet_counts,
    _pacman_quadrant,
    PELLET,
    POWER_PELLET,
)


def _count_singleton_pellets(grid: np.ndarray) -> int:
    """
    Count singleton pellets: a (power) pellet with no orthogonally adjacent pellet.

    Args:
        grid: The (height, width) game grid.

    Returns:
        The number of isolated pellets on the board.
    """
    m = (grid == PELLET) | (grid == POWER_PELLET)

    # Mark cells that have at least one pellet neighbour (up / down / left / right).
    nbr = np.zeros_like(m)
    nbr[:-1, :] |= m[1:, :]
    nbr[1:, :] |= m[:-1, :]
    nbr[:, :-1] |= m[:, 1:]
    nbr[:, 1:] |= m[:, :-1]

    singletons = m & ~nbr
    return int(np.count_nonzero(singletons))


def _reset_obs(data: Any) -> Any:
    """Extract the observation from a Gym (obs) or Gymnasium (obs, info) reset return."""
    if isinstance(data, tuple) and len(data) == 2:
        return data[0]
    return data


class T1ThoroughnessWrapper(gym.Wrapper):
    """
    Thoroughness behaviour T1 - Sector by Sector.

    Encourages clearing one 2x2 maze quadrant at a time. Two toggleable signals:
      - penalise_leaving: penalty when Pacman enters a different quadrant while its
        previous quadrant still has pills.
      - reward_staying: bonus for each step Pacman stays in its current quadrant
        while that quadrant still has pills.

    Args:
        env: The environment to wrap
        leave_penalty: Penalty for leaving an unfinished sector (default: 10.0)
        stay_bonus: Bonus per step spent finishing the current sector (default: 2.0)
        use_penalise_leaving: Enable the leaving penalty (default: True)
        use_reward_staying: Enable the staying bonus (default: True)
    """

    def __init__(self, env: gym.Env, leave_penalty: float = 10.0,
                 stay_bonus: float = 2.0, use_penalise_leaving: bool = True,
                 use_reward_staying: bool = True):
        super().__init__(env)
        self.leave_penalty = abs(leave_penalty)
        self.stay_bonus = stay_bonus
        self.use_penalise_leaving = use_penalise_leaving
        self.use_reward_staying = use_reward_staying

        self._prev_quadrant: Optional[int] = None

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        self._prev_quadrant = _pacman_quadrant(_reshape_grid(_reset_obs(data)))
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        grid = _reshape_grid(obs)
        cur = _pacman_quadrant(grid)
        counts = _quadrant_pellet_counts(grid)

        if cur is not None and self._prev_quadrant is not None:
            if (self.use_penalise_leaving and cur != self._prev_quadrant
                    and counts[self._prev_quadrant] > 0):
                reward -= self.leave_penalty
            if (self.use_reward_staying and cur == self._prev_quadrant
                    and counts[cur] > 0):
                reward += self.stay_bonus

        if cur is not None:
            self._prev_quadrant = cur
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


class T2ThoroughnessWrapper(gym.Wrapper):
    """
    Thoroughness behaviour T2 - Leaves a Single Pill.

    Discourages leaving isolated pills behind. Two toggleable signals:
      - use_count: per-step penalty proportional to the number of singleton pills
        currently on the board.
      - use_on_create: penalty applied when the singleton count rises (the act of
        leaving a pill isolated).

    Args:
        env: The environment to wrap
        weight: Per-step penalty per singleton pill (default: 2.0)
        create_penalty: Penalty per newly created singleton pill (default: 10.0)
        use_count: Enable the per-step count penalty (default: True)
        use_on_create: Enable the singleton-created penalty (default: True)
    """

    def __init__(self, env: gym.Env, weight: float = 2.0,
                 create_penalty: float = 10.0, use_count: bool = True,
                 use_on_create: bool = True):
        super().__init__(env)
        self.weight = abs(weight)
        self.create_penalty = abs(create_penalty)
        self.use_count = use_count
        self.use_on_create = use_on_create

        self._prev_singletons = 0

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        self._prev_singletons = _count_singleton_pellets(_reshape_grid(_reset_obs(data)))
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        singletons = _count_singleton_pellets(_reshape_grid(obs))

        if self.use_count:
            reward -= self.weight * singletons
        if self.use_on_create:
            created = singletons - self._prev_singletons
            if created > 0:
                reward -= self.create_penalty * created

        self._prev_singletons = singletons
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
