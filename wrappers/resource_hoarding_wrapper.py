import gym
from gym import spaces
from typing import Any, Optional, Tuple

import numpy as np

from .speed_wrapper import _reshape_grid
from .aggression_wrapper import CHERRY, _cherry_cell, _pacman_cell, _reset_obs


class R1ResourceHoardingWrapper(gym.Wrapper):
    """
    Resource hoarding behaviour R1 - Average Time For Pac-Man to Eat Cherry.

    Rewards collecting a newly appeared cherry quickly. Steps are counted from
    the moment a cherry (grid cell value 10) appears; when it disappears with
    Pacman within eat_distance of its last cell it counts as eaten, and a
    bonus is added that scales inversely with the steps it was out (full bonus
    at or under ref_steps, shrinking beyond it), the same scaling pattern as
    Sp1SpeedWrapper. A cherry that vanishes with Pacman far away is treated as
    a timer despawn and pays nothing. Every cherry collected in the episode
    pays its own bonus.

    Args:
        env: The environment to wrap
        cherry_bonus: Maximum bonus granted for a fast collection (default: 100.0)
        ref_steps: Steps on screen at or under which the full bonus is granted
            (default: 20)
        eat_distance: Manhattan distance from Pacman to the cherry's last cell
            for a disappearance to count as eaten (default: 2.0)
    """

    def __init__(self, env: gym.Env, cherry_bonus: float = 100.0,
                 ref_steps: int = 20, eat_distance: float = 2.0):
        super().__init__(env)
        self.cherry_bonus = cherry_bonus
        self.ref_steps = max(1, ref_steps)
        self.eat_distance = eat_distance

        self._cherry: Optional[Tuple[int, int]] = None
        self._cherry_steps = 0

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        obs = _reset_obs(data)
        self._cherry = _cherry_cell(_reshape_grid(obs))
        self._cherry_steps = 0
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        grid = _reshape_grid(obs)
        cherry = _cherry_cell(grid)

        if cherry is not None:
            # Count steps the cherry has been out; a fresh cherry restarts at 1.
            self._cherry_steps = self._cherry_steps + 1 if self._cherry else 1
        elif self._cherry is not None:
            # Cherry vanished: eaten if Pacman ends the step beside its cell,
            # otherwise it despawned on its timer.
            cell = _pacman_cell(grid)
            if cell is not None:
                distance = abs(cell[0] - self._cherry[0]) + abs(cell[1] - self._cherry[1])
                if distance <= self.eat_distance:
                    steps = max(self._cherry_steps, self.ref_steps)
                    reward += self.cherry_bonus * (self.ref_steps / steps)

        self._cherry = cherry
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
