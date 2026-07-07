import gym
from gym import spaces
from typing import Any, Dict, Tuple


class NormaliseRewardWrapper(gym.Wrapper):
    def step(self, action):
        # Get the data from Unity
        data = self.env.step(action)

        # Check if we got 4 values (Unity/Old Gym) or 5 values (Gymnasium) (future-proofing)
        if len(data) == 4:
            obs, reward, done, info = data
            # Scale the reward
            reward = reward / 1000.0

            return obs, reward, done, info

        elif len(data) == 5:
            obs, reward, terminated, truncated, info = data

            # Scale the reward
            reward = reward / 1000.0

            return obs, reward, terminated, truncated, info

class StepRewardWrapper(gym.Wrapper):
    def __init__(self, env, step_value=0.0):
        super().__init__(env)
        self.step_value = step_value

    def step(self, action):
        data = self.env.step(action)

        # 4 values: Unity / old Gym API
        if len(data) == 4:
            obs, reward, done, info = data
            reward = reward + (-abs(self.step_value))
            return obs, reward, done, info

        # 5 values: Gymnasium API
        elif len(data) == 5:
            obs, reward, terminated, truncated, info = data
            reward = reward + (-abs(self.step_value))
            return obs, reward, terminated, truncated, info