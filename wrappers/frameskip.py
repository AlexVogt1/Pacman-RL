import gym
from gym import spaces
from typing import Any, Dict, Tuple

class FrameSkipWrapper(gym.Wrapper):
    """
    Wrapper that repeats the same action for multiple frames.

    This wrapper:
    - Repeats the given action for 'skip' frames
    - Sums all rewards from the skipped frames
    - Returns the last observation

    Args:
        env: The environment to wrap
        skip: Number of frames to repeat each action (default: 4)
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)

        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")

        self.skip = skip

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Repeat action for skip frames and sum rewards.

        Args:
            action: The action to repeat

        Returns:
            observation: The last observation
            reward: Sum of all rewards from skipped frames
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Information dictionary (from last step)
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None

        # Execute action for 'skip' frames
        for i in range(self.skip):
            obs, reward, terminated, info = self.env.step(action)
            total_reward += reward

            # Stop early if episode ends
            if terminated:
                break

        return obs, total_reward, terminated, info