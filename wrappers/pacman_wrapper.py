import gym
from gym import spaces
from typing import Any, Dict, Tuple
from .frameskip import FrameSkipWrapper
def wrap_env(env: Any, skip: int = 4) -> gym.Env:
    """
    Wrap an environment with frame skipping.

    Args:
        env: The environment to wrap (custom or standard)
        skip: Number of frames to skip

    Returns:
        The wrapped environment

    Example:
        custom_env = MyCustomEnv()
        wrapped = make_frameskip_env(custom_env, skip=4)
    """
    if skip is not None:
        return FrameSkipWrapper(env, skip=skip)
    else:
        return env