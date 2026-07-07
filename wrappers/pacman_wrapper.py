import gym
from gym import spaces
from typing import Any, Dict, Tuple, Optional

from .frameskip import FrameSkipWrapper
from .RewardWrapper import NormaliseRewardWrapper, StepRewardWrapper
from .speed_wrapper import Sp1SpeedWrapper, Sp2SpeedWrapper
from .caution_wrapper import (
    C1aCautionWrapper,
    C1bCautionWrapper,
    C2aCautionWrapper,
    C2bCautionWrapper,
)
from .thoroughness_wrapper import T1ThoroughnessWrapper, T2ThoroughnessWrapper
from .aggression_wrapper import (
    A1AggressionWrapper,
    A3AggressionWrapper,
    A6AggressionWrapper,
)

def wrap_env(env: Any, skip: int = 4, wrap_reward: str = None,step_reward:Optional[float]=None, caution: Optional[bool]=None,
             aggression: Optional[bool]=None,
             planning: Optional[bool]=None, speed: Optional[bool]=None,
             cfg: Optional[Dict]=None) -> gym.Env:
    """
    Wrap an environment with frame skipping.

    Args:
        env: The environment to wrap (custom or standard)
        skip: Number of frames to skip
        wrap_reward: If specified, wrap reward function
        step_reward: If specified, step reward function
        caution: If specified, wrap reward function
        speed: If specified, enable speed behaviour wrappers
        cfg: Per-behaviour config; keys (e.g. 'Sp1', 'Sp2') enable a wrapper when
            truthy. A dict value supplies reward params, True uses the defaults.

    Returns:
        The wrapped environment

    """
    # 1) First check to add frameskip
    if skip is not None:
        env = FrameSkipWrapper(env, skip=skip)

    if step_reward is not None:
        env = StepRewardWrapper(env, step_value=step_reward)

    # Behaviour wrappers, driven by cfg keys. Applied before normalisation so their
    # rewards are scaled consistently with the rest of the reward structure.
    if cfg:
        if cfg.get('Sp1'):
            params = cfg['Sp1'] if isinstance(cfg['Sp1'], dict) else {}
            env = Sp1SpeedWrapper(env, **params)
        if cfg.get('Sp2'):
            params = cfg['Sp2'] if isinstance(cfg['Sp2'], dict) else {}
            env = Sp2SpeedWrapper(env, **params)
        if cfg.get('C1a'):
            params = cfg['C1a'] if isinstance(cfg['C1a'], dict) else {}
            env = C1aCautionWrapper(env, **params)
        if cfg.get('C1b'):
            params = cfg['C1b'] if isinstance(cfg['C1b'], dict) else {}
            env = C1bCautionWrapper(env, **params)
        if cfg.get('C2a'):
            params = cfg['C2a'] if isinstance(cfg['C2a'], dict) else {}
            env = C2aCautionWrapper(env, **params)
        if cfg.get('C2b'):
            params = cfg['C2b'] if isinstance(cfg['C2b'], dict) else {}
            env = C2bCautionWrapper(env, **params)
        if cfg.get('T1'):
            params = cfg['T1'] if isinstance(cfg['T1'], dict) else {}
            env = T1ThoroughnessWrapper(env, **params)
        if cfg.get('T2'):
            params = cfg['T2'] if isinstance(cfg['T2'], dict) else {}
            env = T2ThoroughnessWrapper(env, **params)
        if cfg.get('A1'):
            params = cfg['A1'] if isinstance(cfg['A1'], dict) else {}
            env = A1AggressionWrapper(env, **params)
        if cfg.get('A3'):
            params = cfg['A3'] if isinstance(cfg['A3'], dict) else {}
            env = A3AggressionWrapper(env, **params)
        if cfg.get('A6'):
            params = cfg['A6'] if isinstance(cfg['A6'], dict) else {}
            env = A6AggressionWrapper(env, **params)

    if wrap_reward == 'normalise':
        env = NormaliseRewardWrapper(env)

    return env