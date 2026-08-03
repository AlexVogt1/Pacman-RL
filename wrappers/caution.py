import numpy as np
import gym
from gym import Wrapper


# ─────────────────────────────────────────────
#  Re-use obs indices from reward_shaping.py
# ─────────────────────────────────────────────
PACMAN_ATTACK   = 3
GHOST_OFFSET    = 7
GHOST_STRIDE    = 3
N_GHOSTS        = 4
GHOST_DISTANCES = [30, 31, 32, 33]

def ghost_state(i): return GHOST_OFFSET + i * GHOST_STRIDE + 2


class GhostState:
    HOME       = 0
    SCATTER    = 1
    CHASE      = 2
    FRIGHTENED = 3
    EATEN      = 4


# ─────────────────────────────────────────────
#  Caution config
# ─────────────────────────────────────────────
class CautionConfig:
    # Distance thresholds
    DANGER_RADIUS = 3.0   # within this → strong penalty
    CAUTION_RADIUS = 7.0  # within this → mild penalty
    SAFE_RADIUS = 10.0    # beyond this → small reward for maintaining distance

    # Reward/penalty magnitudes
    DANGER_PENALTY  = -3.0   # per ghost inside DANGER_RADIUS
    CAUTION_PENALTY = -1.0   # per ghost inside CAUTION_RADIUS
    SAFE_BONUS      = 0.5    # per ghost beyond SAFE_RADIUS

    # Whether to scale penalty by how threatening the ghost state is
    # chase > scatter > home in terms of threat
    SCALE_BY_STATE  = True

    GAMMA = 0.99


# ─────────────────────────────────────────────
#  State threat multipliers
# ─────────────────────────────────────────────
STATE_THREAT = {
    GhostState.CHASE:      1.0,   # most threatening
    GhostState.SCATTER:    0.6,   # moving away but still dangerous
    GhostState.HOME:       0.1,   # not on the board yet
    GhostState.FRIGHTENED: 0.0,   # not a threat — pacman is hunting
    GhostState.EATEN:      0.0,   # no threat
}


# ─────────────────────────────────────────────
#  Obs parsing (caution-specific)
# ─────────────────────────────────────────────
def parse_caution_obs(obs):
    """
    Extract only what caution shaping needs from the flat obs.
    Returns pacman_attack flag and per-ghost (dist, state, threat).
    """
    pacman_attack = bool(obs[PACMAN_ATTACK])

    ghosts = []
    for i in range(N_GHOSTS):
        state = int(obs[ghost_state(i)])
        dist  = float(obs[GHOST_DISTANCES[i]])
        ghosts.append({
            "state":  state,
            "dist":   dist,
            "threat": STATE_THREAT.get(state, 0.0),
        })

    return {
        "pacman_attack": pacman_attack,
        "ghosts":        ghosts,
    }


# ─────────────────────────────────────────────
#  Caution potential  Φ_caution(s)
# ─────────────────────────────────────────────
def caution_potential(state, cfg):
    """
    Φ_caution(s): encodes how safe pacman's current position is
    relative to all non-frightened ghosts.

    Only active when pacman is NOT in attack mode.
    Three zones per ghost:
      - Beyond SAFE_RADIUS    → small positive (reward safe distance)
      - Inside CAUTION_RADIUS → mild negative (nudge away)
      - Inside DANGER_RADIUS  → strong negative (urgent avoidance)

    Each ghost's contribution is scaled by its threat level if
    SCALE_BY_STATE is enabled — chase ghosts matter more than
    scatter or home ghosts.
    """
    if state["pacman_attack"]:
        return 0.0  # caution suspended during attack/hunt mode

    phi = 0.0
    for ghost in state["ghosts"]:
        dist   = ghost["dist"]
        threat = ghost["threat"] if cfg.SCALE_BY_STATE else 1.0

        if threat == 0.0:
            continue  # frightened or eaten — not a threat

        if dist <= cfg.DANGER_RADIUS:
            # Steep penalty — scales inversely with distance so
            # getting closer makes it sharply worse
            phi += cfg.DANGER_PENALTY * threat * (1.0 / (1.0 + dist))

        elif dist <= cfg.CAUTION_RADIUS:
            # Mild penalty — linear falloff between caution and danger
            phi += cfg.CAUTION_PENALTY * threat * (1.0 / (1.0 + dist))

        elif dist >= cfg.SAFE_RADIUS:
            # Small reward for maintaining a safe gap
            phi += cfg.SAFE_BONUS * threat * (dist / (cfg.SAFE_RADIUS + dist))

    return phi


# ─────────────────────────────────────────────
#  Caution shaped reward
# ─────────────────────────────────────────────
def caution_shaped_reward(obs, next_obs, base_reward, cfg=None):
    """
    Potential-based caution shaping: r' = r + γΦ(s') - Φ(s)

    Suspended entirely when pacman is in attack mode so it does
    not conflict with ghost hunting behaviour.

    Args:
        obs:         flat obs array before step
        next_obs:    flat obs array after step
        base_reward: reward from Unity environment
        cfg:         CautionConfig (uses defaults if None)

    Returns:
        float: shaped reward
    """
    if cfg is None:
        cfg = CautionConfig()

    state      = parse_caution_obs(obs)
    next_state = parse_caution_obs(next_obs)

    phi_s      = caution_potential(state, cfg)
    phi_s_next = caution_potential(next_state, cfg)
    shaping    = cfg.GAMMA * phi_s_next - phi_s

    return base_reward + shaping


# ─────────────────────────────────────────────
#  Gymnasium Wrapper
# ─────────────────────────────────────────────
class CautionWrapper(Wrapper):
    """
    Reward shaping wrapper that encourages pacman to maintain
    distance from ghosts when NOT in attack mode.

    Can be used standalone or stacked on top of PacmanRewardShapingWrapper:

    STANDALONE:
        env = CautionWrapper(env)

    STACKED (caution + ghost hunting):
        env = PacmanRewardShapingWrapper(env)   # ghost hunting
        env = CautionWrapper(env)               # caution on top

    The caution wrapper automatically suspends when pacman enters
    attack mode (obs[3] == 1) so hunting behaviour is never penalised.
    """

    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.cfg       = cfg or CautionConfig()
        self._last_obs = None

        self._episode_base_reward    = 0.0
        self._episode_caution_reward = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_obs               = obs.copy()
        self._episode_base_reward    = 0.0
        self._episode_caution_reward = 0.0
        return obs

    def step(self, action):
        next_obs, base_reward, done, info = self.env.step(action)

        total_reward = caution_shaped_reward(
            obs         = self._last_obs,
            next_obs    = next_obs,
            base_reward = base_reward,
            cfg         = self.cfg,
        )

        self._episode_base_reward    += base_reward
        self._episode_caution_reward += total_reward - base_reward

        info["base_reward"]    = base_reward
        info["caution_bonus"]  = total_reward - base_reward

        if done:
            info["episode_base_reward"]    = self._episode_base_reward
            info["episode_caution_reward"] = self._episode_caution_reward

        self._last_obs = next_obs.copy()
        return next_obs, total_reward, done, info
