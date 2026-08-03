import numpy as np
import gym
from gym import Wrapper


# ─────────────────────────────────────────────
#  Obs indices (mirrors reward_shaping.py)
# ─────────────────────────────────────────────
PACMAN_ATTACK   = 3
GHOST_OFFSET    = 7
GHOST_STRIDE    = 3
N_GHOSTS        = 4
GHOST_DISTANCES = [30, 31, 32, 33]
REMAINING_PILLS = 26   # normalised; multiply by 4 for raw count

def ghost_state(i): return GHOST_OFFSET + i * GHOST_STRIDE + 2


class GhostState:
    HOME       = 0
    SCATTER    = 1
    CHASE      = 2
    FRIGHTENED = 3
    EATEN      = 4


# ─────────────────────────────────────────────
#  Aggression config
# ─────────────────────────────────────────────
class AggressionConfig:
    # Potential weight — reward closing distance to nearest frightened ghost
    CHASE_WEIGHT = 15.0

    # Event reward — each frightened ghost eaten during this attack window
    GHOST_EATEN_BONUS = 30.0

    # Event reward — bonus for clearing ALL frightened ghosts in one window
    FULL_CLEAR_BONUS = 100.0

    # Urgency: bonus for eating the first ghost quickly after activation
    # bonus = URGENCY_BONUS * max(0, URGENCY_WINDOW - steps_to_first_eat) / URGENCY_WINDOW
    URGENCY_BONUS  = 60.0
    URGENCY_WINDOW = 20    # steps — after this, urgency bonus drops to 0

    # Penalty for letting frightened phase expire with ghosts uneaten
    # Encourages commitment — once you activate, you should follow through
    EXPIRY_PENALTY_PER_GHOST = 25.0

    GAMMA = 0.99


# ─────────────────────────────────────────────
#  Obs parsing
# ─────────────────────────────────────────────
def parse_aggression_obs(obs):
    pacman_attack = bool(obs[PACMAN_ATTACK])

    ghosts = []
    for i in range(N_GHOSTS):
        state = int(obs[ghost_state(i)])
        ghosts.append({
            "state":      state,
            "dist":       float(obs[GHOST_DISTANCES[i]]),
            "frightened": state == GhostState.FRIGHTENED,
            "eaten":      state == GhostState.EATEN,
        })

    frightened_dists = [g["dist"] for g in ghosts if g["frightened"]]

    return {
        "pacman_attack":     pacman_attack,
        "ghosts":            ghosts,
        "any_frightened":    len(frightened_dists) > 0,
        "n_frightened":      len(frightened_dists),
        "nearest_frightened_dist": min(frightened_dists) if frightened_dists else None,
    }


# ─────────────────────────────────────────────
#  Aggression potential  Φ_aggression(s)
# ─────────────────────────────────────────────
def aggression_potential(state, cfg):
    """
    Φ_aggression: high when pacman is close to the nearest frightened ghost.
    Only active when at least one ghost is frightened.

    Targets only the nearest frightened ghost specifically — committing
    to a single target is more aggressive than vaguely chasing all of them.
    """
    if not state["any_frightened"]:
        return 0.0

    nearest = state["nearest_frightened_dist"]
    return cfg.CHASE_WEIGHT * (1.0 / (1.0 + nearest))


# ─────────────────────────────────────────────
#  Aggression Wrapper
# ─────────────────────────────────────────────
class AggressionWrapper(Wrapper):
    """
    Reward shaping wrapper that encourages aggressive ghost hunting:

    1. CONSISTENT CHASING — potential-based reward for closing distance
       to the nearest frightened ghost.

    2. URGENCY — bonus for eating the first frightened ghost quickly
       after activation. Bonus decays linearly over URGENCY_WINDOW steps.

    3. CONSISTENCY — per-ghost bonus for each frightened ghost eaten
       during a single attack window.

    4. FULL CLEAR — bonus for eating ALL frightened ghosts before the
       frightened phase ends.

    5. EXPIRY PENALTY — penalty for letting the frightened window end
       with ghosts uneaten (discourages half-hearted activation).

    Tracks attack-window state internally:
      - _attack_active: are we currently in a frightened phase?
      - _ghosts_eaten_this_window: count of ghosts eaten this window
      - _steps_since_attack_start: for urgency calculation
      - _first_eat_recorded: prevents urgency bonus firing twice

    Usage:
        env = PacmanAggressionWrapper(your_env)

    Or stacked:
        env = PacmanRewardShapingWrapper(your_env)
        env = PacmanAggressionWrapper(env)
    """

    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.cfg       = cfg or AggressionConfig()
        self._last_obs = None

        # Attack-window tracking
        self._attack_active            = False
        self._ghosts_eaten_this_window = 0
        self._steps_since_attack_start = 0
        self._first_eat_recorded       = False

        # Episode logging
        self._episode_base_reward       = 0.0
        self._episode_aggression_reward = 0.0

    # ── helpers ──────────────────────────────
    def _reset_attack_window(self):
        self._attack_active            = False
        self._ghosts_eaten_this_window = 0
        self._steps_since_attack_start = 0
        self._first_eat_recorded       = False

    def _count_frightened_to_eaten_transitions(self, state, next_state):
        """Returns number of ghosts that went FRIGHTENED -> EATEN this step."""
        count = 0
        for i in range(N_GHOSTS):
            if state["ghosts"][i]["frightened"] and \
               next_state["ghosts"][i]["state"] == GhostState.EATEN:
                count += 1
        return count

    # ── core shaping logic ───────────────────
    def _compute_aggression_reward(self, state, next_state):
        """
        Compute total aggression reward for a transition.
        Returns (reward, info_dict) for logging.
        """
        cfg = self.cfg
        info = {
            "chase_shaping":      0.0,
            "urgency_bonus":      0.0,
            "ghosts_eaten_bonus": 0.0,
            "full_clear_bonus":   0.0,
            "expiry_penalty":     0.0,
        }

        # ── 1. Detect attack-window transitions ──
        was_attacking = self._attack_active
        is_attacking  = next_state["any_frightened"] or next_state["pacman_attack"]

        # Window started this step (any_frightened became True)
        if not was_attacking and is_attacking:
            self._attack_active            = True
            self._ghosts_eaten_this_window = 0
            self._steps_since_attack_start = 0
            self._first_eat_recorded       = False

        # Window ended this step
        if was_attacking and not is_attacking:
            # Penalise any frightened ghosts that were never eaten
            uneaten = max(0, N_GHOSTS - self._ghosts_eaten_this_window)
            # Only penalise if some ghosts were available but not all eaten
            if self._ghosts_eaten_this_window > 0 and uneaten > 0:
                info["expiry_penalty"] = -cfg.EXPIRY_PENALTY_PER_GHOST * uneaten

            # Full-clear bonus: ate all 4 ghosts in this window
            if self._ghosts_eaten_this_window >= N_GHOSTS:
                info["full_clear_bonus"] = cfg.FULL_CLEAR_BONUS

            self._reset_attack_window()

        # ── 2. Potential-based chase reward ──
        # γΦ(s') - Φ(s), only active when ghosts are frightened
        phi_s      = aggression_potential(state, cfg)
        phi_s_next = aggression_potential(next_state, cfg)
        info["chase_shaping"] = cfg.GAMMA * phi_s_next - phi_s

        # ── 3. Per-ghost-eaten bonus ──
        eaten_now = self._count_frightened_to_eaten_transitions(state, next_state)
        if eaten_now > 0:
            info["ghosts_eaten_bonus"] = eaten_now * cfg.GHOST_EATEN_BONUS
            self._ghosts_eaten_this_window += eaten_now

            # ── 4. Urgency bonus — first ghost only ──
            if not self._first_eat_recorded:
                self._first_eat_recorded = True
                steps = self._steps_since_attack_start
                urgency = max(0.0, (cfg.URGENCY_WINDOW - steps) / cfg.URGENCY_WINDOW)
                info["urgency_bonus"] = cfg.URGENCY_BONUS * urgency

        # ── 5. Tick window step counter ──
        if self._attack_active:
            self._steps_since_attack_start += 1

        total = sum(info.values())
        return total, info

    # ── gym API ──────────────────────────────
    def reset(self):
        obs = self.env.reset()
        self._last_obs                  = obs.copy()
        self._reset_attack_window()
        self._episode_base_reward       = 0.0
        self._episode_aggression_reward = 0.0
        return obs

    def step(self, action):
        next_obs, base_reward, done, info = self.env.step(action)

        state      = parse_aggression_obs(self._last_obs)
        next_state = parse_aggression_obs(next_obs)

        aggression_r, breakdown = self._compute_aggression_reward(state, next_state)
        total_reward = base_reward + aggression_r

        self._episode_base_reward       += base_reward
        self._episode_aggression_reward += aggression_r

        info["base_reward"]      = base_reward
        info["aggression_bonus"] = aggression_r
        info["aggression_breakdown"] = breakdown

        if done:
            info["episode_base_reward"]       = self._episode_base_reward
            info["episode_aggression_reward"] = self._episode_aggression_reward

        self._last_obs = next_obs.copy()
        return next_obs, total_reward, done, info
