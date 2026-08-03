import gym
from gym import spaces
from typing import Any, Optional, Tuple

import numpy as np

from .speed_wrapper import _reshape_grid, PACMAN_NORMAL, PACMAN_ATTACK
from .aggression_wrapper import (
    _reset_obs,
    _ghost_house_centroid,
    _pacman_cell,
    _count_pellets,
)

# Observation layout (see wrappers/README.md):
ATTACK_STATE_INDEX = 3          # Pacman attack state (bool): 1 = on a pill / hunt mode
SCORE_INDEX = 23                # Normalised score (score / 3200)
LIVES_INDEX = 24                # Normalised lives (lives / 3)
REMAINING_PELLETS_INDEX = 25    # Normalised remaining pellets (/ 244)
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


def _score(obs: Any) -> float:
    """Normalised score (score / 3200) from the observation."""
    return float(np.asarray(obs)[SCORE_INDEX])


def _lives(obs: Any) -> float:
    """Normalised lives (lives / 3) from the observation."""
    return float(np.asarray(obs)[LIVES_INDEX])


def _pellets_remaining(obs: Any) -> float:
    """Normalised remaining pellet count from the observation."""
    return float(np.asarray(obs)[REMAINING_PELLETS_INDEX])


def _min_ghost_distance(obs: Any) -> float:
    """Manhattan distance from Pacman to the nearest ghost."""
    distances = np.asarray(obs)[GHOST_DISTANCE_SLICE]
    return float(np.min(distances))


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


class C7CautionWrapper(gym.Wrapper):
    """
    Caution behaviour C7 - Killed at Ghost House.

    Adds an extra death penalty when Pacman dies collecting dots around the
    ghost house. The ghost house is located once at reset as the centroid of
    the ghost cells (as in A1). A death (episode termination, as in C1b - no
    life tracking) pays the penalty when Pacman's last known cell is within
    house_distance of the house and, optionally, a dot was eaten within the
    last dot_window steps (dot eating detected from the pellet count dropping,
    as in A6).

    Args:
        env: The environment to wrap
        death_penalty: Extra reward subtracted on a death at the house (default: 200.0)
        house_distance: Manhattan distance to the house centroid that counts as
            "around the ghost house" (default: 5.0)
        require_dots: Only penalise if a dot was eaten recently, so the death
            counts as "collecting dots" rather than just passing by (default: True)
        dot_window: Steps since the last dot eaten for a death to count (default: 8)
    """

    def __init__(self, env: gym.Env, death_penalty: float = 200.0,
                 house_distance: float = 5.0, require_dots: bool = True,
                 dot_window: int = 8):
        super().__init__(env)
        self.death_penalty = abs(death_penalty)
        self.house_distance = house_distance
        self.require_dots = require_dots
        self.dot_window = max(0, dot_window)

        self._house: Optional[Tuple[float, float]] = None
        self._last_cell: Optional[Tuple[int, int]] = None
        self._prev_pellets: Optional[int] = None
        self._steps_since_dot: Optional[int] = None

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        grid = _reshape_grid(_reset_obs(data))
        self._house = _ghost_house_centroid(grid)
        self._last_cell = _pacman_cell(grid)
        self._prev_pellets = _count_pellets(grid)
        self._steps_since_dot = None
        return data

    def _near_house(self) -> bool:
        """Return True if Pacman's last known cell is close to the ghost house."""
        if self._house is None or self._last_cell is None:
            return False
        distance = (abs(self._last_cell[0] - self._house[0])
                    + abs(self._last_cell[1] - self._house[1]))
        return distance <= self.house_distance

    def _shape_reward(self, obs: Any, reward: float, done: bool) -> float:
        grid = _reshape_grid(obs)

        # Keep the last known cell: Pacman may be absent from the terminal grid.
        cell = _pacman_cell(grid)
        if cell is not None:
            self._last_cell = cell

        # Track how recently a dot was eaten (pellet count dropping, as in A6).
        pellets = _count_pellets(grid)
        if self._prev_pellets is not None and pellets < self._prev_pellets:
            self._steps_since_dot = 0
        elif self._steps_since_dot is not None:
            self._steps_since_dot += 1
        self._prev_pellets = pellets

        if done and self._near_house():
            collecting = (not self.require_dots
                          or (self._steps_since_dot is not None
                              and self._steps_since_dot <= self.dot_window))
            if collecting:
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


class C4CautionWrapper(gym.Wrapper):
    """
    Caution behaviour C4 - Caught After Hunt.

    Penalises overstaying a hunt: chasing the ghosts until after the pill wears
    off and losing a life for it. When attack mode ends while Pacman is still
    close to a ghost (i.e. mid-chase), a grace window opens; if the episode ends
    (Pacman caught, detected from the terminal flag as in C1b) within that
    window, an extra death penalty is applied on top of the environment's
    existing death reward.

    Args:
        env: The environment to wrap
        death_penalty: Extra reward subtracted on a post-hunt death (default: 200.0)
        grace_steps: Steps after the hunt ends during which a death counts (default: 8)
        require_chasing: Only open the window if a ghost is nearby when the
            pill wears off (default: True)
        chase_distance: Manhattan distance to the nearest ghost that counts as
            still chasing at hunt end (default: 5.0)
    """

    def __init__(self, env: gym.Env, death_penalty: float = 200.0,
                 grace_steps: int = 8, require_chasing: bool = True,
                 chase_distance: float = 5.0):
        super().__init__(env)
        self.death_penalty = abs(death_penalty)
        self.grace_steps = max(1, grace_steps)
        self.require_chasing = require_chasing
        self.chase_distance = chase_distance

        self._was_attacking = False
        self._steps_since_hunt_end: Optional[int] = None

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        obs = _reset_obs(data)
        self._was_attacking = _is_attacking(obs)
        self._steps_since_hunt_end = None
        return data

    def _shape_reward(self, obs: Any, reward: float, done: bool) -> float:
        attacking = _is_attacking(obs)

        # The hunt just ended: open the grace window if Pacman was still chasing.
        if self._was_attacking and not attacking:
            chasing = (not self.require_chasing
                       or _min_ghost_distance(obs) <= self.chase_distance)
            self._steps_since_hunt_end = 0 if chasing else None
        elif self._steps_since_hunt_end is not None:
            self._steps_since_hunt_end += 1
            if self._steps_since_hunt_end > self.grace_steps:
                self._steps_since_hunt_end = None

        if done and self._steps_since_hunt_end is not None:
            reward -= self.death_penalty

        self._was_attacking = attacking
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


class C5CautionWrapper(gym.Wrapper):
    """
    Caution behaviour C5 - Moves With No Points Scored.

    Penalises traversals of empty space: each step Pacman moves to a new grid
    cell without scoring any points (normalised score obs[23] unchanged), a
    small penalty is applied. Standing still or moving onto a dot costs
    nothing, so cleared corridors become expensive to wander through.

    Args:
        env: The environment to wrap
        move_penalty: Reward subtracted per pointless move (default: 2.0)
    """

    def __init__(self, env: gym.Env, move_penalty: float = 2.0):
        super().__init__(env)
        self.move_penalty = abs(move_penalty)

        self._prev_cell: Optional[Tuple[int, int]] = None
        self._prev_score: Optional[float] = None

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        obs = _reset_obs(data)
        self._prev_cell = _pacman_cell(_reshape_grid(obs))
        self._prev_score = _score(obs)
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        cell = _pacman_cell(_reshape_grid(obs))
        score = _score(obs)

        moved = (cell is not None and self._prev_cell is not None
                 and cell != self._prev_cell)
        scored = self._prev_score is not None and score > self._prev_score
        if moved and not scored:
            reward -= self.move_penalty

        if cell is not None:
            self._prev_cell = cell
        self._prev_score = score
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


class C6CautionWrapper(gym.Wrapper):
    """
    Caution behaviour C6 - Points Scored per Life Lost.

    Rewards making each life count: when a life is lost, a bonus proportional
    to the normalised score gained during that life (obs[23], score / 3200) is
    added, so a productive life offsets more of the death penalty while dying
    early with few points earns almost nothing. A life loss is either the
    lives observation (obs[24]) dropping or the episode ending with pellets
    still on the board (the final death - a level clear ends with none left).

    Args:
        env: The environment to wrap
        life_bonus: Max bonus per life, granted when the whole normalised
            score (3200 points) is earned in a single life (default: 200.0)
    """

    def __init__(self, env: gym.Env, life_bonus: float = 200.0):
        super().__init__(env)
        self.life_bonus = life_bonus

        self._life_start_score: Optional[float] = None
        self._prev_lives: Optional[float] = None

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        obs = _reset_obs(data)
        self._life_start_score = _score(obs)
        self._prev_lives = _lives(obs)
        return data

    def _shape_reward(self, obs: Any, reward: float, done: bool) -> float:
        score = _score(obs)
        lives = _lives(obs)

        life_lost = self._prev_lives is not None and lives < self._prev_lives
        terminal_death = done and not life_lost and _pellets_remaining(obs) > 0
        if (life_lost or terminal_death) and self._life_start_score is not None:
            reward += self.life_bonus * max(score - self._life_start_score, 0.0)
            self._life_start_score = score

        self._prev_lives = lives
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
