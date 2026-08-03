import gym
from gym import spaces
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np

from .speed_wrapper import _reshape_grid, PELLET, POWER_PELLET
from .aggression_wrapper import (
    _count_eaten_ghosts,
    _is_attacking,
    _pacman_cell,
    _reset_obs,
    GHOST_DISTANCE_SLICE,
)


def _ghost_distances(obs: Any) -> np.ndarray:
    """Return the four Manhattan ghost distances from the observation."""
    return np.asarray(obs)[GHOST_DISTANCE_SLICE]


def _count_dots(grid: np.ndarray) -> int:
    """Count remaining dots (plain pellets, not power pellets) on the board."""
    return int(np.count_nonzero(grid == PELLET))


def _nearest_pill_distance(grid: np.ndarray) -> Optional[float]:
    """
    Manhattan distance from Pacman to the closest active power pellet.

    Args:
        grid: The (height, width) game grid.

    Returns:
        The distance in cells, or None if Pacman or all power pellets are gone.
    """
    cell = _pacman_cell(grid)
    if cell is None:
        return None

    rows, cols = np.where(grid == POWER_PELLET)
    if len(rows) == 0:
        return None

    distances = np.abs(rows - cell[0]) + np.abs(cols - cell[1])
    return float(np.min(distances))


class _LureUpdate(NamedTuple):
    """Lure state transitions observed on a single step."""
    in_lure: bool                 # Pacman is waiting beside a pill (not attacking)
    moved: bool                   # Pacman changed grid cell this step
    was_waiting: bool             # a lure was in progress before this step
    pill_eaten_after_lure: bool   # attack started after waiting >= min_wait steps


class _LureTrackerWrapper(gym.Wrapper):
    """
    Base class for the P1 planning behaviours: Lure Ghosts to Power Pill.

    A lure is Pacman holding position beside an active power pellet without
    eating it (not in attack mode) while ghosts approach. The tracker counts
    consecutive steps spent in that state and flags the moment the pill is
    eaten after a sufficient wait, so subclasses only shape their own signal.

    Args:
        env: The environment to wrap
        pill_distance: Manhattan distance to a power pellet that counts as
            "beside" it (default: 3.0)
        min_wait: Waiting steps required for an attack start to count as a
            completed lure (default: 4)
    """

    def __init__(self, env: gym.Env, pill_distance: float = 3.0, min_wait: int = 4):
        super().__init__(env)
        self.pill_distance = pill_distance
        self.min_wait = max(1, min_wait)

        self._wait_steps = 0
        self._prev_cell: Optional[Tuple[int, int]] = None
        self._was_attacking = False

    def _init_lure_state(self, obs: Any) -> None:
        """Initialise the lure tracker from a reset observation."""
        self._wait_steps = 0
        self._prev_cell = _pacman_cell(_reshape_grid(obs))
        self._was_attacking = _is_attacking(obs)

    def _on_reset(self, obs: Any) -> None:
        """Hook for subclasses that track extra per-episode state."""

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        obs = _reset_obs(data)
        self._init_lure_state(obs)
        self._on_reset(obs)
        return data

    def _update_lure(self, obs: Any) -> _LureUpdate:
        """Advance the lure state machine one step and report transitions."""
        grid = _reshape_grid(obs)
        cell = _pacman_cell(grid)
        attacking = _is_attacking(obs)

        moved = (
            cell is not None
            and self._prev_cell is not None
            and cell != self._prev_cell
        )
        was_waiting = self._wait_steps > 0

        # A completed lure: attack mode just started after enough waiting.
        pill_eaten_after_lure = (
            attacking and not self._was_attacking
            and self._wait_steps >= self.min_wait
        )

        pill_dist = _nearest_pill_distance(grid)
        in_lure = (
            not attacking
            and pill_dist is not None
            and pill_dist <= self.pill_distance
        )
        self._wait_steps = self._wait_steps + 1 if in_lure else 0

        if cell is not None:
            self._prev_cell = cell
        self._was_attacking = attacking

        return _LureUpdate(in_lure, bool(moved), was_waiting, pill_eaten_after_lure)


class P1aPlanningWrapper(_LureTrackerWrapper):
    """
    Planning behaviour P1.a - Lure: Count Moves While Waiting for Ghosts.

    Encourages holding position beside a power pellet while the ghosts are
    still far away. Two toggleable signals, both active only while in the lure
    state with every ghost at least ghost_far away:
      - reward_waiting: bonus per step Pacman keeps his grid cell.
      - penalise_moves: penalty per step Pacman changes grid cell.

    Args:
        env: The environment to wrap
        wait_bonus: Reward added per stationary waiting step (default: 2.0)
        move_penalty: Reward subtracted per moving step (default: 2.0)
        ghost_far: Manhattan distance all ghosts must exceed for the wait to
            count as luring (default: 10.0)
        reward_waiting: Enable the waiting bonus (default: True)
        penalise_moves: Enable the movement penalty (default: True)
        pill_distance: Manhattan distance that counts as beside a pill (default: 3.0)
        min_wait: Waiting steps for a completed lure (default: 4)
    """

    def __init__(self, env: gym.Env, wait_bonus: float = 2.0,
                 move_penalty: float = 2.0, ghost_far: float = 10.0,
                 reward_waiting: bool = True, penalise_moves: bool = True,
                 pill_distance: float = 3.0, min_wait: int = 4):
        super().__init__(env, pill_distance=pill_distance, min_wait=min_wait)
        self.wait_bonus = wait_bonus
        self.move_penalty = abs(move_penalty)
        self.ghost_far = ghost_far
        self.reward_waiting = reward_waiting
        self.penalise_moves = penalise_moves

    def _shape_reward(self, obs: Any, reward: float) -> float:
        update = self._update_lure(obs)

        ghosts_far = bool(np.all(_ghost_distances(obs) >= self.ghost_far))
        if update.in_lure and ghosts_far:
            if update.moved:
                if self.penalise_moves:
                    reward -= self.move_penalty
            elif self.reward_waiting:
                reward += self.wait_bonus
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


class P1bPlanningWrapper(_LureTrackerWrapper):
    """
    Planning behaviour P1.b - Lure: All Ghosts Lured.

    Rewards completing a lure with every ghost drawn in: when the power pellet
    is eaten after waiting beside it, a bonus is added if all four ghosts are
    within lure_distance. An optional per-ghost bonus grants partial credit
    for each close ghost on the same transition (off by default).

    Args:
        env: The environment to wrap
        all_lured_bonus: Reward added when all four ghosts are close at the
            moment the pill is eaten after a lure (default: 100.0)
        per_ghost_bonus: Reward added per close ghost on that transition
            (default: 0.0, disabled)
        lure_distance: Manhattan distance under which a ghost counts as lured
            (default: 8.0)
        pill_distance: Manhattan distance that counts as beside a pill (default: 3.0)
        min_wait: Waiting steps for a completed lure (default: 4)
    """

    def __init__(self, env: gym.Env, all_lured_bonus: float = 100.0,
                 per_ghost_bonus: float = 0.0, lure_distance: float = 8.0,
                 pill_distance: float = 3.0, min_wait: int = 4):
        super().__init__(env, pill_distance=pill_distance, min_wait=min_wait)
        self.all_lured_bonus = all_lured_bonus
        self.per_ghost_bonus = per_ghost_bonus
        self.lure_distance = lure_distance

    def _shape_reward(self, obs: Any, reward: float) -> float:
        update = self._update_lure(obs)
        if not update.pill_eaten_after_lure:
            return reward

        distances = _ghost_distances(obs)
        lured = int(np.count_nonzero(distances <= self.lure_distance))
        if lured == len(distances):
            reward += self.all_lured_bonus
        if self.per_ghost_bonus:
            reward += self.per_ghost_bonus * lured
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


class P1cPlanningWrapper(_LureTrackerWrapper):
    """
    Planning behaviour P1.c - Lure: Number Ghosts Eaten After Lure.

    Rewards ghosts eaten during the hunt that follows a completed lure, on top
    of the environment's existing ghost-eaten reward. The hunt window opens
    when the pill is eaten after waiting beside it and closes when attack mode
    ends. Kills are detected as the eaten-ghost count (grid cells with value 9)
    rising between steps, as in A3AggressionWrapper.

    Args:
        env: The environment to wrap
        kill_bonus: Reward added per ghost eaten inside the hunt window
            (default: 40.0)
        pill_distance: Manhattan distance that counts as beside a pill (default: 3.0)
        min_wait: Waiting steps for a completed lure (default: 4)
    """

    def __init__(self, env: gym.Env, kill_bonus: float = 40.0,
                 pill_distance: float = 3.0, min_wait: int = 4):
        super().__init__(env, pill_distance=pill_distance, min_wait=min_wait)
        self.kill_bonus = kill_bonus
        self._hunting = False
        self._prev_eaten = 0

    def _on_reset(self, obs: Any) -> None:
        self._hunting = False
        self._prev_eaten = _count_eaten_ghosts(_reshape_grid(obs))

    def _shape_reward(self, obs: Any, reward: float) -> float:
        update = self._update_lure(obs)
        eaten = _count_eaten_ghosts(_reshape_grid(obs))

        if update.pill_eaten_after_lure:
            self._hunting = True
        elif self._hunting and not _is_attacking(obs):
            self._hunting = False

        if self._hunting:
            kills = eaten - self._prev_eaten
            if kills > 0:
                reward += self.kill_bonus * kills

        self._prev_eaten = eaten
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


class P1dPlanningWrapper(_LureTrackerWrapper):
    """
    Planning behaviour P1.d - Lure: Caught Before Eating Pill.

    Penalises failing a lure: the episode ends (Pacman caught) while he was
    waiting beside a power pellet without having eaten it. The extra penalty
    is added on top of the environment's existing death reward. The death is
    detected from the terminal flag, as in C1bCautionWrapper, so no life
    tracking is needed.

    Args:
        env: The environment to wrap
        death_penalty: Extra reward subtracted on a death during a lure
            (default: 200.0)
        pill_distance: Manhattan distance that counts as beside a pill (default: 3.0)
        min_wait: Waiting steps for a completed lure (default: 4)
    """

    def __init__(self, env: gym.Env, death_penalty: float = 200.0,
                 pill_distance: float = 3.0, min_wait: int = 4):
        super().__init__(env, pill_distance=pill_distance, min_wait=min_wait)
        self.death_penalty = abs(death_penalty)

    def _shape_reward(self, obs: Any, reward: float, done: bool) -> float:
        update = self._update_lure(obs)
        caught_luring = update.was_waiting and not _is_attacking(obs)
        if done and caught_luring:
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


class P3PlanningWrapper(gym.Wrapper):
    """
    Planning behaviour P3 - Dots Eaten Before 1st Pill.

    Rewards clearing dots before committing to the first power pill. Dots
    eaten (plain pellet grid cells, value 1, disappearing between steps) are
    counted from the episode start, and when the first pill is eaten
    (attack mode starts for the first time) a one-time bonus is added that
    scales with the dots collected so far, capped at pill_bonus once
    ref_dots have been eaten. No bonus is given if a pill is never eaten.

    Args:
        env: The environment to wrap
        pill_bonus: Maximum bonus granted at the first pill (default: 100.0)
        ref_dots: Dots eaten at or above which the full bonus is granted
            (default: 100)
    """

    def __init__(self, env: gym.Env, pill_bonus: float = 100.0,
                 ref_dots: int = 100):
        super().__init__(env)
        self.pill_bonus = pill_bonus
        self.ref_dots = max(1, ref_dots)

        self._dots_eaten = 0
        self._rewarded = False
        self._was_attacking = False
        self._prev_dots = 0

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        obs = _reset_obs(data)
        self._dots_eaten = 0
        self._rewarded = False
        self._was_attacking = _is_attacking(obs)
        self._prev_dots = _count_dots(_reshape_grid(obs))
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        attacking = _is_attacking(obs)
        dots = _count_dots(_reshape_grid(obs))

        if not self._rewarded:
            eaten = self._prev_dots - dots
            if eaten > 0:
                self._dots_eaten += eaten

            # First pill of the episode: attack mode just started.
            if attacking and not self._was_attacking:
                scale = min(self._dots_eaten / self.ref_dots, 1.0)
                reward += self.pill_bonus * scale
                self._rewarded = True

        self._was_attacking = attacking
        self._prev_dots = dots
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


class P4aPlanningWrapper(gym.Wrapper):
    """
    Planning behaviour P4.a - Average Speed Hunting 1st Ghost.

    Rewards catching the first ghost quickly once a hunt begins. Moves are
    counted on every step Pacman is in attack mode; when the first ghost is
    caught, a bonus is added that scales inversely with the moves spent
    hunting (full bonus at or under ref_moves, shrinking beyond it), the same
    scaling pattern as Sp1SpeedWrapper. Catches are detected as the eaten-ghost
    count (grid cells with value 9) rising between steps, as in
    A3AggressionWrapper. No bonus is given if a ghost is never caught.

    By default the bonus fires once per episode, on the first catch of the
    game, with moves accumulated across all attacking steps. With
    per_hunt=True the move counter restarts whenever attack mode begins and
    the bonus fires on the first catch of each hunt window.

    Args:
        env: The environment to wrap
        catch_bonus: Maximum bonus granted for a fast first catch (default: 100.0)
        ref_moves: Hunting-move count at or under which the full bonus is
            granted (default: 20)
        per_hunt: Reward the first catch of every hunt window instead of only
            the first catch of the episode (default: False)
    """

    def __init__(self, env: gym.Env, catch_bonus: float = 100.0,
                 ref_moves: int = 20, per_hunt: bool = False):
        super().__init__(env)
        self.catch_bonus = catch_bonus
        self.ref_moves = max(1, ref_moves)
        self.per_hunt = per_hunt

        self._hunt_moves = 0
        self._rewarded = False
        self._was_attacking = False
        self._prev_eaten = 0

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        obs = _reset_obs(data)
        self._hunt_moves = 0
        self._rewarded = False
        self._was_attacking = _is_attacking(obs)
        self._prev_eaten = _count_eaten_ghosts(_reshape_grid(obs))
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        attacking = _is_attacking(obs)
        eaten = _count_eaten_ghosts(_reshape_grid(obs))

        # A new hunt window opens when attack mode starts.
        if attacking and not self._was_attacking and self.per_hunt:
            self._hunt_moves = 0
            self._rewarded = False

        if attacking:
            self._hunt_moves += 1
            if eaten > self._prev_eaten and not self._rewarded:
                moves = max(self._hunt_moves, self.ref_moves)
                reward += self.catch_bonus * (self.ref_moves / moves)
                self._rewarded = True

        self._was_attacking = attacking
        self._prev_eaten = eaten
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


class P4bPlanningWrapper(gym.Wrapper):
    """
    Planning behaviour P4.b - Average Speed Hunting 2nd Ghost.

    Rewards catching the second ghost quickly after the first: moves are
    counted on every attacking step from the first catch onwards, and when
    the second ghost is caught a bonus is added that scales inversely with
    those moves (full bonus at or under ref_moves, shrinking beyond it), the
    same scaling pattern as Sp1SpeedWrapper. Catches are detected as the
    eaten-ghost count (grid cells with value 9) rising between steps, as in
    A3AggressionWrapper. Composes with P4aPlanningWrapper, which rewards the
    first catch, without counting the same moves twice.

    By default the bonus fires once per episode, on the second catch of the
    game, with catches and moves accumulated across hunt windows. With
    per_hunt=True the catch and move counters restart whenever attack mode
    begins and the bonus fires on the second catch of each hunt window.

    Args:
        env: The environment to wrap
        catch_bonus: Maximum bonus granted for a fast second catch (default: 100.0)
        ref_moves: Move count from the first catch at or under which the full
            bonus is granted (default: 20)
        per_hunt: Reward the second catch of every hunt window instead of only
            the second catch of the episode (default: False)
    """

    def __init__(self, env: gym.Env, catch_bonus: float = 100.0,
                 ref_moves: int = 20, per_hunt: bool = False):
        super().__init__(env)
        self.catch_bonus = catch_bonus
        self.ref_moves = max(1, ref_moves)
        self.per_hunt = per_hunt

        self._catches = 0
        self._hunt_moves = 0
        self._rewarded = False
        self._was_attacking = False
        self._prev_eaten = 0

    def reset(self, **kwargs):
        data = self.env.reset(**kwargs)
        obs = _reset_obs(data)
        self._catches = 0
        self._hunt_moves = 0
        self._rewarded = False
        self._was_attacking = _is_attacking(obs)
        self._prev_eaten = _count_eaten_ghosts(_reshape_grid(obs))
        return data

    def _shape_reward(self, obs: Any, reward: float) -> float:
        attacking = _is_attacking(obs)
        eaten = _count_eaten_ghosts(_reshape_grid(obs))

        # A new hunt window opens when attack mode starts.
        if attacking and not self._was_attacking and self.per_hunt:
            self._catches = 0
            self._hunt_moves = 0
            self._rewarded = False

        if attacking:
            # Moves towards the second catch only count from the first catch.
            if self._catches >= 1:
                self._hunt_moves += 1
            kills = eaten - self._prev_eaten
            if kills > 0:
                self._catches += kills
                if self._catches >= 2 and not self._rewarded:
                    moves = max(self._hunt_moves, self.ref_moves)
                    reward += self.catch_bonus * (self.ref_moves / moves)
                    self._rewarded = True

        self._was_attacking = attacking
        self._prev_eaten = eaten
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
