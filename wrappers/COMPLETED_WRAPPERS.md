# Completed Behaviour Reward Wrappers

Tracks the behaviour reward wrappers that have been implemented. Each is a
`gym.Wrapper` that handles both the 4-tuple (Unity / old Gym) and 5-tuple
(Gymnasium) `step` signatures, uses only values from the observation, and is
enabled through the `cfg` dict passed to `wrap_env` (see `pacman_wrapper.py`).

Rewards are weighted against the base reward structure (pellet +20, power pellet
+70, ghost +40, level clear +1000, death −1000) and are applied before the
optional `NormaliseRewardWrapper` (÷1000).

## How to enable

```python
from wrappers import wrap_env

env = wrap_env(
    env,
    skip=4,
    wrap_reward='normalise',
    cfg={
        'Sp1': True,                       # enable with defaults
        'A1': {'house_distance': 4.0},     # enable with custom params
        'C2b': None,                       # disabled (falsy)
    },
)
```

A `cfg` value that is truthy enables the wrapper: pass `True` for defaults, or a
`dict` of constructor params. Behaviours may be combined freely.

## Status overview

| Class             | Behaviour                                        | cfg key | Wrapper class               | File                           | Status |
|:------------------|:-------------------------------------------------|:--------|:----------------------------|:-------------------------------|:------:|
| Speed             | Sp1 - Average Cycles Per Sector                  | `Sp1`   | `Sp1SpeedWrapper`           | `speed_wrapper.py`             |   ✅    |
| Speed             | Sp2 - Average States                             | `Sp2`   | `Sp2SpeedWrapper`           | `speed_wrapper.py`             |   ✅    |
| Caution           | C1a - Times Trapped By Ghosts                    | `C1a`   | `C1aCautionWrapper`         | `caution_wrapper.py`           |   ✅    |
| Caution           | C1b - Times Trapped and Killed By Ghosts         | `C1b`   | `C1bCautionWrapper`         | `caution_wrapper.py`           |   ✅    |
| Caution           | C2a - Average Distance to Ghosts                 | `C2a`   | `C2aCautionWrapper`         | `caution_wrapper.py`           |   ✅    |
| Caution           | C2b - Average Distance During Hunt               | `C2b`   | `C2bCautionWrapper`         | `caution_wrapper.py`           |   ✅    |
| Caution           | C4 - Caught After Hunt                           | `C4`    | `C4CautionWrapper`          | `caution_wrapper.py`           |   ✅    |
| Caution           | C5 - Moves With No Points Scored                 | `C5`    | `C5CautionWrapper`          | `caution_wrapper.py`           |   ✅    |
| Caution           | C6 - Points Scored per Life Lost                 | `C6`    | `C6CautionWrapper`          | `caution_wrapper.py`           |   ✅    |
| Caution           | C7 - Killed at Ghost House                       | `C7`    | `C7CautionWrapper`          | `caution_wrapper.py`           |   ✅    |
| Thoroughness      | T1 - Sector by Sector                            | `T1`    | `T1ThoroughnessWrapper`     | `thoroughness_wrapper.py`      |   ✅    |
| Thoroughness      | T2 - Leaves a Single Pill                        | `T2`    | `T2ThoroughnessWrapper`     | `thoroughness_wrapper.py`      |   ✅    |
| Aggression        | A1 - Hunt Close To Ghost House                   | `A1`    | `A1AggressionWrapper`       | `aggression_wrapper.py`        |   ✅    |
| Aggression        | A2 - Chase Ghosts or Eat New Cherry              | `A2`    | `A2AggressionWrapper`       | `aggression_wrapper.py`        |   ✅    |
| Aggression        | A3 - Ghost Kills                                 | `A3`    | `A3AggressionWrapper`       | `aggression_wrapper.py`        |   ✅    |
| Aggression        | A6 - Chase Ghosts or Collect Dots                | `A6`    | `A6AggressionWrapper`       | `aggression_wrapper.py`        |   ✅    |
| Planning          | P1a - Lure: Count Moves While Waiting for Ghosts | `P1a`   | `P1aPlanningWrapper`        | `planning_wrapper.py`          |   ✅    |
| Planning          | P1b - Lure: All Ghosts Lured                     | `P1b`   | `P1bPlanningWrapper`        | `planning_wrapper.py`          |   ✅    |
| Planning          | P1c - Lure: Number Ghosts Eaten After Lure       | `P1c`   | `P1cPlanningWrapper`        | `planning_wrapper.py`          |   ✅    |
| Planning          | P1d - Lure: Caught Before Eating Pill            | `P1d`   | `P1dPlanningWrapper`        | `planning_wrapper.py`          |   ✅    |
| Planning          | P3 - Dots Eaten Before 1st Pill                  | `P3`    | `P3PlanningWrapper`         | `planning_wrapper.py`          |   ✅    |
| Planning          | P4a - Average Speed Hunting 1st Ghost            | `P4a`   | `P4aPlanningWrapper`        | `planning_wrapper.py`          |   ✅    |
| Planning          | P4b - Average Speed Hunting 2nd Ghost            | `P4b`   | `P4bPlanningWrapper`        | `planning_wrapper.py`          |   ✅    |
| Resource Hoarding | R1 - Average Time For Pac-Man to Eat Cherry      | `R1`    | `R1ResourceHoardingWrapper` | `resource_hoarding_wrapper.py` |   ✅    |

## Details

### Speed - `speed_wrapper.py`

**Sp1 `Sp1SpeedWrapper`** - Average Cycles Per Sector
Encourages clearing each maze sector (a 2×2 grid quadrant) in fewer cycles.
Per-step penalty plus a speed-scaled bonus when a quadrant is fully cleared.
- `step_penalty=2.0` - subtracted each step
- `clear_bonus=200.0` - max bonus when a sector is cleared
- `ref_cycles=50` - reference cycle count for bonus scaling

**Sp2 `Sp2SpeedWrapper`** - Average States
Per-step penalty over the whole episode to encourage clearing a level in fewer
moves (completion itself is already rewarded by the +1000 level clear).
- `step_penalty=2.0` - subtracted each step

### Caution - `caution_wrapper.py`

**C1a `C1aCautionWrapper`** - Times Trapped By Ghosts
Per-step penalty while trapped. Two toggleable detectors:
- `use_proximity=True` - ≥ `min_ghosts` (default 2) ghosts within `trap_distance` (default 3.0, Manhattan, from `obs[30:34]`)
- `use_directional=True` - a dangerous ghost (grid 5–7) on opposite sides of Pacman in the grid
- `trap_penalty=10.0`

**C1b `C1bCautionWrapper`** - Times Trapped and Killed By Ghosts
Extra penalty when the episode ends while the trapped condition holds (death
detected from the terminal flag, no life tracking). Same detector params as C1a.
- `death_penalty=200.0`

**C2a `C2aCautionWrapper`** - Average Distance to Ghosts
While **not** on a pill, bonus proportional to the mean normalised ghost distance
(`obs[30:34]`).
- `weight=5.0` - max per-step bonus

**C2b `C2bCautionWrapper`** - Average Distance During Hunt
Same as C2a but only while **in hunt mode** (rewards keeping distance even while
powered up). Note: opposite of A6.
- `weight=5.0`

**C4 `C4CautionWrapper`** - Caught After Hunt
Extra penalty when chasing the ghosts until after the pill wears off costs a
life. When attack mode ends (attack flag `obs[3]` falling) while still chasing,
a grace window opens; a death (terminal flag, as in C1b - no life tracking)
inside the window pays the penalty.
- `death_penalty=200.0` - extra penalty on a post-hunt death
- `grace_steps=8` - window length after the hunt ends
- `require_chasing=True` - only open the window if a ghost is within
  `chase_distance` when the pill wears off
- `chase_distance=5.0` - Manhattan, nearest ghost from `obs[30:34]`

**C5 `C5CautionWrapper`** - Moves With No Points Scored
Penalty each step Pacman moves to a new grid cell without scoring any points
(normalised score `obs[23]` unchanged) - a traversal of empty space. Standing
still or moving onto a dot costs nothing.
- `move_penalty=2.0` - subtracted per pointless move

**C6 `C6CautionWrapper`** - Points Scored per Life Lost
Bonus when a life is lost, proportional to the normalised score (`obs[23]`)
gained during that life, so a productive life offsets more of the death
penalty. A life loss is the lives observation (`obs[24]`) dropping, or the
episode ending with pellets still on the board (`obs[25]` > 0) - the final
death, covering single-life builds without life tracking.
- `life_bonus=200.0` - max bonus, granted when the whole normalised score
  (3200 points) is earned in a single life

**C7 `C7CautionWrapper`** - Killed at Ghost House
Extra penalty when Pacman dies collecting dots around the ghost house. The
house is the ghost-cell centroid captured at reset (as in A1); the death is the
terminal flag (as in C1b - no life tracking) with Pacman's last known grid cell
within range of the house.
- `death_penalty=200.0` - extra penalty on a death at the house
- `house_distance=5.0` - Manhattan distance to the house centroid that counts
  as "around the ghost house"
- `require_dots=True` - only penalise if a dot was eaten within the last
  `dot_window` steps (pellet count dropping, as in A6), so the death counts as
  "collecting dots" rather than just passing by
- `dot_window=8` - steps since the last dot for a death to qualify

### Thoroughness - `thoroughness_wrapper.py`

**T1 `T1ThoroughnessWrapper`** - Sector by Sector
Clear one 2×2 quadrant at a time. Two toggleable signals:
- `use_penalise_leaving=True` - penalty (`leave_penalty=10.0`) for entering a new quadrant while the previous one still has pills
- `use_reward_staying=True` - bonus (`stay_bonus=2.0`) per step spent in the current uncleared quadrant

**T2 `T2ThoroughnessWrapper`** - Leaves a Single Pill
Discourages isolated pills (a pill with no orthogonally-adjacent pill). Two
toggleable signals:
- `use_count=True` - per-step penalty `weight` (default 2.0) × singleton count
- `use_on_create=True` - penalty `create_penalty` (default 10.0) per newly created singleton

### Aggression - `aggression_wrapper.py`

**A1 `A1AggressionWrapper`** - Hunt Close To Ghost House
Bonus each step Pacman is attacking and within `house_distance` (default 5.0,
Manhattan) of the ghost-house centroid (captured once at reset from the ghost
cells in the pen).
- `bonus=5.0`, `house_distance=5.0`

**A2 `A2AggressionWrapper`** - Chase Ghosts or Eat New Cherry
Encourages staying on the ghost chase when a cherry appears mid-hunt instead of
abandoning it for the cherry. Two toggleable signals, both active only while
attacking with a cherry out (grid value 10):
- `penalise_abandon=True` - penalty `abandon_penalty` (default 5.0) per step the
  Manhattan distance to the nearest frightened ghost (grid value 8) rises
- `penalise_cherry=True` - one-off penalty `cherry_penalty` (default 20.0) when
  the cherry is eaten while attacking (vanishes with Pacman within
  `eat_distance`, default 2.0, of its cell - R1-style detection)

**A3 `A3AggressionWrapper`** - Ghost Kills
Bonus per ghost eaten (eaten-ghost grid cells, value 9, rising between steps), on
top of the env's existing +40.
- `kill_bonus=40.0`

**A6 `A6AggressionWrapper`** - Chase Ghosts or Collect Dots
Encourages using pills to chase ghosts rather than collect dots. Two toggleable
signals, both active only while attacking:
- `reward_chasing=True` - bonus `chase_weight` (default 5.0) × ghost closeness
- `penalise_dots=True` - penalty `dot_penalty` (default 5.0) per dot eaten while attacking

### Planning - `planning_wrapper.py`

All four P1 behaviours share a lure tracker (`_LureTrackerWrapper`): a **lure**
is Pacman within `pill_distance` (default 3.0, Manhattan) of an active power
pellet (grid value 2) while **not** attacking; consecutive steps in that state
are counted, and a lure **completes** when attack mode starts after at least
`min_wait` (default 4) waiting steps. P1b/P1c/P1d predicate on this shared
detection, per the P1 definition in the README (sub-features depend on the
lure itself). Both params are overridable on every P1 wrapper.

**P1a `P1aPlanningWrapper`** - Lure: Count Moves While Waiting for Ghosts
While in the lure state and every ghost is at least `ghost_far` (default 10.0,
from `obs[30:34]`) away. Two toggleable signals:
- `reward_waiting=True` - bonus `wait_bonus` (default 2.0) per step Pacman holds his grid cell
- `penalise_moves=True` - penalty `move_penalty` (default 2.0) per step he moves

**P1b `P1bPlanningWrapper`** - Lure: All Ghosts Lured
On the pill-eaten-after-lure transition, bonus `all_lured_bonus` (default
100.0) when all four ghosts are within `lure_distance` (default 8.0).
- `per_ghost_bonus=0.0` - optional partial credit per close ghost (disabled)

**P1c `P1cPlanningWrapper`** - Lure: Number Ghosts Eaten After Lure
Bonus `kill_bonus` (default 40.0) per ghost eaten inside the hunt window that
opens on a completed lure and closes when attack mode ends. Kill detection as
in A3 (eaten-ghost grid cells, value 9, rising between steps), on top of the
env's existing +40.

**P1d `P1dPlanningWrapper`** - Lure: Caught Before Eating Pill
Extra penalty `death_penalty` (default 200.0) when the episode ends while
Pacman was waiting beside the pill without having eaten it (terminal flag, as
in C1b - no life tracking).

**P3 `P3PlanningWrapper`** - Dots Eaten Before 1st Pill
One-time milestone bonus when the first power pill of the episode is eaten
(attack mode starting for the first time), scaled by the dots eaten so far
(plain pellet grid cells, value 1, disappearing between steps):
`pill_bonus * min(dots / ref_dots, 1)`. Encourages clearing dots before
committing to the first pill. No bonus if a pill is never eaten.
- `pill_bonus=100.0` - max bonus, granted once `ref_dots` dots are eaten first
- `ref_dots=100` - dots for the full bonus (board starts with 240 dots + 4 pills)

**P4a `P4aPlanningWrapper`** - Average Speed Hunting 1st Ghost
Generalised from "Ghost 1" to the **first ghost caught in the episode**: moves
are counted on every attacking step, and on the first catch (eaten-ghost grid
cells, value 9, rising between steps, as in A3) a bonus is added that scales
inversely with the moves spent hunting, as in Sp1. No bonus if no ghost is
ever caught.
- `catch_bonus=100.0` - max bonus, granted when caught within `ref_moves`
- `ref_moves=20` - hunting-move count for the full bonus; slower catches get
  `catch_bonus * ref_moves / moves`
- `per_hunt=False` - when True, the move counter restarts at each hunt start
  and the bonus fires on the first catch of **each** hunt window instead of
  once per episode

**P4b `P4bPlanningWrapper`** - Average Speed Hunting 2nd Ghost
Generalised from "Ghost 2" to the **second ghost caught in the episode**: moves
are counted on attacking steps **from the first catch onwards**, and on the
second catch (detection as in A3/P4a) a bonus is added that scales inversely
with those moves, as in Sp1. Composes with P4a without counting the same moves
twice. No bonus if a second ghost is never caught.
- `catch_bonus=100.0` - max bonus, granted when the second catch comes within
  `ref_moves` of the first
- `ref_moves=20` - move count for the full bonus; slower catches get
  `catch_bonus * ref_moves / moves`
- `per_hunt=False` - when True, the catch and move counters restart at each
  hunt start and the bonus fires on the second catch of **each** hunt window
  instead of once per episode

### Resource Hoarding - `resource_hoarding_wrapper.py`

**R1 `R1ResourceHoardingWrapper`** - Average Time For Pac-Man to Eat Cherry
Speed-scaled bonus (as in Sp1/P4a) for collecting a cherry (grid value 10)
quickly after it appears: on a disappearance with Pacman within `eat_distance`
of the cherry's last cell, adds `cherry_bonus * ref_steps / max(steps, ref_steps)`
where `steps` is how long the cherry was out. A disappearance with Pacman far
away is treated as a timer despawn and pays nothing (no miss penalty). Every
cherry collected in the episode pays its own bonus.
- `cherry_bonus=100.0` - max bonus, granted when eaten within `ref_steps`
- `ref_steps=20` - on-screen step count for the full bonus
- `eat_distance=2.0` - Manhattan distance that counts a disappearance as eaten

## Notes

- Sp1, T1 use the 2×2 quadrant helpers in `speed_wrapper.py`
  (`_pacman_quadrant`, `_quadrant_pellet_counts`).
- C2b (keep distance during hunt) and A6 (chase during hunt) encode opposite
  playstyles - enabling both at once will partly cancel.
- A2 (penalise cherry-grabbing during a hunt) and R1 (reward fast cherry
  collection) conflict whenever a cherry appears mid-hunt - enabling both will
  partly cancel there.
- The proximity trap detector (C1a/C1b) cannot distinguish frightened ghosts from
  threats; the directional detector excludes frightened/eaten ghosts.
- Numeric defaults are starting points and are all overridable via `cfg`.
