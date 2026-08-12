# Pacman-RL
The purpose of this repo is to conduct RL on Pacman, and create different play-style
behaviours. See [Pacman-Unity_AiPerCog](https://github.com/PipaFlores/Pacman-Unity_AiPerCog) for more info
## Structure
The `pacman_builds` folder contains different pacman executables used for training and are distiguished by the observation
space used. At the moment only `small_obs` is present, and the observation is just Pacmans loacation and movement 
direction.
## Installation & Setup
Create a conda environment using the following command
```
conda create --name pacman python=3.10.12
```
From the repo file location run the following command
```
git clone --branch release_23 https://github.com/Unity-Technologies/ml-agents.git
```
If you want to train using GPU, your will need to install Pytorch before installing `mlagents`. To install m
Pytorch with GPU (windows). Activate the `pacman conda` environment and run the following:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
To install `mlagents` Python package, activate the conda environment and run the following in the command line
```
cd /path/to/ml-agents
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```
At this point `pacman.py` can be run to check installation of `ml-agents`is working.

To install `stable-baselines3` run the following command:
```
pip install stable-baselines3
```
If you run into a shimmy error when trying to run `train_pacman.py`, simply run 
```
pip install shimmy
```

# Training a Behaviour Agent
`train_behaviour_pacman.py` trains a PPO agent whose reward is shaped by one or more
behavlet wrappers. A run is defined by three things:

1. the **behaviour config** (`base_cfg`) inside `train_behaviour_pacman.py` — which reward
   wrappers are active,
2. an **experiment JSON** in `exp/behaviour/` — the build, the wandb run name and the PPO
   hyperparameters,
3. optionally a **SLURM script** such as `behavlets_p1abcd_003.sh` to submit the run on the
   cluster.

## 1. Choosing the reward wrappers
The behaviour wrappers are selected with the `base_cfg` dictionary near the top of
`train_behaviour_pacman.py`. Every key is a behavlet ID and is `False` by default; set a key
to `True` to enable that wrapper. The example below is the config used for the `P1abcd` runs
(all four planning behavlets on, everything else off):
```python
base_cfg = {
    "Sp1": None, "Sp2": None,
    "C1a": None, "C1b": None, "C2a": None, "C2b": None,
    "C4": None, "C5": None, "C6": None, "C7": None,
    "A1": None, "A2": None, "A3": None, "A6": None,
    "P1a": True, "P1b": True, "P1c": True, "P1d": True,
    "P3": None, "P4a": None, "P4b": None,
}
```
A key can also be given a dictionary instead of `True` to override that wrapper's reward
parameters, e.g. `"P1a": {"reward": 0.5}`. This dict is passed straight to `wrap_env`
(`wrappers/pacman_wrapper.py`) — see `wrappers/readme.md` and
`wrappers/COMPLETED_WRAPPERS.md` for what each behavlet does and which parameters it takes.

Note that `base_cfg` is **not** read from the JSON, so it has to be edited in the source
before launching a run. Whatever is set here is also what needs to be mirrored in
`collect_data.py`'s `reward_cfg` when the trained agent is later rolled out.

Other constants in the same block control the env setup:
```python
NUM_ENVS = 16       # parallel Unity envs (SubprocVecEnv)
FRAME_SKIP = True   # apply the frame skip wrapper
FRAMES_TO_SKIP = 4  # frames skipped per agent step
```

## 2. The experiment JSON
The run is otherwise configured by a JSON file under `exp/behaviour/`. For example
`exp/behaviour/p1abcd_003.json`:
```json
{
  "exp_config": {
    "project": "pacman-rl-behavlets",
    "name": "pacman-rl-PPO-P1abcd-003",
    "description": "Training trail 18 from cluster optuna search on full obs and increased pellet reward of 20 with 100 ghost",
    "pacman_path": "./pacman_builds/linux/base_1/AiPerPacman.x86_64",
    "continue_training": false,
    "pretrained_model_path": "./logs/pacman-rl-cluster/pacman-rl-PPO-Trial-284-Single-life-no-ghost-02-kl/models/best_model.zip",
    "step_penalty": -0.0
  },
  "ppo_config": {
    "learning_rate": 0.00013025229343275444,
    "n_steps": 1024,
    "batch_size": 1024,
    "n_epochs": 10,
    "gamma": 0.9992364864459256,
    "gae_lambda": 0.9265249499387012,
    "clip_range": 0.11143626540303606,
    "ent_coef": 1.010432138448156e-05,
    "vf_coef": 0.8956626030059094,
    "max_grad_norm": 2.781844676613703,
    "policy_kwargs": {
      "net_arch": { "pi": [512, 512], "vf": [512, 512] },
      "activation_fn": "tanh"
    }
  }
}
```
`exp_config` fields:

| Field | Purpose |
|-------|---------|
| `project` | wandb project, also the first part of the log directory |
| `name` | wandb run name, also the second part of the log directory |
| `description` | free text, passed to wandb as the run notes |
| `pacman_path` | Unity build to train against (use the `linux/` build on the cluster) |
| `continue_training` | if `true`, resume from `pretrained_model_path` at 20% of the learning rate instead of starting fresh |
| `pretrained_model_path` | checkpoint used when `continue_training` is `true` (ignored otherwise) |
| `step_penalty` | per-step reward applied by `StepRewardWrapper` |

`ppo_config` is passed directly to `stable_baselines3.PPO`. `activation_fn` accepts
`tanh`, `relu` or `leakyRelu` (anything else falls back to Mish).

To set up a new run, copy an existing JSON (e.g. `exp/behaviour/p1abcd_003.json`), give it a
new `name` so it does not overwrite a previous run's logs, and adjust the hyperparameters.

## 3. Launching the run
Pass the JSON path **without** the `.json` extension — the script appends it:
```
conda activate pacman
python train_behaviour_pacman.py --json_path="exp/behaviour/p1abcd_003"
```
Training runs headless (`no_graphics=True`) for 160M timesteps and requires a wandb login.

On the cluster, submit the SLURM wrapper instead. `behavlets_p1abcd_003.sh` is the template
— it activates the conda env and runs the same command:
```bash
#!/bin/bash
#SBATCH -p bigbatch
#SBATCH -N 1
#SBATCH -J pacman_train_behaviour
#SBATCH -o ./logs/cluster/behavlets/pacman_train_behaviour.%N.%j.out
#SBATCH -e ./logs/cluster/behavlets/pacman_train_behaviour.%N.%j.err

source ~/.bashrc
cd ~/Pacman-RL
conda activate pacman

python train_behaviour_pacman.py --json_path="exp/behaviour/p1abcd_003"
```
Submit it with:
```
sbatch behavlets_p1abcd_003.sh
```
For a new experiment, copy the script and change the `--json_path` to point at your new JSON.
The `./logs/cluster/behavlets/` directory must exist or SLURM will fail to write the job
output.

## 4. Outputs
Everything is written to `./logs/<project>/<name>/`, e.g.
`./logs/pacman-rl-behavlets/pacman-rl-PPO-P1abcd-003/`:

- `models/best_model.zip` — best model by mean eval reward
- `PPO_pacman.zip` — final model at the end of training
- `ppo_model_*_steps.zip` — periodic checkpoints (every 2M steps)
- `tensorboard/` — tensorboard logs (also synced to wandb)

Evaluation runs every 50k steps (800k / `NUM_ENVS`) over 10 episodes and logs the custom
metrics in `base_custom_metric_dict` — pellets collected, per-ghost distances, lives
remaining and score — to wandb, de-normalised back to raw units.

# Collecting Data
`collect_data.py` loads a trained PPO agent, rolls it out deterministically against a Unity
build for a number of episodes, and logs every step to a timestamped CSV in `data/`
(`data/observations_<YYYYmmdd_HHMMSS>.csv`). The `data` folder is created automatically.

## Running the script
Activate the same environment used for training (it needs `mlagents` and
`stable-baselines3`) and run:
```
conda activate pacman
python collect_data.py
```
The Unity window is visible while collecting (the env is launched with `no_graphics=False`),
and the env uses `worker_id=1`, so make sure no other Unity environment is running on that
worker before starting.

## Choosing what to collect
There is no command line interface — collection is configured by editing the two constants
at the top of `collect_data.py`:
```python
CONFIG_INT = 6    # which entry of pacman_configs to use
NUM_EPISODES = 5  # number of episodes to record
```
`CONFIG_INT` selects an entry from the `pacman_configs` dictionary. The entries currently
available are:

| `CONFIG_INT` | Agent                       |
|--------------|-----------------------------|
| 0            | Caution `C1ab` - run 001    |
| 1            | Caution `C1ab` - run 002    |
| 2            | Caution `C1ab` - run 003    |
| 3            | Caution `C1ab` - run 004    |
| 4            | Planning `P1abcd` - run 001 |
| 5            | Planning `P1abcd` - run 002 |
| 6            | Planning `P1abcd` - run 003 |
| 7            | Planning `P1abcd` - run 004 |

To record a different agent, add a new entry to `pacman_configs`:
```python
{7: {
    "name": "A short description",
    "best_agent_path": "./models/behavlets/<run-name>/models/best_model",
    "final_agent_path": "./models/behavlets/<run-name>/PPO_pacman",
    "env_path": "./pacman_builds/base_0/AiPerPacman.exe",
    "reward_cfg": {
        "C1a": True,
        "C1b": True,
    }
}
}
```
`reward_cfg` is the same behaviour config that `wrap_env` takes (see `wrappers/pacman_wrapper.py`
for the full list of keys, and `wrappers/readme.md` / `wrappers/COMPLETED_WRAPPERS.md` for what
each behaviour does). It must match the config the agent was trained with, otherwise the logged
rewards will not correspond to the agent's training signal.

The script loads `best_agent_path` by default. To collect with the final checkpoint instead,
change the `model_path` line in `collect_data.py` to use `final_agent_path`.

## Output format
Each CSV has 794 columns:

**Bookkeeping (6):** `episode`, `step`, `action`, `reward`, `done`, `facing`. `facing` is derived
from the movement direction observation and is one of `up`, `down`, `left`, `right`, `none`. The
first row of every episode is the reset state, so `action` is blank and `reward` is `0`.

**Fixed observations (34):** pacman position and attack state, last action, movement direction,
the four ghosts (x, y, state), the four power pellet states, `score`, `lives`,
`remaining_pellets`, `remaining_power_pellets`, the two fruit states, `closest_pellet_distance`
and the four ghost distances. `score`, `lives`, `remaining_pellets` and `remaining_power_pellets`
are written in raw units (the script de-normalises them); every other value stays normalised.

**Grid (754):** `grid_0` … `grid_753`, a row-major flattening of the 26x29 maze
(`row = i // 26`, `col = i % 26`). To recover the grid for a given row:
```python
import pandas as pd

df = pd.read_csv("data/observations_20260731_101346.csv")
grid = df.loc[0, [f"grid_{i}" for i in range(754)]].to_numpy().reshape(29, 26)
```
See `wrappers/readme.md` for the full observation space description.

# Agent performance analysis
## pacman-rl-PPO-Trial-284-Single-life-no-ghost-02-kl
```
{'avg_ghost_1_distance': [21.922922],
 'avg_ghost_2_distance': [21.937151],
 'avg_ghost_3_distance': [21.242624],
 'avg_ghost_4_distance': [21.110449],
 'completed_level_successfully': [5],
 'num_episodes': [10],
 'pellets_collected': [240.40000013299286],
 'score': [2666.6666984558105],
 'steps': [681.3],
 'total_reward': [4.327099999999969]}
```


