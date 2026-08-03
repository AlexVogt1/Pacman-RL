import csv
import os
import time
from typing import List, Optional

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3 import PPO

from wrappers import wrap_env

GRID_WIDTH = 26
GRID_HEIGHT = 29
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT  # 754

# Named columns for the 34 fixed observations (indices 0-33), see wrappers/readme.md
FIXED_OBS_COLUMNS = [
    "pacman_x", "pacman_y", "pacman_z",              # 0-2
    "pacman_attack_state",                           # 3
    "last_action",                                   # 4
    "move_dir_x", "move_dir_y",                      # 5-6
    "ghost1_x", "ghost1_y", "ghost1_state",          # 7-9
    "ghost2_x", "ghost2_y", "ghost2_state",          # 10-12
    "ghost3_x", "ghost3_y", "ghost3_state",          # 13-15
    "ghost4_x", "ghost4_y", "ghost4_state",          # 16-18
    "power_pellet1_state", "power_pellet2_state",    # 19-20
    "power_pellet3_state", "power_pellet4_state",    # 21-22
    "score", "lives",                                # 23-24 (un-normalised)
    "remaining_pellets",                             # 25 (un-normalised)
    "remaining_power_pellets",                       # 26 (un-normalised)
    "fruit1_state", "fruit2_state",                  # 27-28
    "closest_pellet_distance",                       # 29
    "ghost1_distance", "ghost2_distance",            # 30-31
    "ghost3_distance", "ghost4_distance",            # 32-33
]

# grid_i is row-major: row = i // 26, col = i % 26
# reshape with np.array(row[grid_cols]).reshape(29, 26)
GRID_COLUMNS = [f"grid_{i}" for i in range(GRID_SIZE)]

CSV_COLUMNS = ["episode", "step", "action", "reward", "done", "facing"] + FIXED_OBS_COLUMNS + GRID_COLUMNS

# Movement direction observation indices (Unity Vector2: +x = right, +y = up)
MOVE_DIR_X, MOVE_DIR_Y = 5, 6

##########################################
## Global Variables for data collection ##
##########################################
CONFIG_INT = 6 # INTEGER TO DESIGNATE WITH AGENT CONFIG TO USE FROM BELOW
NUM_EPISODES = 5 # number of episodes to record

pacman_configs= {
    0 :{
        'name': 'Caution 1 a & b',
        "best_agent_path":"./models/behavlets/Pacman-rl-PPO-C1ab-001/models/best_model",
        "final_agent_path":"./models/behavlets/Pacman-rl-PPO-C1ab-001/PPO_pacman",
        "env_path":"./pacman_builds/base_0/AiPerPacman.exe",
        "reward_cfg": {
            "C1a": True,
            "C1b": True,
        }
    },
    1 :{
        'name': 'Caution 1 a & b',
        "best_agent_path":"./models/behavlets/Pacman-rl-PPO-C1ab-002/models/best_model",
        "final_agent_path":"./models/behavlets/Pacman-rl-PPO-C1ab-002/PPO_pacman",
        "env_path":"./pacman_builds/base_0/AiPerPacman.exe",
        "reward_cfg": {
            "C1a": True,
            "C1b": True,
        }
    },
    2 :{
        'name': 'Caution 1 a & b',
        "best_agent_path":"./models/behavlets/Pacman-rl-PPO-C1ab-003_1/models/best_model",
        "final_agent_path":"./models/behavlets/Pacman-rl-PPO-C1ab-003_1/PPO_pacman",
        "env_path":"./pacman_builds/base_0/AiPerPacman.exe",
        "reward_cfg": {
            "C1a": True,
            "C1b": True,
        }
    },
    3 :{
        'name': 'Caution 1 a & b',
        "best_agent_path":"./models/behavlets/Pacman-rl-PPO-C1ab-004_1/models/best_model",
        "final_agent_path":"./models/behavlets/Pacman-rl-PPO-C1ab-004_1/PPO_pacman",
        "env_path":"./pacman_builds/base_0/AiPerPacman.exe",
        "reward_cfg": {
            "C1a": True,
            "C1b": True,
        }
    },
    4: {
        "name": "P1 a,b,c & d",
        "best_agent_path":"./models/behavlets/Pacman-rl-PPO-P1abcd-001/models/best_model",
        "final_agent_path":"./models/behavlets/Pacman-rl-PPO-P1abcd-001/PPO_pacman",
        "env_path":"./pacman_builds/base_0/AiPerPacman.exe",
        "reward_cfg": {
            "P1a": True,
            "P1b": True,
            "P1c": True,
            "P1d": True,
        }
    },
    5: {
        "name": "P1 a,b,c & d",
        "best_agent_path":"./models/behavlets/Pacman-rl-PPO-P1abcd-002/models/best_model",
        "final_agent_path":"./models/behavlets/Pacman-rl-PPO-P1abcd-002/PPO_pacman",
        "env_path":"./pacman_builds/base_0/AiPerPacman.exe",
        "reward_cfg": {
            "P1a": True,
            "P1b": True,
            "P1c": True,
            "P1d": True,
        }
    },
    6: {
        "name": "P1 a,b,c & d",
        "best_agent_path":"./models/behavlets/Pacman-rl-PPO-P1abcd-003/models/best_model",
        "final_agent_path":"./models/behavlets/Pacman-rl-PPO-P1abcd-003/PPO_pacman",
        "env_path":"./pacman_builds/base_0/AiPerPacman.exe",
        "reward_cfg": {
            "P1a": True,
            "P1b": True,
            "P1c": True,
            "P1d": True,
        }
    },
    7: {
        "name": "P1 a,b,c & d",
        "best_agent_path":"./models/behavlets/Pacman-rl-PPO-P1abcd-004/models/best_model",
        "final_agent_path":"./models/behavlets/Pacman-rl-PPO-P1abcd-004/PPO_pacman",
        "env_path":"./pacman_builds/base_0/AiPerPacman.exe",
        "reward_cfg": {
            "P1a": True,
            "P1b": True,
            "P1c": True,
            "P1d": True,
        }
    }
}


def direction_from_vector(dx: float, dy: float) -> str:
    """Convert movement.direction (x, y) at obs[5], obs[6] to a string."""
    if dx > 0.5:  return "right"
    if dx < -0.5: return "left"
    if dy > 0.5:  return "up"
    if dy < -0.5: return "down"
    return "none"
# Denormalisation factors, see wrappers/readme.md: score/3200, lives/3,
# remaining pellets/244, remaining power pellets/4
DENORM_FACTORS = {23: 3200.0, 24: 3.0, 25: 244.0, 26: 4.0}


def make_row(episode: int, step: int, action: Optional[int], reward: float,
             done: bool, obs: np.ndarray) -> List:
    values = list(obs)
    for idx, factor in DENORM_FACTORS.items():
        values[idx] = values[idx] * factor
    return [episode, step, "" if action is None else int(action), reward, done,
            direction_from_vector(obs[MOVE_DIR_X], obs[MOVE_DIR_Y])] + values


if __name__ == '__main__':
    unity_env = UnityEnvironment(file_name=pacman_configs[CONFIG_INT]['env_path'],
                                 no_graphics=False, worker_id=1)
    env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=False)
    env = wrap_env(env, skip=4, wrap_reward='normalise', step_reward=-0.0, cfg=pacman_configs[CONFIG_INT]["reward_cfg"])
    print(pacman_configs[CONFIG_INT]["reward_cfg"])
    model_path = pacman_configs[CONFIG_INT]["best_agent_path"]
    # model_path = pacman_configs[0]["final_agent_path"]
    model = PPO.load(model_path, device='cpu')

    num_episodes = NUM_EPISODES

    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", f"observations_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)

        for episode in range(num_episodes):
            obs = env.reset()
            writer.writerow(make_row(episode, 0, None, 0.0, False, obs))

            done = False
            step = 0
            total_reward = 0.0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                step += 1
                total_reward += reward
                writer.writerow(make_row(episode, step, action, reward, done, obs))

            f.flush()
            print(f"Episode {episode + 1} finished: {step} steps, total reward: {total_reward}")

    print(f"Saved observation data to {csv_path}")

    try: # Close envs if they did not close properly
        env.close()
        unity_env.close()
    except:
        pass
