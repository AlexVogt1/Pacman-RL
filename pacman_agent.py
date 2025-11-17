import mlagents
import numpy as np
import os
# from IPython.display import HTML, display
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
# from onnx.examples.check_model import model_path
from stable_baselines3.common.monitor import Monitor
from wrappers import wrap_env

from stable_baselines3 import PPO
from stable_baselines3 import DQN

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    unity_env = UnityEnvironment(file_name="./pacman_builds/grid_data_obs/AiPerPacman.exe", no_graphics=False,worker_id=1)
    # unity_env = UnityEnvironment(file_name="../Pacman-Unity_AiPerCog/server_build/AiPerPacman.exe", no_graphics=True) # path to unity build for faster debugging of observation space
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)
    env = wrap_env(env, skip=4)
    print(env.observation_space)

    # Reset environment
    obs = env.reset()
    # unity_env_eval = UnityEnvironment(file_name="./pacman_builds/dist_obs/AiPerPacman.exe", no_graphics=False,
    #                                   worker_id=2)
    # # unity_env = UnityEnvironment(file_name="../Pacman-Unity_AiPerCog/server_build/AiPerPacman.exe", no_graphics=True) # path to unity build for faster debugging of observation space
    # eval_env = UnityToGymWrapper(unity_env_eval, allow_multiple_obs=False)
    # # eval_env = Monitor(eval_env)
    # eval_obs = eval_env.reset()

    # get model
    # model_path="./baseline_model/ppo_model_1000000_steps.zip"
    # model = PPO.load(model_path)
    # model_path="./logs/pacman-rl-test/pacman-rl-DQN-grid-skip-4/DQN_pacman.zip"
    model_path= "logs/optuna/models/trial_20/model.zip"
    model = PPO.load(model_path)

    # # Run a few episodes of random actions
    num_episodes = 5
    max_steps = 50


    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            # Sample a random action
            # action = env.action_space.sample()
            action, _states = model.predict(obs, deterministic=True)
            # print(action)
            # print(action.shape)
            # obs, reward, done, info = env.step(action)
            # Apply the action
            obs, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                break

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
        print(obs[25])

    env.close()
    unity_env.close()