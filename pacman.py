import mlagents
import numpy as np
import os
# from IPython.display import HTML, display
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from wrappers import wrap_env

def check_pellet_grid(obs: np.ndarray):
    grid = obs[30:]
    grid1 = grid.reshape((29,26))

    np.savetxt("./grid_29_26.txt", grid1.astype(int), fmt="%i",delimiter="")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("anything")
    unity_env = UnityEnvironment(file_name="./pacman_builds/grid_data_obs/AiPerPacman.exe", no_graphics=False,worker_id=1)
    # unity_env = UnityEnvironment(file_name="../Pacman-Unity_AiPerCog/windows/AiPerPacman.exe", no_graphics=False,worker_id=3) # path to unity build for faster debugging of observation space
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)
    env=wrap_env(env, skip=4)
    print(env.observation_space)

    # Reset environment
    obs = env.reset()

    # # Run a few episodes of random actions
    num_episodes = 1
    max_steps = 1

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        for step in range(max_steps):
            # Sample a random action
            action = env.action_space.sample()

            # Apply the action
            obs, reward, done, info = env.step(action)
            print(obs)
            # print(check_pellet_grid(obs))
            total_reward += reward

            if done:
                break
        print(check_pellet_grid(obs))

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()
    # unity_env.close()