import mlagents
import numpy as np
import os
# from IPython.display import HTML, display
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    unity_env = UnityEnvironment(file_name="../Pacman-Unity_AiPerCog/windows/AiPerPacman.exe", no_graphics=False)
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)
    print(env.observation_space)

    # Reset environment
    obs = env.reset()

    # # Run a few episodes of random actions
    num_episodes = 5
    max_steps = 1000

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        for step in range(max_steps):
            # Sample a random action
            action = env.action_space.sample()

            # Apply the action
            obs, reward, done, info = env.step(action)
            # print(obs)
            total_reward += reward

            if done:
                print(f"Episode {episode + 1} finished with total reward: {total_reward}")
                break

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()
    # unity_env.close()