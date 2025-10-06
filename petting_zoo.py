import mlagents
import numpy as np
import os
# from IPython.display import HTML, display
from mlagents_envs.envs import StrikersVsGoalie  # import unity environment

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = StrikersVsGoalie.env()
    num_cycles = 10

    env.reset()
    for agent in env.agent_iter(env.num_agents * num_cycles):
        prev_observe, reward, done, info = env.last()
        if isinstance(prev_observe, dict) and 'action_mask' in prev_observe:
            action_mask = prev_observe['action_mask']
        if done:
            action = None
        else:
            action = env.action_spaces[agent].sample()  # randomly choose an action for example
        env.step(action)