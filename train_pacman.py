import gym
import os
import torch
from pprint import pprint
# from baselines
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

def check_avaliable_devices():
    # Check if CUDA is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # CUDA details if available
    if device.type == 'cuda':
        print('CUDA Device Name:', torch.cuda.get_device_name(0))
        print('Number of Available GPUs:', torch.cuda.device_count())
        print('Current CUDA Device:', torch.cuda.current_device())
        print('Memory Usage:')
        print('  Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('  Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

def train_ppo_unity_baseline(env_path: str,
                             model_save_path: str,
                             timesteps: int = 1_000_000,
                             eval_freq: int = 10_000,
                             n_eval_episodes: int = 5,
                             checkpoint_freq: int = 50_000):
    """
    Train a PPO agent in a Unity environment.

    Args:
        env_path (str): Path to the Unity executable.
        model_save_path (str): Where to save the trained model.
        timesteps (int): Total timesteps for training.
        eval_freq (int): Frequency of evaluation (timesteps).
        n_eval_episodes (int): Episodes to run during evaluation.
        checkpoint_freq (int): Frequency of saving checkpoints.
    """

    # Load Unity Environment
    unity_env = UnityEnvironment(file_name=env_path, no_graphics=False)
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False,flatten_branched=True)

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=model_save_path, name_prefix="ppo_model")
    # eval_callback = EvalCallback(eval_env=Monitor(env),
    #                              best_model_save_path=model_save_path + "/best_model",
    #                              log_path=model_save_path + "/logs",
    #                              eval_freq=eval_freq,
    #                              n_eval_episodes=n_eval_episodes,
    #                              deterministic=True) look at this https://stackoverflow.com/questions/75415737/accessing-training-metrics-in-stable-baselines3

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(model_save_path, "tensorboard"),device='cuda')

    # Train
    model.learn(total_timesteps=timesteps, callback=[checkpoint_callback])
    # model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback]) # eval needs to be fixed. environment may need to be wrapped in monitor

    # Save final model
    model.save(os.path.join(model_save_path, "ppo_pacman"))

    env.close()
    unity_env.close()
    print(f"Training complete. Model saved to {model_save_path}")


def main():
    # Get the exe path
    pacman_exe ="./pacman_builds/small_obs/AiPerPacman.exe"
    # Make config so that we can track variables like what obs was being used as well as hyperparams

    #Check devices for training
    check_avaliable_devices()

    # train the Agent
    train_ppo_unity_baseline(env_path=pacman_exe,
                           model_save_path="baseline_model",
                           timesteps=1_000_000,
                           eval_freq=2_000,
                           n_eval_episodes=5,
                           checkpoint_freq=6_000)




if __name__ == '__main__':
  main()