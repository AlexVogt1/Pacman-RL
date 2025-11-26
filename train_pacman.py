import gym
import os
import argparse
import json
from typing import Callable, Any
from pathlib import Path
import torch
import wandb
# from pprint import pprint
from dataclasses import dataclass
# # from baselines
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, VecEnv, SubprocVecEnv
# from supersuit import observation_lambda_v0

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from wrappers import wrap_env
from mlagents_envs.registry import UnityEnvRegistry, default_registry
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig,
    EngineConfigurationChannel,
)
import numpy as np
from gym import Env

NUM_ENVS = 4
FRAME_SKIP=True
FRAMES_TO_SKIP=4
ENV_PATH=None

# Default values from CLI (See cli_utils.py)
DEFAULT_ENGINE_CONFIG = EngineConfig(
    width=584,
    height=584,
    quality_level= 4,
    time_scale= 5, #anything more than 11 and physics breaks
    target_frame_rate=-1,
    capture_frame_rate=60,
)

# Some config subset of an actual config.yaml file for MLA.
@dataclass
class LimitedConfig:
    # The local path to a Unity executable or the name of an entry in the registry.
    env_path_or_name: str
    base_port: int
    base_seed: int = 0
    num_env: int = 1
    engine_config: EngineConfig = DEFAULT_ENGINE_CONFIG
    visual_obs: bool = False
    # TODO: Decide if we should just tell users to always use MultiInputPolicy so we can simplify the user workflow.
    # WARNING: Make sure to use MultiInputPolicy if you turn this on.
    allow_multiple_obs: bool = True
    env_registry: UnityEnvRegistry = default_registry

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

def _unity_env_from_path_or_registry(env: str, registry: UnityEnvRegistry, **kwargs: Any) -> UnityEnvironment:
    if Path(env).expanduser().absolute().exists():
        return UnityEnvironment(file_name=env, **kwargs)
    elif env in registry:
        return registry.get(env).make(**kwargs)
    else:
        raise ValueError(f"Environment '{env}' wasn't a local path or registry entry")

def make_mla_sb3_env(config: LimitedConfig, **kwargs: Any) -> VecEnv:
    def handle_obs(obs, space):
        if isinstance(space, gym.spaces.Tuple):
            if len(space) == 1:
                return obs[0]
            # Turn the tuple into a dict (stable baselines can handle spaces.Dict but not spaces.Tuple).
            return {str(i): v for i, v in enumerate(obs)}
        return obs

    def handle_obs_space(space):
        if isinstance(space, gym.spaces.Tuple):
            if len(space) == 1:
                return space[0]
            # Turn the tuple into a dict (stable baselines can handle spaces.Dict but not spaces.Tuple).
            return gym.spaces.Dict({str(i): v for i, v in enumerate(space)})
        return space

    def create_env(env: str, worker_id: int) -> Callable[[], Env]:
        def _f() -> Env:
            engine_configuration_channel = EngineConfigurationChannel()
            engine_configuration_channel.set_configuration(config.engine_config)
            kwargs["side_channels"] = kwargs.get("side_channels", []) + [
                engine_configuration_channel
            ]
            unity_env = _unity_env_from_path_or_registry(
                env=env,
                registry=config.env_registry,
                worker_id=worker_id,
                base_port=config.base_port,
                seed=config.base_seed + worker_id,
                **kwargs,
            )
            new_env = UnityToGymWrapper(
                unity_env=unity_env,
                # uint8_visual=config.visual_obs,
                # allow_multiple_obs=config.allow_multiple_obs,
                flatten_branched=True
            )
            # new_env = observation_lambda_v0(new_env, handle_obs, handle_obs_space)
            #frameskip wrap
            if FRAME_SKIP:
                return wrap_env(new_env, skip=FRAMES_TO_SKIP)
            else:
                return new_env

        return _f

    env_facts = [
        create_env(config.env_path_or_name, worker_id=x) for x in range(config.num_env)
    ]
    return SubprocVecEnv(env_facts)

def make_mla_sb3_eval_env(config: LimitedConfig, **kwargs: Any) -> VecEnv:
    def handle_obs(obs, space):
        if isinstance(space, gym.spaces.Tuple):
            if len(space) == 1:
                return obs[0]
            # Turn the tuple into a dict (stable baselines can handle spaces.Dict but not spaces.Tuple).
            return {str(i): v for i, v in enumerate(obs)}
        return obs

    def handle_obs_space(space):
        if isinstance(space, gym.spaces.Tuple):
            if len(space) == 1:
                return space[0]
            # Turn the tuple into a dict (stable baselines can handle spaces.Dict but not spaces.Tuple).
            return gym.spaces.Dict({str(i): v for i, v in enumerate(space)})
        return space

    def create_env(env: str, worker_id: int) -> Callable[[], Env]:
        def _f() -> Env:
            engine_configuration_channel = EngineConfigurationChannel()
            engine_configuration_channel.set_configuration(config.engine_config)
            kwargs["side_channels"] = kwargs.get("side_channels", []) + [
                engine_configuration_channel
            ]
            unity_env = _unity_env_from_path_or_registry(
                env=env,
                registry=config.env_registry,
                worker_id=worker_id,
                base_port=config.base_port,
                seed=config.base_seed + worker_id,
                **kwargs,
            )
            new_env = UnityToGymWrapper(
                unity_env=unity_env,
                # uint8_visual=config.visual_obs,
                # allow_multiple_obs=config.allow_multiple_obs,
                flatten_branched=True
            )
            # new_env = observation_lambda_v0(new_env, handle_obs, handle_obs_space)
            #frameskip wrap
            if FRAME_SKIP:
                return wrap_env(new_env, skip=FRAMES_TO_SKIP)
            else:
                return new_env

        return _f

    env_facts = [
        create_env(config.env_path_or_name, worker_id=config.num_env+1)
    ]
    return SubprocVecEnv(env_facts)

def create_env(is_eval:bool=False):
    """
    Create and return your Pacman environment.
    Replace this with your actual environment creation code.
    """
    if is_eval:
        env = make_mla_sb3_eval_env(
            config=LimitedConfig(
                env_path_or_name=ENV_PATH,
                # Can use any name from a registry or a path to your own unity build.
                base_port=6006,
                base_seed=42,
                num_env=NUM_ENVS,
                allow_multiple_obs=True,
            ),
            no_graphics=True,
            # Set to false if you are running locally and want to watch the environments move around as they train.
        )
        return VecMonitor(env)
    else:
        env = make_mla_sb3_env(
            config=LimitedConfig(
                env_path_or_name=ENV_PATH,
                # Can use any name from a registry or a path to your own unity build.
                base_port=6006,
                base_seed=42,
                num_env=NUM_ENVS,
                allow_multiple_obs=True,
            ),
            no_graphics=True,
            # Set to false if you are running locally and want to watch the environments move around as they train.
        )
        return VecMonitor(env)

class CustomMetricEvalCallback(EvalCallback):
    """
    Evaluation callback that tracks a custom metric from observations.
    Use this for full training runs (not Optuna optimization).
    """

    def __init__(
            self,
            eval_env,
            callback_on_new_best=None,
            n_eval_episodes=5,
            eval_freq=10000,
            log_path=None,
            best_model_save_path=None,
            deterministic=True,
            render=False,
            verbose=1,
            warn=True,
            use_wandb=False,
            custom_metric_idx=None,
            custom_metric_name="custom_metric"
    ):
        """
        Args:
            eval_env: Environment used for evaluation
            callback_on_new_best: Callback triggered when new best model is found
            n_eval_episodes: Number of episodes to evaluate
            eval_freq: Evaluate every N steps
            log_path: Path to save evaluation results
            best_model_save_path: Path to save best model
            deterministic: Whether to use deterministic actions
            render: Whether to render during evaluation
            verbose: Verbosity level
            warn: Whether to warn about Vec/Monitor wrapping
            use_wandb: Whether to log to Weights & Biases
            custom_metric_idx: Index in observation vector to track (None to disable)
            custom_metric_name: Name for the custom metric in logs
        """
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn
        )
        self.use_wandb = use_wandb
        self.custom_metric_idx = custom_metric_idx
        self.custom_metric_name = custom_metric_name
        self.custom_metrics_history = []  # Store all custom metric values
        self.best_custom_metric = -np.inf

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Perform evaluation (with custom metric tracking if enabled)
            if self.custom_metric_idx is not None:
                # Run custom evaluation
                episode_rewards = []
                episode_lengths = []
                episode_custom_metrics = []

                for episode_idx in range(self.n_eval_episodes):
                    obs = self.eval_env.reset()
                    done = False
                    state = None
                    episode_reward = 0.0
                    episode_length = 0
                    last_obs = obs
                    obs_list = []


                    while not done:
                        action, state = self.model.predict(
                            obs, state=state, deterministic=self.deterministic
                        )
                        obs, reward, done, _info = self.eval_env.step(action)

                        # Handle vectorized reward
                        episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                        episode_length += 1
                        last_obs = obs
                        obs_list.append(last_obs)

                        if self.render:
                            self.eval_env.render()

                    # Extract custom metric from last observation
                    last_obs = obs_list[-2]
                    try:
                        # Handle vectorized env (shape: (1, obs_dim))
                        if isinstance(last_obs, np.ndarray) and last_obs.ndim > 1:
                            custom_value = float(last_obs[0][self.custom_metric_idx])
                        else:
                            custom_value = float(last_obs[self.custom_metric_idx])

                        custom_value = (1-custom_value)*244
                        episode_custom_metrics.append(custom_value)
                    except (IndexError, TypeError) as e:
                        if self.verbose > 0:
                            print(f"Warning: Could not extract custom metric: {e}")
                        episode_custom_metrics.append(0.0)

                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

                # Calculate statistics
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_ep_length = np.mean(episode_lengths)
                mean_custom_metric = np.mean(episode_custom_metrics)
                std_custom_metric = np.std(episode_custom_metrics)

                # Update internal state
                self.last_mean_reward = mean_reward
                self.custom_metrics_history.append(mean_custom_metric)

                # Check if this is the best model (based on reward)
                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print(f"New best mean reward: {mean_reward:.2f}")

                    self.best_mean_reward = mean_reward

                    # Save best model
                    # Check if path exists
                    if self.best_model_save_path is not None:
                        # Check directory exists
                        os.makedirs(self.best_model_save_path, exist_ok=True)
                        # Save new best model to path
                        save_path = os.path.join(self.best_model_save_path, "best_model")
                        self.model.save(save_path)
                        if self.verbose > 0:
                            print(f"Saved model to {save_path}")

                    # Trigger callback
                    if self.callback is not None:
                        return self._on_event()

                # Track best custom metric separately
                if mean_custom_metric > self.best_custom_metric:
                    self.best_custom_metric = mean_custom_metric
                    if self.verbose > 0:
                        print(f"New best {self.custom_metric_name}: {mean_custom_metric:.2f}")

                # Print evaluation results
                if self.verbose > 0:
                    print(f"Eval at step {self.num_timesteps}:")
                    print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"  Mean {self.custom_metric_name}: {mean_custom_metric:.2f} +/- {std_custom_metric:.2f}")
                    print(f"  Mean episode length: {mean_ep_length:.2f}")

                # Log to W&B
                if self.use_wandb:
                    wandb.log({
                        "eval/mean_reward": mean_reward,
                        "eval/std_reward": std_reward,
                        "eval/mean_ep_length": mean_ep_length,
                        f"eval/mean_{self.custom_metric_name}": mean_custom_metric,
                        f"eval/std_{self.custom_metric_name}": std_custom_metric,
                        f"eval/best_{self.custom_metric_name}": self.best_custom_metric,
                        "eval/best_mean_reward": self.best_mean_reward,
                        "time/total_timesteps": self.num_timesteps
                    })

                # Log to file if path provided
                if self.log_path is not None:
                    self.evaluations_timesteps.append(self.num_timesteps)
                    self.evaluations_results.append(episode_rewards)
                    self.evaluations_length.append(episode_lengths)

                    np.savez(
                        self.log_path,
                        timesteps=self.evaluations_timesteps,
                        results=self.evaluations_results,
                        ep_lengths=self.evaluations_length,
                        custom_metrics=self.custom_metrics_history
                    )
            else:
                # Use default evaluation (no custom metric)
                super()._on_step()

                # Still log to W&B if enabled
                if self.use_wandb:
                    try:
                        std_reward = np.std(self.evaluations_results[-1]) if len(self.evaluations_results) > 0 else 0.0
                    except (IndexError, TypeError):
                        std_reward = 0.0

                    wandb.log({
                        "eval/mean_reward": self.last_mean_reward,
                        "eval/std_reward": std_reward,
                        "eval/best_mean_reward": self.best_mean_reward,
                        "time/total_timesteps": self.num_timesteps
                    })

        return True

def train_ppo_unity_baseline(env_path: str,
                             model_save_path: str,
                             timesteps: int = 1_000_000,
                             eval_freq: int = 10_000,
                             n_eval_episodes: int = 5,
                             checkpoint_freq: int = 50_000,
                             config: dict = None):
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
    print(f"Using build {env_path}")
    # unity_env = UnityEnvironment(file_name=env_path, no_graphics=True)
    # env = UnityToGymWrapper(unity_env, allow_multiple_obs=False,flatten_branched=True)
    # env = wrap_env(env, skip=4)
    env = create_env(is_eval=False)
    # env = make_mla_sb3_env(
    #     config=LimitedConfig(
    #         env_path_or_name=env_path,
    #         # Can use any name from a registry or a path to your own unity build.
    #         base_port=6006,
    #         base_seed=42,
    #         num_env=NUM_ENVS,
    #         allow_multiple_obs=True,
    #     ),
    #     no_graphics=True,
    #     # Set to false if you are running locally and want to watch the environments move around as they train.
    # )
    env = VecMonitor(env)

    #load evaluation env
    eval_env = create_env(is_eval=True)
    eval_env = VecMonitor(eval_env)

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=model_save_path, name_prefix="ppo_model")
    eval_callback = CustomMetricEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        best_model_save_path=f"./{model_save_path}/models",
        log_path=f"./{model_save_path}/evaluations",
        use_wandb=True,
        custom_metric_idx=25,  # Index of score/lives/pellets in observation
        custom_metric_name="pellets_collected"  # Name for logging
    )

    # Create PPO model
    model = PPO("MlpPolicy", env,
                learning_rate=config['learning_rate'],
                n_steps=config['n_steps'],
                batch_size=config['batch_size'],
                n_epochs=config['n_epochs'],
                gamma=config['gamma'],
                gae_lambda=config['gae_lambda'],
                clip_range=config['clip_range'],
                ent_coef=config['ent_coef'],
                vf_coef=config['vf_coef'],
                max_grad_norm=config['max_grad_norm'],
                policy_kwargs=config['policy_kwargs'],
                verbose=1, tensorboard_log=os.path.join(model_save_path, "tensorboard"),device='cpu',seed=42)
    # model = DQN("MlpPolicy", env,
    #             train_freq=config["train_freq"],
    #             learning_rate=config["learning_rate"],
    #             exploration_fraction=config["exploration_fraction"],
    #             exploration_final_eps=config["exploration_final_eps"],
    #             verbose=1, tensorboard_log=os.path.join(model_save_path, "tensorboard"),device='cuda')

    # Train
    # model.learn(total_timesteps=timesteps, callback=[checkpoint_callback,eval_callback])
    model.learn(total_timesteps=timesteps, callback=[checkpoint_callback,eval_callback]) # eval needs to be fixed. environment may need to be wrapped in monitor

    # Save final model
    # model.save(os.path.join(model_save_path, "ppo_pacman"))
    model.save(os.path.join(model_save_path, "PPO_pacman"))

    env.close()
    # unity_env.close()
    print(f"Training complete. Model saved to {model_save_path}")


def parse_args():
    parser = argparse.ArgumentParser("Pacman Training")
    parser.add_argument("--json_path", type=str, default="./exp/base/exp_001.json",
                        help="directory for expeiment parameters")

    return parser.parse_args()
def main(config: dict):

    # Make config so that we can track variables like what obs was being used as well as hyperparams

    #Check devices for training
    check_avaliable_devices()
    target_freq= 50_000
    correct_freq= target_freq // NUM_ENVS

    # train the Agent
    train_ppo_unity_baseline(env_path=config['pacman_exe_path'],
                            model_save_path=config["model_save_path"],
                            timesteps=2_000_000,
                            eval_freq=correct_freq,
                            n_eval_episodes=10,
                            checkpoint_freq=100_000,
                            config=config["ppo_config"])




if __name__ == '__main__':
    args = parse_args()
    with open(f"{args.json_path}.json", 'r') as f:
        config = json.load(f)
    # wandb config
    wandb_config = config["exp_config"]
    dqn_config = {
        "policy": "MlpPolicy",
        "learning_rate": 1e-3,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 50,
        "gradient_steps": 1,
        "replay_buffer_class": None,
        "replay_buffer_kwargs": None,
        "optimize_memory_usage": False,
        "n_steps": 1,
        "target_update_interval": 10_000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.075,
        "max_grad_norm": 10,
        "stats_window_size": 100,
        "tensorboard_log": None,
        "policy_kwargs": None,
        "verbose": 0,
        "seed": None,
        "device": "cuda",
        "_init_setup_model": True
    }
    # ppo_config = {
    #     "learning_rate": 0.0008541161764262368,
    #     "n_steps": 512,
    #     "batch_size": 64,
    #     "n_epochs": 18,
    #     "gamma": 0.9991964773396101,
    #     "gae_lambda": 0.9011887172355716,
    #     "clip_range": 0.18185377161463595,
    #     "clip_range_vf": None,
    #     "normalize_advantage": True,
    #     "ent_coef": 0.00697442595256507,
    #     "vf_coef": 0.6757818160485769,
    #     "max_grad_norm": 1.0134150593938502,
    #     "use_sde": False,
    #     "sde_sample_freq": -1,
    #     "rollout_buffer_class": None,
    #     "rollout_buffer_kwargs": None,
    #     "target_kl": None,
    #     "stats_window_size": 100,
    #     "tensorboard_log": None,
    #     "policy_kwargs": {"net_arch":  [dict(pi=[256, 256], vf=[256, 256])]},
    #     "verbose": 0,
    #     "seed": None,
    #     "device": 'auto',
    #     "_init_setup_model": True
    # }
    ppo_config= config["ppo_config"]
    config = {
        "wandb_config": wandb_config,
        "pacman_exe_path":wandb_config["pacman_path"],
        "model_save_path": f"./logs/{wandb_config['project']}/{wandb_config['name']}",
        # "dqn_config": dqn_config,
        "ppo_config": ppo_config,
    }
    print(config)
    ENV_PATH = config["pacman_exe_path"]
    os.makedirs(config["model_save_path"], exist_ok=True)
    print(wandb_config)
    wandb.init(project=wandb_config['project'], name=wandb_config['name'],notes=wandb_config['description'],config=config, sync_tensorboard=True)
    main(config)
    wandb.finish()