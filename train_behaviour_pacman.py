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

NUM_ENVS = 16
FRAME_SKIP=True
FRAMES_TO_SKIP=4
ENV_PATH=None
CONTINUE_TRAINING = False
PRETRAINED_MODEL_PATH = None
STEP_REWARD = None
CAUTION = None
AGGRESSION = None
base_cfg = {
    "Sp1":False,
    "Sp2":False,
    "C1a":False,
    "C1b":False,
    "C2a":False,
    "C2b":False,
    # "C3":False,
    "C4":False,
    "C5":False,
    "C6":False,
    "C7":False,
    "A1":False,
    "A2":False,
    "A3":False,
    # "A4":False,
    # "A5":False,
    "A6":False,
    "P1a":True,
    "P1b":True,
    "P1c":True,
    "P1d":True,
    "P3":False,
    "P4a":False,
    "P4b":False,
}

#dict is {name: index in obs}
base_custom_metric_dict = {
    "pellets_collected": 25,
    "ghost_1_distance": 30,
    "ghost_2_distance": 31,
    "ghost_3_distance": 32,
    "ghost_4_distance": 33,
    "lives_remaining": 24,
    "score":23,
}
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
                return wrap_env(new_env, skip=FRAMES_TO_SKIP, wrap_reward='normalise', step_reward=STEP_REWARD, caution=CAUTION, aggression = AGGRESSION, cfg = base_cfg)
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
                return wrap_env(new_env, skip=FRAMES_TO_SKIP, wrap_reward='normalise', step_reward=STEP_REWARD, caution=CAUTION, aggression = AGGRESSION, cfg = base_cfg)
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
    """Callback for evaluating an agent and pruning unpromising trials with multiple custom metrics."""

    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=5000,
                 deterministic=True, use_wandb=False, custom_metrics_dict=None,best_model_save_path=None, log_path=None):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=1
        )
        self.eval_idx = 0
        self.use_wandb = use_wandb
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        # Expects a dict like {"metric_name": idx, ...}
        self.custom_metrics_dict = custom_metrics_dict or {}
        # Stores history of means for each metric: {"metric_name": [mean1, mean2, ...]}
        self.metrics_history = {name: [] for name in self.custom_metrics_dict.keys()}

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.custom_metrics_dict:
                mean_reward, std_reward, mean_metrics = self._evaluate_with_custom_metrics()
                self.last_mean_reward = mean_reward
            else:
                super()._on_step()
                mean_metrics = {}
                mean_reward = self.last_mean_reward
                # Assuming std_reward is available or negligible for the log
                std_reward = 0.0

            self.eval_idx += 1

            if self.use_wandb:
                log_dict = {
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "eval/eval_idx": self.eval_idx,
                    "time/total_timesteps": self.n_calls
                }

                # Dynamically add all mean custom metrics to the log
                for name, value in mean_metrics.items():
                    if name == "pellets_collected":
                        log_dict[f"eval/mean_{name}"] = (1 - value) * 244.0
                    elif name == "lives_remaining":
                        log_dict[f"eval/mean_{name}"] = value * 3
                    elif name == "score":
                        log_dict[f"eval/mean_{name}"] = value * 3200
                    else:
                        log_dict[f"eval/mean_{name}"] = value
                wandb.log(log_dict)

                # Log to file if path provided
                # if self.log_path is not None:
                #     self.evaluations_timesteps.append(self.num_timesteps)
                #     self.evaluations_results.append(episode_rewards)
                #     self.evaluations_length.append(episode_lengths)
                #
                #     np.savez(
                #         self.log_path,
                #         timesteps=self.evaluations_timesteps,
                #         results=self.evaluations_results,
                #         ep_lengths=self.evaluations_length,
                #         custom_metrics=self.custom_metrics_history
                #     )

                # Check if we should save a new best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.best_model_save_path is not None:
                        os.makedirs(self.best_model_save_path, exist_ok=True)
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                        print(f"New best reward: {mean_reward:.2f}! Model saved.")

        return True

    def _evaluate_with_custom_metrics(self):
        episode_rewards = []
        # Stores the calculated result (either mean or last-step) for each episode
        episode_results = {name: [] for name in self.custom_metrics_dict.keys()}

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0

            # Record every step for every metric
            step_values = {name: [] for name in self.custom_metrics_dict.keys()}

            while not done:
                # Capture values from the current state (observation)
                for name, idx in self.custom_metrics_dict.items():
                    val = self._extract_value(obs, idx)
                    step_values[name].append(val)

                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward

            # Distinguish logic based on key name
            for name in self.custom_metrics_dict.keys():
                if "distance" in name.lower():
                    # Calculate average distance over all timesteps in the episode
                    episode_results[name].append(np.mean(step_values[name]))
                else:
                    # Capture only the final value (score, pellets, lives)
                    episode_results[name].append(step_values[name][-1])

            episode_rewards.append(episode_reward)

        # Calculate final means across the batch of evaluation episodes
        final_means = {name: np.mean(values) for name, values in episode_results.items()}

        for name, mean_val in final_means.items():
            self.metrics_history[name].append(mean_val)

        return np.mean(episode_rewards), np.std(episode_rewards), final_means

    def _extract_value(self, obs, idx):
        try:
            if isinstance(obs, np.ndarray):
                return float(obs[0][idx] if obs.ndim > 1 else obs[idx])
            return float(obs[idx])
        except (IndexError, KeyError, TypeError):
            return 0.0

def make_kwargs(activation_fn_name): #TODO: add net_arch to this function

    if activation_fn_name == "tanh":
        activation_fn = torch.nn.Tanh
    elif activation_fn_name == "relu":
        activation_fn = torch.nn.ReLU
    elif activation_fn_name == "leakyRelu":
        activation_fn = torch.nn.LeakyReLU
    else:
        activation_fn = torch.nn.Mish
    return activation_fn

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
        custom_metrics_dict=base_custom_metric_dict
        # custom_metric_idx=25,  # Index of score/lives/pellets in observation
        # custom_metric_name="pellets_collected"  # Name for logging
    )
    #process kwargs
    sb3_kwargs = dict(net_arch=config['policy_kwargs']["net_arch"],activation_fn=make_kwargs(config['policy_kwargs']["activation_fn"]))
    print(sb3_kwargs)
    # Create PPO model
    if CONTINUE_TRAINING:
        new_lr = config['learning_rate']*0.2
        model = PPO.load(PRETRAINED_MODEL_PATH, env=env, device="cpu", custom_objects={'learning_rate': new_lr})
        model.set_random_seed(911)
        model.learn(total_timesteps=timesteps, callback=[checkpoint_callback,eval_callback], reset_num_timesteps=False)
    else:
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
                    policy_kwargs=sb3_kwargs,
                    verbose=1, tensorboard_log=os.path.join(model_save_path, "tensorboard"),device='cpu',seed=911)
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
    parser.add_argument("--json_path", type=str, default="./exp/base/exp_001",
                        help="directory for expeiment parameters")

    return parser.parse_args()
def main(config: dict):

    # Make config so that we can track variables like what obs was being used as well as hyperparams

    #Check devices for training
    check_avaliable_devices()
    target_freq= 800_000
    correct_freq= target_freq // NUM_ENVS

    # train the Agent
    train_ppo_unity_baseline(env_path=config['pacman_exe_path'],
                            model_save_path=config["model_save_path"],
                            timesteps=160_000_000,
                            eval_freq=correct_freq,
                            n_eval_episodes=10,
                            checkpoint_freq=2_000_000,
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
        "continue_training": wandb_config["continue_training"],
        "pretrained_model_path": wandb_config["pretrained_model_path"],
        "step_penalty": wandb_config["step_penalty"],
    }
    print(config)
    ENV_PATH = config["pacman_exe_path"]
    CONTINUE_TRAINING = config["continue_training"]
    PRETRAINED_MODEL_PATH = config["pretrained_model_path"]

    STEP_REWARD = config["step_penalty"]

    try:
        CAUTION = wandb_config["caution"]
        config["caution"] = True
    except:
        print("Caution behaviour reward was not used")
    try:
        AGGRESSION = wandb_config["aggression"]
        config["aggression"] = True
    except:
        print("Aggression behaviour reward was not used")

    os.makedirs(config["model_save_path"], exist_ok=True)
    print(wandb_config)
    wandb.init(project=wandb_config['project'], name=wandb_config['name'],notes=wandb_config['description'],config=config, sync_tensorboard=True)
    main(config)
    wandb.finish()
