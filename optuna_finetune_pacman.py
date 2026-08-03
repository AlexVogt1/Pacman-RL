import optuna
from optuna.pruners import MedianPruner, PatientPruner, PercentilePruner
from optuna.samplers import TPESampler
import argparse
import json
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, VecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from wrappers import wrap_env
from mlagents_envs.registry import UnityEnvRegistry, default_registry
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig,
    EngineConfigurationChannel,
)
from typing import Callable, Any
from pathlib import Path
from dataclasses import dataclass
from gym import Env
import gym

import warnings

warnings.filterwarnings('ignore')
# TODO: Save trail hyperparameters to a usable json to make life easier
# TODO: Look into the ratio thing for batch size to avoid bad trials

# Global variables
NUM_ENVS = 16
FRAME_SKIP = True
FRAMES_TO_SKIP = 4
env_path = "./pacman_builds/grid_data_obs/AiPerPacman.exe"
SIM_SPEED = 3
SAVE_PATH = None
TRAIN_STEPS = None
WANDB_PROJECT = None
STUDY_NAME = None

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
    quality_level=4,
    time_scale=SIM_SPEED,  # anything more than 11 and physics breaks
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


# Import your custom Unity ML-Agents environment
# from mlagents_envs.environment import UnityEnvironment
# from gym_unity.envs import UnityToGymWrapper
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
            # frameskip wrap
            if FRAME_SKIP:
                return wrap_env(new_env, skip=FRAMES_TO_SKIP, wrap_reward='normalise')
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
            # frameskip wrap
            if FRAME_SKIP:
                return wrap_env(new_env, skip=FRAMES_TO_SKIP, wrap_reward='normalise')
            else:
                return new_env

        return _f

    env_facts = [
        create_env(config.env_path_or_name, worker_id=config.num_env + 1)
    ]
    return SubprocVecEnv(env_facts)


def create_env(is_eval: bool = False):
    """
    Create and return your Pacman environment.
    Replace this with your actual environment creation code.
    """
    if is_eval:
        env = make_mla_sb3_eval_env(
            config=LimitedConfig(
                env_path_or_name=env_path,
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
                env_path_or_name=env_path,
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
    raise NotImplementedError("Replace this with your Pacman environment creation code")


def objective(trial, wandb_project="ppo-pacman-optuna", use_wandb=True, base_model_path=None):
    """
    Objective function for Optuna optimization.
    Focuses on the most impactful PPO hyperparameters.
    """

    # Initialize W&B for this trial
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            name=f"trial_{trial.number}",
            group="optuna_optimization",
            config={"trial_number": trial.number},
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            reinit=True
        )

    # MOST IMPORTANT HYPERPARAMETERS FOR PPO

    # 1. Learning rate - affects convergence speed
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-3, log=True)

    # 2. Number of steps per update - critical for PPO performance
    # With 2000-step episodes: 512=1/4 episode, 2048=1 episode, 4096=2 episodes
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])

    # 3. Batch size must be <= n_steps, affects gradient quality
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

    # 4. Number of epochs - how many times to reuse collected data
    n_epochs = trial.suggest_int("n_epochs", 3, 15)

    # 5. Gamma - discount factor for long-term rewards
    gamma = trial.suggest_float("gamma", 0.98, 0.99999, log=True)

    # 6. GAE lambda - balances bias vs variance in advantage estimation
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

    # 7. Clip range - PPO's key parameter, limits policy updates
    clip_range = trial.suggest_float("clip_range", 0.05, 0.4)

    # 8. Entropy coefficient - encourages exploration
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.05, log=True)

    # 9. Value function coefficient - balances value vs policy learning
    vf_coef = trial.suggest_float("vf_coef", 0.3, 2.0)

    # 10. Max gradient norm - prevents unstable updates
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0)

    # 11. Whether or not to reset the optimizers
    reset_optimizer = trial.suggest_categorical("reset_optimizer", [True, False])

    # Create environments (PPO requires vectorized env)

    try:
        env.close()
        eval_env.close()
    except:
        pass

    env = create_env(is_eval=False)
    eval_env = create_env(is_eval=True)

    try:
        updated_hyperparams = {
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "gae_lambda": gae_lambda,
            "gamma": gamma,
            "clip_range": clip_range,
            "tensorboard_log": f"{SAVE_PATH}/runs/trial_{trial.number}",
        }

        # Load the model and override the old parameters using custom_objects
        if reset_optimizer:
            pretrained_model = PPO.load(base_model_path)

            model = PPO("MlpPolicy", env,
                            learning_rate=learning_rate,
                            n_steps=n_steps,
                            batch_size=batch_size,
                            n_epochs=n_epochs,
                            gamma=gamma,
                            gae_lambda=gae_lambda,
                            clip_range=clip_range,
                            ent_coef=ent_coef,
                            vf_coef=vf_coef,
                            max_grad_norm=max_grad_norm,
                            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.LeakyReLU),
                            verbose=1,
                            tensorboard_log=f"{SAVE_PATH}/runs/trial_{trial.number}" if use_wandb else None,
                            device='cpu',
                            seed=42)

            # Load pretrained weights to new model
            model.policy.load_state_dict(pretrained_model.policy.state_dict())
            # model.set_parameters()
        else:
            model = PPO.load(
                base_model_path,
                env=env,
                custom_objects=updated_hyperparams
            )


        # Log hyperparameters to W&B
        if use_wandb:
            wandb.config.update({
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "reset_optimizer": reset_optimizer,
            })

        # Validate batch_size <= n_steps
        if batch_size > n_steps:
            raise ValueError(f"batch_size ({batch_size}) must be <= n_steps ({n_steps})")

        # Create callbacks
        callbacks = []

        if use_wandb:
            wandb_callback = WandbCallback(
                model_save_path=f"{SAVE_PATH}/models/trial_{trial.number}",
                verbose=2,
            )
            callbacks.append(wandb_callback)

        # Evaluation callback with pruning
        target_eval_freq = 100000
        correct_eval_freq = target_eval_freq // NUM_ENVS
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=3,
            eval_freq=5500,
            deterministic=True,
            use_wandb=use_wandb,
            # custom_metric_idx=25
            custom_metrics_dict=base_custom_metric_dict
        )
        callbacks.append(eval_callback)

        # Train the model
        model.learn(
            total_timesteps=TRAIN_STEPS,
            callback=callbacks,
            log_interval=1
        )

        # Final evaluation
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10,
            deterministic=True
        )

        # Log final results to W&B
        if use_wandb:
            wandb.log({
                "final_mean_reward": mean_reward,
                "final_std_reward": std_reward,
                "trial_number": trial.number
            })
            wandb.finish()

        try:
            env.close()
            eval_env.close()
        except:
            pass

        return mean_reward

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        if use_wandb:
            wandb.finish(exit_code=1)
        return float('-inf')

    finally:
        env.close()
        eval_env.close()


class TrialEvalCallback(EvalCallback):
    """Callback for evaluating an agent and pruning unpromising trials with multiple custom metrics."""

    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=5000,
                 deterministic=True, use_wandb=False, custom_metrics_dict=None):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=1
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.use_wandb = use_wandb
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

            # Report to Optuna for pruning
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.is_pruned = True
                if self.use_wandb:
                    wandb.log({"pruned": True, "pruned_at_step": self.n_calls})
                return False

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
    # def _evaluate_with_custom_metrics(self):
    #     """Runs evaluation and extracts multiple metrics from the observation vector."""
    #     episode_rewards = []
    #     # Initialize lists for each custom metric to track across episodes
    #     episode_custom_values = {name: [] for name in self.custom_metrics_dict.keys()}
    #
    #     for _ in range(self.n_eval_episodes):
    #         obs = self.eval_env.reset()
    #         done = False
    #         episode_reward = 0.0
    #         obs_list = [obs]
    #
    #         while not done:
    #             action, _ = self.model.predict(obs, deterministic=self.deterministic)
    #             obs, reward, done, info = self.eval_env.step(action)
    #             episode_reward += reward
    #             obs_list.append(obs)
    #
    #         # Stable Baselines vectorized envs reset automatically; use the penultimate observation
    #         last_obs = obs_list[-2] if len(obs_list) > 1 else obs_list[0]
    #
    #         # Extract each metric based on its index
    #         #TODO: get the episode average for the ghost distances
    #         for name, idx in self.custom_metrics_dict.items():
    #             try:
    #                 if isinstance(last_obs, np.ndarray):
    #                     # Handle (1, obs_dim) or (obs_dim,)
    #                     val = last_obs[0][idx] if last_obs.ndim > 1 else last_obs[idx]
    #                 else:
    #                     val = last_obs[idx]
    #                 episode_custom_values[name].append(float(val))
    #             except (IndexError, KeyError, TypeError):
    #                 episode_custom_values[name].append(0.0)
    #
    #         episode_rewards.append(episode_reward)
    #
    #     # Calculate means
    #     mean_reward = np.mean(episode_rewards)
    #     std_reward = np.std(episode_rewards)
    #
    #     mean_metrics = {}
    #     for name, values in episode_custom_values.items():
    #         avg = np.mean(values)
    #         mean_metrics[name] = avg
    #         self.metrics_history[name].append(avg)
    #
    #     return mean_reward, std_reward, mean_metrics



def optimize_hyperparameters(n_trials=100, timeout=None, n_jobs=1, study_name="ppo_pacman",
                             wandb_project="ppo-pacman-optuna", use_wandb=True,base_model=None):
    """Run hyperparameter optimization."""

    if use_wandb:
        wandb.login()

    # Create study
    sampler = TPESampler(n_startup_trials=50)
    # pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    pruner_config = PatientPruner(
        PercentilePruner(n_startup_trials=50, percentile=50.0, n_warmup_steps=18),  # i report eval id not steps
        patience=2
    )

    # Create/Load the study
    db_name = f"{study_name}.db"
    # db_name = "pacman_optuna.db"

    if os.name == 'nt':  # 'nt' means Windows
        # WINDOWS FIX: Force DB to live with the script to avoid path issues
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        db_path = os.path.join(script_dir, db_name)
    else:
        # LINUX/CLUSTER: Do NOT change directory.
        # Assume the user submitted the job from a writable Scratch folder.
        # Just use the current working directory.
        db_path = os.path.join(os.getcwd(), db_name)

    # Create URL (Handle slashes for SQLAlchemy)
    db_path = db_path.replace("\\", "/")
    storage_path = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner_config,
        storage=storage_path,
        load_if_exists=True
    )
    # Check that trials will be saved before running study save time
    print(f"Checking database write permissions for {storage_path}...")
    try:
        # This forces a tiny write operation to the DB immediately
        study.set_user_attr("db_check", "writable")
        print("SUCCESS: Database is writable. Starting training...")
    except Exception as e:
        print(f"CRITICAL ERROR: Cannot write to database. Stopping now.")
        print(f"Error details: {e}")
        exit(1)  # Kill the script immediately

    try:
        study.optimize(
            lambda trial: objective(trial, wandb_project=wandb_project, use_wandb=use_wandb,base_model_path=base_model),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    # Print results
    print("\n" + "=" * 70)
    print("PPO HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    trial = study.best_trial
    print(f"\n{'=' * 70}")
    print(f"BEST TRIAL: #{trial.number}")
    print(f"{'=' * 70}")
    print(f"Mean Reward: {trial.value:.2f}")

    print("\n" + "-" * 70)
    print("OPTIMAL HYPERPARAMETERS:")
    print("-" * 70)
    for key, value in sorted(trial.params.items()):
        print(f"  {key:.<30} {value}")

    # Save best hyperparameters
    import json
    with open(f"{SAVE_PATH}/{study_name}_best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    print(f"\n{'=' * 70}")
    print(f"Best parameters saved to: {study_name}_best_params.json")

    return study


def parse_args():
    parser = argparse.ArgumentParser("Pacman Training")
    parser.add_argument("--json_path", type=str, default="./exp/optuna/optuna_001",
                        help="directory for expeiment parameters")

    return parser.parse_args()


if __name__ == "__main__":

    # Read in json path
    args = parse_args()
    with open(f"{args.json_path}.json", 'r') as f:
        config = json.load(f)

    # Configuration
    USE_WANDB = True  # Set to False to disable W&B logging
    WANDB_PROJECT = config["wandb_project"]
    N_TRIALS = None  # Number of hyperparameter combinations to try. None allows to use clock better for working on cluster
    env_path = config["env_path"]
    TRAIN_STEPS = config["train_steps"]
    SIM_SPEED = config["sim_speed"]
    STUDY_NAME = config["study_name"]

    SAVE_PATH = f'optuna_logs/{WANDB_PROJECT}/{STUDY_NAME}'
    try:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    except:
        print('Something went wrong and could not make directory "' + SAVE_PATH + '"')

    # Run optimization
    study = optimize_hyperparameters(
        n_trials=N_TRIALS,
        timeout=257400,
        n_jobs=1,  # Set to -1 to use all CPU cores (requires parallel environments)
        study_name=config["study_name"],
        wandb_project=WANDB_PROJECT,
        use_wandb=USE_WANDB,
        base_model=config["base_model_path"],
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
