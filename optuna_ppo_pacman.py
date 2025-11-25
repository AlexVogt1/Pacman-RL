import optuna
from optuna.pruners import MedianPruner, PatientPruner, PercentilePruner
from optuna.samplers import TPESampler
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

NUM_ENVS = 4
FRAME_SKIP=True
FRAMES_TO_SKIP=4
env_path="./pacman_builds/grid_data_obs/AiPerPacman.exe"

# Default values from CLI (See cli_utils.py)
DEFAULT_ENGINE_CONFIG = EngineConfig(
    width=584,
    height=584,
    quality_level=4,
    time_scale=5, #anything more than 11 and physics breaks
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


def objective(trial, wandb_project="ppo-pacman-optuna", use_wandb=True):
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
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)

    # 2. Number of steps per update - critical for PPO performance
    # With 2000-step episodes: 512=1/4 episode, 2048=1 episode, 4096=2 episodes
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])

    # 3. Batch size must be <= n_steps, affects gradient quality
    valid_batch_sizes = [bs for bs in [32, 64, 128, 256, 512] if bs <= n_steps]
    if valid_batch_sizes:
        # Sample from valid options
        batch_size = trial.suggest_categorical("batch_size_resampled", valid_batch_sizes)
        # if trial.number == 0 or trial.number % 10 == 0:  # Log occasionally to avoid spam
        #     print(f"Trial {trial.number}: Resampled batch_size to {batch_size} (n_steps={n_steps})")
    else:
        # Fallback: use n_steps as batch_size
        batch_size = n_steps
        print(f"Trial {trial.number}: Using batch_size={n_steps} (same as n_steps)")

    # 4. Number of epochs - how many times to reuse collected data
    n_epochs = trial.suggest_int("n_epochs", 5, 20)

    # 5. Gamma - discount factor for long-term rewards
    gamma = trial.suggest_float("gamma", 0.95, 0.9999, log=True)

    # 6. GAE lambda - balances bias vs variance in advantage estimation
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

    # 7. Clip range - PPO's key parameter, limits policy updates
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    # 8. Entropy coefficient - encourages exploration
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.01, log=True)

    # 9. Value function coefficient - balances value vs policy learning
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)

    # 10. Max gradient norm - prevents unstable updates
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0)

    # 11. Network architecture - model capacity
    net_arch_type = trial.suggest_categorical("net_arch", ['XX-large', 'extra_large_deep', 'huge', 'massive'])
    net_arch_map = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[128, 128], vf=[128, 128])],
        "large": [dict(pi=[256, 256], vf=[256, 256])],
        "extra-large": [dict(pi=[512, 512], vf=[512, 512])],
        "XX-large": [dict(pi=[1024, 1024], vf=[512, 512])],
        "XXX-large": [dict(pi=[2048, 2048], vf=[512,512])],
        "extra_large_deep": [dict(pi=[512, 512, 256], vf=[512, 512, 256])],
        "huge": [dict(pi=[1024, 512, 256], vf=[1024, 512, 256])],
        "massive": [dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512])]
    }
    net_arch = net_arch_map[net_arch_type]

    # Create environments (PPO requires vectorized env)

    try:
        env.close()
        eval_env.close()
    except:
        pass

    env = create_env(is_eval=False)
    eval_env = create_env(is_eval=True)

    try:
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
                "net_arch": net_arch_type,
            })

        # Validate batch_size <= n_steps
        if batch_size > n_steps:
            raise ValueError(f"batch_size ({batch_size}) must be <= n_steps ({n_steps})")

        # Create PPO model with focused hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
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
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1,
            tensorboard_log=f"optuna_search/runs/trial_{trial.number}" if use_wandb else None,
            # device="cuda" if torch.cuda.is_available() else "cpu",
            device= "cpu",
            seed=trial.number,
        )

        # Create callbacks
        callbacks = []

        if use_wandb:
            wandb_callback = WandbCallback(
                model_save_path=f"optuna_search/models/trial_{trial.number}",
                verbose=2,
            )
            callbacks.append(wandb_callback)

        # Evaluation callback with pruning
        target_eval_freq = 50000
        correct_eval_freq = target_eval_freq // NUM_ENVS
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=8,
            eval_freq=correct_eval_freq,
            deterministic=True,
            use_wandb=use_wandb
        )
        callbacks.append(eval_callback)

        # Train the model (100k steps = ~50 episodes)
        model.learn(
            total_timesteps=200000,
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
    """Callback for evaluating an agent and pruning unpromising trials."""

    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=5000,
                 deterministic=True, use_wandb=False, custom_metric_idx=None):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=0
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.use_wandb = use_wandb
        self.custom_metric_idx = 25  # Index in observation vector for custom metric
        self.custom_metrics = []  # Store custom metrics from each episode

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run custom evaluation if we want to track specific observation values
            if self.custom_metric_idx is not None:
                mean_reward, std_reward, mean_custom_metric = self._evaluate_with_custom_metric()
                self.last_mean_reward = mean_reward
            else:
                super()._on_step()
                mean_custom_metric = None

            self.eval_idx += 1

            if self.use_wandb:
                # Safely get std_reward
                try:
                    std_reward = np.std(self.evaluations_results[-1]) if len(self.evaluations_results) > 0 else 0.0
                except (IndexError, TypeError):
                    std_reward = 0.0

                log_dict = {
                    "eval/mean_reward": self.last_mean_reward,
                    "eval/std_reward": std_reward,
                    "eval/eval_idx": self.eval_idx,
                    "time/total_timesteps": self.n_calls
                }

                # Add custom metric if available
                if mean_custom_metric is not None:
                    log_dict["eval/mean_pellets_collected"] = (1-mean_custom_metric)*244.0

                wandb.log(log_dict)

            # Report and check for pruning
            # You can choose to prune based on reward or custom metric
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.is_pruned = True
                if self.use_wandb:
                    wandb.log({"pruned": True, "pruned_at_step": self.n_calls})
                return False

        return True

    def _evaluate_with_custom_metric(self):
        """
        Custom evaluation that tracks a specific value from observations.
        Returns mean_reward, std_reward, and mean_custom_metric.
        """
        episode_rewards = []
        episode_custom_metrics = []


        for episode_idx in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            last_obs = obs
            obs_list=[]

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                last_obs = obs
                obs_list.append(last_obs)
            last_obs =obs_list[-2] #env resets for some reason so last obs is actually the second last

            # Extract custom metric from last observation of the episode
            # For vectorized env, need to handle differently
            try:
                if isinstance(last_obs, np.ndarray):
                    # For DummyVecEnv, obs shape is (1, obs_dim)
                    if last_obs.ndim > 1:
                        custom_value = last_obs[0][self.custom_metric_idx]
                    else:
                        custom_value = last_obs[self.custom_metric_idx]
                elif isinstance(last_obs, (list, tuple)):
                    custom_value = last_obs[self.custom_metric_idx]
                else:
                    custom_value = 0.0

                episode_custom_metrics.append(float(custom_value))

            except (IndexError, KeyError, TypeError) as e:
                print(f"Warning: Could not extract custom metric: {e}")
                episode_custom_metrics.append(0.0)

            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_custom_metric = np.mean(episode_custom_metrics)

        # Store for later access
        self.custom_metrics.append(mean_custom_metric)

        return mean_reward, std_reward, mean_custom_metric


def optimize_hyperparameters(n_trials=100,timeout=None, n_jobs=1, study_name="ppo_pacman",
                             wandb_project="ppo-pacman-optuna", use_wandb=True):
    """Run hyperparameter optimization."""

    if use_wandb:
        wandb.login()

    # Create study
    sampler = TPESampler(n_startup_trials=10, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    pruner_config = PatientPruner(
        PercentilePruner(percentile=50.0, n_warmup_steps=3), # i report eval id not steps so 3 * 50 000 = 150000
        patience=1
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
            lambda trial: objective(trial, wandb_project=wandb_project, use_wandb=use_wandb),
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
    with open(f"optuna_search/{study_name}_best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    print(f"\n{'=' * 70}")
    print(f"Best parameters saved to: {study_name}_best_params.json")

    # Create visualizations
    try:
        import optuna.visualization as vis

        fig = vis.plot_optimization_history(study)
        fig.write_html(f"optuna_search/{study_name}_optimization_history.html")

        fig = vis.plot_param_importances(study)
        fig.write_html(f"optuna_search/{study_name}_param_importances.html")

        fig = vis.plot_slice(study)
        fig.write_html(f"optuna_search/{study_name}_slice_plot.html")

        print(f"Visualizations saved as HTML files")

    except ImportError:
        print("\nNote: Install plotly for visualizations: pip install plotly")


    return study


if __name__ == "__main__":
    # Configuration
    USE_WANDB = True  # Set to False to disable W&B logging
    WANDB_PROJECT = "ppo-pacman-optuna-search-dump"
    N_TRIALS = None  # Number of hyperparameter combinations to try. None allows to use clock better for working on cluster

    # Run optimization
    study = optimize_hyperparameters(
        n_trials=N_TRIALS,
        timeout = 257400,
        n_jobs=1,  # Set to -1 to use all CPU cores (requires parallel environments)
        study_name="ppo_pacman_optuna_optimization_1",
        wandb_project=WANDB_PROJECT,
        use_wandb=USE_WANDB
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review the best parameters in 'ppo_pacman_optimization_best_params.json'")
    print(f"2. Check W&B dashboard for detailed analysis: wandb.ai")
    print(f"3. Use the optimal hyperparameters to train your final model")
    print("=" * 70 + "\n")