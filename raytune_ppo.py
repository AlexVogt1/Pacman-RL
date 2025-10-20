import os
import gym
import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.wandb import WandbLoggerCallback  # optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

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

# -------------------------------
# Define your training function
# -------------------------------
def train_ppo(config, checkpoint_dir=None):
    # Create the environment
    pacman_exe = "./pacman_builds/data_obs_headless_bigger_reward/AiPerPacman.exe"
    unity_env = UnityEnvironment(file_name=pacman_exe, no_graphics=True)
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False, flatten_branched=True)
    env = DummyVecEnv([lambda: env])

    # Define model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        n_epochs=config["n_epochs"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=0,
    )

    # Optionally load from checkpoint
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.zip")
        if os.path.exists(checkpoint_path):
            model.set_parameters(checkpoint_path)

    # Evaluation callback (every eval_freq steps)
    eval_env = DummyVecEnv([lambda: UnityToGymWrapper(unity_env, allow_multiple_obs=False, flatten_branched=True)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=config["eval_freq"],
        deterministic=True,
        render=False,
    )

    # Train
    model.learn(total_timesteps=config["total_timesteps"], callback=eval_callback)

    # Evaluate the model
    mean_reward, _ = evaluate_model(model, eval_env)
    tune.report(mean_reward=mean_reward)  # Report metric to Ray Tune

    # Save final checkpoint
    with tune.checkpoint_dir(step=config["total_timesteps"]) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint.zip")
        model.save(path)

    env.close()
    eval_env.close()


# -------------------------------
# Helper function for evaluation
# -------------------------------
def evaluate_model(model, env, n_eval_episodes=5):
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        total_reward = 0.0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
    return float(sum(episode_rewards) / len(episode_rewards)), episode_rewards

def train_ppo_without_eval(config, checkpoint_dir=None):
    # Use your own custom environment here
    def make_env():
        # Replace with your own class or gym.make call
        pacman_exe = "./pacman_builds/data_obs_headless_bigger_reward/AiPerPacman.exe"
        unity_env = UnityEnvironment(file_name=pacman_exe, no_graphics=True)
        env = UnityToGymWrapper(unity_env, allow_multiple_obs=False, flatten_branched=True)
        return env

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        n_epochs=config["n_epochs"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=0,
    )

    # Load checkpoint if resuming
    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.zip")
        if os.path.exists(ckpt_path):
            model.set_parameters(ckpt_path)

    # Training loop with manual evaluation
    total_timesteps = config["total_timesteps"]
    eval_interval = config["eval_freq"]

    for step in range(0, total_timesteps, eval_interval):
        # Train incrementally
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)

        # Evaluate policy
        mean_reward, _ = evaluate_model(model, eval_env)
        tune.report(mean_reward=mean_reward)

        # Save checkpoint for this interval
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            model.save(os.path.join(checkpoint_dir, "checkpoint.zip"))

    env.close()
    eval_env.close()
# -------------------------------
# Main Ray Tune setup
# -------------------------------
if __name__ == "__main__":
    check_avaliable_devices()

    ray.init(ignore_reinit_error=True)

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 3e-3),
        "n_steps": tune.choice([128, 256, 512, 1024, 2048]),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "gamma": tune.uniform(0.9, 0.999),
        "gae_lambda": tune.uniform(0.8, 1.0),
        "clip_range": tune.uniform(0.1, 0.3),
        "ent_coef": tune.loguniform(1e-4, 0.02),
        "vf_coef": tune.uniform(0.3, 1.0),
        "n_epochs": tune.choice([2, 5, 10, 20]),
        "total_timesteps": 100_000,  # per trial
        "eval_freq": 10_000,
    }

    scheduler = ASHAScheduler(
        metric="mean_reward",
        mode="max",
        max_t=200_000,
        grace_period=50_000,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(train_ppo, {"cpu": 2, "gpu": 0.25}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=20,  # number of trials
        ),
        run_config=tune.RunConfig(
            name="ppo_raytune_stablebaselines",
            stop={"training_iteration": 1},
            local_dir="./ray_results",
            verbose=1,
            callbacks=[
                # Uncomment if you use Weights & Biases
                # WandbLoggerCallback(project="ppo_tuning", log_config=True)
            ],
        ),
    )

    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result(metric="mean_reward", mode="max")
    print("Best hyperparameters:", best_result.config)
