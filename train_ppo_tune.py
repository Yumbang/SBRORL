import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
import numpy as np
import datetime as dt
import warnings
import polars as pl

# Import your custom environment and the necessary wrappers
from gymnasium_env.SBROEnvironment import SBROEnv
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import NormalizeObservation, TransformObservation


# class MinMaxNormalizeObservation(gym.ObservationWrapper):
#     """
#     Applies min-max scaling to the observations of an environment to a [0, 1] range.
#     This version correctly handles both `step` and `reset` methods.
#     """
#     def __init__(self, env: gym.Env):
#         super().__init__(env)
#         if not isinstance(env.observation_space, spaces.Box):
#             raise ValueError("This wrapper only works with Box observation spaces.")
        
#         self.obs_low = self.observation_space.low
#         self.obs_high = self.observation_space.high
#         self.obs_range = self.obs_high - self.obs_low
#         # Avoid division by zero
#         self.obs_range[self.obs_range == 0] = 1e-6
        
#         # Update the environment's observation space to reflect the new [0, 1] range
#         self.observation_space = spaces.Box(
#             low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32
#         )

#     def observation(self, obs: np.ndarray) -> np.ndarray:
#         """Applies the min-max scaling to the observation from a step."""
#         clipped_obs = np.clip(obs, self.obs_low, self.obs_high)
#         scaled_obs = (clipped_obs - self.obs_low) / self.obs_range
#         return scaled_obs.astype(np.float32)


def sbro_env_creator(env_config):
    """
    Creates an SBROEnv instance for a specific worker using a unique configuration.
    """
    worker_index = env_config.worker_index
    worker_specific_config = env_config["worker_configs"][
        worker_index % len(env_config["worker_configs"])
    ]

    print(
        f"Worker {worker_index} creating environment with config: "
        f"URL='{worker_specific_config['base_url']}', "
        f"T_feed_mean={worker_specific_config['scenario_condition']['T_feed_mean']}"
    )

    # --- START: CORRECTED SECTION ---
    # Call the SBROEnv constructor using the `worker_specific_config` dictionary
    # that was selected above. This ensures each worker gets its unique settings.
    env = SBROEnv(
        base_url=worker_specific_config["base_url"],
        scenario_condition=worker_specific_config["scenario_condition"],
        objective_condition=worker_specific_config["objective_condition"],
        reward_conf=worker_specific_config["reward_conf"],
        dt=env_config.get("dt", 30.0),  # Global settings can still be used
    )
    # --- END: CORRECTED SECTION ---
    # return MinMaxNormalizeObservation(env)
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high

    new_observation_space = spaces.Box(
        low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
    )   

    # return TransformObservation(env, lambda obs: )
    return NormalizeObservation(env)
    # return env


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init(num_gpus=1)
    print("Ray Cluster Resources:", ray.cluster_resources())

    num_environments = 24
    # num_environments = 4
    ports = range(8100, 8100 + num_environments)
    temp_means = np.linspace(15.0, 25.0, num=num_environments)

    SAVE_DIR = os.path.join("./result", str(dt.datetime.now()))
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "parquets"))

    class EpisodeReturn(RLlibCallback):
        def __init__(self):
            super().__init__()

            # Keep some global state in between individual callback events.

            self.episode_num = 0

        def on_episode_end(self, *, episode, **kwargs):
            self.episode_num += 1

            numpy_episode = episode.to_numpy()

            # obs = numpy_episode.get_observations()

            # acts = numpy_episode.get_actions()

            # rewards = numpy_episode.get_rewards()

            # print(obs.shape)

            # print(acts[0].shape)

            # print(acts[1].shape)

            # print(rewards.shape)

            prev_obs = numpy_episode.get_observations()[:-2, :]

            curr_obs = numpy_episode.get_observations()[1:-1, :]

            acts = numpy_episode.get_actions()

            cont_acts = acts[0][1:, :]

            disc_acts = acts[1][1:]

            rewards = numpy_episode.get_rewards()[1:]

            episode_dict = {}

            for obs_dim in range(prev_obs.shape[1]):
                episode_dict[f"Previous_obs_{obs_dim}"] = prev_obs[:, obs_dim]

            for obs_dim in range(curr_obs.shape[1]):
                episode_dict[f"Current_obs_{obs_dim}"] = curr_obs[:, obs_dim]

            for act_dim in range(cont_acts.shape[1]):
                episode_dict[f"Act_{act_dim}"] = cont_acts[:, act_dim]

            episode_dict["Act_2"] = disc_acts

            episode_dict["Reward"] = rewards

            save_path = os.path.join(
                SAVE_DIR, "parquets", f"{dt.datetime.now()}.parquet"
            )
            episode_df = pl.DataFrame(episode_dict)
            episode_df.write_parquet(save_path)
            print(f"Reward: {episode_df['Reward'].sum():.2f}")

    worker_configs = []
    for i in range(num_environments):
        config = {
            "base_url": f"http://localhost:{ports[i]}",
            "scenario_condition": {
                "T_feed_mean": temp_means[i],
                "T_feed_std": 0.5,
                "C_feed_mean": 0.05,
                "C_feed_std": 0.005,
            },
            "objective_condition": {
                "time_objective_low": 28800.0,
                "time_objective_high": 28800.0,
                "V_perm_objective_low": 14.0,
                "V_perm_objective_high": 16.0,
            },
            "reward_conf": {
                "penalty_truncation": 1.0,
                "penalty_τ": 1e-6,
                "penalty_SEC": (0.5) / 3600.0 / 1000.0,
                "incentive_V_perm": 0.0,
                "penalty_V_disp": 0.25,
            },
        }
        worker_configs.append(config)

    ENV_CONFIG = {"worker_configs": worker_configs, "dt": 30.0}

    tune.register_env("sbro_env_v1", sbro_env_creator)

    config = (
        PPOConfig()
        .environment(env="sbro_env_v1", env_config=ENV_CONFIG)
        .env_runners(num_env_runners=num_environments)
        .framework("torch")
        .resources(
            # The main Algorithm driver runs on the CPU, so it requires 0 GPUs.
            num_gpus=0
        )
        .learners(
            # The Learner process gets the GPU.
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .training(
            gamma=0.99,
            lr=5e-5,
            lambda_=0.95,
            # Add the model dictionary here to enable the RNN
            model={
                # Use a Long Short-Term Memory (LSTM) network.
                # You could also use "use_gru": True
                "use_gru": True,
                # The number of hidden units in the LSTM layer.
                # This is the size of the agent's memory.
                "lstm_cell_size": 256,
                # The length of the sequence of observations to pass to the RNN for training.
                # This is a critical hyperparameter to tune.
                "max_seq_len": 20,
                # By default, the value function will also share the LSTM encoder.
                # Set this to False to have a separate feedforward network for the value function.
                "vf_share_layers": True,
            },
        )
        .callbacks(EpisodeReturn)
    )

    stop_criteria = {"training_iteration": 1500}
    storage_path = os.path.abspath(SAVE_DIR)

    run_config = tune.RunConfig(
        stop=stop_criteria,
        storage_path=storage_path,
        name="PPO_SBRO_Custom_Configs_run_FIXED",
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_score_attribute="env_runners/episode_return_mean",
            checkpoint_score_order="max",
            num_to_keep=3,
            checkpoint_frequency=50,
        ),
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=run_config,
    )

    results = tuner.fit()

    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean", mode="max"
    )
    print("\nTraining finished.", best_result)

    ray.shutdown()
    print("Ray shut down.")


if __name__ == "__main__":
    main()
