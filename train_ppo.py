import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback

# Import your custom environment and the necessary wrappers
import gymnasium as gym
import gymnasium.spaces as spaces

from gymnasium_env.SBROEnvironment import SBROEnv

import numpy as np

import datetime as dt

import os

import polars as pl


class MinMaxNormalizeObservation(gym.ObservationWrapper):
    """
    Applies min-max scaling to the observations of an environment.

    This scales the observations to a fixed range of [0, 1] based on the
    defined `low` and `high` bounds of the environment's observation space.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # The observation space must be a Box for this wrapper to work
        if not isinstance(env.observation_space, spaces.Box):
            raise ValueError(
                "MinMaxNormalizeObservation wrapper only works with Box observation spaces."
            )

        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high
        self.obs_range = self.obs_high - self.obs_low

        # Avoid division by zero if the range is zero for some dimension
        self.obs_range[self.obs_range == 0] = 1e-6

        # Update the environment's observation space to reflect the new [0, 1] range
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Applies the min-max scaling to the observation."""
        # First, clip the observation to be within the defined bounds
        clipped_obs = np.clip(obs, self.obs_low, self.obs_high)

        # Apply the scaling formula: (obs - low) / (high - low)
        scaled_obs = (clipped_obs - self.obs_low) / self.obs_range

        return scaled_obs.astype(np.float32)


def sbro_env_creator(env_config):
    """
    Creates, configures, and wraps the SBROEnv.
    """
    env = SBROEnv(
        base_url=env_config["base_url"],
        scenario_condition=env_config["scenario_condition"],
        objective_condition=env_config["objective_condition"],
        reward_conf=env_config["reward_conf"],
        dt=env_config.get("dt", 30.0),
    )
    return MinMaxNormalizeObservation(env)


ENV_CONFIG = {
    "base_url": "http://localhost:8081",
    "scenario_condition": {
        "T_feed_mean": 15.0,
        "T_feed_std": 0.5,
        "C_feed_mean": 0.05,
        "C_feed_std": 0.005,
    },
    "objective_condition": {
        "time_objective_low": 28800.0,
        "time_objective_high": 43200.0,
        "V_perm_objective_low": 12.0,
        "V_perm_objective_high": 16.0,
    },
    "reward_conf": {
        "penalty_truncation": 1.0,
        "penalty_τ": 0.01,
        "penalty_SEC": (1.0) / 3600.0 / 1000.0,
        "incentive_V_perm": 0.01,
        "penalty_V_disp": 0.1,
    },
}


def main():
    # --- START: MODIFIED SECTION 1 ---
    # Step 1: Explicitly tell Ray that 1 GPU is available for the whole cluster.
    # This makes the resource visible to all components.
    ray.init(num_gpus=1)

    # You can now programmatically verify that Ray has detected the GPU.
    print("Ray Cluster Resources:", ray.cluster_resources())
    # --- END: MODIFIED SECTION 1 ---

    tune.register_env("sbro_env_v1", sbro_env_creator)

    SAVE_DIR = os.path.join("./result", str(dt.datetime.now()))

    os.makedirs(os.path.join(SAVE_DIR, "parquets"), exist_ok=True)

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

    config = (
        PPOConfig()
        .environment(env="sbro_env_v1", env_config=ENV_CONFIG)
        .env_runners(num_env_runners=1, rollout_fragment_length="auto")
        .framework("torch")
        # --- START: MODIFIED SECTION 2 ---
        # Step 2: Request 1 GPU for the PPO learner/trainer process.
        # This tells the algorithm to place itself on the GPU made available in ray.init().
        .resources(num_gpus=1)
        .learners(
            # Explicitly assign the GPU to the learner process
            num_learners=1,
            num_gpus_per_learner=1,
        )
        # --- END: MODIFIED SECTION 2 ---
        .training(
            gamma=0.99,
            lr=5e-5,
            lambda_=0.95,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
        )
        .callbacks(EpisodeReturn)
    )

    algo = config.build_algo()

    print("\nStarting GPU training...")
    for i in range(100):
        result = algo.train()

        print(f"Iteration: {i + 1}")

        env_runners_results = result.get("env_runners", {})
        mean_reward = env_runners_results["agent_episode_returns_mean"]["default_agent"]
        steps_trained = env_runners_results.get("num_env_steps_sampled", 0)
        # sps = result.get("sps", 0)

        print(f"  Episode reward mean: {mean_reward:.3f}")
        print(f"  Total env steps: {steps_trained}")
        # print(f"  SPS: {sps:.0f}")

    checkpoint_dir = algo.save()
    print(f"\nCheckpoint saved in directory: {checkpoint_dir}")

    algo.stop()
    ray.shutdown()
    print("Training finished and Ray shut down.")


if __name__ == "__main__":
    main()
