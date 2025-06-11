import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback

# Import your custom environment and the necessary wrappers
from gymnasium_env.SBROEnvironment import SBROEnv
from gymnasium.wrappers import NormalizeObservation

import datetime as dt

import os

import polars as pl


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
    return NormalizeObservation(env)


ENV_CONFIG = {
    "base_url": "http://localhost:8100",
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
        "penalty_τ": 0.01,  # (s)⁻¹
        "penalty_SEC": (0.01) / 3600.0 / 1000.0,  # (kWh/m³)⁻¹ → (Ws/m³)⁻¹
        "penalty_conc": 5.0,  # (kg/m³)⁻¹
        "incentive_V_perm": 0.1,  # (m³)⁻¹
    },
}


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        # Keep some global state in between individual callback events.
        self.episode_num = 0
        self.save_dir = os.path.join("./result", str(dt.datetime.now()))
        os.makedirs(self.save_dir, exist_ok=True)

    def on_episode_end(self, *, episode, **kwargs):
        self.episode_num += 1

        numpy_episode = episode.to_numpy()
        obs = numpy_episode.get_observations()
        acts = numpy_episode.get_actions()
        rewards = numpy_episode.get_rewards()

        # print(obs.shape)
        # print(acts[0].shape)
        # print(acts[1].shape)
        # print(rewards.shape)

        obs = numpy_episode.get_observations()[:-2, :]
        acts = numpy_episode.get_actions()
        cont_acts = acts[0][1:, :]
        disc_acts = acts[1][1:]
        rewards = numpy_episode.get_rewards()[1:]

        episode_dict = {}

        for obs_dim in range(obs.shape[1]):
            episode_dict[f"Obs_{obs_dim}"] = obs[:, obs_dim]

        for act_dim in range(cont_acts.shape[1]):
            episode_dict[f"Act_{act_dim}"] = cont_acts[:, act_dim]

        episode_dict["Act_2"] = disc_acts

        episode_dict["Reward"] = rewards

        save_path = os.path.join(self.save_dir, f"{dt.datetime.now()}.parquet")

        episode_df = pl.DataFrame(episode_dict)

        episode_df.write_parquet(save_path)


def main():
    # --- START: MODIFIED SECTION 1 ---
    # Step 1: Explicitly tell Ray that 1 GPU is available for the whole cluster.
    # This makes the resource visible to all components.
    ray.init(num_gpus=1)

    # You can now programmatically verify that Ray has detected the GPU.
    print("Ray Cluster Resources:", ray.cluster_resources())
    # --- END: MODIFIED SECTION 1 ---

    tune.register_env("sbro_env_v1", sbro_env_creator)

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
