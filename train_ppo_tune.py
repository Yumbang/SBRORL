import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
import numpy as np
import datetime as dt
import warnings
import polars as pl

# Import your custom environment and the necessary wrappers
from gymnasium_env.SBROEnvironment import SBROEnv
from gymnasium_env.utils import sbro_env_creator, generate_entropy_schedule


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init(num_gpus=1)
    print("Ray Cluster Resources:", ray.cluster_resources())

    num_environments = 24
    # num_environments = 4
    ports = range(8100, 8100 + num_environments)

    Q_obj = (
        1.69 / 2
    )  # Based on FT106 (24-hr moving average: mean=1.69, std=1.52. FT106 contains both aeration product + RO product, thus divide by 2)

    n_time_scenarios = 8
    n_temp_scenarios = int(np.floor(num_environments/n_time_scenarios))

    if num_environments % n_time_scenarios != 0:
        print("Number of scenarios does not match with the number of available environments."
              "Some of the environments are not activated.")

    temp_means = np.linspace(15.0, 18.0, num=n_temp_scenarios)
    time_means = np.linspace(3.0, 12.0, num=n_time_scenarios)
    time_highs = time_means * 1.05
    time_lows = time_means * 0.95
    V_highs = time_highs * Q_obj
    V_lows = time_lows * Q_obj

    algorithm = "PPO"

    SAVE_DIR = os.path.join("./result/", algorithm, str(dt.datetime.now()))
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "csvs"))

    class EpisodeReturn(RLlibCallback):
        def __init__(self):
            super().__init__()

            # Keep some global state in between individual callback events.

            self.episode_num = 0

        def on_episode_end(self, *, episode, **kwargs):
            self.episode_num += 1
            numpy_episode = episode.to_numpy()
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

            save_path = os.path.join(SAVE_DIR, "csvs", f"{dt.datetime.now()}.csv")
            episode_df = pl.DataFrame(episode_dict)
            episode_df.write_csv(save_path)
            print(f"Reward: {episode_df['Reward'].sum():.2f}")

    worker_configs = []

    REWARD_CONFIG = {
        "penalty_truncation": 5.0,
        "incentive_termination": 5.0,
        "penalty_τ": 5.0e-3,
        "penalty_SEC": (0.25) / 3600.0 / 1000.0,
        "penalty_conc": 1000.0 / 3600.0,
        "incentive_V_perm": 0.0,
        "penalty_V_disp": 0.0,
        "penalty_V_feed": 1.0,
        "penalty_V_perm": 0.01,
        # "penalty_rapid_change": 0.0,
        "penalty_rapid_change": 0.05 / 60.0,
    }

    for i in range(len(temp_means)):
        for j in range(len(time_means)):
            config = {
                "base_url": f"http://localhost:{ports[i * len(time_means) + j]}",
                "scenario_condition": {
                    "T_feed_mean": temp_means[i],
                    "T_feed_std": 0.25,
                    "C_feed_mean": 0.05,
                    "C_feed_std": 0.001,
                },
                "objective_condition": {
                    "time_objective_low": time_lows[j] * 3600.0,
                    "time_objective_high": time_highs[j] * 3600.0,
                    "V_perm_objective_low": V_lows[j],
                    "V_perm_objective_high": V_highs[j],
                },
                "reward_conf": REWARD_CONFIG,
            }
            worker_configs.append(config)

    ENV_CONFIG = {"worker_configs": worker_configs, "dt": 60.0}

    tune.register_env("sbro_env_v1", sbro_env_creator)

    max_train_step = 1000

    # entropy_coefficient_schedule = generate_entropy_schedule(max_train_step, 0.25, 0.05, 10)

    if algorithm == "PPO":
        config = (
            PPOConfig()
            .environment(env="sbro_env_v1", env_config=ENV_CONFIG)
            .env_runners(
                num_env_runners=num_environments,
                batch_mode="complete_episodes",
                gym_env_vectorize_mode="SYNC",
            )
            .framework("torch")
            .resources(
                num_gpus=0
            )
            .learners(
                # The Learner process gets the GPU.
                num_learners=1,
                num_gpus_per_learner=1,
                num_cpus_per_learner=4,
            )
            .training(
                use_critic=True,
                use_gae=True,
                gamma=0.995,
                lr=5e-5,
                lambda_=0.95,
                # Add the model dictionary here to enable the RNN
                model={
                    # Use a Long Short-Term Memory (LSTM) network.
                    # You could also use "use_gru": True
                    "use_lstm": True,
                    # The number of hidden units in the LSTM layer.
                    # This is the size of the agent's memory.
                    "lstm_cell_size": 256,
                    # The length of the sequence of observations to pass to the RNN for training.
                    # This is a critical hyperparameter to tune.
                    "max_seq_len": 256,
                    # By default, the value function will also share the LSTM encoder.
                    # Set this to False to have a separate feedforward network for the value function.
                    "vf_share_layers": True,
                    "fcnet_hiddens": [32, 64, 128, 256],
                },
                entropy_coeff=0.0,
                # train_batch_size_per_learner=1024*4
            )
            .callbacks(EpisodeReturn)
        )
    elif algorithm == "APPO":
        config = (
            APPOConfig()
            .environment(env="sbro_env_v1", env_config=ENV_CONFIG)
            .env_runners(
                num_env_runners=num_environments,
                batch_mode="complete_episodes",
                gym_env_vectorize_mode="ASYNC",
            )
            .framework("torch")
            .resources(
                # The main Algorithm driver runs on the CPU, so it requires 0 GPUs.
                num_gpus=0
            )
            .learners(
                # The Learner process gets the GPU.
                num_learners=1,
                num_gpus_per_learner=1,
                num_cpus_per_learner=4,
            )
            .training(
                # use_critic=True,
                # use_gae=True,
                gamma=0.995,
                lr=5e-5,
                lambda_=0.95,
                # Add the model dictionary here to enable the RNN
                model={
                    # Use a Long Short-Term Memory (LSTM) network.
                    # You could also use "use_gru": True
                    "use_lstm": True,
                    # The number of hidden units in the LSTM layer.
                    # This is the size of the agent's memory.
                    "lstm_cell_size": 256,
                    # The length of the sequence of observations to pass to the RNN for training.
                    # This is a critical hyperparameter to tune.
                    "max_seq_len": 256,
                    # By default, the value function will also share the LSTM encoder.
                    # Set this to False to have a separate feedforward network for the value function.
                    "vf_share_layers": True,
                    "fcnet_hiddens": [32, 64, 128, 256],
                },
                # entropy_coeff=entropy_coefficient_schedule,
                # train_batch_size_per_learner=1024*4
            )
            .callbacks(EpisodeReturn)
        )

    stop_criteria = {"training_iteration": max_train_step}
    storage_path = os.path.abspath(SAVE_DIR)

    run_config = tune.RunConfig(
        stop=stop_criteria,
        storage_path=storage_path,
        name=f"{algorithm}_SBRO",
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_score_attribute="env_runners/episode_return_mean",
            checkpoint_score_order="max",
            # num_to_keep=3,
            checkpoint_frequency=50,
        ),
    )

    tuner = tune.Tuner(
        algorithm,
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
