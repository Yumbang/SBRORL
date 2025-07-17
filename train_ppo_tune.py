import datetime as dt
import os
import warnings
import argparse
from functools import partial

import numpy as np
import polars as pl
import ray
from ray import tune
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)
from scipy import stats

from gymnasium_env.SBROEnvironment import SBROEnv
from gymnasium_env.utils import (
    discrete_sbro_env_creator,
    sbro_env_creator,
)


def generate_env_settings(
    reward_config,
    num_environments=24,
    starting_port=8100,
    temp_range=[15.0, 18.0],
    time_range=[3.0, 12.0],
    Q_obj=0.85,
):
    ports = range(starting_port, starting_port + num_environments)

    Q_obj = Q_obj  # Based on FT106 (24-hr moving average: mean=1.69, std=1.52. FT106 contains both aeration product + RO product, thus divide by 2)

    n_time_scenarios = 8
    n_temp_scenarios = int(np.floor(num_environments / n_time_scenarios))

    if num_environments % n_time_scenarios != 0:
        print(
            "Number of scenarios does not match with the number of available environments."
            "Some of the environments are not activated."
        )

    temp_means = np.linspace(temp_range[0], temp_range[1], num=n_temp_scenarios)
    time_means = np.linspace(time_range[0], time_range[1], num=n_time_scenarios)
    time_highs = time_means * 1.05
    time_lows = time_means * 0.95
    V_highs = time_highs * Q_obj
    V_lows = time_lows * Q_obj

    worker_configs = []

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
                "reward_conf": reward_config,
            }
            worker_configs.append(config)

    return worker_configs


def generate_env_settings_v2(
    reward_config,
    num_environments=24,
    starting_port=8100,
    temp_range=[15.0, 18.0],
    time_range=[3.0, 12.0],
    Q_obj_range=[0.85, 1.70],
):
    ports = range(starting_port, starting_port + num_environments)

    Q_obj_low, Q_obj_high = Q_obj_range

    n_time_scenarios = 8
    n_temp_scenarios = int(np.floor(num_environments / n_time_scenarios))

    if num_environments % n_time_scenarios != 0:
        print(
            "Number of scenarios does not match with the number of available environments."
            "Some of the environments are not activated."
        )

    time_grid = np.linspace(*time_range, n_time_scenarios + 1)
    time_highs = time_grid[:-1]
    time_lows = time_grid[1:]

    temp_means = np.linspace(temp_range[0], temp_range[1], num=n_temp_scenarios)

    worker_configs = []

    for i in range(n_temp_scenarios):
        for j in range(n_time_scenarios):
            config = {
                "base_url": f"http://localhost:{ports[i * n_time_scenarios + j]}",
                "scenario_condition": {
                    "T_feed_mean": temp_means[i],
                    "T_feed_std": 0.25,
                    # "C_feed_mean": 0.05,
                    "C_feed_mean_low": 0.05,
                    "C_feed_mean_high": 0.1,
                    "C_feed_std": 0.001,
                },
                "objective_condition": {
                    "time_objective_low": time_lows[j] * 3600.0,
                    "time_objective_high": time_highs[j] * 3600.0,
                    "Q_perm_objective_low": Q_obj_low,
                    "Q_perm_objective_high": Q_obj_high,
                },
                "reward_conf": reward_config,
            }
            worker_configs.append(config)

    return worker_configs


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO model with optional curriculum learning."
    )
    parser.add_argument(
        "--use-curriculum",
        type=str,
        choices=["y", "n"],
        default="y",
        help="Enable or disable curriculum learning ('y' or 'n').",
    )
    args = parser.parse_args()
    USE_CURRICULUM = args.use_curriculum == "y"

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init(num_gpus=1)
    print("Ray Cluster Resources:", ray.cluster_resources())

    # Experiment informations
    algorithm = "PPO"
    num_environments = 24
    starting_port = 8100

    # Curriculum informations

    N_CURRICULUM = 4
    # Q_OBJ_CURRICULUM = np.linspace(
    #     start=1.69 / 2, stop=1.69 * 1.5 / 2, num=N_CURRICULUM
    # )
    Q_obj_range = np.linspace(0.8, 1.6, 2 + 2 * N_CURRICULUM)
    Q_obj_curr = [
        np.array([Q_obj_range[curr], Q_obj_range[-(curr + 1)]])
        for curr in range(N_CURRICULUM)
    ]
    Q_obj_curr.reverse()
    Q_OBJ_CURRICULUM = Q_obj_curr
    time_low_curriculum = np.linspace(start=3.0, stop=3.0, num=N_CURRICULUM)
    time_high_curriculum = np.linspace(start=18.0, stop=18.0, num=N_CURRICULUM)
    TIME_CURRICULUM = [
        [time_low_curriculum[level], time_high_curriculum[level]]
        for level in range(N_CURRICULUM)
    ]

    REWARD_CONFIG = {
        "penalty_truncation": 5.0,
        # "incentive_termination": 5.0,
        "incentive_termination": 0.0,
        "penalty_τ": 5.0e-3,
        # Virtually it is penalty about energy, not SEC.
        "penalty_SEC": (1.0) / 3600.0 / 1000.0,
        "penalty_conc": 1000.0 / 3600.0,
        "incentive_V_perm": 0.0,
        "penalty_V_disp": 0.0,
        "penalty_V_feed": 1.0,
        "penalty_V_perm": 0.01,
        "penalty_rapid_change": 0.05 / 60.0,
    }

    SAVE_DIR = os.path.join("./result/", algorithm, str(dt.datetime.now()))
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "csvs"))

    # Necessary Callbacks
    class EpisodeReturn(RLlibCallback):
        def __init__(self):
            super().__init__()

            # Keep some global state in between individual callback events.

            self.episode_num = 0

        def on_episode_end(self, *, episode, **kwargs):
            self.episode_num += 1
            numpy_episode = episode.to_numpy()
            numpy_observations = numpy_episode.get_observations()
            prev_obs = numpy_observations[:-1, :]
            curr_obs = numpy_observations[1:, :]

            acts = numpy_episode.get_actions()
            cont_acts = acts[0]
            disc_acts = acts[1]

            rewards = numpy_episode.get_rewards()

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
            # print(f"Reward: {episode_df['Reward'].sum():.2f}")

    class DiscreteEpisodeReturn(RLlibCallback):
        def __init__(self):
            super().__init__()

            # Keep some global state in between individual callback events.

            self.episode_num = 0

        def on_episode_end(self, *, episode, **kwargs):
            self.episode_num += 1
            # numpy_episode = episode.to_numpy()
            numpy_episode = episode
            numpy_observations = numpy_episode.get_observations()
            prev_obs = np.array(
                [
                    numpy_observation["observations"]
                    for numpy_observation in numpy_observations[:-1]
                ]
            )
            curr_obs = np.array(
                [
                    numpy_observation["observations"]
                    for numpy_observation in numpy_observations[1:]
                ]
            )

            acts = numpy_episode.get_actions()
            # print(f"{type(acts) = }")
            # print(f"{acts = }")
            # cont_acts = acts[0][1:, :]
            # disc_acts = acts[1][1:]

            rewards = numpy_episode.get_rewards()

            episode_dict = {}

            for obs_dim in range(prev_obs.shape[1]):
                episode_dict[f"Previous_obs_{obs_dim}"] = prev_obs[:, obs_dim]

            for obs_dim in range(curr_obs.shape[1]):
                episode_dict[f"Current_obs_{obs_dim}"] = curr_obs[:, obs_dim]

            # for act_dim in range(cont_acts.shape[1]):
            #     episode_dict[f"Act_{act_dim}"] = cont_acts[:, act_dim]

            # episode_dict["Act_2"] = disc_acts
            episode_dict["Act"] = acts

            episode_dict["Reward"] = rewards

            save_path = os.path.join(SAVE_DIR, "csvs", f"{dt.datetime.now()}.csv")
            episode_df = pl.DataFrame(episode_dict)
            episode_df.write_csv(save_path)
            print(f"Reward: {episode_df['Reward'].sum():.2f}")

    class CurriculumHandler(RLlibCallback):
        def __init__(self):
            super().__init__()
            self.train_iter_count = 0
            self.level = 0
            self.max_level = N_CURRICULUM - 1
            self.return_mean_log = [[]]
            self.window = 30
            self.div_thresh = 0.01
            self.alpha = 0.05
            self.improvement_tol = 3
            self.improvement_tol_count = 0

        def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
            # print(algorithm.env_runner_group.healthy_env_runner_ids())
            # Logging mean (or min 😒) return.
            # For diversive curriculum (Plan_B.jl), min return-based handling is more reasonable.
            self.train_iter_count += 1
            self.return_mean_log[-1].append(
                result["env_runners"]["episode_return_mean"]
                # result["env_runners"]["episode_return_min"]
            )

            # Perform KL divergence test to determine convergence
            if self.train_iter_count > 3 * self.window + 50:
                recent_mean_returns = np.array(self.return_mean_log[-1][-self.window :])
                past_mean_returns = np.array(
                    self.return_mean_log[-1][-2 * self.window : -self.window]
                )

                KL_div = stats.entropy(
                    pk=recent_mean_returns,
                    qk=past_mean_returns,
                )
                is_evolving = KL_div > self.div_thresh

                if is_evolving:
                    self.improvement_tol_count = 0
                    print(
                        f"Iter #{self.train_iter_count} showed difference! (KL_div = {KL_div:.5f})"
                    )
                else:
                    converged_mean_returns = np.array(
                        self.return_mean_log[-1][-2 * self.window :]
                    )
                    evolving_mean_returns = np.array(
                        self.return_mean_log[-1][: -2 * self.window]
                    )
                    # Perform T-test to determine improvement
                    _, p_val = stats.ttest_ind(
                        converged_mean_returns,
                        evolving_mean_returns,
                        alternative="greater",
                    )
                    if p_val < self.alpha:
                        self.improvement_tol_count += 1
                        print(
                            f"Iter #{self.train_iter_count} converged and improved! (KL_div = {KL_div:.5f} | p_val = {p_val:.4f})"
                        )
                    else:
                        print(
                            f"Iter #{self.train_iter_count} converged but did not improve! (KL_div = {KL_div:.5f} | p_val = {p_val:.4f})"
                        )

                if self.improvement_tol_count == self.improvement_tol:
                    print(
                        f"Recent {self.improvement_tol + 2 * self.window} iterations converged and improved! (KL_div = {KL_div:.5f} | p_val = {p_val:.4f})"
                    )
                    if self.level < self.max_level:
                        print(
                            f"📚 Moving from curriculum [#{self.level} / {self.max_level}] to [#{self.level + 1} / {self.max_level}]"
                        )
                        # Prepare the next curriculum
                        self.return_mean_log.append([])
                        self.train_iter_count = 0
                        self.improvement_tol_count = 0
                        self.level += 1

                        # Generate next curriculum's settings
                        new_conditions = generate_env_settings_v2(
                            reward_config=REWARD_CONFIG,
                            num_environments=num_environments,
                            starting_port=starting_port,
                            temp_range=[15.0, 18.0],
                            time_range=TIME_CURRICULUM[self.level],
                            Q_obj_range=Q_OBJ_CURRICULUM[self.level],
                        )

                        def _update_conditions(env_runner, conditions):
                            env_runner.config.environment(
                                env_config={
                                    "worker_configs": conditions
                                }  # Runner ID ranges from 1 to num_environments
                            )
                            env_runner.make_env()
                            return None

                        # Apply the generated conditions to the environments
                        algorithm.env_runner_group.foreach_env_runner(
                            partial(_update_conditions, conditions=new_conditions)
                        )

                    else:
                        print(
                            f"🧑‍🎓 Completed all of the curriculums! [#{self.level} / {self.max_level}]"
                        )
                        # If the end of the curriculum was reached, (gently) terminate the training
                        result["done"] = True

            result["level"] = self.level

            return None

    class EarlyStopHandler(RLlibCallback):
        def __init__(self):
            super().__init__()
            self.train_iter_count = 0
            self.return_mean_log = [[]]
            self.window = 30
            self.div_thresh = 0.01
            self.alpha = 0.01
            self.improvement_tol = 3
            self.improvement_tol_count = 0

        def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
            # print(algorithm.env_runner_group.healthy_env_runner_ids())
            # Logging mean (or min 😒) return.
            # For diversive curriculum (Plan_B.jl), min return-based handling is more reasonable.
            self.train_iter_count += 1
            self.return_mean_log[-1].append(
                result["env_runners"]["episode_return_mean"]
                # result["env_runners"]["episode_return_min"]
            )

            # Perform KL divergence test to determine convergence
            if self.train_iter_count > 3 * self.window + 50:
                recent_mean_returns = np.array(self.return_mean_log[-1][-self.window :])
                past_mean_returns = np.array(
                    self.return_mean_log[-1][-2 * self.window : -self.window]
                )

                KL_div = stats.entropy(
                    pk=recent_mean_returns,
                    qk=past_mean_returns,
                )
                is_evolving = KL_div > self.div_thresh

                if is_evolving:
                    self.improvement_tol_count = 0
                    print(
                        f"Iter #{self.train_iter_count} showed difference! (KL_div = {KL_div:.5f})"
                    )
                else:
                    converged_mean_returns = np.array(
                        self.return_mean_log[-1][-2 * self.window :]
                    )
                    evolving_mean_returns = np.array(
                        self.return_mean_log[-1][: -2 * self.window]
                    )
                    # Perform T-test to determine improvement
                    _, p_val = stats.ttest_ind(
                        converged_mean_returns,
                        evolving_mean_returns,
                        alternative="greater",
                    )
                    if p_val < self.alpha:
                        self.improvement_tol_count += 1
                        print(
                            f"Iter #{self.train_iter_count} converged and improved! (KL_div = {KL_div:.5f} | p_val = {p_val:.4f})"
                        )
                    else:
                        print(
                            f"Iter #{self.train_iter_count} converged but did not improve! (KL_div = {KL_div:.5f} | p_val = {p_val:.4f})"
                        )

                if self.improvement_tol_count == self.improvement_tol:
                    print(
                        f"Recent {self.improvement_tol + 2 * self.window} iterations converged and improved! (KL_div = {KL_div:.5f} | p_val = {p_val:.4f})"
                    )

                    print("🧑‍🎓 Completed all of the curriculums!")
                    # If the end of the curriculum was reached, (gently) terminate the training
                    result["done"] = True

            return None

    if USE_CURRICULUM:
        # Start with the first curriculum
        worker_configs = generate_env_settings_v2(
            reward_config=REWARD_CONFIG,
            num_environments=num_environments,
            starting_port=starting_port,
            temp_range=[15.0, 18.0],
            time_range=TIME_CURRICULUM[0],
            Q_obj_range=Q_OBJ_CURRICULUM[0],
        )
        callbacks = [EpisodeReturn, CurriculumHandler]
    else:
        # Start with the last curriculum
        worker_configs = generate_env_settings_v2(
            reward_config=REWARD_CONFIG,
            num_environments=num_environments,
            starting_port=starting_port,
            temp_range=[15.0, 18.0],
            time_range=TIME_CURRICULUM[-1],
            Q_obj_range=Q_OBJ_CURRICULUM[-1],
        )
        # callbacks = [EpisodeReturn, EarlyStopHandler]
        callbacks = [EpisodeReturn]

    env_config = {"worker_configs": worker_configs, "dt": 60.0}

    # Adjust maximum training step!
    max_train_step = 1000

    # entropy_coefficient_schedule = generate_entropy_schedule(max_train_step, 0.25, 0.05, 10)

    if algorithm == "PPO":
        # Register the environment to use
        tune.register_env("sbro_env_v1", sbro_env_creator)

        config = (
            PPOConfig()
            .environment(env="sbro_env_v1", env_config=env_config)
            .env_runners(
                num_env_runners=num_environments,
                batch_mode="complete_episodes",
                gym_env_vectorize_mode="ASYNC",
            )
            .framework("torch")
            .resources(num_gpus=0)
            .learners(
                # The Learner process gets the GPU.
                num_learners=1,
                num_gpus_per_learner=1,
                num_cpus_per_learner=4,
            )
            .training(
                use_critic=True,
                use_gae=True,
                gamma=0.999,
                lr=5e-5,
                lambda_=0.95,
                # Add the model dictionary here to enable the RNN
                # Configurations about model construction did not work (25.07.09)
                model={
                    # Use a Long Short-Term Memory (LSTM) network.
                    "use_lstm": True,
                    # The number of hidden units in the LSTM layer.
                    "lstm_cell_size": 256,
                    # The length of the sequence of observations to pass to the RNN for training.
                    "max_seq_len": 2048,
                    # Set this to False to have a separate feedforward network for the value function.
                    "vf_share_layers": True,
                    # This do not work. RLLib just construct encoder consists of 3 layers with 256 nodes
                    "fcnet_hiddens": [
                        32,
                        64,
                        128,
                        256,
                    ],
                },
                entropy_coeff=0.0,
                # train_batch_size_per_learner=1024*4
            )
            .callbacks(callbacks)
        )
    elif algorithm == "APPO":
        config = (
            APPOConfig()
            .environment(env="sbro_env_v1", env_config=env_config)
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
                gamma=0.999,
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
    elif algorithm == "DiscretePPO":
        tune.register_env("sbro_env_v1", discrete_sbro_env_creator)
        config = (
            PPOConfig()
            .environment(env="sbro_env_v1", env_config=env_config)
            .env_runners(
                num_env_runners=num_environments,
                batch_mode="complete_episodes",
                gym_env_vectorize_mode="SYNC",
            )
            .framework("torch")
            .resources(num_gpus=0)
            .learners(
                # The Learner process gets the GPU.
                num_learners=1,
                num_gpus_per_learner=1,
                num_cpus_per_learner=4,
            )
            .rl_module(
                # We need to explicitly specify here RLModule to use and
                # the catalog needed to build it.
                rl_module_spec=RLModuleSpec(
                    module_class=ActionMaskingTorchRLModule,
                    model_config={
                        "head_fcnet_hiddens": [64, 128, 256],
                        "head_fcnet_activation": "relu",
                    },
                ),
            )
            .training(
                use_critic=True,
                use_gae=True,
                gamma=0.999,
                lr=5e-5,
                lambda_=0.95,
                entropy_coeff=0.0,
                # train_batch_size_per_learner=1024*4
            )
            .callbacks(DiscreteEpisodeReturn)
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
            checkpoint_frequency=10,
        ),
    )

    tuner = tune.Tuner(
        algorithm,
        param_space=config.to_dict(),
        run_config=run_config,
    )

    results = tuner.fit()

    print("Training finished.")

    ray.shutdown()
    print("Ray shut down.")


if __name__ == "__main__":
    main()
