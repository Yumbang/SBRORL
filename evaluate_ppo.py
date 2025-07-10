# %%
import asyncio
import datetime as dt
import os
from copy import deepcopy

import gymnasium as gym
import numpy as np
import polars as pl
import ray
import torch
from matplotlib import pyplot as plt
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule as PPOModule,
)
from tqdm import tqdm

from gymnasium_env.SBROEnvironment import SBROEnv
from gymnasium_env.utils import MinMaxNormalizeObservation, sbro_env_creator

# %%
SAVE_DIR = os.curdir


# %%
def generate_eval_env_settings_v2(
    reward_config,
    port=8100,
    temp_range=[15.0, 16.5, 18.0],
    time_range=[3.0, 18.0],
    Q_obj_range=[0.8, 1.6],
    C_feed_range=[0.05, 0.1],
    n_time_scenarios=100,
    n_Q_obj_scenarios=100,
    n_C_feed_scenarios=100,
):
    evaluation_environment_settings = []

    temp_values = temp_range
    time_values = np.linspace(*time_range, num=n_time_scenarios)
    Q_obj_values = np.linspace(*Q_obj_range, num=n_Q_obj_scenarios)
    C_feed_values = np.linspace(*C_feed_range, num=n_C_feed_scenarios)

    for time in time_values:
        for temp in temp_values:
            for Q_obj in Q_obj_values:
                for C_feed in C_feed_values:
                    config = {
                        # "base_url": f"http://localhost:{port}",
                        "scenario_condition": {
                            "T_feed_mean": temp,
                            "T_feed_std": 0.25,
                            # "C_feed_mean": 0.05,
                            "C_feed_mean_low": C_feed,
                            "C_feed_mean_high": C_feed,
                            "C_feed_std": 0.001,
                        },
                        "objective_condition": {
                            "time_objective_low": time * 3600.0,
                            "time_objective_high": time * 3600.0,
                            "Q_perm_objective_low": Q_obj,
                            "Q_perm_objective_high": Q_obj,
                        },
                        "reward_conf": reward_config,
                        "dt": 60.0,
                    }
                    evaluation_environment_settings.append(config)

    return evaluation_environment_settings


# %%
def decode_action_determ(nn_output: torch.Tensor, done_mask: list = None):
    mean1, _ = nn_output[:, 0], nn_output[:, 2]
    mean2, _ = nn_output[:, 1], nn_output[:, 3]
    logits_discrete = nn_output[:, 4:]

    # action_1 = mean1.to("cpu")
    # action_2 = mean2.to("cpu")
    action_1 = torch.tanh(mean1).to("cpu")
    action_2 = torch.tanh(mean2).to("cpu")
    action_3 = torch.argmax(logits_discrete, dim=-1).to("cpu")

    return np.array([action_1, action_2, action_3]).transpose()

    # combined_action = np.array([action_1, action_2, action_3]).transpose()

    # if done_mask is None:
    #     assigned_action = [action for action in combined_action]
    # else:
    #     action_iterator = iter(combined_action)
    #     assigned_action = [
    #         next(action_iterator) if not (is_true) else None for is_true in done_mask
    #     ]

    # return assigned_action


def refactor_as_df(observations, actions, rewards):
    numpy_observations = np.array(observations)
    prev_obs = numpy_observations[:-1, :]
    curr_obs = numpy_observations[1:, :]

    acts = np.array(actions)
    cont_acts = acts[:, :2]
    disc_acts = acts[:, 2]

    rewards = np.array(rewards)

    episode_dict = {}

    for obs_dim in range(prev_obs.shape[1]):
        episode_dict[f"Previous_obs_{obs_dim}"] = prev_obs[:, obs_dim]

    for obs_dim in range(curr_obs.shape[1]):
        episode_dict[f"Current_obs_{obs_dim}"] = curr_obs[:, obs_dim]

    for act_dim in range(cont_acts.shape[1]):
        episode_dict[f"Act_{act_dim}"] = cont_acts[:, act_dim]

    episode_dict["Act_2"] = disc_acts

    episode_dict["Reward"] = rewards

    episode_df = pl.DataFrame(episode_dict)

    return episode_df
    # save_path = os.path.join(SAVE_DIR, "csvs", f"{dt.datetime.now()}.csv")
    # episode_df.write_csv(save_path)


# # %%
# tune.register_env("sbro_env_v1", sbro_env_creator)
# agent = PPO.from_checkpoint(
#     "/home/ybang-eai/research/2025/SBRO/SBRORL/result/PPO/2025-07-09 09:05:18.189259/PPO_SBRO/PPO_sbro_env_v1_65bcf_00000_0_2025-07-09_09-05-18/checkpoint_000000"
# )
# # %%
# ppomodule = agent.get_module()
# # %%
# # ppomodule.from_checkpoint(
# #     "/home/ybang-eai/research/2025/SBRO/SBRORL/result/PPO/2025-07-09 09:05:18.189259/PPO_SBRO/PPO_sbro_env_v1_65bcf_00000_0_2025-07-09_09-05-18/checkpoint_000089/learner_group/learner/rl_module/default_policy"
# # )

# %%

ppomodule = PPOModule.from_checkpoint(
    "/home/ybang-eai/research/2025/SBRO/SBRORL/result/PPO/2025-07-09 09:05:18.189259/PPO_SBRO/PPO_sbro_env_v1_65bcf_00000_0_2025-07-09_09-05-18/checkpoint_000075/learner_group/learner/rl_module/default_policy"
)
ppomodule.to("cuda:0")
# %%
for named_module in ppomodule.named_modules():
    print("\n+++++++++++++++++++++++++++++\n")
    print(named_module)
# %%
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
N_ENVS = 25
STARTING_PORT = 8100

env_settings = generate_eval_env_settings_v2(
    reward_config=REWARD_CONFIG,
    n_C_feed_scenarios=5,
    n_Q_obj_scenarios=5,
    n_time_scenarios=5,
)


# %%
async def request_step(env, action):
    if action is None:
        return None
    else:
        return await asyncio.to_thread(env.step, action)


ppomodule = PPOModule.from_checkpoint(
    "/home/ybang-eai/research/2025/SBRO/SBRORL/result/PPO/2025-07-09 09:05:18.189259/PPO_SBRO/PPO_sbro_env_v1_65bcf_00000_0_2025-07-09_09-05-18/checkpoint_000075/learner_group/learner/rl_module/default_policy"
)
ppomodule.to("cuda:0")


async def evaluate_settings():
    # env_settings = env_settings[:8]
    n_settings = len(env_settings)
    split_settings_idx = np.arange(0, n_settings, step=N_ENVS)
    # split_settings_idx = np.arange(0, n_settings, step=2)
    scheduled_settings = [
        [split_settings_idx[i], split_settings_idx[i + 1]]
        for i in range(len(split_settings_idx) - 1)
    ]

    evaluation_dfs = []

    for scheduled_setting in scheduled_settings:
        target_settings = deepcopy(
            env_settings[scheduled_setting[0] : scheduled_setting[1]]
        )
        obss = []

        for env_n, target_setting in enumerate(target_settings):
            target_setting["base_url"] = f"http://localhost:{STARTING_PORT + env_n}"

        target_vector_envs = [
            MinMaxNormalizeObservation(SBROEnv(**target_settings[0])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[1])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[2])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[3])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[4])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[5])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[6])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[7])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[8])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[9])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[10])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[11])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[12])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[13])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[14])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[15])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[16])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[17])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[18])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[19])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[20])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[21])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[22])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[23])),
            MinMaxNormalizeObservation(SBROEnv(**target_settings[24])),
        ]

        results = [
            {"observations": [], "actions": [], "rewards": []}
            for _ in target_vector_envs
        ]

        for env, result in zip(target_vector_envs, results):
            obs, _ = env.reset()
            result["observations"].append(obs)
            obss.append(obs)

        done_mask = [False for _ in target_vector_envs]

        next_obss = []
        for obs, done in zip(obss, done_mask):
            if not done:
                next_obss.append(obs)

        next_tensor_obss = torch.tensor(next_obss).to("cuda:0")

        ppo_outs = ppomodule.forward_inference({"obs": next_tensor_obss})

        next_actions = decode_action_determ(
            ppo_outs["action_dist_inputs"], done_mask=done_mask
        )

        task_schedule = [
            request_step(env, action)
            for env, action in zip(target_vector_envs, next_actions)
        ]

        while not all(done_mask):
            # 1. Collect observations ONLY from environments that are not done
            active_obss = [
                result["observations"][-1]
                for result, done in zip(results, done_mask)
                if not done
            ]

            # If all environments finished in the last step, break
            if not active_obss:
                break

            # 2. Get actions for ONLY the active environments
            next_tensor_obss = torch.tensor(active_obss, dtype=torch.float32).to(
                "cuda:0"
            )
            ppo_outs = ppomodule.forward_inference({"obs": next_tensor_obss})
            # This gives you an array of actions, e.g., shape (24, 3)
            active_actions = decode_action_determ(ppo_outs["action_dist_inputs"])

            # 3. Create the full action schedule, inserting `None` for done environments
            active_action_iter = iter(active_actions)
            next_actions = [
                next(active_action_iter) if not done else None for done in done_mask
            ]

            # 4. Schedule and execute the steps
            task_schedule = [
                request_step(env, action)
                for env, action in zip(target_vector_envs, next_actions)
            ]
            step_results = await asyncio.gather(*task_schedule)

            # 5. Correctly update results and done_mask
            for i, step_result in enumerate(step_results):
                # Only process results for environments that took a step
                if step_result is not None:
                    obs, reward, terminated, truncated, _ = step_result
                    results[i]["observations"].append(obs)
                    # Use the action that was actually taken for this environment
                    results[i]["actions"].append(next_actions[i])
                    results[i]["rewards"].append(reward)
                    done_mask[i] = terminated or truncated

        evaluation_dfs += [refactor_as_df(**result_dict) for result_dict in results]

    return evaluation_dfs


# evaluation_dfs = await asyncio.gather(evaluate_settings())


# %%
def evaluate_sequentially(env_settings, ppo_module):
    """
    Evaluates each environment setting sequentially in a single process.

    Args:
        env_settings (list): A list of configuration dictionaries for the environment.
        ppo_module (torch.nn.Module): The trained PPO RLModule.

    Returns:
        list: A list of Polars DataFrames, one for each completed episode.
    """
    evaluation_dfs = []

    # Use tqdm for a progress bar over all the settings
    for setting in tqdm(env_settings, desc="Evaluating Settings"):
        # 1. Initialize a single environment with the current setting
        setting["base_url"] = f"http://localhost:{STARTING_PORT}"
        env = MinMaxNormalizeObservation(SBROEnv(**setting))

        # 2. Store data for this single episode
        episode_observations = []
        episode_actions = []
        episode_rewards = []

        # 3. Reset the environment and store the initial observation
        obs, _ = env.reset()
        episode_observations.append(obs)

        # 4. Run the episode until it's done
        while True:
            # Prepare the observation for the model (add batch dim)
            obs_tensor = torch.tensor([obs], dtype=torch.float32).to("cuda:0")

            # Get action parameters from the policy module
            with torch.no_grad():
                ppo_outs = ppo_module.forward_inference({"obs": obs_tensor})

            # Decode the single action from the output batch
            # decode_actions returns a numpy array, so we take the first row
            action = decode_action_determ(ppo_outs["action_dist_inputs"])[0]

            # Take a step in the environment
            obs, reward, terminated, truncated, _ = env.step(action)

            # Store the results for this step
            episode_observations.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)

            # 5. Check for episode termination
            if terminated or truncated:
                break

        # 6. After the episode, refactor the collected data into a DataFrame
        # Note: refactor_as_df expects rewards for n-1 steps
        episode_df = refactor_as_df(
            observations=episode_observations,
            actions=episode_actions,
            rewards=episode_rewards,
        )
        evaluation_dfs.append(episode_df)

    return evaluation_dfs


evaluation_dfs = evaluate_sequentially(env_settings, ppomodule)
# %%
os.makedirs(os.path.join(SAVE_DIR, "eval_csvs"))
for i, eval_df in enumerate(evaluation_dfs[0]):
    save_path = os.path.join(SAVE_DIR, "eval_csvs", f"{i}.csv")
    eval_df.write_csv(save_path)

# %%
for settings in env_settings:
    env = MinMaxNormalizeObservation(SBROEnv(**settings))
# %%
env = MinMaxNormalizeObservation(SBROEnv(**env_settings[0]))

obs, info = env.reset()

# %%
(obs)

# %%
ppomodule.get_inference_action_dist_cls()
# %%
# ppo_out = ppomodule({'obs':torch.tensor([obs]).to("cuda:0")})
# embedding = ppo_out["embeddings"]
# action_dist_inputs = ppo_out["action_dist_inputs"]

# # %%
# embedding

# # %%
# action_dist_inputs
# %%
ppo_out = ppomodule.forward_inference({"obs": torch.tensor([obs]).to("cuda:0")})

ppo_out["action_dist_inputs"]
# %%
next_action = decode_action_determ(ppo_out["action_dist_inputs"])

print(next_action)

observation, reward, terminated, truncated, info = env.step(
    action=next_action.squeeze()
)

print(observation)
# %%

for iter in range(50, 70):
    ppomodule = PPOModule.from_checkpoint(
        f"/home/ybang-eai/research/2025/SBRO/SBRORL/result/PPO/2025-07-09 09:05:18.189259/PPO_SBRO/PPO_sbro_env_v1_65bcf_00000_0_2025-07-09_09-05-18/checkpoint_0000{iter}/learner_group/learner/rl_module/default_policy"
    )
    ppomodule.to("cuda:0")

    terminated = False
    truncated = False

    observations = []
    actions = []
    rewards = []

    observation, _ = env.reset()
    observations.append(observation)

    with torch.no_grad():
        while not (terminated or truncated):
            agent_output = ppomodule.forward_inference(
                {"obs": torch.tensor([observation]).to("cuda:0")}
            )
            next_action = decode_action_determ(agent_output["action_dist_inputs"])
            observation, reward, terminated, truncated, info = env.step(
                action=next_action.squeeze()
            )
            observations.append(observation)
            actions.append(next_action)
            rewards.append(reward)

    actions = np.array(actions).squeeze()

    for i in range(3):
        plt.plot(actions[:, i])
        plt.title(f"action {i}")
        plt.show()

    plt.plot(np.array(rewards).squeeze())
    plt.show()

    plt.plot(np.cumsum(np.array(rewards).squeeze()))
    plt.show()

# %%
refactor_as_df(observations, actions, rewards)
# %%
