import argparse
import os

import numpy as np
import polars as pl
import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule as PPOModule,
)

from gymnasium_env.SBROEnvironment import SBROEnv
from gymnasium_env.utils import MinMaxNormalizeObservation


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


def decode_action_determ(nn_output: torch.Tensor):
    """
    Decodes the deterministic action from the 6-element nn_output
    based on the correct [4, 2] input split.

    Args:
        nn_output: Tensor of shape (batch_size, 6).
    """
    # 1. Correctly slice the tensor based on input_lens=[4, 2]
    means_continuous = nn_output[:, :2]  # Indices 0, 1
    logits_discrete = nn_output[:, 4:]  # Indices 4, 5

    # 2. Calculate the deterministic actions
    action_discrete = torch.argmax(logits_discrete, dim=-1)
    actions_continuous = torch.tanh(means_continuous)

    # 3. Combine and format the output
    action_discrete_unsqueezed = action_discrete.unsqueeze(1)
    combined_actions = torch.cat(
        (actions_continuous, action_discrete_unsqueezed.float()), dim=1
    )

    return combined_actions.cpu().numpy()


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


def main():
    device = "cuda:0"
    parser = argparse.ArgumentParser(description="Evaluate PPO models.")
    parser.add_argument(
        "--checkpoints-path",
        type=str,
    )
    parser.add_argument(
        "--checkpoint-number",
        type=int,
    )
    parser.add_argument("--port", type=int)

    args = parser.parse_args()

    checkpoints_path = args.checkpoints_path
    checkpoint_number = args.checkpoint_number
    port_number = args.port

    target_checkpoint_path = os.path.join(
        checkpoints_path,
        f"checkpoint_0000{str(checkpoint_number).rjust(2, '0')}/learner_group/learner/rl_module/default_policy",
    )

    ppomodule = PPOModule.from_checkpoint(target_checkpoint_path)
    ppomodule.to(device)

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

    env_settings = generate_eval_env_settings_v2(
        reward_config=REWARD_CONFIG,
        n_C_feed_scenarios=5,
        n_Q_obj_scenarios=5,
        n_time_scenarios=5,
    )

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
        # for setting in tqdm(env_settings, desc="Evaluating Settings"):
        for setting in env_settings:
            # 1. Initialize a single environment with the current setting
            setting["base_url"] = f"http://localhost:{port_number}"
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
                obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(
                    device
                )

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

    save_path = os.path.join(
        checkpoints_path,
        "evaluation_results",
        f"checkpoint_0000{str(checkpoint_number).rjust(2, '0')}",
    )
    os.makedirs(save_path, exist_ok=True)

    for n, evaluation_df in enumerate(evaluation_dfs):
        evaluation_df.write_csv(os.path.join(save_path, f"{n}.csv"))

    return None


if __name__ == "__main__":
    main()
