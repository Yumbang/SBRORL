# %%
import os
import re
import sys

import numpy as np
import pandas as pd
import polars as pl
import scikit_posthocs as sp
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import f_oneway, friedmanchisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm


# %% Copied from read_result_files.py
def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def get_cycles_from_df(df: pl.DataFrame) -> list[pl.DataFrame]:
    """
    Splits a DataFrame into cycles of 'CC' and 'purge' modes using vectorized Polars operations.

    A cycle is defined as a sequence of 'CC' states followed by a sequence of
    'purge' states. If the data starts with 'purge', it's treated as a
    single initial cycle.

    Args:
        df: The input Polars DataFrame. Must contain an 'Act_2' column.

    Returns:
        A list of Polars DataFrames, where each DataFrame represents a cycle.
    """
    if df.is_empty():
        return []

    df = df.with_columns(
        pl.when(pl.col("Act_2") == 0)
        .then(pl.lit("CC"))
        .otherwise(pl.lit("purge"))
        .alias("mode")
    )

    # Get the start indices of each consecutive block of 'mode'
    block_start_indices = (
        df.select(
            (pl.col("mode").ne(pl.col("mode").shift(1))).fill_null(True).arg_true()
        )
        .to_series()
        .to_list()
    )

    # Get the mode type for each block
    block_modes = (
        df.select(
            pl.col("mode").filter(
                pl.col("mode").ne(pl.col("mode").shift(1)).fill_null(True)
            )
        )
        .to_series()
        .to_list()
    )

    cycles = []
    current_block_idx = 0

    # Handle initial purge cycle if the DataFrame starts with 'purge'
    if block_modes and block_modes[0] == "purge":
        # The first block is a purge block
        purge_block_end_row_idx = (
            block_start_indices[1] if len(block_start_indices) > 1 else len(df)
        )
        cycles.append(df.slice(0, purge_block_end_row_idx))
        current_block_idx = 1  # Move to the next block

    # Process subsequent CC-purge cycles
    while current_block_idx < len(block_modes):
        if block_modes[current_block_idx] == "CC":
            cc_start_row_idx = block_start_indices[current_block_idx]

            # Check if there's a subsequent purge block
            if (
                current_block_idx + 1 < len(block_modes)
                and block_modes[current_block_idx + 1] == "purge"
            ):
                purge_end_row_idx = (
                    block_start_indices[current_block_idx + 2]
                    if current_block_idx + 2 < len(block_start_indices)
                    else len(df)
                )
                cycles.append(
                    df.slice(cc_start_row_idx, purge_end_row_idx - cc_start_row_idx)
                )
                current_block_idx += 2  # Move past both CC and purge blocks
            else:  # CC block is not followed by a purge block, or it's the last block
                cc_end_row_idx = (
                    block_start_indices[current_block_idx + 1]
                    if current_block_idx + 1 < len(block_start_indices)
                    else len(df)
                )
                cycles.append(
                    df.slice(cc_start_row_idx, cc_end_row_idx - cc_start_row_idx)
                )
                current_block_idx += 1  # Move past the CC block
        else:  # This case should ideally not be hit if initial purge is handled, but for safety
            current_block_idx += (
                1  # Skip unexpected purge block (e.g., if a CC-purge cycle was broken)
            )

    return cycles


def get_cyclic_recovery(df):
    split_dfs = get_cycles_from_df(df)
    processed_dfs = []
    for i, split_df in enumerate(split_dfs):
        # Assuming 'Current Q_disp' and 'Current Q_perm' are columns in split_df
        feed_Q = split_df["Current Q_disp"].sum() + split_df["Current Q_perm"].sum()
        perm_Qs = np.cumsum(split_df["Current Q_perm"])
        cyclic_recovery = np.array(perm_Qs / feed_Q)
        processed_dfs.append(
            split_df.with_columns(
                pl.Series(name="Cyclic recovery", values=cyclic_recovery)
            )
        )
    merged_df = pl.concat(processed_dfs)
    return merged_df


# %%


class EvalResultReader:
    """
    Analyzes and visualizes evaluation results generated at different checkpoints
    during a reinforcement learning training process.
    """

    def __init__(self, exp_path):
        """
        Initializes the reader by locating, indexing, and pre-processing all
        evaluation result files.

        Args:
            exp_path (str): The root path of the experiment directory.
        """
        self.exp_path = exp_path
        self.discount_rate = 0.999  # Gamma for discounted reward calculation
        self.target_path = os.path.join(self.exp_path, "evaluation_results")
        if not os.path.exists(self.target_path):
            raise FileNotFoundError(
                f"The evaluation results directory does not exist: {self.target_path}"
            )

        # --- Index all evaluation CSV files and map them to their checkpoints ---
        eval_paths = [
            os.path.join(self.target_path, d)
            for d in os.listdir(self.target_path)
            if os.path.isdir(os.path.join(self.target_path, d))
        ]

        eval_csv_path_dicts = []
        for checkpoint_path in eval_paths:
            checkpoint_name = os.path.basename(checkpoint_path)
            match = re.search(r"\d+", checkpoint_name)
            if not match:
                continue

            checkpoint_num = int(match.group())

            try:
                csv_files = os.listdir(checkpoint_path)
                # Correctly extract episode number for sorting
                csv_files.sort(key=lambda path: int(os.path.splitext(path)[0]))

                for csv_path in csv_files:
                    episode_n = int(os.path.splitext(csv_path)[0])
                    eval_csv_path_dicts.append(
                        {
                            "checkpoint_name": checkpoint_name,
                            "checkpoint_num": checkpoint_num,
                            "episode_n": episode_n,
                            "full_path": os.path.join(checkpoint_path, csv_path),
                        }
                    )
            except Exception as e:
                print(f"Could not process directory {checkpoint_path}: {e}")

        if not eval_csv_path_dicts:
            raise FileNotFoundError(
                f"No valid evaluation CSV files found in {self.target_path}"
            )

        self.eval_csv_path_df = pl.DataFrame(eval_csv_path_dicts).sort(
            ["checkpoint_num", "episode_n"]
        )

        # --- Create directory for saving figures ---
        self.figure_dir = os.path.join(
            "figures", f"{os.path.basename(self.exp_path)}_evaluation"
        )
        os.makedirs(self.figure_dir, exist_ok=True)

        self.stats_dir = os.path.join(self.figure_dir, "statistical_tests")
        os.makedirs(self.stats_dir, exist_ok=True)

        # --- Environment-specific ranges for denormalization ---
        self.obs_range_dict = {
            "T_feed": [0.0, 50.0],
            "C_feed": [0.0, 0.25],
            "C_pipe_c_out": [0.0, 10.0],
            "P_m_in": [0.0, 25e5],
            "P_m_out": [0.0, 25e5],
            "Q_circ": [0.0, 10.0],
            "Q_disp": [0.0, 10.0],
            "Q_perm": [0.0, 10.0],
            "C_perm": [0.0, 0.25],
            "time_remaining": [-86400.0, 86400.0],
            "V_perm_remaining": [-16.0, 32.0],
        }
        self.action_range_dict = {"Q0": [4.0, 6.0], "R_sp": [0.1, 0.3], "mode": None}

        # --- Pre-calculate and store reward sums ---
        self._calculate_and_store_reward_summaries()

        print(f"🚀 EvalResultReader is initiated for {self.exp_path}!")
        print(
            f"Found and processed {self.eval_csv_path_df.height} evaluation files across {self.eval_csv_path_df['checkpoint_name'].n_unique()} checkpoints."
        )

    def _calculate_discounted_reward(self, rewards: pl.Series) -> float:
        """Calculates the discounted sum of a series of rewards."""
        discounts = self.discount_rate ** np.arange(len(rewards))
        return np.dot(rewards.to_numpy(), discounts)

    def _calculate_and_store_reward_summaries(self):
        """
        Reads all episode files to calculate simple and discounted reward sums,
        then appends them as new columns to the main DataFrame.
        """
        print(
            "Calculating reward summaries for all episodes (this may take a moment)..."
        )

        reward_sums = []
        discounted_reward_sums = []

        for path in tqdm(
            self.eval_csv_path_df["full_path"], desc="Processing episodes"
        ):
            rewards = pl.read_csv(path)["Reward"]
            reward_sums.append(rewards.sum())
            discounted_reward_sums.append(self._calculate_discounted_reward(rewards))

        self.eval_csv_path_df = self.eval_csv_path_df.with_columns(
            [
                pl.Series("reward_sum", reward_sums),
                pl.Series("discounted_reward_sum", discounted_reward_sums),
            ]
        )

    def get_summary_df(self, sort_by="mean_discounted_reward_sum", descending=True):
        """
        Calculates summary statistics for each checkpoint and sorts the results.

        Args:
            sort_by (str): The column to sort by. Can be
                           "mean_discounted_reward_sum" or "median_discounted_reward_sum".
            descending (bool): Whether to sort in descending order.

        Returns:
            polars.DataFrame: A DataFrame with summary statistics for each checkpoint.
        """
        summary_df = self.eval_csv_path_df.group_by(
            ["checkpoint_name", "checkpoint_num"]
        ).agg(
            pl.mean("discounted_reward_sum").alias("mean_discounted_reward_sum"),
            pl.median("discounted_reward_sum").alias("median_discounted_reward_sum"),
            pl.min("discounted_reward_sum").alias("min_discounted_reward_sum"),
            pl.std("discounted_reward_sum").alias("std_discounted_reward_sum"),
            pl.count().alias("num_episodes"),
        )

        return summary_df.sort(by=sort_by, descending=descending)

    def _denormalize_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Denormalizes specific observation columns of a Polars DataFrame."""
        feature_names = list(self.obs_range_dict.keys())
        denormalization_exprs = []
        cols_to_drop = []

        for i, feature_name in enumerate(feature_names):
            for obs_type in ["Previous", "Current"]:
                col_name = f"{obs_type}_obs_{i}"
                if col_name in df.columns:
                    min_val, max_val = self.obs_range_dict[feature_name]
                    expr = (pl.col(col_name) * (max_val - min_val) + min_val).alias(
                        f"{obs_type} {feature_name}"
                    )
                    denormalization_exprs.append(expr)
                    cols_to_drop.append(col_name)

        if not denormalization_exprs:
            return df
        return df.with_columns(denormalization_exprs).drop(cols_to_drop)

    def plot_episode(self, show=True, episode_n=-1):
        if episode_n < 0:
            episode_n = self.eval_csv_path_df.height + episode_n

        target_episode_row = self.eval_csv_path_df.row(episode_n, named=True)
        target_episode_path = target_episode_row["full_path"]
        checkpoint_name = target_episode_row["checkpoint_name"]
        episode_num = target_episode_row["episode_n"]

        episode_save_dir = os.path.join(
            self.figure_dir, checkpoint_name, str(episode_num)
        )
        try:
            os.makedirs(episode_save_dir)

            target_episode_df = pl.read_csv(target_episode_path)

            denorm_target_df = self._denormalize_dataframe(df=target_episode_df)

            obs_names = list(self.obs_range_dict.keys())
            action_names = list(self.action_range_dict.keys())

            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            fig.suptitle(f"Episode [{episode_num}] - Observations")

            for i in range(11):
                ax = axes[i // 4, i % 4]
                ax.plot(denorm_target_df[f"Previous {obs_names[i]}"], lw=0.5)
                ax.set_title(f"{obs_names[i]}")
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, linestyle="--", alpha=0.7)

            if i == 10:
                fig.delaxes(axes[2, 3])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(episode_save_dir, "Previous_obs_all.pdf"))

            if show:
                plt.show()
            else:
                plt.close()

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Episode [{episode_num}] - Actions", fontsize=16)

            for j in range(2):
                ax = axes[j // 2, j % 2]
                ax.plot(
                    np.tanh(
                        denorm_target_df[f"Act_{j}"],
                    ),
                    lw=0.5,
                )
                ax.set_title(f"Act_{j} ({action_names[j]})")
                ax.set_ylim([-1, 1])
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, linestyle="--", alpha=0.7)

            ax = axes[1, 0]
            ax.plot(denorm_target_df["Act_2"], lw=0.5)
            ax.set_title(f"Act_2 ({action_names[2]})")
            ax.set_ylim([-0.1, 1.1])
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, linestyle="--", alpha=0.7)

            fig.delaxes(axes[1, 1])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(episode_save_dir, "Actions_all.pdf"))

            if show:
                plt.show()
            else:
                plt.close()

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Episode [{episode_num}] - Rewards", fontsize=16)

            ax = axes[0]
            ax.plot(denorm_target_df["Reward"], lw=0.5)
            ax.set_title("Reward")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, linestyle="--", alpha=0.7)

            ax = axes[1]
            ax.plot(np.cumsum(denorm_target_df["Reward"]), lw=0.5)
            ax.set_title("Cumulative Reward")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, linestyle="--", alpha=0.7)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(episode_save_dir, "Reward_summary.pdf"))

            if show:
                plt.show()
            else:
                plt.close()
            return True

        except FileExistsError as e:
            print("The directory already exists\n", e)
            return False

    def plot_episode_post_analysis(self, show=True, episode_n=-1):
        if episode_n < 0:
            episode_n = self.eval_csv_path_df.height + episode_n

        target_episode_row = self.eval_csv_path_df.row(episode_n, named=True)
        target_episode_path = target_episode_row["full_path"]
        checkpoint_name = target_episode_row["checkpoint_name"]
        episode_num = target_episode_row["episode_n"]

        episode_save_dir = os.path.join(
            self.figure_dir, checkpoint_name, str(episode_num)
        )

        os.makedirs(episode_save_dir, exist_ok=True)

        target_episode_df = pl.read_csv(target_episode_path)

        denorm_target_df = self._denormalize_dataframe(df=target_episode_df)

        target_df_with_cyclic_recovery = get_cyclic_recovery(denorm_target_df)

        plt.figure(figsize=(5, 5))
        plt.plot(target_df_with_cyclic_recovery["Cyclic recovery"], lw=0.5)
        plt.grid(True, linestyle=":", alpha=0.7)

        plt.tick_params(axis="x", rotation=45)
        plt.ylabel("Cyclic recovery")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.gca().set_yticks(np.arange(0, 1.05, 0.05), minor=True)
        plt.ylim([0.0, 1.05])
        plt.title("Cyclic recovery")
        plt.tight_layout()
        plt.savefig(os.path.join(episode_save_dir, "Cyclic recovery.pdf"))
        if show:
            plt.show()
        else:
            plt.close()

        cycle_lengths = [
            len(split_df) for split_df in get_cycles_from_df(denorm_target_df)
        ]
        plt.figure(figsize=(5, 5))
        plt.hist(cycle_lengths, bins=30, alpha=0.5)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.xlabel("Cycle length")
        plt.ylabel("Frequency")
        plt.title("Cycle length distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(episode_save_dir, "Cycle length distribution.pdf"))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_all_episodes(self):
        for i in tqdm(
            range(self.eval_csv_path_df.height),
            desc="Plotting all evaluation episodes",
        ):
            done = self.plot_episode(show=False, episode_n=i)
            if done:
                self.plot_episode_post_analysis(show=False, episode_n=i)


# %%

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_evaluation_files.py <experiment_path>")
        sys.exit(1)

    experiment_path = sys.argv[1]
    reader = EvalResultReader(experiment_path)

    # %%
    reader.plot_all_episodes()
    # %%
    checkpoint_split_dfs = reader.eval_csv_path_df.partition_by(by="checkpoint_name")
    # %%
    # Apply Friedman test to all of the checkpoints
    checkpoint_split_returns = [
        np.array(checkpoint_split_df["discounted_reward_sum"])
        for checkpoint_split_df in checkpoint_split_dfs
    ]

    if len(set(len(arr) for arr in checkpoint_split_returns)) > 1:
        print("Warning: The number of episodes is not the same for all checkpoints.")
        print(
            "The Friedman test requires a balanced design. Consider using only the common episodes across all checkpoints."
        )
    else:
        # Unpack the list of arrays into separate arguments for the function
        statistic, p_value = friedmanchisquare(*checkpoint_split_returns)

        print(f"Friedman Test Statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.4f}")

        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            print("\nThe p-value is less than 0.05. We can reject the null hypothesis.")
            print(
                "Conclusion: There is a statistically significant difference in performance between the checkpoints."
            )
        else:
            print(
                "\nThe p-value is not less than 0.05. We cannot reject the null hypothesis."
            )
            print(
                "Conclusion: There is not enough evidence to say that there are significant differences in performance between the checkpoints."
            )

    # %%
    # Get and print the summary DataFrame sorted by mean
    mean_sorted_summary = reader.get_summary_df(sort_by="mean_discounted_reward_sum")
    print("--- Summary Sorted by Mean Discounted Reward Sum ---")
    print(mean_sorted_summary)

    # Get and print the summary DataFrame sorted by median
    median_sorted_summary = reader.get_summary_df(
        sort_by="median_discounted_reward_sum"
    )
    print("\n--- Summary Sorted by Median Discounted Reward Sum ---")
    print(median_sorted_summary)

    # %% Friedman - Conover analysis of top N mean checkpoints

    TOP_N = 4
    top_N_mean_checkpoints = mean_sorted_summary["checkpoint_name"][:TOP_N]

    top_N_mean_data = [
        reader.eval_csv_path_df.filter(pl.col("checkpoint_name") == checkpoint_name)[
            "discounted_reward_sum"
        ]
        for checkpoint_name in top_N_mean_checkpoints
    ]
    statistic, p_value = friedmanchisquare(*top_N_mean_data)

    print(f"Friedman Test Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("\nThe p-value is less than 0.05. We can reject the null hypothesis.")
        print(
            "Conclusion: There is a statistically significant difference in performance between the checkpoints."
        )
    else:
        print(
            "\nThe p-value is not less than 0.05. We cannot reject the null hypothesis."
        )
        print(
            "Conclusion: There is not enough evidence to say that there are significant differences in performance between the checkpoints."
        )

    # Post-hoc analysis (Conover - Iman test)
    top_N_mean_checkpoints = mean_sorted_summary["checkpoint_name"][:TOP_N]

    top_N_mean_data = [
        reader.eval_csv_path_df.filter(pl.col("checkpoint_name") == checkpoint_name)[
            "discounted_reward_sum"
        ]
        for checkpoint_name in top_N_mean_checkpoints
    ]
    statistic, p_value = friedmanchisquare(*top_N_mean_data)

    print(f"Friedman Test Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("\nThe p-value is less than 0.05. We can reject the null hypothesis.")
        print(
            "Conclusion: There is a statistically significant difference in performance between the checkpoints."
        )
    else:
        print(
            "\nThe p-value is not less than 0.05. We cannot reject the null hypothesis."
        )
        print(
            "Conclusion: There is not enough evidence to say that there are significant differences in performance between the checkpoints."
        )

    labels = top_N_mean_checkpoints.to_list()

    data_for_test = pd.DataFrame(np.array(top_N_mean_data).transpose(), columns=labels)

    conover_p_value_matrix = sp.posthoc_conover_friedman(data_for_test)

    # --- Save statistical test results ---
    friedman_results_text = f"""Friedman Test Statistic: {statistic:.4f}
    P-value: {p_value:.4f}
    """
    with open(os.path.join(reader.stats_dir, "friedman_results_mean.txt"), "w") as f:
        f.write(friedman_results_text)

    conover_p_value_matrix.to_csv(
        os.path.join(reader.stats_dir, "conover_p_values_mean.csv")
    )

    # --- Customize labels for the heatmap ---
    def format_checkpoint_name(name: str) -> str:
        try:
            # Extracts the numeric part of the checkpoint name and converts it to an integer
            number = int(name.split("_")[-1])
            return f"Checkpoint {number}"
        except (ValueError, IndexError):
            # Returns the original name if parsing fails
            return name

    conover_p_value_matrix.columns = [
        format_checkpoint_name(col) for col in conover_p_value_matrix.columns
    ]
    conover_p_value_matrix.index = [
        format_checkpoint_name(idx) for idx in conover_p_value_matrix.index
    ]

    # --- Create the heatmap ---
    plt.figure(figsize=(8, 7))
    heatmap = sns.heatmap(
        conover_p_value_matrix,
        annot=True,  # Write the p-value in each cell
        fmt=".3f",  # Format p-values to 3 decimal places
        cmap="viridis",  # Use a color-blind friendly colormap
        linewidths=0.5,  # Draw lines between cells
        linecolor="black",
        vmin=0,  # Set the minimum of the colorbar to 0
        vmax=1,  # Set the maximum of the colorbar to 1
    )

    # --- Final plot adjustments ---
    plt.xticks(rotation=45, ha="right")  # Rotate xticks for better readability
    plt.yticks(rotation=0, va="center")  # Rotate yticks
    plt.title("Conover-Friedman Post-Hoc Test P-Values")
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(os.path.join(reader.stats_dir, "conover_heatmap_mean.pdf"))
    plt.show()
    plt.close()

    # --- Boxplot for Top N Checkpoints (Mean) ---

    # Filter data for top N checkpoints based on mean rewards
    top_N_mean_df = reader.eval_csv_path_df.filter(
        pl.col("checkpoint_name").is_in(top_N_mean_checkpoints)
    ).to_pandas()

    # Apply the custom formatter to the checkpoint names
    top_N_mean_df["checkpoint_name"] = top_N_mean_df["checkpoint_name"].apply(
        format_checkpoint_name
    )

    # Define the order for the plot based on the sorted summary
    plot_order_mean = [
        format_checkpoint_name(c) for c in top_N_mean_checkpoints.to_list()
    ]

    # --- Violinplot for Top N Checkpoints (Mean) ---
    plt.figure(figsize=(6, 8))
    plt.grid(True, alpha=0.25)
    sns.violinplot(
        x="checkpoint_name",
        y="discounted_reward_sum",
        data=top_N_mean_df,
        order=plot_order_mean,
        # color='white',
        inner="box",  # Can be 'box', 'quartile', 'point', 'stick', or None
        width=0.55,
        palette=sns.color_palette(),
    )
    plt.title("Distribution of Discounted Reward Sum for Top 5 Checkpoints")
    plt.xlabel("Checkpoint")
    plt.ylabel("Discounted Reward Sum")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(reader.stats_dir, "violinplot_mean.pdf"))
    plt.show()
    plt.close()

    # %% ANOVA - Tukey HSD analysis of top N mean checkpoints

    # Prepare data for ANOVA
    # Concatenate all data into a single Series and create a group indicator
    all_data_mean = pl.concat(top_N_mean_data)
    groups_mean = pl.Series(
        [labels[i] for i, data in enumerate(top_N_mean_data) for _ in range(len(data))]
    )

    # Perform ANOVA
    f_statistic_mean, p_value_anova_mean = f_oneway(
        *[data.to_numpy() for data in top_N_mean_data]
    )

    print(f"\n--- ANOVA for Top {TOP_N} Mean Checkpoints ---")
    print(f"ANOVA F-statistic: {f_statistic_mean:.4f}")
    print(f"ANOVA P-value: {p_value_anova_mean:.4f}")

    # Save ANOVA results
    anova_results_text_mean = f"""ANOVA F-statistic: {f_statistic_mean:.4f}
P-value: {p_value_anova_mean:.4f}
"""
    with open(os.path.join(reader.stats_dir, "anova_results_mean.txt"), "w") as f:
        f.write(anova_results_text_mean)

    # Interpretation
    if p_value_anova_mean < alpha:
        print("\nThe p-value is less than 0.05. We can reject the null hypothesis.")
        print(
            "Conclusion: There is a statistically significant difference in performance between the checkpoints."
        )
        # Perform Tukey's HSD post-hoc test
        tukey_result_mean = pairwise_tukeyhsd(
            endog=all_data_mean.to_numpy(), groups=groups_mean.to_numpy(), alpha=alpha
        )
        print("\n--- Tukey's HSD Post-Hoc Test Results (Mean) ---")
        print(tukey_result_mean)

        # Save Tukey's HSD results
        with open(
            os.path.join(reader.stats_dir, "tukey_hsd_results_mean.txt"), "w"
        ) as f:
            f.write(str(tukey_result_mean))

        # Convert Tukey's HSD results to a DataFrame for heatmap
        tukey_df_mean = pd.DataFrame(
            data=tukey_result_mean._results_table.data[1:],
            columns=tukey_result_mean._results_table.data[0],
        )
        tukey_p_value_matrix_mean = pd.pivot_table(
            tukey_df_mean,
            values="p-adj",
            index="group1",
            columns="group2",
            fill_value=1.0,
        )
        # Fill diagonal with 1.0 (comparison with itself is not significant)
        for col in tukey_p_value_matrix_mean.columns:
            if col in tukey_p_value_matrix_mean.index:
                tukey_p_value_matrix_mean.loc[col, col] = 1.0

        # Ensure symmetry
        for i in range(len(tukey_p_value_matrix_mean.index)):
            for j in range(i + 1, len(tukey_p_value_matrix_mean.columns)):
                val = tukey_p_value_matrix_mean.iloc[i, j]
                tukey_p_value_matrix_mean.iloc[j, i] = val

        tukey_p_value_matrix_mean.to_csv(
            os.path.join(reader.stats_dir, "tukey_hsd_p_values_mean.csv")
        )

        # Customize labels for the heatmap
        tukey_p_value_matrix_mean.columns = [
            format_checkpoint_name(col) for col in tukey_p_value_matrix_mean.columns
        ]
        tukey_p_value_matrix_mean.index = [
            format_checkpoint_name(idx) for idx in tukey_p_value_matrix_mean.index
        ]

        # Create the heatmap for Tukey's HSD
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            tukey_p_value_matrix_mean,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            linecolor="black",
            vmin=0,
            vmax=1,
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0, va="center")
        plt.title("Tukey's HSD Post-Hoc Test P-Values (Mean)")
        plt.tight_layout()
        plt.savefig(os.path.join(reader.stats_dir, "tukey_hsd_heatmap_mean.pdf"))
        plt.show()
        plt.close()

    else:
        print(
            "\nThe p-value is not less than 0.05. We cannot reject the null hypothesis."
        )
        print(
            "Conclusion: There is not enough evidence to say that there are significant differences in performance between the checkpoints."
        )

    # %% Friedman - Conover analysis of top N median checkpoints

    top_N_median_checkpoints = median_sorted_summary["checkpoint_name"][:TOP_N]

    top_N_median_data = [
        reader.eval_csv_path_df.filter(pl.col("checkpoint_name") == checkpoint_name)[
            "discounted_reward_sum"
        ]
        for checkpoint_name in top_N_median_checkpoints
    ]
    statistic, p_value = friedmanchisquare(*top_N_median_data)

    print(f"Friedman Test Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("\nThe p-value is less than 0.05. We can reject the null hypothesis.")
        print(
            "Conclusion: There is a statistically significant difference in performance between the checkpoints."
        )
    else:
        print(
            "\nThe p-value is not less than 0.05. We cannot reject the null hypothesis."
        )
        print(
            "Conclusion: There is not enough evidence to say that there are significant differences in performance between the checkpoints."
        )

    labels = top_N_median_checkpoints.to_list()

    data_for_test = pd.DataFrame(
        np.array(top_N_median_data).transpose(), columns=labels
    )

    conover_p_value_matrix = sp.posthoc_conover_friedman(data_for_test)
    # --- Save statistical test results ---
    friedman_results_text = f"""Friedman Test Statistic: {statistic:.4f}
    P-value: {p_value:.4f}
    """
    with open(os.path.join(reader.stats_dir, "friedman_results_median.txt"), "w") as f:
        f.write(friedman_results_text)

    conover_p_value_matrix.to_csv(
        os.path.join(reader.stats_dir, "conover_p_values_median.csv")
    )

    conover_p_value_matrix.columns = [
        format_checkpoint_name(col) for col in conover_p_value_matrix.columns
    ]
    conover_p_value_matrix.index = [
        format_checkpoint_name(idx) for idx in conover_p_value_matrix.index
    ]

    # --- Create the heatmap ---
    plt.figure(figsize=(8, 7))
    heatmap = sns.heatmap(
        conover_p_value_matrix,
        annot=True,  # Write the p-value in each cell
        fmt=".3f",  # Format p-values to 3 decimal places
        cmap="viridis_r",  # Use a color-blind friendly colormap
        linewidths=0.5,  # Draw lines between cells
        linecolor="black",
        vmin=0,  # Set the minimum of the colorbar to 0
        vmax=1,  # Set the maximum of the colorbar to 1
    )

    # --- Final plot adjustments ---
    plt.xticks(rotation=45, ha="right")  # Rotate xticks for better readability
    plt.yticks(rotation=0, va="center")  # Rotate yticks
    plt.title("Conover-Friedman Post-Hoc Test P-Values")
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(os.path.join(reader.stats_dir, "conover_heatmap_median.pdf"))
    plt.show()
    plt.close()

    # --- Boxplot for Top N Checkpoints (Median) ---

    # Filter data for top N checkpoints based on median rewards
    top_N_median_df = reader.eval_csv_path_df.filter(
        pl.col("checkpoint_name").is_in(top_N_median_checkpoints)
    ).to_pandas()

    # Apply the custom formatter to the checkpoint names
    top_N_median_df["checkpoint_name"] = top_N_median_df["checkpoint_name"].apply(
        format_checkpoint_name
    )

    # Define the order for the plot based on the sorted summary
    plot_order_median = [
        format_checkpoint_name(c) for c in top_N_median_checkpoints.to_list()
    ]

    # --- Violinplot for Top N Checkpoints (Median) ---
    plt.figure(figsize=(6, 8))
    plt.grid(True, alpha=0.25)
    sns.violinplot(
        x="checkpoint_name",
        y="discounted_reward_sum",
        data=top_N_median_df,
        order=plot_order_median,
        inner="box",  # Can be 'box', 'quartile', 'point', 'stick', or None
        width=0.55,
        palette=sns.color_palette(),
    )
    plt.title("Distribution of Discounted Reward Sum for Top 5 Checkpoints")
    plt.xlabel("Checkpoint")
    plt.ylabel("Discounted Reward Sum")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(reader.stats_dir, "violinplot_median.pdf"))
    plt.show()

    # %% ANOVA - Tukey HSD analysis of top N median checkpoints

    # Prepare data for ANOVA
    # Concatenate all data into a single Series and create a group indicator
    all_data_median = pl.concat(top_N_median_data)
    groups_median = pl.Series(
        [
            labels[i]
            for i, data in enumerate(top_N_median_data)
            for _ in range(len(data))
        ]
    )

    # Perform ANOVA
    f_statistic_median, p_value_anova_median = f_oneway(
        *[data.to_numpy() for data in top_N_median_data]
    )

    print(f"\n--- ANOVA for Top {TOP_N} Median Checkpoints ---")
    print(f"ANOVA F-statistic: {f_statistic_median:.4f}")
    print(f"ANOVA P-value: {p_value_anova_median:.4f}")

    # Save ANOVA results
    anova_results_text_median = f"""ANOVA F-statistic: {f_statistic_median:.4f}
P-value: {p_value_anova_median:.4f}
"""
    with open(os.path.join(reader.stats_dir, "anova_results_median.txt"), "w") as f:
        f.write(anova_results_text_median)

    # Interpretation
    if p_value_anova_median < alpha:
        print("\nThe p-value is less than 0.05. We can reject the null hypothesis.")
        print(
            "Conclusion: There is a statistically significant difference in performance between the checkpoints."
        )
        # Perform Tukey's HSD post-hoc test
        tukey_result_median = pairwise_tukeyhsd(
            endog=all_data_median.to_numpy(),
            groups=groups_median.to_numpy(),
            alpha=alpha,
        )
        print("\n--- Tukey's HSD Post-Hoc Test Results (Median) ---")
        print(tukey_result_median)

        # Save Tukey's HSD results
        with open(
            os.path.join(reader.stats_dir, "tukey_hsd_results_median.txt"), "w"
        ) as f:
            f.write(str(tukey_result_median))

        # Convert Tukey's HSD results to a DataFrame for heatmap
        tukey_df_median = pd.DataFrame(
            data=tukey_result_median._results_table.data[1:],
            columns=tukey_result_median._results_table.data[0],
        )
        tukey_p_value_matrix_median = pd.pivot_table(
            tukey_df_median,
            values="p-adj",
            index="group1",
            columns="group2",
            fill_value=1.0,
        )
        # Fill diagonal with 1.0 (comparison with itself is not significant)
        for col in tukey_p_value_matrix_median.columns:
            if col in tukey_p_value_matrix_median.index:
                tukey_p_value_matrix_median.loc[col, col] = 1.0

        # Ensure symmetry
        for i in range(len(tukey_p_value_matrix_median.index)):
            for j in range(i + 1, len(tukey_p_value_matrix_median.columns)):
                val = tukey_p_value_matrix_median.iloc[i, j]
                tukey_p_value_matrix_median.iloc[j, i] = val

        tukey_p_value_matrix_median.to_csv(
            os.path.join(reader.stats_dir, "tukey_hsd_p_values_median.csv")
        )

        # Customize labels for the heatmap
        tukey_p_value_matrix_median.columns = [
            format_checkpoint_name(col) for col in tukey_p_value_matrix_median.columns
        ]
        tukey_p_value_matrix_median.index = [
            format_checkpoint_name(idx) for idx in tukey_p_value_matrix_median.index
        ]

        # Create the heatmap for Tukey's HSD
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            tukey_p_value_matrix_median,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            linecolor="black",
            vmin=0,
            vmax=1,
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0, va="center")
        plt.title("Tukey's HSD Post-Hoc Test P-Values (Median)")
        plt.tight_layout()
        plt.savefig(os.path.join(reader.stats_dir, "tukey_hsd_heatmap_median.pdf"))
        plt.show()
        plt.close()

    else:
        print(
            "\nThe p-value is not less than 0.05. We cannot reject the null hypothesis."
        )
        print(
            "Conclusion: There is not enough evidence to say that there are significant differences in performance between the checkpoints."
        )

    # %% Friedman - Conover analysis of top N min checkpoints

    min_sorted_summary = reader.get_summary_df(
        sort_by="min_discounted_reward_sum", descending=True
    )

    top_N_min_checkpoints = min_sorted_summary["checkpoint_name"][:TOP_N]

    top_N_min_data = [
        reader.eval_csv_path_df.filter(pl.col("checkpoint_name") == checkpoint_name)[
            "discounted_reward_sum"
        ]
        for checkpoint_name in top_N_min_checkpoints
    ]
    statistic, p_value = friedmanchisquare(*top_N_min_data)

    print(f"Friedman Test Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("\nThe p-value is less than 0.05. We can reject the null hypothesis.")
        print(
            "Conclusion: There is a statistically significant difference in performance between the checkpoints."
        )
    else:
        print(
            "\nThe p-value is not less than 0.05. We cannot reject the null hypothesis."
        )
        print(
            "Conclusion: There is not enough evidence to say that there are significant differences in performance between the checkpoints."
        )

    labels = top_N_min_checkpoints.to_list()

    data_for_test = pd.DataFrame(np.array(top_N_min_data).transpose(), columns=labels)

    conover_p_value_matrix = sp.posthoc_conover_friedman(data_for_test)

    # --- Save statistical test results ---
    friedman_results_text = f"""Friedman Test Statistic: {statistic:.4f}
    P-value: {p_value:.4f}
    """
    with open(os.path.join(reader.stats_dir, "friedman_results_min.txt"), "w") as f:
        f.write(friedman_results_text)

    conover_p_value_matrix.to_csv(
        os.path.join(reader.stats_dir, "conover_p_values_min.csv")
    )

    conover_p_value_matrix.columns = [
        format_checkpoint_name(col) for col in conover_p_value_matrix.columns
    ]
    conover_p_value_matrix.index = [
        format_checkpoint_name(idx) for idx in conover_p_value_matrix.index
    ]

    # --- Create the heatmap ---
    plt.figure(figsize=(8, 7))
    heatmap = sns.heatmap(
        conover_p_value_matrix,
        annot=True,  # Write the p-value in each cell
        fmt=".3f",  # Format p-values to 3 decimal places
        cmap="viridis_r",  # Use a color-blind friendly colormap
        linewidths=0.5,  # Draw lines between cells
        linecolor="black",
        vmin=0,  # Set the minimum of the colorbar to 0
        vmax=1,  # Set the maximum of the colorbar to 1
    )

    # --- Final plot adjustments ---
    plt.xticks(rotation=45, ha="right")  # Rotate xticks for better readability
    plt.yticks(rotation=0, va="center")  # Rotate yticks
    plt.title("Conover-Friedman Post-Hoc Test P-Values")
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(os.path.join(reader.stats_dir, "conover_heatmap_min.pdf"))
    plt.show()

    # %% ANOVA - Tukey HSD analysis of top N min checkpoints

    # Prepare data for ANOVA
    # Concatenate all data into a single Series and create a group indicator
    all_data_min = pl.concat(top_N_min_data)
    groups_min = pl.Series(
        [labels[i] for i, data in enumerate(top_N_min_data) for _ in range(len(data))]
    )

    # Perform ANOVA
    f_statistic_min, p_value_anova_min = f_oneway(
        *[data.to_numpy() for data in top_N_min_data]
    )

    print(f"\n--- ANOVA for Top {TOP_N} Min Checkpoints ---")
    print(f"ANOVA F-statistic: {f_statistic_min:.4f}")
    print(f"ANOVA P-value: {p_value_anova_min:.4f}")

    # Save ANOVA results
    anova_results_text_min = f"""ANOVA F-statistic: {f_statistic_min:.4f}
P-value: {p_value_anova_min:.4f}
"""
    with open(os.path.join(reader.stats_dir, "anova_results_min.txt"), "w") as f:
        f.write(anova_results_text_min)

    # Interpretation
    if p_value_anova_min < alpha:
        print("\nThe p-value is less than 0.05. We can reject the null hypothesis.")
        print(
            "Conclusion: There is a statistically significant difference in performance between the checkpoints."
        )
        # Perform Tukey's HSD post-hoc test
        tukey_result_min = pairwise_tukeyhsd(
            endog=all_data_min.to_numpy(), groups=groups_min.to_numpy(), alpha=alpha
        )
        print("\n--- Tukey's HSD Post-Hoc Test Results (Min) ---")
        print(tukey_result_min)

        # Save Tukey's HSD results
        with open(
            os.path.join(reader.stats_dir, "tukey_hsd_results_min.txt"), "w"
        ) as f:
            f.write(str(tukey_result_min))

        # Convert Tukey's HSD results to a DataFrame for heatmap
        tukey_df_min = pd.DataFrame(
            data=tukey_result_min._results_table.data[1:],
            columns=tukey_result_min._results_table.data[0],
        )
        tukey_p_value_matrix_min = pd.pivot_table(
            tukey_df_min,
            values="p-adj",
            index="group1",
            columns="group2",
            fill_value=1.0,
        )
        # Fill diagonal with 1.0 (comparison with itself is not significant)
        for col in tukey_p_value_matrix_min.columns:
            if col in tukey_p_value_matrix_min.index:
                tukey_p_value_matrix_min.loc[col, col] = 1.0

        # Ensure symmetry
        for i in range(len(tukey_p_value_matrix_min.index)):
            for j in range(i + 1, len(tukey_p_value_matrix_min.columns)):
                val = tukey_p_value_matrix_min.iloc[i, j]
                tukey_p_value_matrix_min.iloc[j, i] = val

        tukey_p_value_matrix_min.to_csv(
            os.path.join(reader.stats_dir, "tukey_hsd_p_values_min.csv")
        )

        # Customize labels for the heatmap
        tukey_p_value_matrix_min.columns = [
            format_checkpoint_name(col) for col in tukey_p_value_matrix_min.columns
        ]
        tukey_p_value_matrix_min.index = [
            format_checkpoint_name(idx) for idx in tukey_p_value_matrix_min.index
        ]

        # Create the heatmap for Tukey's HSD
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            tukey_p_value_matrix_min,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            linecolor="black",
            vmin=0,
            vmax=1,
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0, va="center")
        plt.title("Tukey's HSD Post-Hoc Test P-Values (Min)")
        plt.tight_layout()
        plt.savefig(os.path.join(reader.stats_dir, "tukey_hsd_heatmap_min.pdf"))
        plt.show()
        plt.close()

    else:
        print(
            "\nThe p-value is not less than 0.05. We cannot reject the null hypothesis."
        )
        print(
            "Conclusion: There is not enough evidence to say that there are significant differences in performance between the checkpoints."
        )
