# %%
import os

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from rich.progress import track
from tqdm import tqdm


# %%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


# %%
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


# %%
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
class PPOResultReader:
    def __init__(self, experiment_path):
        self.exp_path = experiment_path

        self.csv_path = os.path.join(self.exp_path, "csvs")
        self.csv_paths = os.listdir(self.csv_path)
        self.csv_paths.sort()
        self.csv_paths = list(
            map(lambda path: os.path.join(self.csv_path, path), self.csv_paths)
        )
        self.csv_dfs_lazy = [pl.scan_csv(csv_path) for csv_path in self.csv_paths]

        ppo_sbro_dir = os.path.join(self.exp_path, "PPO_SBRO")
        candidates = [
            d for d in os.listdir(ppo_sbro_dir) if d.startswith("PPO_sbro_env")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No directory starting with 'PPO_sbro_env' found in {ppo_sbro_dir}"
            )
        candidates.sort()
        self.progress_path = os.path.join(ppo_sbro_dir, candidates[-1])
        self.progress_csv_path = os.path.join(self.progress_path, "progress.csv")
        self.progress_df = pl.read_csv(self.progress_csv_path)

        self.figure_dir = os.path.join("figures", os.path.basename(self.exp_path))
        try:
            os.makedirs(self.figure_dir)
        except FileExistsError:
            print(f'The directory "{self.figure_dir}" already exists!')

        self.simple_reward_sum = [
            csv_df_lazy.select("Reward").sum().collect().item()
            for csv_df_lazy in track(
                self.csv_dfs_lazy, description="Reading reward sum..."
            )
        ]

        self.obs_range_dict = {
            "T_feed": [0.0, 50.0],  # °C
            "C_feed": [0.0, 0.25],  # kg/m3
            "C_pipe_c_out": [0.0, 10.0],  # kg/m3
            "P_m_in": [0.0, 25e5],  # Pa
            "P_m_out": [0.0, 25e5],  # Pa
            "Q_circ": [0.0, 10.0],  # m3/hr
            "Q_disp": [0.0, 10.0],  # m3/hr
            "Q_perm": [0.0, 10.0],  # m3/hr
            "C_perm": [0.0, 0.25],  # kg/m3
            "time_remaining": [-86400.0, 86400.0],  # seconds
            "V_perm_remaining": [-16.0, 32.0],  # m3
        }

        self.action_range_dict = {"Q0": [4.0, 6.0], "R_sp": [0.1, 0.3], "mode": None}

        print(f"🚀 PPOResultReader is initiated for {self.exp_path}!")

    def update(self):
        print("Reading incremental file paths ... ", end="")
        new_csv_paths = os.listdir(self.csv_path)
        new_csv_paths.sort()
        new_csv_paths = list(
            map(lambda path: os.path.join(self.csv_path, path), new_csv_paths)
        )
        incremental_files = list(set(new_csv_paths) - set(self.csv_paths))
        incremental_files.sort()
        self.csv_paths += incremental_files.copy()
        print("Done!")

        new_csv_dfs_lazy = [
            pl.scan_csv(new_csv_path)
            for new_csv_path in track(
                incremental_files,
                description=f"Loading incremental files (Total: {len(incremental_files)}) ... ",
            )
        ]
        self.csv_dfs_lazy += new_csv_dfs_lazy.copy()

        print(len(new_csv_dfs_lazy))
        self.simple_reward_sum += [
            csv_df_lazy.select("Reward").sum().collect().item()
            for csv_df_lazy in track(
                new_csv_dfs_lazy,
                description=f"Reading incremental reward sum (Total: {len(incremental_files)}) ... ",
            )
        ]

        self.progress_df = pl.read_csv(self.progress_csv_path)

        return None

    def plot_reward_summary(self, update=False, show=True, episode_max=-1):
        if update:
            self.update()

        ma_window = 24 * 4
        reward_sum = np.array(self.simple_reward_sum)
        ma_reward_sum = moving_average(reward_sum, ma_window)

        plt.figure(figsize=(15, 10))
        plt.plot(reward_sum[:episode_max], lw=0.5, alpha=0.5, label="Reward")
        plt.plot(
            ma_reward_sum[:episode_max],
            ls="--",
            lw=2.0,
            color="red",
            label=f"{ma_window} episode moving average",
        )
        # plt.ylim([-80, 10])
        # plt.xlim([0, 24000])
        plt.xlabel("Episode")
        plt.ylabel("Reward sum")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "reward_sum.pdf"))

        if show:
            plt.show()

        return None

    def plot_episode_len_dist(self, update=False, show=True):
        if update:
            self.update()

        plt.hist(
            [
                csv_df_lazy.select(pl.len(csv_df_lazy)).collect().item()
                for csv_df_lazy in self.csv_dfs_lazy
            ],
            bins=100,
            alpha=0.5,
        )
        plt.title("Episode length distirbution")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "Episode length.pdf"))

        if show:
            plt.show()

        return None

    def plot_episode(self, update=False, show=True, episode_n=-1):
        if update:
            self.update()

        if episode_n < 0:
            episode_n = len(self.csv_paths) + episode_n

        episode_save_dir = os.path.join(self.figure_dir, str(episode_n))
        try:
            os.makedirs(episode_save_dir)
        except FileExistsError:
            print(f'The directory "{episode_save_dir}" already exists!')

        target_episode_df = self.csv_dfs_lazy[episode_n].collect()

        denorm_target_df = self._denormalize_dataframe(
            df=target_episode_df, denormalize=True
        )

        obs_names = list(self.obs_range_dict.keys())
        action_names = list(self.action_range_dict.keys())

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f"Episode [{episode_n}] - Observations")

        for i in range(11):
            ax = axes[i // 4, i % 4]
            ax.plot(denorm_target_df[f"Previous {obs_names[i]}"], lw=0.5)
            ax.set_title(f"{obs_names[i]}")
            # ax.set_ylim([0, 1])
            ax.tick_params(axis="x", rotation=45)
            # Add grid lines for better visualization (optional)
            ax.grid(True, linestyle="--", alpha=0.7)

        # Remove the last subplot if there are an odd number of plots
        if i == 10:
            fig.delaxes(axes[2, 3])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(episode_save_dir, "Previous_obs_all.pdf"))

        if show:
            plt.show()
        else:
            plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Episode [{episode_n}] - Actions", fontsize=16)

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
        fig.suptitle(f"Episode [{episode_n}] - Rewards", fontsize=16)

        ax = axes[0]
        ax.plot(denorm_target_df["Reward"], lw=0.5)
        # ax.set_ylim([-0.5, 0.1])
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

        return None

    def plot_episode_post_analysis(self, update=False, show=True, episode_n=-1):
        if update:
            self.update()

        if episode_n < 0:
            episode_n = len(self.csv_paths) + episode_n

        episode_save_dir = os.path.join(self.figure_dir, str(episode_n))
        try:
            os.makedirs(episode_save_dir)
        except FileExistsError:
            print(f'The directory "{episode_save_dir}" already exists!')

        target_episode_df = self.csv_dfs_lazy[episode_n].collect()

        denorm_target_df = self._denormalize_dataframe(
            df=target_episode_df, denormalize=True
        )

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

        return None

    def plot_progress(self, update=False, plot_level=True, show=False):
        if update:
            self.update()

        progress_save_dir = os.path.join(self.figure_dir, "progress")
        try:
            os.makedirs(progress_save_dir)
        except FileExistsError:
            print(f'The directory "{progress_save_dir}" already exists!')

        trained_samples = self.progress_df[
            "learners/__all_modules__/num_env_steps_trained_lifetime"
        ]

        try:
            levels = self.progress_df["level"].unique()
            level_dfs = [
                self.progress_df.filter(pl.col("level") == level) for level in levels
            ]
            level_ranges = [
                [
                    level_df[
                        "learners/__all_modules__/num_env_steps_trained_lifetime"
                    ].min(),
                    level_df[
                        "learners/__all_modules__/num_env_steps_trained_lifetime"
                    ].max(),
                ]
                for level_df in level_dfs
            ]
        except pl.exceptions.ColumnNotFoundError as e:
            print("`level` column is not found: ", e)
            plot_level = False

        for col in tqdm(self.progress_df.columns):
            try:
                plt.figure()  # Create a new figure for each column
                plt.plot(trained_samples, self.progress_df[col], lw=0.5)
                if plot_level:
                    for level_range in level_ranges:
                        plt.vlines(
                            [level_range[1]],
                            ymin=self.progress_df[col].min(),
                            ymax=self.progress_df[col].max(),
                            color="k",
                            lw=0.5,
                        )
                plt.xlabel("Train sample")
                plt.title(col)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(progress_save_dir, f"{col.replace('/', '-')}.pdf")
                )
                if show:
                    plt.show()
                plt.close()  # Close the figure to free memory
            except TypeError as e:
                print(f"{col} is not plottable:\n{e}")
            except Exception as e:
                print(f"{col} is not plottable:\n{e}")

        return None

    def plot_objective_distribution(self, update=False, show=True):
        if update:
            self.update()

        lazy_expressions = [
            csv_df_lazy.select(
                pl.first("Previous_obs_9").alias("Previous time_remaining"),
                pl.first("Previous_obs_10").alias("Previous V_perm_remaining"),
            )
            for csv_df_lazy in self.csv_dfs_lazy
        ]

        combined_lazy_df = pl.concat(lazy_expressions)

        time_range = self.obs_range_dict["time_remaining"]
        v_perm_range = self.obs_range_dict["V_perm_remaining"]

        denormalized_lazy_df = combined_lazy_df.select(
            (
                pl.col("Previous time_remaining") * (time_range[1] - time_range[0])
                + time_range[0]
            ),
            (
                pl.col("Previous V_perm_remaining")
                * (v_perm_range[1] - v_perm_range[0])
                + v_perm_range[0]
            ),
        )

        objective_conditions_df = denormalized_lazy_df.collect()
        objective_conditions = objective_conditions_df.to_numpy()

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            objective_conditions[:, 0],  # X-axis: Previous time_remaining
            objective_conditions[:, 1],  # Y-axis: Previous V_perm_remaining
            c=np.arange(len(objective_conditions)),  # Color based on episode order
            cmap="viridis",  # Colormap (you can choose others like 'plasma', 'magma', 'cividis')
            s=10,  # Marker size
            alpha=0.7,  # Transparency
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Episode Order")

        plt.xlabel("Previous Time Remaining (seconds)")
        plt.ylabel("Previous V_perm Remaining (m³)")
        plt.title("Objective Conditions by Episode Order")
        plt.grid(True, linestyle="--", alpha=0.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "objective_conditions_scatter.pdf"))
        plt.show()

        return None

    def get_denormalized_df(self, update=False, episode_n=-1) -> pl.DataFrame:
        if update:
            self.update()

        if episode_n < 0:
            episode_n = len(self.csv_paths) + episode_n

        target_episode_df = self.csv_dfs_lazy[episode_n].collect()

        denorm_target_df = self._denormalize_dataframe(
            df=target_episode_df, denormalize=True
        )

        return denorm_target_df

    def _denormalize_dataframe(
        self, df: pl.DataFrame, denormalize: bool = True
    ) -> pl.DataFrame:
        """
        Denormalizes specific columns of a Polars DataFrame and keeps all other columns.

        Args:
            df: The input Polars DataFrame with normalized columns.

        Returns:
            A new Polars DataFrame with specified columns denormalized and
            all other original columns retained.
        """
        feature_names = list(self.obs_range_dict.keys())
        denormalization_exprs = []

        # This list will keep track of the original normalized columns to be dropped later
        cols_to_drop = []

        for i in range(len(feature_names)):
            prev_col_name = f"Previous_obs_{i}"
            curr_col_name = f"Current_obs_{i}"

            # Check if the 'Previous_obs_{i}' column exists in the DataFrame
            if prev_col_name in df.columns:
                original_feature_name = feature_names[i]
                min_val, max_val = self.obs_range_dict[original_feature_name]
                if denormalize:
                    expr = (
                        pl.col(prev_col_name) * (max_val - min_val) + min_val
                    ).alias("Previous " + original_feature_name)
                else:
                    expr = pl.col(prev_col_name).alias(
                        "Previous " + original_feature_name
                    )

                denormalization_exprs.append(expr)
                cols_to_drop.append(prev_col_name)

            # Check if the 'Current_obs_{i}' column exists in the DataFrame
            if curr_col_name in df.columns:
                original_feature_name = feature_names[i]
                min_val, max_val = self.obs_range_dict[original_feature_name]
                if denormalize:
                    expr = (
                        pl.col(curr_col_name) * (max_val - min_val) + min_val
                    ).alias("Current " + original_feature_name)
                else:
                    expr = pl.col(curr_col_name).alias(
                        "Current " + original_feature_name
                    )

                denormalization_exprs.append(expr)
                cols_to_drop.append(curr_col_name)

        # Use .with_columns() to add the new denormalized columns,
        # then use .drop() to remove the old ones.
        if not denormalization_exprs:
            return df  # Return original df if no columns were processed

        denormalized_df = df.with_columns(denormalization_exprs).drop(cols_to_drop)

        return denormalized_df


# %%
reader = PPOResultReader(
    experiment_path="/home/ybang-eai/research/2025/SBRO/SBRORL/result/PPO/2025-07-09 09:05:18.189259"
)

# %%
reader.update()
reader.plot_reward_summary()

# %%
reader.plot_episode(episode_n=-1)
reader.plot_episode_post_analysis(episode_n=-1)

# %%
reader.plot_progress(show=False, plot_level=False)

# %%
reader.plot_episode_post_analysis(episode_n=-1)


# %%
reader.update()
for ep in range(1, 48):
    reader.plot_episode(update=False, show=False, episode_n=-1 * ep)
    reader.plot_episode_post_analysis(update=False, show=False, episode_n=-1 * ep)

# %%
reader.plot_objective_distribution()

# %%
reader_without_curriculum = PPOResultReader(
    experiment_path="result/PPO/2025-07-08 12:54:48.371418"
)

# %%
reader_without_curriculum.update()
reader_without_curriculum.plot_reward_summary()


# %%
reader_without_curriculum.plot_episode(episode_n=-1)
reader_without_curriculum.plot_episode_post_analysis(episode_n=-1)
# reader_without_curriculum.plot_progress(show=False)

# %%
for ep in range(1, 48):
    reader_without_curriculum.plot_episode(update=False, show=False, episode_n=-1 * ep)
    reader_without_curriculum.plot_episode_post_analysis(
        update=False, show=False, episode_n=-1 * ep
    )

# %%
reader_without_curriculum.plot_objective_distribution()

# %%
optimal_checkpoint_with_curriculum = 66
optimal_checkpoint_without_curriculum = 18

plt.figure(figsize=(10, 7.5))
plt.plot(
    reader_without_curriculum.progress_df[
        "learners/__all_modules__/num_env_steps_trained_lifetime"
    ],
    reader_without_curriculum.progress_df["env_runners/episode_return_mean"],
    label="Without curriculum",
    lw=1.0,
)
plt.scatter(
    reader_without_curriculum.progress_df[
        "learners/__all_modules__/num_env_steps_trained_lifetime"
    ][18 * 10 : (18 + 1) * 10],
    reader_without_curriculum.progress_df["env_runners/episode_return_mean"][
        18 * 10 : (18 + 1) * 10
    ],
    label="Optimal point",
    marker="+",
)
plt.fill_between(
    x=reader_without_curriculum.progress_df[
        "learners/__all_modules__/num_env_steps_trained_lifetime"
    ],
    y1=reader_without_curriculum.progress_df["env_runners/episode_return_min"],
    y2=reader_without_curriculum.progress_df["env_runners/episode_return_max"],
    color="C0",
    alpha=0.15,
)

plt.plot(
    reader.progress_df["learners/__all_modules__/num_env_steps_trained_lifetime"],
    reader.progress_df["env_runners/episode_return_mean"],
    label="With curriculum",
    lw=1.0,
)
plt.scatter(
    reader.progress_df["learners/__all_modules__/num_env_steps_trained_lifetime"][
        66 * 10 : (66 + 1) * 10
    ],
    reader.progress_df["env_runners/episode_return_mean"][66 * 10 : (66 + 1) * 10],
    label="Optimal point",
    marker="+",
)
plt.fill_between(
    x=reader.progress_df["learners/__all_modules__/num_env_steps_trained_lifetime"],
    y1=reader.progress_df["env_runners/episode_return_min"],
    y2=reader.progress_df["env_runners/episode_return_max"],
    color="C1",
    alpha=0.15,
)
plt.xlabel("Train sample")
# plt.xlim([0, 6e10])
plt.ylim([-300, 0])
plt.legend()
plt.title("Reward sum (Return) trend comparison")
plt.tight_layout()
plt.savefig("Reward sum comparison.pdf")
plt.show()
# %%
plt.figure(figsize=(5, 7.5))
plt.plot(
    reader_without_curriculum.progress_df[
        "learners/__all_modules__/num_env_steps_trained_lifetime"
    ],
    reader_without_curriculum.progress_df["env_runners/episode_return_mean"],
    label="Without curriculum",
    lw=1.0,
)

plt.fill_between(
    x=reader_without_curriculum.progress_df[
        "learners/__all_modules__/num_env_steps_trained_lifetime"
    ],
    y1=reader_without_curriculum.progress_df["env_runners/episode_return_min"],
    y2=reader_without_curriculum.progress_df["env_runners/episode_return_max"],
    color="C0",
    alpha=0.15,
)

plt.plot(
    reader.progress_df["learners/__all_modules__/num_env_steps_trained_lifetime"],
    reader.progress_df["env_runners/episode_return_mean"],
    label="With curriculum",
    lw=1.0,
)

plt.fill_between(
    x=reader.progress_df["learners/__all_modules__/num_env_steps_trained_lifetime"],
    y1=reader.progress_df["env_runners/episode_return_min"],
    y2=reader.progress_df["env_runners/episode_return_max"],
    color="C1",
    alpha=0.15,
)

plt.xlim([5e10, 6e10])
plt.ylim([-60, None])
plt.tight_layout()
plt.show()
# %%
lazy_expressions = [
    csv_df_lazy.select(
        pl.last("Previous_obs_9").alias("Previous time_remaining"),
    )
    for csv_df_lazy in reader.csv_dfs_lazy
]

combined_lazy_df = pl.concat(lazy_expressions)

time_range = reader.obs_range_dict["time_remaining"]

denormalized_lazy_df = combined_lazy_df.select(
    (
        pl.col("Previous time_remaining") * (time_range[1] - time_range[0])
        + time_range[0]
    ),
)

time_remaining_df_with_curr = denormalized_lazy_df.collect()
time_remaining_with_curr = time_remaining_df_with_curr.to_numpy().squeeze()

# %%
lazy_expressions = [
    csv_df_lazy.select(
        pl.last("Previous_obs_9").alias("Previous time_remaining"),
    )
    for csv_df_lazy in reader_without_curriculum.csv_dfs_lazy
]

combined_lazy_df = pl.concat(lazy_expressions)

time_range = reader_without_curriculum.obs_range_dict["time_remaining"]

denormalized_lazy_df = combined_lazy_df.select(
    (
        pl.col("Previous time_remaining") * (time_range[1] - time_range[0])
        + time_range[0]
    ),
)

time_remaining_df_without_curr = denormalized_lazy_df.collect()
time_remaining_without_curr = time_remaining_df_without_curr.to_numpy().squeeze()

# %%
fig, axs = plt.subplots(2, 1, figsize=(7.5, 10), sharex=True, sharey=True)

fig.suptitle("Remaining time after termination")

axs[0].plot(time_remaining_with_curr, lw=0.1, label="With curriculum", alpha=0.5)
axs[0].plot(
    moving_average(time_remaining_with_curr, w=24 * 4),
    lw=2.0,
    label="Moving average",
    ls="--",
    color="red",
)
axs[0].grid(True, linestyle="--", alpha=0.7)
axs[0].set_ylabel("Time remaining [s]")
axs[0].legend()

axs[1].plot(time_remaining_without_curr, lw=0.1, label="Without curriculum", alpha=0.5)
axs[1].plot(
    moving_average(time_remaining_without_curr, w=24 * 4),
    lw=2.0,
    label="Moving average",
    ls="--",
    color="red",
)
axs[1].grid(True, linestyle="--", alpha=0.7)
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Time remaining [s]")
axs[1].legend()

plt.tight_layout()
plt.show()
# %%
tau_range = np.linspace(1.5, 18, 7)
Q_obj_range = np.linspace(0.8, 1.6, 16)
Q_obj_curriculum = [
    np.array([Q_obj_range[curr], Q_obj_range[-curr]]) for curr in range(1, 7)
]
Q_obj_curriculum.reverse()

curr_samples = []

for curr in range(6):
    samples = []
    for iter in range(1000):
        for i in range(6):
            tau_i = np.random.uniform(tau_range[i], tau_range[i + 1])
            Q_obj_curr = np.random.uniform(
                Q_obj_curriculum[curr][0], Q_obj_curriculum[curr][1]
            )
            V_obj = tau_i * Q_obj_curr
            samples.append(np.array([tau_i, V_obj]))
    curr_samples.append(np.array(samples))

plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(curr_samples)))

for curr, samples_arr in enumerate(curr_samples):
    plt.scatter(
        samples_arr[:, 0],
        samples_arr[:, 1],
        alpha=1.0,
        s=0.5,
        color=colors[curr],
        label=f"Curriculum # {curr}",
    )
plt.xlabel("Time limit [hr]")
plt.ylabel("Objective Volume [m³]")
plt.xlim([0, None])
plt.ylim([0, None])
plt.grid(True, linestyle=":")
plt.title("Scatterplot of mission objective scenario samples")
plt.legend()
plt.tight_layout()
plt.savefig("objective_scenario_samples.pdf")
plt.show()
# %%

# Pre-process all episodes to get cycle lengths, one by one to save memory

all_episode_cycle_lengths = []
total_episodes = len(reader.csv_dfs_lazy)

for i, lazy_df in enumerate(tqdm(reader.csv_dfs_lazy, desc="Processing episodes")):
    # Denormalize and collect one lazy_df at a time
    eager_df = reader._denormalize_dataframe(lazy_df, denormalize=True).collect()

    # Get cycles from the eager_df
    episode_split_dfs = get_cycles_from_df(eager_df)
    mean_C_feed = eager_df["Previous C_feed"].mean()

    # Extend the main list with cycle lengths
    all_episode_cycle_lengths.append(
        [
            [
                [
                    len(cycle),
                    cycle["Previous Q_perm"].mean(),
                    cycle["Previous T_feed"].mean(),
                    cycle["Previous C_perm"].mean(),
                ]
                for cycle in episode_split_dfs
            ],
            mean_C_feed,
        ]
    )

    # Explicitly delete to free memory
    del eager_df
    del episode_split_dfs

# %%
cycle_info_samples = []

for ep, data in enumerate(all_episode_cycle_lengths):
    for cycle_info in data[0]:
        cycle_info_samples.append(
            {
                "Episode": ep,
                "Cycle_length": cycle_info[0],
                "Mean_permeate_flowrate": cycle_info[1],
                "Mean_feed_temperature": cycle_info[2],
                "Mean_permeate_concentration": cycle_info[3],
                "Mean_feed_concentration": data[1],
            }
        )

cycle_length_samples = pl.DataFrame(cycle_info_samples)

# %%
plt.hist(cycle_length_samples["Mean feed concentration"], bins=100)
plt.show()

plt.hist(cycle_length_samples["Episode"], bins=100)
plt.show()

plt.hist(cycle_length_samples["Cycle length"], bins=100)
plt.show()

# %%
samples_after_convergence = cycle_length_samples.filter(pl.col("Episode") > 8000)

# %%
plt.scatter(
    samples_after_convergence["Mean_feed_concentration"],
    samples_after_convergence["Cycle_length"],
    s=1,
    marker="o",
    alpha=0.1,
)

plt.hist2d(
    samples_after_convergence["Mean_feed_concentration"],
    samples_after_convergence["Cycle_length"],
    bins=(100, 100),
    alpha=0.5,
    cmin=10,
    cmap="viridis",
)

plt.ylabel("Cycle length")
plt.xlabel("Mean feed concentration")
# plt.ylim([None, 100])
plt.show()

# %% Poisson regression analysis
import statsmodels.formula.api as smf  # noqa: E402

# %%
# --- 1. Fit the Negative Binomial Regression Model ---
# The formula remains the same, but we use negativebinomial() instead of poisson()
formula = "Cycle_length ~ Mean_feed_concentration"

# Convert to pandas for statsmodels compatibility
data_pd = samples_after_convergence.to_pandas()

# Fit the model
model_nb = smf.negativebinomial(formula=formula, data=data_pd).fit()

# Print the new model summary
print("--- Negative Binomial Regression Results ---")
print(model_nb.summary())


# --- 2. Get Predictions and Add to Polars DataFrame ---
# Get predictions from the new model
predictions_pd_nb = model_nb.predict(data_pd)

# Add the NBR predictions as a new column to your Polars DataFrame
samples_with_nb_predictions = samples_after_convergence.with_columns(
    pl.Series(name="predicted_cycle_length_nb", values=predictions_pd_nb)
)
# %%
# --- 3. Visualize the NBR Results ---
fig, ax = plt.subplots(figsize=(7, 6))

# Draw the 2D histogram using Matplotlib's hist2d
# It returns (counts, xedges, yedges, image_mappable)
h = ax.hist2d(
    samples_with_nb_predictions["Mean_feed_concentration"],
    samples_with_nb_predictions["Cycle_length"],
    bins=(np.arange(0.05, 0.1, 0.001), np.arange(0, 300, 1)),
    alpha=1.0,
    cmin=8,
    cmap="viridis",
)

# Add the colorbar, linked to the returned image object (h[3])
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label("Counts")

# Overlay the line plot for the predicted counts
ax.plot(
    samples_with_nb_predictions["Mean_feed_concentration"],
    samples_with_nb_predictions["predicted_cycle_length_nb"],
    color="red",
    lw=3,
    label="Predicted count",
)

# --- 2. Finalize the plot details ---
ax.set_ylim([None, 100])
ax.set_xlabel("Mean Feed Concentration [kg/m³]")
ax.set_ylabel("Cycle Length [minute]")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

plt.savefig("C_feed_Cycle_length_NBR.pdf")
plt.show()

# %%

# Pre-process all episodes to get cycle lengths, one by one to save memory

all_episode_cycle_lengths_without_curriculum = []
total_episodes = len(reader_without_curriculum.csv_dfs_lazy)

for i, lazy_df in enumerate(
    tqdm(reader_without_curriculum.csv_dfs_lazy, desc="Processing episodes")
):
    # Denormalize and collect one lazy_df at a time
    eager_df = reader_without_curriculum._denormalize_dataframe(
        lazy_df, denormalize=True
    ).collect()

    # Get cycles from the eager_df
    episode_split_dfs = get_cycles_from_df(eager_df)
    mean_C_feed = eager_df["Previous C_feed"].mean()

    # Extend the main list with cycle lengths
    all_episode_cycle_lengths_without_curriculum.append(
        [
            [
                [
                    len(cycle),
                    cycle["Previous Q_perm"].mean(),
                    cycle["Previous T_feed"].mean(),
                    cycle["Previous C_perm"].mean(),
                ]
                for cycle in episode_split_dfs
            ],
            mean_C_feed,
        ]
    )

    # Explicitly delete to free memory
    del eager_df
    del episode_split_dfs

# %%
cycle_info_samples_without_curriculum = []

for ep, data in enumerate(all_episode_cycle_lengths_without_curriculum):
    for cycle_info in data[0]:
        cycle_info_samples_without_curriculum.append(
            {
                "Episode": ep,
                "Cycle_length": cycle_info[0],
                "Mean_permeate_flowrate": cycle_info[1],
                "Mean_feed_temperature": cycle_info[2],
                "Mean_permeate_concentration": cycle_info[3],
                "Mean_feed_concentration": data[1],
            }
        )

cycle_length_samples_without_curriculum = pl.DataFrame(
    cycle_info_samples_without_curriculum
)

# %%
plt.hist(cycle_length_samples_without_curriculum["Mean_feed_concentration"], bins=100)
plt.show()

plt.hist(cycle_length_samples_without_curriculum["Episode"], bins=100)
plt.show()

plt.hist(cycle_length_samples_without_curriculum["Cycle_length"], bins=100)
plt.show()

# %%
samples_after_convergence_without_curriculum = (
    cycle_length_samples_without_curriculum.filter(pl.col("Episode") > 12000)
)

# %%
plt.scatter(
    samples_after_convergence_without_curriculum["Mean_feed_concentration"],
    samples_after_convergence_without_curriculum["Cycle_length"],
    s=1,
    marker="o",
    alpha=0.1,
)

plt.hist2d(
    samples_after_convergence_without_curriculum["Mean_feed_concentration"],
    samples_after_convergence_without_curriculum["Cycle_length"],
    bins=(100, 100),
    alpha=0.5,
    cmin=10,
    cmap="viridis",
)

plt.ylabel("Cycle length")
plt.xlabel("Mean feed concentration")
# plt.ylim([None, 100])
plt.show()
# %%
# --- 1. Fit the Negative Binomial Regression Model ---
# The formula remains the same, but we use negativebinomial() instead of poisson()
formula = "Cycle_length ~ Mean_feed_concentration"

# Convert to pandas for statsmodels compatibility
data_pd = samples_after_convergence_without_curriculum.to_pandas()

# Fit the model
model_nb = smf.negativebinomial(formula=formula, data=data_pd).fit()

# Print the new model summary
print("--- Negative Binomial Regression Results ---")
print(model_nb.summary())


# --- 2. Get Predictions and Add to Polars DataFrame ---
# Get predictions from the new model
predictions_pd_nb = model_nb.predict(data_pd)

# Add the NBR predictions as a new column to your Polars DataFrame
samples_with_nb_predictions_without_curriculum = (
    samples_after_convergence_without_curriculum.with_columns(
        pl.Series(name="predicted_cycle_length_nb", values=predictions_pd_nb)
    )
)

# --- 3. Visualize the NBR Results ---
fig, ax = plt.subplots(figsize=(7, 6))

# Draw the 2D histogram using Matplotlib's hist2d
# It returns (counts, xedges, yedges, image_mappable)
h = ax.hist2d(
    samples_with_nb_predictions_without_curriculum["Mean_feed_concentration"],
    samples_with_nb_predictions_without_curriculum["Cycle_length"],
    bins=(np.arange(0.05, 0.1, 0.001), np.arange(0, 300, 2)),
    alpha=1.0,
    cmin=8,
    cmap="viridis",
)

# Add the colorbar, linked to the returned image object (h[3])
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label("Counts")

# Overlay the line plot for the predicted counts
ax.plot(
    samples_with_nb_predictions_without_curriculum["Mean_feed_concentration"],
    samples_with_nb_predictions_without_curriculum["predicted_cycle_length_nb"],
    color="red",
    lw=3,
    label="Predicted count",
)

# --- 2. Finalize the plot details ---
ax.set_ylim([None, 150])
ax.set_xlabel("Mean Feed Concentration [kg/m³]")
ax.set_ylabel("Cycle Length [minute]")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

plt.savefig("C_feed_Cycle_length_NBR.pdf")
plt.show()

# %%
from copy import deepcopy

print(len(cycle_length_samples_without_curriculum["Episode"].unique()))

ep_range = np.arange(5000, 17000, 3000, dtype=np.int64)

for i in range(3):
    episodes = (ep_range[i], ep_range[i + 1])
    print(episodes)
    target_samples = deepcopy(
        cycle_length_samples_without_curriculum[episodes[0] : episodes[1]]
    )

    formula = "Cycle_length ~ Mean_feed_concentration"

    # Convert to pandas for statsmodels compatibility
    data_pd = target_samples.to_pandas()

    # Fit the model
    model_nb = smf.negativebinomial(formula=formula, data=data_pd).fit()

    # Print the new model summary
    print("--- Negative Binomial Regression Results ---")
    print(model_nb.summary())

    # --- 2. Get Predictions and Add to Polars DataFrame ---
    # Get predictions from the new model
    predictions_pd_nb = model_nb.predict(data_pd)

    # Add the NBR predictions as a new column to your Polars DataFrame
    samples_with_nb_predictions_without_curriculum = target_samples.with_columns(
        pl.Series(name="predicted_cycle_length_nb", values=predictions_pd_nb)
    )

    # --- 3. Visualize the NBR Results ---
    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw the 2D histogram using Matplotlib's hist2d
    # It returns (counts, xedges, yedges, image_mappable)
    h = ax.hist2d(
        samples_with_nb_predictions_without_curriculum["Mean_feed_concentration"],
        samples_with_nb_predictions_without_curriculum["Cycle_length"],
        bins=(np.arange(0.05, 0.1, 0.001), np.arange(0, 300, 2)),
        alpha=1.0,
        cmin=5,
        cmap="viridis",
    )

    # Add the colorbar, linked to the returned image object (h[3])
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label("Counts")

    # Overlay the line plot for the predicted counts
    ax.plot(
        samples_with_nb_predictions_without_curriculum["Mean_feed_concentration"],
        samples_with_nb_predictions_without_curriculum["predicted_cycle_length_nb"],
        color="red",
        lw=3,
        label="Predicted count",
    )

    # --- 2. Finalize the plot details ---
    # ax.set_ylim([None, 150])
    ax.set_xlabel("Mean Feed Concentration [kg/m³]")
    ax.set_ylabel("Cycle Length [minute]")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("C_feed_Cycle_length_NBR.pdf")
    plt.show()

# %%
## Fig. 1:Tree기반 시각화
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.tree import DecisionTreeRegressor, plot_tree

# # 데이터 로드
# df = pd.read_csv(r"D:\@ZerogapRl_code\@@Figure에쓰인엑셀파일\td3_allSteps_tree-based-100.csv")

# # Feature 및 Target 정의
# features = ['Action1', 'Action2', 'Action3']
# target = 'Reward'
# X = df[features]
# y = df[target]

# # Tree 모델 학습
# tree = DecisionTreeRegressor(max_depth=4, random_state=42)
# tree.fit(X, y)

# # Figure 1: Decision Tree 시각화
# plt.figure(figsize=(20, 10))
# plot_tree(tree, feature_names=features, filled=True, rounded=True, precision=5, node_ids=True)
# plt.title("Figure 1. Tree-based Interpretation of TD3 Policy (Actions → Reward)")
# plt.savefig(r"D:\@ZerogapRl_code\@@Figure에쓰인엑셀파일\Fig1_tree_based-100.png", dpi=300, bbox_inches='tight')
# # plt.show()

# # 리프 노드 정보 추가
# df['Leaf'] = tree.apply(X)

# # State 값 이름 변경 (있을 경우)
# df = df.rename(columns={
#     'State2': 'prod_H2',
#     'State3': 'P_total',
#     'State4': 'V_cell'
# })
# # Leaf 별 성능지표 요약
# summary = df.groupby('Leaf').agg({
#     'prod_H2': 'mean',
#     'P_total': 'mean',
#     'V_cell': 'mean',
#     'Reward': ['mean', 'count']
# }).reset_index()
# summary.columns = ['Leaf', 'Mean_H2', 'Mean_Power', 'Mean_Vcell', 'Mean_Reward', 'Sample_Count']

# ##-----------------------------------------------------------------------------------------##
# ## Fig. 2: Tree기반 평균, 표준편차 표현
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.tree import DecisionTreeRegressor
# import pandas as pd
# import numpy as np
# #
# # # 데이터 로드
# # df = pd.read_csv(r"D:\@ZerogapRl_code\@@Figure에쓰인엑셀파일\td3_allSteps_tree-based-1000.csv")
# #
# # # 트리 학습
# # features = ['Action1', 'Action2', 'Action3']
# # target = 'Reward'
# # X = df[features]
# # y = df[target]
# #
# # tree = DecisionTreeRegressor(max_depth=4, random_state=42)
# # tree.fit(X, y)
# # # Leaf 할당
# # df['Leaf'] = tree.apply(X)

# # 시각화
# leaf_ids = sorted(df['Leaf'].unique())
# num_cols = 4
# num_rows = int(np.ceil(len(leaf_ids) / num_cols))

# fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 3.5 * num_rows))
# axes = axes.flatten()

# for idx, leaf in enumerate(leaf_ids):
#     ax = axes[idx]
#     subset = df[df['Leaf'] == leaf]['Reward']
#     sns.histplot(subset, kde=True, stat='density', ax=ax,
#                  bins=25, color="skyblue", edgecolor="black", linewidth=0.5)

#     mu = subset.mean()
#     sigma = subset.std()

#     ax.set_title(f"Leaf {leaf}\nμ={mu:.2f}, σ={sigma:.2f}", fontsize=10)
#     ax.set_xlabel("Reward")
#     ax.set_ylabel("Density")

# # 남는 subplot 제거
# for i in range(len(leaf_ids), len(axes)):
#     fig.delaxes(axes[i])

# plt.tight_layout()
# plt.savefig(r"D:\@ZerogapRl_code\@@Figure에쓰인엑셀파일\Fig2_tree-based-100.png", dpi=300)
