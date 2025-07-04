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

    def plot_progress(self, update=False, show=False):
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

        for col in self.progress_df.columns:
            try:
                plt.figure()  # Create a new figure for each column
                plt.plot(trained_samples, self.progress_df[col], lw=0.5)
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
reader = PPOResultReader(experiment_path="result/PPO/2025-07-04 10:55:27.949935")

# %%
reader.update()
reader.plot_reward_summary()
reader.plot_episode(episode_n=-1)
reader.plot_episode_post_analysis(episode_n=-1)

# %%
reader.plot_progress(show=False)

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
    experiment_path="result/PPO/2025-07-03 18:54:53.379802"
)

# %%
reader_without_curriculum.update()
reader_without_curriculum.plot_episode(episode_n=-1)
reader_without_curriculum.plot_episode_post_analysis(episode_n=-1)


# %%
reader_without_curriculum.plot_reward_summary()
reader_without_curriculum.plot_progress(show=False)

# %%
for ep in range(1, 48):
    reader_without_curriculum.plot_episode(update=False, show=False, episode_n=-1 * ep)
    reader_without_curriculum.plot_episode_post_analysis(
        update=False, show=False, episode_n=-1 * ep
    )

# %%
reader_without_curriculum.plot_objective_distribution()

# %%
plt.figure(figsize=(10, 7.5))
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
plt.xlabel("Train sample")
plt.xlim([0, 6e10])
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
fig, axs = plt.subplots(2, 1, figsize=(7.5, 10), sharex=True)

fig.suptitle("Remaining time after termination")

axs[0].plot(time_remaining_with_curr, lw=0.1, label="With curriculum", alpha=0.5)
axs[0].plot(
    moving_average(time_remaining_with_curr, w=24 * 4),
    lw=2.0,
    label="Moving average",
    ls="--",
    color="red",
)
axs[0].grid(True, linestyle="-", alpha=0.7)
axs[0].legend()

axs[1].plot(time_remaining_without_curr, lw=0.1, label="Without curriculum", alpha=0.5)
axs[1].plot(
    moving_average(time_remaining_without_curr, w=24 * 4),
    lw=2.0,
    label="Moving average",
    ls="--",
    color="red",
)
axs[1].grid(True, linestyle="-", alpha=0.7)
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

    # Extend the main list with cycle lengths
    all_episode_cycle_lengths.extend([len(cycle) for cycle in episode_split_dfs])

    # Explicitly delete to free memory
    del eager_df
    del episode_split_dfs
# Dynamic plotting of cycle length distributions from pre-processed data


def plot_cycle_length_distribution_blocks(cycle_lengths_data, episode_block_size=2400):
    total_lengths = len(cycle_lengths_data)
    num_blocks = (total_lengths + episode_block_size - 1) // episode_block_size

    fig, axs = plt.subplots(1, num_blocks, figsize=(6 * num_blocks, 5), squeeze=False)
    axs = axs.flatten()

    current_idx = 0
    for i in range(num_blocks):
        start_idx = current_idx
        end_idx = min(current_idx + episode_block_size, total_lengths)

        block_cycle_lengths = cycle_lengths_data[start_idx:end_idx]

        if not block_cycle_lengths:
            continue

        # Plotting for the current block
        axs[i].hist(block_cycle_lengths, bins=100)
        axs[i].set_title(
            f"Cycle Length Distribution (Lengths {start_idx}-{end_idx - 1})"
        )
        axs[i].set_xlabel("Cycle Length")
        axs[i].set_ylabel("Frequency")

        current_idx = end_idx

    plt.tight_layout()
    plt.show()


# Example usage:
plot_cycle_length_distribution_blocks(
    all_episode_cycle_lengths, episode_block_size=2400
)

# You can now call plot_cycle_length_distribution_blocks with different block sizes or modify plotting design
# For example:
# plot_cycle_length_distribution_blocks(all_episode_cycle_lengths, episode_block_size=1000)
# plot_cycle_length_distribution_blocks(all_episode_cycle_lengths, episode_block_size=5000)

# %%
