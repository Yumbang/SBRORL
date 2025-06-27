# %%
import os

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from tqdm import tqdm

# %%


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


# %%
target_dir = "result/PPO/2025-06-27 14:28:27.591269"

data_dir = os.path.join(target_dir, "csvs")

SAVE_DIR = os.path.join("figures", os.path.basename(target_dir))

try:
    os.makedirs(SAVE_DIR)
except FileExistsError:
    print(f'The directory "{SAVE_DIR}" already exists!')


target_files = os.listdir(data_dir)

target_files = [os.path.join(data_dir, target_file) for target_file in target_files]

target_files.sort()

print(f"Total {len(target_files)} csv files")
result_dfs = []

for target_file in tqdm(target_files[:-1]):
    try:
        # result_dfs.append(pl.scan_csv(target_file).collect())
        result_dfs.append(pl.read_csv(target_file))
    except Exception as e:
        print(f"Failed to read {target_file}: {e}")


ma_window = 24 * 4

simple_reward_sum = np.array([result_df[:, "Reward"].sum() for result_df in result_dfs])
ma_reward_sum = moving_average(simple_reward_sum, ma_window)

episode_max = -1
plt.figure(figsize=(15, 10))
plt.plot(simple_reward_sum[:episode_max], lw=0.1, alpha=0.5, label="Reward")
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
plt.savefig(os.path.join(SAVE_DIR, "reward_sum.pdf"))
plt.show()

# %%
plt.hist2d(
    [len(df) for df in result_dfs],
    [df["Reward"].sum() for df in result_dfs],
    bins=(100, 100),
    cmap=plt.cm.jet,
    alpha=1.0,
)
# plt.ylim([-40, 0])
plt.show()

plt.scatter(
    [len(df) for df in result_dfs],
    [df["Reward"].sum() for df in result_dfs],
    alpha=0.1,
    s=10,
    marker="+",
)
plt.show()

plt.hist([len(df) for df in result_dfs], bins=100, alpha=0.5)
plt.title("Episode length distirbution")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Episode length.pdf"))
plt.show()

# %%
for i, df in enumerate(result_dfs):
    if df["Reward"].is_null().any():
        print(f"DataFrame {i} contains None values in 'Reward' column")
        result_dfs[i] = df.drop_nulls()
obs_range_dict = {
    "T_feed": [0.0, 50.0],  # °C
    "C_feed": [0.0, 1.0],  # kg/m3
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

action_range_dict = {"Q0": [4.0, 6.0], "R_sp": [0.1, 0.3], "mode": None}

obs_names = list(obs_range_dict.keys())
action_names = list(action_range_dict.keys())


def denormalize_dataframe(
    df: pl.DataFrame, obs_range: dict, denormalize: bool = True
) -> pl.DataFrame:
    """
    Denormalizes specific columns of a Polars DataFrame and keeps all other columns.

    Args:
        df: The input Polars DataFrame with normalized columns.
        obs_range: A dictionary where keys are original feature names and
                   values are [min, max] ranges.

    Returns:
        A new Polars DataFrame with specified columns denormalized and
        all other original columns retained.
    """
    feature_names = list(obs_range.keys())
    denormalization_exprs = []

    # This list will keep track of the original normalized columns to be dropped later
    cols_to_drop = []

    for i in range(len(feature_names)):
        prev_col_name = f"Previous_obs_{i}"
        curr_col_name = f"Current_obs_{i}"

        # Check if the 'Previous_obs_{i}' column exists in the DataFrame
        if prev_col_name in df.columns:
            original_feature_name = feature_names[i]
            min_val, max_val = obs_range[original_feature_name]
            if denormalize:
                expr = (pl.col(prev_col_name) * (max_val - min_val) + min_val).alias(
                    "Previous " + original_feature_name
                )
            else:
                expr = pl.col(prev_col_name).alias("Previous " + original_feature_name)

            denormalization_exprs.append(expr)
            cols_to_drop.append(prev_col_name)

        # Check if the 'Current_obs_{i}' column exists in the DataFrame
        if curr_col_name in df.columns:
            original_feature_name = feature_names[i]
            min_val, max_val = obs_range[original_feature_name]
            if denormalize:
                expr = (pl.col(curr_col_name) * (max_val - min_val) + min_val).alias(
                    "Current " + original_feature_name
                )
            else:
                expr = pl.col(curr_col_name).alias("Current " + original_feature_name)

            denormalization_exprs.append(expr)
            cols_to_drop.append(curr_col_name)

    # Use .with_columns() to add the new denormalized columns,
    # then use .drop() to remove the old ones.
    if not denormalization_exprs:
        return df  # Return original df if no columns were processed

    denormalized_df = df.with_columns(denormalization_exprs).drop(cols_to_drop)

    return denormalized_df


# %%
from copy import deepcopy

result_dfs_by_reward = deepcopy(result_dfs)

result_dfs_by_reward.sort(key=lambda df: sum(df["Reward"]), reverse=True)


# %%
episode_n = 0
#
denorm_target_df = denormalize_dataframe(
    result_dfs_by_reward[episode_n], obs_range_dict
)
# denorm_target_df = denormalize_dataframe(result_dfs_by_reward[episode_n], obs_range_dict, denormalize=False)

print(f"REWARD: {denorm_target_df['Reward'].sum()}")

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle(f"Episode [{episode_n}] - Observations")


for i in range(11):
    ax = axes[i // 4, i % 4]
    ax.plot(denorm_target_df[f"Previous {obs_names[i]}"], lw=0.5)
    ax.set_title(f"{obs_names[i]}")
    # ax.set_ylim([0, 1])
    ax.tick_params(axis="x", rotation=45)

# Remove the last subplot if there are an odd number of plots
if i == 10:
    fig.delaxes(axes[2, 3])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, "Previous_obs_all.pdf"))
plt.show()

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

ax = axes[1, 0]
ax.plot(denorm_target_df["Act_2"], lw=0.5)
ax.set_title(f"Act_2 ({action_names[2]})")
ax.set_ylim([-0.1, 1.1])
ax.tick_params(axis="x", rotation=45)

fig.delaxes(axes[1, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, "Actions_all.pdf"))
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f"Episode [{episode_n}] - Rewards", fontsize=16)

ax = axes[0]
ax.plot(denorm_target_df["Reward"], lw=0.5)
# ax.set_ylim([-0.5, 0.1])
ax.set_title("Reward")
ax.tick_params(axis="x", rotation=45)

ax = axes[1]
ax.plot(np.cumsum(denorm_target_df["Reward"]), lw=0.5)
ax.set_title("Cumulative Reward")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, "Reward_summary.pdf"))
plt.show()

# %%
episode_n = -12

denorm_target_df = denormalize_dataframe(result_dfs[episode_n], obs_range_dict)

print(f"REWARD: {denorm_target_df['Reward'].sum()}")

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle(f"Episode [{episode_n}] - Observations")


for i in range(11):
    ax = axes[i // 4, i % 4]
    ax.plot(denorm_target_df[f"Previous {obs_names[i]}"], lw=0.5)
    ax.set_title(f"{obs_names[i]}")
    # ax.set_ylim([0, 1])
    ax.tick_params(axis="x", rotation=45)

# Remove the last subplot if there are an odd number of plots
if i == 10:
    fig.delaxes(axes[2, 3])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, "Previous_obs_all.pdf"))
plt.show()

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

ax = axes[1, 0]
ax.plot(denorm_target_df["Act_2"], lw=0.5)
ax.set_title(f"Act_2 ({action_names[2]})")
ax.set_ylim([-0.1, 1.1])
ax.tick_params(axis="x", rotation=45)

fig.delaxes(axes[1, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, "Actions_all.pdf"))
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f"Episode [{episode_n}] - Rewards", fontsize=16)

ax = axes[0]
ax.plot(denorm_target_df["Reward"], lw=0.5)
# ax.set_ylim([-0.5, 0.1])
ax.set_title("Reward")
ax.tick_params(axis="x", rotation=45)

ax = axes[1]
ax.plot(np.cumsum(denorm_target_df["Reward"]), lw=0.5)
ax.set_title("Cumulative Reward")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, "Reward_summary.pdf"))
plt.show()

# %%
process_df = pl.read_csv(
    "result/PPO/2025-06-27 14:28:27.591269/PPO_SBRO/PPO_sbro_env_v1_8dc3f_00000_0_2025-06-27_14-28-27/progress.csv"
)

for col in process_df.columns:
    print(col)

# plt.plot(process_df["learners/default_policy/total_loss"])
# plt.title("learners/default_policy/total_loss")
# plt.show()

# plt.plot(process_df["learners/default_policy/policy_loss"])
# plt.title("learners/default_policy/policy_loss")
# plt.show()


# plt.plot(process_df["learners/default_policy/vf_loss"])
# plt.title("learners/default_policy/vf_loss")
# plt.show()

# plt.plot(process_df["env_runners/episode_return_mean"])
# plt.title("env_runners/episode_return_max")
# plt.show()

for col in process_df.columns:
    plt.plot(process_df[col], lw=0.5)
    plt.title(col)
    plt.tight_layout()
    plt.show()

# %%
from scipy import stats

mean_return = np.array(process_df["env_runners/episode_return_mean"])
window = 15

divergence = []
for i in range(len(mean_return) - 2 * window):
    past_10 = mean_return[i : i + window]
    recent_10 = mean_return[i + window : i + 2 * window]
    divergence.append(stats.entropy(past_10, recent_10))

p_val = []
for i in range(len(mean_return) - 2 * window):
    past_10 = mean_return[i : i + window]
    recent_10 = mean_return[i + window : i + 2 * window]
    p_val.append(stats.ttest_ind(past_10, recent_10, alternative="two-sided"))

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

for ax in axes:
    ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))

axes[0].plot(np.arange(len(mean_return)), mean_return, lw=0.5)
axes[0].set_xlim([0, len(mean_return)])
axes[0].set_title("Episode Return Mean")
axes[0].grid(True, which="major", linestyle="-", linewidth=0.7)
axes[0].grid(True, which="minor", linestyle=":", linewidth=0.5)

axes[1].plot(np.arange(2 * window, len(mean_return)), divergence, lw=0.5)
axes[1].set_xlim([0, len(mean_return)])
axes[1].set_ylim([0.0, 0.1])
axes[1].set_title(f"KL Divergence (Entropy) between Past and Recent {window}")
axes[1].grid(True, which="major", linestyle="-", linewidth=0.7)
axes[1].grid(True, which="minor", linestyle=":", linewidth=0.5)

axes[2].plot(
    np.arange(2 * window, len(mean_return)),
    [p[1] if hasattr(p, "__getitem__") else p.pvalue for p in p_val],
    lw=0.5,
)
axes[2].set_title(f"p-value (t-test) between Past and Recent {window}")
axes[2].set_xlim([0, len(mean_return)])
axes[2].set_ylim([0.0, 1.0])
axes[2].grid(True, which="major", linestyle="-", linewidth=0.7)
axes[2].grid(True, which="minor", linestyle=":", linewidth=0.5)

plt.tight_layout()
plt.show()

# %%
divergence

# %%
from pathlib import Path  # <-- Make sure you have this import

import orjson


def load_rllib_results(file_path):
    """
    Loads an RLlib result.json file into a pandas DataFrame.

    Each line in result.json is a separate JSON object. This function
    reads the file line by line, parses each line with orjson, and
    compiles them into a DataFrame.

    Args:
        file_path: The path to the result.json file.

    Returns:
        A pandas DataFrame containing the experiment results.
    """
    data = []
    print(f"Loading results from: {file_path}")

    # Ensure the file exists before proceeding
    if not file_path.is_file():
        raise FileNotFoundError(f"Result file not found at: {file_path}")

    # Open the file and read it line by line
    with open(file_path, "r") as f:
        for line in f:
            # Skip empty lines, which can sometimes occur at the end of the file
            if not line.strip():
                continue
            try:
                # orjson.loads() parses a single JSON object from a string/bytes
                data.append(orjson.loads(line))
            except orjson.JSONDecodeError:
                print(f"Warning: Skipping a malformed line: {line.strip()}")

    if not data:
        print(
            "Warning: No data was loaded. The file might be empty or contain only malformed lines."
        )
        return pl.DataFrame()

    print(f"Successfully loaded {len(data)} training iterations.")

    # Convert the list of dictionaries to a pandas DataFrame
    return pl.DataFrame(data)


# Your file path as a string
file_path_str = "result/PPO/2025-06-17 20:01:08.419844/PPO_SBRO_Custom_Configs_run_FIXED/PPO_sbro_env_v1_5f350_00000_0_2025-06-17_20-01-08/result.json"

# Convert the string to a Path object before calling the function
path_obj = Path(file_path_str)

# Now call the function with the Path object
result_json = load_rllib_results(path_obj)

result_json

# %%
