# %%
import polars as pl
import os
from matplotlib import pyplot as plt
import numpy as np

# %%
target_dir = (
    "/home/ybang-eai/research/2025/SBRO/SBRORL/result/2025-06-11 23:05:02.661691"
)

parquet_dir = os.path.join(target_dir, 'parquets')
# parquet_dir = target_dir

target_files = os.listdir(parquet_dir)

target_files.sort()

print(f"Total {len(target_files)} parquet files")
# %%
result_dfs = []

for target_file in target_files[:-1]:
    result_dfs.append(pl.scan_parquet(os.path.join(parquet_dir, target_file)).collect())


# %%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


simple_reward_sum = np.array([result_df[:, "Reward"].sum() for result_df in result_dfs])
ma_reward_sum = moving_average(simple_reward_sum, 100)

water_temps = [result_df[:, "Previous_obs_0"].mean() for result_df in result_dfs]

# %%
plt.plot(simple_reward_sum, lw=0.1)
plt.plot(ma_reward_sum)
plt.show()

# %%
episode_n = -1

for i in range(9, 11):
    plt.plot(result_dfs[episode_n][f"Previous_obs_{i}"])
    plt.title(f"Previous_obs_{i}")
    plt.show()

for j in range(2):
    plt.plot(np.tanh(result_dfs[episode_n][f"Act_{j}"]))
    plt.title(f"Act_{j}")
    plt.show()

plt.plot(result_dfs[episode_n]["Act_2"])
plt.title("Act_2")
plt.show()

plt.plot(result_dfs[episode_n]["Reward"])
plt.title("Reward")
plt.show()

plt.plot(np.cumsum(result_dfs[episode_n]["Reward"]))
plt.title("Cumulative Reward")
plt.show()
# %%
Q_disps = [result_df[:, "Previous_obs_6"] for result_df in result_dfs[-10:]]
for Q_disp in Q_disps:
    plt.plot(Q_disp)
plt.show()
# %%
process_df = pl.read_csv(
    "/home/ybang-eai/research/2025/SBRO/SBRORL/result/2025-06-11 23:05:02.661691/PPO_SBRO_Custom_Configs_run_FIXED/PPO_sbro_env_v1_11a73_00000_0_2025-06-11_23-05-02/progress.csv"
)
# %%
for col in process_df.columns:
    print(col)
# %%
plt.plot(process_df["learners/default_policy/total_loss"])
# %%
plt.plot(process_df["learners/default_policy/policy_loss"])

# %%
