# %%
import polars as pl
import os
from matplotlib import pyplot as plt
import numpy as np

# %%
target_dir = "/home/ybang-eai/research/2025/SBRO/SBRORL/result/2025-06-09 20:45:22.283794/parquets"
target_files = os.listdir(target_dir)
# %%
result_dfs = []
for target_file in target_files:
    result_dfs.append(pl.scan_parquet(os.path.join(target_dir, target_file)).collect())

# %%
simple_reward_sum = [result_df[:, "Reward"].sum() for result_df in result_dfs]

water_temps = [result_df[:, "Previous_obs_0"].mean() for result_df in result_dfs]
# %%
plt.scatter(water_temps, simple_reward_sum)
# %%
plt.plot(water_temps, lw=0.1)
# %%
water_temps = [result_df[:, "Previous_obs_0"] for result_df in result_dfs[-10:]]
plt.plot(water_temps[1])
# %%
print(water_temps)
# %%
V_objectives = [result_df[:, "Previous_obs_10"] for result_df in result_dfs[-10:]]
# %%
plt.plot(V_objectives[1])
# %%
Q_disps = [result_df[:, "Previous_obs_6"] for result_df in result_dfs[-10:]]
for Q_disp in Q_disps:
    plt.plot(Q_disp)
plt.show()
# %%
