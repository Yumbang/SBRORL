# %%
import os

import numpy as np
import polars as pl
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule as PPOModule,
)
from tqdm import tqdm

from read_evaluation_files import EvalResultReader

# %%
checkpoints_path = "/home/ybang-eai/research/2025/SBRO/SBRORL/result/PPO/2025-07-08 12:54:48.371418/PPO_SBRO/PPO_sbro_env_v1_4afbe_00000_0_2025-07-08_12-54-48"
checkpoint_number = 58

# %%
device = "cuda:0"
target_checkpoint_path = os.path.join(
    checkpoints_path,
    f"checkpoint_0000{str(checkpoint_number).rjust(2, '0')}/learner_group/learner/rl_module/default_policy",
)

ppomodule = PPOModule.from_checkpoint(target_checkpoint_path)
ppomodule.to(device)

# %%
print("++++++++++ Critic Encoder ++++++++++")
print(ppomodule.encoder.critic_encoder)
print("++++++++++ Critic Head ++++++++++")
print(ppomodule.vf)
# %%
reader = EvalResultReader(checkpoints_path)

# %%
shap_result = np.load(os.path.join(reader.figure_dir, "shap_result.npy"))

# %%
shap_result.shape
# %%
shap_result = shap_result.squeeze()
# %%
dfs_to_explain = []
observations_to_explain = []
for row in tqdm(
    reader.eval_csv_path_df.filter(
        pl.col("checkpoint_num") == checkpoint_number
    ).iter_rows()
):
    df = pl.read_csv(row[-3])
    dfs_to_explain.append(df)
    observations_to_explain.append(np.array(df[:, :11], dtype=np.float32))

observations_to_explain_numpy = np.concatenate(observations_to_explain)
df_to_explain = reader._denormalize_dataframe(pl.concat(dfs_to_explain, how="vertical"))
# %%
shap_result_df = pl.DataFrame(
    shap_result, ["Previous " + feature for feature in reader.obs_range_dict.keys()]
)
# %%
shap_result_df
# %%
sns.set_theme(style="ticks")
plt.figure(figsize=(12, 10))
sns.violinplot(shap_result_df[:, :9].to_pandas())
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(12, 10))
sns.violinplot(shap_result_df[:, 9:].to_pandas())
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
shap.summary_plot(
    shap_result,
    features=observations_to_explain_numpy,
    feature_names=shap_result_df.columns,
)
plt.show()

# %%
shap.dependence_plot(
    "Previous V_perm_remaining",
    shap_result,
    features=observations_to_explain_numpy,
    feature_names=shap_result_df.columns,
)

# %%
shap_result_df_renamed = shap_result_df.rename(lambda x: "SHAP-" + x.replace("\n", " "))

shap_result_df_combined = pl.concat(
    [df_to_explain, shap_result_df_renamed], how="horizontal"
)
# %%
shap_result_df_combined
# %%
for feature in reader.obs_range_dict.keys():
    sns.scatterplot(
        shap_result_df_combined,
        x="Previous " + feature,
        y="SHAP-Previous " + feature,
        # hue="Previous C_feed",
        markers="+",
        alpha=0.5,
        size=0.05,
        legend=False,
    )
    plt.grid(True)
    plt.title(f"Original and SHAP value of {feature}")
    plt.tight_layout()
    plt.show()

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(df_to_explain.corr().to_pandas(), vmin=-1.0, vmax=1.0, cmap="viridis")
plt.show()
# %%
result = reader.eval_csv_path_df.group_by("checkpoint_num").agg(
    pl.col("discounted_reward_sum").min().alias("min discounted_reward_sum"),
    pl.col("discounted_reward_sum").max().alias("max discounted_reward_sum"),
    pl.col("discounted_reward_sum").mean().alias("mean discounted_reward_sum"),
    pl.col("discounted_reward_sum").median().alias("median discounted_reward_sum"),
)
result = result.with_columns(
    result["mean discounted_reward_sum"].rank(descending=True).alias("mean rank")
)
result = result.with_columns(
    result["median discounted_reward_sum"].rank(descending=True).alias("median rank")
)

# %%
plt.figure(figsize=(8, 6))
plt.plot(result["checkpoint_num"], result["median discounted_reward_sum"], lw=0.5)
sns.scatterplot(
    result,
    x="checkpoint_num",
    y="median discounted_reward_sum",
    hue="median rank",
    legend=True,
)
plt.fill_between(
    x=result["checkpoint_num"],
    y1=result["min discounted_reward_sum"],
    y2=result["max discounted_reward_sum"],
    color="C0",
    alpha=0.15,
)
plt.xlabel("Saved checkpoints")
plt.ylabel("Evaluation return")
plt.grid(True, alpha=0.5, ls=":")
plt.ylim([-150, None])
plt.show()
# %%

# %%
