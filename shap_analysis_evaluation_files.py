# %%
import os
import pickle as pkl
import argparse

import numpy as np
import polars as pl
import shap
import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule as PPOModule,
)
from shap import KernelExplainer
from tqdm import tqdm

from read_evaluation_files import EvalResultReader


def main(checkpoints_path, checkpoint_number):
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
    observations = []
    for checkpoint_n in range(checkpoint_number - 2, checkpoint_number + 2):
        for row in tqdm(
            reader.eval_csv_path_df.filter(
                pl.col("checkpoint_num") == checkpoint_n
            ).iter_rows()
        ):
            df = pl.read_csv(row[-3])
            observations.append(np.array(df[:, :11], dtype=np.float32))

    observations_numpy = np.concatenate(observations)

    # %%
    observations_to_explain = []
    for row in tqdm(
        reader.eval_csv_path_df.filter(
            pl.col("checkpoint_num") == checkpoint_number
        ).iter_rows()
    ):
        df = pl.read_csv(row[-3])
        observations_to_explain.append(np.array(df[:, :11], dtype=np.float32))

    observations_to_explain_numpy = np.concatenate(observations_to_explain)

    # %%
    def critic_wrapper(obs_numpy):
        obs_tensor = torch.tensor(obs_numpy, device=device)
        encoder_output = ppomodule.encoder.critic_encoder({"obs": obs_tensor})
        value_prediction = ppomodule.vf(encoder_output["encoder_out"])
        return value_prediction.cpu().detach().numpy()

    def predict_fn(numpy_data):
        """
        A single, efficient function that takes a NumPy array of observations,
        runs them through the critic model, and returns a NumPy array of values.
        """
        # Convert numpy data to a GPU tensor
        tensor_data = torch.from_numpy(numpy_data).float().to(device)

        with torch.no_grad():
            # 1. Pass observations through the main encoder
            # The encoder expects a dictionary input.
            encoder_output = ppomodule.encoder.critic_encoder({"obs": tensor_data})

            # 2. Pass the encoder's output to the value function head
            predictions = ppomodule.vf(encoder_output["encoder_out"])

        # Return predictions as a CPU numpy array
        return predictions.cpu().numpy()

    # %%
    print("Summarizing background data with k-means...")
    # background_summary = shap.kmeans(observations_numpy, 250)

    # with open(os.path.join(reader.figure_dir, "background_summary.pkl"), "wb") as f:
    #     pkl.dump(background_summary, f)

    with open(os.path.join(reader.figure_dir, "background_summary.pkl"), "rb") as f:
        background_summary = pkl.load(f)

    # %%

    # --- 2. Create the explainer using the SMALL summary ---
    # This will now be memory-efficient and will not crash the kernel.
    # explainer = shap.DeepExplainer(wrapped_critic, background_summary_tensor)
    explainer = KernelExplainer(predict_fn, background_summary)

    print("SHAP explainer created successfully with summarized background data.")

    # --- 3. You can now calculate SHAP values as before ---
    # shap_values = explainer.shap_values(observations_to_explain_tensor)

    # %%
    batch_size = 2048  # You can adjust this based on your GPU memory

    # 2. Prepare to collect the results
    all_shap_values = []

    print(f"Calculating SHAP values in batches of {batch_size}...")

    # 3. Loop through the tensor in batches
    for i in tqdm(range(0, len(observations_to_explain_numpy), batch_size)):
        # Get the current batch of observations
        batch_obs = observations_to_explain_numpy[i : i + batch_size]

        # Calculate SHAP values for just this batch
        batch_shap_values = explainer.shap_values(batch_obs)

        # Store the results (they will be NumPy arrays)
        all_shap_values.append(batch_shap_values)

    # 4. Combine the results from all batches into one large array
    final_shap_values = np.concatenate(all_shap_values, axis=0)

    print(
        f"\nSuccessfully calculated SHAP values for all {final_shap_values.shape[0]} instances."
    )

    np.save(os.path.join(reader.figure_dir, "shap_result.npy"), final_shap_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-path", type=str, required=True)
    parser.add_argument("--checkpoint-number", type=int, required=True)
    args = parser.parse_args()
    main(args.checkpoints_path, args.checkpoint_number)
