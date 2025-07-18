import os
import numpy as np

eval_script_path = (
    "/home/ybang-eai/research/2025/SBRO/SBRORL/evaluation/evaluate_ppo_sequentially.py"
)
checkpoints_dir = "/home/ybang-eai/research/2025/SBRO/SBRORL/result/PPO/2025-07-08 12:54:48.371418/PPO_SBRO/PPO_sbro_env_v1_4afbe_00000_0_2025-07-08_12-54-48"
shell_dir = os.path.join(checkpoints_dir, "shell")

os.makedirs(shell_dir, exist_ok=True)

N_ENVS = 25
STARTING_PORT = 8100

n_checkpoints = 74

checkpoints_range = np.arange(1, n_checkpoints)

distributed_checkpoints = np.array_split(checkpoints_range, N_ENVS)


def execution_line(checkpoint_number, port):
    execution_text = (
        f'uv run "{eval_script_path}" '
        f'--checkpoints-path "{checkpoints_dir}" --checkpoint-number {checkpoint_number} --port {port}\n'
    )
    return execution_text


def bash_execution_line(shell_file_number):
    shell_path = os.path.join(shell_dir, f"{shell_file_number}.sh")
    return f'bash "{shell_path}"\n'


for n_env, checkpoints in enumerate(distributed_checkpoints):
    shell_script = [
        execution_line(checkpoint, STARTING_PORT + n_env) for checkpoint in checkpoints
    ]

    with open(os.path.join(shell_dir, f"{n_env}.sh"), "w", encoding="utf-8") as f:
        for line in shell_script:
            f.write(line)

shell_number_range = np.arange(N_ENVS)
distributed_shell_numbers = np.array_split(shell_number_range, 4)

for i, shell_numbers in enumerate(distributed_shell_numbers):
    shell_script = [bash_execution_line(shell_number) for shell_number in shell_numbers]

    with open(os.path.join(shell_dir, f"final_{i}.sh"), "w", encoding="utf-8") as f:
        for line in shell_script:
            f.write(line)
