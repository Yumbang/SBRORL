# SBRORL: Reinforcement Learning for SBRO Process Control

This repository contains the code for training a Reinforcement Learning (RL) agent to control a simulated Semi-Batch Reverse Osmosis (SBRO) process. The project uses `ray[rllib]` for distributed training of a Proximal Policy Optimization (PPO) agent.

The RL agent (Python client) communicates with a separate backend simulation service (assumed to be a Julia-based application) via a REST API. This architecture decouples the learning algorithm from the environment simulation.

## Features

- **Custom Gymnasium Environment**: A `gymnasium`-compatible environment (`SBROEnv`) that interfaces with the SBRO simulation backend.
- **Distributed Training**: Leverages `ray[rllib]` for efficient, distributed RL training using the PPO algorithm.
- **Curriculum Learning**: Implements a curriculum learning strategy to improve training stability and agent performance by gradually increasing task difficulty.
- **Client-Server Architecture**: The Python-based RL agent is decoupled from the process simulation, allowing for modularity and scalability.
- **Detailed Logging**: Saves training metrics, episode data (in CSV format), and model checkpoints for analysis.

## Project Structure

```
/
├── gymnasium_env/        # Contains the custom SBRO Gymnasium environment
│   └── SBROEnvironment.py
├── docker/                 # Docker configuration for the backend service
│   └── docker-compose.yml
├── result/                 # Default directory for saving training results
├── train_ppo_tune.py       # Main script for training the RL agent
├── pyproject.toml          # Project metadata and Python dependencies
└── README.md               # This file
```

## Dependencies
We highly recommend using `uv` to manage the Python environment.
- `uv`

The main Python dependencies are listed in `pyproject.toml` and include:
- `ray[rllib]`
- `stable-baselines3`
- `gymnasium`
- `torch`
- `httpx`
- `polars`
- `numpy`

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd SBRORL
    ```

2.  **Install Python dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    uv sync
    ```

3.  **Backend Simulation Service:**
    This project requires the SBRO simulation backend to be running. The backend is expected to expose a REST API that the `SBROEnv` can connect to. The `docker/docker-compose.yml` file is likely used to start these services.

    Start the backend services (assuming Docker is installed):
    ```bash
    cd docker
    docker compose up
    ```
    This will likely start multiple instances of the simulation, listening on ports starting from 8100, as configured in `train_ppo_tune.py`.

## How to Run Training

Once the backend simulation services are running, you can start the RL agent training by executing the main training script:

```bash
python train_ppo_tune.py
```

or

```bash
uv run train_ppo_tune.py
```

### Configuration

Key parameters in `train_ppo_tune.py` can be modified to control the training process:
- `algorithm`: The RL algorithm to use (e.g., "PPO", "APPO").
- `num_environments`: The number of parallel simulation environments to use for training. This should match the number of running backend services.
- `USE_CURRICULUM`: A boolean flag to enable or disable curriculum learning.
- `REWARD_CONFIG`: A dictionary defining the weights for different components of the reward function.
- `SAVE_DIR`: The root directory where training results, logs, and checkpoints will be saved.

## Environment Details

### Observation Space
The agent receives an 11-dimensional continuous observation vector with the following features:
1.  `T_feed` (°C)
2.  `C_feed` (kg/m³)
3.  `C_pipe_c_out` (kg/m³)
4.  `P_m_in` (Pa)
5.  `P_m_out` (Pa)
6.  `Q_circ` (m³/hr)
7.  `Q_disp` (m³/hr)
8.  `Q_perm` (m³/hr)
9.  `C_perm` (kg/m³)
10. `time_remaining` (seconds)
11. `V_perm_remaining` (m³)

### Action Space
The agent's action space is a `Tuple` consisting of:
1.  `spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)`: Two continuous actions.
2.  `spaces.Discrete(2)`: One binary discrete action.

These normalized actions are decoded by the environment into physical values (`Q0`, `R_sp`, `mode`) before being sent to the simulation backend.
