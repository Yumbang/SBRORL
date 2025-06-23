import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium_env.SBROEnvironment import SBROEnv


class MinMaxNormalizeObservation(gym.ObservationWrapper):
    """
    Applies min-max scaling to the observations of an environment to a [0, 1] range.
    This version correctly handles both `step` and `reset` methods.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise ValueError("This wrapper only works with Box observation spaces.")

        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high
        self.obs_range = self.obs_high - self.obs_low
        # Avoid division by zero
        self.obs_range[self.obs_range == 0] = 1e-6

        # Update the environment's observation space to reflect the new [0, 1] range
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Applies the min-max scaling to the observation from a step."""
        clipped_obs = np.clip(obs, self.obs_low, self.obs_high)
        scaled_obs = (clipped_obs - self.obs_low) / self.obs_range
        return scaled_obs.astype(np.float32)


# class DiscreteSBROWrapper(gym.ActionWrapper):
#     """
#     Applies action discretization for Q_0 and R_sp, by letting the agent add or subtract
#     values from the previous action.
#     """
#     def __init__(self, env: gym.Env):
#         super().__init__(env)


def sbro_env_creator(env_config):
    """
    Creates an SBROEnv instance for a specific worker using a unique configuration.
    """
    worker_index = env_config.worker_index
    worker_specific_config = env_config["worker_configs"][
        worker_index % len(env_config["worker_configs"])
    ]

    print(
        f"Worker {worker_index} creating environment with config: "
        f"URL='{worker_specific_config['base_url']}', "
        f"T_feed_mean={worker_specific_config['scenario_condition']['T_feed_mean']}"
    )

    # --- START: CORRECTED SECTION ---
    # Call the SBROEnv constructor using the `worker_specific_config` dictionary
    # that was selected above. This ensures each worker gets its unique settings.
    env = SBROEnv(
        base_url=worker_specific_config["base_url"],
        scenario_condition=worker_specific_config["scenario_condition"],
        objective_condition=worker_specific_config["objective_condition"],
        reward_conf=worker_specific_config["reward_conf"],
        dt=env_config.get("dt", 30.0),  # Global settings can still be used
    )

    # return NormalizeObservation(env)
    return MinMaxNormalizeObservation(env)


def generate_entropy_schedule(
    max_timesteps: int, start_value: float, end_value: float, num_points: int = 10
) -> list:
    """
    Generates a linear schedule for a hyperparameter like entropy_coef.

    This is useful for annealing a value over the course of training, for example,
    to encourage more exploration at the beginning and less at the end.

    Args:
        max_timesteps: The total number of training timesteps over which to schedule.
        start_value: The initial value of the hyperparameter (at timestep 0).
        end_value: The final value of the hyperparameter (at max_timesteps).
        num_points: The number of points to define in the schedule.

    Returns:
        A list of [timestep, value] pairs formatted for RLlib's scheduling config.
    """
    # Generate `num_points` evenly spaced timesteps from 0 to max_timesteps.
    timesteps = np.linspace(0, max_timesteps, num=num_points, dtype=int)

    # Generate `num_points` linearly interpolated values from start_value to end_value.
    values = np.linspace(start_value, end_value, num=num_points)

    # Combine them into the desired format: [[timestep, value], ...]
    schedule = [[int(t), float(v)] for t, v in zip(timesteps, values)]

    return schedule
