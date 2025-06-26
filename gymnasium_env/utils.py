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


class FlattenedActionWrapper(gym.ActionWrapper):
    """
    Convert the SBROEnv hybrid action space
        (Box(2,), Discrete(2))
    into a single Discrete(total_actions) space suitable for RLlib's
    ParametricActionsModel.  Also provides a flat 0/1 action_mask.
    """

    def __init__(self, env: gym.Env, num_discrete_steps: int = 7):
        """
        Args
        ----
        env : SBROEnv
        num_discrete_steps : odd integer  ≥3.
            The two continuous controls are quantised into this many bins each.
            Example: 7 → deltas [-3 … +3], centre bin = no-op.
        """
        super().__init__(env)

        if num_discrete_steps % 2 == 0:
            raise ValueError(
                f"num_discrete_steps must be odd; got {num_discrete_steps}"
            )

        self.num_steps = num_discrete_steps
        self.num_steps_aside = (num_discrete_steps - 1) // 2

        # Total flat actions = bins₁ × bins₂ × modes
        self.total_actions = num_discrete_steps * num_discrete_steps * 2
        self.action_space = spaces.Discrete(self.total_actions)

        # Observation = raw obs + FLAT mask
        self.observation_space = spaces.Dict(
            {
                "observations": env.observation_space,  # unchanged
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.total_actions,),
                    dtype=np.float32,
                ),
            }
        )
        # self.observation_space = env.observation_space

        # Pre-compute step size table for the two continuous controls
        step_unit = 1.0 / (60.0 * 10.0 * 2.0 * self.num_steps_aside) * env.env.dt
        self.deltas = np.linspace(
            -self.num_steps_aside * step_unit,
            +self.num_steps_aside * step_unit,
            num_discrete_steps,
            dtype=np.float32,
        )

        # Track current continuous state
        self._a_cont = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers to map flat index  ↔  (idx1, idx2, mode)
    # ------------------------------------------------------------------
    def _index_to_components(self, index: int):
        mode = index % 2
        idx = index // 2
        idx2 = idx % self.num_steps
        idx1 = idx // self.num_steps
        return idx1, idx2, mode

    # ------------------------------------------------------------------
    # Gymnasium ActionWrapper interface
    # ------------------------------------------------------------------
    def action(self, flat_index: int):
        """
        Convert the flat discrete index into the original hybrid tuple:
            (np.array([Q0_delta, R_sp_delta], dtype=float32),  int(mode))
        """
        idx1, idx2, mode = self._index_to_components(flat_index)
        delta_vec = np.array([self.deltas[idx1], self.deltas[idx2]], dtype=np.float32)

        # Apply delta in place and clip
        self._a_cont = np.clip(self._a_cont + delta_vec, -1.0, 1.0)
        return self._a_cont.copy(), mode  # tuple expected by SBROEnv

    def reset(self, *, seed=None, options=None):
        self._a_cont[:] = 0.0  # reset internal state
        raw_obs, info = self.env.reset(seed=seed, options=options)
        return self._format_obs(raw_obs), info

    def step(self, flat_index: int):
        raw_obs, rew, term, trunc, info = self.env.step(self.action(flat_index))
        return self._format_obs(raw_obs), rew, term, trunc, info

    # ------------------------------------------------------------------
    # Mask logic
    # ------------------------------------------------------------------
    def _format_obs(self, raw_obs):
        """
        Return dict with flat mask (float32 0/1) + original observation.
        """
        mask = self._compute_flat_mask()
        return {"observations": raw_obs, "action_mask": mask}

    def _compute_flat_mask(self) -> np.ndarray:
        """
        Build a flat mask of length total_actions.
        Invalid actions get 0.0, valid actions 1.0.
        """
        mask1 = np.ones(self.num_steps, dtype=np.float32)
        mask2 = np.ones(self.num_steps, dtype=np.float32)

        # mask for first continuous control
        for i, d in enumerate(self.deltas):
            if not (-1.0 <= self._a_cont[0] + d <= 1.0):
                mask1[i] = 0.0
        # mask for second continuous control
        for i, d in enumerate(self.deltas):
            if not (-1.0 <= self._a_cont[1] + d <= 1.0):
                mask2[i] = 0.0

        # mode (third dim) is always valid
        mask3 = np.ones(2, dtype=np.float32)

        # Kronecker-style product to flatten
        flat_mask = np.kron(np.kron(mask1, mask2), mask3)
        return flat_mask  # shape (total_actions,)


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


def discrete_sbro_env_creator(env_config):
    env = sbro_env_creator(env_config)
    return FlattenedActionWrapper(env=env, num_discrete_steps=5)


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
