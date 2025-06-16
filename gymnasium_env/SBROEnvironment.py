import orjson
import numpy as np
import httpx
import gymnasium as gym
from gymnasium import spaces


class SBROEnv(gym.Env):
    """
    Gymnasium-compatible environment that interfaces with a backend simulation via HTTP REST API.
    """

    metadata = {"render_modes": []}  # no render modes supported for now

    def __init__(
        self,
        base_url: str,
        scenario_condition: dict,
        objective_condition: dict,
        reward_conf: dict,
        dt: float = 30.0,
        initial_action=None,
        timeout: float = 5.0,
    ):
        """
        Initialize the SBROEnv environment.
        :param base_url: Base URL of the REST API (e.g., "http://localhost:8000").
        :param scenario_condition: Scenario configuration to send to /reset_scenario.
        :param objective_condition: Objective configuration to send to /reset_scenario.
        :param dt: Time step to use in the simulation (for /reset and presumably each step).
        :param initial_action: Optional initial action to use on reset. If None, a default (zeros) will be used.
        :param timeout: Timeout in seconds for HTTP requests (defaults to 5s).
        """
        super().__init__()
        # Save parameters
        self.base_url = base_url.rstrip("/")  # ensure no trailing slash
        self.scenario_condition = scenario_condition
        self.objective_condition = objective_condition
        self.dt = dt
        self.timeout = timeout
        self.reward_conf = reward_conf

        # Define action space: 2 continuous + 1 discrete (binary)
        # Using a Tuple space: (Box(2,), Discrete(2))
        self.action_space = spaces.Tuple(
            (
                spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                ),  # continuous actions
                spaces.Discrete(2),  # binary action (0 or 1)
            )
        )

        # self.last_action = np.zeros(3, dtype=np.float32)

        # If needed, allow user to adjust Box bounds later to actual limits.
        # Define observation space: 11 continuous values
        self.obs_range_dict = {
            "T_feed": [0.0, 50.0],  # °C
            "C_feed": [0.0, 1.0],  # kg/m3
            "C_pipe_c_out": [0.0, 10.0],  # kg/m3
            "P_m_in": [0.0, 25e5],  # Pa
            "P_m_out": [0.0, 25e5],  # Pa
            "Q_circ": [0.0, 10.0],  # m3/hr
            "Q_disp": [0.0, 10.0],  # m3/hr
            "Q_perm": [0.0, 10.0],  # m3/hr
            "C_perm": [0.0, 0.5],  # kg/m3
            "time_remaining": [-86400.0, 86400.0],  # seconds
            "V_perm_remaining": [-16.0, 32.0],  # m3
        }

        self.action_range_dict = {"Q0": [4.0, 6.0], "R_sp": [0.1, 0.9], "mode": None}

        obs_low = np.array(
            [obs_range[0] for (_, obs_range) in self.obs_range_dict.items()],
            dtype=np.float32,
        )
        obs_high = np.array(
            [obs_range[1] for (_, obs_range) in self.obs_range_dict.items()],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Store the default initial action for resets
        if initial_action is not None:
            # Validate initial_action format (should be tuple of (array, int) or list of 3 values)
            self.initial_action = initial_action
        else:
            # Default initial action: 0.5 for continuous, 0 for discrete
            self.initial_action = [0.5, 0.5, 0.0]
            self.initial_action = self._decode_action(self.initial_action)

        # Initialize HTTP client (one per environment instance for thread-safety and isolation)
        self.client = httpx.Client(timeout=self.timeout)
        print(
            f"🔗 Connection to SBRO backend @({self.base_url}) successfully established."
        )

        # Hard reset the environment before starting.
        # TODO: Modify server-side handler to receive dt and time_max from client.
        # (Issue: the payloads are treated as Int, not Float)
        hard_reset_payload = {"dt": None, "time_max": None}
        # Send POST request to /hard_reset
        try:
            hard_reset_json_payload = orjson.dumps(
                hard_reset_payload, option=orjson.OPT_SERIALIZE_NUMPY
            )
            resp = self.client.post(
                f"{self.base_url}/hard_reset",
                headers={"Content-Type": "application/json"},
                content=hard_reset_json_payload,
            )
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(f"HTTP request failed during step: {e}") from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            raise RuntimeError(
                f"step returned status {status}, URL: {e.request.url!r}"
            ) from e

        print(f"🚀 Hard reset is completed @({self.base_url})")

        reward_conf_payload = self.reward_conf
        # Send POST request to /update_reward_conf
        try:
            reward_conf_json_payload = orjson.dumps(
                reward_conf_payload, option=orjson.OPT_SERIALIZE_NUMPY
            )
            resp = self.client.post(
                f"{self.base_url}/update_reward_conf",
                headers={"Content-Type": "application/json"},
                content=reward_conf_json_payload,
            )
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(f"HTTP request failed during step: {e}") from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            raise RuntimeError(
                f"step returned status {status}, URL: {e.request.url!r}"
            ) from e

        return None

    def reset(self, *, seed: int = None, options: dict = None):
        """
        Reset the environment to an initial state by calling the backend API.
        Returns:
            observation (np.ndarray), info (dict)
        """
        # Gymnasium requires seed handling in reset
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(
                seed
            )  # seed internal RNG if needed
        else:
            # Gymnasium's Env.reset might handle seeding via super().reset
            # (In gymnasium, super().reset() will set np_random if seed is not None)
            # We call super().reset() to be compliant with Gymnasium's seeding procedure.
            super().reset(seed=seed)
        # Optionally, scenario_condition or objective_condition could be changed via options
        # if options is not None:
        #     # Allow dynamic scenario/objective override via options if provided
        #     if "scenario_condition" in options:
        #         self.scenario_condition = options["scenario_condition"]
        #     if "objective_condition" in options:
        #         self.objective_condition = options["objective_condition"]
        #     if "initial_action" in options:
        #         self.initial_action = options["initial_action"]

        # 1. Call /reset_scenario to set up the scenario and objective
        try:
            scenario_condition = [
                self.scenario_condition["T_feed_mean"],
                self.scenario_condition["T_feed_std"],
                self.scenario_condition["C_feed_mean"],
                self.scenario_condition["C_feed_std"],
            ]
            objective_condition = [
                self.objective_condition["time_objective_low"],
                self.objective_condition["time_objective_high"],
                self.objective_condition["V_perm_objective_low"],
                self.objective_condition["V_perm_objective_high"],
            ]
            reset_scenario_json_payload = orjson.dumps(
                {
                    "scenario_condition": scenario_condition,
                    "objective_condition": objective_condition,
                },
                option=orjson.OPT_SERIALIZE_NUMPY,
            )

            resp = self.client.post(
                f"{self.base_url}/reset_scenario",
                headers={"Content-Type": "application/json"},
                content=reset_scenario_json_payload,
            )
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(f"HTTP request failed during reset_scenario: {e}") from e
        except httpx.HTTPStatusError as e:
            # Include status code in the error message
            status = e.response.status_code if e.response is not None else "N/A"
            raise RuntimeError(
                f"reset_scenario returned status {status}, URL: {e.request.url!r}"
            ) from e

        # 2. Call /reset to reset the state and get initial observation
        # Prepare payload for /reset
        # Determine the initial action values to send
        init_action = self.initial_action
        if isinstance(init_action, tuple):
            # If tuple (cont, disc), convert to list
            cont_part, disc_part = init_action
            # Ensure cont_part is a list of float (or convert np array to list)
            if isinstance(cont_part, np.ndarray):
                action_cont_list = cont_part.tolist()
            else:
                action_cont_list = list(cont_part)
            action_disc_val = int(disc_part)  # binary as int
            action_payload = action_cont_list + [action_disc_val]
        else:
            # If initial_action was given directly as a list of 3 values
            action_payload = list(init_action)

        reset_payload = {
            "action": action_payload,
            "u_initial": None,  # no specific initial control override
            "dt": self.dt,
        }
        try:
            reset_json_payload = orjson.dumps(
                reset_payload, option=orjson.OPT_SERIALIZE_NUMPY
            )
            resp = self.client.post(
                f"{self.base_url}/reset",
                headers={"Content-Type": "application/json"},
                content=reset_json_payload,
            )
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(f"HTTP request failed during reset: {e}") from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            raise RuntimeError(
                f"reset returned status {status}, URL: {e.request.url!r}"
            ) from e

        # Parse JSON response
        try:
            data = orjson.loads(resp.content)
        except orjson.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode JSON from reset response: {e}") from e

        # Expected keys: observation (list of 11 values), info (dict)
        if "observation" not in data or "info" not in data:
            raise RuntimeError(f"reset response JSON missing keys: {data}")

        # self.last_action = init_action

        observation = np.array(data["observation"], dtype=np.float32)
        info = data.get("info", {})  # info might be present as dict

        # Ensure observation shape is correct
        if observation.shape != (11,):
            # If shape is unexpected, try to reshape or raise error
            try:
                observation = observation.astype(np.float32).reshape(11)
            except Exception as e:
                raise RuntimeError(
                    f"Invalid observation shape from reset: {observation.shape}"
                ) from e
        return observation, info

    def step(self, action):
        """
        Send the given action to the environment (backend) and return the result.
        :param action: A tuple (continuous_action_array, discrete_action) matching the action_space.
        :return: observation (np.ndarray), reward (float), terminated (bool), truncated (bool), info (dict)
        """
        # Convert action into JSON-serializable format
        # The action is expected as a tuple: (np.ndarray or list of 2 floats, int (0 or 1))
        if isinstance(action, tuple):
            cont_part, disc_part = action
        else:
            # If action comes in as a single array-like of length 3 (not tuple), handle that too
            # For example, some agents might provide a NumPy array [a1, a2, d] directly.
            arr = np.asarray(action, dtype=float)
            if arr.shape == (3,):
                cont_part = arr[0:2]
                disc_part = int(arr[2])
            else:
                raise ValueError(f"Unrecognized action format: {action}")

        # Prepare action payload list
        if isinstance(cont_part, np.ndarray):
            action_cont_list = cont_part.tolist()
        else:
            action_cont_list = list(cont_part)

        action_disc_val = int(disc_part)
        action_payload = action_cont_list + [action_disc_val]
        action_payload = self._decode_action(action=action_payload)

        # Send POST request to /step
        try:
            action_json_payload = orjson.dumps(
                {"action": action_payload}, option=orjson.OPT_SERIALIZE_NUMPY
            )
            resp = self.client.post(
                f"{self.base_url}/step",
                headers={"Content-Type": "application/json"},
                content=action_json_payload,
            )
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(f"HTTP request failed during step: {e}") from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            raise RuntimeError(
                f"step returned status {status}, URL: {e.request.url!r}"
            ) from e
        # Parse response JSON
        try:
            data = orjson.loads(resp.content)
        except orjson.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode JSON from step response: {e}") from e

        # Expected keys: observation, reward, terminated, truncated, info
        if (
            "observation" not in data
            or "reward" not in data
            or "terminated" not in data
            or "truncated" not in data
        ):
            raise RuntimeError(f"step response JSON missing keys: {data}")

        # self.last_action =

        # Extract and convert to appropriate types
        obs = np.array(data["observation"], dtype=np.float32)
        rew = float(data["reward"])
        terminated = bool(data["terminated"])
        truncated = bool(data["truncated"])
        info = data.get("info", {})
        # Validate observation shape
        if obs.shape != (11,):
            try:
                obs = obs.astype(np.float32).reshape(11)
            except Exception as e:
                raise RuntimeError(
                    f"Invalid observation shape from step: {obs.shape}"
                ) from e
        return obs, rew, terminated, truncated, info

    def render(self, mode="text"):
        """
        Rendering is not implemented. This method is a no-op for now.
        """
        # We can optionally print the latest state or a message if needed.
        # For now, just do nothing or return no visualization.

        return None

    def reset_scenario(
        self, new_scneario_condition: dict, new_objective_condition: dict
    ):
        """
        Reset scenario generation setting with new scenario and objective conditions.
        The setting is used when starting a new episode with .reset().
        """
        self.scenario_condition = new_scneario_condition
        self.objective_condition = new_objective_condition
        return None

    def hard_reset(self, dt: float, time_max: float):
        hard_reset_payload = {"dt": dt, "time_max": time_max}
        # Send POST request to /step
        try:
            hard_reset_json_payload = orjson.dumps(
                hard_reset_payload, option=orjson.OPT_SERIALIZE_NUMPY
            )
            resp = self.client.post(
                f"{self.base_url}/hard_reset",
                headers={"Content-Type": "application/json"},
                content=hard_reset_json_payload,
            )
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(f"HTTP request failed during step: {e}") from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            raise RuntimeError(
                f"step returned status {status}, URL: {e.request.url!r}"
            ) from e

        return None

    def update_reward_conf(self, reward_conf: dict):
        self.reward_conf = reward_conf
        reward_conf_payload = reward_conf
        # Send POST request to /update_reward_conf
        try:
            reward_conf_json_payload = orjson.dumps(
                reward_conf_payload, option=orjson.OPT_SERIALIZE_NUMPY
            )
            resp = self.client.post(
                f"{self.base_url}/update_reward_conf",
                headers={"Content-Type": "application/json"},
                content=reward_conf_json_payload,
            )
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(f"HTTP request failed during step: {e}") from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            raise RuntimeError(
                f"step returned status {status}, URL: {e.request.url!r}"
            ) from e

        return None

    def _decode_action(self, action: list):
        """
        Decode normalized action into usable values for Julia SBRO environment.
        """
        decoded_Q0 = (action[0] + 1.0) / 2 * (
            self.action_range_dict["Q0"][1] - self.action_range_dict["Q0"][0]
        ) + self.action_range_dict["Q0"][0]

        decoded_R_sp = (action[1] + 1.0) / 2 * (
            self.action_range_dict["R_sp"][1] - self.action_range_dict["R_sp"][0]
        ) + self.action_range_dict["R_sp"][0]

        decoded_mode = action[2]
        return np.array([decoded_Q0, decoded_R_sp, decoded_mode])

    def close(self):
        """Clean up the environment (close HTTP client)."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass  # ignore errors on close
            self.client = None
