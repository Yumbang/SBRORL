# %% Import necessary libraries
import numpy as np
from gymnasium_env.SBROEnvironment import SBROEnv
from gymnasium.wrappers import NormalizeObservation, FrameStackObservation

# %% Initialize SBRO server
server_url = "http://localhost:8101"

scenario_condition = {
    "T_feed_mean": 15.0,
    "T_feed_std": 0.5,
    "C_feed_mean": 0.05,
    "C_feed_std": 0.005,
}

objective_condition = {
    "time_objective_low": 28800.0,
    "time_objective_high": 43200.0,
    "V_perm_objective_low": 12.0,
    "V_perm_objective_high": 16.0,
}

sbro_env = SBROEnv(
    base_url=server_url,
    scenario_condition=scenario_condition,
    objective_condition=objective_condition,
)
# %% Basic environment reset
reset_observation, reset_info = sbro_env.reset()
print(reset_observation)
print(reset_info)
# %% Stacked environment reset
stacked_sbro_env = FrameStackObservation(sbro_env, stack_size=5)
stacked_reset_observation, stacked_reset_info = stacked_sbro_env.reset()
print(stacked_reset_observation)
print(stacked_reset_info)
# %% Scaled environment reset
scaled_sbro_env = NormalizeObservation(sbro_env)
scaled_reset_observation, scaled_reset_info = scaled_sbro_env.reset()
print(scaled_reset_observation)
print(scaled_reset_info)
# %% Scaled & Stacked environment reset
stacked_scaled_sbro_env = FrameStackObservation(
    NormalizeObservation(sbro_env), stack_size=5
)
stacked_scaled_reset_observation, stacked_scaled_reset_info = (
    stacked_scaled_sbro_env.reset()
)
print(stacked_scaled_reset_observation)
print(stacked_scaled_reset_info)
# %% Test step
dummy_action = np.array([0.0, 0.0, 0.0])

step_obs, step_reward, terminated, truncated, step_info = sbro_env.step(dummy_action)

print(step_obs)
print(step_reward)
print(step_info)
# %%
scaled_step_obs, scaled_step_reward, terminated, truncated, step_info = (
    NormalizeObservation(sbro_env).step(dummy_action)
)

print(scaled_step_obs)
print(scaled_step_reward)
print(step_info)
# %%
stacked_step_obs, stacked_step_reward, terminated, truncated, step_info = (
    stacked_sbro_env.step(dummy_action)
)

print(stacked_step_obs)
print(stacked_step_reward)
print(step_info)
# %%
sbro_env.hard_reset(dt=None, time_max=None)
