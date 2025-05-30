#%% Testing http server communication
import httpx

server_url = "http://localhost:8081/"

r = httpx.get(server_url+"health")

print(f"Status: {r.status_code}\nPayload: {r.text}")
# %% Defining variables for JSON communication with server
import orjson
import numpy as np

dt = 30.0

scenario_condition  = np.array([15.0, 0.5, 0.05, 0.01])
objective_condition = np.array([28800.0, 43200.0, 12.0, 16.0])

dummy_action = np.array([5.0, 0.5, 0.0])

dummy_u_initial = None

# %% Constituting proper JSON payloads

reset_scenario_payload = {
    "scenario_condition": scenario_condition,
    "objective_condition": objective_condition
}

reset_payload = {
    "action": dummy_action,
    "u_initial": dummy_u_initial,
    "dt": dt
}

step_payload = {
    "action": dummy_action
}

opts = orjson.OPT_SERIALIZE_NUMPY          # + other flags if you like

json_reset_scenario = orjson.dumps(reset_scenario_payload, option=opts)
json_reset          = orjson.dumps(reset_payload,          option=opts)
json_step           = orjson.dumps(step_payload,           option=opts)

# %% Request 30 steps in row
for _ in range(1, 30):
    resp_step = httpx.post(
        f"{server_url}step",
        headers={"Content-Type": "application/json"},
        content=json_step)
    # print(resp_step.status_code, resp_step.text)

# %% Request many number of repetitive reset - step
import time

reset_step_times = []

for i in range(1, 1000):
    start_time = time.perf_counter()
    
    resp_reset = httpx.post(
        f"{server_url}reset",
        headers={"Content-Type": "application/json"},
        content=json_reset)
    # print(resp_reset.status_code, resp_reset.text)

    resp_step = httpx.post(
        f"{server_url}step",
        headers={"Content-Type": "application/json"},
        content=json_step)
    # print(resp_step.status_code, resp_step.text)
    
    loop_time = time.perf_counter() - start_time
    reset_step_times.append(loop_time)
    
    if i % 100 == 0:  # Print stats every 100 iterations
        mean_time = np.mean(reset_step_times)
        std_time = np.std(reset_step_times)
        print(f"Statistics after {i} iterations:")
        print(f"Mean time: {mean_time:.4f}s")
        print(f"Std dev: {std_time:.4f}s")
        print(f"Min time: {min(reset_step_times):.4f}s")
        print(f"Max time: {max(reset_step_times):.4f}s")