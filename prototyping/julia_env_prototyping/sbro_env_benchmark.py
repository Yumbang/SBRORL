# %% Testing http server communication
import httpx

server_url = "http://localhost:8080/"

r = httpx.get(server_url + "health")

print(f"Status: {r.status_code}\nPayload: {r.text}")
# %% Defining variables for JSON communication with server
import orjson
import numpy as np

dt = 30.0

scenario_condition = np.array([15.0, 0.5, 0.05, 0.01])
objective_condition = np.array([28800.0, 43200.0, 12.0, 16.0])

dummy_action = np.array([5.0, 0.5, 0.0])

dummy_u_initial = None

# %% Constituting proper JSON payloads

reset_scenario_payload = {
    "scenario_condition": scenario_condition,
    "objective_condition": objective_condition,
}

reset_payload = {"action": dummy_action, "u_initial": dummy_u_initial, "dt": dt}

step_payload = {"action": dummy_action}

opts = orjson.OPT_SERIALIZE_NUMPY  # + other flags if you like

json_reset_scenario = orjson.dumps(reset_scenario_payload, option=opts)
json_reset = orjson.dumps(reset_payload, option=opts)
json_step = orjson.dumps(step_payload, option=opts)

# %% Decode and print the jsonized payloads
print(json_reset_scenario.decode())
print(json_reset.decode())
print(json_step.decode())

# %% Request scenario reset to sbro environment server
resp_reset_scenario = httpx.post(
    f"{server_url}reset_scenario",
    headers={"Content-Type": "application/json"},
    content=json_reset_scenario,
)
print(resp_reset_scenario.status_code, resp_reset_scenario.text)

# %% Request environment reset to sbro environment server
resp_reset = httpx.post(
    f"{server_url}reset",
    headers={"Content-Type": "application/json"},
    content=json_reset,
)
print(resp_reset.status_code, resp_reset.text)

# %% Request many number of repetitive reset - step
import time

reset_step_times = []

for i in range(1, 1000):
    start_time = time.perf_counter()

    resp_reset = httpx.post(
        f"{server_url}reset",
        headers={"Content-Type": "application/json"},
        content=json_reset,
    )
    # print(resp_reset.status_code, resp_reset.text)

    resp_step = httpx.post(
        f"{server_url}step",
        headers={"Content-Type": "application/json"},
        content=json_step,
    )
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

# Without --cpus tag
# Statistics after 900 iterations:
# Mean time: 0.0229s
# Std dev: 0.0031s
# Min time: 0.0211s
# Max time: 0.0432s

# With --cpus 1 tag
# Statistics after 900 iterations:
# Mean time: 0.0231s
# Std dev: 0.0033s
# Min time: 0.0212s
# Max time: 0.0458s

# With --cpus 2 tag
# Statistics after 900 iterations:
# Mean time: 0.0230s
# Std dev: 0.0062s
# Min time: 0.0212s
# Max time: 0.1819s
