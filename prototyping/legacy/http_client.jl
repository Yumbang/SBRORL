using HTTP, JSON3, BenchmarkTools

const BASE_URL = "http://127.0.0.1:8081"

# helper to post JSON and return parsed response
function post_json(path::String, payload)
    url = "$BASE_URL$path"
    body = JSON3.write(payload)
    resp = HTTP.post(url,
        ["Content-Type" => "application/json"],
        body)
    return resp.status, resp.body
end

# Run the following lines after starting server with `http_json_api_loop.jl` file

# 1. reset_scenario
status, resp = post_json("/reset_scenario", Dict(
    "scenario_condition"  => [15.0, 0.5, 0.05, 0.01],
    "objective_condition" => [28800.0, 43200.0, 12.0, 16.0]
))
@info "reset_scenario →" status String(resp)  # expect (200, "Scenario set")

# 2. reset
status, resp_reset = post_json("/reset", Dict(
    "action"    => [5.0, 0.5, 0.0],
    "u_initial" => nothing,
    "dt"        => 30.0
))
resp_decoded = JSON3.read(String(resp_reset))
@info "reset →" status, resp_decoded["observation"], resp_decoded["info"]

# 3. step
status, resp_step = post_json("/step", Dict(
    "action" => [5.0, 0.5, 0.0]
))
resp_decoded = JSON3.read(String(resp_step))
@info "step →" status, resp_decoded["observation"], resp_decoded["reward"], resp_decoded["terminated"], resp_decoded["truncated"], resp_decoded["info"]

resp_render = HTTP.get(BASE_URL*"/render",
    ["Content-Type" => "application/json"],
    JSON3.write(Dict("mode" => "text")))

resp_health = HTTP.get(BASE_URL*"/health")

status       = resp_render.status
rendered_env = String(resp_render.body) |> print

# Benchmarking
# Interesting, or not, single- and double- threaded run made big differences!

@benchmark begin
    status, resp = post_json("/reset_scenario", Dict(
    "scenario_condition"  => [15.0, 0.5, 0.05, 0.01],
    "objective_condition" => [28800.0, 43200.0, 12.0, 16.0]
    ))
end #  Time  (mean ± σ):   870.051 μs ± 192.143 μs <THREAD = 1>
    #  Time  (mean ± σ):   143.731 μs ± 41.396 μs  <THREAD = 2>

@benchmark begin
    status, resp_reset = post_json("/reset", Dict(
    "action"    => [5.0, 0.5, 0.0],
    "u_initial" => nothing,
    "dt"        => 30.0
    ))
    resp_decoded = JSON3.read(String(resp_reset))
end #  Time  (mean ± σ):   897.830 μs ± 352.694 μs <THREAD = 1>
    #  Time  (mean ± σ):   440.690 μs ± 128.529 μs <THREAD = 2>

@benchmark begin
    _ = post_json("/reset", Dict(
        "action"    => [5.0, 0.5, 0.0],
        "u_initial" => nothing,
        "dt"        => 30.0
    ))
    status, resp_step = post_json("/step", Dict(
    "action" => [5.0, 0.5, 0.0]
    ))
    resp_decoded = JSON3.read(String(resp_step))
end #  Time  (mean ± σ):   4.460 ms ± 966.078 μs   <THREAD = 1>
    #  Time  (mean ± σ):   853.933 μs ± 194.030 μs <THREAD = 2>

@benchmark begin
    resp_render = HTTP.get(BASE_URL*"/render",
        ["Content-Type" => "application/json"],
        JSON3.write(Dict("mode" => "text")))

    status       = resp_render.status
    rendered_env = String(resp_render.body)
end #  Time  (mean ± σ):   262.658 μs ± 31.451 μs <THREAD = 1>
    #  Time  (mean ± σ):   128.903 μs ± 38.247 μs <THREAD = 2>

@benchmark begin
    resp_health = HTTP.get(BASE_URL*"/health")
    status = resp_health.status
    health = resp_health.body |> String
end #  Time  (mean ± σ):   216.172 μs ± 81.038 μs <THREAD = 1>
    #  Time  (mean ± σ):   85.569 μs ± 46.613 μs  <THREAD = 2> 