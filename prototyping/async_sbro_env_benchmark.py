# async_bench_csv.py  ────────────────────────────────────────────────────────
import asyncio, time, numpy as np, httpx, orjson, csv, pathlib, datetime as dt

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
SERVER_PORTS = list(range(8100, 8116))   # 16 containers: 8100–8115
SERVER_URLS  = [f"http://localhost:{p}/" for p in SERVER_PORTS]

ITERATIONS_PER_SERVER = 1000             # total reset→step pairs per server
PARALLEL_PER_SERVER   = 1                # concurrent pipelines per server
OUTFILE               = (
    pathlib.Path(__file__).with_suffix("")  # same dir/name
    .with_name(f"results_{dt.datetime.utcnow():%Y%m%dT%H%M%SZ}.csv")
)
# --------------------------------------------------------------------------

# ---------- request payloads ----------------------------------------------
opts = orjson.OPT_SERIALIZE_NUMPY
HEADERS = {"Content-Type": "application/json"}

json_reset_scenario = orjson.dumps(
    {
        "scenario_condition":  [15.0, 0.5, 0.05, 0.01],
        "objective_condition": [28800.0, 43200.0, 12.0, 16.0],
    },
    option=opts,
)
json_reset = orjson.dumps(
    {"action": [5.0, 0.5, 0.0], "u_initial": None, "dt": 30.0}, option=opts
)
json_step = orjson.dumps({"action": [5.0, 0.5, 0.0]}, option=opts)

# ---------- helpers --------------------------------------------------------
async def warmup(client: httpx.AsyncClient, url: str):
    await client.post(url + "reset_scenario", headers=HEADERS, content=json_reset_scenario)

async def one_cycle(client: httpx.AsyncClient, url: str) -> float:
    """Run reset+step against *url* and return elapsed wall-clock seconds."""
    t0 = time.perf_counter()
    await client.post(url + "reset", headers=HEADERS, content=json_reset)
    await client.post(url + "step",  headers=HEADERS, content=json_step)
    return time.perf_counter() - t0

async def bench_one_server(url: str, n_iter: int, parallel: int, results: list):
    port = int(url.split(":")[2].rstrip("/"))
    limits = httpx.Limits(max_connections=parallel, max_keepalive_connections=parallel)
    async with httpx.AsyncClient(limits=limits, timeout=30) as client:
        await warmup(client, url)

        pending = {asyncio.create_task(one_cycle(client, url)) for _ in range(parallel)}
        done_count = 0
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                duration = task.result()
                done_count += 1
                # collect *after* we have the duration
                results.append(
                    {
                        "port": port,
                        "iteration": done_count,
                        "duration_sec": duration,
                        "ts_utc": dt.datetime.utcnow().isoformat(timespec="microseconds"),
                    }
                )
                if done_count < n_iter:
                    pending.add(asyncio.create_task(one_cycle(client, url)))

async def main():
    results: list[dict] = []
    coros = [
        bench_one_server(url, ITERATIONS_PER_SERVER, PARALLEL_PER_SERVER, results)
        for url in SERVER_URLS
    ]
    # gather with live stats
    total_target = ITERATIONS_PER_SERVER * len(SERVER_URLS)
    last_print   = 0
    for fut in asyncio.as_completed(coros):
        await fut            # each coroutine finishes independently
        if len(results) - last_print >= 100:
            last_print = len(results)
            arr = np.fromiter((r["duration_sec"] for r in results), float)
            print(
                f"[{len(results):4d}/{total_target}] "
                f"mean={arr.mean():.4f}s  std={arr.std():.4f}s "
                f"min={arr.min():.4f}s   max={arr.max():.4f}s"
            )

    # ---------------- write CSV *after* the benchmark ----------------------
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTFILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["port", "iteration", "duration_sec", "ts_utc"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("── Benchmark complete ──")
    print(f"Wrote {len(results)} rows to {OUTFILE}")

if __name__ == "__main__":
    asyncio.run(main())
