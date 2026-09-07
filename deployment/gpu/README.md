# RTX 4090 Policy Stack

2026-09-07 update: new resolved runs select `cec.historical_depth_source=online_history`.
Historical CEC depth is retained from the shared causal stream; legacy configs
without this field still use canonical replay. See
[`CEC_ONLINE_HISTORY_UPDATE_20260907_CN.md`](../../CEC_ONLINE_HISTORY_UPDATE_20260907_CN.md)
for costs, evidence, receipt fields and the next-start-only deployment boundary.

The RTX side runs the loopback-only MemNav, frozen NavDP and CEC hub services.
It is an internal half of the Full-Mono stack; normal operation starts it from
the Jetson:

```bash
bash deployment/go2/nav_stack.sh start \
  --config deployment/config/experiments/fullmono_imagegoal.json
```

There is no `deployment/gpu/.env`. Licensed external source/checkpoint paths,
ports, Python, runtime root and the measured camera height are tracked in
`deployment/config/system.json`. The Jetson resolves that file with the
experiment file, records a `config_id`, copies the exact resolved JSON to
`runtime/config/` on this machine and requires matching Git revisions before
startup.

The GPU leaf commands accept only that resolved file:

```bash
bash deployment/gpu/scripts/preflight.sh --config runtime/config/CONFIG_ID.json
bash deployment/gpu/scripts/run_policy_stack.sh --config runtime/config/CONFIG_ID.json
bash deployment/gpu/scripts/stop_policy_stack.sh --config runtime/config/CONFIG_ID.json
```

The licensed MemNav source remains in its separate research worktree. Apply
the tracked current-frame depth-reuse patch once after checking out or updating
that worktree; the GPU preflight then verifies the applied patch on every start:

```bash
bash deployment/gpu/scripts/apply_memnav_source_patch.sh \
  --config runtime/config/CONFIG_ID.json
```

The patch reuses the depth tensor already produced by the post-warmup flow
gate for the same frame-bound MDTEC transaction. It also exposes append,
retrieval, depth-prediction and depth-materialization timings in the runtime
receipt. It never reuses depth across frame indices or RGB hashes.

Ports `8888`, `18888` and `18889` bind to loopback. Do not expose them on the
LAN; the Jetson launcher owns the SSH local forward. These services have no
Unitree dependency and no actuator path.

Check Python syntax without starting services or issuing motion commands:

```bash
/home/asus/miniconda3/envs/memnav-realworld/bin/python -m compileall -q deployment/gpu
```

No unit-test suite is maintained. Use the resolved-config preflight and service
health receipts for runtime validation; syntax checks alone do not prove readiness.

## Resident models, isolated episodes

The Episode workflow uses `fullmono.sh park` after sealing Survey and after a
Revisit ends. This stops the Jetson motion stack first, removes the GPU hub,
drains NavDP, and resets NavDP/LingBot/CEC episode state. Model weights remain
loaded. CPU garbage collection and CUDA unused-cache release run at episode
boundaries, never in the navigation loop. Existing recordings, sealed datasets
and per-episode RGB buffers are not deleted. An idle GPU therefore still shows
the VRAM used by model weights; zero VRAM is not the expected idle state.

On the next start, only a managed, parked, empty and compatible model session
is reused. The hub is newly created with the new immutable run config (including
CEC versus Mono-native authority). Model/code/weight metadata changes cause a
cold start of the owned parked session. Active or unidentified sessions are
not replaced. Failed cleanup discards the owned model processes rather than
letting the next experiment inherit uncertain state.

Every formal arm still performs the original reset, sealed manifest/hash checks,
full causal Survey replay, exact frozen goal binding and query-start FIFO prime.
No controller, speed, termination, sensor or replay rule is relaxed. Seeded
NavDP reset applies the seed after model initialization too, so loading weights
cannot shift the random sequence on cold starts but not warm starts.

`stop` remains the explicit full shutdown command to release the model weights
as well. GPU-only lifecycle commands (no actuator path):

```bash
bash deployment/gpu/scripts/park_policy_stack.sh --config runtime/config/CONFIG_ID.json
# Full shutdown, including weights:
bash deployment/gpu/scripts/stop_policy_stack.sh --config runtime/config/CONFIG_ID.json
```

While parked the hub is intentionally absent; model idle receipts are available
on loopback `/resident/status` at ports 18888 and 8888. Cleanup receipts with
PIDs, zero frame/queue counts and CUDA allocated/reserved bytes are appended to
`runtime/gpu/logs/resident_lifecycle.jsonl`. Reuse saves model loading, not Survey
replay or recording finalization; neither constant-time preparation nor identical
stochastic trajectories is implied.
