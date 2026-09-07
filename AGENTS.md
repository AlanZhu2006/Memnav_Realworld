# MemNav-RealWorld working and operation rules

These instructions apply to `/home/unitree/MemNav-RealWorld` and its deployed
counterpart on `work-pc`.

## Validation policy

- Do not create, restore, or run unit tests unless the user explicitly asks for
  them in a later request. Repository-owned unit-test files have been removed.
- Use syntax/import checks, existing configuration and data-integrity tools,
  documented preflight, and motion-locked observation as appropriate.
- Keep experiment evaluation, data capture, and hardware diagnostic tools. They
  are not disposable unit tests. Preserve runtime data unless its exact removal
  is authorized.
- Report what was actually verified. Static checks and stationary telemetry do
  not prove navigation success. Never initiate motion just to validate code.

## Motion authorization: no repeated confirmation

- A direct user request to start Revisit/navigation is the motion authorization
  for that run. A Foxglove `REVISIT` click has the same meaning. Carry forward
  the user's already-stated onsite safety, controller, and emergency-stop
  readiness; do not ask for a second confirmation or a fixed authorization
  phrase merely because preparation took time or code was synchronized.
- Perform the existing automated preflight before motion. Authorization does
  not override actual sensor, feedback, connection, or emergency-stop faults.
  If evidence contradicts the stated onsite conditions, stop and explain the
  concrete issue rather than issuing a generic reconfirmation request.
- Requests to edit code, commit, push, synchronize, capture a goal, Prepare, or
  record Survey do not implicitly authorize autonomous motion.
- A user Stop request cancels the current motion authorization. Keep the robot
  locked until the user issues a new motion request; do not auto-resume after a
  fault or deploy merely because an earlier run was authorized.
- Exception explicitly requested by the user: a temporary RGB-D freshness pause
  inside an active run preserves that run's authorization. At age >2 seconds,
  command zero, discard the old action, wait for post-stop RGB-D and a new
  accepted plan, then continue. This never clears estop or re-enables a stopped
  run. Hard faults and the existing overall run timeout still terminate it.
- `nav_stack.sh start` remains observation-only: `enabled=false`, `estop=true`.
  Do not change this default. Only use `run`, clear estop, enable execution, or
  publish motion commands within a user-requested motion run.

## Deployment identity and preservation

- Jetson host: `unitree-dog`, Orin NX 16 GB, Ubuntu 22.04, L4T R36.4.3,
  ROS 2 Humble, JetPack 6.2.1 user-space components.
- Preserve machine-local paths, hostname, and interface in
  `deployment/config/system.json`. Verify actual deployed revisions with Git;
  do not reset a newer checkout to a historical revision.
- Support workspaces live under `/home/unitree/.local/share/memnav`:
  CycloneDDS, Unitree SDK2 Python, Tinynav, RealSense ROS, message_filters, Odin.
- Runtime evidence, checkpoints, and resolved contracts live under `runtime/`.
  Use the current run's resolved contract; do not substitute a historical hash.
- NavDP uses `.venv-navdp` and NVIDIA PyTorch/CUDA 12.6.
- D435i validated stream: aligned RGB-D 848x480x30, USB SuperSpeed;
  librealsense/realsense-ros 2.58.1, firmware 5.17.0.10.
- Go2 link: `enP8p1s0`, local `192.168.123.164/24`, robot `192.168.123.161`.
  Prefer observation-only SDK telemetry for link checks.
- Odin1 native 0.14 is installed; do not claim live hardware validation until
  USB device `2207:0019` is actually observed.
- `ssh work-pc` accesses the RTX 4090 with a dedicated key. Preserve unrelated
  pre-existing `cec-realworld` GPU sessions; ports may already be occupied.
  GPU checkout: `/home/asus/Research/MemNav-RealWorld`.
- Last documented Jetson power mode: 15 W, four CPU cores. Verify current
  state when relevant. Switching to 25 W/MAXN requires explicit confirmation
  that power delivery and cooling are adequate.
- Query-time RGB updates during an IMU turn do not establish reliable LingBot
  translation. IMU currently controls the actuator only; the camera-height
  scale receipt is not a pose-drift correction. The 2026-09-07 latency check
  was stationary, and a forward-only arc U-turn has not been ported. Preserve
  these limits when documenting or deploying the execution adaptation.

## Experiment names and the Baseline entry point

- User-facing names: **Baseline = Mono-native** (`--arm mono_native`,
  `authority_mode=native`); **our method = CEC** (`--arm mono_cec`,
  `authority_mode=cec`). Use these names consistently in discussion and run
  records. Do not use the ambiguous phrase "run native" / "原生跑".
- In the current paired Episode workflow, a request for Baseline means the
  existing RTX 4090 Full-Mono `mono_native` arm, not the Jetson-local RGB-D
  profile. Prepare it through `deployment/go2/offboard/revisit_experiment.sh`
  `formal-start ... --arm mono_native`, supplying the current Episode's sealed
  dataset, exact frozen goal/hash, unique run ID and required registration (or
  explicitly identified engineering-run parameters). Use the matching resolved
  contract for supervised execution. Keep all motion/preflight rules above.
- Both paired arms run NavDP on the RTX 4090 and use the same causal LingBot
  monocular-depth pipeline. D435i metric depth stays in local safety/evidence,
  not in either paired arm's policy input. Preserve the paired goal, starting
  pose, controller settings, budget and termination procedure; verify actual
  runtime authority and goal/dataset identities before motion.
- "Without memory" here means **without CEC long-term retrieval / direct-pose
  control authority**, not a stateless policy. Mono-native ignores retrieved
  navigation proposals and routes to native ImageGoal NavDP (`cec_takeover=false`,
  no selected anchor). NavDP retains its own short-term observation FIFO.
  The shared LingBot causal state, sealed-dataset replay and depth-producing
  probe remain; do not claim no history, no Survey processing, no retrieval
  computation, or no memory anywhere. Independent formal query preparation
  initializes the NavDP FIFO from the current query-start frame rather than
  filling it from Survey history. Describe the ablation as disabling memory
  guidance, not removing every stateful component.
- `deployment/config/experiments/native_imagegoal.json` is a separate
  **Jetson-local NavDP RGB-D diagnostic profile**, not this paired Baseline.
  Do not substitute it, or propose rebuilding a 4090 Baseline that already
  exists, merely because the user says "Baseline" or "Mono-native".
- The experimental `fullmono_local_approach.json` explicitly adds the same
  height-scaled terminal adapter to both paired arms. In that profile,
  describe the comparator as **Mono-native + shared terminal adapter**, not
  untouched Mono-native; preserve `terminal_approach` in resolved/formal records.
  The default `bearing_only` profile retains the baseline definition above.
  Do not pool these configurations into the old pair registry or formal SR.

## Canonical locked operation

### Protected experiment registry

- Read `runtime/go2/experiment_pairs/index.json` before collecting, labeling,
  counting, or cleaning experiments. The user designated `pair_001` as the
  first and currently sole valid completed pair on 2026-09-06 (local time).
  Start subsequent collection as a new pair; do not overwrite or recategorize
  the protected pair based on old GUI Episode status.
- Pair 001: Episode `episode_20260905T175456_573243Z`; CEC capture
  `episode_20260905T175456_573243Z_retry_202653Z` is **human success**;
  Baseline (Mono-native) capture
  `episode_20260905T175456_573243Z_baseline_204358Z` is **human failure**.
  `runtime/go2/experiment_pairs/pair_001/pair.json` binds the two captures,
  goal, shared Survey, runtime contracts, and retained GPU dependencies.
- Protect all dependencies named in that registry, not only the two MCAPs.
  The CEC recording was tail-trimmed at the user's request; preserve this
  provenance. User-valid pair status does not mean independent GT validation
  or automatic arrival, and a failed Baseline does not invalidate the pair.
- The user authorized deleting earlier test data during the pair-001 cleanup.
  This is not standing permission to delete future experiments. Consult the
  registry and obtain new scope for any later deletion.

Before starting the stack, check `rs-enumerate-devices`, USB SuperSpeed, and
`ping 192.168.123.161`. These are automated checks, not conversational approval
steps. Immediately after a locked start, verify `/navdp/status` reports
`enabled:false` and `estop:true` before proceeding. An offline Go2 may be
observed in camera-only mode but cannot begin navigation.

The following commands apply only to the separate Jetson-local RGB-D diagnostic
profile, when explicitly requested; they are not the Baseline launch sequence:

```bash
source /opt/ros/humble/setup.bash
bash deployment/go2/nav_stack.sh start --config deployment/config/experiments/native_imagegoal.json --refresh
bash deployment/go2/nav_stack.sh status --config deployment/config/experiments/native_imagegoal.json
bash deployment/go2/nav_stack.sh stop --config deployment/config/experiments/native_imagegoal.json
```

Use the active episode's documented offboard workflow and resolved contract
for Revisit; the local RGB-D example above is not a replacement for that workflow.
