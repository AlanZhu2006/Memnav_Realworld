# MemNav Real-World

Real-world monocular ImageGoal and revisit navigation on a Unitree Go2. An RTX
4090 workstation runs one causal RGB stream through LingBot-backed MemNav,
Certified Episodic Compass (CEC), and frozen NavDP. The Jetson Orin NX keeps
aligned depth only inside the local collision-safety layer, together with
trajectory tracking, motor safety and final stop authority on the robot.

## Experiment Handbook

The complete Chinese field protocol, from scene registration and sealed Survey
through supervised motion, dual-view evidence, SR/SPL and failure handling, is
maintained in
[REALWORLD_EXPERIMENT_HANDBOOK_CN.md](REALWORLD_EXPERIMENT_HANDBOOK_CN.md).
Use it as the primary experiment and handoff entry point, together with the
latest claim boundary in [CURRENT_STATUS.md](CURRENT_STATUS.md).

The **un-deployed 2026-09-07 adaptation branch** is documented in
[the real-world / simulation audit](REALWORLD_RECEDING_HORIZON_AUDIT_20260907.md).
It restores rolling replanning and query-time geometry updates, and provides
an explicit shared local-approach experiment. Offline receipt reassessment is
not a new navigation result; existing robot runs remain unchanged.

## Runtime Entry Points

For an ad-hoc baseline or Full-Mono run, `nav_stack.sh` provides one profile
and arrival vocabulary:

~~~bash
bash deployment/go2/nav_stack.sh list
bash deployment/go2/nav_stack.sh describe native-navdp-rgbd
bash deployment/go2/nav_stack.sh describe fullmono-lingbot-cec
bash deployment/go2/nav_stack.sh status
~~~

The facade accepts one tracked experiment JSON. It resolves that file together
with `deployment/config/system.json` into one hash-verified runtime contract;
native and Full-Mono launchers receive only that file. Full-Mono copies the
same bytes to the RTX and verifies both `config_id` and Git revision. There is
no GPU `.env` or environment-variable override layer.

## Reference Platform

<p align="center">
  <img src="media/go2_showcase.jpg" width="720" alt="Unitree Go2 with a front-facing Intel RealSense D435i and Jetson Orin NX">
</p>

| Role | Compute | Platform / sensor | Responsibility |
| --- | --- | --- | --- |
| Policy workstation | NVIDIA RTX 4090 | Ubuntu workstation | One causal RGB stream, LingBot dense mono-depth readout, CEC proof, frozen NavDP |
| Robot computer | Jetson Orin NX 16 GB | Unitree Go2 + RealSense D435i | RGB transport, local aligned-depth collision guard, trajectory tracking, watchdog and Unitree control |

This deployment does **not** use TinyNav VIO, mapping or planning. The working
TinyNav Python environment may only be reused for CycloneDDS and Unitree SDK
packages on the Jetson.

## System Architecture

<p align="center">
  <a href="media/system_architecture.png">
    <img src="media/system_architecture.png" width="100%" alt="One causal RGB stream with LingBot short-range geometry, CEC long-range memory, frozen NavDP control and Jetson-local safety">
  </a>
</p>

The workstation never publishes velocity and does not load the Unitree SDK.
It returns one 24-point robot-local trajectory through a loopback-only service
reached by an SSH tunnel. The Jetson converts that path into
<code>/navdp/cmd_vel</code> only after its own RGB/depth freshness,
depth-clearance, command-age, estop and operator-enable checks pass. Jetson
depth is never forwarded into the navigation policy.

## Real-Robot Demo

The clips below are legacy engineering reference footage supplied on
2026-08-27, not a formal SR/SPL result. The external view shows physical Go2
motion; the first-person dashboard shows the ImageGoal, current RGB, aligned safety depth, visual match,
candidate trajectories, selected trajectory and live control state.

<table width="100%">
  <tr>
    <td width="35%" align="center" valign="top">
      <strong>Third-person view</strong><br>
      <a href="media/demo/revisit_reference_third_view.mp4">
        <img src="media/demo/revisit_reference_third_view.gif" width="65%" alt="Third-person Unitree Go2 engineering demo">
      </a><br>
      <a href="media/demo/revisit_reference_third_view.mp4">H.264 MP4</a>
    </td>
    <td width="65%" align="center" valign="top">
      <strong>First-person dashboard</strong><br>
      <a href="media/demo/revisit_reference_dashboard.mp4">
        <img src="media/demo/revisit_reference_dashboard.gif" width="100%" alt="NavDP first-person dashboard engineering demo">
      </a><br>
      <a href="media/demo/revisit_reference_dashboard.mp4">H.264 MP4</a>
    </td>
  </tr>
</table>

Formal runs use one run ID to bind the MCAP rosbag, readable CEC/status
receipts, imported Foxglove recording and external third-view master into a
SHA-256 manifest. See
[EXPERIMENT_DATA_COLLECTION.md](EXPERIMENT_DATA_COLLECTION.md).

### Online ImageGoal Route

1. A survey pass records exact causal RGB memory and memory-excluded supported goal candidates.
2. The sealed dataset is restarted and verified before the formal Revisit pass.
3. The task-boundary transaction selects and installs one candidate, then reconstructs NavDP's short observation FIFO.
4. Each query appends current RGB exactly once; LingBot exposes dense short-range mono depth and sparse long-range proof evidence from the same state.
5. CEC either certifies a scale-free revisit bearing or abstains.
6. An accepted bearing is normalized onto a frozen 2.5 m local PointGoal; rejection uses exact mono-native ImageGoal NavDP.
7. A failed causal stream update latches <code>reset_required</code>; it cannot silently fall back to metric depth.
8. The Jetson tracks the returned local path at a tested default limit of <code>0.30 m/s</code>.

MemNav is therefore a certified directional memory layer, not a metric global
planner. NavDP remains the sole local trajectory policy, but its observation
depth is reconstructed from the same causal monocular stream. See
[ARCHITECTURE.md](ARCHITECTURE.md) for state and failure semantics.

## Planned Real-World Evaluation

The registered conference campaign uses four frozen scenes and five matched
native/CEC blocks per scene: 20 pairs and 40 physical rollouts.  Ten blocks are
native-first and ten CEC-first.  No formal run has been entered; every dash is
an intentionally blank value, not zero.

| Method | Planned rollouts | Completed | Novel SR | Revisit SR | Overall SR | SPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen mono NavDP | `20` | `0` | `—` | `—` | `—` | `—` |
| Frozen mono NavDP + CEC | `20` | `0` | `—` | `—` | `—` | `—` |

`SPL_i = S_i L_i / max(L_i, P_i)`, where `L_i` is the independently
predeclared shortest feasible scene path and `P_i` is an independently
measured physical Go2 path. Results remain blocked until the separate arrival
and path-measurement contracts are frozen. See the
[paired scene registry](REALWORLD_EVALUATION.md) and
[machine-readable plan](manifests/realworld_paired_evaluation_plan_v2.json).

## Independent Odin1 Reference Lane

An evaluation-only Odin1 stack is included under
[`deployment/odin1_gt/`](deployment/odin1_gt/README_CN.md). It performs one
out-and-back mode-1 mapping survey, hash-binds the D435i goal image to an Odin
map pose, requires stable mode-2 `map -> odom` relocalization, integrates
independent odometry path length and computes frozen-grid A* SPL receipts.

Odin data never enters CEC, NavDP, D435i collision safety or Go2 control. The
honest claim is **independent reference SLAM**, not motion-capture-grade
metrological ground truth. The code is implemented and tested offline, but the
official `v0.14.0` native-Mode1 driver has been compiled locally. The current
release has not yet produced a hash-bound live/calibration receipt on this Go2,
so all formal SR/SPL fields remain blank. The old 0.13.1 cold-start patch is an
explicit legacy fallback only.

## Safety Contract

- Motion is locked at startup and requires an explicit ROS service call.
- RGB/depth timeout, excessive synchronization skew, stale trajectory or invalid local safety depth produces zero velocity.
- The Go2 bridge has an independent <code>0.35 s</code> watchdog and hand-controller priority.
- The policy service is loopback-only; the robot reaches it through an SSH local forward.
- Certificate rejection falls back exactly to mono-native NavDP. A causal stream failure or uncertain NavDP state requires a full reset.
- The RTX workstation has no direct actuator path; the Jetson retains final authority.

The software guards do not replace an onsite operator, a clear test area,
tethering for first motion, or the Unitree hand controller.

## Repository Layout

| Path | Contents |
| --- | --- |
| <code>deployment/go2/</code> | D435i, ROS 2 adapter, Foxglove, RGB arrival gate, Go2 bridge and tests |
| <code>deployment/go2/nav_stack.sh</code> | Unified profile launcher separating native NavDP, Full-Mono CEC/LingBot and arrival authority |
| <code>deployment/go2/offboard/</code> | Jetson-to-workstation SSH tunnel and dual-machine launcher |
| <code>deployment/go2/offboard/experiment_capture.sh</code> | ROS bag, receipt, Foxglove and third-view evidence binding for each run |
| <code>deployment/odin1_gt/</code> | Independent Odin mapping, relocalization, arrival, path and A* SPL evidence lane |
| <code>deployment/gpu/</code> | Auditable CEC router, fixed-bearing adapter, GPU launch scripts and tests |
| <code>baselines/navdp/</code> | Frozen NavDP plus audited mono-sidecar and state-safe inference interfaces |
| <code>baselines/x-navdp/</code> | Preserved upstream simulator source; not part of the real-world launcher |
| <code>REALWORLD_EXPERIMENT_HANDBOOK_CN.md</code> | Complete Chinese experiment, safety, evidence, metric and handoff protocol |
| <code>REALWORLD_EVALUATION.md</code> | Planned four-scene, five-paired-block native/CEC registry with empty result slots |
| <code>docs/</code> | Documentation index and archived historical integration/release records |

Model checkpoints, research datasets, local environments, runtime buffers and
raw experiment evidence are intentionally excluded. Curated engineering demo
derivatives are indexed in [media/README.md](media/README.md).

## Reproduction

### 1. Verify the Checkout

~~~bash
git clone git@github.com:AlanZhu2006/MemNav-RealWorld.git
cd MemNav-RealWorld

python3 tools/verify_public_baseline.py --workspace .
python3 -m pip install -r deployment/gpu/requirements.txt
python3 -m compileall -q deployment/go2 deployment/gpu deployment/odin1_gt
~~~

These static checks do not connect to the robot or issue motion commands.
The repository does not maintain a unit-test suite. Use syntax/import checks,
the documented preflight, and motion-locked runtime observation for validation;
these do not establish physical navigation performance. See [AGENTS.md](AGENTS.md)
for operation and authorization rules.

### 2. Configure Both Machines

Edit the tracked `deployment/config/system.json` for Jetson/RTX paths, model
artifacts, ports, camera and safety values. Edit or copy an experiment under
`deployment/config/experiments/` for the profile, ImageGoal, arrival module and
optional processes. Do not create a `.env` or export `NAVDP_*`/`CEC_*` values.

The configured RTX SSH alias is `work-pc`. Full-Mono startup uses it to sync
both source and the exact resolved configuration contract.

### 3. Prepare the Jetson

~~~bash
bash deployment/go2/scripts/download_weights.sh navdp
bash deployment/go2/scripts/setup_jetson.sh
~~~

### 4. Capture and Select an Image Goal

With the camera publishing and navigation stopped, move the robot to the goal
with the hand controller and capture RGB explicitly:

~~~bash
bash deployment/go2/scripts/capture_image_goal.sh \
  --output deployment/go2/goals/image_goal.png
~~~

Set that path in `experiment.navigation.image_goal`. Goal files stay in the
ignored `deployment/go2/goals/` runtime directory; the resolver records image
dimensions and SHA-256 so a later file replacement cannot be silent.

### 5. Resolve and Dry-Run

~~~bash
bash deployment/go2/nav_stack.sh start \
  --config deployment/config/experiments/fullmono_imagegoal.json \
  --dry-run
~~~

The output names the final ImageGoal, content hash, source revision and
`config_id`. No process is started in dry-run mode.

### 6. Supervised Run

For a supervised native RGB-D episode, the explicit motion command is:

~~~bash
bash deployment/go2/nav_stack.sh run
~~~

It reuses a healthy current stack and cold-starts only when the session is
absent, stale, incomplete, or unhealthy. A single fail-closed agent prints
relative phase timings, locks motion, resets the policy, validates one fresh
trajectory and the live goal view, arms, monitors arrival, and stops on any
error, interruption, or timeout. Use `nav_stack.sh start --config ...` when the
intent is to start services while retaining the motion lock. Repeating `start`
with the same healthy contract confirms `disabled + estop` and reuses the
running processes; add `--refresh` only when a deliberate cold replacement is
needed. The same normal `start` command is used when the Go2 is disconnected:
the camera, policy, adapter, previews, and Foxglove stay available and locked,
while the Go2 bridge waits in the background and Status reports `GO2 OFFLINE`.

The tracked Foxglove dashboard is also published as the organization layout
`MemNav Go2 Navigation` by `.github/workflows/sync-foxglove-layout.yml` whenever
its JSON changes on `main`. The workflow reads `FOXGLOVE_API_KEY` only from a
GitHub Actions secret and updates one stable layout ID in place, so operators
select the cloud layout once instead of repeatedly importing JSON.
The built-in Tab panel keeps the default `Operate` view terse while separate
`Planning` and `System` tabs expose candidate markers, command plots, safety
transitions, the ROS graph, and raw status messages without a custom extension.

Full-Mono formal runs remain separate and start locked:

~~~bash
bash deployment/go2/nav_stack.sh start \
  --config deployment/config/experiments/fullmono_imagegoal.json

# On the operator computer, connect Foxglove to ws://JETSON_IP:8765 and import
# deployment/go2/config/navdp_debug.foxglove-layout.json.

source /opt/ros/humble/setup.bash
ros2 service call /navdp_go2_adapter/set_enabled \
  std_srvs/srv/SetBool "{data: true}"
~~~

Full-Mono startup remains disabled and estopped; the service call is a separate
onsite operator decision. For formal Survey/Revisit, use
`deployment/go2/offboard/revisit_experiment.sh`, which derives immutable
survey/formal configs from the same base file.

Immediate stop:

~~~bash
ros2 topic pub --once /navdp/estop std_msgs/msg/Bool "{data: true}"
ros2 service call /navdp_go2_adapter/set_enabled \
  std_srvs/srv/SetBool "{data: false}"
bash deployment/go2/nav_stack.sh stop
~~~

## Current Status

As of **2026-08-29**, the repository contains the protocol-v3/bearing-v2
Full-Mono stack, immutable two-pass Revisit datasets, online goal installation,
persistent CEC receipts and the dual-view experiment collector described here.
The base Go2 tracker and safety chain have moved successfully. A tuned RGB-only
commissioning gate has also completed one powered near-goal automatic stop.
That result is not a full arbitrary-start rollout or a formal STOP contract;
formal Full-Mono SR/SPL remains unverified.

For the optional independent Odin1 reference workflow, including the complete
mapping-survey and formal-run command order, use
[`deployment/odin1_gt/README_CN.md`](deployment/odin1_gt/README_CN.md). Do not
set `--gt-source odin1` until its hardware preflight reports a stable
`reference_ready=true` lane.

See [CURRENT_STATUS.md](CURRENT_STATUS.md) before any new experiment.

## Documentation

Use the centralized [documentation index](docs/README.md). Start with
[CURRENT_STATUS.md](CURRENT_STATUS.md) before an experiment; dated release and
integration records live under `docs/archive/` and are not current runbooks.

## Upstream NavDP

This repository is built on
[InternRobotics/NavDP](https://github.com/InternRobotics/NavDP) and retains its
benchmark and baseline source history. Upstream code is distributed under the
terms stated by that project; X-NavDP and bundled third-party components keep
their own license files. This repository does not redistribute model weights.

If this project is useful, cite the upstream NavDP work:

~~~bibtex
@misc{navdp,
  title={NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance},
  author={Wenzhe Cai and Jiaqi Peng and Yuqiang Yang and Yujian Zhang and Meng Wei and Hanqing Wang and Yilun Chen and Tai Wang and Jiangmiao Pang},
  year={2025},
  booktitle={arXiv}
}
~~~
