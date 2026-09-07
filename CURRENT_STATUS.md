# Current Full-Mono Real-World Status

2026-09-07 实现更新：新的运行配置默认使用 `online_history`，为 CEC 复用观察时
保存的历史 LingBot 深度。保留当前帧 depth 复用、resident 模型管理及旧配置的
canonical 行为，不改控制／到达／急停；现有进程不热切换。本次是代码和配置更新，
不是新增真机成功结果。详见 [更新说明](CEC_ONLINE_HISTORY_UPDATE_20260907_CN.md)。

Snapshot: **2026-08-30, Foxglove + MCAP migration**

现场实验、交接、双机架构、两阶段数据、控制安全、证据采集和SR/SPL的统一操作入口为
`REALWORLD_EXPERIMENT_HANDBOOK_CN.md`。本文件继续作为最新claim boundary；若旧日期文档
与本文件冲突，以本文件和当前代码为准。

新增的两阶段真机框架已经把长程手柄 survey 冻结为 exact-byte episodic dataset，并能在
独立的第二次运行中校验、重放、安装目标和初始化 formal query。入口与剩余边界见
`TWO_PASS_REVISIT_RUNBOOK.md`。该更新解决实验生命周期和持久化，不改变下文
“正式 arrival/STOP 尚未建立”的结论。

实验采集侧车现在无界面地保存 ROS 2 MCAP 和 CEC/status JSONL。Foxglove dashboard与
第三人称原片从操作电脑/外部相机逐字节导入，再与同一 run ID、Git revision 和 SHA-256
manifest绑定；Jetson不再依赖VNC、X11或本地可视化进程。完整操作见
`EXPERIMENT_DATA_COLLECTION.md`。

Foxglove实时相机显示现在走独立的带宽受限预览侧车：RGB为640×360@15 Hz JPEG，深度为
200--4000 mm着色后的640×360@10 Hz JPEG，ImageGoal为2 Hz、arrival debug最多5 Hz。
原始图仍由NavDP、arrival和可选full MCAP直接消费，预览不是策略或安全输入；Bridge明确
屏蔽四路原始图以防旧布局绕过限流。已有Foxglove本地布局需重新导入版本化JSON后才会切到
新topic。

原 `4 scenes x 5 repeats = 20 CEC runs` 模板已归档。会议要求的 controlling v2 协议
改为 4 scenes × 5 matched native/CEC blocks：20 pairs、40 rollouts，方法顺序 10/10
平衡，并要求两条 Novel、两条 Revisit scene contracts。所有结果槽仍为空；该修复只冻结
公平比较设计，不代表已经执行实验，也不改变 arrival/STOP 和 SR/SPL 尚未建立的结论。
见 `REALWORLD_EVALUATION.md` 与
`manifests/realworld_paired_evaluation_plan_v2.json`。

2026-08-28 增加了完全隔离的Odin1参考评测栈：mode-1往返建图、D435i目标图与Odin
地图位姿绑定、mode-2 `map -> odom`重定位门、局部odom路径积分、融合到达证据和冻结
A* SPL收据。Odin不进入CEC/NavDP/Go2控制。当前Odin未连接，本轮没有硬件标定或正式
结果；正确命名是independent reference SLAM，不是计量级absolute GT。完整边界和命令见
`deployment/odin1_gt/README_CN.md`。

## Bottom line

The two-machine Full-Mono CEC stack is synchronized and fail-closed, and it
has completed real powered navigation trials.  A near-goal commissioning run
on 2026-08-29 triggered the RGB-only arrival latch and stopped the powered
robot automatically.  This is an engineering milestone, not a completed
arbitrary-start ImageGoal rollout or a formal STOP contract.  The remaining P0
is a separately validated, scale-free terminal visual-servo / arrival contract.

The robot is currently motion-locked (`disabled + estop`). The native stack,
camera and observation-first Foxglove Bridge may remain live for inspection.
Its only controls are a dedicated fail-closed STOP service and a camera-recovery
service that first locks motion and verifies fresh RGB-D after restart. Neither
can grant actuator authority, reset policy state or clear estop.

The paired campaign now has a fail-closed executable arm boundary.  The Jetson
formal entry point requires `--arm mono_native|mono_cec`, forwards the choice
to the RTX hub, verifies it from `/healthz`, and binds it into every plan
receipt.  The native arm still consumes the same causal-monocular depth stream
but skips both long-range certificate and direct-local bearing authority.  No
formal rollout has been executed under this new control.

`formal-start` now also closes the exact-goal startup gate.  Every registered
run must provide its scene ID, run ID, external frozen goal JPEG, goal SHA-256,
and sealed-dataset SHA-256.  The launcher verifies the source bytes before
startup, verifies the committed goal and dataset from RTX health after replay,
and writes a role-hidden `formal_ready.json`; it never receives a Novel/Revisit
label.  Automatic Survey-candidate selection remains an engineering path and
cannot silently enter the paired campaign.

The paired result boundary is also executable rather than manual.  The
outcome-blank preregistration remains immutable; the read-only
`tools/verify_realworld_paired_campaign.py` binds the 40 registered run IDs to
finalized capture manifests, rechecks Odin SPL receipts and explicit authority
modes, and derives paired statistics only when every run passes.  Its current
plan-only audit reports 40 structurally valid registered runs and, correctly,
zero verified outcomes.  The expanded pre-Formal audit exposes 65 explicit
freeze blockers: the campaign-level held-out arrival/registry seal plus every
scene role, dataset, goal, start, shortest path, budget, and artifact binding.
`tools/freeze_realworld_paired_campaign.py` now closes those blockers only from
verified field bytes and writes a new outcome-blank plan; it has no ROS, policy,
or motion path.  `formal-start` now also requires that plan and records its SHA;
the only unregistered route is explicitly marked engineering/debug and the
final verifier rejects it.  No field registry or arrival calibration has yet
been frozen.

Calibration capture now has a pre-observation physical-label contract.  A
`trial-kind=calibration` run cannot create its evidence directory unless the
held-out scene ID, independently measured distance, yaw, and measurement
method are supplied.  Capture schema v3 records that these labels precede this
run's arrival-score logger; it does not claim that an operator never viewed a
separate live display and does not mark calibration as passed.  This closes a
receipt-timing gap only; physical calibration data are still absent.

The opt-in RGB-only arrival gate added on 2026-08-28 is a commissioning aid,
not an established STOP contract.  Its first powered A -> D trial produced a
false negative and is recorded below.  After controlled threshold tuning and
offline replay, a powered near-D retry latched successfully.  That single
near-goal result does not establish route-level success or false-positive
performance.

## Current architecture and authority

- episode protocol: server-enforced v3
  `memory_recording -> prepare_revisit -> revisit_query`;
- policy observation: causal monocular RGB plus ImageGoal;
- short-range expert: frozen LingBot dense mono-depth readout into frozen
  NavDP's existing depth encoder;
- long-range expert: CEC history retrieval, LightGlue/PnP certificate and a
  scale-free bearing;
- direct-local expert: current-to-goal certified scale-free bearing;
- controller: frozen NavDP trajectory decoder; rearward direct bearings are
  executed only as a bounded Jetson atomic turn;
- D435i metric depth: Jetson collision safety only, never a policy input;
- certificate/proof loss: return to the preceding native or long-range route;
- metric PnP translation: diagnostic only, with no control or STOP authority;
- automatic STOP: disabled for formal operation until an independent
  convergence proof is calibrated and confirmed; an opt-in RGB-only
  commissioning gate has one successful powered near-goal latch.
- optional Odin1 lane: evaluation-only map/relocalization/path/arrival evidence;
  it has no policy, motion or estop authority.
- Foxglove operator UI: selected trajectory is the default 3D signal; verbose
  candidate/Q-value markers are opt-in, arrival comparison preserves its native
  wide aspect, and a read-only status card summarizes lock/freshness/clearance/
  command/error while raw JSON remains available for detailed diagnosis. The
  red STOP control can only disable + estop + command zero; the orange camera
  recovery control applies that same lock, restarts RGB-D and never resumes motion.

The terminal wire schema is
`cec_direct_bearing_handoff_v2_20260824`.  Both reset and launcher preflight
now compare the hub-advertised schema with the schema imported from the actual
Jetson executor source.  A partial file copy (v2 hub with v1 executor, or the
reverse) therefore refuses startup instead of silently changing motion
authority.

## Powered field evidence

| Trial | Observed result | Formal outcome |
| --- | --- | --- |
| Q -> R, CEC Revisit | moved 3.01 m; auxiliary distance 3.507 -> min 1.498 m; long-range and direct bearings became inconsistent near the goal | failure: `safety_abort_path_length_limit` |
| R -> Q, native Novel | 1.167 -> min 1.019 -> final 1.022 m; only 0.615 m path | failure: old controller/Go2 velocity-floor ordering caused left-right hunting; execution contract subsequently fixed |
| S -> Q, native Full-Mono after controller fix | first command 0.297 m/s forward; 1.226 -> min 0.993 -> final 3.729 m; 18.54 m path | failure: `operator_stop`; the robot passed the high-covisibility window without a valid arrival decision |
| A -> D, native Novel plus temporary RGB gate | the D board pair was visible around frames 47--48, then the approach turned right and the right board filled the image; local D435i clearance reached 0.418--0.430 m | failure: RGB gate false negative followed by correct `obstacle_stop`; later disabled/static/operator-moved frames contaminated the episode |
| Near D -> D, native Novel plus tuned RGB gate | detector confirmed a geometrically consistent D view; adapter latched arrival, disabled motion, asserted estop and held zero command | engineering success: powered automatic stop from a near-goal start; not a formal arbitrary-start rollout |

### 2026-08-28 temporary RGB-gate failure

The retained diagnostic episode is
`work-pc:/home/asus/Research/Memnav_Realworld/runtime/gpu/buffer/run_20260828T142614Z_4106982/ep_0002`
with 363 numeric RGB frames.  It is not a formal Novel memory and must not be
used for Revisit.  A full replay against the frozen D ImageGoal produced zero
strict matches.  The important boundary is:

- frames 29--37 saw the complete D board pair while it was still too small
  (`image_scale` approximately `0.33--0.44`);
- frames 47--48 were the only near-view candidates, with `56/59` good matches,
  `43/38` inliers, scale `0.897/0.874`, and current-view inlier coverage
  `0.050/0.045`;
- the temporary gate required 60 good matches, 45 inliers, coverage 0.12 and
  three consecutive matches, so both frames were rejected rather than
  producing a late or failed stop command;
- from frame 58 onward the right D board dominated the image.  The local
  metric-depth safety gate then correctly held zero velocity at
  `0.418--0.430 m` clearance;
- the later restart/reposition/static segment does not preserve a causal A ->
  D trace.  At final shutdown the detector remained unlatched.  Both Jetson
  and RTX tmux stacks were explicitly stopped.

The earlier failed-run replay used to choose the strict defaults had four
clean full-frame matches and would have confirmed at frame 240.  The new trial
shows that this calibration did not cover lateral/tilted approaches or a
partially visible target.

### 2026-08-29 powered near-goal commissioning success

The commissioning thresholds were changed to 45 good matches, 30 inliers,
0.45 inlier ratio, 0.07 minimum coverage, image scale 0.60--1.45, 16 degree
rotation, 4 px reprojection error and one required frame.  A 39-frame trimmed
offline replay only matched source frames 123--125.  This provided a narrow
positive window before powered testing rather than evidence of route-level
reliability.

At the near-D powered start, the preflight image did not match (`51` good
matches, `34` inliers, current coverage `0.0182`, image scale `0.3921`).  After
motion began, the detector confirmed D with `71` good matches, `44` inliers,
inlier ratio `0.6197`, target/current coverage `0.1687/0.0756`, center offset
`0.0202`, image scale `0.6735`, rotation `0.6353` degrees and reprojection error
`0.4549` px.  The adapter then set `enabled=false`, `estop=true`, latched
arrival and emitted zero velocity.

This proves the detector-to-adapter stop transport can act on a powered robot.
It does not yet prove a complete A -> D navigation, robustness to other
approaches, or a calibrated false-positive rate.  Before formal use, the rule
still needs negative-route replay and repeated full-route trials; a
target-region/multi-reference rule or dedicated fiducial remains a valid
alternative if the full-frame homography is unstable.

The controller repair restored the formal `0.30/0.55` limits and applies the
`8 deg` heading deadband before the Go2 `0.10/0.20` command floors.  It removes
the earlier hunting mechanism, but does not solve arrival.

## Why direct PnP cannot authorize STOP

On the S -> Q trace, frames 325--328 pass the full
LightGlue -> LingBot depth -> PnP -> certificate chain.  Its predicted metric
distances fall from `0.769` to `0.125 m`, while the independent evaluator says
the entire run never came closer than `0.993 m`.  The terminal metric scale
therefore underestimates by at least `7.9x` on this trace.  V2 keeps only the
certified direction and projects it to the frozen `2.5 m` residual.

A new read-only audit scanned all 431 recorded frames against the same goal:

- only 15 frames reached the already-frozen two-view certificate precheck;
- frame 326 was the strongest supported near-view, with 331 LightGlue
  matches, 299 fundamental inliers, query/reference hull coverage
  `0.712/0.398`, and normalized median identity flow `0.0613`;
- nevertheless, the run's true minimum distance remained `0.993 m`;
- low identity-flow values from other frames were often supported by only a
  few spurious matches, so view error must always be conditioned on proof;
- three additional disabled/static traces contain strong covisibility but no
  physical arrival labels and therefore cannot freeze a success threshold.

The collector is
`deployment/gpu/audit_visual_convergence.py`; immutable outputs are under
`Nav-graph-blind/.diagnostics/realworld_visual_convergence_20260825/`.
This is measurement-only evidence, not a deployed STOP policy.

## Causal goal-selection repair

The first automated goal lifecycle smoke selected online anchor 215 even
though candidate construction had frozen an eligible ceiling of 200.  That
transport smoke is not a causal Revisit result.  The hub now carries the
selected candidate's `eligible_anchor_ceiling` into every retrieval probe and
rejects an accepted candidate whose server receipt omits or widens that
ceiling.  Operator-supplied frozen ImageGoals remain the formal benchmark
route; automatic goal selection remains an optional lifelong demo.

## Verification and synchronization

- focused RTX/standalone regression: **71 passed**;
- Python compile, shell syntax and `git diff --check`: passed;
- Jetson targeted runtime regression after synchronization: **31 passed**;
- matching v2 schema accepted; stale v1 schema rejected;
- updated Jetson files match the workstation SHA-256 values;
- pre-sync Jetson files are recoverable from
  `.deployment_backups/20260825_bearing_v2_contract_pre_sync/`;
- one stale ROS process that only published `estop=true` was removed by exact
  PID; no navigation process remains.
- 2026-08-30 live fail-closed camera recovery was exercised after RGB-D became
  stale for about 212 seconds: the service verified 11 fresh RGB and 10 fresh
  aligned-depth frames, reduced `rgbd_age` to 0.027 s, and left
  `enabled=false`, `estop=true`, and the commanded velocity at zero.

Synchronization did not start the camera, ROS adapter, Go2 bridge or motors.

## What is established and what is not

Established:

1. the real camera/transport/Full-Mono transaction reaches the frozen policy;
2. the repaired Go2 controller produces forward motion without the previous
   left-right hunting mechanism;
3. CEC can recover a useful Revisit direction and direct two-view proof can
   refine it;
4. monocular PnP direction and metric distance require different authority;
5. stale or partially synchronized terminal schemas now fail closed.
6. formal-run evidence now has a single run ID and hash-bound ROS/dashboard/
   third-view collection contract.
7. an Odin1 reference implementation can bind `S_i/L_i/P_i/SPL_i` without
   exposing Odin observations to the navigation method.
8. the official Odin driver `v0.14.0` native-Mode1 profile is compiled in
   `/home/nvidia/twork/odin_ws` and passes its no-hardware source/dependency
   preflight.

Not established:

1. autonomous Novel or Revisit arrival and STOP;
2. a physically calibrated relationship from visual convergence to target
   success radius;
3. real-world SR/SPL or a statistically meaningful number of trials;
4. that automatic target selection satisfies the same causal contract as an
   externally frozen benchmark goal beyond the repaired software check.
5. Odin1 0.14 Mode1 has been reported normal in a prior operator test, but this
   release still lacks a hash-bound live USB/topic receipt, current-Go2
   mount/extrinsic calibration, relocalization repeatability, metric arrival
   thresholds and path-accuracy validation.

## Next safe experiment

Do not run another formal navigation trial. First connect Odin1 with the robot
disabled, use the installed native `v0.14.0` driver profile, verify live
USB/topics and measure the current mount,
obstacle height band and Go2 footprint. Then record one debug out-and-back map,
test mode-2 relocalization across restarts and collect the predeclared
`0, 0.25, 0.5, 1.0 m` by `0, +/-10, +/-20 deg` arrival offsets with independent
tape labels. Use one location to choose the combined metric/visual convergence
rule and different locations to test it.

Only after that gate passes should the terminal controller gain a two-stage
authority:

1. strong direct proof may enter a zero-translation visual-alignment hold;
2. persistent, independently calibrated convergence may authorize STOP.

Until then, real trials require an external evaluator/operator termination
and cannot be reported as autonomous ImageGoal success.
