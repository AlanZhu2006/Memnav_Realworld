#!/usr/bin/env bash
set -euo pipefail

# Engineering M-point workflow:
#   record-start -> manual Unitree hand-controller drive -> record-stop
#   -> later revisit-prepare -> user-requested motion (no repeated confirmation).
# Survey motion is always policy-locked and the live matcher is observation-only.

OFFBOARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO2_DIR="$(cd "$OFFBOARD_DIR/.." && pwd)"
REPO_ROOT="$(cd "$GO2_DIR/../.." && pwd)"
CONFIG_TOOL="$REPO_ROOT/deployment/runtime_config.py"
NAV_STACK="$GO2_DIR/nav_stack.sh"
REVISIT="$OFFBOARD_DIR/revisit_experiment.sh"
MONITOR_RUNNER="$GO2_DIR/scripts/run_revisit_goal_monitor.sh"
BASE_EXPERIMENT="$REPO_ROOT/deployment/config/experiments/fullmono_imagegoal.json"
NATIVE_EXPERIMENT="$REPO_ROOT/deployment/config/experiments/native_imagegoal.json"
DEBUG_ROOT="$REPO_ROOT/runtime/go2/revisit_debug"
ACTIVE_STATE="$DEBUG_ROOT/active.json"

usage() {
  cat <<'EOF'
Usage:
  revisit_debug.sh record-prepare DATASET_ID --goal FROZEN_M_IMAGE [--point-label M]
  revisit_debug.sh record-start DATASET_ID --goal FROZEN_M_IMAGE [--point-label M]
  revisit_debug.sh status
  revisit_debug.sh record-stop
  revisit_debug.sh revisit-prepare [--run-id RUN_ID]
  revisit_debug.sh stop
  revisit_debug.sh park

record-prepare:
  Prepares the same one-way Survey but leaves recording PAUSED at zero frames.
  Start it later with Foxglove's START SURVEY button so the first saved frame
  has an explicit operator boundary.

record-start:
  Starts MemNav + LingBot + CEC in an engineering one-way Survey. Autonomous
  motion remains disabled and the Go2 command bridge is absent; drive only with
  the Unitree hand controller. Foxglove continuously shows M-vs-live MATCH.

record-stop:
  Requires the persistent dataset seal gates (normally >=40 memory frames and
  exactly zero Survey candidates because M is external), then locks and stops
  the motion stack, keeping GPU model weights resident with episode state reset.
  If a seal gate is not ready, the stack stays locked and recording so history
  is not accidentally discarded.

revisit-prepare:
  Accepts a validated STOP SURVEY receipt directly from Foxglove, rebinds the
  hub and motion-locked client, verifies and replays the sealed history, installs the exact
  frozen M image under mono_cec authority, starts the Go2 watchdog bridge, and
  leaves motion disabled + estop. It never starts physical motion.
EOF
}

die() { echo "revisit-debug: $*" >&2; exit 1; }

validate_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] \
    || die "invalid id: $1"
}

state_value() {
  local key="$1"
  [[ -f "$ACTIVE_STATE" ]] || die "no active Revisit debug state: $ACTIVE_STATE"
  python3 - "$ACTIVE_STATE" "$key" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get(sys.argv[2])
if value is None:
    raise SystemExit(1)
print(value)
PY
}

write_start_state() {
  local dataset_id="$1" goal="$2" goal_sha="$3" point_label="$4" experiment="$5"
  local mode="${6:-recording}"
  mkdir -p "$DEBUG_ROOT"
  python3 - "$ACTIVE_STATE" "$dataset_id" "$goal" "$goal_sha" \
      "$point_label" "$experiment" "$mode" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = {
    "schema": "memnav_revisit_debug_state_v1",
    "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "updated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "mode": sys.argv[7],
    "dataset_id": sys.argv[2],
    "goal_path": sys.argv[3],
    "goal_sha256": sys.argv[4],
    "point_label": sys.argv[5],
    "experiment_path": sys.argv[6],
    "collection_mode": "manual_one_way_external_goal_debug",
    "candidate_capture": "disabled_external_frozen_goal_required",
    "motion_authority": "unitree_hand_controller_only",
    "dataset_manifest_sha256": None,
    "run_id": None,
}
temporary = path.with_name("." + path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

update_state() {
  local mode="$1" manifest_sha="${2:-}" run_id="${3:-}"
  python3 - "$ACTIVE_STATE" "$mode" "$manifest_sha" "$run_id" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["mode"] = sys.argv[2]
if sys.argv[3]:
    payload["dataset_manifest_sha256"] = sys.argv[3]
if sys.argv[4]:
    payload["run_id"] = sys.argv[4]
payload["updated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
temporary = path.with_name("." + path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

make_debug_experiment() {
  local dataset_id="$1" goal="$2" output="$3"
  mkdir -p "$(dirname "$output")"
  python3 - "$BASE_EXPERIMENT" "$goal" "$dataset_id" "$output" <<'PY'
import json
from pathlib import Path
import sys

base = Path(sys.argv[1]).resolve()
goal = Path(sys.argv[2]).resolve()
dataset_id = sys.argv[3]
output = Path(sys.argv[4])
payload = json.loads(base.read_text(encoding="utf-8"))
system = Path(payload["system_config"])
if not system.is_absolute():
    system = (base.parent / system).resolve()
payload["system_config"] = str(system)
experiment = payload["experiment"]
experiment["id"] = f"revisit-debug-{dataset_id}"
experiment["profile"] = "fullmono-lingbot-cec"
experiment["authority_mode"] = "cec"
experiment["navigation"]["image_goal"] = str(goal)
experiment["navigation"]["revisit_image_goal"] = None
experiment["arrival"]["module"] = "rgb-homography"
experiment["arrival"]["image_goal"] = str(goal)
experiment["arrival"]["allowed_phases"] = ["memory_recording", "revisit_query"]
experiment["launch"] = {"camera": True, "go2_bridge": False, "foxglove": True}
experiment["control"]["profile"] = "formal"
temporary = output.with_name("." + output.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
PY
}

stop_native_if_running() {
  if tmux has-session -t navdp-go2 2>/dev/null; then
    echo "Locking and stopping the native NavDP stack before Full-Mono startup..."
    bash "$NAV_STACK" stop --config "$NATIVE_EXPERIMENT"
  fi
}

launch_match_monitor() {
  local resolved="$1" goal="$2" point_label="$3"
  local session log command expected_goal_sha waiter_pid monitor_ready=false
  session="$(python3 "$CONFIG_TOOL" get --config "$resolved" sites.jetson.sessions.fullmono)"
  log="$DEBUG_ROOT/$(state_value dataset_id)/revisit_match_monitor.log"
  expected_goal_sha="$(state_value goal_sha256)"
  mkdir -p "$(dirname "$log")"
  tmux kill-window -t "$session:arrival" 2>/dev/null || true

  # ROS Humble's generated setup references variables that may be unset. Keep
  # the outer script strict, but do not let nounset abort environment loading.
  set +u
  source /opt/ros/humble/setup.bash
  set -u

  # Keep one typed subscriber alive across monitor startup. Repeated short-lived
  # `ros2 topic echo` processes can continually repay DDS discovery latency on
  # Jetson and miss the transient status sample even though the monitor is live.
  timeout 60 python3 - "$expected_goal_sha" >/dev/null 2>&1 <<'PY' &
import json
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

expected_goal_sha = sys.argv[1]
rclpy.init(args=[])
node = Node("memnav_revisit_monitor_waiter")
ready = False


def on_status(message):
    global ready
    try:
        payload = json.loads(message.data)
    except (TypeError, ValueError):
        return
    ready = (
        payload.get("schema") == "memnav_revisit_goal_monitor_v1"
        and payload.get("authority") == "observation_only"
        and payload.get("goal_sha256") == expected_goal_sha
    )


qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
subscription = node.create_subscription(
    String, "/navdp/rgb_arrival_status", on_status, qos
)
while rclpy.ok() and not ready:
    rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_subscription(subscription)
node.destroy_node()
rclpy.shutdown()
raise SystemExit(0 if ready else 1)
PY
  waiter_pid=$!

  printf -v command 'exec %q --config %q --goal %q --point-label %q >>%q 2>&1' \
    "$MONITOR_RUNNER" "$resolved" "$goal" "$point_label" "$log"
  tmux new-window -d -t "$session:" -n m-match "$command"

  while kill -0 "$waiter_pid" 2>/dev/null; do
    if ! tmux list-windows -t "$session" -F '#{window_name}' 2>/dev/null \
        | grep -Fxq m-match; then
      kill -TERM "$waiter_pid" 2>/dev/null || true
      wait "$waiter_pid" 2>/dev/null || true
      break
    fi
    sleep 0.25
  done
  if wait "$waiter_pid" 2>/dev/null; then
    monitor_ready=true
  fi
  [[ "$monitor_ready" == true ]] \
    || die "observation-only M match monitor did not become ready; see $log"
}

record_begin() {
  local start_immediately="$1"
  shift
  local action="record-start"
  [[ "$start_immediately" == true ]] || action="record-prepare"
  [[ $# -ge 1 ]] || die "$action requires DATASET_ID"
  local dataset_id="$1" goal="" point_label="M"
  shift
  validate_id "$dataset_id"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --goal) [[ $# -ge 2 ]] || die "--goal requires a path"; goal="$2"; shift 2 ;;
      --point-label) [[ $# -ge 2 ]] || die "--point-label requires a value"; point_label="$2"; shift 2 ;;
      *) die "unknown record-start option: $1" ;;
    esac
  done
  [[ -n "$goal" ]] || die "record-start requires --goal FROZEN_M_IMAGE"
  [[ -f "$goal" ]] || die "goal image does not exist: $goal"
  goal="$(readlink -f "$goal")"
  local goal_sha experiment_dir experiment survey_config
  goal_sha="$(sha256sum "$goal" | awk '{print $1}')"
  experiment_dir="$DEBUG_ROOT/$dataset_id"
  experiment="$experiment_dir/debug_experiment.json"
  survey_config="$REPO_ROOT/runtime/go2/two_pass_revisit/$dataset_id/survey_config.json"
  [[ ! -e "$experiment_dir" ]] \
    || die "debug dataset state already exists: $experiment_dir"
  ! tmux has-session -t navdp-go2-offboard 2>/dev/null \
    || die "Full-Mono stack is already running; stop or finish it first"

  make_debug_experiment "$dataset_id" "$goal" "$experiment"
  stop_native_if_running
  local state_mode="prepared"
  if [[ "$start_immediately" == true ]]; then
    bash "$REVISIT" --config "$experiment" survey-start "$dataset_id" \
      --collection-mode manual_one_way_external_goal_debug
    state_mode="recording"
  else
    bash "$REVISIT" --config "$experiment" survey-prepare "$dataset_id" \
      --collection-mode manual_one_way_external_goal_debug
  fi
  write_start_state "$dataset_id" "$goal" "$goal_sha" "$point_label" \
    "$experiment" "$state_mode"

  launch_match_monitor "$survey_config" "$goal" "$point_label"

  echo
  if [[ "$start_immediately" == true ]]; then
    echo "M-point history recording is ACTIVE."
  else
    echo "M-point history recording is PREPARED and PAUSED."
  fi
  echo "  dataset:       $dataset_id"
  echo "  frozen goal:   $goal"
  echo "  goal sha256:   $goal_sha"
  echo "  policy input:  causal RGB -> LingBot/MemNav/CEC on RTX"
  echo "  robot control: Unitree hand controller only (Go2 bridge absent)"
  echo "  Foxglove:      goal panel=M; match panel updates continuously"
  [[ "$start_immediately" == true ]] \
    || echo "  start:          click START SURVEY in Foxglove"
  echo "  finish:        $0 record-stop"
}

record_prepare() { record_begin false "$@"; }

record_start() { record_begin true "$@"; }

show_status() {
  local mode experiment
  mode="$(state_value mode)"
  experiment="$(state_value experiment_path)"
  echo "Debug state: mode=$mode dataset=$(state_value dataset_id) point=$(state_value point_label)"
  echo "Goal SHA-256: $(state_value goal_sha256)"
  case "$mode" in
    prepared|recording)
      bash "$REVISIT" --config "$experiment" survey-status
      echo
      echo "Live M match (observation-only):"
      set +u
      source /opt/ros/humble/setup.bash
      set -u
      timeout 5 ros2 topic echo --once /navdp/rgb_arrival_status --field data || true
      ;;
    sealed|formal_ready)
      [[ "$mode" != formal_ready ]] \
        || bash "$REVISIT" --config "$experiment" formal-status
      python3 -m json.tool "$ACTIVE_STATE"
      ;;
    *) python3 -m json.tool "$ACTIVE_STATE" ;;
  esac
}

record_stop() {
  local mode dataset_id experiment receipt manifest_sha
  mode="$(state_value mode)"
  [[ "$mode" == prepared || "$mode" == recording ]] \
    || die "record-stop requires mode=prepared|recording, got $mode"
  dataset_id="$(state_value dataset_id)"
  experiment="$(state_value experiment_path)"

  local status_payload
  status_payload="$(curl -fsS --max-time 5 http://127.0.0.1:18889/dataset/status)"
  python3 - "$status_payload" "$dataset_id" <<'PY' >/dev/null
import json, sys
p = json.loads(sys.argv[1])
assert p["recording"] is True and p["dataset_id"] == sys.argv[2]
assert int(p["memory_frames"]) >= int(p["minimum_frames"])
PY
  echo "Sealing persistent history before stopping either machine..."
  # Reassert the one-way operator lock first.  The adapter's survey_seal
  # service then pauses and drains the current RGB transaction, atomically
  # seals the dataset, and writes the fail-closed receipt used by Revisit.
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  timeout 8 ros2 service call /navdp_go2_adapter/operator_stop \
    std_srvs/srv/Trigger '{}' >/dev/null 2>&1 || true
  if ! bash "$REVISIT" --config "$experiment" survey-seal "$dataset_id"; then
    echo >&2
    echo "revisit-debug: seal is not ready; the stack remains motion-locked and recording." >&2
    echo "Run '$0 status', continue the manual route if needed, then retry record-stop." >&2
    exit 1
  fi
  receipt="$REPO_ROOT/runtime/go2/two_pass_revisit/$dataset_id/survey_seal.json"
  manifest_sha="$(python3 - "$receipt" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text())["manifest_sha256"])
PY
)"
  update_state sealed "$manifest_sha"
  if ! bash "$REVISIT" --config "$experiment" park; then
    # The seal is already durable. A cleanup failure must never make the GUI
    # resume recording into that immutable dataset as if the seal had failed.
    echo "Warning: dataset sealed, but GPU parking failed; inspect cleanup before the next start." >&2
  fi
  echo
  echo "Persistent history sealed; GPU residency/cleanup result is reported above."
  echo "  dataset:        $dataset_id"
  echo "  manifest sha:   $manifest_sha"
  echo "  next:            $0 revisit-prepare"
}

adopt_foxglove_seal() {
  local mode dataset_id receipt manifest_sha
  mode="$(state_value mode)"
  [[ "$mode" == sealed ]] && return 0
  [[ "$mode" == prepared || "$mode" == recording ]] \
    || die "revisit-prepare requires mode=sealed, got $mode"

  dataset_id="$(state_value dataset_id)"
  receipt="$REPO_ROOT/runtime/go2/two_pass_revisit/$dataset_id/survey_seal.json"
  [[ -f "$receipt" ]] || die \
    "STOP SURVEY has not produced a seal receipt for dataset $dataset_id"
  manifest_sha="$(python3 - "$receipt" "$dataset_id" <<'PY'
import json
from pathlib import Path
import re
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload.get("dataset_id") == sys.argv[2]
assert payload.get("recording_active") is False
assert payload.get("motion_enabled") is False
assert payload.get("estop") is True
assert payload.get("evaluation_depth_consumed_by_policy") is False
assert int(payload.get("goal_memory_exact_sha_overlap", -1)) == 0
digest = payload.get("manifest_sha256")
assert isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
print(digest)
PY
)" || die "STOP SURVEY receipt failed its fail-closed contract"
  update_state sealed "$manifest_sha"
  echo "Adopted validated Foxglove STOP SURVEY receipt for $dataset_id."
}

revisit_prepare() {
  local run_id=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-id) [[ $# -ge 2 ]] || die "--run-id requires a value"; run_id="$2"; shift 2 ;;
      *) die "unknown revisit-prepare option: $1" ;;
    esac
  done
  local mode dataset_id experiment goal goal_sha dataset_sha scene_id
  adopt_foxglove_seal
  mode="$(state_value mode)"
  [[ "$mode" == sealed ]] || die "revisit-prepare requires mode=sealed, got $mode"
  dataset_id="$(state_value dataset_id)"
  experiment="$(state_value experiment_path)"
  goal="$(state_value goal_path)"
  goal_sha="$(state_value goal_sha256)"
  dataset_sha="$(state_value dataset_manifest_sha256)"
  [[ -n "$run_id" ]] || run_id="${dataset_id}_cec_$(date -u +%Y%m%dT%H%M%SZ)"
  validate_id "$run_id"
  scene_id="debug_${dataset_id}"
  validate_id "$scene_id"
  stop_native_if_running

  bash "$REVISIT" --config "$experiment" formal-start "$dataset_id" \
    --scene-id "$scene_id" --run-id "$run_id" --arm mono_cec \
    --engineering-unregistered \
    --goal "$goal" --expected-goal-sha256 "$goal_sha" \
    --expected-dataset-sha256 "$dataset_sha"
  update_state formal_ready "$dataset_sha" "$run_id"
  echo
  echo "M-point Revisit is formal-ready and MOTION-LOCKED."
  echo "Preparation never arms the robot. After automated preflight, an existing"
  echo "user request to start Revisit is sufficient; no second confirmation is needed."
}

stop_debug() {
  local experiment
  experiment="$(state_value experiment_path)"
  bash "$REVISIT" --config "$experiment" "${1:-stop}"
  echo "Revisit motion stack stopped; sealed dataset files were not deleted."
}

command="${1:-}"
[[ $# -eq 0 ]] || shift
case "$command" in
  record-prepare) record_prepare "$@" ;;
  record-start) record_start "$@" ;;
  status) [[ $# -eq 0 ]] || die "status takes no arguments"; show_status ;;
  record-stop) [[ $# -eq 0 ]] || die "record-stop takes no arguments"; record_stop ;;
  revisit-prepare) revisit_prepare "$@" ;;
  stop) [[ $# -eq 0 ]] || die "stop takes no arguments"; stop_debug ;;
  park) [[ $# -eq 0 ]] || die "park takes no arguments"; stop_debug park ;;
  -h|--help|help|"") usage ;;
  *) die "unknown command: $command" ;;
esac
