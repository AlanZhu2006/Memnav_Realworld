#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
gpu_require_config "$@"
GO2_DIR="$REPO_ROOT/deployment/go2"
source "$GO2_DIR/offboard/runtime_contract.sh"
SESSION="$CFG_GPU_SESSION"
require_executable tmux
mkdir -p "$CEC_OUT_ROOT/logs" "$CEC_OUT_ROOT/buffer"
exec 9>"$CEC_OUT_ROOT/policy_lifecycle.lock"
flock -w 150 9
signature="$(python3 "$GPU_DIR/resident_policy.py" signature --config "$RUN_CONFIG")"
warm=false

if tmux has-session -t "$SESSION" 2>/dev/null; then
  [[ "$(tmux show-environment -t "$SESSION" MEMNAV_RESIDENT_MANAGER 2>/dev/null)" == MEMNAV_RESIDENT_MANAGER=v1 ]] \
    && [[ "$(tmux show-environment -t "$SESSION" MEMNAV_RESIDENT_STATE 2>/dev/null)" == MEMNAV_RESIDENT_STATE=parked ]] \
    || { echo "GPU session is active or unmanaged; refusing to replace it" >&2; exit 1; }
  if [[ "$(tmux show-environment -t "$SESSION" MEMNAV_MODEL_SIGNATURE)" == "MEMNAV_MODEL_SIGNATURE=$signature" ]] \
      && python3 "$GPU_DIR/resident_policy.py" verify-idle --config "$RUN_CONFIG"; then
    warm=true
  else
    echo "Parked model contract changed or service unhealthy; cold-starting managed models."
    tmux kill-session -t "$SESSION"
  fi
fi
if [[ "$warm" == true ]] && ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$CEC_HUB_PORT$"; then
  echo "Hub port is occupied while resident models are parked; refusing to bind" >&2
  exit 1
fi
if [[ "$warm" == false ]]; then
  bash "$SCRIPT_DIR/preflight.sh" --config "$RUN_CONFIG"
  # A new tmux daemon otherwise inherits FD 9 and holds the lifecycle flock
  # forever after this launcher exits. Only the launcher may own that lock.
  tmux new-session -d -s "$SESSION" -n memnav \
    "exec '$SCRIPT_DIR/run_memnav_server.sh' --config '$RUN_CONFIG' >'$CEC_OUT_ROOT/logs/memnav.log' 2>&1" 9>&-
fi
# From here on this launcher owns the session; an incomplete bind cannot be reused.
trap 'tmux kill-session -t "$SESSION" 2>/dev/null || true' ERR
tmux set-environment -t "$SESSION" MEMNAV_RESIDENT_MANAGER v1
tmux set-environment -t "$SESSION" MEMNAV_RESIDENT_STATE starting
tmux set-environment -t "$SESSION" MEMNAV_MODEL_SIGNATURE "$signature"
if [[ "$warm" == false ]]; then
  tmux new-window -t "$SESSION" -n navdp \
    "exec '$SCRIPT_DIR/run_navdp_server.sh' --config '$RUN_CONFIG' >'$CEC_OUT_ROOT/logs/navdp.log' 2>&1"
fi
tmux new-window -t "$SESSION" -n hub \
  "exec '$SCRIPT_DIR/run_cec_hub.sh' --config '$RUN_CONFIG' >'$CEC_OUT_ROOT/logs/hub.log' 2>&1"
tmux set-environment -t "$SESSION" MEMNAV_RUN_CONFIG "$RUN_CONFIG"
tmux set-environment -t "$SESSION" MEMNAV_CONFIG_ID "$CFG_CONFIG_ID"

ready=false
for _ in $(seq 1 "$CFG_GPU_READY_TIMEOUT_S"); do
  health="$(curl -fsS --max-time 1 \
      "http://127.0.0.1:$CEC_HUB_PORT/healthz" 2>/dev/null || true)"
  if [[ -n "$health" ]] \
      && cec_validate_health_contract "$health" "$GO2_DIR" \
      && ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$MEMNAV_PORT$" \
      && ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$NAVDP_PORT$"; then
    ready=true
    break
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "CEC policy stack failed to become ready" >&2
  for log in "$CEC_OUT_ROOT"/logs/*.log; do
    echo "===== $log" >&2
    tail -n 100 "$log" >&2 || true
  done
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  exit 1
fi
tmux set-environment -t "$SESSION" MEMNAV_RESIDENT_STATE active
trap - ERR

echo "CEC real-world policy stack ready"
echo "  resident weight reuse: $warm"
echo "  sensor: causal monocular RGB (client depth is local safety only)"
echo "  config: $RUN_CONFIG"
echo "  config_id: $CFG_CONFIG_ID"
echo "  camera optical-center height: ${CFG_CAMERA_HEIGHT_M} m"
echo "  hub:    http://127.0.0.1:$CEC_HUB_PORT"
echo "  memnav: http://127.0.0.1:$MEMNAV_PORT"
echo "  navdp:  http://127.0.0.1:$NAVDP_PORT"
echo "  logs:   $CEC_OUT_ROOT/logs"
echo "  tmux:   tmux attach -t $SESSION"
