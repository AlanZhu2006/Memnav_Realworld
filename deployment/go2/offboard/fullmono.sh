#!/usr/bin/env bash
set -euo pipefail

# Jetson-side transactional entry point for the two-machine Full-Mono stack.
# One immutable resolved JSON is copied byte-for-byte to the RTX. Startup never
# grants motion authority.

OFFBOARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO2_DIR="$(cd "$OFFBOARD_DIR/.." && pwd)"
source "$GO2_DIR/scripts/common.sh"
source "$OFFBOARD_DIR/runtime_contract.sh"

SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=8
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)

usage() {
  cat <<'EOF'
Usage:
  fullmono.sh start  --config RESOLVED_CONFIG.json
  fullmono.sh status --config RESOLVED_CONFIG.json
  fullmono.sh park   --config RESOLVED_CONFIG.json
  fullmono.sh stop   --config RESOLVED_CONFIG.json

This is an internal launcher. Normally use nav_stack.sh with a tracked
experiment JSON. No environment-variable configuration is accepted.
EOF
}

die() { echo "fullmono: $*" >&2; exit 1; }
shell_quote() { printf '%q' "$1"; }
remote_exec() { ssh "${SSH_OPTIONS[@]}" "$CFG_GPU_HOST" "$1"; }

load_config() {
  navdp_require_config_arg "$@"
  navdp_load_config "$NAVDP_RUN_CONFIG"
  [[ "$CFG_PROFILE" == fullmono-lingbot-cec ]] \
    || die "resolved config is not the Full-Mono profile"
  GPU_CONFIG="$CFG_GPU_REPO/runtime/config/$CFG_CONFIG_ID.json"
}

remote_session_exists() {
  remote_exec "tmux has-session -t $(shell_quote "$CFG_GPU_SESSION") 2>/dev/null"
}

remote_health() {
  remote_exec "curl -fsS --max-time 3 http://127.0.0.1:${CFG_HUB_PORT}/healthz"
}

validate_health() {
  local payload="$1"
  cec_validate_health_contract "$payload" "$GO2_DIR" "$CFG_AUTHORITY_MODE"
  python3 - "$payload" "$CFG_CAMERA_HEIGHT_M" <<'PY'
import json, math, sys
p = json.loads(sys.argv[1])
actual = float(p["camera_height_m"])
expected = float(sys.argv[2])
assert math.isfinite(actual) and abs(actual - expected) <= 1e-9
print(f"health=fullmono-v3-bearing-v2 camera_height_m={actual:.3f}")
PY
}

sync_config_to_gpu() {
  local remote_dir temporary
  remote_dir="$(dirname "$GPU_CONFIG")"
  temporary="$GPU_CONFIG.tmp.$$"
  remote_exec "mkdir -p $(shell_quote "$remote_dir")"
  scp -q "${SSH_OPTIONS[@]}" "$NAVDP_RUN_CONFIG" "$CFG_GPU_HOST:$temporary"
  remote_exec \
    "mv $(shell_quote "$temporary") $(shell_quote "$GPU_CONFIG") && cd $(shell_quote "$CFG_GPU_REPO") && python3 deployment/runtime_config.py verify --config $(shell_quote "$GPU_CONFIG") --site gpu"
}

assert_remote_session_config() {
  local remote_id
  remote_id="$(remote_exec "tmux show-environment -t $(shell_quote "$CFG_GPU_SESSION") MEMNAV_CONFIG_ID 2>/dev/null | sed -n 's/^MEMNAV_CONFIG_ID=//p'")"
  [[ "$remote_id" == "$CFG_CONFIG_ID" ]] \
    || die "RTX session uses config_id=${remote_id:-unknown}, requested $CFG_CONFIG_ID; stop it first"
}

start_stack() {
  load_config "$@"
  for cmd in ssh scp tmux curl python3; do command -v "$cmd" >/dev/null || die "missing command: $cmd"; done
  [[ -f "$CFG_IMAGE_GOAL" ]] || die "ImageGoal is missing: $CFG_IMAGE_GOAL"
  if tmux has-session -t "$CFG_FULLMONO_SESSION" 2>/dev/null; then
    die "Jetson session already exists: $CFG_FULLMONO_SESSION"
  fi
  ssh "${SSH_OPTIONS[@]}" "$CFG_GPU_HOST" true \
    || die "passwordless SSH failed: $CFG_GPU_HOST"
  remote_exec \
    "test -x $(shell_quote "$CFG_GPU_REPO/deployment/gpu/scripts/preflight.sh") && test -x $(shell_quote "$CFG_GPU_REPO/deployment/gpu/scripts/run_policy_stack.sh")" \
    || die "RTX source is missing under $CFG_GPU_REPO"
  sync_config_to_gpu

  local started_gpu=false start_complete=false
  rollback_partial_start() {
    local status=$?
    if [[ "$start_complete" != true ]]; then
      bash "$OFFBOARD_DIR/stop_offboard_stack.sh" --config "$NAVDP_RUN_CONFIG" >/dev/null 2>&1 || true
      if [[ "$started_gpu" == true ]]; then
        remote_exec "cd $(shell_quote "$CFG_GPU_REPO") && bash deployment/gpu/scripts/stop_policy_stack.sh --config $(shell_quote "$GPU_CONFIG")" >/dev/null 2>&1 || true
      fi
    fi
    return "$status"
  }
  trap rollback_partial_start EXIT

  local health
  local resident_state=""
  if remote_session_exists; then
    resident_state="$(remote_exec "tmux show-environment -t $(shell_quote "$CFG_GPU_SESSION") MEMNAV_RESIDENT_STATE 2>/dev/null | sed -n 's/^MEMNAV_RESIDENT_STATE=//p'")"
  fi
  if remote_session_exists && [[ "$resident_state" != parked ]]; then
    [[ -z "$resident_state" || "$resident_state" == active ]] \
      || die "RTX lifecycle is $resident_state; incomplete preparation must be stopped first"
    assert_remote_session_config
    health="$(remote_health)" || die "RTX session exists but hub is unhealthy"
    validate_health "$health" >/dev/null || die "RTX session advertises wrong policy contract"
    echo "Reusing healthy RTX policy session: $CFG_GPU_SESSION"
  else
    echo "Starting/rebinding RTX policy stack through $CFG_GPU_HOST ..."
    remote_exec \
      "cd $(shell_quote "$CFG_GPU_REPO") && bash deployment/gpu/scripts/run_policy_stack.sh --config $(shell_quote "$GPU_CONFIG")"
    started_gpu=true
    health="$(remote_health)" || die "RTX policy stack started but health is unavailable"
    validate_health "$health"
  fi

  bash "$OFFBOARD_DIR/run_offboard_stack.sh" --config "$NAVDP_RUN_CONFIG"
  bash "$OFFBOARD_DIR/preflight_offboard.sh" --config "$NAVDP_RUN_CONFIG"
  start_complete=true
  trap - EXIT
  echo
  echo "Full-Mono stack is ready."
  echo "  config_id:      $CFG_CONFIG_ID"
  echo "  config Jetson:  $NAVDP_RUN_CONFIG"
  echo "  config RTX:     $CFG_GPU_HOST:$GPU_CONFIG"
  echo "  RTX session:    $CFG_GPU_SESSION"
  echo "  Jetson session: $CFG_FULLMONO_SESSION"
  echo "  ImageGoal:      $CFG_IMAGE_GOAL"
  echo "  authority:      $CFG_AUTHORITY_MODE"
  echo "  Go2 bridge:     $CFG_WITH_GO2"
  echo "  Motion:         LOCKED"
}

status_stack() {
  load_config "$@"
  local failures=0
  if tmux has-session -t "$CFG_FULLMONO_SESSION" 2>/dev/null; then
    echo "Jetson session: RUNNING ($CFG_FULLMONO_SESSION)"
    tmux list-windows -t "$CFG_FULLMONO_SESSION" -F '  window=#{window_name} dead=#{pane_dead}'
  else
    echo "Jetson session: STOPPED ($CFG_FULLMONO_SESSION)"
    failures=$((failures + 1))
  fi
  if ssh "${SSH_OPTIONS[@]}" "$CFG_GPU_HOST" true 2>/dev/null && remote_session_exists; then
    local resident_state
    resident_state="$(remote_exec "tmux show-environment -t $(shell_quote "$CFG_GPU_SESSION") MEMNAV_RESIDENT_STATE 2>/dev/null | sed -n 's/^MEMNAV_RESIDENT_STATE=//p'")"
    if [[ "$resident_state" == parked ]]; then
      echo "RTX models: RESIDENT / IDLE (hub absent; episode state cleared)"
      echo "Motion stack is stopped; start a new phase to bind a fresh hub."
      return "$failures"
    fi
    echo "RTX session: RUNNING ($CFG_GPU_HOST:$CFG_GPU_SESSION)"
    assert_remote_session_config || failures=$((failures + 1))
    local health
    if health="$(remote_health 2>/dev/null)"; then
      validate_health "$health" || failures=$((failures + 1))
    else
      echo "RTX hub: UNHEALTHY"
      failures=$((failures + 1))
    fi
  else
    echo "RTX session: STOPPED or unreachable"
    failures=$((failures + 1))
  fi
  echo "Motion state must be confirmed from /navdp/status."
  return "$failures"
}

stop_stack() {
  navdp_require_config_arg "$@"
  navdp_read_config "$NAVDP_RUN_CONFIG"
  GPU_CONFIG="$CFG_GPU_REPO/runtime/config/$CFG_CONFIG_ID.json"
  bash "$OFFBOARD_DIR/stop_offboard_stack.sh" --config "$NAVDP_RUN_CONFIG"
  if ssh "${SSH_OPTIONS[@]}" "$CFG_GPU_HOST" true 2>/dev/null; then
    remote_exec "cd $(shell_quote "$CFG_GPU_REPO") && bash deployment/gpu/scripts/stop_policy_stack.sh --config $(shell_quote "$GPU_CONFIG")"
  else
    echo "Warning: RTX is unreachable; its policy session was not stopped." >&2
    return 1
  fi
  echo "Full-Mono Jetson and RTX sessions are stopped."
}

park_stack() {
  navdp_require_config_arg "$@"
  navdp_read_config "$NAVDP_RUN_CONFIG"
  # Stop command production and its in-flight client before touching GPU state.
  bash "$OFFBOARD_DIR/stop_offboard_stack.sh" --config "$NAVDP_RUN_CONFIG"
  if remote_session_exists; then
    local remote_state
    remote_state="$(remote_exec "tmux show-environment -t $(shell_quote "$CFG_GPU_SESSION") MEMNAV_RESIDENT_STATE 2>/dev/null | sed -n 's/^MEMNAV_RESIDENT_STATE=//p'")"
    if [[ "$remote_state" == parked ]]; then
      echo "RTX models already parked; motion stack remains stopped."
      return
    fi
    assert_remote_session_config
    GPU_CONFIG="$CFG_GPU_REPO/runtime/config/$CFG_CONFIG_ID.json"
    remote_exec "cd $(shell_quote "$CFG_GPU_REPO") && bash deployment/gpu/scripts/park_policy_stack.sh --config $(shell_quote "$GPU_CONFIG")"
  else
    # Distinguish no session from a lost SSH connection.
    remote_exec true || die "RTX unreachable; cannot confirm episode memory cleanup"
  fi
  echo "Motion stack stopped; GPU weights retained with episode state cleared."
}

action="${1:-}"
[[ $# -eq 0 ]] || shift
case "$action" in
  start) start_stack "$@" ;;
  status) status_stack "$@" ;;
  stop) stop_stack "$@" ;;
  park) park_stack "$@" ;;
  -h|--help|help|"") usage ;;
  *) die "unknown action: $action" ;;
esac
