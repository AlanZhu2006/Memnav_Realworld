#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
if [[ $# -ne 2 || "$1" != --config ]]; then
  echo "Usage: ${0##*/} --config RESOLVED_CONFIG.json" >&2
  exit 2
fi
RUN_CONFIG="$(readlink -f "$2")"
gpu_read_config "$RUN_CONFIG"
SESSION="$CFG_GPU_SESSION"
mkdir -p "$CEC_OUT_ROOT"
exec 9>"$CEC_OUT_ROOT/policy_lifecycle.lock"
flock -w 150 9
if tmux has-session -t "$SESSION" 2>/dev/null; then
  remote_id="$(tmux show-environment -t "$SESSION" MEMNAV_CONFIG_ID 2>/dev/null || true)"
  resident_state="$(tmux show-environment -t "$SESSION" MEMNAV_RESIDENT_STATE 2>/dev/null || true)"
  managed="$(tmux show-environment -t "$SESSION" MEMNAV_RESIDENT_MANAGER 2>/dev/null || true)"
  if [[ "$remote_id" != "MEMNAV_CONFIG_ID=$CFG_CONFIG_ID" ]] \
      && ! { [[ "$managed" == MEMNAV_RESIDENT_MANAGER=v1 ]] \
          && [[ "$resident_state" == MEMNAV_RESIDENT_STATE=parked ]]; }; then
    echo "Refusing to stop another/unidentified active GPU configuration" >&2
    exit 1
  fi
  tmux kill-session -t "$SESSION"
  echo "Stopped tmux session $SESSION"
else
  echo "No tmux session named $SESSION"
fi
