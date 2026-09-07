#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
# Stop/reset remains usable even when source has changed since this run.
[[ $# -eq 2 && "$1" == --config ]] || exit 2
RUN_CONFIG="$(readlink -f "$2")"
gpu_read_config "$RUN_CONFIG"
mkdir -p "$CEC_OUT_ROOT"
exec 9>"$CEC_OUT_ROOT/policy_lifecycle.lock"
flock -w 150 9
python3 "$GPU_DIR/resident_policy.py" park --config "$RUN_CONFIG"
