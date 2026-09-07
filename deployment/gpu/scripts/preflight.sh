#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
gpu_require_config "$@"
failures=0
pass() { printf '[PASS] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }

[[ -x "$MEMNAV_PY" ]] || command -v "$MEMNAV_PY" >/dev/null 2>&1 \
  && pass "GPU Python" || fail "GPU Python: $MEMNAV_PY"
[[ -f "$CFG_MEMNAV_CKPT" ]] && pass "MemNav checkpoint" || fail "MemNav checkpoint"
[[ -f "$CFG_NAVDP_CKPT" ]] && pass "NavDP checkpoint" || fail "NavDP checkpoint"
MEMNAV_SERVER="$CFG_MEMNAV_SOURCE_ROOT/NavDP/baselines/memnav/memnav_server.py"
MEMNAV_DEPTH_RUNTIME="$CFG_MEMNAV_SOURCE_ROOT/MemNavData/monocular_depth_runtime.py"
REPO_DEPTH_RUNTIME="$REPO_ROOT/deployment/gpu/monocular_depth_runtime.py"
MEMNAV_LATENCY_PATCH="$REPO_ROOT/deployment/gpu/patches/memnav_reuse_flow_depth.patch"
[[ -f "$MEMNAV_SERVER" ]] \
  && pass "MemNav server" || fail "MEMNAV_SERVER"
[[ -f "$MEMNAV_DEPTH_RUNTIME" ]] \
  && pass "monocular depth runtime" || fail "MEMNAV_SOURCE_ROOT/monocular_depth_runtime.py"
if [[ -f "$MEMNAV_DEPTH_RUNTIME" && -f "$REPO_DEPTH_RUNTIME" ]] \
    && cmp -s "$MEMNAV_DEPTH_RUNTIME" "$REPO_DEPTH_RUNTIME"; then
  pass "monocular depth runtime matches the tracked source"
else
  fail "external monocular depth runtime differs from the tracked source"
fi
grep -q 'monocular_depth_query' \
  "$MEMNAV_SERVER" \
  2>/dev/null \
  && pass "MemNav protocol-v2 depth endpoint" \
  || fail "MemNav server lacks /monocular_depth_query"
grep -q 'goal_candidate_support' \
  "$MEMNAV_SERVER" \
  2>/dev/null \
  && pass "MemNav read-only goal support endpoint" \
  || fail "MemNav server lacks /goal_candidate_support"
if [[ -f "$MEMNAV_LATENCY_PATCH" ]] \
    && git -C "$CFG_MEMNAV_SOURCE_ROOT" apply --reverse --check \
      "$MEMNAV_LATENCY_PATCH" >/dev/null 2>&1; then
  pass "MemNav current-frame depth reuse patch"
else
  fail "MemNav latency patch missing; run apply_memnav_source_patch.sh"
fi
[[ -f "$CFG_LINGBOT_WEIGHTS" ]] && pass "LingBot weights" || fail "LingBot weights"
[[ -d "$CFG_LIGHTGLUE_REPO" ]] && pass "LightGlue source" || fail "LightGlue source"
[[ -f "$REPO_ROOT/baselines/navdp/navdp_server.py" ]] \
  && pass "Full-Mono NavDP server" || fail "Full-Mono NavDP server"
if python3 - "$CFG_CAMERA_HEIGHT_M" <<'PY'
import math
import sys

value = float(sys.argv[1])
assert math.isfinite(value) and 0.1 <= value <= 2.0
PY
then
  pass "measured camera height: ${CFG_CAMERA_HEIGHT_M} m"
else
  fail "camera height must be finite and in [0.1, 2.0] m"
fi
case "$CFG_AUTHORITY_MODE" in
  cec|native) pass "authority mode: $CFG_AUTHORITY_MODE" ;;
  *) fail "configured authority_mode must be cec or native" ;;
esac
case "$CFG_HISTORICAL_DEPTH_SOURCE" in
  canonical|online_history) pass "CEC history depth: $CFG_HISTORICAL_DEPTH_SOURCE" ;;
  *) fail "unknown CEC historical depth source" ;;
esac
if ! grep -q -- '--certified_reference_depth_source' "$MEMNAV_SERVER"; then
  fail "external MemNav source lacks the configured CEC depth-source interface"
fi
for port in "$MEMNAV_PORT" "$NAVDP_PORT" "$CEC_HUB_PORT"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$port$"; then
    fail "port already in use: $port"
  else
    pass "port available: $port"
  fi
done

printf '\nGPU preflight complete: failures=%d\n' "$failures"
exit "$failures"
