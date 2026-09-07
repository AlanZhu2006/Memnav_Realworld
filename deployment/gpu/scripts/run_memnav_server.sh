#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
gpu_require_config "$@"
MEMNAV_SOURCE_ROOT="$CFG_MEMNAV_SOURCE_ROOT"
MEMNAV_CKPT="$CFG_MEMNAV_CKPT"
INTERNNAV_ROOT="$CFG_INTERNNAV_ROOT"
LINGBOT_REPO="$CFG_LINGBOT_REPO"
LINGBOT_WEIGHTS="$CFG_LINGBOT_WEIGHTS"
LIGHTGLUE_REPO="$CFG_LIGHTGLUE_REPO"
DEPENDENCY_ROOT="$CFG_DEPENDENCY_ROOT"
MEMNAV_SERVER="$MEMNAV_SOURCE_ROOT/NavDP/baselines/memnav/memnav_server.py"
# A per-process namespace prevents a restart from erasing the preceding trace.
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
BUFFER_ROOT="$CEC_OUT_ROOT/buffer/run_$RUN_STAMP"
require_executable "$MEMNAV_PY"
require_file "$MEMNAV_SERVER"
require_file "$MEMNAV_CKPT"
require_file "$LINGBOT_WEIGHTS"
require_dir "$INTERNNAV_ROOT"
require_dir "$LINGBOT_REPO"
require_dir "$LIGHTGLUE_REPO"
require_dir "$DEPENDENCY_ROOT"
mkdir -p "$BUFFER_ROOT"
echo "realworld_memnav_buffer_root=$BUFFER_ROOT"

extra_args=()
if [[ "$CFG_EAGER_DEPTH_CACHE" == true ]]; then
  extra_args+=(--certified_eager_depth_cache)
fi
server_pythonpath="$MEMNAV_SOURCE_ROOT:$DEPENDENCY_ROOT:$LIGHTGLUE_REPO:$INTERNNAV_ROOT/src/diffusion-policy${PYTHONPATH:+:$PYTHONPATH}"

cd "$CEC_OUT_ROOT"
exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="$server_pythonpath" \
  LINGBOT_REPO="$LINGBOT_REPO" LINGBOT_WEIGHTS="$LINGBOT_WEIGHTS" \
  MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
  MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
  MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
  MEMNAV_REPORT_TO=none \
  "$MEMNAV_PY" -u "$REPO_ROOT/deployment/gpu/resident_memnav_server.py" "$MEMNAV_SERVER" \
    --host 127.0.0.1 --port "$MEMNAV_PORT" --checkpoint "$MEMNAV_CKPT" \
    --internnav_root "$INTERNNAV_ROOT" --num_samples 16 \
    --exclude_recent 32 --retrieval raw \
    --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
    --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
    --flow_gate auto --buffer_root "$BUFFER_ROOT" \
    --certified_relocalization \
    --certified_reference_depth_source "$CFG_HISTORICAL_DEPTH_SOURCE" \
    --lightglue_repo "$LIGHTGLUE_REPO" \
    --lightglue_dependency_root "$DEPENDENCY_ROOT" \
    --lightglue_max_keypoints 2048 \
    "${extra_args[@]}"
