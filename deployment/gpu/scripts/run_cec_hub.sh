#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
gpu_require_config "$@"
require_executable "$MEMNAV_PY"
CEC_GOAL_CANDIDATE_DIR="$CEC_OUT_ROOT/goal_candidates"
CEC_EPISODIC_DATASET_ROOT="$CEC_OUT_ROOT/episodic_datasets"
mkdir -p "$CEC_GOAL_CANDIDATE_DIR" "$CEC_EPISODIC_DATASET_ROOT"
dataset_args=()
if [[ "$CFG_DATASET_AUTO_OPEN" == true ]]; then
  dataset_args+=(--auto-dataset-id "$CFG_DATASET_ID")
  dataset_args+=(
    --auto-dataset-metadata-json
    "$CFG_DATASET_METADATA"
  )
fi

cd "$REPO_ROOT"
exec env PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$MEMNAV_PY" -u -m deployment.gpu.realworld_cec_hub \
    --host 127.0.0.1 --port "$CEC_HUB_PORT" \
    --memnav-url "http://127.0.0.1:$MEMNAV_PORT" \
    --navdp-url "http://127.0.0.1:$NAVDP_PORT" \
    --camera-height-m "$CFG_CAMERA_HEIGHT_M" \
    --authority-mode "$CFG_AUTHORITY_MODE" \
    --terminal-approach "$CFG_TERMINAL_APPROACH" \
    --goal-candidate-dir "$CEC_GOAL_CANDIDATE_DIR" \
    --goal-score-stride "$CFG_GOAL_SCORE_STRIDE" \
    --goal-min-frame-gap "$CFG_GOAL_MIN_FRAME_GAP" \
    --goal-min-inliers "$CFG_GOAL_MIN_INLIERS" \
    --goal-max-cos "$CFG_GOAL_MAX_COS" \
    --episodic-dataset-root "$CEC_EPISODIC_DATASET_ROOT" \
    --episodic-dataset-min-frames "$CFG_DATASET_MIN_FRAMES" \
    "${dataset_args[@]}"
