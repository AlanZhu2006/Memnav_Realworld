from PIL import Image
from flask import Flask, request, jsonify
from policy_agent import NavDP_Agent
from deterministic_seed import apply_seed
import io
import numpy as np
import cv2
import imageio
import time
import datetime
import hashlib
import json
import os
import requests
from PIL import Image, ImageDraw, ImageFont
import argparse
import gc
import threading
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument("--checkpoint",type=str,default="/home/PJLAB/caiwenzhe/Desktop/navdp_bench/baselines/navdp/checkpoints/cross-waic-final4-125.ckpt")
parser.add_argument(
    "--depth_source",
    choices=["metric_request", "zero", "monocular_sidecar"],
    default="metric_request",
    help="observation depth source; policy weights remain frozen",
)
parser.add_argument(
    "--allow_depth_source_override",
    action="store_true",
    help="allow navigator_reset to select a paired Gate-D depth arm",
)
parser.add_argument(
    "--monocular_depth_url",
    type=str,
    default="",
    help="MemNav/CEC sidecar /monocular_depth_query URL",
)
parser.add_argument("--monocular_depth_timeout_s", type=float, default=60.0)
parser.add_argument(
    "--require_monocular_depth_transaction",
    action="store_true",
    help=("require a SHA/frame-bound token when a current monocular depth "
          "payload is not already in NavDP's verified one-frame cache"),
)
args = parser.parse_known_args()[0]

app = Flask(__name__)
navdp_navigator = None
navdp_fps_writer = None
active_depth_source = args.depth_source
monocular_depth_cache = {}
episode_request_lock = threading.Lock()


@app.before_request
def serialize_episode_requests():
    # Reset/release must not race an in-flight inference or its depth request.
    episode_request_lock.acquire()


@app.teardown_request
def unlock_episode_request(error):
    episode_request_lock.release()


def release_unused_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@app.route("/resident/status")
def resident_status():
    return jsonify({
        "resident_contract": "episode_reset_v1", "pid": os.getpid(),
        "weights_loaded": navdp_navigator is not None,
        "queue_lengths": ([] if navdp_navigator is None else
                          [len(q) for q in navdp_navigator.memory_queue]),
        "depth_transactions": len(monocular_depth_cache),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
    })


@app.route("/resident/release", methods=["POST"])
def release_episode():
    global navdp_fps_writer
    if navdp_navigator is not None:
        navdp_navigator.reset(navdp_navigator.batch_size,
                             navdp_navigator.stop_threshold)
    monocular_depth_cache.clear()
    if navdp_fps_writer is not None:
        navdp_fps_writer.close()
        navdp_fps_writer = None
    release_unused_memory()
    return resident_status()


def _decode_request_depth(depth_file, batch_size):
    if depth_file is None:
        raise ValueError("metric_request depth source requires uploaded depth")
    depth = Image.open(depth_file.stream).convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    return depth.reshape((batch_size, -1, depth.shape[1], 1))


def _resolve_observation_depth(
    image_bytes,
    image_bgr,
    depth_file,
    batch_size,
    *,
    transaction_token=None,
    expected_frame_index=None,
):
    """Resolve exactly one configured depth arm and return an audit receipt."""

    global monocular_depth_cache
    source = active_depth_source
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    if source == "metric_request":
        depth = _decode_request_depth(depth_file, batch_size)
        return depth, {
            "depth_source": source,
            "metric_depth_sensor_consumed": True,
            "image_sha256": image_digest,
        }
    if source == "zero":
        height, width = image_bgr.shape[:2]
        depth = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        return depth, {
            "depth_source": source,
            "metric_depth_sensor_consumed": False,
            "image_sha256": image_digest,
            "scale_state": "explicit_zero_depth_arm",
        }
    if source != "monocular_sidecar":
        raise RuntimeError(f"unsupported active depth source {source!r}")
    if batch_size != 1:
        raise ValueError("monocular sidecar currently requires batch_size=1")
    cached = monocular_depth_cache.get(image_digest)
    if (
        cached is not None
        and transaction_token
        and cached[1].get("monocular_depth_transaction_token")
        != str(transaction_token)
    ):
        # Byte-identical observations may recur at different stream indices.
        # A digest-only cache hit can therefore belong to the older append.
        cached = None
    if cached is None:
        if not args.monocular_depth_url:
            raise RuntimeError("monocular depth URL is not configured")
        if args.require_monocular_depth_transaction and not transaction_token:
            raise RuntimeError(
                "monocular depth transaction token is required for this frame"
            )
        sidecar_data = {"expected_image_sha256": image_digest}
        if transaction_token:
            sidecar_data["monocular_depth_transaction_token"] = str(
                transaction_token
            )
        if expected_frame_index is not None:
            sidecar_data["expected_frame_index"] = str(
                int(expected_frame_index)
            )
        response = requests.post(
            args.monocular_depth_url,
            data=sidecar_data,
            timeout=float(args.monocular_depth_timeout_s),
        )
        if not response.ok:
            raise RuntimeError(
                "monocular depth sidecar rejected the bound frame: "
                f"HTTP {response.status_code} {response.text[:4096]}"
            )
        payload = response.json()
        from deployment.gpu.monocular_depth_runtime import (
            decode_monocular_depth_payload,
            validate_monocular_depth_transaction,
        )

        if transaction_token:
            validate_monocular_depth_transaction(
                payload,
                expected_token=str(transaction_token),
                expected_image_sha256=image_digest,
                expected_frame_index=(
                    None if expected_frame_index is None
                    else int(expected_frame_index)
                ),
            )
        depth_2d, metadata = decode_monocular_depth_payload(
            payload, expected_image_sha256=image_digest
        )
        cached = (depth_2d, metadata)
        monocular_depth_cache = {image_digest: cached}
    depth_2d, metadata = cached
    depth = depth_2d[None, :, :, None].astype(np.float32, copy=False)
    return depth, {
        "depth_source": source,
        "metric_depth_sensor_consumed": False,
        **dict(metadata),
    }


def _depth_response_fields(receipt):
    return {
        "depth_source": receipt["depth_source"],
        "metric_depth_sensor_consumed": bool(
            receipt["metric_depth_sensor_consumed"]
        ),
        "monocular_depth_receipt": (
            receipt if receipt["depth_source"] == "monocular_sidecar" else None
        ),
    }


def memory_queue_fingerprints(navigator):
    """Content fingerprints for fail-closed read-only resample contracts."""
    fingerprints = []
    for queue in navigator.memory_queue:
        digest = hashlib.sha256()
        digest.update(str(len(queue)).encode())
        for item in queue:
            array = np.ascontiguousarray(np.asarray(item))
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        fingerprints.append(digest.hexdigest())
    return fingerprints


def _form_boolean(name, default=False):
    value = request.form.get(name)
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean form value for {name}: {value}")


def _score_timesteps_from_form():
    raw = request.form.get("score_timesteps")
    if raw is None or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(parsed, list):
        raise ValueError("score_timesteps must be a JSON list or CSV sequence")
    return [int(value) for value in parsed]


def _critic_receipt(values):
    scores = np.asarray(values, dtype=np.float64)
    critic_max = float(scores.max())
    return {
        "critic_max": critic_max,
        "critic_min": float(scores.min()),
        "critic_threshold": float(navdp_navigator.stop_threshold),
        "critic_fallback_applied": bool(
            critic_max < float(navdp_navigator.stop_threshold)),
    }


@app.route("/navigator_reset",methods=['POST'])
def navdp_reset():
    global navdp_navigator,navdp_fps_writer,active_depth_source,monocular_depth_cache
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify({"error": "reset payload must be a JSON object"}), 400
    intrinsic = np.asarray(payload.get('intrinsic'), dtype=np.float64)
    threshold = float(payload.get('stop_threshold'))
    batchsize = int(payload.get('batch_size'))
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        return jsonify({"error": "intrinsic must be a finite 3x3 matrix"}), 400
    if not np.isfinite(threshold):
        return jsonify({"error": "stop_threshold must be finite"}), 400
    if batchsize < 1:
        return jsonify({"error": "batch_size must be positive"}), 400
    requested_depth_source = payload.get("depth_source", args.depth_source)
    allowed_depth_sources = {"metric_request", "zero", "monocular_sidecar"}
    if requested_depth_source not in allowed_depth_sources:
        return jsonify({"error": "invalid depth_source"}), 400
    if (requested_depth_source != args.depth_source
            and not args.allow_depth_source_override):
        return jsonify({
            "error": "depth_source override is disabled",
            "configured_depth_source": args.depth_source,
        }), 400
    if requested_depth_source == "monocular_sidecar" and not args.monocular_depth_url:
        return jsonify({"error": "monocular_depth_url is required"}), 400
    active_depth_source = requested_depth_source
    monocular_depth_cache = {}
    seed = payload.get('seed')
    apply_seed(seed)
    if navdp_navigator is None:
        navdp_navigator = NavDP_Agent(intrinsic,
                                image_size=224,
                                memory_size=8,
                                predict_size=24,
                                temporal_depth=16,
                                heads=8,
                                token_dim=384,
                                navi_model=args.checkpoint,
                                device='cuda:0')
        navdp_navigator.reset(batchsize,threshold)
    else:
        navdp_navigator.reset(batchsize,threshold)

    navdp_navigator.image_intrinsic = intrinsic
    # Weight initialization must not consume a seeded run's diffusion RNG on
    # cold starts only. Cold and resident resets now have the same seed boundary.
    apply_seed(seed)
    release_unused_memory()

    if os.environ.get("NAVDP_DISABLE_VIDEO", "0") == "1":
        navdp_fps_writer = None
    elif navdp_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    else:
        navdp_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    return jsonify({
        "algo": "navdp",
        "stop_threshold": threshold,
        "threshold_semantics": "critic_score_fallback",
        "batch_size": batchsize,
        "checkpoint_contract": navdp_navigator.checkpoint_contract,
        "depth_source": active_depth_source,
        "metric_depth_sensor_consumed_by_config": (
            active_depth_source == "metric_request"
        ),
        "depth_source_override_enabled": bool(args.allow_depth_source_override),
        "monocular_depth_url_configured": bool(args.monocular_depth_url),
        "monocular_depth_transaction_required": bool(
            args.require_monocular_depth_transaction
        ),
    })

@app.route("/navigator_reset_env",methods=['POST'])
def navdp_reset_env():
    global navdp_navigator
    navdp_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo":"navdp"})


@app.route("/memory_replay_step", methods=['POST'])
def navdp_memory_replay_step():
    """Append one frozen decision image without running diffusion."""
    global navdp_navigator
    if navdp_navigator is None:
        return jsonify({"error": "navigator is not initialized"}), 409
    image_file = request.files['image']
    batch_size = navdp_navigator.batch_size
    image = Image.open(image_file.stream).convert('RGB')
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    queue_lengths = navdp_navigator.append_observation(image)
    return jsonify({
        "algo": "navdp",
        "queue_lengths": queue_lengths,
        "memory_size": int(navdp_navigator.memory_size),
        "diffusion_sampled": False,
    })

@app.route("/pointgoal_step",methods=['POST'])
def navdp_step_xy():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
    batch_size = navdp_navigator.batch_size

    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pointgoal(goal,image,depth)
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))

    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        **_critic_receipt(all_values),
    })


@app.route("/pixelgoal_step",methods=['POST'])
def navdp_step_pixel():
    global navdp_navigator,navdp_fps_writer

    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y),axis=1)
    batch_size = navdp_navigator.batch_size

    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pixelgoal(goal,image,depth)
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist(),
                    'diffusion_seed': diffusion_seed})

@app.route("/imagegoal_step",methods=['POST'])
def navdp_step_image():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files.get('depth')
    goal_file = request.files['goal']
    batch_size = navdp_navigator.batch_size

    phase1_time = time.time()
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("image.jpg",image)
    current_image_bgr = image
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    goal = Image.open(goal_file.stream)
    goal = goal.convert('RGB')
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    cv2.imwrite("goal.jpg",goal)
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))

    depth, depth_receipt = _resolve_observation_depth(
        image_bytes, current_image_bgr,
        depth_file, batch_size,
        transaction_token=request.form.get(
            "monocular_depth_transaction_token"),
        expected_frame_index=request.form.get(
            "monocular_depth_frame_index"),
    )

    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_imagegoal(goal,image,depth)
    queue_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        'queue_lengths': queue_lengths,
        'memory_size': int(navdp_navigator.memory_size),
        **_critic_receipt(all_values),
        **_depth_response_fields(depth_receipt),
    })


@app.route("/imagegoal_resample", methods=["POST"])
def navdp_resample_imagegoal():
    """Resample candidates from the current FIFO without appending an image."""
    global navdp_navigator
    if navdp_navigator is None:
        return jsonify({"error": "navigator is not initialized"}), 409
    batch_size = navdp_navigator.batch_size

    image_bytes = request.files['image'].read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    current_image_bgr = image
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    goal = Image.open(request.files['goal'].stream).convert('RGB')
    goal = cv2.cvtColor(np.asarray(goal), cv2.COLOR_RGB2BGR)
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))

    depth, depth_receipt = _resolve_observation_depth(
        image_bytes, current_image_bgr,
        request.files.get('depth'), batch_size,
        transaction_token=request.form.get(
            "monocular_depth_transaction_token"),
        expected_frame_index=request.form.get(
            "monocular_depth_frame_index"),
    )

    before_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    before_hashes = memory_queue_fingerprints(navdp_navigator)
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, _trajectory_mask = (
        navdp_navigator.resample_imagegoal(goal, image, depth))
    after_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    after_hashes = memory_queue_fingerprints(navdp_navigator)
    if after_lengths != before_lengths or after_hashes != before_hashes:
        return jsonify({"error": "resampling mutated NavDP memory"}), 500
    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        **_critic_receipt(all_values),
        'memory_mutated': False,
        'queue_lengths': after_lengths,
        'queue_hashes_before': before_hashes,
        'queue_hashes_after': after_hashes,
        **_depth_response_fields(depth_receipt),
    })

@app.route("/nogoal_step",methods=['POST'])
def navdp_step_nogoal():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    batch_size = navdp_navigator.batch_size

    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_nogoal(image,depth)
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist(),
                    'diffusion_seed': diffusion_seed})

@app.route("/navdp_step_ip_mixgoal",methods=['POST'])
def navdp_step_ip_mixgoal():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files.get('depth')
    batch_size = navdp_navigator.batch_size

    point_goal_data = json.loads(request.form.get('goal_data'))
    point_goal_x = np.array(point_goal_data['goal_x'])
    point_goal_y = np.array(point_goal_data['goal_y'])
    point_goal = np.stack((point_goal_x,point_goal_y,np.zeros_like(point_goal_x)),axis=1)

    image_goal_file = request.files['image_goal']
    image_goal = Image.open(image_goal_file.stream)
    image_goal = image_goal.convert('RGB')
    image_goal = np.asarray(image_goal)
    image_goal = cv2.cvtColor(image_goal, cv2.COLOR_RGB2BGR)
    cv2.imwrite("goal.jpg",image_goal)
    image_goal = image_goal.reshape((batch_size, -1, image_goal.shape[1], 3))

    phase1_time = time.time()
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    current_image_bgr = image
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth, depth_receipt = _resolve_observation_depth(
        image_bytes, current_image_bgr, depth_file, batch_size,
        transaction_token=request.form.get(
            "monocular_depth_transaction_token"),
        expected_frame_index=request.form.get(
            "monocular_depth_frame_index"),
    )

    phase2_time = time.time()
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_point_image_goal(point_goal,image_goal,image,depth)
    pointgoal_diagnostic = {
        'schema': 'navdp_pointgoal_input_v1',
        'received_xyz': point_goal.tolist(),
        'processed_xyz': navdp_navigator.process_pointgoal(point_goal).tolist(),
        'observation_only': True,
    }
    queue_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    phase3_time = time.time()
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({
        'pointgoal_diagnostic': pointgoal_diagnostic,
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        'queue_lengths': queue_lengths,
        'memory_size': int(navdp_navigator.memory_size),
        **_critic_receipt(all_values),
        **_depth_response_fields(depth_receipt),
    })


@app.route("/mixgoal_resample", methods=["POST"])
def navdp_resample_mixgoal():
    """Read-only mixed image/point proposal from the current NavDP FIFO."""
    global navdp_navigator
    if navdp_navigator is None:
        return jsonify({"error": "navigator is not initialized"}), 409
    batch_size = navdp_navigator.batch_size
    point_goal_data = json.loads(request.form.get('goal_data'))
    point_goal_x = np.asarray(point_goal_data['goal_x'])
    point_goal_y = np.asarray(point_goal_data['goal_y'])
    point_goal = np.stack(
        (point_goal_x, point_goal_y, np.zeros_like(point_goal_x)), axis=1)

    image_bytes = request.files['image'].read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    current_image_bgr = image
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    image_goal = Image.open(request.files['image_goal'].stream).convert('RGB')
    image_goal = cv2.cvtColor(np.asarray(image_goal), cv2.COLOR_RGB2BGR)
    image_goal = image_goal.reshape(
        (batch_size, -1, image_goal.shape[1], 3))
    control_goal = None
    if 'control_goal' in request.files:
        control_goal = Image.open(
            request.files['control_goal'].stream).convert('RGB')
        control_goal = cv2.cvtColor(
            np.asarray(control_goal), cv2.COLOR_RGB2BGR)
        control_goal = control_goal.reshape(
            (batch_size, -1, control_goal.shape[1], 3))
    depth, depth_receipt = _resolve_observation_depth(
        image_bytes, current_image_bgr, request.files.get('depth'), batch_size,
        transaction_token=request.form.get(
            "monocular_depth_transaction_token"),
        expected_frame_index=request.form.get(
            "monocular_depth_frame_index"),
    )

    try:
        score_goal_contrast = _form_boolean("score_goal_contrast", False)
        score_timesteps = _score_timesteps_from_form()
        score_noise_samples = int(
            request.form.get("score_noise_samples", 1))
        score_seed = int(request.form.get(
            "score_seed", request.form.get('diffusion_seed', 0)))
        if score_noise_samples < 1 or score_noise_samples > 16:
            raise ValueError("score_noise_samples must lie in [1, 16]")
        if control_goal is not None and not score_goal_contrast:
            raise ValueError(
                "control_goal requires score_goal_contrast=true")
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    before_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    before_hashes = memory_queue_fingerprints(navdp_navigator)
    diffusion_seed = apply_seed(request.form.get('diffusion_seed'))
    execute_trajectory, all_trajectory, all_values, _trajectory_mask = (
        navdp_navigator.resample_point_image_goal(
            point_goal, image_goal, image, depth))
    goal_contrast = None
    if score_goal_contrast:
        try:
            goal_contrast = navdp_navigator.score_imagegoal_trajectories(
                image_goal,
                image,
                depth,
                all_trajectory,
                control_imagegoal=control_goal,
                timesteps=score_timesteps,
                noise_samples=score_noise_samples,
                seed=score_seed,
            )
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
    after_lengths = [len(queue) for queue in navdp_navigator.memory_queue]
    after_hashes = memory_queue_fingerprints(navdp_navigator)
    if after_lengths != before_lengths or after_hashes != before_hashes:
        return jsonify({"error": "mixed resampling mutated NavDP memory"}), 500
    payload = {
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
        'diffusion_seed': diffusion_seed,
        **_critic_receipt(all_values),
        'memory_mutated': False,
        'queue_lengths': after_lengths,
        'queue_hashes_before': before_hashes,
        'queue_hashes_after': after_hashes,
        **_depth_response_fields(depth_receipt),
    }
    if goal_contrast is not None:
        payload['goal_contrast'] = goal_contrast
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=args.port)
