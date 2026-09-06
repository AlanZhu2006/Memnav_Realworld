#!/usr/bin/env python3
"""Resolve, validate and transport one immutable MemNav-RealWorld run config.

The operator edits tracked JSON.  Launchers consume only the resolved JSON
emitted here; environment variables are intentionally not a configuration API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
from typing import Any, Mapping

from go2.stack_profiles import ARRIVAL_MODULES, PROFILES, validate_combination


SYSTEM_SCHEMA = "memnav-realworld-system-v1"
EXPERIMENT_SCHEMA = "memnav-realworld-experiment-v1"
RESOLVED_SCHEMA = "memnav-realworld-resolved-v1"
PHASES = {"memory_recording", "revisit_query"}


class ConfigError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be an object: {path}")
    return value


def _exact_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"unknown {label} field(s): {', '.join(sorted(unknown))}")


def _required_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    if missing:
        raise ConfigError(f"missing {label} field(s): {', '.join(sorted(missing))}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    return dict(value)


def _number(value: Any, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ConfigError(f"{label} must be >= {minimum}")
    return result


def _positive_port(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ConfigError(f"{label} must be an integer in [1, 65535]")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def _integer_range(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ConfigError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _payload_id(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("config_id", None)
    return hashlib.sha256(_canonical_bytes(unhashed)).hexdigest()


def _git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigError(f"cannot determine Git revision under {repo}") from exc


def _repo_path(repo: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{label} must be a non-empty path string")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    return candidate.resolve()


def _image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 9 <= len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            offset += 2
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if offset + 2 > len(data):
                break
            length = int.from_bytes(data[offset : offset + 2], "big")
            if length < 2 or offset + length > len(data):
                break
            if marker in {
                0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
            }:
                height = int.from_bytes(data[offset + 3 : offset + 5], "big")
                width = int.from_bytes(data[offset + 5 : offset + 7], "big")
                return width, height
            offset += length
    raise ConfigError(f"ImageGoal is not a readable PNG/JPEG: {path}")


def _image_artifact(repo: Path, raw: Any, label: str) -> dict[str, Any]:
    path = _repo_path(repo, raw, label)
    if not path.is_file():
        raise ConfigError(f"{label} does not exist: {path}")
    width, height = _image_size(path)
    try:
        repository_path: str | None = path.relative_to(repo).as_posix()
    except ValueError:
        repository_path = None
    return {
        "path": str(path),
        "repository_path": repository_path,
        "sha256": _sha256_file(path),
        "width": width,
        "height": height,
    }


def _optional_image_artifact(
    repo: Path, raw: Any, label: str
) -> dict[str, Any] | None:
    if raw is None:
        return None
    return _image_artifact(repo, raw, label)


def _validate_system(system: Mapping[str, Any]) -> None:
    _exact_keys(system, {"schema", "sites", "stack"}, "system")
    _required_keys(system, {"schema", "sites", "stack"}, "system")
    if system["schema"] != SYSTEM_SCHEMA:
        raise ConfigError(f"unsupported system schema: {system['schema']!r}")
    sites = _object(system["sites"], "sites")
    _exact_keys(sites, {"jetson", "gpu"}, "sites")
    _required_keys(sites, {"jetson", "gpu"}, "sites")
    jetson = _object(sites["jetson"], "sites.jetson")
    jetson_fields = {
        "hostname", "repository", "python", "ros_setup",
        "realsense_setup", "message_filters_setup", "runtime_root",
        "sessions", "camera", "unitree", "native_policy",
    }
    _exact_keys(
        jetson,
        jetson_fields,
        "sites.jetson",
    )
    _required_keys(jetson, jetson_fields, "sites.jetson")
    gpu = _object(sites["gpu"], "sites.gpu")
    gpu_fields = {
        "hostname", "ssh_host", "repository", "python", "runtime_root",
        "session", "ports", "models", "ready_timeout_s",
    }
    _exact_keys(
        gpu,
        gpu_fields,
        "sites.gpu",
    )
    _required_keys(gpu, gpu_fields, "sites.gpu")
    for site_name, site in (("jetson", jetson), ("gpu", gpu)):
        for field in ("hostname", "repository", "python", "runtime_root"):
            if not isinstance(site.get(field), str) or not site[field]:
                raise ConfigError(f"sites.{site_name}.{field} must be a string")
    ports = _object(gpu.get("ports"), "sites.gpu.ports")
    _exact_keys(ports, {"memnav", "navdp", "hub", "tunnel_local"}, "GPU ports")
    _required_keys(ports, {"memnav", "navdp", "hub", "tunnel_local"}, "GPU ports")
    for name, value in ports.items():
        _positive_port(value, f"sites.gpu.ports.{name}")
    if len(set(ports.values())) != 4:
        # hub and tunnel_local deliberately bind on different machines.
        allowed = ports["hub"] == ports["tunnel_local"]
        if not allowed or len({ports["memnav"], ports["navdp"], ports["hub"]}) != 3:
            raise ConfigError("GPU service ports collide")
    sessions = _object(jetson["sessions"], "sites.jetson.sessions")
    _exact_keys(sessions, {"native", "fullmono"}, "Jetson sessions")
    _required_keys(sessions, {"native", "fullmono"}, "Jetson sessions")
    camera_fields = {
        "minimum_firmware", "depth_profile", "color_profile", "ready_timeout_s",
        "rgb_topic", "depth_topic", "camera_info_topic",
    }
    camera = _object(jetson["camera"], "sites.jetson.camera")
    _exact_keys(camera, camera_fields, "Jetson camera")
    _required_keys(camera, camera_fields, "Jetson camera")
    unitree_fields = {
        "network_interface", "sdk_python_path", "cyclonedds_home", "python",
        "cmd_topic", "timeout_s", "max_vx", "max_vy", "max_wz",
        "min_cmd_v", "min_cmd_w",
    }
    unitree = _object(jetson["unitree"], "sites.jetson.unitree")
    _exact_keys(unitree, unitree_fields, "Unitree")
    _required_keys(unitree, unitree_fields, "Unitree")
    native_fields = {
        "host", "port", "device", "ready_timeout_s", "checkpoint",
        "checkpoint_sha256",
    }
    native = _object(jetson["native_policy"], "sites.jetson.native_policy")
    _exact_keys(native, native_fields, "native policy")
    _required_keys(native, native_fields, "native policy")
    _positive_port(native["port"], "sites.jetson.native_policy.port")
    model_fields = {
        "memnav_source_root", "memnav_checkpoint", "internnav_root",
        "lingbot_repository", "lingbot_weights", "lightglue_repository",
        "dependency_root", "navdp_checkpoint",
    }
    models = _object(gpu["models"], "sites.gpu.models")
    _exact_keys(models, model_fields, "GPU models")
    _required_keys(models, model_fields, "GPU models")
    stack_fields = {
        "adapter_params", "foxglove", "camera_height_m", "formal_limits",
        "memory", "cec", "arrival", "evidence", "adapter_ready_timeout_s",
        "tunnel_ready_timeout_s",
    }
    stack = _object(system["stack"], "stack")
    _exact_keys(stack, stack_fields, "stack")
    _required_keys(stack, stack_fields, "stack")
    nested_fields = {
        "foxglove": {"layout", "address", "port", "preview"},
        "formal_limits": {"max_linear_mps", "max_angular_rps"},
        "memory": {
            "navigate_during_recording", "pause_recording",
            "auto_goal_candidate_interval_frames", "auto_goal_candidate_max",
            "auto_goal_candidate_post_guard_frames",
            "auto_goal_candidate_capture_enabled", "auto_select_goal_candidate",
        },
        "cec": {
            "goal_score_stride", "goal_min_frame_gap", "goal_min_inliers",
            "goal_max_cos", "episodic_dataset_min_frames", "eager_depth_cache",
        },
        "arrival": {
            "rate_hz", "required_consecutive", "min_image_scale", "max_image_scale",
        },
        "evidence": {"capture_root", "session_prefix"},
    }
    for name, fields in nested_fields.items():
        nested = _object(stack[name], f"stack.{name}")
        _exact_keys(nested, fields, f"stack.{name}")
        _required_keys(nested, fields, f"stack.{name}")
    foxglove = _object(stack["foxglove"], "stack.foxglove")
    if not isinstance(foxglove["address"], str) or not foxglove["address"]:
        raise ConfigError("stack.foxglove.address must be a non-empty string")
    _positive_port(foxglove["port"], "stack.foxglove.port")
    preview = _object(foxglove["preview"], "stack.foxglove.preview")
    preview_fields = {
        "rgb_topic", "depth_topic", "goal_topic", "arrival_topic", "status_topic",
        "width", "height", "status_width", "status_height", "rgb_fps", "depth_fps",
        "goal_fps", "arrival_fps", "status_fps", "rgb_jpeg_quality",
        "depth_jpeg_quality", "goal_jpeg_quality", "arrival_jpeg_quality",
        "status_jpeg_quality", "arrival_preserve_resolution", "depth_min_mm",
        "depth_max_mm",
    }
    _exact_keys(preview, preview_fields, "stack.foxglove.preview")
    _required_keys(preview, preview_fields, "stack.foxglove.preview")
    for field in (
        "rgb_topic", "depth_topic", "goal_topic", "arrival_topic", "status_topic"
    ):
        if not isinstance(preview[field], str) or not preview[field].startswith("/"):
            raise ConfigError(
                f"stack.foxglove.preview.{field} must be an absolute ROS topic"
            )
    for field in (
        "width", "height", "status_width", "status_height", "rgb_fps", "depth_fps",
        "goal_fps", "arrival_fps", "status_fps",
    ):
        _positive_integer(preview[field], f"stack.foxglove.preview.{field}")
    if preview["status_width"] < 480 or preview["status_height"] < 220:
        raise ConfigError(
            "stack.foxglove.preview status card must be at least 480x220"
        )
    for field in (
        "rgb_jpeg_quality", "depth_jpeg_quality", "goal_jpeg_quality",
        "arrival_jpeg_quality", "status_jpeg_quality",
    ):
        _integer_range(preview[field], f"stack.foxglove.preview.{field}", 1, 100)
    if not isinstance(preview["arrival_preserve_resolution"], bool):
        raise ConfigError(
            "stack.foxglove.preview.arrival_preserve_resolution must be boolean"
        )
    depth_min = _number(
        preview["depth_min_mm"], "stack.foxglove.preview.depth_min_mm", 0
    )
    depth_max = _number(
        preview["depth_max_mm"], "stack.foxglove.preview.depth_max_mm", 0
    )
    if depth_max <= depth_min:
        raise ConfigError(
            "stack.foxglove.preview.depth_max_mm must exceed depth_min_mm"
        )
    height = _number(
        stack.get("camera_height_m"),
        "stack.camera_height_m",
    )
    if not 0.1 <= height <= 2.0:
        raise ConfigError("stack.camera_height_m must be in [0.1, 2.0]")
    for label, value in (
        ("sites.jetson.native_policy.ready_timeout_s", native["ready_timeout_s"]),
        ("sites.gpu.ready_timeout_s", gpu["ready_timeout_s"]),
        ("stack.adapter_ready_timeout_s", stack["adapter_ready_timeout_s"]),
        ("stack.tunnel_ready_timeout_s", stack["tunnel_ready_timeout_s"]),
        ("sites.jetson.camera.ready_timeout_s", camera["ready_timeout_s"]),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{label} must be a positive integer")


def _validate_experiment(experiment_file: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(experiment_file, {"schema", "system_config", "experiment"}, "experiment file")
    _required_keys(experiment_file, {"schema", "system_config", "experiment"}, "experiment file")
    if experiment_file["schema"] != EXPERIMENT_SCHEMA:
        raise ConfigError(f"unsupported experiment schema: {experiment_file['schema']!r}")
    experiment = _object(experiment_file["experiment"], "experiment")
    _exact_keys(
        experiment,
        {"id", "profile", "authority_mode", "terminal_approach", "navigation", "arrival", "launch", "control"},
        "experiment",
    )
    _required_keys(
        experiment,
        {"id", "profile", "authority_mode", "navigation", "arrival", "launch", "control"},
        "experiment",
    )
    if not isinstance(experiment["id"], str) or not experiment["id"]:
        raise ConfigError("experiment.id must be a non-empty string")
    if experiment["profile"] not in PROFILES:
        raise ConfigError(f"unknown profile: {experiment['profile']!r}")
    if experiment["authority_mode"] not in {"native", "cec"}:
        raise ConfigError("experiment.authority_mode must be native or cec")
    if experiment.get("terminal_approach", "bearing_only") not in {"bearing_only", "height_scaled_local"}:
        raise ConfigError("invalid experiment.terminal_approach")
    if experiment["profile"] == "native-navdp-rgbd" and experiment["authority_mode"] != "native":
        raise ConfigError("native-navdp-rgbd requires authority_mode=native")
    navigation = _object(experiment["navigation"], "experiment.navigation")
    _exact_keys(navigation, {"image_goal", "revisit_image_goal"}, "navigation")
    _required_keys(navigation, {"image_goal", "revisit_image_goal"}, "navigation")
    arrival = _object(experiment["arrival"], "experiment.arrival")
    _exact_keys(arrival, {"module", "image_goal", "allowed_phases"}, "arrival")
    _required_keys(arrival, {"module", "image_goal", "allowed_phases"}, "arrival")
    if arrival["module"] not in ARRIVAL_MODULES:
        raise ConfigError(f"unknown arrival module: {arrival['module']!r}")
    try:
        validate_combination(experiment["profile"], arrival["module"])
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    phases = arrival["allowed_phases"]
    if not isinstance(phases, list) or not phases or any(p not in PHASES for p in phases):
        raise ConfigError("arrival.allowed_phases must be a non-empty valid phase list")
    launch = _object(experiment["launch"], "experiment.launch")
    _exact_keys(launch, {"camera", "go2_bridge", "foxglove"}, "launch")
    _required_keys(launch, {"camera", "go2_bridge", "foxglove"}, "launch")
    if any(not isinstance(value, bool) for value in launch.values()):
        raise ConfigError("all launch fields must be booleans")
    if experiment["profile"] == "fullmono-lingbot-cec" and not launch["camera"]:
        raise ConfigError("Full-Mono owns the camera; launch.camera must be true")
    control = _object(experiment["control"], "experiment.control")
    _exact_keys(control, {"profile"}, "control")
    if control.get("profile") not in {"formal", "acceptance"}:
        raise ConfigError("control.profile must be formal or acceptance")
    return experiment


def _write_resolved(payload: dict[str, Any], output: Path) -> Path:
    payload["config_id"] = _payload_id(payload)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def resolve(experiment_path: Path, output: Path | None) -> Path:
    experiment_path = experiment_path.resolve()
    repo = Path(__file__).resolve().parents[1]
    experiment_file = _load_json(experiment_path)
    experiment = _validate_experiment(experiment_file)
    system_ref = experiment_file["system_config"]
    if not isinstance(system_ref, str) or not system_ref:
        raise ConfigError("system_config must be a path string")
    system_path = (experiment_path.parent / system_ref).resolve()
    system = _load_json(system_path)
    _validate_system(system)

    stack = _object(system["stack"], "stack")
    limits = _object(stack["formal_limits"], "stack.formal_limits")
    memory = _object(stack["memory"], "stack.memory")
    cec = _object(stack["cec"], "stack.cec")
    arrival_defaults = _object(stack["arrival"], "stack.arrival")
    navigation = _object(experiment["navigation"], "experiment.navigation")
    arrival = _object(experiment["arrival"], "experiment.arrival")
    jetson = _object(_object(system["sites"], "sites")["jetson"], "sites.jetson")
    gpu = _object(_object(system["sites"], "sites")["gpu"], "sites.gpu")

    native_policy = _object(jetson["native_policy"], "sites.jetson.native_policy")
    native_policy["checkpoint"] = str(
        _repo_path(repo, native_policy["checkpoint"], "native checkpoint")
    )
    jetson["native_policy"] = native_policy
    jetson["repository"] = str(Path(jetson["repository"]).resolve())
    gpu["repository"] = str(Path(gpu["repository"]).resolve())
    stack["adapter_params"] = str(
        _repo_path(repo, stack["adapter_params"], "adapter params")
    )
    foxglove = _object(stack["foxglove"], "stack.foxglove")
    foxglove["layout"] = str(
        _repo_path(repo, foxglove["layout"], "Foxglove layout")
    )
    stack["foxglove"] = foxglove
    goal = _image_artifact(repo, navigation["image_goal"], "navigation ImageGoal")
    revisit_goal = _optional_image_artifact(
        repo, navigation["revisit_image_goal"], "revisit ImageGoal"
    )
    arrival_goal = _image_artifact(repo, arrival["image_goal"], "arrival ImageGoal")

    payload: dict[str, Any] = {
        "schema": RESOLVED_SCHEMA,
        "source": {
            "experiment_file": str(experiment_path),
            "experiment_sha256": _sha256_file(experiment_path),
            "system_file": str(system_path),
            "system_sha256": _sha256_file(system_path),
            "repository": str(repo),
            "git_revision": _git_revision(repo),
        },
        "experiment": {
            "id": experiment["id"],
            "profile": experiment["profile"],
            "phase": "standard",
        },
        "formal": None,
        "sites": {"jetson": jetson, "gpu": gpu},
        "stack": stack,
        "navigation": {
            "backend": "navdp",
            "mode": "imagegoal",
            "two_phase": experiment["profile"] == "fullmono-lingbot-cec",
            "image_goal": goal,
            "revisit_image_goal": revisit_goal,
            "selected_goal_image_path": None,
            "selected_goal_depth_path": None,
        },
        "arrival": {
            "module": arrival["module"],
            "image_goal": arrival_goal,
            "allowed_phases": list(arrival["allowed_phases"]),
            **arrival_defaults,
        },
        "launch": dict(experiment["launch"]),
        "control": {
            "profile": experiment["control"]["profile"],
            "max_linear_mps": _number(limits["max_linear_mps"], "max_linear_mps", 0),
            "max_angular_rps": _number(limits["max_angular_rps"], "max_angular_rps", 0),
        },
        "memory": memory,
        "cec": {**cec, "authority_mode": experiment["authority_mode"],
                "terminal_approach": experiment.get("terminal_approach", "bearing_only")},
        "dataset": {"auto_open": False, "id": None, "metadata": {}},
    }
    if output is None:
        prospective_id = _payload_id(payload)
        output = repo / "runtime" / "config" / f"{prospective_id}.json"
    return _write_resolved(payload, output)


def load_resolved(path: Path) -> dict[str, Any]:
    payload = _load_json(path.resolve())
    if payload.get("schema") != RESOLVED_SCHEMA:
        raise ConfigError(f"not a {RESOLVED_SCHEMA} file: {path}")
    expected = _payload_id(payload)
    if payload.get("config_id") != expected:
        raise ConfigError(
            f"config_id mismatch: recorded={payload.get('config_id')!r} computed={expected}"
        )
    return payload


def verify(path: Path, site: str) -> dict[str, Any]:
    payload = load_resolved(path)
    if site not in {"jetson", "gpu"}:
        raise ConfigError("site must be jetson or gpu")
    repo = Path(payload["sites"][site]["repository"])
    if not repo.is_dir():
        raise ConfigError(f"configured {site} repository is missing: {repo}")
    revision = _git_revision(repo)
    if revision != payload["source"]["git_revision"]:
        raise ConfigError(
            f"{site} source revision {revision} != config revision "
            f"{payload['source']['git_revision']}"
        )
    if site == "jetson":
        artifacts = [
            ("navigation ImageGoal", payload["navigation"]["image_goal"]),
            ("arrival ImageGoal", payload["arrival"]["image_goal"]),
        ]
        if payload["navigation"]["revisit_image_goal"] is not None:
            artifacts.append(
                ("revisit ImageGoal", payload["navigation"]["revisit_image_goal"])
            )
        verified_paths: set[Path] = set()
        for label, artifact in artifacts:
            goal = Path(artifact["path"])
            if goal in verified_paths:
                continue
            verified_paths.add(goal)
            if not goal.is_file() or _sha256_file(goal) != artifact["sha256"]:
                raise ConfigError(f"{label} changed or disappeared: {goal}")
        for field in ("python", "ros_setup"):
            if not Path(payload["sites"]["jetson"][field]).exists():
                raise ConfigError(f"configured Jetson {field} is missing")
    else:
        gpu = payload["sites"]["gpu"]
        if not Path(gpu["python"]).is_file():
            raise ConfigError(f"configured GPU Python is missing: {gpu['python']}")
        models = gpu["models"]
        file_fields = {"memnav_checkpoint", "lingbot_weights", "navdp_checkpoint"}
        for field, raw in models.items():
            path_value = Path(raw)
            if field in file_fields and not path_value.is_file():
                raise ConfigError(f"configured GPU file is missing: {field}={path_value}")
            if field not in file_fields and not path_value.is_dir():
                raise ConfigError(f"configured GPU directory is missing: {field}={path_value}")
    return payload


def _derive(
    base_path: Path,
    output: Path,
    phase: str,
    dataset_id: str,
    run_root: Path | None,
    *,
    scene_id: str | None = None,
    run_id: str | None = None,
    authority_mode: str | None = None,
    frozen_goal: Path | None = None,
    expected_goal_sha256: str | None = None,
    expected_dataset_sha256: str | None = None,
    collection_mode: str = "manual_long_out_and_back",
) -> Path:
    payload = load_resolved(base_path)
    if payload["experiment"]["profile"] != "fullmono-lingbot-cec":
        raise ConfigError("survey/formal derivation requires the Full-Mono profile")
    if not dataset_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in dataset_id):
        raise ConfigError("invalid dataset id")
    payload.pop("config_id", None)
    payload["source"]["derived_from_config_id"] = load_resolved(base_path)["config_id"]
    payload["experiment"]["phase"] = phase
    if phase == "survey":
        if collection_mode not in {
            "manual_long_out_and_back",
            "manual_one_way_external_goal_debug",
        }:
            raise ConfigError(f"unsupported survey collection mode: {collection_mode}")
        one_way_external_debug = (
            collection_mode == "manual_one_way_external_goal_debug"
        )
        payload["dataset"] = {
            "auto_open": True,
            "id": dataset_id,
            "metadata": {
                "dataset_id": dataset_id,
                "collection_mode": collection_mode,
                "robot": "unitree_go2",
                "motion_authority": "unitree_hand_controller_only",
                "adapter_enabled": False,
                "source_observation_contract": (
                    "memnav_rgbd_source_observation_v1"
                ),
                "candidate_contract": (
                    "external_frozen_goal_only_no_survey_candidate"
                    if one_way_external_debug
                    else "memory_excluded_with_post_guard"
                ),
                "goal_selection_contract": (
                    "operator_frozen_external_required"
                    if one_way_external_debug
                    else "survey_supported_candidate_required"
                ),
                "goal_candidates_required": not one_way_external_debug,
            },
        }
        payload["memory"]["navigate_during_recording"] = False
        # Open the exact-byte dataset during the atomic upstream reset, but do
        # not append camera frames until the operator explicitly starts the
        # Survey.  This gives the Foxglove START SURVEY service a real causal
        # boundary without allowing the browser to start processes or gain
        # motion authority.
        payload["memory"]["pause_recording"] = True
        payload["memory"]["auto_goal_candidate_capture_enabled"] = False
        payload["memory"]["auto_select_goal_candidate"] = True
        payload["launch"]["go2_bridge"] = False
    else:
        if run_root is None:
            raise ConfigError("formal derivation requires --run-root")
        run_root = run_root.resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        if not scene_id or not run_id:
            raise ConfigError("formal derivation requires scene id and run id")
        if authority_mode not in {"native", "cec"}:
            raise ConfigError("formal authority mode must be native or cec")
        if frozen_goal is None or expected_goal_sha256 is None or expected_dataset_sha256 is None:
            raise ConfigError("formal derivation requires frozen goal and expected hashes")
        for label, digest in (
            ("goal", expected_goal_sha256),
            ("dataset", expected_dataset_sha256),
        ):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ConfigError(f"expected {label} SHA-256 must be lowercase hex")
        artifact = _image_artifact(
            Path(payload["source"]["repository"]), str(frozen_goal), "formal frozen goal"
        )
        if artifact["sha256"] != expected_goal_sha256:
            raise ConfigError(
                f"formal frozen goal SHA mismatch: expected {expected_goal_sha256}, "
                f"got {artifact['sha256']}"
            )
        payload["formal"] = {
            "scene_id": scene_id,
            "run_id": run_id,
            "arm": "mono_cec" if authority_mode == "cec" else "mono_native",
            "authority_mode": authority_mode,
            "terminal_approach": payload["cec"].get("terminal_approach", "bearing_only"),
            "expected_goal_sha256": expected_goal_sha256,
            "expected_dataset_sha256": expected_dataset_sha256,
            "runtime_role_visibility": "none",
        }
        payload["cec"]["authority_mode"] = authority_mode
        payload["dataset"] = {
            "auto_open": False,
            "id": None,
            "metadata": {"formal_dataset_id": dataset_id},
        }
        payload["navigation"]["selected_goal_image_path"] = str(
            run_root / "selected_goal.jpg"
        )
        payload["navigation"]["selected_goal_depth_path"] = str(
            run_root / "selected_goal_depth.png"
        )
        payload["navigation"]["revisit_image_goal"] = artifact
        payload["arrival"]["image_goal"] = artifact
        payload["memory"]["navigate_during_recording"] = False
        payload["memory"]["pause_recording"] = True
        payload["memory"]["auto_select_goal_candidate"] = False
        payload["launch"]["go2_bridge"] = True
    return _write_resolved(payload, output)


def _get(payload: Any, dotted: str) -> Any:
    current = payload
    for component in dotted.split("."):
        if not isinstance(current, dict) or component not in current:
            raise ConfigError(f"configuration field does not exist: {dotted}")
        current = current[component]
    return current


def _shell_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def shell_exports(payload: Mapping[str, Any], site: str) -> str:
    if site not in {"jetson", "gpu"}:
        raise ConfigError("site must be jetson or gpu")
    j = payload["sites"]["jetson"]
    g = payload["sites"]["gpu"]
    fields = {
        "CFG_CONFIG_ID": payload["config_id"],
        "CFG_GIT_REVISION": payload["source"]["git_revision"],
        "CFG_EXPERIMENT_ID": payload["experiment"]["id"],
        "CFG_EXPERIMENT_PHASE": payload["experiment"]["phase"],
        "CFG_PROFILE": payload["experiment"]["profile"],
        "CFG_AUTHORITY_MODE": payload["cec"]["authority_mode"],
        "CFG_TERMINAL_APPROACH": payload["cec"].get("terminal_approach", "bearing_only"),
        "CFG_NAV_BACKEND": payload["navigation"]["backend"],
        "CFG_NAV_MODE": payload["navigation"]["mode"],
        "CFG_TWO_PHASE": payload["navigation"]["two_phase"],
        "CFG_IMAGE_GOAL": payload["navigation"]["image_goal"]["path"],
        "CFG_IMAGE_GOAL_SHA256": payload["navigation"]["image_goal"]["sha256"],
        "CFG_REVISIT_IMAGE_GOAL": (payload["navigation"]["revisit_image_goal"] or {}).get("path"),
        "CFG_SELECTED_GOAL_IMAGE": payload["navigation"]["selected_goal_image_path"],
        "CFG_SELECTED_GOAL_DEPTH": payload["navigation"]["selected_goal_depth_path"],
        "CFG_ARRIVAL_MODULE": payload["arrival"]["module"],
        "CFG_ARRIVAL_GOAL": payload["arrival"]["image_goal"]["path"],
        "CFG_ARRIVAL_PHASES": payload["arrival"]["allowed_phases"],
        "CFG_ARRIVAL_RATE_HZ": payload["arrival"]["rate_hz"],
        "CFG_ARRIVAL_CONSECUTIVE": payload["arrival"]["required_consecutive"],
        "CFG_ARRIVAL_MIN_SCALE": payload["arrival"]["min_image_scale"],
        "CFG_ARRIVAL_MAX_SCALE": payload["arrival"]["max_image_scale"],
        "CFG_WITH_CAMERA": payload["launch"]["camera"],
        "CFG_WITH_GO2": payload["launch"]["go2_bridge"],
        "CFG_WITH_FOXGLOVE": payload["launch"]["foxglove"],
        "CFG_CONTROL_PROFILE": payload["control"]["profile"],
        "CFG_MAX_LINEAR_MPS": payload["control"]["max_linear_mps"],
        "CFG_MAX_ANGULAR_RPS": payload["control"]["max_angular_rps"],
        "CFG_NAVIGATE_DURING_RECORDING": payload["memory"]["navigate_during_recording"],
        "CFG_PAUSE_RECORDING": payload["memory"]["pause_recording"],
        "CFG_AUTO_GOAL_INTERVAL": payload["memory"]["auto_goal_candidate_interval_frames"],
        "CFG_AUTO_GOAL_MAX": payload["memory"]["auto_goal_candidate_max"],
        "CFG_AUTO_GOAL_GUARD": payload["memory"]["auto_goal_candidate_post_guard_frames"],
        "CFG_AUTO_GOAL_CAPTURE": payload["memory"]["auto_goal_candidate_capture_enabled"],
        "CFG_AUTO_SELECT_GOAL": payload["memory"]["auto_select_goal_candidate"],
        "CFG_DATASET_AUTO_OPEN": payload["dataset"]["auto_open"],
        "CFG_DATASET_ID": payload["dataset"]["id"],
        "CFG_DATASET_METADATA": payload["dataset"]["metadata"],
        "CFG_CAMERA_HEIGHT_M": payload["stack"]["camera_height_m"],
        "CFG_ADAPTER_PARAMS": payload["stack"]["adapter_params"],
        "CFG_FOXGLOVE_LAYOUT": payload["stack"]["foxglove"]["layout"],
        "CFG_FOXGLOVE_ADDRESS": payload["stack"]["foxglove"]["address"],
        "CFG_FOXGLOVE_PORT": payload["stack"]["foxglove"]["port"],
        "CFG_FOXGLOVE_PREVIEW_RGB_TOPIC": payload["stack"]["foxglove"]["preview"]["rgb_topic"],
        "CFG_FOXGLOVE_PREVIEW_DEPTH_TOPIC": payload["stack"]["foxglove"]["preview"]["depth_topic"],
        "CFG_FOXGLOVE_PREVIEW_GOAL_TOPIC": payload["stack"]["foxglove"]["preview"]["goal_topic"],
        "CFG_FOXGLOVE_PREVIEW_ARRIVAL_TOPIC": payload["stack"]["foxglove"]["preview"]["arrival_topic"],
        "CFG_FOXGLOVE_PREVIEW_STATUS_TOPIC": payload["stack"]["foxglove"]["preview"]["status_topic"],
        "CFG_FOXGLOVE_PREVIEW_WIDTH": payload["stack"]["foxglove"]["preview"]["width"],
        "CFG_FOXGLOVE_PREVIEW_HEIGHT": payload["stack"]["foxglove"]["preview"]["height"],
        "CFG_FOXGLOVE_PREVIEW_STATUS_WIDTH": payload["stack"]["foxglove"]["preview"]["status_width"],
        "CFG_FOXGLOVE_PREVIEW_STATUS_HEIGHT": payload["stack"]["foxglove"]["preview"]["status_height"],
        "CFG_FOXGLOVE_PREVIEW_RGB_FPS": payload["stack"]["foxglove"]["preview"]["rgb_fps"],
        "CFG_FOXGLOVE_PREVIEW_DEPTH_FPS": payload["stack"]["foxglove"]["preview"]["depth_fps"],
        "CFG_FOXGLOVE_PREVIEW_GOAL_FPS": payload["stack"]["foxglove"]["preview"]["goal_fps"],
        "CFG_FOXGLOVE_PREVIEW_ARRIVAL_FPS": payload["stack"]["foxglove"]["preview"]["arrival_fps"],
        "CFG_FOXGLOVE_PREVIEW_STATUS_FPS": payload["stack"]["foxglove"]["preview"]["status_fps"],
        "CFG_FOXGLOVE_PREVIEW_RGB_JPEG_QUALITY": payload["stack"]["foxglove"]["preview"]["rgb_jpeg_quality"],
        "CFG_FOXGLOVE_PREVIEW_DEPTH_JPEG_QUALITY": payload["stack"]["foxglove"]["preview"]["depth_jpeg_quality"],
        "CFG_FOXGLOVE_PREVIEW_GOAL_JPEG_QUALITY": payload["stack"]["foxglove"]["preview"]["goal_jpeg_quality"],
        "CFG_FOXGLOVE_PREVIEW_ARRIVAL_JPEG_QUALITY": payload["stack"]["foxglove"]["preview"]["arrival_jpeg_quality"],
        "CFG_FOXGLOVE_PREVIEW_STATUS_JPEG_QUALITY": payload["stack"]["foxglove"]["preview"]["status_jpeg_quality"],
        "CFG_FOXGLOVE_PREVIEW_ARRIVAL_PRESERVE_RESOLUTION": payload["stack"]["foxglove"]["preview"]["arrival_preserve_resolution"],
        "CFG_FOXGLOVE_PREVIEW_DEPTH_MIN_MM": payload["stack"]["foxglove"]["preview"]["depth_min_mm"],
        "CFG_FOXGLOVE_PREVIEW_DEPTH_MAX_MM": payload["stack"]["foxglove"]["preview"]["depth_max_mm"],
        "CFG_ADAPTER_READY_TIMEOUT_S": payload["stack"]["adapter_ready_timeout_s"],
        "CFG_TUNNEL_READY_TIMEOUT_S": payload["stack"]["tunnel_ready_timeout_s"],
        "CFG_JETSON_PYTHON": j["python"],
        "CFG_JETSON_RUNTIME_ROOT": j["runtime_root"],
        "CFG_ROS_SETUP": j["ros_setup"],
        "CFG_REALSENSE_SETUP": j["realsense_setup"],
        "CFG_MESSAGE_FILTERS_SETUP": j["message_filters_setup"],
        "CFG_NATIVE_SESSION": j["sessions"]["native"],
        "CFG_FULLMONO_SESSION": j["sessions"]["fullmono"],
        "CFG_CAMERA_MIN_FW": j["camera"]["minimum_firmware"],
        "CFG_CAMERA_DEPTH_PROFILE": j["camera"]["depth_profile"],
        "CFG_CAMERA_COLOR_PROFILE": j["camera"]["color_profile"],
        "CFG_CAMERA_READY_TIMEOUT_S": j["camera"]["ready_timeout_s"],
        "CFG_RGB_TOPIC": j["camera"]["rgb_topic"],
        "CFG_DEPTH_TOPIC": j["camera"]["depth_topic"],
        "CFG_CAMERA_INFO_TOPIC": j["camera"]["camera_info_topic"],
        "CFG_UNITREE_NET_IF": j["unitree"]["network_interface"],
        "CFG_UNITREE_SDK_PATH": j["unitree"]["sdk_python_path"],
        "CFG_CYCLONEDDS_HOME": j["unitree"]["cyclonedds_home"],
        "CFG_GO2_PYTHON": j["unitree"]["python"],
        "CFG_GO2_CMD_TOPIC": j["unitree"]["cmd_topic"],
        "CFG_GO2_TIMEOUT_S": j["unitree"]["timeout_s"],
        "CFG_GO2_MAX_VX": j["unitree"]["max_vx"],
        "CFG_GO2_MAX_VY": j["unitree"]["max_vy"],
        "CFG_GO2_MAX_WZ": j["unitree"]["max_wz"],
        "CFG_GO2_MIN_CMD_V": j["unitree"]["min_cmd_v"],
        "CFG_GO2_MIN_CMD_W": j["unitree"]["min_cmd_w"],
        "CFG_NATIVE_HOST": j["native_policy"]["host"],
        "CFG_NATIVE_PORT": j["native_policy"]["port"],
        "CFG_NATIVE_DEVICE": j["native_policy"]["device"],
        "CFG_NATIVE_READY_TIMEOUT_S": j["native_policy"]["ready_timeout_s"],
        "CFG_NATIVE_CHECKPOINT": j["native_policy"]["checkpoint"],
        "CFG_NATIVE_CHECKPOINT_SHA256": j["native_policy"]["checkpoint_sha256"],
        "CFG_GPU_HOST": g["ssh_host"],
        "CFG_GPU_REPO": g["repository"],
        "CFG_GPU_PYTHON": g["python"],
        "CFG_GPU_RUNTIME_ROOT": g["runtime_root"],
        "CFG_GPU_READY_TIMEOUT_S": g["ready_timeout_s"],
        "CFG_GPU_SESSION": g["session"],
        "CFG_MEMNAV_PORT": g["ports"]["memnav"],
        "CFG_NAVDP_PORT": g["ports"]["navdp"],
        "CFG_HUB_PORT": g["ports"]["hub"],
        "CFG_TUNNEL_LOCAL_PORT": g["ports"]["tunnel_local"],
        "CFG_MEMNAV_SOURCE_ROOT": g["models"]["memnav_source_root"],
        "CFG_MEMNAV_CKPT": g["models"]["memnav_checkpoint"],
        "CFG_INTERNNAV_ROOT": g["models"]["internnav_root"],
        "CFG_LINGBOT_REPO": g["models"]["lingbot_repository"],
        "CFG_LINGBOT_WEIGHTS": g["models"]["lingbot_weights"],
        "CFG_LIGHTGLUE_REPO": g["models"]["lightglue_repository"],
        "CFG_DEPENDENCY_ROOT": g["models"]["dependency_root"],
        "CFG_NAVDP_CKPT": g["models"]["navdp_checkpoint"],
        "CFG_GOAL_SCORE_STRIDE": payload["cec"]["goal_score_stride"],
        "CFG_GOAL_MIN_FRAME_GAP": payload["cec"]["goal_min_frame_gap"],
        "CFG_GOAL_MIN_INLIERS": payload["cec"]["goal_min_inliers"],
        "CFG_GOAL_MAX_COS": payload["cec"]["goal_max_cos"],
        "CFG_DATASET_MIN_FRAMES": payload["cec"]["episodic_dataset_min_frames"],
        "CFG_EAGER_DEPTH_CACHE": payload["cec"]["eager_depth_cache"],
    }
    if site == "gpu":
        gpu_keys = {
            "CFG_CONFIG_ID",
            "CFG_GIT_REVISION",
            "CFG_EXPERIMENT_ID",
            "CFG_EXPERIMENT_PHASE",
            "CFG_PROFILE",
            "CFG_AUTHORITY_MODE",
            "CFG_TERMINAL_APPROACH",
            "CFG_DATASET_AUTO_OPEN",
            "CFG_DATASET_ID",
            "CFG_DATASET_METADATA",
            "CFG_CAMERA_HEIGHT_M",
            "CFG_GPU_PYTHON",
            "CFG_GPU_RUNTIME_ROOT",
            "CFG_GPU_READY_TIMEOUT_S",
            "CFG_GPU_SESSION",
            "CFG_MEMNAV_PORT",
            "CFG_NAVDP_PORT",
            "CFG_HUB_PORT",
            "CFG_MEMNAV_SOURCE_ROOT",
            "CFG_MEMNAV_CKPT",
            "CFG_INTERNNAV_ROOT",
            "CFG_LINGBOT_REPO",
            "CFG_LINGBOT_WEIGHTS",
            "CFG_LIGHTGLUE_REPO",
            "CFG_DEPENDENCY_ROOT",
            "CFG_NAVDP_CKPT",
            "CFG_GOAL_SCORE_STRIDE",
            "CFG_GOAL_MIN_FRAME_GAP",
            "CFG_GOAL_MIN_INLIERS",
            "CFG_GOAL_MAX_COS",
            "CFG_DATASET_MIN_FRAMES",
            "CFG_EAGER_DEPTH_CACHE",
        }
        fields = {key: value for key, value in fields.items() if key in gpu_keys}
    return "\n".join(
        f"{key}={shlex.quote(_shell_value(value))}" for key, value in fields.items()
    )


def system_shell_exports(system: Mapping[str, Any], site: str) -> str:
    _validate_system(system)
    if site != "jetson":
        raise ConfigError("system-shell currently supports site=jetson")
    jetson = system["sites"]["jetson"]
    fields = {
        "CFG_JETSON_PYTHON": jetson["python"],
        "CFG_ROS_SETUP": jetson["ros_setup"],
        "CFG_REALSENSE_SETUP": jetson["realsense_setup"],
        "CFG_MESSAGE_FILTERS_SETUP": jetson["message_filters_setup"],
        "CFG_RGB_TOPIC": jetson["camera"]["rgb_topic"],
        "CFG_DEPTH_TOPIC": jetson["camera"]["depth_topic"],
        "CFG_CAPTURE_ROOT": system["stack"]["evidence"]["capture_root"],
        "CFG_CAPTURE_SESSION_PREFIX": system["stack"]["evidence"]["session_prefix"],
        "CFG_NATIVE_SESSION": jetson["sessions"]["native"],
        "CFG_FULLMONO_SESSION": jetson["sessions"]["fullmono"],
        "CFG_GPU_HOST": system["sites"]["gpu"]["ssh_host"],
        "CFG_GPU_SESSION": system["sites"]["gpu"]["session"],
    }
    return "\n".join(
        f"{key}={shlex.quote(_shell_value(value))}" for key, value in fields.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_resolve = sub.add_parser("resolve")
    p_resolve.add_argument("--config", required=True, type=Path)
    p_resolve.add_argument("--output", type=Path)
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--config", required=True, type=Path)
    p_verify.add_argument("--site", required=True, choices=("jetson", "gpu"))
    p_shell = sub.add_parser("shell")
    p_shell.add_argument("--config", required=True, type=Path)
    p_shell.add_argument("--site", required=True, choices=("jetson", "gpu"))
    p_system_shell = sub.add_parser("system-shell")
    p_system_shell.add_argument("--config", required=True, type=Path)
    p_system_shell.add_argument("--site", required=True, choices=("jetson",))
    p_get = sub.add_parser("get")
    p_get.add_argument("--config", required=True, type=Path)
    p_get.add_argument("field")
    p_survey = sub.add_parser("derive-survey")
    p_survey.add_argument("--config", required=True, type=Path)
    p_survey.add_argument("--dataset-id", required=True)
    p_survey.add_argument("--output", required=True, type=Path)
    p_survey.add_argument(
        "--collection-mode",
        choices=(
            "manual_long_out_and_back",
            "manual_one_way_external_goal_debug",
        ),
        default="manual_long_out_and_back",
    )
    p_formal = sub.add_parser("derive-formal")
    p_formal.add_argument("--config", required=True, type=Path)
    p_formal.add_argument("--dataset-id", required=True)
    p_formal.add_argument("--run-root", required=True, type=Path)
    p_formal.add_argument("--scene-id", required=True)
    p_formal.add_argument("--run-id", required=True)
    p_formal.add_argument("--authority-mode", required=True, choices=("native", "cec"))
    p_formal.add_argument("--frozen-goal", required=True, type=Path)
    p_formal.add_argument("--expected-goal-sha256", required=True)
    p_formal.add_argument("--expected-dataset-sha256", required=True)
    p_formal.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        if args.command == "resolve":
            print(resolve(args.config, args.output))
        elif args.command == "verify":
            payload = verify(args.config, args.site)
            print(f"config_id={payload['config_id']} site={args.site} verified=true")
        elif args.command == "shell":
            print(shell_exports(load_resolved(args.config), args.site))
        elif args.command == "system-shell":
            print(system_shell_exports(_load_json(args.config.resolve()), args.site))
        elif args.command == "get":
            value = _get(load_resolved(args.config), args.field)
            if isinstance(value, (dict, list, bool)) or value is None:
                print(json.dumps(value, ensure_ascii=False, sort_keys=True))
            else:
                print(value)
        elif args.command == "derive-survey":
            print(_derive(
                args.config,
                args.output,
                "survey",
                args.dataset_id,
                None,
                collection_mode=args.collection_mode,
            ))
        else:
            print(_derive(
                args.config,
                args.output,
                "formal",
                args.dataset_id,
                args.run_root,
                scene_id=args.scene_id,
                run_id=args.run_id,
                authority_mode=args.authority_mode,
                frozen_goal=args.frozen_goal,
                expected_goal_sha256=args.expected_goal_sha256,
                expected_dataset_sha256=args.expected_dataset_sha256,
            ))
        return 0
    except ConfigError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
