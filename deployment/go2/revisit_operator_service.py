#!/usr/bin/env python3
"""Persistent Foxglove orchestration for complete Survey/Revisit episodes.

The operator-facing sequence is intentionally fixed: capture an immutable RGB-D
goal, start a new Survey, seal it, then run Revisit.  Every action is persisted
and recorded, while the browser is never allowed to supply paths, identifiers,
commands, or motion parameters.  ``start_revisit`` is the sole motion-authority
action and still delegates arming to the fail-closed navigation runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np


ACTIVE_STATE_SCHEMA = "memnav_revisit_debug_state_v1"
STATUS_SCHEMA = "memnav_revisit_operator_status_v1"
EPISODE_SCHEMA = "memnav_foxglove_episode_v1"
GOAL_CAPTURE_SCHEMA = "memnav_revisit_goal_capture_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_EPISODE_STATES = {"complete", "failed", "cancelled", "stopped"}


def stop_reason_from_legacy_outcome(outcome: str) -> str:
    return {"success": "automatic_arrival", "failure": "navigation_error",
            "aborted": "operator_stop"}.get(outcome, outcome)
BUSY_EPISODE_STATES = {
    "capturing_goal",
    "survey_preparing",
    "survey_stopping",
    "revisit_preparing",
    "revisiting",
    "finalizing",
    "stopping",
}


class ContractError(RuntimeError):
    """The frozen Revisit state is not safe to execute."""


@dataclass(frozen=True)
class StartContract:
    dataset_id: str
    goal_path: Path
    goal_sha256: str
    experiment_path: Path
    mode: str
    seal_receipt_path: Path
    dataset_manifest_sha256: str


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ContractError(f"{label} is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is invalid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _required_string(value: dict[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ContractError(f"{label} requires non-empty {key}")
    return item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _image_rows(message: Any, bytes_per_pixel: int) -> np.ndarray:
    width = int(message.width)
    height = int(message.height)
    expected_step = width * bytes_per_pixel
    step = int(message.step)
    if width <= 0 or height <= 0 or step < expected_step:
        raise ContractError("camera frame has invalid dimensions or row stride")
    expected_size = height * step
    data = memoryview(message.data)
    if data.nbytes < expected_size:
        raise ContractError(
            f"camera frame contains {data.nbytes} bytes; expected {expected_size}"
        )
    return np.frombuffer(data[:expected_size], dtype=np.uint8).reshape(height, step)


def _rgb_to_bgr(message: Any) -> np.ndarray:
    encoding = str(message.encoding).lower()
    if encoding in {"rgb8", "bgr8", "8uc3"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8", "8uc4"}:
        channels = 4
    else:
        raise ContractError(f"unsupported RGB encoding: {message.encoding}")
    packed = _image_rows(message, channels)[:, : int(message.width) * channels]
    image = np.ascontiguousarray(
        packed.reshape(int(message.height), int(message.width), channels)
    )
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding in {"rgba8", "8uc4"}:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _depth_to_u16(message: Any) -> np.ndarray:
    if str(message.encoding).lower() not in {"16uc1", "mono16"}:
        raise ContractError(f"unsupported depth encoding: {message.encoding}")
    packed = np.ascontiguousarray(
        _image_rows(message, 2)[:, : int(message.width) * 2]
    )
    byte_order = ">u2" if bool(message.is_bigendian) else "<u2"
    depth = packed.view(np.dtype(byte_order)).reshape(
        int(message.height), int(message.width)
    )
    return np.ascontiguousarray(depth.astype(np.uint16, copy=False))


def image_stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def episode_identity(now: datetime | None = None) -> tuple[str, str]:
    instant = now or datetime.now(timezone.utc)
    token = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"episode_{token}", f"m_episode_{token}"


def allowed_actions_for_state(state: str | None, *, busy: bool) -> list[str]:
    if busy:
        return ["stop-navigation"]
    if state in TERMINAL_EPISODE_STATES:
        return ["capture-goal", "stop-navigation"]
    if not state:
        return ["capture-goal"]
    if state == "goal_captured":
        return ["start-survey", "stop-navigation"]
    if state == "surveying":
        return ["stop-survey", "stop-navigation"]
    if state == "survey_sealed":
        return ["revisit", "stop-navigation"]
    return ["stop-navigation"]


def freeze_goal_pair(
    *,
    rgb_message: Any,
    depth_message: Any,
    episode_dir: Path,
    rgb_topic: str,
    depth_topic: str,
    captured_utc: str,
) -> dict[str, Any]:
    """Losslessly freeze one aligned RGB-D pair and its sensor timestamps."""

    rgb_stamp = image_stamp_ns(rgb_message)
    depth_stamp = image_stamp_ns(depth_message)
    delta_ns = abs(rgb_stamp - depth_stamp)
    if delta_ns > 100_000_000:
        raise ContractError(
            f"RGB/depth timestamps differ by {delta_ns / 1e6:.1f} ms"
        )
    color = _rgb_to_bgr(rgb_message)
    depth = _depth_to_u16(depth_message)
    if color.shape[:2] != depth.shape[:2]:
        raise ContractError(
            f"RGB/depth dimensions differ: {color.shape[:2]} vs {depth.shape[:2]}"
        )
    color_ok, color_png = cv2.imencode(".png", color)
    depth_ok, depth_png = cv2.imencode(".png", depth)
    if not color_ok or not depth_ok:
        raise ContractError("OpenCV could not encode the frozen RGB-D goal")
    goal_path = episode_dir / "revisit_goal.png"
    depth_path = episode_dir / "revisit_goal_depth.png"
    _atomic_write_bytes(goal_path, color_png.tobytes())
    _atomic_write_bytes(depth_path, depth_png.tobytes())
    return {
        "schema": GOAL_CAPTURE_SCHEMA,
        "captured_utc": captured_utc,
        "pair_delta_ms": delta_ns / 1e6,
        "rgb": {
            "topic": rgb_topic,
            "stamp_ns": rgb_stamp,
            "stamp": {
                "sec": int(rgb_message.header.stamp.sec),
                "nanosec": int(rgb_message.header.stamp.nanosec),
            },
            "frame_id": str(rgb_message.header.frame_id),
            "encoding": str(rgb_message.encoding),
            "width": int(rgb_message.width),
            "height": int(rgb_message.height),
            "path": str(goal_path.resolve()),
            "sha256": _sha256(goal_path),
            "policy_goal_authority": True,
        },
        "depth": {
            "topic": depth_topic,
            "stamp_ns": depth_stamp,
            "stamp": {
                "sec": int(depth_message.header.stamp.sec),
                "nanosec": int(depth_message.header.stamp.nanosec),
            },
            "frame_id": str(depth_message.header.frame_id),
            "encoding": str(depth_message.encoding),
            "width": int(depth_message.width),
            "height": int(depth_message.height),
            "path": str(depth_path.resolve()),
            "sha256": _sha256(depth_path),
            "policy_goal_authority": False,
        },
    }


def validate_start_contract(repo_root: Path, state_path: Path) -> StartContract:
    """Validate the fixed, argument-free contract used by the start service."""
    state = _load_object(state_path, "active Revisit state")
    if state.get("schema") != ACTIVE_STATE_SCHEMA:
        raise ContractError("active Revisit state has an unsupported schema")

    dataset_id = _required_string(state, "dataset_id", "active Revisit state")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", dataset_id) is None:
        raise ContractError("active Revisit dataset_id is invalid")
    mode = _required_string(state, "mode", "active Revisit state")
    if mode not in {"prepared", "recording", "sealed"}:
        raise ContractError(f"Revisit requires a stopped Survey, not mode={mode}")

    goal_path = Path(_required_string(state, "goal_path", "active Revisit state"))
    experiment_path = Path(
        _required_string(state, "experiment_path", "active Revisit state")
    )
    goal_sha256 = _required_string(
        state, "goal_sha256", "active Revisit state"
    )
    if SHA256_RE.fullmatch(goal_sha256) is None:
        raise ContractError("active Revisit goal_sha256 is invalid")
    if not goal_path.is_file():
        raise ContractError(f"frozen Revisit goal is missing: {goal_path}")
    if _sha256(goal_path) != goal_sha256:
        raise ContractError("frozen Revisit goal SHA-256 changed")
    if not experiment_path.is_file():
        raise ContractError(f"Revisit experiment is missing: {experiment_path}")

    seal_receipt_path = (
        repo_root
        / "runtime/go2/two_pass_revisit"
        / dataset_id
        / "survey_seal.json"
    )
    receipt = _load_object(seal_receipt_path, "Survey stop receipt")
    if receipt.get("dataset_id") != dataset_id:
        raise ContractError("Survey stop receipt belongs to a different dataset")
    if receipt.get("recording_active") is not False:
        raise ContractError("Survey stop receipt still reports active recording")
    if receipt.get("motion_enabled") is not False or receipt.get("estop") is not True:
        raise ContractError("Survey stop receipt does not prove disabled + estop")
    if receipt.get("evaluation_depth_consumed_by_policy") is not False:
        raise ContractError("Survey stop receipt violates RGB-only evaluation")
    if int(receipt.get("goal_memory_exact_sha_overlap", -1)) != 0:
        raise ContractError("Survey stop receipt reports goal/memory SHA overlap")
    manifest_sha256 = str(receipt.get("manifest_sha256") or "")
    if SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ContractError("Survey stop receipt has an invalid manifest SHA-256")
    state_manifest = state.get("dataset_manifest_sha256")
    if mode == "sealed" and state_manifest != manifest_sha256:
        raise ContractError("active state and Survey stop receipt manifest differ")

    return StartContract(
        dataset_id=dataset_id,
        goal_path=goal_path,
        goal_sha256=goal_sha256,
        experiment_path=experiment_path,
        mode=mode,
        seal_receipt_path=seal_receipt_path,
        dataset_manifest_sha256=manifest_sha256,
    )


def prepare_command(repo_root: Path, run_id: str) -> list[str]:
    return [
        "bash",
        str(repo_root / "deployment/go2/offboard/revisit_debug.sh"),
        "revisit-prepare",
        "--run-id",
        run_id,
    ]


def survey_prepare_command(
    repo_root: Path, dataset_id: str, goal_path: Path
) -> list[str]:
    return [
        "bash",
        str(repo_root / "deployment/go2/offboard/revisit_debug.sh"),
        "record-prepare",
        dataset_id,
        "--goal",
        str(goal_path),
        "--point-label",
        "M",
    ]


def survey_stop_command(repo_root: Path) -> list[str]:
    return [
        "bash",
        str(repo_root / "deployment/go2/offboard/revisit_debug.sh"),
        "record-stop",
    ]


def capture_start_command(
    repo_root: Path, episode_id: str, dataset_id: str
) -> list[str]:
    return [
        "bash",
        str(repo_root / "deployment/go2/offboard/experiment_capture.sh"),
        "start",
        episode_id,
        "--dataset",
        dataset_id,
        "--trial-kind",
        "revisit",
        "--profile",
        "full",
        "--allow-observer",
        "--onboard-episode",
    ]


def capture_stop_command(repo_root: Path, episode_id: str) -> list[str]:
    return [
        "bash",
        str(repo_root / "deployment/go2/offboard/experiment_capture.sh"),
        "stop",
        episode_id,
    ]


def capture_pause_command(repo_root: Path, episode_id: str) -> list[str]:
    return [
        "bash",
        str(repo_root / "deployment/go2/offboard/experiment_capture.sh"),
        "pause",
        episode_id,
    ]


def capture_resume_command(repo_root: Path, episode_id: str) -> list[str]:
    return [
        "bash",
        str(repo_root / "deployment/go2/offboard/experiment_capture.sh"),
        "resume",
        episode_id,
    ]


def capture_resume_survey_command(repo_root: Path, episode_id: str) -> list[str]:
    return [
        "bash",
        str(repo_root / "deployment/go2/offboard/experiment_capture.sh"),
        "resume-survey",
        episode_id,
    ]


def capture_finalize_command(
    repo_root: Path, episode_id: str, outcome: str, *, allow_incomplete: bool
) -> list[str]:
    command = [
        "bash",
        str(repo_root / "deployment/go2/offboard/experiment_capture.sh"),
        "finalize",
        episode_id,
        "unreviewed",
        "--termination-reason",
        stop_reason_from_legacy_outcome(outcome),
        "--notes",
        "Foxglove-managed phase-gated RGB-D Episode",
    ]
    if allow_incomplete:
        command.append("--allow-incomplete")
    return command


def navigation_command(
    repo_root: Path, formal_config: Path, timeout_s: float
) -> list[str]:
    return [
        "bash",
        str(repo_root / "deployment/go2/scripts/run_navigation.sh"),
        "--config",
        str(formal_config),
        "--timeout-s",
        f"{timeout_s:g}",
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_realsense_link(camera_summary: str, usb_tree: str) -> None:
    if re.search(r"Intel RealSense D435I?", camera_summary, re.IGNORECASE) is None:
        raise ContractError("RealSense D435i was not enumerated")
    video_speeds = [
        int(speed)
        for speed in re.findall(
            r"Class=Video[^\n]*?,\s*([0-9]+)M", usb_tree, re.IGNORECASE
        )
    ]
    if not video_speeds or max(video_speeds) < 5000:
        raise ContractError("RealSense video interfaces are not on USB SuperSpeed")


class RevisitOperatorService:
    def __init__(
        self,
        *,
        repo_root: Path,
        state_path: Path,
        episodes_root: Path,
        capture_root: Path,
        capture_session_prefix: str,
        rgb_topic: str,
        depth_topic: str,
        timeout_s: float,
        robot_ip: str,
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image
        from std_msgs.msg import Bool, String
        from std_srvs.srv import Trigger

        class NodeImpl(Node):
            pass

        self.rclpy = rclpy
        self.Bool = Bool
        self.String = String
        self.Image = Image
        self.Trigger = Trigger
        self.repo_root = repo_root.resolve()
        self.state_path = state_path.resolve()
        self.episodes_root = episodes_root.resolve()
        self.capture_root = capture_root.resolve()
        self.capture_session_prefix = capture_session_prefix
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.timeout_s = float(timeout_s)
        self.robot_ip = robot_ip
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self.active_episode_path = self.episodes_root / "active.json"
        self.node = NodeImpl("memnav_revisit_operator")
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        command_qos = QoSProfile(depth=10)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.status_pub = self.node.create_publisher(
            String, "/navdp/operator/revisit_workflow", state_qos
        )
        self.episode_event_pub = self.node.create_publisher(
            String, "/navdp/operator/episode_event", state_qos
        )
        self.goal_pub = self.node.create_publisher(
            Image, "/navdp/image_goal", state_qos
        )
        self.enabled_pub = self.node.create_publisher(
            Bool, "/navdp/enabled", command_qos
        )
        self.estop_pub = self.node.create_publisher(
            Bool, "/navdp/estop", command_qos
        )
        self.adapter_stop = self.node.create_client(
            Trigger, "/navdp_go2_adapter/operator_stop"
        )
        self.node.create_subscription(Image, rgb_topic, self._on_rgb, sensor_qos)
        self.node.create_subscription(Image, depth_topic, self._on_depth, sensor_qos)
        self.node.create_service(
            Trigger, "/memnav_operator/capture_goal", self._capture_goal
        )
        self.node.create_service(
            Trigger, "/memnav_operator/start_survey", self._start_survey
        )
        self.node.create_service(
            Trigger, "/memnav_operator/stop_survey", self._stop_survey
        )
        self.node.create_service(
            Trigger, "/memnav_operator/start_revisit", self._start_revisit
        )
        self.node.create_service(
            Trigger, "/memnav_operator/operator_stop", self._operator_stop
        )
        self.node.create_timer(0.5, self._tick)

        self._mutex = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._cancel = threading.Event()
        self._lock_until = 0.0
        self._latest_rgb: tuple[Any, float] | None = None
        self._latest_depth: tuple[Any, float] | None = None
        self._episode: dict[str, Any] | None = self._restore_episode()
        self._status: dict[str, Any] = {
            "schema": STATUS_SCHEMA,
            "state": "idle" if self._episode is None else str(self._episode["state"]),
            "detail": self._initial_detail(),
            "active": False,
            "updated_utc": utc_now(),
        }
        if self._episode is not None:
            self._status.update(self._episode_context())
        self._status["allowed_actions"] = allowed_actions_for_state(
            None if self._episode is None else str(self._episode["state"]),
            busy=False,
        )
        self._publish_goal()
        self._tick()

    def _restore_episode(self) -> dict[str, Any] | None:
        if not self.active_episode_path.is_file():
            return None
        try:
            episode = _load_object(self.active_episode_path, "active Episode")
            if episode.get("schema") != EPISODE_SCHEMA:
                raise ContractError("active Episode has an unsupported schema")
            _required_string(episode, "episode_id", "active Episode")
            _required_string(episode, "dataset_id", "active Episode")
            _required_string(episode, "state", "active Episode")
            return episode
        except (ContractError, OSError, ValueError) as exc:
            self.node.get_logger().error(f"Cannot restore active Episode: {exc}")
            return None

    def _initial_detail(self) -> str:
        if self._episode is None:
            return "At the destination, capture a Revisit goal"
        state = str(self._episode.get("state") or "")
        if state == "goal_captured":
            return "Goal captured · return to the Survey start"
        if state == "surveying":
            return "Survey recording · drive with the Unitree controller"
        if state == "survey_sealed":
            return "Survey sealed · ready for Revisit"
        if state in BUSY_EPISODE_STATES:
            return "Previous action was interrupted · press STOP"
        return str(self._episode.get("detail") or "Ready for a new Episode")

    def _episode_context(self) -> dict[str, Any]:
        if self._episode is None:
            return {}
        goal = self._episode.get("goal")
        goal = goal if isinstance(goal, dict) else {}
        return {
            "episode_id": self._episode.get("episode_id"),
            "dataset_id": self._episode.get("dataset_id"),
            "episode_state": self._episode.get("state"),
            "termination_reason": self._episode.get("termination_reason"),
            "goal_captured_utc": goal.get("captured_utc"),
            "goal_rgb_stamp_ns": (goal.get("rgb") or {}).get("stamp_ns")
            if isinstance(goal.get("rgb"), dict)
            else None,
            "capture_active": bool(self._episode.get("capture_active")),
            "capture_root": self._episode.get("capture_root"),
        }

    def _set_status(
        self, state: str, detail: str, *, active: bool | None = None, **fields: Any
    ) -> None:
        if active is None:
            active = state in BUSY_EPISODE_STATES
        episode_state = (
            None if self._episode is None else str(self._episode.get("state") or "")
        )
        with self._mutex:
            self._status = {
                "schema": STATUS_SCHEMA,
                "state": state,
                "detail": detail,
                "active": bool(active),
                "updated_utc": utc_now(),
                **self._episode_context(),
                "allowed_actions": allowed_actions_for_state(
                    episode_state, busy=bool(active)
                ),
                **fields,
            }
        self._publish_status()

    def _publish_status(self) -> None:
        with self._mutex:
            payload = dict(self._status)
        message = self.String()
        message.data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.status_pub.publish(message)

    def _assert_motion_lock(self) -> None:
        disabled = self.Bool()
        disabled.data = False
        estop = self.Bool()
        estop.data = True
        self.enabled_pub.publish(disabled)
        self.estop_pub.publish(estop)

    def _request_adapter_stop(self) -> None:
        self._assert_motion_lock()
        if self.adapter_stop.service_is_ready():
            self.adapter_stop.call_async(self.Trigger.Request())

    def _tick(self) -> None:
        if time.monotonic() < self._lock_until:
            self._assert_motion_lock()
        self._publish_status()

    def _on_rgb(self, message: Any) -> None:
        with self._mutex:
            self._latest_rgb = (message, time.monotonic())

    def _on_depth(self, message: Any) -> None:
        with self._mutex:
            self._latest_depth = (message, time.monotonic())

    def _fresh_goal_pair(self) -> tuple[Any, Any]:
        now = time.monotonic()
        with self._mutex:
            rgb = self._latest_rgb
            depth = self._latest_depth
        if rgb is None or depth is None:
            raise ContractError("Waiting for live aligned RGB-D frames")
        if now - rgb[1] > 1.0 or now - depth[1] > 1.0:
            raise ContractError("Live RGB-D frames are stale")
        delta_ms = abs(image_stamp_ns(rgb[0]) - image_stamp_ns(depth[0])) / 1e6
        if delta_ms > 100.0:
            raise ContractError(f"RGB/depth timestamps differ by {delta_ms:.1f} ms")
        return rgb[0], depth[0]

    def _episode_dir(self, episode_id: str | None = None) -> Path:
        identity = episode_id
        if identity is None:
            if self._episode is None:
                raise ContractError("No active Episode")
            identity = _required_string(self._episode, "episode_id", "active Episode")
        return self.episodes_root / identity

    def _write_episode(self) -> None:
        if self._episode is None:
            return
        episode_dir = self._episode_dir()
        _atomic_write_json(episode_dir / "episode.json", self._episode)
        _atomic_write_json(self.active_episode_path, self._episode)

    def _append_episode_event(self, event: str, **fields: Any) -> None:
        if self._episode is None:
            return
        payload = {
            "schema": "memnav_foxglove_episode_event_v1",
            "event": event,
            "utc": utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "episode_id": self._episode.get("episode_id"),
            "dataset_id": self._episode.get("dataset_id"),
            **fields,
        }
        event_path = self._episode_dir() / "events.jsonl"
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        message = self.String()
        message.data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.episode_event_pub.publish(message)

    def _transition(self, state: str, detail: str, **fields: Any) -> None:
        if self._episode is None:
            raise ContractError("No active Episode")
        if "outcome" in fields:
            fields["termination_reason"] = stop_reason_from_legacy_outcome(fields["outcome"])
            fields["outcome"] = "unreviewed"
        if state in TERMINAL_EPISODE_STATES:
            state = "stopped"
            detail += " · Saved for human review (not automatically scored)"
        self._episode = {
            **self._episode,
            "state": state,
            "detail": detail,
            "updated_utc": utc_now(),
            **fields,
        }
        self._write_episode()
        self._append_episode_event(state, detail=detail)
        self._set_status(state, detail)

    def _publish_goal(self) -> None:
        if self._episode is None:
            return
        goal = self._episode.get("goal")
        if not isinstance(goal, dict) or not isinstance(goal.get("rgb"), dict):
            return
        rgb = goal["rgb"]
        path = Path(str(rgb.get("path") or ""))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self.node.get_logger().error(f"Cannot publish frozen goal: {path}")
            return
        message = self.Image()
        stamp = rgb.get("stamp") if isinstance(rgb.get("stamp"), dict) else {}
        message.header.stamp.sec = int(stamp.get("sec") or 0)
        message.header.stamp.nanosec = int(stamp.get("nanosec") or 0)
        message.header.frame_id = str(rgb.get("frame_id") or "camera_color_optical_frame")
        message.height, message.width = image.shape[:2]
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = int(message.width) * 3
        message.data = image.tobytes()
        self.goal_pub.publish(message)

    def _worker_is_active(self) -> bool:
        with self._mutex:
            return self._worker is not None and self._worker.is_alive()

    def _start_worker(self, target: Any, name: str, *args: Any) -> None:
        self._cancel.clear()

        def entry() -> None:
            try:
                target(*args)
            finally:
                with self._mutex:
                    if self._worker is threading.current_thread():
                        self._worker = None

        worker = threading.Thread(target=entry, name=name, daemon=True)
        with self._mutex:
            self._worker = worker
        worker.start()

    def _capture_goal(self, _request: Any, response: Any) -> Any:
        if self._worker_is_active():
            response.success = False
            response.message = "Another Episode action is still running"
            return response
        if (
            self._episode is not None
            and self._episode.get("state") not in TERMINAL_EPISODE_STATES
        ):
            response.success = False
            response.message = "Finish or stop the current Episode first"
            return response
        try:
            rgb, depth = self._fresh_goal_pair()
        except (ContractError, ValueError) as exc:
            response.success = False
            response.message = str(exc)
            self._set_status("blocked", str(exc), active=False)
            return response
        episode_id, dataset_id = episode_identity()
        self._start_worker(
            self._run_goal_capture,
            "memnav-goal-capture",
            episode_id,
            dataset_id,
            rgb,
            depth,
        )
        response.success = True
        response.message = "Capturing the current RGB-D frame as Revisit goal"
        return response

    def _run_goal_capture(
        self,
        episode_id: str,
        dataset_id: str,
        rgb_message: Any,
        depth_message: Any,
    ) -> None:
        captured_utc = utc_now()
        episode_dir = self._episode_dir(episode_id)
        try:
            episode_dir.mkdir(parents=True, exist_ok=False)
            self._episode = {
                "schema": EPISODE_SCHEMA,
                "episode_id": episode_id,
                "dataset_id": dataset_id,
                "state": "capturing_goal",
                "detail": "Freezing the current RGB-D goal",
                "created_utc": captured_utc,
                "updated_utc": utc_now(),
                "goal": None,
                "capture_profile": "full",
                "capture_root": str((self.capture_root / episode_id).resolve()),
                "capture_active": False,
                "capture_phase": "awaiting_survey",
                "outcome": None,
            }
            self._write_episode()
            self._set_status("capturing_goal", "Freezing the current RGB-D goal")
            goal = freeze_goal_pair(
                rgb_message=rgb_message,
                depth_message=depth_message,
                episode_dir=episode_dir,
                rgb_topic=self.rgb_topic,
                depth_topic=self.depth_topic,
                captured_utc=captured_utc,
            )
            _atomic_write_json(episode_dir / "goal_capture.json", goal)
            self._episode = {**self._episode, "updated_utc": utc_now(), "goal": goal}
            self._write_episode()
            self._append_episode_event("goal_frame_frozen")
            self._publish_goal()
            self._transition(
                "goal_captured",
                "Goal captured · return to the Survey start",
                capture_active=False,
                capture_phase="awaiting_survey",
            )
            self._publish_goal()
        except Exception as exc:
            if self._episode is not None:
                self._transition(
                    "failed",
                    f"Goal capture failed: {exc}",
                    capture_active=False,
                    outcome="system_failure",
                )
            else:
                self._set_status("failed", f"Goal capture failed: {exc}", active=False)

    def _start_survey(self, _request: Any, response: Any) -> Any:
        if self._worker_is_active():
            response.success = False
            response.message = "Another Episode action is still running"
            return response
        if self._episode is None or self._episode.get("state") != "goal_captured":
            response.success = False
            response.message = "Capture a Revisit goal before starting Survey"
            return response
        self._start_worker(self._run_survey_start, "memnav-survey-start")
        response.success = True
        response.message = "Survey preparation started"
        return response

    def _run_survey_start(self) -> None:
        assert self._episode is not None
        dataset_id = _required_string(self._episode, "dataset_id", "active Episode")
        goal = self._episode.get("goal")
        if not isinstance(goal, dict) or not isinstance(goal.get("rgb"), dict):
            self._transition("failed", "Frozen Revisit goal is missing")
            return
        goal_path = Path(str(goal["rgb"].get("path") or ""))
        log_path = self._episode_dir() / "operator.log"
        try:
            self._transition(
                "survey_preparing", "Starting the reusable RTX Survey stack"
            )
            self._lock_until = time.monotonic() + 30.0
            self._request_adapter_stop()
            with log_path.open("ab", buffering=0) as log:
                code = self._run_command(
                    survey_prepare_command(self.repo_root, dataset_id, goal_path), log
                )
            if code != 0 or self._cancel.is_set():
                raise ContractError(f"Survey preparation exited with code {code}")
            active = _load_object(self.state_path, "active Revisit state")
            if active.get("dataset_id") != dataset_id or active.get("mode") != "prepared":
                raise ContractError("Survey stack prepared a different Dataset")
            active.update(
                {
                    "episode_id": self._episode["episode_id"],
                    "goal_capture_receipt": str(
                        (self._episode_dir() / "goal_capture.json").resolve()
                    ),
                    "capture_run_id": self._episode["episode_id"],
                }
            )
            _atomic_write_json(self.state_path, active)
            self._publish_goal()
            self._set_status(
                "survey_preparing", "Survey ready · starting RGB-D capture"
            )
            with log_path.open("ab", buffering=0) as log:
                code = self._run_command(
                    capture_start_command(
                        self.repo_root,
                        str(self._episode["episode_id"]),
                        dataset_id,
                    ),
                    log,
                )
            if code != 0 or self._cancel.is_set():
                raise ContractError(f"Survey RGB-D recorder exited with code {code}")
            self._call_trigger_service("/navdp_go2_adapter/survey_start", 20.0)
            self._assert_motion_lock()
            self._transition(
                "surveying",
                "Survey recording · drive with the Unitree controller",
                survey_started_utc=utc_now(),
                capture_active=True,
                capture_phase="survey",
                survey_capture_started_utc=utc_now(),
            )
        except Exception as exc:
            self._lock_until = time.monotonic() + 30.0
            self._request_adapter_stop()
            self._finish_capture("system_failure")
            self._cleanup_stack_path(log_path)
            self._transition(
                "failed",
                f"Survey start failed: {exc}",
                capture_active=False,
                outcome="system_failure",
            )

    def _stop_survey(self, _request: Any, response: Any) -> Any:
        if self._worker_is_active():
            response.success = False
            response.message = "Another Episode action is still running"
            return response
        if self._episode is None or self._episode.get("state") != "surveying":
            response.success = False
            response.message = "Survey is not recording"
            return response
        self._start_worker(self._run_survey_stop, "memnav-survey-stop")
        response.success = True
        response.message = "Stopping and validating Survey"
        return response

    def _run_survey_stop(self) -> None:
        log_path = self._episode_dir() / "operator.log"
        paused = False
        try:
            self._transition("survey_stopping", "Stopping Survey RGB-D capture")
            with log_path.open("ab", buffering=0) as log:
                code = self._run_command(
                    capture_pause_command(
                        self.repo_root, str(self._episode["episode_id"])
                    ),
                    log,
                )
            if code != 0:
                raise ContractError(f"Survey RGB-D recorder pause exited with code {code}")
            paused = True
            self._episode = {
                **self._episode,
                "capture_active": False,
                "capture_phase": "paused_between_segments",
                "survey_capture_stopped_utc": utc_now(),
                "detail": "Survey RGB-D stopped · sealing and validating data",
                "updated_utc": utc_now(),
            }
            self._write_episode()
            self._set_status(
                "survey_stopping",
                "Survey RGB-D stopped · sealing and validating data",
            )
            self._lock_until = time.monotonic() + 30.0
            self._request_adapter_stop()
            with log_path.open("ab", buffering=0) as log:
                code = self._run_command(survey_stop_command(self.repo_root), log)
            if self._cancel.is_set():
                raise ContractError("Survey stop was cancelled")
            if code != 0:
                with log_path.open("ab", buffering=0) as log:
                    resume_code = self._run_command(
                        capture_resume_survey_command(
                            self.repo_root, str(self._episode["episode_id"])
                        ),
                        log,
                    )
                if resume_code != 0:
                    raise ContractError(
                        f"Survey stop failed and RGB-D resume exited with code {resume_code}"
                    )
                self._call_trigger_service("/navdp_go2_adapter/survey_start", 20.0)
                paused = False
                self._transition(
                    "surveying",
                    "Survey is not ready to stop · continue until at least 40 frames",
                    capture_active=True,
                    capture_phase="survey",
                )
                return
            active = _load_object(self.state_path, "active Revisit state")
            if active.get("mode") != "sealed":
                raise ContractError("Survey stop did not commit a sealed Dataset")
            self._transition(
                "survey_sealed",
                "Survey sealed · ready for Revisit",
                survey_stopped_utc=utc_now(),
                dataset_manifest_sha256=active.get("dataset_manifest_sha256"),
            )
        except Exception as exc:
            if self._cancel.is_set():
                self._abort_episode("Stopped by operator")
            else:
                if paused:
                    try:
                        with log_path.open("ab", buffering=0) as log:
                            resume_code = self._run_command(
                                capture_resume_survey_command(
                                    self.repo_root, str(self._episode["episode_id"])
                                ),
                                log,
                            )
                        if resume_code != 0:
                            raise ContractError(
                                f"RGB-D resume exited with code {resume_code}"
                            )
                        self._call_trigger_service(
                            "/navdp_go2_adapter/survey_start", 20.0
                        )
                    except Exception as resume_exc:
                        self._finish_capture("system_failure")
                        self._cleanup_stack_path(log_path)
                        self._transition(
                            "failed",
                            f"Survey stop failed and capture could not resume: {resume_exc}",
                            capture_active=False,
                            outcome="system_failure",
                        )
                        return
                self._transition(
                    "surveying",
                    f"Survey stop failed: {exc}",
                    capture_active=True,
                    capture_phase="survey",
                )

    def _start_revisit(self, _request: Any, response: Any) -> Any:
        if self._worker_is_active():
            response.success = False
            response.message = "Another Episode action is still running"
            return response
        if self._episode is None or self._episode.get("state") != "survey_sealed":
            response.success = False
            response.message = "Stop and seal Survey before starting Revisit"
            return response
        try:
            contract = validate_start_contract(self.repo_root, self.state_path)
            if contract.dataset_id != self._episode.get("dataset_id"):
                raise ContractError("Sealed Dataset belongs to a different Episode")
            goal = self._episode.get("goal")
            expected_goal = (
                (goal.get("rgb") or {}).get("sha256")
                if isinstance(goal, dict)
                else None
            )
            if contract.goal_sha256 != expected_goal:
                raise ContractError("Frozen goal belongs to a different Episode")
        except (ContractError, OSError, ValueError) as exc:
            response.success = False
            response.message = str(exc)
            self._set_status("blocked", str(exc), active=False)
            return response

        self._lock_until = time.monotonic() + 2.0
        self._request_adapter_stop()
        self._start_worker(
            self._run_transaction,
            "memnav-revisit-transaction",
            contract,
        )
        response.success = True
        response.message = "Revisit accepted; locked stack preparation started"
        return response

    def _operator_stop(self, _request: Any, response: Any) -> Any:
        self._cancel.set()
        self._lock_until = time.monotonic() + 30.0
        self._request_adapter_stop()
        active = self._worker_is_active()
        if active:
            self._set_status(
                "stopping", "Stopping Episode · disabled + estop asserted"
            )
        elif (
            self._episode is not None
            and self._episode.get("state") not in TERMINAL_EPISODE_STATES
        ):
            self._start_worker(
                self._abort_episode,
                "memnav-episode-stop",
                "Stopped by operator",
            )
        else:
            self._set_status(
                "stopped", "No active Episode · motion lock asserted", active=False
            )
        response.success = True
        response.message = "Episode stop accepted; motion lock asserted"
        return response

    def _call_trigger_service(self, service: str, timeout_s: float) -> str:
        result = subprocess.run(
            [
                "timeout",
                f"{timeout_s:g}",
                "ros2",
                "service",
                "call",
                service,
                "std_srvs/srv/Trigger",
                "{}",
            ],
            cwd=self.repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s + 5.0,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0 or re.search(
            r"success[=:]\s*[Tt]rue", output
        ) is None:
            raise ContractError(f"{service} rejected the request: {output[-300:]}")
        return output

    def _hardware_preflight(self) -> None:
        camera = subprocess.run(
            ["rs-enumerate-devices", "-s"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        camera_text = camera.stdout + camera.stderr
        if camera.returncode != 0:
            raise ContractError("RealSense D435i was not enumerated")
        usb = subprocess.run(
            ["lsusb", "-t"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if usb.returncode != 0:
            raise ContractError("USB topology could not be inspected")
        validate_realsense_link(camera_text, usb.stdout + usb.stderr)
        robot = subprocess.run(
            ["ping", "-c", "1", "-W", "2", self.robot_ip],
            check=False,
            capture_output=True,
            timeout=5,
        )
        if robot.returncode != 0:
            raise ContractError(f"Go2 is unreachable at {self.robot_ip}")

    def _run_command(self, argv: list[str], log: Any) -> int:
        log.write(("\n$ " + " ".join(argv) + "\n").encode("utf-8"))
        log.flush()
        process = subprocess.Popen(
            argv,
            cwd=self.repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self._mutex:
            self._process = process
        signalled_at: Optional[float] = None
        terminated_at: Optional[float] = None
        while process.poll() is None:
            if self._cancel.is_set():
                now = time.monotonic()
                try:
                    if signalled_at is None:
                        os.killpg(process.pid, signal.SIGINT)
                        signalled_at = now
                    elif terminated_at is None and now - signalled_at >= 5.0:
                        os.killpg(process.pid, signal.SIGTERM)
                        terminated_at = now
                    elif terminated_at is not None and now - terminated_at >= 5.0:
                        os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.2)
        with self._mutex:
            if self._process is process:
                self._process = None
        return int(process.returncode or 0)

    def _cleanup_stack(self, log: Any) -> None:
        self._lock_until = time.monotonic() + 30.0
        self._request_adapter_stop()
        try:
            subprocess.run(
                [
                    "bash",
                    str(self.repo_root / "deployment/go2/offboard/revisit_debug.sh"),
                    "park",
                ],
                cwd=self.repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.write(b"\nRevisit cleanup timed out; motion lock remains asserted.\n")
            log.flush()

    def _cleanup_stack_path(self, log_path: Path) -> None:
        with log_path.open("ab", buffering=0) as log:
            self._cleanup_stack(log)

    def _capture_session_running(self) -> bool:
        if self._episode is None:
            return False
        session = f"{self.capture_session_prefix}-{self._episode['episode_id']}"
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def _mirror_episode_evidence(self) -> None:
        if self._episode is None:
            return
        destination = self.capture_root / str(self._episode["episode_id"])
        if not destination.is_dir():
            return
        episode_dir = self._episode_dir()
        copies = {
            episode_dir / "revisit_goal.png": destination / "media/revisit_goal.png",
            episode_dir
            / "revisit_goal_depth.png": destination / "media/revisit_goal_depth.png",
            episode_dir
            / "goal_capture.json": destination / "receipts/goal_capture.json",
            episode_dir / "events.jsonl": destination / "logs/episode_manager.jsonl",
            episode_dir / "episode.json": destination / "receipts/episode.json",
        }
        for source, target in copies.items():
            if source.is_file():
                _atomic_write_bytes(target, source.read_bytes())

    def _finish_capture(self, outcome: str) -> None:
        if self._episode is None:
            return
        episode_id = str(self._episode["episode_id"])
        log_path = self._episode_dir() / "operator.log"
        if self._capture_session_running():
            with log_path.open("ab", buffering=0) as log:
                result = subprocess.run(
                    capture_stop_command(self.repo_root, episode_id),
                    cwd=self.repo_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                    check=False,
                )
            clean = result.returncode == 0
        else:
            clean = False
        self._episode = {
            **self._episode,
            "capture_active": False,
            "capture_stopped_utc": utc_now(),
            "capture_stop_clean": clean,
            "outcome": "unreviewed",
            "termination_reason": stop_reason_from_legacy_outcome(outcome),
        }
        self._write_episode()
        self._append_episode_event(
            "capture_stopped", termination_reason=stop_reason_from_legacy_outcome(outcome),
            outcome="unreviewed", capture_stop_clean=clean
        )
        self._mirror_episode_evidence()
        finalized = False
        manifest_sha256 = None
        if clean:
            allow_incomplete = outcome != "success"
            with log_path.open("ab", buffering=0) as log:
                result = subprocess.run(
                    capture_finalize_command(
                        self.repo_root,
                        episode_id,
                        outcome,
                        allow_incomplete=allow_incomplete,
                    ),
                    cwd=self.repo_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                    check=False,
                )
                if result.returncode != 0 and not allow_incomplete:
                    result = subprocess.run(
                        capture_finalize_command(
                            self.repo_root,
                            episode_id,
                            outcome,
                            allow_incomplete=True,
                        ),
                        cwd=self.repo_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                        check=False,
                    )
            finalized = result.returncode == 0
            manifest = self.capture_root / episode_id / "manifest.json"
            if finalized and manifest.is_file():
                manifest_sha256 = _sha256(manifest)
        self._episode = {
            **self._episode,
            "capture_finalized": finalized,
            "capture_manifest_sha256": manifest_sha256,
        }
        self._write_episode()

    def _abort_episode(self, detail: str) -> None:
        if self._episode is None:
            self._set_status("stopped", detail, active=False)
            return
        self._cancel.set()
        self._lock_until = time.monotonic() + 30.0
        self._request_adapter_stop()
        log_path = self._episode_dir() / "operator.log"
        self._transition("stopping", "Stopping Episode · motion remains locked")
        self._finish_capture("aborted")
        self._cleanup_stack_path(log_path)
        self._transition(
            "cancelled", detail, capture_active=False, outcome="aborted"
        )

    def _run_transaction(self, contract: StartContract) -> None:
        assert self._episode is not None
        log_dir = self._episode_dir()
        run_id = f"{contract.dataset_id}_cec_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        log_path = log_dir / "operator.log"
        try:
            with log_path.open("ab", buffering=0) as log:
                self._transition(
                    "revisit_preparing",
                    "Checking D435i SuperSpeed and Go2 link",
                    run_id=run_id,
                    revisit_started_utc=utc_now(),
                )
                self._hardware_preflight()
                if self._cancel.is_set():
                    raise ContractError("Cancelled before stack preparation")

                self._set_status(
                    "revisit_preparing",
                    "Restarting stack and replaying sealed Survey",
                    run_id=run_id,
                )
                code = self._run_command(prepare_command(self.repo_root, run_id), log)
                if code != 0 or self._cancel.is_set():
                    raise ContractError(f"Revisit preparation exited with code {code}")

                active = _load_object(self.state_path, "active Revisit state")
                if active.get("mode") != "formal_ready" or active.get("run_id") != run_id:
                    raise ContractError("Revisit preparation did not commit formal_ready state")
                formal_config = (
                    self.repo_root
                    / "runtime/go2/two_pass_revisit"
                    / contract.dataset_id
                    / run_id
                    / "formal_config.json"
                )
                if not formal_config.is_file():
                    raise ContractError(f"formal config is missing: {formal_config}")
                if self._cancel.is_set():
                    raise ContractError("Cancelled before motion preflight")

                self._set_status(
                    "revisit_preparing", "Stack ready · starting Revisit RGB-D capture"
                )
                code = self._run_command(
                    capture_resume_command(
                        self.repo_root, str(self._episode["episode_id"])
                    ),
                    log,
                )
                if code != 0 or self._cancel.is_set():
                    raise ContractError(f"Revisit RGB-D recorder resume exited with code {code}")
                self._lock_until = 0.0
                self._transition(
                    "revisiting",
                    "Stack ready; supervised Revisit preflight/navigation active",
                    run_id=run_id,
                    capture_active=True,
                    capture_phase="revisit",
                    revisit_capture_started_utc=utc_now(),
                )
                code = self._run_command(
                    navigation_command(self.repo_root, formal_config, self.timeout_s), log
                )
                if self._cancel.is_set():
                    self._lock_until = time.monotonic() + 180.0
                    self._request_adapter_stop()
                    self._finish_capture("aborted")
                    self._cleanup_stack(log)
                    self._transition(
                        "cancelled",
                        "Stopped by operator",
                        capture_active=False,
                        outcome="aborted",
                    )
                elif code == 0:
                    self._lock_until = time.monotonic() + 180.0
                    self._request_adapter_stop()
                    self._finish_capture("success")
                    self._cleanup_stack(log)
                    self._transition(
                        "complete",
                        "Automatic arrival signal stopped motion",
                        capture_active=False,
                        outcome="success",
                        completed_utc=utc_now(),
                    )
                else:
                    self._lock_until = time.monotonic() + 180.0
                    self._request_adapter_stop()
                    stop_cause = {2: "timeout", 3: "operator_intervention"}.get(code, "navigation_error")
                    self._finish_capture(stop_cause)
                    self._cleanup_stack(log)
                    self._transition(
                        "failed",
                        f"Revisit stopped with navigation code {code}",
                        capture_active=False,
                        outcome=stop_cause,
                    )
        except Exception as exc:
            self._lock_until = time.monotonic() + 180.0
            self._request_adapter_stop()
            self._finish_capture(
                "aborted" if self._cancel.is_set() else "system_failure"
            )
            self._cleanup_stack_path(log_path)
            state = "cancelled" if self._cancel.is_set() else "failed"
            self._transition(
                state,
                "Stopped by operator"
                if self._cancel.is_set()
                else f"Revisit failed: {exc}",
                capture_active=False,
                outcome="aborted" if self._cancel.is_set() else "system_failure",
            )
        finally:
            with self._mutex:
                self._process = None

    def close(self) -> None:
        self._cancel.set()
        self._lock_until = time.monotonic() + 2.0
        self._request_adapter_stop()
        with self._mutex:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        self.node.destroy_node()


def main() -> int:
    import argparse
    import rclpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--capture-session-prefix", default="navdp-capture")
    parser.add_argument(
        "--rgb-topic", default="/camera/camera/color/image_raw"
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera/camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--robot-ip", default="192.168.123.161")
    args = parser.parse_args()
    if not 0 < args.timeout_s <= 900:
        parser.error("--timeout-s must be in (0, 900]")

    rclpy.init()
    service = RevisitOperatorService(
        repo_root=args.repo_root,
        state_path=args.state,
        episodes_root=args.episodes_root
        or args.repo_root / "runtime/go2/episodes",
        capture_root=args.capture_root
        or args.repo_root / "runtime/go2/experiment_capture",
        capture_session_prefix=args.capture_session_prefix,
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        timeout_s=args.timeout_s,
        robot_ip=args.robot_ip,
    )
    try:
        rclpy.spin(service.node)
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
