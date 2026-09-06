#!/usr/bin/env python3
"""One-command, fail-closed supervisor for a NavDP navigation run.

The stack launcher deliberately starts motion-locked.  This agent turns the
operator's explicit run command into one bounded transaction: lock, establish
the requested policy state, verify one fresh plan, arm, monitor, and stop on
every non-arrival exit. It prints relative timestamps so slow phases are
visible without requiring ad-hoc topic or tmux inspection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import signal
import sys
import time
from typing import Optional

import numpy as np

from image_goal_io import load_rgb_image
from rgb_goal_arrival import RgbGoalArrivalVerifier
from trajectory_control import ControllerConfig, trajectory_to_command


@dataclass(frozen=True)
class PathAssessment:
    poses: int
    path_length_m: float
    target_x_m: float
    target_y_m: float
    predicted_vx: float
    predicted_wz: float
    reverse: bool
    motion_source: str = "trajectory"


def assess_path(
    path_xy: np.ndarray,
    *,
    max_linear_mps: float,
    max_angular_rps: float,
) -> PathAssessment:
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError(f"selected trajectory has invalid shape {path.shape}")
    if not np.isfinite(path).all():
        raise ValueError("selected trajectory contains non-finite coordinates")
    command = trajectory_to_command(
        path,
        ControllerConfig(
            max_linear_mps=float(max_linear_mps),
            max_angular_rps=float(max_angular_rps),
        ),
    )
    if command.path_length < 0.10:
        raise ValueError(
            f"selected trajectory is too short ({command.path_length:.3f} m)"
        )
    if command.reverse or command.linear_x < -1e-6:
        raise ValueError("selected trajectory requests reverse motion")
    if abs(command.linear_x) > float(max_linear_mps) + 1e-6:
        raise ValueError("selected trajectory exceeds the linear speed limit")
    if abs(command.angular_z) > float(max_angular_rps) + 1e-6:
        raise ValueError("selected trajectory exceeds the angular speed limit")
    return PathAssessment(
        poses=len(path),
        path_length_m=float(command.path_length),
        target_x_m=float(command.target_x),
        target_y_m=float(command.target_y),
        predicted_vx=float(command.linear_x),
        predicted_wz=float(command.angular_z),
        reverse=bool(command.reverse),
    )


def assess_motion(
    path_xy: np.ndarray,
    status: dict,
    *,
    max_linear_mps: float,
    max_angular_rps: float,
) -> PathAssessment:
    """Assess effective motion, including body-heading feedback turns.

    The adapter validates the policy proof before publishing its override
    receipt. A turn replaces the trajectory command, so its displayed XY path
    can be zero even though the Go2 receives a pure-yaw command. Unknown
    overrides and holds must never fall back to arming a trajectory that the
    adapter will not execute.
    """
    override = status.get("terminal_motion_override")
    if not override or (
        isinstance(override, dict) and override.get("applied") is False
    ):
        execution = status.get("trajectory_execution") or {}
        age = execution.get("feedback_age_s")
        if (status.get("position_reference_available") is not True
                or not isinstance(age, (int, float)) or isinstance(age, bool)
                or not math.isfinite(age) or not 0 <= age <= 0.35):
            raise ValueError("fresh position aligned to the plan observation is required")
        return assess_path(
            path_xy, max_linear_mps=max_linear_mps,
            max_angular_rps=max_angular_rps,
        )
    if not isinstance(override, dict) or override.get("applied") is not True:
        raise ValueError("invalid motion override")
    heading = status.get("heading_turn") or {}
    age = heading.get("feedback_age_s")
    if (status.get("heading_reference_available") is not True
            or not isinstance(age, (int, float)) or isinstance(age, bool)
            or not math.isfinite(age) or not 0 <= age <= 0.35):
        raise ValueError("fresh body heading aligned to the goal observation is required")
    reason = override.get("reason")
    if reason not in {"local_goal_heading_turn", "rear_goal_heading_turn"}:
        raise ValueError(f"motion override does not authorize a turn: {reason}")
    if override.get("assert_estop") is not False:
        raise ValueError("motion override requests estop")
    command = override.get("command")
    if not isinstance(command, dict):
        raise ValueError("motion override command missing")
    values = [command.get(key) for key in ("linear_x", "angular_z")]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) for value in values
    ):
        raise ValueError("motion override command is not finite")
    vx, wz = values
    if (
        not math.isclose(vx, 0.0, abs_tol=1e-6)
        or vx > max_linear_mps
        or command.get("reverse") is not False
    ):
        raise ValueError("heading turn must have zero translation")
    if not 0.0 < abs(wz) <= max_angular_rps:
        raise ValueError("certified turn angular speed is zero or exceeds limit")
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError(f"selected trajectory has invalid shape {path.shape}")
    if not np.isfinite(path).all():
        raise ValueError("selected trajectory contains non-finite coordinates")
    return PathAssessment(
        poses=len(path), path_length_m=0.0, target_x_m=0.0, target_y_m=0.0,
        predicted_vx=vx, predicted_wz=wz, reverse=False, motion_source=reason,
    )


def locked_preflight_issue(
    status: dict,
    *,
    min_clearance_m: float,
    max_rgbd_age_s: float = 0.60,
    max_plan_age_s: float = 1.50,
) -> str:
    if bool(status.get("enabled")):
        return "adapter_is_enabled"
    if not bool(status.get("estop")):
        return "estop_is_not_asserted"
    if not bool(status.get("server_initialized")):
        return "policy_not_initialized"
    if not bool(status.get("image_goal_loaded")):
        return "image_goal_not_loaded"
    if bool(status.get("arrival_latched")):
        return "arrival_latch_not_reset"
    if status.get("last_error"):
        return f"adapter_error:{status['last_error']}"
    rgbd_age = status.get("rgbd_age_s")
    if rgbd_age is None or float(rgbd_age) > float(max_rgbd_age_s):
        return "rgbd_not_fresh"
    plan_age = status.get("plan_age_s")
    if plan_age is None or float(plan_age) > float(max_plan_age_s):
        return "trajectory_not_fresh"
    clearance = status.get("clearance_m")
    if clearance is None:
        return "depth_clearance_unavailable"
    if float(clearance) < float(min_clearance_m):
        return f"clearance_below_{float(min_clearance_m):.2f}m"
    return ""


def preserved_revisit_issue(
    status: dict,
    *,
    expected_dataset_id: str,
    expected_dataset_sha256: str,
    expected_goal_sha256: str,
) -> str:
    """Prove Formal Revisit state without issuing a destructive policy reset."""
    if status.get("phase") != "revisit_query":
        return "revisit_phase_not_active"
    if status.get("active_goal_sha256") != expected_goal_sha256:
        return "revisit_goal_changed"
    receipt = status.get("begin_revisit_receipt")
    if not isinstance(receipt, dict):
        return "revisit_receipt_missing"
    if receipt.get("loaded_dataset_id") != expected_dataset_id:
        return "revisit_dataset_changed"
    if receipt.get("loaded_dataset_manifest_sha256") != expected_dataset_sha256:
        return "revisit_dataset_manifest_changed"
    selected_goal = receipt.get("selected_goal")
    if not isinstance(selected_goal, dict):
        return "revisit_goal_receipt_missing"
    if selected_goal.get("sha256") != expected_goal_sha256:
        return "revisit_goal_receipt_changed"
    return ""


def live_fault(
    status: dict,
    *,
    max_linear_mps: float,
    max_angular_rps: float,
    max_rgbd_age_s: float = 2.00,
    max_plan_age_s: float = 5.00,
) -> str:
    if status.get("last_error"):
        return f"adapter_error:{status['last_error']}"
    if status.get("stop_reason") in {"obstacle_stop", "depth_unavailable_stop", "inference_error"}:
        return str(status["stop_reason"])
    if (status.get("rgbd_recovery") or {}).get("pending") is True:
        # This is a zero-command pause inside the same run, not a failure.
        # Do not mistake the intentionally consumed old plan for stale inference.
        velocities = (status.get("cmd_vx"), status.get("cmd_wz"))
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) and abs(v) < 1e-3 for v in velocities):
            return "rgbd_pause_nonzero_command"
        return ""
    rgbd_age = status.get("rgbd_age_s")
    # New adapters own the 20 Hz camera-age pause transaction. Do not abort
    # on a status sampled just before that control tick commits its pause.
    # Legacy adapters without this contract still get the stale-data fault.
    if "rgbd_recovery" not in status and (
        rgbd_age is None or float(rgbd_age) > float(max_rgbd_age_s)
    ):
        return "rgbd_stale"
    heading = status.get("heading_turn") or {}
    if heading.get("active") is True:
        age = heading.get("feedback_age_s")
        if (not isinstance(age, (int, float)) or isinstance(age, bool)
                or not math.isfinite(age) or not 0 <= age <= 0.35):
            return "heading_feedback_stale"
    elif (status.get("trajectory_execution") or {}).get("active") is True:
        age = status["trajectory_execution"].get("feedback_age_s")
        if (not isinstance(age, (int, float)) or isinstance(age, bool)
                or not math.isfinite(age) or not 0 <= age <= 0.35):
            return "position_feedback_stale"
    else:
        plan_age = status.get("plan_age_s")
        post_execution_replan = any(
            receipt.get("phase") in {"complete", "stalled_replan"}
            and isinstance(receipt.get("completed_age_s"), (int, float))
            and 0 <= receipt["completed_age_s"] <= max_plan_age_s
            for receipt in (heading, status.get("trajectory_execution") or {})
        )
        if not post_execution_replan and (plan_age is None or float(plan_age) > float(max_plan_age_s)):
            return "trajectory_stale"
    if abs(float(status.get("cmd_vx") or 0.0)) > float(max_linear_mps) + 0.01:
        return "linear_command_limit_violation"
    if abs(float(status.get("cmd_wz") or 0.0)) > float(max_angular_rps) + 0.01:
        return "angular_command_limit_violation"
    return ""


class NavigationRunAgent:
    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from cv_bridge import CvBridge, CvBridgeError
        from nav_msgs.msg import Path
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from sensor_msgs.msg import Image
        from std_msgs.msg import Bool, String
        from std_srvs.srv import SetBool, Trigger

        class NodeImpl(Node):
            pass

        self.rclpy = rclpy
        self.Bool = Bool
        self.SetBool = SetBool
        self.Trigger = Trigger
        self.bridge_error = CvBridgeError
        self.args = args
        self.started = time.monotonic()
        self.node = NodeImpl("navdp_navigation_run_agent")
        self.bridge = CvBridge()
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        command_qos = QoSProfile(depth=10)
        self.status: Optional[dict] = None
        self.arrival_status: Optional[dict] = None
        self.rgb: Optional[np.ndarray] = None
        self.path_xy: Optional[np.ndarray] = None
        self.path_after_reset: Optional[np.ndarray] = None
        self.reset_started_at: Optional[float] = None
        self.status_received_at = 0.0
        self.node.create_subscription(
            String, "/navdp/status", self._on_status, state_qos
        )
        self.node.create_subscription(
            String,
            "/navdp/rgb_arrival_status",
            self._on_arrival_status,
            state_qos,
        )
        self.node.create_subscription(
            Image, args.rgb_topic, self._on_rgb, qos_profile_sensor_data
        )
        self.node.create_subscription(
            Path, "/navdp/trajectory", self._on_path, state_qos
        )
        self.estop_pub = self.node.create_publisher(
            Bool, "/navdp/estop", command_qos
        )
        self.stop_client = self.node.create_client(
            Trigger, "/navdp_go2_adapter/operator_stop"
        )
        self.reset_client = self.node.create_client(
            Trigger, "/navdp_go2_adapter/reset_policy"
        )
        self.enable_client = self.node.create_client(
            SetBool, "/navdp_go2_adapter/set_enabled"
        )

    def _log(self, phase: str, message: str) -> None:
        elapsed = time.monotonic() - self.started
        print(f"[+{elapsed:5.1f}s] {phase:<9} {message}", flush=True)

    def _on_status(self, message) -> None:
        try:
            self.status = json.loads(message.data)
            self.status_received_at = time.monotonic()
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    def _on_arrival_status(self, message) -> None:
        try:
            self.arrival_status = json.loads(message.data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    def _on_rgb(self, message) -> None:
        try:
            self.rgb = np.asarray(
                self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"),
                dtype=np.uint8,
            ).copy()
        except self.bridge_error:
            pass

    def _on_path(self, message) -> None:
        path = np.asarray(
            [
                [pose.pose.position.x, pose.pose.position.y]
                for pose in message.poses
            ],
            dtype=np.float64,
        )
        self.path_xy = path
        if self.reset_started_at is not None:
            self.path_after_reset = path

    def _spin_until(self, predicate, timeout_s: float) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.10)
            if predicate():
                return True
        return bool(predicate())

    def _call(self, client, request, label: str, timeout_s: float = 5.0):
        if not client.wait_for_service(timeout_sec=float(timeout_s)):
            raise RuntimeError(f"{label} service is unavailable")
        future = client.call_async(request)
        self.rclpy.spin_until_future_complete(
            self.node, future, timeout_sec=float(timeout_s)
        )
        if not future.done() or future.result() is None:
            raise RuntimeError(f"{label} service timed out")
        response = future.result()
        if hasattr(response, "success") and not response.success:
            raise RuntimeError(f"{label} rejected: {response.message}")
        return response

    def _operator_stop(self, reason: str) -> None:
        try:
            response = self._call(
                self.stop_client,
                self.Trigger.Request(),
                "operator_stop",
                timeout_s=3.0,
            )
            self._log("LOCKED", f"{reason}: {response.message}")
        except Exception as exc:  # node shutdown remains the final fallback
            self._log("LOCK-ERR", f"{reason}: {exc}")

    @staticmethod
    def _locked_and_zero(status: Optional[dict]) -> bool:
        if not status:
            return False
        return (
            not bool(status.get("enabled"))
            and bool(status.get("estop"))
            and abs(float(status.get("cmd_vx") or 0.0)) < 1e-3
            and abs(float(status.get("cmd_wz") or 0.0)) < 1e-3
        )

    def _wait_for_reset_ready(self) -> tuple[bool, str]:
        last_issue = "waiting_for_status"

        def ready() -> bool:
            nonlocal last_issue
            if self.status is None:
                last_issue = "waiting_for_status"
                return False
            if self.path_after_reset is None:
                last_issue = "waiting_for_post_reset_trajectory"
                return False
            # A cached status may describe an older turn than the fresh Path.
            # Require an adapter plan computed after this run's boundary.
            plan_time = self.status.get("plan_monotonic_s")
            if (
                self.reset_started_at is None
                or isinstance(plan_time, bool)
                or not isinstance(plan_time, (int, float))
                or not math.isfinite(plan_time)
                or not self.reset_started_at < plan_time <= self.status_received_at
                or time.monotonic() - self.status_received_at > 2.0
            ):
                last_issue = "waiting_for_post_reset_plan_status"
                return False
            if self.arrival_status is None:
                last_issue = "waiting_for_arrival_module"
                return False
            if self.args.preserve_policy_state:
                last_issue = preserved_revisit_issue(
                    self.status,
                    expected_dataset_id=self.args.expected_dataset_id,
                    expected_dataset_sha256=self.args.expected_dataset_sha256,
                    expected_goal_sha256=self.args.expected_goal_sha256,
                )
                if last_issue:
                    return False
            command_subscribers = self.node.get_subscriptions_info_by_topic(
                "/navdp/cmd_vel"
            )
            if not any(
                item.node_name == "navdp_go2_cmd_bridge"
                for item in command_subscribers
            ):
                last_issue = "waiting_for_go2_bridge"
                return False
            if bool(self.arrival_status.get("arrival_latched")):
                last_issue = "waiting_for_arrival_latch_reset"
                return False
            if str(self.status.get("phase") or "") not in self.args.arrival_phases:
                last_issue = "arrival_phase_not_allowed"
                return False
            last_issue = locked_preflight_issue(
                self.status,
                min_clearance_m=self.args.min_clearance_m,
            )
            return not last_issue

        return self._spin_until(ready, self.args.ready_timeout_s), last_issue

    def _arm(self) -> str:
        self._log("ARM", "releasing software estop")
        if not self._spin_until(
            lambda: self.estop_pub.get_subscription_count() > 0, 3.0
        ):
            raise RuntimeError("adapter did not subscribe to /navdp/estop")
        message = self.Bool()
        message.data = False
        for _ in range(3):
            self.estop_pub.publish(message)
            self.rclpy.spin_once(self.node, timeout_sec=0.08)
            if self.status and not bool(self.status.get("estop")):
                break
        if not self._spin_until(
            lambda: bool(self.status)
            and not bool(self.status.get("enabled"))
            and not bool(self.status.get("estop")),
            3.0,
        ):
            raise RuntimeError("software estop release was not acknowledged")
        response = self._call(
            self.enable_client,
            self.SetBool.Request(data=True),
            "set_enabled",
            timeout_s=4.0,
        )
        self._log("ENABLED", response.message)

        def enabled_or_arrived() -> bool:
            return bool(self.status) and (
                bool(self.status.get("enabled"))
                or bool(self.status.get("arrival_latched"))
            )

        if not self._spin_until(enabled_or_arrived, 4.0):
            raise RuntimeError("motion enable was not observed in /navdp/status")
        if bool(self.status.get("arrival_latched")):
            return "arrival_latched"
        return "enabled"

    def _monitor(self) -> tuple[int, str]:
        deadline = time.monotonic() + self.args.timeout_s
        next_report = 0.0
        observed_enabled = False
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.15)
            status = self.status
            if status is None:
                continue
            if bool(status.get("arrival_latched")):
                return 0, "arrival_latched"
            if bool(status.get("enabled")):
                observed_enabled = True
            fault = live_fault(
                status,
                max_linear_mps=self.args.max_linear_mps,
                max_angular_rps=self.args.max_angular_rps,
            )
            if fault:
                return 1, fault
            if observed_enabled and (
                not bool(status.get("enabled")) or bool(status.get("estop"))
            ):
                return 3, "stopped_externally"
            if time.monotonic() - self.status_received_at > 2.0:
                return 1, "status_stale"
            now = time.monotonic()
            if now >= next_report:
                recovery = status.get("rgbd_recovery") or {}
                if recovery.get("pending") is True:
                    self._log("WAIT-RGBD", "motion paused; waiting for fresh post-stop RGB-D and a new plan")
                    next_report = now + 1.0
                    continue
                heading = status.get("heading_turn") or {}
                if heading.get("active") is True:
                    self._log(
                        "TURNING",
                        "measured heading error={:.1f}deg feedback={:.3f}s wz={:.2f}".format(
                            math.degrees(float(heading.get("error_rad") or 0)),
                            float(heading.get("feedback_age_s") or 0),
                            float(status.get("cmd_wz") or 0),
                        ),
                    )
                    next_report = now + 1.0
                    continue
                execution = status.get("trajectory_execution") or {}
                if execution.get("active") is True:
                    self._log(
                        "TRACKING",
                        "remaining={:.2f}m endpoint={:.2f}m pose={:.3f}s vx={:.2f} wz={:.2f}".format(
                            float(execution.get("remaining_m") or 0),
                            float(execution.get("endpoint_distance_m") or 0),
                            float(execution.get("feedback_age_s") or 0),
                            float(status.get("cmd_vx") or 0), float(status.get("cmd_wz") or 0),
                        ),
                    )
                    next_report = now + 1.0
                    continue
                self._log(
                    "RUNNING",
                    "vx={:.2f} wz={:.2f} clearance={:.2f}m "
                    "rgbd={:.2f}s plan={:.2f}s".format(
                        float(status.get("cmd_vx") or 0.0),
                        float(status.get("cmd_wz") or 0.0),
                        float(status.get("clearance_m") or math.nan),
                        float(status.get("rgbd_age_s") or math.nan),
                        float(status.get("plan_age_s") or math.nan),
                    ),
                )
                next_report = now + 1.0
        return 2, f"navigation_timeout_{self.args.timeout_s:.0f}s"

    def run(self) -> int:
        self._log("CONNECT", "waiting for locked stack state, RGB, and arrival module")
        if not self._spin_until(
            lambda: self.status is not None
            and self.rgb is not None
            and self.arrival_status is not None,
            10.0,
        ):
            self._operator_stop("initial state unavailable")
            return 1

        self._operator_stop("run transaction opened")
        if not self._spin_until(lambda: self._locked_and_zero(self.status), 3.0):
            self._operator_stop("failed to confirm initial lock")
            return 1

        if self.args.preserve_policy_state:
            assert self.status is not None
            issue = preserved_revisit_issue(
                self.status,
                expected_dataset_id=self.args.expected_dataset_id,
                expected_dataset_sha256=self.args.expected_dataset_sha256,
                expected_goal_sha256=self.args.expected_goal_sha256,
            )
            if issue:
                self._log("BLOCKED", issue)
                self._operator_stop("prepared Revisit contract changed")
                return 1
            self._log(
                "PRESERVE",
                f"revisit_query dataset={self.args.expected_dataset_id}; reset skipped",
            )
        else:
            response = self._call(
                self.reset_client,
                self.Trigger.Request(),
                "reset_policy",
                timeout_s=5.0,
            )
            self._log("RESET", response.message)
        # Reject the transient-local trajectory delivered at subscription time.
        # Only a path published after this transaction boundary may arm motion.
        self.path_after_reset = None
        self.reset_started_at = time.monotonic()
        ready, issue = self._wait_for_reset_ready()
        if not ready:
            self._log("BLOCKED", f"preflight timed out: {issue}")
            self._operator_stop("preflight failed")
            return 1

        assert self.path_after_reset is not None
        path = assess_motion(
            self.path_after_reset,
            self.status,
            max_linear_mps=self.args.max_linear_mps,
            max_angular_rps=self.args.max_angular_rps,
        )
        self._log(
            "PLAN",
            "{}: {} poses, {:.2f}m; first command vx={:.2f}, wz={:.2f}; "
            "clearance={:.2f}m".format(
                path.motion_source,
                path.poses,
                path.path_length_m,
                path.predicted_vx,
                path.predicted_wz,
                float(self.status["clearance_m"]),
            ),
        )
        verifier = RgbGoalArrivalVerifier(
            load_rgb_image(self.args.arrival_goal),
            min_image_scale=self.args.min_image_scale,
            max_image_scale=self.args.max_image_scale,
            required_consecutive_matches=1,
        )
        goal_match = verifier.evaluate(self.rgb)
        self._log(
            "GOAL",
            "matched={} reason={} good={} inliers={} scale={:.2f}".format(
                goal_match.matched,
                goal_match.reason,
                goal_match.good_matches,
                goal_match.inliers,
                goal_match.image_scale,
            ),
        )
        if goal_match.matched:
            self._operator_stop("already at ImageGoal; motion not armed")
            self._log("COMPLETE", "already at ImageGoal")
            return 0

        final_issue = locked_preflight_issue(
            self.status,
            min_clearance_m=self.args.min_clearance_m,
        )
        if final_issue:
            self._log("BLOCKED", f"state changed before arm: {final_issue}")
            self._operator_stop("final preflight failed")
            return 1

        # Recheck the effective command as well as freshness before arming.
        assess_motion(
            self.path_after_reset, self.status,
            max_linear_mps=self.args.max_linear_mps,
            max_angular_rps=self.args.max_angular_rps,
        )

        arm_result = self._arm()
        if arm_result == "arrival_latched":
            code, event = 0, "arrival_latched_during_arm"
        else:
            code, event = self._monitor()

        if code == 0:
            if not self._spin_until(lambda: self._locked_and_zero(self.status), 3.0):
                self._operator_stop("arrival did not settle to a zero command")
                return 1
            result = (self.arrival_status or {}).get("result") or {}
            self._log(
                "ARRIVED",
                "matched={} good={} inliers={} scale={}".format(
                    result.get("matched"),
                    result.get("good_matches"),
                    result.get("inliers"),
                    result.get("image_scale"),
                ),
            )
            self._log("COMPLETE", event)
            return 0

        self._log("STOPPING", event)
        self._operator_stop(event)
        return code

    def close(self) -> None:
        self.node.destroy_node()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrival-goal", required=True)
    parser.add_argument("--rgb-topic", required=True)
    parser.add_argument("--arrival-phases", default="revisit_query")
    parser.add_argument("--min-image-scale", type=float, default=0.60)
    parser.add_argument("--max-image-scale", type=float, default=1.45)
    parser.add_argument("--max-linear-mps", type=float, required=True)
    parser.add_argument("--max-angular-rps", type=float, required=True)
    parser.add_argument("--min-clearance-m", type=float, default=0.80)
    parser.add_argument("--ready-timeout-s", type=float, default=25.0)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--preserve-policy-state", action="store_true")
    parser.add_argument("--expected-dataset-id", default="")
    parser.add_argument("--expected-dataset-sha256", default="")
    parser.add_argument("--expected-goal-sha256", default="")
    args = parser.parse_args(argv)
    args.arrival_phases = frozenset(
        item.strip() for item in args.arrival_phases.split(",") if item.strip()
    )
    if not args.arrival_phases:
        parser.error("--arrival-phases must not be empty")
    if args.ready_timeout_s <= 0 or args.timeout_s <= 0:
        parser.error("timeouts must be positive")
    if args.preserve_policy_state and not all(
        (
            args.expected_dataset_id,
            args.expected_dataset_sha256,
            args.expected_goal_sha256,
        )
    ):
        parser.error(
            "--preserve-policy-state requires expected dataset ID, dataset SHA, and goal SHA"
        )
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    import rclpy

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_on_sigterm(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    rclpy.init()
    agent: Optional[NavigationRunAgent] = None
    try:
        agent = NavigationRunAgent(args)
        return agent.run()
    except KeyboardInterrupt:
        if agent is not None:
            agent._operator_stop("interrupted by operator")
        return 130
    except Exception as exc:
        if agent is not None:
            agent._log("FAILED", f"{type(exc).__name__}: {exc}")
            agent._operator_stop("run agent failure")
        else:
            print(f"navigation run agent failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if agent is not None:
            agent.close()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    sys.exit(main())
