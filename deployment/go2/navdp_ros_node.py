#!/usr/bin/env python3
"""ROS 2 adapter from aligned RGB-D NavDP trajectories to safe Go2 Twist commands."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path as FilePath
import threading
import time
from typing import Optional

import message_filters
import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PoseStamped, Twist, Vector3Stamped
from nav_msgs.msg import Path as NavPath, Odometry
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import Marker, MarkerArray

from debug_visualization import ranked_candidates, score_rgb
from image_goal_io import load_rgb_image
from latency_motion_guard import (
    LatencyMotionGuard,
    LatencyMotionGuardConfig,
)
from navdp_client import NavDPClient
from navigation_diagnostics import plan_diagnostics
from terminal_motion_override import terminal_motion_override
from heading_turn import HeadingTurn
from trajectory_execution import PlanCycle, TrajectoryExecution
from trajectory_control import (
    ControllerConfig,
    DepthSafetyConfig,
    VelocityCommand,
    apply_depth_safety,
    front_clearance,
    slew_limit,
    trajectory_to_command,
)


SOURCE_OBSERVATION_SCHEMA = "memnav_rgbd_source_observation_v1"


class NavDPGo2Adapter(Node):
    """Run NavDP asynchronously and publish fail-closed velocity commands."""

    def __init__(self) -> None:
        super().__init__("navdp_go2_adapter")
        self._declare_parameters()
        self._load_parameters()

        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._inference_event = threading.Event()
        self._inference_busy = False

        self._rgb: Optional[np.ndarray] = None
        self._depth_m: Optional[np.ndarray] = None
        self._intrinsic: Optional[np.ndarray] = None
        self._rgbd_monotonic = 0.0
        self._rgbd_source_stamp_ns = 0
        self._rgbd_source_age_at_receive_s: Optional[float] = None
        self._rgb_depth_skew_s: Optional[float] = None
        self._image_goal: Optional[np.ndarray] = load_rgb_image(
            self.image_goal_path
        )
        self._revisit_image_goal: Optional[np.ndarray] = None
        if self.revisit_image_goal_path:
            self._revisit_image_goal = load_rgb_image(
                self.revisit_image_goal_path
            )
        self._startup_image_goal = (
            None if self._image_goal is None else self._image_goal.copy()
        )
        self._startup_pause_memory_recording = self.pause_memory_recording

        self._enabled = bool(self.get_parameter("enable_on_start").value)
        self._estop = self.estop_on_start
        self._server_initialized = False
        self._reset_requested = True
        self._trajectory: Optional[np.ndarray] = None
        self._candidate_trajectories = np.empty((0, 0, 2), dtype=np.float32)
        self._candidate_values = np.empty((0,), dtype=np.float32)
        self._target_command = VelocityCommand()
        self._last_command = VelocityCommand()
        self._latency_motion_guard = LatencyMotionGuard(
            self.latency_motion_guard_config
        )
        self._plan_cycle = PlanCycle(self.settle_before_sense_s)
        self._rgbd_pause_reason = ""
        self._rgbd_pause_started_s = None
        self._trajectory_execution = TrajectoryExecution(
            completion_tolerance_m=self.local_completion_tolerance_m,
            stagnation_timeout_s=self.stagnation_timeout_s,
            stagnation_min_progress_m=self.stagnation_min_progress_m,
            stagnation_min_linear_mps=self.stagnation_min_linear_mps,
        )
        self._latency_motion_receipt: dict = {}
        self._heading_turn = HeadingTurn()
        self._turn_image_ns = 0
        self._plan_monotonic = 0.0
        self._last_inference_s = 0.0
        self._last_error = ""
        self._stop_reason = "disabled" if not self._enabled else "waiting_for_plan"
        self._last_warn: dict[str, float] = {}
        # Protocol-v3 two-phase episode state (hub is authoritative; these
        # mirror its receipts for status reporting and local gating only).
        self._phase: Optional[str] = None
        self._frames_recorded = 0
        self._goal_candidates_captured = 0
        self._last_auto_candidate_after_frame = -1
        self._auto_candidate_guard_remaining = 0
        self._auto_candidate_capture_started_after_frame = 0
        self._active_goal_id: Optional[int] = None
        self._active_goal_sha256: Optional[str] = None
        self._last_phase_receipt: dict = {}
        self._last_plan_receipt: dict = {}
        self._terminal_motion_receipt: dict = {}
        self._rgbd_diagnostic: dict = {}
        self._last_receipt_event = ""
        self._survey_seal_receipt: dict = {}
        self._survey_last_action = ""
        self._survey_last_success: Optional[bool] = None
        self._survey_last_message = ""
        self._survey_recording_active: Optional[bool] = None
        self._arrival_latched = False
        self._client_lock = threading.Lock()

        self._client = NavDPClient(
            self.server_url,
            self.connect_timeout_s,
            self.request_timeout_s,
        )
        if self.attach_existing_hub_on_start:
            self._attach_existing_hub()

        command_qos = QoSProfile(depth=10)
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, command_qos)
        self._path_pub = self.create_publisher(NavPath, self.path_topic, state_qos)
        self._status_pub = self.create_publisher(String, self.status_topic, state_qos)
        self._receipt_pub = self.create_publisher(
            String, self.cec_receipt_topic, state_qos
        )
        self._debug_markers_pub = None
        self._image_goal_pub = None
        if self.debug_visualization:
            self._debug_markers_pub = self.create_publisher(
                MarkerArray, self.debug_markers_topic, state_qos
            )
        self._image_goal_pub = self.create_publisher(
            Image, self.image_goal_debug_topic, state_qos
        )

        # Keep RGB-D conversion from starving the 20 Hz pose/control path.
        # Each group remains internally serialized; the executor may run the
        # independent groups concurrently and shared state is guarded by
        # self._lock.
        self._rgbd_callback_group = MutuallyExclusiveCallbackGroup()
        self._pose_callback_group = MutuallyExclusiveCallbackGroup()
        self._control_callback_group = MutuallyExclusiveCallbackGroup()
        self._rgb_sub = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile_sensor_data,
            callback_group=self._rgbd_callback_group,
        )
        self._depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile_sensor_data,
            callback_group=self._rgbd_callback_group,
        )
        self._rgbd_sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=self.rgbd_sync_queue_size,
            slop=self.max_rgb_depth_skew_s,
        )
        self._rgbd_sync.registerCallback(self._on_rgbd)
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data,
            callback_group=self._rgbd_callback_group,
        )
        self.create_subscription(
            Bool, self.enable_topic, self._on_enable, command_qos,
            callback_group=self._control_callback_group,
        )
        self.create_subscription(
            Vector3Stamped, "/navdp/go2/body_heading", self._on_body_heading, 1,
            callback_group=self._pose_callback_group,
        )
        self.create_subscription(
            Odometry, "/navdp/go2/odometry", self._on_body_pose, 1,
            callback_group=self._pose_callback_group,
        )
        self.create_subscription(
            Bool, self.estop_topic, self._on_estop, command_qos,
            callback_group=self._control_callback_group,
        )
        self.create_subscription(
            Bool, self.arrival_topic, self._on_arrival, command_qos,
            callback_group=self._control_callback_group,
        )

        self.create_service(
            Trigger, "~/operator_stop", self._operator_stop_service,
            callback_group=self._control_callback_group,
        )
        self.create_service(
            SetBool, "~/set_enabled", self._set_enabled_service,
            callback_group=self._control_callback_group,
        )
        self.create_service(
            Trigger, "~/reset_policy", self._reset_policy_service,
            callback_group=self._control_callback_group,
        )
        if self.two_phase_episode:
            self.create_service(
                Trigger, "~/capture_goal_candidate",
                self._capture_goal_candidate_service,
                callback_group=self._control_callback_group,
            )
            self.create_service(
                SetBool,
                "~/set_auto_goal_candidate_capture",
                self._set_auto_goal_candidate_capture_service,
                callback_group=self._control_callback_group,
            )
            self.create_service(
                Trigger, "~/begin_revisit", self._begin_revisit_service,
                callback_group=self._control_callback_group,
            )
            if self.survey_dataset_id:
                self.create_service(
                    Trigger, "~/survey_start", self._survey_start_service,
                    callback_group=self._control_callback_group,
                )
                self.create_service(
                    Trigger, "~/survey_seal", self._survey_seal_service,
                    callback_group=self._control_callback_group,
                )

        self.create_timer(1.0 / self.planning_rate_hz, self._request_inference)
        self.create_timer(
            1.0 / self.control_rate_hz,
            self._control_tick,
            callback_group=self._control_callback_group,
        )
        self.create_timer(0.5, self._publish_status)

        self._worker = threading.Thread(
            target=self._inference_worker,
            name="navdp-inference",
            daemon=True,
        )
        self._worker.start()

        self.get_logger().info(
            "NavDP Go2 adapter ready: "
            f"backend={self.backend}, mode={self.mode}, server={self.server_url}, "
            f"cmd={self.cmd_vel_topic}, enabled={self._enabled}, odometry=disabled, "
            f"rgbd_sync=approximate(queue={self.rgbd_sync_queue_size}, "
            f"slop={self.max_rgb_depth_skew_s:.3f}s)"
        )
        self.get_logger().warning(
            f"imagegoal loaded {self.image_goal_path}; policy has no internal arrival signal, "
            "use an explicit arrival module or operator termination"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "backend": "navdp",
            "mode": "imagegoal",
            "server_url": "http://127.0.0.1:8888",
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "image_goal_path": "",
            "revisit_image_goal_path": "",
            "selected_goal_image_path": "",
            "selected_goal_depth_path": "",
            "image_goal_debug_topic": "/navdp/image_goal",
            "cmd_vel_topic": "/navdp/cmd_vel",
            "path_topic": "/navdp/trajectory",
            "status_topic": "/navdp/status",
            "cec_receipt_topic": "/navdp/cec_receipt",
            "debug_markers_topic": "/navdp/debug/markers",
            "enable_topic": "/navdp/enabled",
            "estop_topic": "/navdp/estop",
            "arrival_topic": "/navdp/arrival",
            "base_frame": "base_link",
            "debug_visualization": True,
            "debug_max_candidates": 6,
            "enable_on_start": False,
            "estop_on_start": True,
            "plan_while_disabled": True,
            "two_phase_episode": False,
            "navigate_during_memory_recording": False,
            "pause_memory_recording": False,
            "auto_goal_candidate_interval_frames": 24,
            "auto_goal_candidate_max": 6,
            "auto_goal_candidate_post_guard_frames": 4,
            "auto_goal_candidate_capture_enabled": True,
            "auto_select_goal_candidate": True,
            "attach_existing_hub_on_start": False,
            "survey_dataset_id": "",
            "survey_seal_receipt_path": "",
            "planning_rate_hz": 2.0,
            "control_rate_hz": 20.0,
            "connect_timeout_s": 3.0,
            "request_timeout_s": 180.0,
            "sensor_timeout_s": 2.00,
            "trajectory_timeout_s": 5.00,
            "max_rgb_depth_skew_s": 0.10,
            "rgbd_sync_queue_size": 5,
            "depth_scale_m": 0.001,
            "lookahead_m": 0.60,
            "max_linear_mps": 0.30,
            "max_angular_rps": 0.60,
            "heading_deadband_rad": math.radians(8.0),
            "rotate_in_place_angle_rad": 0.70,
            "rotate_gain": 1.50,
            "slow_path_length_m": 0.30,
            "local_completion_tolerance_m": 0.15,
            "stagnation_timeout_s": 4.0,
            "stagnation_min_progress_m": 0.02,
            "stagnation_min_linear_mps": 0.08,
            "allow_reverse": False,
            "reverse_lateral_angle_rad": 0.55,
            "max_linear_accel_mps2": 0.50,
            "max_angular_accel_rps2": 1.20,
            "latency_motion_guard_enabled": True,
            "latency_max_plan_input_age_s": 1.50,
            "settle_before_sense_s": 0.15,
            "depth_hard_stop_m": 0.35,
            "depth_percentile": 10.0,
            "depth_roi_left": 0.35,
            "depth_roi_right": 0.65,
            "depth_roi_top": 0.30,
            "depth_roi_bottom": 0.70,
            "depth_min_valid_fraction": 0.03,
            "depth_max_valid_m": 5.0,
            "depth_fail_closed": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.backend = str(self.get_parameter("backend").value).strip().lower()
        self.mode = str(self.get_parameter("mode").value).strip().lower()
        if self.backend != "navdp" or self.mode != "imagegoal":
            raise ValueError(
                "the MemNav real-world adapter requires backend=navdp "
                "and mode=imagegoal"
            )
        self.two_phase_episode = bool(
            self.get_parameter("two_phase_episode").value
        )
        self.navigate_during_memory_recording = bool(
            self.get_parameter("navigate_during_memory_recording").value
        )
        self.pause_memory_recording = bool(
            self.get_parameter("pause_memory_recording").value
        )
        self.auto_goal_candidate_interval_frames = max(
            0,
            int(self.get_parameter("auto_goal_candidate_interval_frames").value),
        )
        self.auto_goal_candidate_max = max(
            0, int(self.get_parameter("auto_goal_candidate_max").value)
        )
        self.auto_goal_candidate_post_guard_frames = max(
            0,
            int(
                self.get_parameter(
                    "auto_goal_candidate_post_guard_frames"
                ).value
            ),
        )
        self.auto_goal_candidate_capture_enabled = bool(
            self.get_parameter(
                "auto_goal_candidate_capture_enabled"
            ).value
        )
        self.auto_select_goal_candidate = bool(
            self.get_parameter("auto_select_goal_candidate").value
        )
        self.attach_existing_hub_on_start = bool(
            self.get_parameter("attach_existing_hub_on_start").value
        )
        if self.navigate_during_memory_recording and not self.two_phase_episode:
            raise ValueError(
                "navigate_during_memory_recording requires two_phase_episode"
            )

        for name in (
            "server_url",
            "rgb_topic",
            "depth_topic",
            "camera_info_topic",
            "image_goal_path",
            "revisit_image_goal_path",
            "selected_goal_image_path",
            "selected_goal_depth_path",
            "image_goal_debug_topic",
            "cmd_vel_topic",
            "path_topic",
            "status_topic",
            "cec_receipt_topic",
            "debug_markers_topic",
            "enable_topic",
            "estop_topic",
            "arrival_topic",
            "base_frame",
            "survey_dataset_id",
            "survey_seal_receipt_path",
        ):
            setattr(self, name, str(self.get_parameter(name).value))

        for name in (
            "planning_rate_hz",
            "control_rate_hz",
            "connect_timeout_s",
            "request_timeout_s",
            "sensor_timeout_s",
            "trajectory_timeout_s",
            "max_rgb_depth_skew_s",
            "depth_scale_m",
            "max_linear_accel_mps2",
            "max_angular_accel_rps2",
            "settle_before_sense_s",
            "local_completion_tolerance_m",
            "stagnation_timeout_s",
            "stagnation_min_progress_m",
            "stagnation_min_linear_mps",
        ):
            setattr(self, name, float(self.get_parameter(name).value))
        self.plan_while_disabled = bool(self.get_parameter("plan_while_disabled").value)
        self.estop_on_start = bool(
            self.get_parameter("estop_on_start").value
        )
        self.debug_visualization = bool(
            self.get_parameter("debug_visualization").value
        )
        self.debug_max_candidates = max(
            0, int(self.get_parameter("debug_max_candidates").value)
        )
        self.rgbd_sync_queue_size = int(
            self.get_parameter("rgbd_sync_queue_size").value
        )

        if self.planning_rate_hz <= 0.0 or self.control_rate_hz <= 0.0:
            raise ValueError("planning_rate_hz and control_rate_hz must be positive")
        if self.sensor_timeout_s <= 0.0 or self.max_rgb_depth_skew_s <= 0.0:
            raise ValueError("sensor_timeout_s and max_rgb_depth_skew_s must be positive")
        if self.rgbd_sync_queue_size <= 0:
            raise ValueError("rgbd_sync_queue_size must be positive")
        if self.settle_before_sense_s < 0.0:
            raise ValueError("settle_before_sense_s must be nonnegative")
        if (self.local_completion_tolerance_m <= 0.0
                or self.stagnation_timeout_s <= 0.0
                or self.stagnation_min_progress_m <= 0.0
                or self.stagnation_min_linear_mps <= 0.0):
            raise ValueError("trajectory execution thresholds must be positive")

        self.controller_config = ControllerConfig(
            lookahead_m=float(self.get_parameter("lookahead_m").value),
            max_linear_mps=float(self.get_parameter("max_linear_mps").value),
            max_angular_rps=float(self.get_parameter("max_angular_rps").value),
            heading_deadband_rad=float(
                self.get_parameter("heading_deadband_rad").value
            ),
            rotate_in_place_angle_rad=float(
                self.get_parameter("rotate_in_place_angle_rad").value
            ),
            rotate_gain=float(self.get_parameter("rotate_gain").value),
            slow_path_length_m=float(self.get_parameter("slow_path_length_m").value),
            allow_reverse=bool(self.get_parameter("allow_reverse").value),
            reverse_lateral_angle_rad=float(
                self.get_parameter("reverse_lateral_angle_rad").value
            ),
        )
        self.depth_safety_config = DepthSafetyConfig(
            hard_stop_m=float(self.get_parameter("depth_hard_stop_m").value),
            percentile=float(self.get_parameter("depth_percentile").value),
            roi_left=float(self.get_parameter("depth_roi_left").value),
            roi_right=float(self.get_parameter("depth_roi_right").value),
            roi_top=float(self.get_parameter("depth_roi_top").value),
            roi_bottom=float(self.get_parameter("depth_roi_bottom").value),
            min_valid_fraction=float(
                self.get_parameter("depth_min_valid_fraction").value
            ),
            max_valid_depth_m=float(self.get_parameter("depth_max_valid_m").value),
            fail_closed=bool(self.get_parameter("depth_fail_closed").value),
        )
        self.latency_motion_guard_config = LatencyMotionGuardConfig(
            enabled=bool(
                self.get_parameter("latency_motion_guard_enabled").value
            ),
            max_plan_input_age_s=float(
                self.get_parameter("latency_max_plan_input_age_s").value
            ),
        )

    @staticmethod
    def _stamp_to_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _stamp_to_nanoseconds(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _source_observation(input_timing: dict) -> Optional[dict]:
        rgb_stamp_ns = input_timing.get("rgb_stamp_ns")
        depth_stamp_ns = input_timing.get("depth_stamp_ns")
        if not isinstance(rgb_stamp_ns, int) or not isinstance(depth_stamp_ns, int):
            return None
        return {
            "schema": SOURCE_OBSERVATION_SCHEMA,
            "rgb_stamp_ns": rgb_stamp_ns,
            "depth_stamp_ns": depth_stamp_ns,
            "pair_received_ros_ns": input_timing.get("pair_received_ros_ns"),
        }

    def _on_rgbd(self, rgb_msg: Image, depth_msg: Image) -> None:
        rgb_stamp_s = self._stamp_to_seconds(rgb_msg.header.stamp)
        depth_stamp_s = self._stamp_to_seconds(depth_msg.header.stamp)
        rgb_stamp_ns = self._stamp_to_nanoseconds(rgb_msg.header.stamp)
        depth_stamp_ns = self._stamp_to_nanoseconds(depth_msg.header.stamp)
        skew_s = abs(rgb_stamp_s - depth_stamp_s)
        if skew_s > self.max_rgb_depth_skew_s:
            self._warn_throttled(
                "rgbd_pair_skew",
                f"Rejected RGB-D pair with {skew_s:.3f}s timestamp skew",
            )
            return
        try:
            image = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            raw_depth = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except CvBridgeError as exc:
            self._warn_throttled("rgbd_convert", f"RGB-D conversion failed: {exc}")
            return

        rgb = np.asarray(image, dtype=np.uint8)
        depth = np.asarray(raw_depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        encoding = depth_msg.encoding.upper()
        if encoding in {"16UC1", "MONO16", "16SC1"} or np.issubdtype(depth.dtype, np.integer):
            depth_m = depth.astype(np.float32) * self.depth_scale_m
        elif encoding in {"32FC1", "64FC1"} or np.issubdtype(depth.dtype, np.floating):
            depth_m = depth.astype(np.float32)
        else:
            self._warn_throttled(
                "depth_encoding", f"Unsupported depth encoding {depth_msg.encoding!r}"
            )
            return
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0.0] = 0.0
        if rgb.shape[:2] != depth_m.shape[:2]:
            self._warn_throttled(
                "rgbd_shape",
                f"Rejected RGB-D pair with shapes {rgb.shape[:2]} and {depth_m.shape[:2]}",
            )
            return

        # Timing diagnostics are observation-only: an unavailable ROS clock
        # must never prevent an otherwise valid synchronized pair from being
        # admitted.
        try:
            pair_received_ros_ns = int(self.get_clock().now().nanoseconds)
            pair_received_ros_s = pair_received_ros_ns / 1e9
        except (AttributeError, RuntimeError):
            pair_received_ros_ns = None
            pair_received_ros_s = None
        source_stamp_ns = min(rgb_stamp_ns, depth_stamp_ns)
        source_age_at_receive_s = (
            None
            if pair_received_ros_ns is None or source_stamp_ns <= 0
            else max(0.0, (pair_received_ros_ns - source_stamp_ns) / 1e9)
        )

        with self._lock:
            self._rgb = rgb.copy()
            self._depth_m = depth_m.copy()
            self._rgbd_monotonic = time.monotonic()
            self._rgbd_source_stamp_ns = source_stamp_ns
            self._rgbd_source_age_at_receive_s = source_age_at_receive_s
            self._rgb_depth_skew_s = skew_s
            self._rgbd_diagnostic = {
                "rgb_stamp_s": rgb_stamp_s,
                "depth_stamp_s": depth_stamp_s,
                "rgb_stamp_ns": rgb_stamp_ns,
                "depth_stamp_ns": depth_stamp_ns,
                "pair_received_monotonic_s": self._rgbd_monotonic,
                "pair_received_ros_s": pair_received_ros_s,
                "pair_received_ros_ns": pair_received_ros_ns,
                "source_stamp_ns": source_stamp_ns,
                "source_age_at_receive_s": source_age_at_receive_s,
            }

    def _on_camera_info(self, msg: CameraInfo) -> None:
        intrinsic = np.asarray(msg.k, dtype=np.float32).reshape(3, 3)
        if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            self._warn_throttled("camera_info", "Rejected invalid camera intrinsics")
            return
        with self._lock:
            if self._intrinsic is None or not np.allclose(self._intrinsic, intrinsic):
                self._intrinsic = intrinsic
                self._reset_requested = True

    def _on_enable(self, msg: Bool) -> None:
        self._set_enabled(bool(msg.data), "enable topic")

    def _on_body_heading(self, msg: Vector3Stamped) -> None:
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self._lock:
            self._heading_turn.observe(stamp, float(msg.vector.z))

    def _on_body_pose(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        pose = msg.pose.pose
        q = pose.orientation
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        with self._lock:
            self._trajectory_execution.observe(stamp, pose.position.x, pose.position.y, yaw)

    def _reset_motion_guards_locked(self) -> None:
        # Keep emergency/episode locking usable during partial initialization.
        self._rgbd_pause_reason = ""
        self._rgbd_pause_started_s = None
        plan_cycle = getattr(self, "_plan_cycle", None)
        if plan_cycle is not None:
            plan_cycle.reset()
        certified_turn = getattr(self, "_heading_turn", None)
        if certified_turn is not None:
            certified_turn.reset()
        execution = getattr(self, "_trajectory_execution", None)
        if execution is not None:
            execution.reset()

    def _on_estop(self, msg: Bool) -> None:
        with self._lock:
            self._estop = bool(msg.data)
            if msg.data:
                self._reset_motion_guards_locked()
        if msg.data:
            self._publish_zero("estop")
            self.get_logger().warning("NavDP emergency stop asserted")
        else:
            self.get_logger().info("NavDP emergency stop released; enable state unchanged")

    def _on_arrival(self, msg: Bool) -> None:
        if not msg.data:
            return
        with self._lock:
            first_latch = not self._arrival_latched
            self._arrival_latched = True
            self._estop = True
            self._enabled = False
            self._reset_motion_guards_locked()
            if self.two_phase_episode and self._phase == "memory_recording":
                self.pause_memory_recording = True
            receipt = {
                "arrival_latched": True,
                "phase": self._phase,
                "frames_recorded": self._frames_recorded,
                "memory_recording_paused": self.pause_memory_recording,
            }
        self._publish_zero("rgb_imagegoal_arrival")
        if first_latch:
            self._publish_receipt("rgb_imagegoal_arrival", receipt)
            self.get_logger().warning(
                "RGB ImageGoal arrival latched: motion disabled, estop asserted, "
                "and Novel memory recording paused"
            )

    def _set_enabled_service(self, request, response):
        self._set_enabled(bool(request.data), "set_enabled service")
        response.success = True
        response.message = f"NavDP motion {'enabled' if request.data else 'disabled'}"
        return response

    def _operator_stop_service(self, _request, response):
        """Latch the adapter into its fail-closed state from an operator UI.

        This endpoint is deliberately one-way: it cannot clear estop, enable
        motion, reset policy state or change a goal.  Foxglove exposes only
        this service, never the general motion-control topics or services.
        """

        with self._lock:
            self._enabled = False
            self._estop = True
            self._target_command = VelocityCommand()
            self._reset_motion_guards_locked()
        self._publish_zero("operator_stop")
        response.success = True
        response.message = (
            "Operator STOP latched: motion disabled, estop asserted, "
            "zero command active"
        )
        self.get_logger().warning(response.message)
        return response

    def _lock_survey_motion(self, reason: str, *, pause: bool) -> None:
        """Revoke policy motion before any operator dataset transition."""

        with self._lock:
            self._enabled = False
            self._estop = True
            self._target_command = VelocityCommand()
            self.pause_memory_recording = bool(pause)
            self._reset_motion_guards_locked()
        self._publish_zero(reason)

    def _finish_survey_action(
        self,
        response,
        *,
        action: str,
        success: bool,
        summary: str,
        recording_active: bool,
        receipt: Optional[dict] = None,
    ):
        """Make every Survey button result visible in status and receipts."""

        payload = {
            "operator_summary": str(summary),
            **dict(receipt or {}),
            "dataset_id": self.survey_dataset_id,
            "action": str(action),
            "success": bool(success),
            "recording_active": bool(recording_active),
            "motion_enabled": False,
            "estop": True,
            "motion_authority_changed": False,
        }
        with self._lock:
            self._survey_last_action = str(action)
            self._survey_last_success = bool(success)
            self._survey_last_message = str(summary)
            self._survey_recording_active = bool(recording_active)
            if "memory_frames" in payload:
                self._frames_recorded = max(
                    self._frames_recorded, int(payload["memory_frames"])
                )
        event = action if success else f"{action}_rejected"
        self._publish_receipt(event, payload)
        # Keep the two severity calls on distinct Python call sites.  rclpy
        # keys logger call-site state by source location and rejects changing
        # severity when one indirect call site is reused for INFO and WARN.
        if success:
            self.get_logger().info(summary)
        else:
            self.get_logger().warning(summary)
        response.success = bool(success)
        response.message = json.dumps(payload, ensure_ascii=False)
        return response

    def _attach_existing_hub(self) -> None:
        """Fail-closed recovery for an already initialized two-phase hub.

        This is intentionally opt-in.  It exists for recovering the Jetson ROS
        adapter after a local process failure without resetting or overwriting
        a non-empty RTX Survey.  Recording always reattaches paused and motion
        authority remains governed by the normal disabled + estop defaults.
        """

        if not self.two_phase_episode:
            raise RuntimeError(
                "attach_existing_hub_on_start requires two_phase_episode"
            )
        health = self._client.health()
        if health.get("initialized") is not True:
            raise RuntimeError("cannot attach: hub is not initialized")
        phase = str(health.get("phase", ""))
        if phase not in {"memory_recording", "revisit_query"}:
            raise RuntimeError(f"cannot attach: unsupported hub phase {phase!r}")

        dataset = health.get("episodic_dataset")
        if not isinstance(dataset, dict):
            raise RuntimeError("cannot attach: hub omitted episodic dataset status")
        if phase == "memory_recording" and self.survey_dataset_id:
            if (
                dataset.get("recording") is not True
                or dataset.get("dataset_id") != self.survey_dataset_id
            ):
                raise RuntimeError(
                    "cannot attach: active hub dataset does not match the "
                    f"resolved Survey {self.survey_dataset_id!r}"
                )

        self._server_initialized = True
        self._reset_requested = False
        self._phase = phase
        self._frames_recorded = int(health.get("frames_recorded", 0))
        self._goal_candidates_captured = int(
            health.get("goal_candidates_captured", 0)
        )
        active_goal_id = health.get("active_goal_id")
        self._active_goal_id = (
            None if active_goal_id is None else int(active_goal_id)
        )
        active_goal_sha256 = health.get("active_goal_sha256")
        self._active_goal_sha256 = (
            None if active_goal_sha256 is None else str(active_goal_sha256)
        )
        if phase == "memory_recording":
            self.pause_memory_recording = True
            self._stop_reason = "memory_recording_paused"
        else:
            if self._revisit_image_goal is None or self._active_goal_sha256 is None:
                raise RuntimeError(
                    "cannot attach to revisit_query without the frozen Revisit "
                    "goal and an active hub goal identity"
                )
            self._image_goal = self._revisit_image_goal.copy()
            prepare = health.get("last_prepare_receipt")
            self._last_phase_receipt = (
                dict(prepare) if isinstance(prepare, dict) else {}
            )
            self._stop_reason = "disabled"
        self.get_logger().warning(
            "Attached to existing initialized hub without reset: "
            f"phase={phase}, frames={self._frames_recorded}; recording PAUSED; "
            "motion LOCKED"
        )

    def _survey_start_service(self, _request, response):
        """Start or resume an already prepared Survey without arming motion.

        Dataset identity is frozen in the resolved Survey config.  Foxglove
        therefore sends an empty Trigger request and cannot select an
        arbitrary dataset, reset either policy, or alter motor authority.
        """

        with self._lock:
            initialized = self._server_initialized
            phase = self._phase
            busy = self._inference_busy
            frames = self._frames_recorded
            active_recording = (
                initialized
                and phase == "memory_recording"
                and not self.pause_memory_recording
            )
        self._lock_survey_motion(
            "survey_start", pause=not active_recording
        )
        if active_recording:
            receipt = {
                "resumed": True,
                "memory_frames": frames,
            }
            return self._finish_survey_action(
                response,
                action="survey_start",
                success=True,
                summary=(
                    f"SURVEY ALREADY ACTIVE | {self.survey_dataset_id} | "
                    f"frames={frames} | motion LOCKED"
                ),
                recording_active=True,
                receipt=receipt,
            )

        reserved = False
        with self._lock:
            if initialized and phase == "memory_recording" and not busy:
                self._inference_busy = True
                reserved = True
        if not initialized:
            return self._finish_survey_action(
                response,
                action="survey_start",
                success=False,
                summary="START REJECTED | policy is not initialized; retry after PREPARE",
                recording_active=False,
                receipt={"memory_frames": frames},
            )
        if phase != "memory_recording":
            return self._finish_survey_action(
                response,
                action="survey_start",
                success=False,
                summary=f"START REJECTED | requires memory_recording, not {phase}",
                recording_active=False,
                receipt={"memory_frames": frames},
            )
        if busy:
            return self._finish_survey_action(
                response,
                action="survey_start",
                success=False,
                summary="START REJECTED | Survey transition is busy; retry",
                recording_active=False,
                receipt={"memory_frames": frames},
            )

        try:
            with self._client_lock:
                status = self._client.dataset_status()
            if (
                status.get("recording") is not True
                or status.get("dataset_id") != self.survey_dataset_id
            ):
                raise RuntimeError(
                    "prepared dataset identity/recording state does not match "
                    f"{self.survey_dataset_id!r}"
                )
            frames = int(status.get("memory_frames", frames))
            receipt = {
                "resumed": bool(frames > 0),
                "memory_frames": frames,
            }
            with self._lock:
                self.pause_memory_recording = False
                self._stop_reason = "memory_recording"
            return self._finish_survey_action(
                response,
                action="survey_start",
                success=True,
                summary=(
                    f"SURVEY {'RESUMED' if frames > 0 else 'STARTED'} | "
                    f"{self.survey_dataset_id} | frames={frames} | motion LOCKED"
                ),
                recording_active=True,
                receipt=receipt,
            )
        except Exception as exc:
            with self._lock:
                self.pause_memory_recording = True
            return self._finish_survey_action(
                response,
                action="survey_start",
                success=False,
                summary=(
                    f"START REJECTED | {type(exc).__name__}: {exc} | "
                    "recording PAUSED; motion LOCKED"
                ),
                recording_active=False,
                receipt={"memory_frames": frames},
            )
        finally:
            if reserved:
                with self._lock:
                    self._inference_busy = False

    def _survey_seal_service(self, _request, response):
        """Pause, lock and atomically seal the config-bound Survey dataset."""

        self._lock_survey_motion("survey_seal", pause=True)
        with self._lock:
            initialized = self._server_initialized
            phase = self._phase
            cached = dict(self._survey_seal_receipt)
            frames = self._frames_recorded
        if cached:
            return self._finish_survey_action(
                response,
                action="survey_seal",
                success=True,
                summary=(
                    f"SURVEY ALREADY SEALED | {self.survey_dataset_id} | "
                    f"frames={cached.get('memory_frames', frames)} | motion LOCKED"
                ),
                recording_active=False,
                receipt=cached,
            )
        if not initialized:
            return self._finish_survey_action(
                response,
                action="survey_seal",
                success=False,
                summary="SEAL REJECTED | Survey policy is not initialized",
                recording_active=False,
                receipt={"memory_frames": frames},
            )
        if phase != "memory_recording":
            return self._finish_survey_action(
                response,
                action="survey_seal",
                success=False,
                summary=f"SEAL REJECTED | requires memory_recording, not {phase}",
                recording_active=False,
                receipt={"memory_frames": frames},
            )

        # Pausing above prevents a new memory transaction from starting.  Let
        # the one already in flight finish, then reserve the same slot used by
        # the inference worker before calling the immutable seal endpoint.
        deadline = time.monotonic() + min(max(self.request_timeout_s, 1.0), 30.0)
        reserved = False
        while time.monotonic() < deadline:
            with self._lock:
                if not self._inference_busy:
                    self._inference_busy = True
                    reserved = True
                    break
            time.sleep(0.01)
        if not reserved:
            return self._finish_survey_action(
                response,
                action="survey_seal",
                success=False,
                summary=(
                    "SEAL REJECTED | timed out waiting for the current Survey "
                    "frame | recording PAUSED; click START SURVEY to resume"
                ),
                recording_active=False,
                receipt={"memory_frames": frames},
            )

        try:
            with self._client_lock:
                status = self._client.dataset_status()
                frames = int(status.get("memory_frames", frames))
                if status.get("recording") is True:
                    if status.get("dataset_id") != self.survey_dataset_id:
                        raise RuntimeError(
                            "active dataset identity does not match the resolved "
                            f"Survey config {self.survey_dataset_id!r}"
                        )
                    receipt = self._client.seal_dataset()
                else:
                    receipt = next(
                        (
                            dict(item)
                            for item in status.get("sealed_datasets", [])
                            if item.get("dataset_id") == self.survey_dataset_id
                        ),
                        None,
                    )
                    if receipt is None:
                        raise RuntimeError(
                            "configured Survey dataset is neither recording nor sealed"
                        )
            if receipt.get("dataset_id") != self.survey_dataset_id:
                raise RuntimeError("seal receipt dataset identity mismatch")
            receipt = {
                **dict(receipt),
                "recording_active": False,
                "motion_enabled": False,
                "estop": True,
                "motion_authority_changed": False,
            }
            if self.survey_seal_receipt_path:
                encoded = (
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8")
                self._atomic_write(
                    FilePath(self.survey_seal_receipt_path).expanduser(), encoded
                )
            with self._lock:
                self._survey_seal_receipt = dict(receipt)
                self._stop_reason = "survey_sealed"
            return self._finish_survey_action(
                response,
                action="survey_seal",
                success=True,
                summary=(
                    f"SURVEY SEALED | {self.survey_dataset_id} | "
                    f"frames={receipt.get('memory_frames', frames)} | motion LOCKED"
                ),
                recording_active=False,
                receipt=receipt,
            )
        except Exception as exc:
            return self._finish_survey_action(
                response,
                action="survey_seal",
                success=False,
                summary=(
                    f"SEAL REJECTED | {type(exc).__name__}: {exc} | recording "
                    "PAUSED; click START SURVEY to resume; motion LOCKED"
                ),
                recording_active=False,
                receipt={"memory_frames": frames},
            )
        finally:
            with self._lock:
                self._inference_busy = False

    def _reset_policy_service(self, _request, response):
        if self.attach_existing_hub_on_start:
            with self._lock:
                self._reset_requested = False
            self._publish_zero("attached_hub_reset_rejected")
            response.success = False
            response.message = (
                "policy reset is disabled while attached to an existing hub; "
                "restart the complete stack for a new episode"
            )
            return response
        with self._lock:
            self._reset_requested = True
            self._server_initialized = False
            self._trajectory = None
            self._candidate_trajectories = np.empty((0, 0, 2), dtype=np.float32)
            self._candidate_values = np.empty((0,), dtype=np.float32)
            self._plan_monotonic = 0.0
            self._last_error = ""
            self._active_goal_id = None
            self._active_goal_sha256 = None
            self._last_phase_receipt = {}
            self._last_plan_receipt = {}
            self._terminal_motion_receipt = {}
            self._latency_motion_guard.reset()
            self._latency_motion_receipt = {}
            self._reset_motion_guards_locked()
            self._last_receipt_event = ""
            self._survey_seal_receipt = {}
            self._survey_last_action = ""
            self._survey_last_success = None
            self._survey_last_message = ""
            self._survey_recording_active = None
            self._arrival_latched = False
            self.pause_memory_recording = self._startup_pause_memory_recording
            self._last_auto_candidate_after_frame = -1
            self._auto_candidate_guard_remaining = 0
            self._image_goal = (
                None
                if self._startup_image_goal is None
                else self._startup_image_goal.copy()
            )
        self._publish_zero("policy_reset")
        self._inference_event.set()
        response.success = True
        response.message = "NavDP reset queued"
        return response

    def _publish_receipt(self, event: str, receipt: dict) -> None:
        payload = {
            "schema": "cec_realworld_runtime_receipt_v1_20260824",
            "event": str(event),
            "monotonic_s": round(time.monotonic(), 6),
            "receipt": dict(receipt),
        }
        with self._lock:
            self._last_receipt_event = str(event)
        message = String()
        message.data = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        self._receipt_pub.publish(message)

    def _capture_goal_candidate_service(self, _request, response):
        with self._lock:
            rgb = None if self._rgb is None else self._rgb.copy()
            depth_m = None if self._depth_m is None else self._depth_m.copy()
            initialized = self._server_initialized
            phase = self._phase
        if not initialized or rgb is None:
            response.success = False
            response.message = "server not initialized or no RGB frame yet"
            return response
        if phase != "memory_recording":
            response.success = False
            response.message = (
                f"goal candidates require the memory_recording phase, not {phase}"
            )
            return response
        try:
            with self._client_lock:
                receipt = self._client.goal_candidate(
                    rgb, evaluation_depth_m=depth_m
                )
        except Exception as exc:
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
            return response
        with self._lock:
            self._goal_candidates_captured += 1
        self.get_logger().info(f"goal candidate captured: {receipt}")
        self._publish_receipt("goal_candidate", receipt)
        response.success = True
        response.message = json.dumps(receipt, ensure_ascii=False)
        return response

    def _set_auto_goal_candidate_capture_service(self, request, response):
        """Arm automatic candidate capture only at a declared return leg.

        The adapter has no trustworthy global pose with which to infer the
        physical turnaround.  Making this boundary explicit prevents an
        outbound view from silently becoming the formal Revisit target.
        """
        with self._lock:
            if not self._server_initialized or self._phase != "memory_recording":
                response.success = False
                response.message = (
                    "automatic candidate capture requires initialized "
                    f"memory_recording, not {self._phase}"
                )
                return response
            enabled = bool(request.data)
            self.auto_goal_candidate_capture_enabled = enabled
            self._last_auto_candidate_after_frame = -1
            self._auto_candidate_guard_remaining = 0
            self._auto_candidate_capture_started_after_frame = (
                self._frames_recorded
            )
            receipt = {
                "enabled": enabled,
                "started_after_frame": self._frames_recorded,
                "phase": self._phase,
                "motion_authority_changed": False,
            }
        self._publish_receipt("auto_goal_candidate_capture", receipt)
        response.success = True
        response.message = json.dumps(receipt, ensure_ascii=False)
        return response

    def _begin_revisit_service(self, _request, response):
        with self._lock:
            initialized = self._server_initialized
            phase = self._phase
            busy = self._inference_busy
            query_start_rgb = None if self._rgb is None else self._rgb.copy()
            # Reserve the only stateful inference slot atomically with the
            # idle check.  Without this reservation, the timer worker can
            # start one final Novel request between this check and acquisition
            # of ``_client_lock``; that stale request then reaches the hub
            # after its phase has already changed to Revisit.
            if initialized and phase == "memory_recording" and not busy:
                self._inference_busy = True
        if not initialized:
            response.success = False
            response.message = "server not initialized"
            return response
        if phase != "memory_recording":
            response.success = False
            response.message = f"begin_revisit requires memory_recording, not {phase}"
            return response
        if busy:
            response.success = False
            response.message = "inference busy; retry when the recording step settles"
            return response
        try:
            with self._client_lock:
                if self._revisit_image_goal is not None:
                    receipt, selected_goal = self._client.prepare_revisit_goal(
                        self._revisit_image_goal,
                        query_start_rgb=query_start_rgb,
                    )
                elif self.auto_select_goal_candidate:
                    receipt, selected_goal = self._client.prepare_revisit(
                        query_start_rgb=query_start_rgb
                    )
                else:
                    receipt = self._client.begin_revisit(
                        query_start_rgb=query_start_rgb
                    )
                    selected_goal = None
        except Exception as exc:
            with self._lock:
                self._inference_busy = False
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
            return response
        with self._lock:
            self._phase = "revisit_query"
            self._server_initialized = True
            self._reset_requested = False
            self.pause_memory_recording = False
            self._inference_busy = False
            self._frames_recorded = int(
                receipt.get("frames_recorded", self._frames_recorded)
            )
            candidate_scores = receipt.get("candidate_scores")
            if isinstance(candidate_scores, list):
                self._goal_candidates_captured = len(candidate_scores)
            if selected_goal is not None:
                selected = receipt.get("selected_goal", {})
                self._image_goal = selected_goal.copy()
                candidate_id = selected.get("candidate_id")
                self._active_goal_id = (
                    None if candidate_id is None else int(candidate_id)
                )
                self._active_goal_sha256 = str(selected["sha256"])
                self._persist_selected_goal_artifacts(receipt)
            self._last_phase_receipt = dict(receipt)
        self.get_logger().info(f"revisit phase started: {receipt}")
        self._publish_receipt("prepare_revisit", receipt)
        self._publish_image_goal()
        response.success = True
        response.message = json.dumps(receipt, ensure_ascii=False)
        return response

    @staticmethod
    def _atomic_write(path: FilePath, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    def _persist_selected_goal_artifacts(self, receipt: dict) -> None:
        """Persist the committed goal and optional offline depth evidence.

        Navigation continues to use the RTX-owned committed JPEG.  The depth
        file has no route back into NavDP or the RGB arrival gate; it is kept
        only for offline audit without manually copying artifacts between
        stages.
        """

        goal_jpeg = self._client.last_goal_jpeg
        depth_png = self._client.last_goal_evaluation_depth_png
        depth_scale_m = self._client.last_goal_evaluation_depth_scale_m
        selected_goal_image_path = getattr(
            self, "selected_goal_image_path", ""
        )
        selected_goal_depth_path = getattr(
            self, "selected_goal_depth_path", ""
        )
        if selected_goal_image_path and goal_jpeg:
            goal_path = FilePath(selected_goal_image_path).expanduser()
            self._atomic_write(goal_path, goal_jpeg)
            receipt["jetson_selected_goal_image_path"] = str(goal_path)
            receipt["jetson_selected_goal_image_sha256"] = hashlib.sha256(
                goal_jpeg
            ).hexdigest()
        if selected_goal_depth_path and depth_png:
            depth_path = FilePath(selected_goal_depth_path).expanduser()
            self._atomic_write(depth_path, depth_png)
            receipt["jetson_selected_goal_depth_path"] = str(depth_path)
            receipt["jetson_selected_goal_depth_sha256"] = hashlib.sha256(
                depth_png
            ).hexdigest()
            receipt["selected_goal_depth_policy_authority"] = False
            receipt["selected_goal_depth_scale_m"] = depth_scale_m

    def _set_enabled(self, enabled: bool, source: str) -> None:
        with self._lock:
            changed = enabled != self._enabled
            self._enabled = enabled
            if not enabled:
                self._reset_motion_guards_locked()
        if not enabled:
            self._publish_zero("disabled")
        elif self._estop:
            self.get_logger().warning("Motion enable accepted, but estop is still asserted")
        if changed:
            self.get_logger().info(
                f"NavDP motion {'enabled' if enabled else 'disabled'} by {source}"
            )

    def _plan_cycle_phase_locked(self, now_s: float) -> str:
        return self._plan_cycle.phase(
            now_s=now_s,
            latest_rgbd_source_ns=self._rgbd_source_stamp_ns,
        )

    def _request_inference(self) -> None:
        now = time.monotonic()
        with self._lock:
            if getattr(self, "_trajectory_execution", None) is not None and self._trajectory_execution.active:
                return
            if getattr(self, "_heading_turn", None) is not None and self._heading_turn.active:
                return
            should_plan = self.plan_while_disabled or self._enabled
            if self._inference_busy:
                should_plan = False
            elif self._enabled:
                phase = self._plan_cycle_phase_locked(now)
                should_plan = self._plan_cycle.planning_allowed(phase)
        if should_plan:
            self._inference_event.set()

    def _snapshot_inference_input(self):
        now = time.monotonic()
        with self._lock:
            if getattr(self, "_trajectory_execution", None) is not None and self._trajectory_execution.active:
                return None, "trajectory_tracking"
            if getattr(self, "_heading_turn", None) is not None and self._heading_turn.active:
                return None, "heading_turn_active"
            if (
                self.two_phase_episode
                and self._server_initialized
                and self._phase == "memory_recording"
                and self.pause_memory_recording
            ):
                return None, "memory_recording_paused"
            if (
                self._enabled
                and self._plan_monotonic > 0.0
            ):
                phase = self._plan_cycle_phase_locked(now)
                if not self._plan_cycle.planning_allowed(phase):
                    return None, f"plan_cycle_{phase}"
            if self._rgb is None or self._depth_m is None or self._intrinsic is None:
                return None, "waiting_for_rgbd_or_camera_info"
            if now - self._rgbd_monotonic > self.sensor_timeout_s:
                return None, "rgbd_stale"
            if self._rgbd_source_age_at_receive_s is not None:
                source_age = self._rgbd_source_age_at_receive_s + max(
                    0.0, now - self._rgbd_monotonic
                )
                if source_age > self.sensor_timeout_s:
                    return None, "rgbd_source_stale"
            if (
                self.two_phase_episode
                and self._phase in (None, "memory_recording")
                and not self.navigate_during_memory_recording
            ):
                goal_condition = None
            elif self._image_goal is None:
                return None, "waiting_for_image_goal"
            else:
                goal_condition = self._image_goal.copy()
            return (
                self._rgb.copy(),
                self._depth_m.copy(),
                self._intrinsic.copy(),
                goal_condition,
                self._reset_requested,
                dict(self._rgbd_diagnostic),
            ), "ready"

    def _inference_worker(self) -> None:
        while not self._stop_event.is_set():
            self._inference_event.wait(timeout=0.2)
            self._inference_event.clear()
            if self._stop_event.is_set():
                break

            snapshot, reason = self._snapshot_inference_input()
            if snapshot is None:
                with self._lock:
                    self._stop_reason = reason
                continue

            rgb, depth_m, intrinsic, goal_condition, reset_requested, input_timing = snapshot
            if reset_requested and self.attach_existing_hub_on_start:
                # An attached adapter must never reset the authoritative RTX
                # episode.  Clear stale/local reset requests and continue with
                # the already verified hub phase.
                with self._lock:
                    self._reset_requested = False
                reset_requested = False
            with self._lock:
                if self._inference_busy:
                    continue
                self._inference_busy = True
            started = time.monotonic()
            try:
                if reset_requested or not self._server_initialized:
                    with self._client_lock:
                        algorithm = self._client.reset(intrinsic)
                    with self._lock:
                        self._server_initialized = True
                        self._reset_requested = False
                        self._frames_recorded = 0
                        self._goal_candidates_captured = 0
                        self._last_auto_candidate_after_frame = -1
                        self._auto_candidate_guard_remaining = 0
                        self._auto_candidate_capture_started_after_frame = 0
                        self._active_goal_id = None
                        self._active_goal_sha256 = None
                        self._last_phase_receipt = {}
                        self._last_plan_receipt = {}
                        self._terminal_motion_receipt = {}
                        self._latency_motion_guard.reset()
                        self._latency_motion_receipt = {}
                        self._last_receipt_event = ""
                        self._survey_seal_receipt = {}
                        self._survey_last_action = ""
                        self._survey_last_success = None
                        self._survey_last_message = ""
                        self._survey_recording_active = None
                        self._image_goal = (
                            None
                            if self._startup_image_goal is None
                            else self._startup_image_goal.copy()
                        )
                        self._phase = (
                            "memory_recording"
                            if self.two_phase_episode
                            else "revisit_query"
                        )
                    self.get_logger().info(f"Policy server initialized: {algorithm}")

                with self._lock:
                    recording = (
                        self.two_phase_episode
                        and self._phase == "memory_recording"
                    )
                planned_in_recording = False
                if recording:
                    if self.pause_memory_recording:
                        finished = time.monotonic()
                        with self._lock:
                            self._last_inference_s = finished - started
                            self._last_error = ""
                            self._stop_reason = "memory_recording_paused"
                        continue
                    # The default remains record-only.  Formal autonomous
                    # Novel->Revisit trials opt into a single hub transaction
                    # that writes this RGB to CEC memory and executes frozen
                    # native NavDP toward the independently frozen Novel goal.
                    with self._lock:
                        guard_remaining = self._auto_candidate_guard_remaining
                        if guard_remaining > 0:
                            self._auto_candidate_guard_remaining -= 1
                    if guard_remaining > 0:
                        finished = time.monotonic()
                        with self._lock:
                            self._last_inference_s = finished - started
                            self._last_error = ""
                            self._stop_reason = "goal_candidate_guard"
                        # Do not append near-adjacent frames after a selected
                        # candidate.  This prevents the candidate from becoming
                        # a trivial near-self match in the recorded history.
                        continue
                    with self._lock:
                        candidate_elapsed = (
                            self._frames_recorded
                            - self._auto_candidate_capture_started_after_frame
                        )
                        auto_candidate = (
                            self.auto_goal_candidate_capture_enabled
                            and self.auto_goal_candidate_interval_frames > 0
                            and candidate_elapsed
                            >= self.auto_goal_candidate_interval_frames
                            and candidate_elapsed
                            % self.auto_goal_candidate_interval_frames == 0
                            and self._last_auto_candidate_after_frame
                            != self._frames_recorded
                            and self._goal_candidates_captured
                            < self.auto_goal_candidate_max
                        )
                    if auto_candidate:
                        try:
                            with self._client_lock:
                                candidate_receipt = self._client.goal_candidate(
                                    rgb,
                                    validate_support=True,
                                    evaluation_depth_m=depth_m,
                                )
                            if candidate_receipt.get("registered") is True:
                                finished = time.monotonic()
                                with self._lock:
                                    self._goal_candidates_captured += 1
                                    self._last_auto_candidate_after_frame = (
                                        self._frames_recorded
                                    )
                                    self._auto_candidate_guard_remaining = (
                                        self.auto_goal_candidate_post_guard_frames
                                    )
                                    self._last_inference_s = finished - started
                                    self._last_error = ""
                                    self._stop_reason = "memory_recording"
                                self._publish_receipt(
                                    "auto_goal_candidate", candidate_receipt
                                )
                                # The candidate RGB is intentionally not
                                # appended.  A short guard also excludes the
                                # immediately adjacent views.
                                continue
                            self._publish_receipt(
                                "auto_goal_candidate_rejected", candidate_receipt
                            )
                        except Exception as exc:
                            with self._lock:
                                # The request may have committed remotely even
                                # if its response was lost.  Never append this
                                # ambiguous RGB to memory; mark the scheduling
                                # point consumed so the next frame resumes
                                # normal recording.
                                self._last_auto_candidate_after_frame = (
                                    self._frames_recorded
                                )
                            self._warn_throttled(
                                "auto_goal_candidate",
                                "Automatic goal-candidate capture was "
                                f"ambiguous; dropping this frame: {exc}",
                                period_s=2.0,
                            )
                            continue
                    if self.navigate_during_memory_recording:
                        if goal_condition is None:
                            raise RuntimeError(
                                "Novel recording requires a loaded ImageGoal"
                            )
                        with self._client_lock:
                            trajectory, all_trajectories, all_values = (
                                self._client.novel_imagegoal_step(
                                    goal_condition,
                                    rgb,
                                    depth_m,
                                    source_observation=self._source_observation(
                                        input_timing
                                    ),
                                )
                            )
                            plan_receipt = dict(self._client.last_plan_receipt)
                        with self._lock:
                            self._frames_recorded = int(
                                plan_receipt.get(
                                    "frames_recorded", self._frames_recorded + 1
                                )
                            )
                        planned_in_recording = True
                    else:
                        with self._client_lock:
                            receipt = self._client.memory_step(
                                rgb,
                                source_observation=self._source_observation(
                                    input_timing
                                ),
                            )
                        finished = time.monotonic()
                        with self._lock:
                            self._frames_recorded = int(
                                receipt.get(
                                    "frames_recorded", self._frames_recorded + 1
                                )
                            )
                            self._last_inference_s = finished - started
                            self._last_error = ""
                            self._stop_reason = "memory_recording"
                        continue

                if not planned_in_recording:
                    with self._lock:
                        active_goal_sha256 = self._active_goal_sha256
                    with self._client_lock:
                        trajectory, all_trajectories, all_values = self._client.imagegoal_step(
                            goal_condition,
                            rgb,
                            depth_m,
                            installed_goal_sha256=active_goal_sha256,
                        )
                        plan_receipt = dict(self._client.last_plan_receipt)
                path = self._normalize_trajectory(trajectory)
                candidates, candidate_values = ranked_candidates(
                    all_trajectories, all_values, self.debug_max_candidates
                )
                target = trajectory_to_command(path, self.controller_config)
                try:
                    diagnostics = plan_diagnostics(
                        path, all_trajectories, all_values, plan_receipt, target, depth_m,
                    )
                    diagnostics["input_timing"] = input_timing
                    diagnostics["plan_completed_monotonic_s"] = time.monotonic()
                    plan_receipt["navigation_diagnostics"] = diagnostics
                except Exception as diagnostic_error:
                    # Diagnostics must not change policy execution or its authority.
                    plan_receipt["navigation_diagnostics"] = {
                        "error": f"{type(diagnostic_error).__name__}: {diagnostic_error}",
                        "observation_only": True,
                    }
                # Upstream NavDP encodes a low-critic recovery by rewriting the
                # selected path to x=0, y=+/-1.  A trajectory follower reads
                # that sentinel as a full 90-degree turn.  That behavior is
                # unsafe for a legged robot whose swept footprint is not
                # certified by the forward depth ROI.  Hold position and wait
                # for a post-stop RGB-D observation instead.
                if bool(plan_receipt.get("critic_fallback_applied")):
                    finished = time.monotonic()
                    plan_receipt["low_critic_motion_guard"] = {
                        "schema": "navdp_low_critic_motion_guard_v1",
                        "action": "hold_and_replan",
                        "critic_max": plan_receipt.get("critic_max"),
                        "critic_threshold": plan_receipt.get("critic_threshold"),
                        "upstream_fallback_trajectory_suppressed": True,
                    }
                    with self._lock:
                        # Discard an inference that crossed an arming or stop
                        # boundary, just as for an ordinary accepted plan.
                        if self._enabled and (
                            self._heading_turn.active
                            or self._trajectory_execution.active
                            or (
                                self._plan_cycle.sense_after_ns is not None
                                and int(input_timing.get("rgb_stamp_ns") or 0)
                                <= self._plan_cycle.sense_after_ns
                            )
                        ):
                            continue
                        self._trajectory = None
                        self._trajectory_execution.reset()
                        self._candidate_trajectories = candidates
                        self._candidate_values = candidate_values
                        self._target_command = VelocityCommand()
                        self._plan_monotonic = finished
                        self._rgbd_pause_reason = "low_critic_fallback"
                        self._rgbd_pause_started_s = finished
                        self._last_inference_s = finished - started
                        self._last_error = ""
                        self._stop_reason = "low_critic_hold"
                        self._last_plan_receipt = plan_receipt
                        self._terminal_motion_receipt = {}
                        self._latency_motion_receipt = {}
                        self._plan_cycle.install_plan(finished)
                    self._publish_zero("low_critic_hold")
                    self._note_action_stopped_after_zero(finished)
                    self._publish_receipt(
                        "imagegoal_plan_rejected_low_critic", plan_receipt
                    )
                    continue
                terminal = terminal_motion_override(
                    plan_receipt,
                    rotate_gain=self.controller_config.rotate_gain,
                    max_angular_rps=self.controller_config.max_angular_rps,
                )
                if terminal.applied:
                    assert terminal.command is not None
                    target = terminal.command
                finished = time.monotonic()
                pair_received = input_timing.get("pair_received_monotonic_s")
                source_age_at_receive = input_timing.get(
                    "source_age_at_receive_s"
                )
                plan_input_age_s = (
                    None
                    if pair_received is None
                    else finished
                    - float(pair_received)
                    + (
                        float(source_age_at_receive)
                        if source_age_at_receive is not None
                        else 0.0
                    )
                )
                guarded = self._latency_motion_guard.apply(
                    target,
                    plan_input_age_s=plan_input_age_s,
                )
                target = guarded.command
                with self._lock:
                    # Discard an inference that crossed arming or a stop
                    # boundary. It must not replace an executing frozen path.
                    if self._enabled and (
                        self._heading_turn.active or self._trajectory_execution.active
                        or (self._plan_cycle.sense_after_ns is not None
                            and int(input_timing.get("rgb_stamp_ns") or 0) <= self._plan_cycle.sense_after_ns)
                    ):
                        continue
                    if self._rgbd_pause_reason and guarded.reason not in {"pass", "disabled"}:
                        # Stay paused and try another fresh observation; do not
                        # consume the recovery gate with an inadmissible plan.
                        continue
                    self._trajectory = path
                    self._trajectory_execution.reset()
                    self._candidate_trajectories = candidates
                    self._candidate_values = candidate_values
                    self._target_command = target
                    self._plan_monotonic = finished
                    self._plan_cycle.install_plan(finished)
                    self._rgbd_pause_reason = ""
                    self._rgbd_pause_started_s = None
                    self._last_inference_s = finished - started
                    self._last_error = ""
                    self._stop_reason = "ready"
                    self._last_plan_receipt = plan_receipt
                    self._terminal_motion_receipt = terminal.audit_dict()
                    self._turn_image_ns = int(input_timing.get("rgb_stamp_ns") or 0)
                    self._latency_motion_receipt = guarded.audit_dict()
                    if terminal.assert_estop:
                        self._estop = True
                        self._enabled = False
                        self._stop_reason = "certified_terminal_stop"
                self._publish_receipt(
                    "novel_recording_plan"
                    if planned_in_recording else "imagegoal_plan",
                    plan_receipt,
                )
                if not self._stop_event.is_set() and rclpy.ok():
                    self._publish_path(path)
                    self._publish_debug_markers()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._stop_reason = "inference_error"
                if not self._stop_event.is_set() and rclpy.ok():
                    self._warn_throttled(
                        "inference", f"NavDP inference failed: {exc}", period_s=2.0
                    )
            finally:
                with self._lock:
                    self._inference_busy = False

    @staticmethod
    def _normalize_trajectory(trajectory: np.ndarray) -> np.ndarray:
        path = np.asarray(trajectory, dtype=np.float32)
        while path.ndim > 2 and path.shape[0] == 1:
            path = path[0]
        if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] < 2:
            raise ValueError(f"invalid trajectory shape {path.shape}")
        if not np.isfinite(path[:, :2]).all():
            raise ValueError("trajectory contains non-finite values")
        return path[:, :3].copy()

    def _motion_block_reason(self, now: float) -> Optional[str]:
        if not self._enabled:
            return "disabled"
        if self._estop:
            return "estop"
        if self._rgb is None or self._depth_m is None:
            return "waiting_for_rgbd_or_camera_info"
        if now - self._rgbd_monotonic > self.sensor_timeout_s:
            return "rgbd_stale"
        if self._rgbd_source_age_at_receive_s is not None:
            source_age = self._rgbd_source_age_at_receive_s + max(
                0.0, now - self._rgbd_monotonic
            )
            if source_age > self.sensor_timeout_s:
                return "rgbd_source_stale"
        if self._image_goal is None:
            return "waiting_for_image_goal"
        if self._rgbd_pause_reason:
            return "awaiting_rgbd_replan"
        # A latched body-heading target is executed by fresh IMU feedback.
        # No new visual plan is needed during this single continuous turn.
        if self._heading_turn.active:
            return None
        if self._trajectory_execution.active:
            return None
        if self._trajectory is None or self._plan_monotonic <= 0.0:
            return "waiting_for_plan"
        if now - self._plan_monotonic > self.trajectory_timeout_s:
            return "trajectory_stale"
        if self._last_error:
            return "inference_error"
        phase = self._plan_cycle_phase_locked(now)
        if not self._plan_cycle.motion_allowed(phase):
            return "awaiting_fresh_plan"
        return None

    def _ros_now_ns(self) -> Optional[int]:
        try:
            value = int(self.get_clock().now().nanoseconds)
        except (AttributeError, RuntimeError):
            return None
        return value if value > 0 else None

    def _note_action_stopped_after_zero(self, now_s: float) -> None:
        stopped_ros_ns = self._ros_now_ns()
        with self._lock:
            if self._trajectory_execution.active:
                self._trajectory_execution.active = False
                self._trajectory_execution.phase = "interrupted"
            self._plan_cycle.note_action_stopped(now_s, stopped_ros_ns)

    def _control_tick(self) -> None:
        now = time.monotonic()
        dt = 1.0 / self.control_rate_hz
        with self._lock:
            reason = self._motion_block_reason(now)
            depth = None if self._depth_m is None else self._depth_m.copy()
            target = self._target_command

        if reason in {"rgbd_stale", "rgbd_source_stale"}:
            with self._lock:
                self._publish_zero(reason)
                if not self._rgbd_pause_reason:
                    self._rgbd_pause_reason = reason
                    self._rgbd_pause_started_s = now
                    # Abandon both the path and any latched rear-goal turn.
                    # Recovery must use an observation captured after stopping.
                    self._heading_turn.reset()
                    self._trajectory_execution.reset()
                    self._target_command = VelocityCommand()
                    self._terminal_motion_receipt = {}
                    self._plan_cycle.install_plan(now)
                    self._note_action_stopped_after_zero(now)
            return

        if reason == "awaiting_rgbd_replan":
            self._publish_zero(reason)
            self._request_inference()
            return

        if reason == "awaiting_fresh_plan":
            self._request_inference()

        if reason is not None:
            if self._heading_turn.active:
                self._fail_heading_turn(reason)
                return
            self._publish_zero(reason)
            if reason not in {"disabled", "estop", "waiting_for_plan"}:
                # A sensor/policy interruption ends this action; never resume
                # its old command merely because the fault later clears.
                self._note_action_stopped_after_zero(now)
            return

        with self._lock:
            turning = self._heading_turn.active
            bearing = self._terminal_motion_receipt.get("bearing_rad")
            if not turning and bearing is not None:
                # Start only from a fresh, accepted plan and the heading at
                # its RGB exposure. Never re-anchor a turn at inference return.
                if self._latency_motion_receipt.get("reason") not in {"pass", "disabled"}:
                    self._publish_zero("heading_plan_rejected")
                    self._note_action_stopped_after_zero(now)
                    return
                turning = self._heading_turn.start(
                    bearing, self._turn_image_ns, self._ros_now_ns() or 0, now
                )
                if not turning:
                    self._fail_heading_turn(self._heading_turn.phase)
                    return
            if turning:
                target = self._heading_turn.step(
                    self._ros_now_ns() or 0, now,
                    self.controller_config.rotate_gain,
                    self.controller_config.max_angular_rps,
                )
                phase = self._heading_turn.phase
                if phase == "complete":
                    self._publish_zero("heading_turn_complete")
                    # Consume this plan, wait for post-stop RGB-D, and replan.
                    self._terminal_motion_receipt = {}
                    self._target_command = VelocityCommand()
                    self._plan_cycle.install_plan(now)
                    self._note_action_stopped_after_zero(now)
                    return
                if phase != "turning":
                    self._fail_heading_turn(phase)
                    return

            if not turning:
                if self._latency_motion_receipt.get("reason") not in {"pass", "disabled"}:
                    self._publish_zero("plan_input_rejected")
                    self._note_action_stopped_after_zero(now)
                    return
                if self._terminal_motion_receipt.get("applied") is True:
                    # A policy hold or stop must not fall through to tracking.
                    self._publish_zero("policy_hold")
                    self._note_action_stopped_after_zero(now)
                    return
                if not self._trajectory_execution.active:
                    started = self._trajectory_execution.start(
                        self._trajectory, self._turn_image_ns, self._ros_now_ns() or 0
                    )
                    if not started:
                        if self._trajectory_execution.phase == "empty_path":
                            self._publish_zero("empty_path")
                            self._note_action_stopped_after_zero(now)
                        else:
                            self._fail_trajectory(self._trajectory_execution.phase)
                        return
                    self._plan_cycle.start_execution()
                target = self._trajectory_execution.step(
                    self._ros_now_ns() or 0, now, self.controller_config
                )
                if self._trajectory_execution.phase in {"complete", "stalled_replan"}:
                    phase = self._trajectory_execution.phase
                    reason = (
                        "trajectory_complete"
                        if phase == "complete"
                        else "trajectory_stalled_replan"
                    )
                    self._publish_zero(reason)
                    self._target_command = VelocityCommand()
                    self._note_action_stopped_after_zero(now)
                    if phase == "stalled_replan":
                        self._warn_throttled(
                            "trajectory_stalled_replan",
                            "Go2 made insufficient path progress; stopped the local action "
                            "and requested a fresh post-stop plan",
                            period_s=1.0,
                        )
                    return
                if not self._trajectory_execution.active:
                    self._fail_trajectory(self._trajectory_execution.phase)
                    return

        safety = apply_depth_safety(target, depth, self.depth_safety_config)
        if safety.reason in {"obstacle_stop", "depth_unavailable_stop"}:
            if turning:
                self._fail_heading_turn(safety.reason)
                return
            self._publish_zero(safety.reason)
            self._note_action_stopped_after_zero(now)
            return

        command = slew_limit(
            self._last_command,
            safety.command,
            dt,
            self.max_linear_accel_mps2,
            self.max_angular_accel_rps2,
        )
        self._publish_command(command, safety.reason)

    def _fail_trajectory(self, reason: str) -> None:
        with self._lock:
            self._trajectory_execution.active = False
            self._trajectory_execution.phase = reason
            self._enabled = False
            self._estop = True
            self._last_error = reason
        self._publish_zero(reason)
        self.get_logger().error(f"Trajectory execution stopped: {reason}")

    def _fail_heading_turn(self, reason: str) -> None:
        with self._lock:
            self._heading_turn.active = False
            self._heading_turn.phase = reason
            self._enabled = False
            self._estop = True
            self._last_error = reason
        self._publish_zero(reason)
        self.get_logger().error(f"Heading turn stopped: {reason}")

    def _publish_command(self, command: VelocityCommand, reason: str) -> None:
        msg = Twist()
        msg.linear.x = float(command.linear_x)
        msg.angular.z = float(command.angular_z)
        self._cmd_pub.publish(msg)
        with self._lock:
            self._last_command = command
            self._stop_reason = reason

    def _publish_zero(self, reason: str) -> None:
        msg = Twist()
        self._cmd_pub.publish(msg)
        with self._lock:
            self._last_command = VelocityCommand()
            self._stop_reason = reason

    def _publish_path(self, trajectory: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        msg = NavPath()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        xy = trajectory[:, :2]
        for index, point in enumerate(trajectory):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.03
            previous = xy[max(0, index - 1)]
            following = xy[min(xy.shape[0] - 1, index + 1)]
            delta = following - previous
            yaw = math.atan2(float(delta[1]), float(delta[0]))
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            msg.poses.append(pose)
        self._path_pub.publish(msg)

    def _publish_image_goal(self) -> None:
        if self._image_goal_pub is None or self._image_goal is None:
            return
        message = self._bridge.cv2_to_imgmsg(self._image_goal, encoding="rgb8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame
        self._image_goal_pub.publish(message)

    @staticmethod
    def _line_marker(
        header,
        namespace: str,
        marker_id: int,
        trajectory: np.ndarray,
        width: float,
        color: tuple[float, float, float, float],
        height: float,
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.points.append(Point(x=0.0, y=0.0, z=height))
        for waypoint in trajectory:
            marker.points.append(
                Point(x=float(waypoint[0]), y=float(waypoint[1]), z=height)
            )
        return marker

    @staticmethod
    def _text_marker(
        header,
        namespace: str,
        marker_id: int,
        text: str,
        x: float,
        y: float,
        z: float,
        height: float,
        color: tuple[float, float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.z = height
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.text = text
        return marker

    def _publish_debug_markers(self) -> None:
        if self._debug_markers_pub is None:
            return
        with self._lock:
            trajectory = None if self._trajectory is None else self._trajectory.copy()
            candidates = self._candidate_trajectories.copy()
            values = self._candidate_values.copy()
            target = self._target_command
            command = self._last_command
            enabled = self._enabled
            estop = self._estop
            stop_reason = self._stop_reason
            inference_s = self._last_inference_s
            depth = None if self._depth_m is None else self._depth_m
            phase = self._phase
            goal_candidates = self._goal_candidates_captured
            active_goal_id = self._active_goal_id
            cec_takeover = self._last_plan_receipt.get("cec_takeover")
            cec_reason = self._last_plan_receipt.get("cec_reason")

        stamp = self.get_clock().now().to_msg()
        header = PoseStamped().header
        header.stamp = stamp
        header.frame_id = self.base_frame
        markers = []

        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        markers.append(clear)

        footprint = Marker()
        footprint.header = header
        footprint.ns = "robot"
        footprint.id = 0
        footprint.type = Marker.CUBE
        footprint.action = Marker.ADD
        footprint.pose.position.x = 0.0
        footprint.pose.position.y = 0.0
        footprint.pose.position.z = 0.025
        footprint.pose.orientation.w = 1.0
        footprint.scale.x = 0.55
        footprint.scale.y = 0.32
        footprint.scale.z = 0.05
        footprint.color.r = 0.55
        footprint.color.g = 0.58
        footprint.color.b = 0.62
        footprint.color.a = 0.75
        markers.append(footprint)

        if candidates.shape[0] > 0:
            finite_values = values[np.isfinite(values)]
            minimum = float(finite_values.min()) if finite_values.size else 0.0
            maximum = float(finite_values.max()) if finite_values.size else 0.0
            for index, (candidate, value) in enumerate(zip(candidates, values)):
                red, green, blue = score_rgb(float(value), minimum, maximum)
                markers.append(
                    self._line_marker(
                        header,
                        "candidates",
                        index,
                        candidate,
                        0.018,
                        (red, green, blue, 0.55),
                        0.015,
                    )
                )
                endpoint = candidate[-1]
                markers.append(
                    self._text_marker(
                        header,
                        "candidate_scores",
                        index,
                        f"Q {float(value):.2f}",
                        float(endpoint[0]),
                        float(endpoint[1]),
                        0.10,
                        0.09,
                        (red, green, blue, 0.9),
                    )
                )

        if trajectory is not None:
            markers.append(
                self._line_marker(
                    header,
                    "selected",
                    0,
                    trajectory,
                    0.055,
                    (0.1, 1.0, 0.25, 1.0),
                    0.045,
                )
            )

        lookahead = Marker()
        lookahead.header = header
        lookahead.ns = "lookahead"
        lookahead.id = 0
        lookahead.type = Marker.SPHERE
        lookahead.action = Marker.ADD
        lookahead.pose.position.x = target.target_x
        lookahead.pose.position.y = target.target_y
        lookahead.pose.position.z = 0.09
        lookahead.pose.orientation.w = 1.0
        lookahead.scale.x = 0.13
        lookahead.scale.y = 0.13
        lookahead.scale.z = 0.13
        lookahead.color.r = 0.0
        lookahead.color.g = 0.9
        lookahead.color.b = 1.0
        lookahead.color.a = 1.0
        markers.append(lookahead)

        clearance = (
            None if depth is None else front_clearance(depth, self.depth_safety_config)
        )
        state = "ESTOP" if estop else ("ENABLED" if enabled else "DISABLED")
        state_color = (
            (1.0, 0.1, 0.1, 1.0)
            if estop or stop_reason == "inference_error"
            else ((0.2, 1.0, 0.2, 1.0) if enabled else (1.0, 0.8, 0.1, 1.0))
        )
        clearance_text = "n/a" if clearance is None else f"{clearance:.2f}m"
        markers.append(
            self._text_marker(
                header,
                "status",
                0,
                (
                    f"{state} | {stop_reason}\n"
                    f"phase {phase or 'n/a'} | goals {goal_candidates}"
                    f" | active {active_goal_id if active_goal_id is not None else '-'}\n"
                    f"CEC {cec_takeover if cec_takeover is not None else '-'}"
                    f" | {cec_reason or '-'}\n"
                    f"cmd {command.linear_x:+.2f} m/s  {command.angular_z:+.2f} rad/s\n"
                    f"depth {clearance_text} | infer {inference_s:.2f}s"
                ),
                -0.25,
                -0.65,
                0.38,
                0.12,
                state_color,
            )
        )

        message = MarkerArray()
        message.markers = markers
        self._debug_markers_pub.publish(message)

    @staticmethod
    def _age(now: float, stamp: float) -> Optional[float]:
        return None if stamp <= 0.0 else round(max(0.0, now - stamp), 3)

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self._lock:
            clearance = (
                None
                if self._depth_m is None
                else front_clearance(self._depth_m, self.depth_safety_config)
            )
            if self._survey_seal_receipt:
                survey_state = "SEALED"
            elif self._phase == "memory_recording":
                survey_state = (
                    "PAUSED" if self.pause_memory_recording else "ACTIVE"
                )
            else:
                survey_state = "INACTIVE"
            rgbd_source_age_s = (
                None
                if self._rgbd_source_age_at_receive_s is None
                else round(
                    self._rgbd_source_age_at_receive_s
                    + max(0.0, now - self._rgbd_monotonic),
                    3,
                )
            )
            command_execution = self._plan_cycle.audit_dict(
                now_s=now,
                latest_rgbd_source_ns=self._rgbd_source_stamp_ns,
            )
            payload = {
                "backend": self.backend,
                "mode": self.mode,
                "odometry": False,
                "enabled": self._enabled,
                "estop": self._estop,
                "server_initialized": self._server_initialized,
                "inference_busy": self._inference_busy,
                "rgb_age_s": self._age(now, self._rgbd_monotonic),
                "depth_age_s": self._age(now, self._rgbd_monotonic),
                "rgbd_age_s": self._age(now, self._rgbd_monotonic),
                "rgbd_source_age_s": rgbd_source_age_s,
                "rgbd_source_stamp_ns": self._rgbd_source_stamp_ns,
                "rgb_depth_skew_s": (
                    None
                    if self._rgb_depth_skew_s is None
                    else round(self._rgb_depth_skew_s, 4)
                ),
                "goal_age_s": None,
                "image_goal_loaded": self._image_goal is not None,
                "revisit_image_goal_loaded": (
                    self._revisit_image_goal is not None
                ),
                "plan_age_s": self._age(now, self._plan_monotonic),
                # Adapter and run supervisor share this Jetson monotonic clock.
                "plan_monotonic_s": self._plan_monotonic,
                "rgbd_diagnostic": self._rgbd_diagnostic,
                "target_command_before_safety": {
                    "vx": self._target_command.linear_x,
                    "wz": self._target_command.angular_z,
                },
                "last_inference_s": round(self._last_inference_s, 3),
                "candidate_count": int(self._candidate_trajectories.shape[0]),
                "clearance_m": None if clearance is None else round(clearance, 3),
                "depth_hard_stop_m": self.depth_safety_config.hard_stop_m,
                "stop_reason": self._stop_reason,
                "last_error": self._last_error,
                "phase": self._phase,
                "frames_recorded": self._frames_recorded,
                "navigate_during_memory_recording": (
                    self.navigate_during_memory_recording
                ),
                "pause_memory_recording": self.pause_memory_recording,
                "survey_dataset_id": self.survey_dataset_id,
                "survey_state": survey_state,
                "survey_last_action": self._survey_last_action,
                "survey_last_success": self._survey_last_success,
                "survey_last_message": self._survey_last_message,
                "survey_recording_active": self._survey_recording_active,
                "arrival_latched": self._arrival_latched,
                "goal_candidates_captured": self._goal_candidates_captured,
                "auto_goal_candidate_interval_frames": (
                    self.auto_goal_candidate_interval_frames
                ),
                "auto_goal_candidate_max": self.auto_goal_candidate_max,
                "auto_goal_candidate_post_guard_frames": (
                    self.auto_goal_candidate_post_guard_frames
                ),
                "auto_goal_candidate_capture_enabled": (
                    self.auto_goal_candidate_capture_enabled
                ),
                "auto_candidate_capture_started_after_frame": (
                    self._auto_candidate_capture_started_after_frame
                ),
                "auto_candidate_guard_remaining": (
                    self._auto_candidate_guard_remaining
                ),
                "auto_select_goal_candidate": self.auto_select_goal_candidate,
                "active_goal_id": self._active_goal_id,
                "active_goal_sha256": self._active_goal_sha256,
                "last_receipt_event": self._last_receipt_event,
                "begin_revisit_receipt": self._last_phase_receipt,
                "cec_takeover": self._last_plan_receipt.get("cec_takeover"),
                "cec_reason": self._last_plan_receipt.get("cec_reason"),
                "cec_controller": self._last_plan_receipt.get("cec_controller"),
                "cec_selected_anchor": self._last_plan_receipt.get(
                    "cec_selected_anchor"
                ),
                "terminal_handoff_disposition": self._last_plan_receipt.get(
                    "terminal_handoff_disposition"
                ),
                "terminal_local_latched": self._last_plan_receipt.get(
                    "terminal_local_latched"
                ),
                "terminal_predicted_distance_m": self._last_plan_receipt.get(
                    "terminal_predicted_distance_m"
                ),
                "terminal_predicted_bearing_deg": self._last_plan_receipt.get(
                    "terminal_predicted_bearing_deg"
                ),
                "terminal_stop_streak": self._last_plan_receipt.get(
                    "terminal_stop_streak"
                ),
                "terminal_stop_authorized": self._last_plan_receipt.get(
                    "terminal_stop_authorized"
                ),
                "terminal_motion_override": self._terminal_motion_receipt,
                "latency_motion_guard": self._latency_motion_receipt,
                "command_execution_mode": "measured_trajectory",
                "command_execution_phase": command_execution["phase"],
                "command_execution": command_execution,
                "rgbd_recovery": {
                    "pending": bool(self._rgbd_pause_reason),
                    "reason": self._rgbd_pause_reason,
                    "pause_age_s": (None if self._rgbd_pause_started_s is None
                                    else now - self._rgbd_pause_started_s),
                    "timeout_s": self.sensor_timeout_s,
                },
                "trajectory_execution": self._trajectory_execution.audit(self._ros_now_ns() or 0, now),
                "position_reference_available": self._trajectory_execution.reference(self._turn_image_ns) is not None,
                "settle_before_sense_s": self.settle_before_sense_s,
                "heading_turn": self._heading_turn.audit(self._ros_now_ns() or 0, time.monotonic()),
                "heading_reference_available": self._heading_turn.reference(self._turn_image_ns) is not None,
                "monocular_depth_receipt": self._last_plan_receipt.get(
                    "monocular_depth_receipt"
                ),
                "controller_max_linear_mps": round(
                    self.controller_config.max_linear_mps, 3
                ),
                "controller_max_angular_rps": round(
                    self.controller_config.max_angular_rps, 3
                ),
                "controller_heading_deadband_deg": round(
                    math.degrees(self.controller_config.heading_deadband_rad), 3
                ),
                "cmd_vx": round(self._last_command.linear_x, 3),
                "cmd_wz": round(self._last_command.angular_z, 3),
            }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._status_pub.publish(msg)
        self._publish_debug_markers()
        self._publish_image_goal()

    def _warn_throttled(self, key: str, message: str, period_s: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._last_warn.get(key, 0.0) >= period_s:
            self._last_warn[key] = now
            self.get_logger().warning(message)

    def stop(self) -> None:
        self._stop_event.set()
        self._inference_event.set()
        if rclpy.ok():
            try:
                self._publish_zero("shutdown")
            except RuntimeError:
                pass
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavDPGo2Adapter()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        executor.remove_node(node)
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
