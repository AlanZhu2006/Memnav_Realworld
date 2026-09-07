#!/usr/bin/env python3
"""Temporary RGB-only ImageGoal arrival gate for controlled Go2 trials.

The NavDP policy returns trajectories but no goal-reached action.  This node
therefore compares the live RGB view with the frozen Novel ImageGoal and emits
an arrival latch after the configured number of geometrically consistent
views.  The controlled commissioning profile currently defaults to one view.
It never subscribes to depth and never writes camera frames.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Optional

import cv2
import numpy as np

from image_goal_io import load_rgb_image, validate_rgb_image


@dataclass(frozen=True)
class RgbArrivalResult:
    matched: bool
    confirmed: bool
    reason: str
    consecutive_matches: int
    target_keypoints: int
    current_keypoints: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    target_coverage: float
    current_coverage: float
    center_offset_norm: float
    image_scale: float
    rotation_deg: float
    reprojection_error_px: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, float):
                payload[key] = round(value, 4) if math.isfinite(value) else None
        return payload


class RgbGoalArrivalVerifier:
    """SIFT/homography arrival verifier with a consecutive-frame latch."""

    def __init__(
        self,
        target_rgb: np.ndarray,
        *,
        image_width: int = 480,
        ratio_test: float = 0.72,
        min_good_matches: int = 45,
        min_inliers: int = 30,
        min_inlier_ratio: float = 0.45,
        min_coverage: float = 0.07,
        max_center_offset_norm: float = 0.22,
        min_image_scale: float = 0.60,
        max_image_scale: float = 1.45,
        max_rotation_deg: float = 16.0,
        max_reprojection_error_px: float = 4.0,
        required_consecutive_matches: int = 1,
    ) -> None:
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV SIFT support is required")
        self.image_width = max(160, int(image_width))
        self.ratio_test = float(ratio_test)
        self.min_good_matches = max(4, int(min_good_matches))
        self.min_inliers = max(4, int(min_inliers))
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_coverage = float(min_coverage)
        self.max_center_offset_norm = float(max_center_offset_norm)
        self.min_image_scale = float(min_image_scale)
        self.max_image_scale = float(max_image_scale)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.required_consecutive_matches = max(
            1, int(required_consecutive_matches)
        )
        if not 0.0 < self.ratio_test < 1.0:
            raise ValueError("ratio_test must be in (0, 1)")
        if not 0.0 < self.min_image_scale <= self.max_image_scale:
            raise ValueError("invalid image scale interval")

        self.detector = cv2.SIFT_create(
            nfeatures=1200, contrastThreshold=0.025
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.target_rgb = self._resize(target_rgb)
        target_gray = cv2.cvtColor(self.target_rgb, cv2.COLOR_RGB2GRAY)
        self.target_keypoints, self.target_descriptors = (
            self.detector.detectAndCompute(target_gray, None)
        )
        if (
            self.target_descriptors is None
            or len(self.target_keypoints) < self.min_good_matches
        ):
            raise ValueError(
                "ImageGoal has too little texture for RGB arrival: "
                f"{len(self.target_keypoints)} SIFT features"
            )
        self.consecutive_matches = 0
        self.last_debug_rgb = self.target_rgb.copy()

    def _resize(self, rgb: np.ndarray) -> np.ndarray:
        image = validate_rgb_image(rgb)
        scale = self.image_width / float(image.shape[1])
        height = max(1, int(round(image.shape[0] * scale)))
        return cv2.resize(
            image, (self.image_width, height), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def _coverage(points: np.ndarray, width: int, height: int) -> float:
        if len(points) < 3:
            return 0.0
        hull = cv2.convexHull(points.astype(np.float32))
        return float(cv2.contourArea(hull) / max(1.0, width * height))

    def reset(self) -> None:
        self.consecutive_matches = 0

    def _empty(
        self, reason: str, current_keypoints: int, good_matches: int = 0
    ) -> RgbArrivalResult:
        self.consecutive_matches = 0
        return RgbArrivalResult(
            False,
            False,
            reason,
            0,
            len(self.target_keypoints),
            current_keypoints,
            good_matches,
            0,
            0.0,
            0.0,
            0.0,
            math.inf,
            0.0,
            math.inf,
            math.inf,
        )

    def evaluate(self, current_rgb: np.ndarray) -> RgbArrivalResult:
        current = self._resize(current_rgb)
        if current.shape != self.target_rgb.shape:
            current = cv2.resize(
                current,
                (self.target_rgb.shape[1], self.target_rgb.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        current_keypoints, current_descriptors = self.detector.detectAndCompute(
            current_gray, None
        )
        self.last_debug_rgb = current.copy()
        if current_descriptors is None or len(current_keypoints) < 2:
            return self._empty(
                "insufficient_current_features", len(current_keypoints)
            )

        pairs = self.matcher.knnMatch(
            self.target_descriptors, current_descriptors, k=2
        )
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < self.ratio_test * second.distance
        ]
        if len(good) < 4:
            return self._empty(
                "insufficient_good_matches", len(current_keypoints), len(good)
            )

        target_points = np.float32(
            [self.target_keypoints[item.queryIdx].pt for item in good]
        )
        current_points = np.float32(
            [current_keypoints[item.trainIdx].pt for item in good]
        )
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        try:
            homography, mask = cv2.findHomography(
                target_points, current_points, method, 3.0
            )
        except cv2.error:
            homography, mask = None, None
        if homography is None or mask is None:
            return self._empty(
                "homography_failed", len(current_keypoints), len(good)
            )
        keep = mask.reshape(-1).astype(bool)
        inlier_target = target_points[keep]
        inlier_current = current_points[keep]
        inliers = int(keep.sum())
        if inliers < 4:
            return self._empty(
                "insufficient_inliers", len(current_keypoints), len(good)
            )
        inlier_ratio = inliers / max(1, len(good))
        height, width = self.target_rgb.shape[:2]
        target_coverage = self._coverage(inlier_target, width, height)
        current_coverage = self._coverage(inlier_current, width, height)

        projected = cv2.perspectiveTransform(
            inlier_target.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        reprojection_error = float(
            np.median(np.linalg.norm(projected - inlier_current, axis=1))
        )
        corners = np.float32(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ]
        ).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if not np.isfinite(warped).all():
            return self._empty(
                "invalid_homography", len(current_keypoints), len(good)
            )
        image_center = np.array(
            [(width - 1.0) / 2.0, (height - 1.0) / 2.0]
        )
        warped_center = warped.mean(axis=0)
        center_offset = float(
            np.linalg.norm(warped_center - image_center)
            / math.hypot(width, height)
        )
        warped_area = abs(float(cv2.contourArea(warped.astype(np.float32))))
        image_scale = math.sqrt(warped_area / max(1.0, width * height))
        top_edge = warped[1] - warped[0]
        rotation_deg = abs(
            math.degrees(math.atan2(float(top_edge[1]), float(top_edge[0])))
        )

        checks = (
            (len(good) >= self.min_good_matches, "insufficient_good_matches"),
            (inliers >= self.min_inliers, "insufficient_inliers"),
            (inlier_ratio >= self.min_inlier_ratio, "low_inlier_ratio"),
            (
                min(target_coverage, current_coverage) >= self.min_coverage,
                "insufficient_coverage",
            ),
            (center_offset <= self.max_center_offset_norm, "center_mismatch"),
            (
                self.min_image_scale <= image_scale <= self.max_image_scale,
                "scale_mismatch",
            ),
            (rotation_deg <= self.max_rotation_deg, "rotation_mismatch"),
            (
                reprojection_error <= self.max_reprojection_error_px,
                "reprojection_mismatch",
            ),
        )
        matched = True
        reason = "matched"
        for passed, failure_reason in checks:
            if not passed:
                matched = False
                reason = failure_reason
                break
        self.consecutive_matches = (
            self.consecutive_matches + 1 if matched else 0
        )
        confirmed = (
            self.consecutive_matches >= self.required_consecutive_matches
        )

        # The goal already has its own Foxglove panel.  Rendering it beside the
        # live frame doubled this diagnostic's width and made Match dominate
        # the dashboard.  Keep the useful evidence on one current-frame view:
        # inlier locations plus the projected goal boundary.
        debug_bgr = cv2.cvtColor(current, cv2.COLOR_RGB2BGR)
        color = (0, 180, 0) if matched else (0, 0, 220)
        outline = np.rint(warped).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(debug_bgr, [outline], True, color, 2, cv2.LINE_AA)
        for point in inlier_current[:80]:
            x, y = np.rint(point).astype(int)
            cv2.circle(debug_bgr, (x, y), 2, (0, 210, 255), -1, cv2.LINE_AA)
        cv2.rectangle(debug_bgr, (0, 0), (width - 1, 48), (24, 27, 32), -1)
        cv2.putText(
            debug_bgr,
            f"{'MATCH' if matched else 'NO MATCH'}  {reason}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_bgr,
            f"good {len(good)}  inliers {inliers}  scale {image_scale:.2f}",
            (10, 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (235, 238, 242),
            1,
            cv2.LINE_AA,
        )
        self.last_debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)
        return RgbArrivalResult(
            matched,
            confirmed,
            reason,
            self.consecutive_matches,
            len(self.target_keypoints),
            len(current_keypoints),
            len(good),
            inliers,
            inlier_ratio,
            target_coverage,
            current_coverage,
            center_offset,
            image_scale,
            rotation_deg,
            reprojection_error,
        )


class RgbGoalArrivalNode:
    def __init__(self, args) -> None:
        import rclpy
        from cv_bridge import CvBridge, CvBridgeError
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from sensor_msgs.msg import Image
        from std_msgs.msg import Bool, String
        from std_srvs.srv import SetBool

        class NodeImpl(Node):
            pass

        self.rclpy = rclpy
        self.bool_type = Bool
        self.string_type = String
        self.cv_bridge_error = CvBridgeError
        self.node = NodeImpl("navdp_rgb_goal_arrival")
        self.bridge = CvBridge()
        self.args = args
        self.allowed_phases = frozenset(
            phase.strip()
            for phase in args.allowed_phases.split(",")
            if phase.strip()
        )
        if not self.allowed_phases:
            raise ValueError("allowed_phases must contain at least one phase")
        self.verifier = RgbGoalArrivalVerifier(
            load_rgb_image(args.goal),
            image_width=args.image_width,
            ratio_test=args.ratio_test,
            min_good_matches=args.min_good_matches,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            min_coverage=args.min_coverage,
            max_center_offset_norm=args.max_center_offset_norm,
            min_image_scale=args.min_image_scale,
            max_image_scale=args.max_image_scale,
            max_rotation_deg=args.max_rotation_deg,
            max_reprojection_error_px=args.max_reprojection_error_px,
            required_consecutive_matches=args.required_consecutive,
        )
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        command_qos = QoSProfile(depth=10)
        self.arrival_pub = self.node.create_publisher(
            Bool, args.arrival_topic, command_qos
        )
        self.estop_pub = self.node.create_publisher(
            Bool, args.estop_topic, command_qos
        )
        self.detector_status_pub = self.node.create_publisher(
            String, args.detector_status_topic, state_qos
        )
        self.debug_pub = self.node.create_publisher(
            Image, args.debug_topic, state_qos
        )
        self.disable_client = self.node.create_client(
            SetBool, args.disable_service
        )
        self.node.create_subscription(
            Image, args.rgb_topic, self._on_rgb, qos_profile_sensor_data
        )
        self.node.create_subscription(
            String, args.navdp_status_topic, self._on_navdp_status, state_qos
        )
        self.node.create_timer(1.0 / args.rate_hz, self._tick)
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_stamp_ns = 0
        self.latest_rgb_received_s = 0.0
        self.latest_status_received_s = 0.0
        self.latest_sequence = 0
        self.processed_sequence = 0
        self.enabled = False
        self.estop = True
        self.phase = ""
        self.armed_since = 0.0
        self.arrival_latched = False
        self.adapter_acknowledged_latch = False
        self.last_result: Optional[RgbArrivalResult] = None
        self.last_error = ""
        self.disable_request_sent = False

    def _on_rgb(self, message) -> None:
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if stamp_ns <= self.latest_rgb_stamp_ns:
            return
        try:
            rgb = np.asarray(
                self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"),
                dtype=np.uint8,
            )
        except self.cv_bridge_error as exc:
            self.last_error = f"rgb_conversion_failed:{exc}"
            return
        self.latest_rgb = rgb.copy()
        self.latest_rgb_stamp_ns = stamp_ns
        self.latest_rgb_received_s = time.monotonic()
        self.latest_sequence += 1

    def _on_navdp_status(self, message) -> None:
        try:
            payload = json.loads(message.data)
            enabled = bool(payload.get("enabled"))
            estop = bool(payload.get("estop"))
            phase = str(payload.get("phase") or "")
            adapter_arrival_latched = bool(payload.get("arrival_latched"))
        except (json.JSONDecodeError, TypeError, ValueError):
            self.last_error = "invalid_navdp_status"
            return
        was_armed = self._armed()
        self.latest_status_received_s = time.monotonic()
        self.enabled = enabled
        self.estop = estop
        self.phase = phase
        if self.arrival_latched and adapter_arrival_latched:
            self.adapter_acknowledged_latch = True
        elif (
            self.arrival_latched
            and self.adapter_acknowledged_latch
            and not adapter_arrival_latched
        ):
            # reset_policy clears the adapter latch.  Clear this node only
            # after observing the previous acknowledgement, so a stale status
            # sample cannot undo a newly detected arrival.
            self.arrival_latched = False
            self.adapter_acknowledged_latch = False
            self.disable_request_sent = False
            self.last_result = None
            self.verifier.reset()
        if self._armed() and not was_armed:
            self.armed_since = time.monotonic()
            self.verifier.reset()
        elif not self._armed() and not self.arrival_latched:
            self.armed_since = 0.0
            self.verifier.reset()

    def _armed(self) -> bool:
        return (
            self.enabled
            and not self.estop
            and self.phase in self.allowed_phases
            and not self.arrival_latched
        )

    def _publish_status(self) -> None:
        payload = {
            "schema": "navdp_rgb_arrival_v1",
            "armed": self._armed(),
            "arrival_latched": self.arrival_latched,
            "phase": self.phase,
            "allowed_phases": sorted(self.allowed_phases),
            "latest_rgb_ready": self.latest_rgb is not None,
            "error": self.last_error,
            "result": None if self.last_result is None else self.last_result.to_dict(),
        }
        message = self.string_type()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.detector_status_pub.publish(message)

    def _latch_arrival(self) -> None:
        self.arrival_latched = True
        command = self.bool_type()
        command.data = True
        for _ in range(3):
            self.arrival_pub.publish(command)
            self.estop_pub.publish(command)
        if self.disable_client.service_is_ready():
            from std_srvs.srv import SetBool

            request = SetBool.Request()
            request.data = False
            self.disable_client.call_async(request)
            self.disable_request_sent = True
        self.node.get_logger().warning(
            "RGB ImageGoal arrival confirmed: estop asserted and Novel memory frozen"
        )

    def _tick(self) -> None:
        if self.arrival_latched:
            self._publish_status()
            return
        if (
            not self._armed()
            or self.latest_rgb is None
            or self.latest_sequence == self.processed_sequence
            or time.monotonic() - self.armed_since < self.args.arm_grace_s
        ):
            self._publish_status()
            return
        self.processed_sequence = self.latest_sequence
        received_age = time.monotonic() - self.latest_rgb_received_s
        source_age = (self.node.get_clock().now().nanoseconds - self.latest_rgb_stamp_ns) / 1e9
        if (received_age > self.args.max_image_age_s or not 0 <= source_age <= self.args.max_image_age_s
                or time.monotonic() - self.latest_status_received_s > 2.0):
            self.verifier.reset()
            self.last_error = "stale_arrival_observation_or_control_state"
            self._publish_status()
            return
        try:
            result = self.verifier.evaluate(self.latest_rgb)
            self.last_result = result
            self.last_error = ""
            debug = self.bridge.cv2_to_imgmsg(
                self.verifier.last_debug_rgb, encoding="rgb8"
            )
            debug.header.stamp = self.node.get_clock().now().to_msg()
            self.debug_pub.publish(debug)
        except Exception as exc:
            self.last_error = f"arrival_evaluation_failed:{type(exc).__name__}:{exc}"
            self._publish_status()
            return
        if result.confirmed:
            # Matching time is part of evidence age, not a free extension of
            # the RGB freshness limit while the robot may still be moving.
            final_age = (self.node.get_clock().now().nanoseconds - self.latest_rgb_stamp_ns) / 1e9
            if not 0 <= final_age <= self.args.max_image_age_s:
                self.verifier.reset()
                self.last_error = "arrival_match_finished_too_late"
            else:
                self._latch_arrival()
        self._publish_status()

    def stop(self) -> None:
        self.node.destroy_node()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RGB-only NavDP arrival gate")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--navdp-status-topic", default="/navdp/status")
    parser.add_argument(
        "--allowed-phases",
        default="memory_recording",
        help="comma-separated adapter phases in which matching is armed",
    )
    parser.add_argument("--arrival-topic", default="/navdp/arrival")
    parser.add_argument("--estop-topic", default="/navdp/estop")
    parser.add_argument(
        "--disable-service", default="/navdp_go2_adapter/set_enabled"
    )
    parser.add_argument(
        "--detector-status-topic", default="/navdp/rgb_arrival_status"
    )
    parser.add_argument("--debug-topic", default="/navdp/rgb_arrival_debug")
    parser.add_argument("--rate-hz", type=float, default=12.0)
    parser.add_argument("--max-image-age-s", type=float, default=0.60)
    parser.add_argument("--arm-grace-s", type=float, default=0.75)
    parser.add_argument("--image-width", type=int, default=480)
    parser.add_argument("--ratio-test", type=float, default=0.72)
    parser.add_argument("--min-good-matches", type=int, default=45)
    parser.add_argument("--min-inliers", type=int, default=30)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.45)
    parser.add_argument("--min-coverage", type=float, default=0.07)
    parser.add_argument("--max-center-offset-norm", type=float, default=0.22)
    parser.add_argument("--min-image-scale", type=float, default=0.60)
    parser.add_argument("--max-image-scale", type=float, default=1.45)
    parser.add_argument("--max-rotation-deg", type=float, default=16.0)
    parser.add_argument("--max-reprojection-error-px", type=float, default=4.0)
    parser.add_argument("--required-consecutive", type=int, default=1)
    return parser


def main() -> None:
    import rclpy

    args, ros_args = build_parser().parse_known_args()
    if (not all(math.isfinite(v) and v > 0 for v in (args.rate_hz, args.max_image_age_s))
            or not math.isfinite(args.arm_grace_s) or args.arm_grace_s < 0.0):
        raise ValueError("rate/image age must be positive and arm grace non-negative")
    rclpy.init(args=ros_args)
    detector = RgbGoalArrivalNode(args)
    try:
        rclpy.spin(detector.node)
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
