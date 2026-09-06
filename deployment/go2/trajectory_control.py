#!/usr/bin/env python3
"""Lightweight local-trajectory tracking and depth safety for NavDP on Go2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
    lookahead_m: float = 0.60
    max_linear_mps: float = 0.30
    max_angular_rps: float = 0.60
    # Match the validated TinyNav real-robot controller: do not chase small
    # alternating heading errors into the Go2 angular command floor.
    heading_deadband_rad: float = math.radians(8.0)
    rotate_in_place_angle_rad: float = 0.70
    rotate_gain: float = 1.50
    # Only taper inside the final local 0.30 m. The earlier 1.0 m value
    # suppressed nearly every short stop-plan-act command twice.
    slow_path_length_m: float = 0.30
    allow_reverse: bool = False
    reverse_lateral_angle_rad: float = 0.55


@dataclass(frozen=True)
class VelocityCommand:
    linear_x: float = 0.0
    angular_z: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    path_length: float = 0.0
    reverse: bool = False


@dataclass(frozen=True)
class DepthSafetyConfig:
    hard_stop_m: float = 0.35
    percentile: float = 10.0
    roi_left: float = 0.35
    roi_right: float = 0.65
    roi_top: float = 0.30
    roi_bottom: float = 0.70
    min_valid_fraction: float = 0.03
    max_valid_depth_m: float = 5.0
    fail_closed: bool = True


@dataclass(frozen=True)
class SafetyResult:
    command: VelocityCommand
    clearance_m: Optional[float]
    reason: str


def _clamp(value: float, limit: float) -> float:
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def _prepare_path(trajectory: np.ndarray) -> np.ndarray:
    path = np.asarray(trajectory, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] < 2:
        raise ValueError(f"trajectory must have shape (N, >=2), got {path.shape}")
    path = path[:, :2]
    path = path[np.isfinite(path).all(axis=1)]
    if path.shape[0] == 0:
        return np.zeros((1, 2), dtype=np.float64)
    if np.linalg.norm(path[0]) > 1e-6:
        path = np.concatenate([np.zeros((1, 2), dtype=np.float64), path], axis=0)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate([[True], segment_lengths > 1e-5])
    return path[keep]


def select_lookahead(trajectory: np.ndarray, lookahead_m: float) -> tuple[np.ndarray, float]:
    path = _prepare_path(trajectory)
    if path.shape[0] < 2:
        return path[-1].copy(), 0.0

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    path_length = float(cumulative[-1])
    target_distance = min(max(float(lookahead_m), 0.0), path_length)
    upper = int(np.searchsorted(cumulative, target_distance, side="left"))
    if upper <= 0:
        return path[0].copy(), path_length
    if upper >= path.shape[0]:
        return path[-1].copy(), path_length

    lower = upper - 1
    span = cumulative[upper] - cumulative[lower]
    ratio = 0.0 if span <= 1e-8 else (target_distance - cumulative[lower]) / span
    target = path[lower] + ratio * (path[upper] - path[lower])
    return target, path_length


def trajectory_to_command(
    trajectory: np.ndarray,
    config: ControllerConfig = ControllerConfig(),
) -> VelocityCommand:
    target, path_length = select_lookahead(trajectory, config.lookahead_m)
    target_x, target_y = float(target[0]), float(target[1])
    distance_sq = target_x * target_x + target_y * target_y
    if distance_sq < 1e-6 or path_length < 1e-4:
        return VelocityCommand(target_x=target_x, target_y=target_y, path_length=path_length)

    rear_heading = math.atan2(target_y, -target_x) if target_x < 0.0 else math.inf
    reverse = bool(
        config.allow_reverse
        and target_x < -0.05
        and abs(rear_heading) <= config.reverse_lateral_angle_rad
    )

    if reverse:
        speed_scale = min(1.0, path_length / max(config.slow_path_length_m, 1e-3))
        linear_x = -config.max_linear_mps * speed_scale
        angular_z = _clamp(2.0 * linear_x * target_y / distance_sq, config.max_angular_rps)
    else:
        heading = math.atan2(target_y, target_x)
        if abs(heading) >= config.rotate_in_place_angle_rad:
            linear_x = 0.0
            angular_z = _clamp(config.rotate_gain * heading, config.max_angular_rps)
        else:
            speed_scale = min(1.0, path_length / max(config.slow_path_length_m, 1e-3))
            heading_scale = max(0.0, math.cos(heading))
            linear_x = config.max_linear_mps * speed_scale * heading_scale
            if abs(heading) < max(0.0, config.heading_deadband_rad):
                angular_z = 0.0
            else:
                angular_z = _clamp(
                    2.0 * linear_x * target_y / distance_sq,
                    config.max_angular_rps,
                )

    return VelocityCommand(
        linear_x=float(linear_x),
        angular_z=float(angular_z),
        target_x=target_x,
        target_y=target_y,
        path_length=path_length,
        reverse=reverse,
    )


def front_clearance(
    depth_m: np.ndarray,
    config: DepthSafetyConfig = DepthSafetyConfig(),
) -> Optional[float]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.size == 0:
        return None

    height, width = depth.shape
    x0 = int(np.clip(config.roi_left, 0.0, 1.0) * width)
    x1 = int(np.clip(config.roi_right, 0.0, 1.0) * width)
    y0 = int(np.clip(config.roi_top, 0.0, 1.0) * height)
    y1 = int(np.clip(config.roi_bottom, 0.0, 1.0) * height)
    if x1 <= x0 or y1 <= y0:
        return None

    roi = depth[y0:y1, x0:x1]
    observed = np.isfinite(roi) & (roi > 0.05)
    if float(observed.mean()) < config.min_valid_fraction:
        return None

    # Depth beyond the local safety horizon still proves that the pixel does
    # not contain an obstacle inside that horizon.  Clamp those observations
    # instead of treating an open, distant view as unavailable depth.
    clearance_depth = np.minimum(roi[observed], config.max_valid_depth_m)
    return float(
        np.percentile(clearance_depth, np.clip(config.percentile, 0.0, 100.0))
    )


def apply_depth_safety(
    command: VelocityCommand,
    depth_m: np.ndarray,
    config: DepthSafetyConfig = DepthSafetyConfig(),
) -> SafetyResult:
    clearance = front_clearance(depth_m, config)
    has_motion = abs(command.linear_x) > 0.0 or abs(command.angular_z) > 0.0
    if not has_motion:
        return SafetyResult(command=command, clearance_m=clearance, reason="clear")

    if clearance is None:
        if not config.fail_closed:
            return SafetyResult(command=command, clearance_m=None, reason="depth_unavailable_open")
        stopped = VelocityCommand(
            target_x=command.target_x,
            target_y=command.target_y,
            path_length=command.path_length,
            reverse=command.reverse,
        )
        return SafetyResult(command=stopped, clearance_m=None, reason="depth_unavailable_stop")

    # The center ROI cannot certify the full swept footprint of a rotation,
    # but a known obstacle inside the hard-stop distance is sufficient reason
    # to prohibit both translation and rotation.  This is conservative and
    # prevents a legged platform from pivoting into an already-visible object.
    if clearance <= config.hard_stop_m:
        stopped = VelocityCommand(
            target_x=command.target_x,
            target_y=command.target_y,
            path_length=command.path_length,
            reverse=command.reverse,
        )
        return SafetyResult(command=stopped, clearance_m=clearance, reason="obstacle_stop")

    return SafetyResult(command=command, clearance_m=clearance, reason="clear")


def slew_limit(
    previous: VelocityCommand,
    target: VelocityCommand,
    dt: float,
    max_linear_accel: float,
    max_angular_accel: float,
) -> VelocityCommand:
    dt = max(float(dt), 1e-3)
    linear_delta = abs(float(max_linear_accel)) * dt
    angular_delta = abs(float(max_angular_accel)) * dt
    linear_x = previous.linear_x + float(
        np.clip(target.linear_x - previous.linear_x, -linear_delta, linear_delta)
    )
    angular_z = previous.angular_z + float(
        np.clip(target.angular_z - previous.angular_z, -angular_delta, angular_delta)
    )
    return VelocityCommand(
        linear_x=linear_x,
        angular_z=angular_z,
        target_x=target.target_x,
        target_y=target.target_y,
        path_length=target.path_length,
        reverse=target.reverse,
    )
