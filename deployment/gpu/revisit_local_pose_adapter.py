"""Role-free direct-bearing and optional local-approach adapter.

Long-range CEC can address a Revisit goal through causal history, while native
NavDP can approach a Novel goal directly.  Once the current and goal views are
geometrically covisible, the same two-view proof supplies a much more local
relative direction. It does not by itself certify metric scale or arrival.

This module therefore grants only the authority supported by the evidence:

* a certified direction may request one bounded atomic turn;
* a direction within NavDP's measured point-token support is projected onto
  the already validated 2.5 m scale-free residual;
* proof loss returns to the preceding route (native or long-range CEC);
* the opt-in height-scaled local approach additionally requires the immutable
  first-40 camera-height receipt, bound to the current frame;
* within one residual radius it shortens the requested translation; within
  0.60 m it faces the goal camera before awaiting the separate visual detector;
* this adapter never authorizes STOP. The existing RGB arrival detector owns
  that decision and can still reject a visually ambiguous goal.

No semantic Novel/Revisit label is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


SCHEMA_VERSION = "cec_local_approach_handoff_v3_20260907"

# Frozen from the measured NavDP point-token transfer function: injected
# bearings remain faithful through +/-60 degrees, while rearward targets can
# collapse.  Outside this support the actuator layer turns atomically first.
NAVDP_POINT_TOKEN_MAX_ABS_BEARING_DEG = 60.0

# The default and long-range projection remain scale-free. The experimental
# height-scaled local mode may only reduce this radius, never extend it.
CERTIFIED_BEARING_RESIDUAL_M = 2.5
LOCAL_ORIENTATION_RADIUS_M = 0.60
LOCAL_ORIENTATION_TOLERANCE_DEG = 8.0


@dataclass(frozen=True)
class LocalPoseHandoffDecision:
    disposition: str
    direct_proof_active: bool
    local_latched: bool
    reason: str
    controller_pointgoal_m: tuple[float, float] | None
    turn_error_left_rad: float | None
    predicted_distance_m: float | None
    predicted_bearing_deg: float | None
    terminal_yaw_right_deg: float | None
    stop_streak: int
    stop_authorized: bool
    metric_approach_active: bool = False
    orientation_alignment_active: bool = False

    def audit_dict(self) -> dict[str, Any]:
        return {
            "terminal_handoff_schema": SCHEMA_VERSION,
            "terminal_handoff_disposition": self.disposition,
            "terminal_local_latched": self.local_latched,
            "terminal_handoff_reason": self.reason,
            "terminal_controller_pointgoal_m": (
                list(self.controller_pointgoal_m)
                if self.controller_pointgoal_m is not None
                else None
            ),
            "terminal_turn_error_left_rad": self.turn_error_left_rad,
            # Only the explicit local-approach mode may use this for motion;
            # it has no STOP authority in either mode.
            "terminal_predicted_distance_m": self.predicted_distance_m,
            "terminal_predicted_distance_control_authority": self.metric_approach_active,
            "terminal_predicted_bearing_deg": self.predicted_bearing_deg,
            "terminal_yaw_right_deg": self.terminal_yaw_right_deg,
            "terminal_stop_streak": self.stop_streak,
            "terminal_stop_authorized": self.stop_authorized,
            "terminal_proof_active": self.direct_proof_active,
            "terminal_point_token_support_deg": (
                NAVDP_POINT_TOKEN_MAX_ABS_BEARING_DEG
            ),
            "terminal_bearing_residual_m": CERTIFIED_BEARING_RESIDUAL_M,
            "terminal_metric_scale_control_authority": self.metric_approach_active,
            "terminal_goal_orientation_control_authority": self.orientation_alignment_active,
            "terminal_stop_authority": (
                "none_until_independent_visual_convergence"
            ),
        }


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x = _finite_number(value[0])
    y = _finite_number(value[1])
    if x is None or y is None or math.hypot(x, y) <= 1e-8:
        return None
    return x, y


def _certified_direction(
    evidence: Mapping[str, object] | None,
) -> tuple[tuple[float, float], float | None, float | None] | None:
    """Return direction plus diagnostic distance/yaw from a valid proof.

    New servers expose the raw scale-free vector explicitly.  The metric
    vector is accepted as a backward-compatible source of direction. Metric
    control additionally requires raw-vector/scale consistency below.
    """

    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("certificate_accepted") is not True:
        return None
    point = _finite_pair(evidence.get("predicted_scale_free_relative_xy"))
    if point is None:
        point = _finite_pair(evidence.get("predicted_relative_xy_m"))
    if point is None:
        return None
    distance = _finite_number(evidence.get("predicted_distance_m"))
    yaw_right = _finite_number(evidence.get("terminal_yaw_right_deg"))
    return point, distance, yaw_right


def decide_local_pose_handoff(
    *,
    long_range_available: bool,
    evidence: Mapping[str, object] | None,
    local_latched: bool = False,
    stop_streak: int = 0,
    metric_approach: bool = False,
    expected_frame_index: int | None = None,
) -> LocalPoseHandoffDecision:
    """Choose native, long-range, local approach, atomic turn, or visual hold.

    ``local_latched`` and ``stop_streak`` remain in the call signature so a
    router can retain its public call signature. Both are cleared: direct
    PnP does not acquire arrival authority through repeated calls.
    """

    if type(long_range_available) is not bool or type(local_latched) is not bool:
        raise TypeError("route and latch states must be bool")
    if type(stop_streak) is not int or stop_streak < 0:
        raise ValueError("stop_streak must be a non-negative integer")

    def result(
        disposition: str,
        proof: bool,
        reason: str,
        *,
        pointgoal: tuple[float, float] | None = None,
        turn: float | None = None,
        distance: float | None = None,
        bearing: float | None = None,
        yaw_right: float | None = None,
        metric_active: bool = False,
        align_active: bool = False,
    ) -> LocalPoseHandoffDecision:
        return LocalPoseHandoffDecision(
            disposition=disposition,
            direct_proof_active=proof,
            local_latched=False,
            reason=reason,
            controller_pointgoal_m=pointgoal,
            turn_error_left_rad=turn,
            predicted_distance_m=distance,
            predicted_bearing_deg=bearing,
            terminal_yaw_right_deg=yaw_right,
            stop_streak=0,
            stop_authorized=False,
            metric_approach_active=metric_active,
            orientation_alignment_active=align_active,
        )

    direct = _certified_direction(evidence)
    if direct is None:
        if long_range_available:
            return result(
                "long_range", False, "direct_bearing_certificate_unavailable"
            )
        return result(
            "native", False, "direct_bearing_certificate_unavailable"
        )

    point, diagnostic_distance, yaw_right = direct
    norm = math.hypot(point[0], point[1])
    unit = point[0] / norm, point[1] / norm
    bearing_rad = math.atan2(unit[1], unit[0])
    bearing_deg = math.degrees(bearing_rad)

    # Optional, explicitly versioned real-robot terminal adapter. The radius
    # can only shrink; scale does not authorize STOP. Use the same first-40
    # receipt as mono depth, never the old GOAT first-64 calibration or GT.
    scale = evidence.get("metric_scale")
    scale = scale if isinstance(scale, Mapping) else {}
    quality = scale.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    raw_point = _finite_pair(evidence.get("predicted_scale_free_relative_xy"))
    scale_value = _finite_number(scale.get("metric_scale_m_per_raw"))
    scaled_norm = (None if raw_point is None or scale_value is None else
                   math.hypot(*raw_point) * scale_value)
    metric_valid = bool(
        metric_approach and expected_frame_index is not None
        and evidence.get("frame_index") == expected_frame_index
        and evidence.get("metric_scale_available") is True
        and evidence.get("metric_scale_policy") == "mdtec_first40"
        and scale.get("available") is True
        and scale.get("frame_count") == 40
        and scale.get("scale_evidence_contract") == "causal_first_prefix_rgb_only_v1"
        and quality.get("scale_clamped") in (False, 0.0)
        and isinstance(scale.get("scale_receipt_sha256"), str)
        and len(scale["scale_receipt_sha256"]) == 64
        and scale_value is not None and scale_value > 0
        and diagnostic_distance is not None and 0 < diagnostic_distance <= CERTIFIED_BEARING_RESIDUAL_M
        and scaled_norm is not None
        and math.isclose(scaled_norm, diagnostic_distance, rel_tol=1e-5, abs_tol=1e-6)
    )
    if metric_valid and diagnostic_distance <= LOCAL_ORIENTATION_RADIUS_M and yaw_right is not None:
        yaw_left = math.atan2(math.sin(-math.radians(yaw_right)),
                              math.cos(-math.radians(yaw_right)))
        if abs(yaw_left) > math.radians(LOCAL_ORIENTATION_TOLERANCE_DEG):
            return result("atomic_turn", True, "near_goal_camera_orientation",
                          turn=yaw_left, distance=diagnostic_distance,
                          bearing=bearing_deg, yaw_right=yaw_right,
                          metric_active=True, align_active=True)
        return result("hold", True, "near_goal_await_independent_visual_arrival",
                      distance=diagnostic_distance, bearing=bearing_deg,
                      yaw_right=yaw_right, metric_active=True)

    if abs(bearing_deg) > NAVDP_POINT_TOKEN_MAX_ABS_BEARING_DEG:
        return result(
            "atomic_turn",
            True,
            "direct_bearing_outside_point_token_support",
            turn=bearing_rad,
            distance=diagnostic_distance,
            bearing=bearing_deg,
            yaw_right=yaw_right,
            metric_active=metric_valid,
        )

    radius = diagnostic_distance if metric_valid else CERTIFIED_BEARING_RESIDUAL_M
    return result(
        "bearing_local",
        True,
        "height_scaled_local_approach" if metric_valid else "direct_scale_free_bearing_certified",
        pointgoal=(radius * unit[0], radius * unit[1]),
        distance=diagnostic_distance,
        bearing=bearing_deg,
        yaw_right=yaw_right,
        metric_active=metric_valid,
    )


__all__ = [
    "CERTIFIED_BEARING_RESIDUAL_M",
    "LocalPoseHandoffDecision",
    "NAVDP_POINT_TOKEN_MAX_ABS_BEARING_DEG",
    "SCHEMA_VERSION",
    "decide_local_pose_handoff",
]
