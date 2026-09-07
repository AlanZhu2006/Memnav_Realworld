"""Reject old visual observations before starting a local action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from trajectory_control import VelocityCommand


@dataclass(frozen=True)
class LatencyMotionGuardConfig:
    enabled: bool = True
    max_plan_input_age_s: float = 1.50


@dataclass(frozen=True)
class LatencyMotionGuardResult:
    command: VelocityCommand
    reason: str
    plan_input_age_s: float | None
    raw_linear_mps: float
    raw_angular_rps: float

    def audit_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = asdict(self.command)
        return payload


def _stopped(command: VelocityCommand) -> VelocityCommand:
    return VelocityCommand(
        target_x=command.target_x,
        target_y=command.target_y,
        path_length=command.path_length,
        reverse=command.reverse,
    )


class LatencyMotionGuard:
    """Reject trajectories inferred from an excessively old observation.

    Paths are anchored at RGB exposure and followed using fresh measured pose.
    Replanning may happen during motion; do not reinterpret an old camera-frame
    path as if it were expressed at inference return. No cross-plan sign vote
    is applied: the executor compensates the coordinates instead.
    """

    def __init__(
        self,
        config: LatencyMotionGuardConfig = LatencyMotionGuardConfig(),
    ) -> None:
        self.config = config

    def reset(self) -> None:
        """Retained for callers that reset all navigation guards."""

    def apply(
        self,
        command: VelocityCommand,
        *,
        plan_input_age_s: float | None,
    ) -> LatencyMotionGuardResult:
        raw_linear = float(command.linear_x)
        raw_angular = float(command.angular_z)
        if not self.config.enabled:
            return LatencyMotionGuardResult(
                command, "disabled", plan_input_age_s, raw_linear, raw_angular
            )

        try:
            age = float(plan_input_age_s)
        except (TypeError, ValueError, OverflowError):
            age = math.nan
        if not math.isfinite(age) or age < 0.0:
            return LatencyMotionGuardResult(
                _stopped(command),
                "invalid_plan_input_age_hold",
                None,
                raw_linear,
                raw_angular,
            )
        if age > self.config.max_plan_input_age_s:
            return LatencyMotionGuardResult(
                _stopped(command),
                "plan_input_too_old_hold",
                age,
                raw_linear,
                raw_angular,
            )
        return LatencyMotionGuardResult(
            command, "pass", age, raw_linear, raw_angular
        )
