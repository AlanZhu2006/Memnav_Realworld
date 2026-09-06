"""Track a frozen local path against measured Go2 position, then replan."""

from collections import deque
from dataclasses import replace
import math

import numpy as np

from heading_turn import wrap
from trajectory_control import VelocityCommand, trajectory_to_command


class PlanCycle:
    """Sequencing only: there is no distance, turn-angle or action-time budget."""

    def __init__(self, settle_s=0.15):
        if not math.isfinite(settle_s) or settle_s < 0:
            raise ValueError("settle time must be finite and nonnegative")
        self.settle_s = settle_s
        self.reset()

    def reset(self):
        self.state = "need_plan"
        self.stopped_s = None
        self.sense_after_ns = None

    def install_plan(self, completed_s):
        self.state = "ready_to_execute"
        self.stopped_s = None
        self.sense_after_ns = None

    def start_execution(self):
        self.state = "execute"

    def note_action_stopped(self, now_s, stopped_ros_ns):
        if self.state not in {"ready_to_execute", "execute"}:
            return
        self.state = "settling"
        self.stopped_s = now_s
        self.sense_after_ns = (None if not stopped_ros_ns else
                               stopped_ros_ns + math.ceil(self.settle_s * 1e9))

    def phase(self, *, now_s, latest_rgbd_source_ns):
        if self.state == "settling" and now_s - self.stopped_s >= self.settle_s:
            self.state = "waiting_for_post_stop_rgbd"
        if (self.state == "waiting_for_post_stop_rgbd" and self.sense_after_ns is not None
                and latest_rgbd_source_ns and latest_rgbd_source_ns > self.sense_after_ns):
            self.state = "ready_to_plan"
        return self.state

    @staticmethod
    def planning_allowed(phase):
        return phase in {"need_plan", "ready_to_plan"}

    @staticmethod
    def motion_allowed(phase):
        return phase in {"ready_to_execute", "execute"}

    def audit_dict(self, *, now_s, latest_rgbd_source_ns):
        return dict(phase=self.phase(now_s=now_s, latest_rgbd_source_ns=latest_rgbd_source_ns),
                    sense_after_ros_ns=self.sense_after_ns)


class TrajectoryExecution:
    """Use SportModeState pose for local execution, never as policy input or GT."""

    def __init__(
        self,
        *,
        completion_tolerance_m=0.15,
        stagnation_timeout_s=4.0,
        stagnation_min_progress_m=0.02,
        stagnation_min_linear_mps=0.08,
    ):
        values = (
            completion_tolerance_m,
            stagnation_timeout_s,
            stagnation_min_progress_m,
            stagnation_min_linear_mps,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("trajectory execution thresholds must be finite and positive")
        self.completion_tolerance_m = float(completion_tolerance_m)
        self.stagnation_timeout_s = float(stagnation_timeout_s)
        self.stagnation_min_progress_m = float(stagnation_min_progress_m)
        self.stagnation_min_linear_mps = float(stagnation_min_linear_mps)
        self.samples = deque(maxlen=400)
        self.reset()

    def reset(self):
        self.active = False
        self.phase = "idle"
        self.path = None
        self.progress_m = 0.0
        self.remaining_m = None
        self.endpoint_distance_m = None
        self.completed_s = None
        self.last_sample = None
        self.stagnation_started_s = None
        self.stagnation_progress_m = None

    def observe(self, stamp_ns, x, y, yaw):
        if stamp_ns <= 0 or not all(math.isfinite(v) for v in (x, y, yaw)):
            return
        if self.samples and stamp_ns <= self.samples[-1][0]:
            return
        self.samples.append((stamp_ns, x, y, wrap(yaw)))

    def age(self, now_ns):
        return None if not self.samples else (now_ns - self.samples[-1][0]) / 1e9

    def reference(self, image_ns):
        if not self.samples or not image_ns:
            return None
        sample = min(self.samples, key=lambda p: abs(p[0] - image_ns))
        return sample if abs(sample[0] - image_ns) <= 150_000_000 else None

    def start(self, local_path, image_ns, now_ns):
        self.reset()
        ref = self.reference(image_ns)
        age = self.age(now_ns)
        if ref is None or age is None or not 0 <= age <= 0.35:
            self.phase = "position_feedback_unavailable"
            return False
        points = np.asarray(local_path, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points[:, :2]).all():
            self.phase = "invalid_execution_path"
            return False
        points = np.vstack(([0., 0.], points[:, :2]))
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5]]
        if len(points) < 2:
            self.phase = "empty_path"
            return False
        _, x, y, yaw = ref
        c, s = math.cos(yaw), math.sin(yaw)
        self.path = points @ np.array([[c, s], [-s, c]]) + [x, y]
        self.lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.arc = np.r_[0., np.cumsum(self.lengths)]
        self.remaining_m = float(self.arc[-1])
        self.last_sample = self.samples[-1]
        self.active = True
        self.phase = "tracking"
        return True

    def step(self, now_ns, now_s, config):
        if not self.active:
            return VelocityCommand()
        age = self.age(now_ns)
        if age is None or not 0 <= age <= 0.35:
            return self._fail("position_feedback_stale")
        sample = self.samples[-1]
        stamp, x, y, yaw = sample
        last_stamp, lx, ly, last_yaw = self.last_sample
        if stamp > last_stamp:
            dt = (stamp - last_stamp) / 1e9
            if (math.hypot(x-lx, y-ly) > 3 * dt + 0.15
                    or abs(wrap(yaw-last_yaw)) > 3 * dt + 0.10):
                return self._fail("position_feedback_discontinuity")
            self.last_sample = sample
        pos = np.array([x, y])
        # Project onto the remaining nearby path. Restrict the search in arc
        # length so a self-crossing cannot jump directly to a later branch.
        best = None
        for i, length in enumerate(self.lengths):
            if self.arc[i+1] < self.progress_m or self.arc[i] > self.progress_m + config.lookahead_m:
                continue
            delta = self.path[i+1] - self.path[i]
            t = float(np.clip(np.dot(pos-self.path[i], delta) / length**2, 0, 1))
            progress = max(self.progress_m, float(self.arc[i] + t * length))
            distance = float(np.linalg.norm(pos - (self.path[i] + t * delta)))
            if best is None or distance < best[0]:
                best = (distance, progress)
        if best is not None:
            self.progress_m = best[1]
        self.remaining_m = max(0., float(self.arc[-1]) - self.progress_m)
        self.endpoint_distance_m = float(np.linalg.norm(self.path[-1]-pos))
        # This completes only the current local action. The post-stop RGB-D
        # cycle still has to produce a fresh plan before motion can continue.
        # Go2 cannot reliably realize the vanishing velocities produced by an
        # 8 cm tail, so stop before entering that gait dead zone.
        if (self.remaining_m <= self.completion_tolerance_m
                and self.endpoint_distance_m <= self.completion_tolerance_m):
            self.active = False
            self.phase = "complete"
            self.completed_s = now_s
            return VelocityCommand()
        target_arc = min(self.progress_m + config.lookahead_m, self.arc[-1])
        index = min(int(np.searchsorted(self.arc, target_arc, side="right"))-1, len(self.lengths)-1)
        fraction = (target_arc-self.arc[index]) / self.lengths[index]
        target = self.path[index] + fraction * (self.path[index+1]-self.path[index])
        delta = target-pos
        c, s = math.cos(yaw), math.sin(yaw)
        body = delta @ np.array([[c, -s], [s, c]])
        command = trajectory_to_command(np.array([[0., 0.], body]), config)
        if abs(command.linear_x) >= self.stagnation_min_linear_mps:
            if self.stagnation_started_s is None:
                self.stagnation_started_s = now_s
                self.stagnation_progress_m = self.progress_m
            elif self.progress_m - self.stagnation_progress_m >= self.stagnation_min_progress_m:
                self.stagnation_started_s = now_s
                self.stagnation_progress_m = self.progress_m
            elif now_s - self.stagnation_started_s >= self.stagnation_timeout_s:
                self.active = False
                self.phase = "stalled_replan"
                self.completed_s = now_s
                return VelocityCommand()
        else:
            # Rotation-only control and intentionally tiny translations do not
            # prove a walking failure. Restart the watchdog when forward motion
            # is requested again.
            self.stagnation_started_s = None
            self.stagnation_progress_m = None
        return replace(command, path_length=self.remaining_m)

    def _fail(self, phase):
        self.active = False
        self.phase = phase
        return VelocityCommand()

    def audit(self, now_ns, now_s):
        return dict(active=self.active, phase=self.phase, progress_m=self.progress_m,
                    remaining_m=self.remaining_m, endpoint_distance_m=self.endpoint_distance_m,
                    feedback_age_s=self.age(now_ns), feedback_source="go2_sportmodestate",
                    completed_age_s=None if self.completed_s is None else now_s-self.completed_s,
                    completion_tolerance_m=self.completion_tolerance_m,
                    stagnation_age_s=(None if self.stagnation_started_s is None
                                      else now_s-self.stagnation_started_s),
                    stagnation_timeout_s=self.stagnation_timeout_s,
                    stagnation_min_progress_m=self.stagnation_min_progress_m,
                    stagnation_min_linear_mps=self.stagnation_min_linear_mps)
