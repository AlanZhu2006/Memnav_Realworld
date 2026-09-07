#!/usr/bin/env python3
"""Live Jetson/GPU performance measurement; never arm or address a motor.

Runs the actual adapter while disabled, with isolated ROS topics. A timed
observation-only window exercises the same request used during an atomic turn,
without pretending that a physical turn happened. Optional arrival work calls
the real image matcher but has no publisher or stop authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

from image_goal_io import load_rgb_image
from navdp_client import NavDPClient
from navdp_ros_node import NavDPGo2Adapter
from rgb_goal_arrival import RgbGoalArrivalVerifier


PHASES = [("warmup", 15), ("policy_a1", 30), ("policy_arrival_b1", 30),
          ("geometry_only", 20), ("policy_arrival_b2", 30), ("policy_a2", 30)]


def quant(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (None if not len(values) else dict(n=len(values),
            p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)), maximum=float(max(values))))


class Snapshot(Node):
    def __init__(self):
        super().__init__("cec_latency_camera_snapshot")
        self.bridge = CvBridge()
        self.rgb = self.intrinsic = None
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.on_rgb,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self.on_info,
                                 qos_profile_sensor_data)

    def on_rgb(self, message):
        self.rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8").copy()

    def on_info(self, message):
        self.intrinsic = np.asarray(message.k).reshape(3, 3)


class MeasuredAdapter(NavDPGo2Adapter):
    def __init__(self, out, goal):
        self.bench_start = time.monotonic()
        self.records = []
        self.record_lock = threading.Lock()
        self.trace = (out / "live_trace.jsonl").open("x")
        self.current_snapshot = None
        self.control_times = []
        self.nonzero_attempts = 0
        self.arrival_stamp = 0
        super().__init__()
        if (self.cmd_vel_topic != "/cec_latency/cmd_vel"
                or self._enabled or not self._estop):
            raise RuntimeError("latency measurement requires isolated, locked adapter")
        self.arrival_verifier = RgbGoalArrivalVerifier(load_rgb_image(goal))
        self.arrival_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(1 / 12, self.measure_arrival, callback_group=self.arrival_group)
        self.original_send = self._client.session.send

        def timed_send(request, **kwargs):
            start = time.monotonic()
            phase = self.phase_at(start)
            try:
                response = self.original_send(request, **kwargs)
                body = response.content  # include response-body transfer
                elapsed = time.monotonic() - start
                receipt = response.json()
                timing = self.current_snapshot or {}
                stamp = timing.get("rgb_stamp_ns")
                self.record("http", phase=phase, endpoint=urlparse(request.url).path,
                            elapsed_s=elapsed, status=response.status_code, response_bytes=len(body),
                            rgb_to_response_s=(None if not stamp else
                                (self.get_clock().now().nanoseconds - stamp) / 1e9),
                            frame=receipt.get("cec_frame_idx", receipt.get("frame_idx")),
                            anchor=receipt.get("cec_selected_anchor"),
                            takeover=receipt.get("cec_takeover"),
                            terminal=receipt.get("terminal_handoff_reason"),
                            probe_timing=receipt.get("cec_retrieval_probe_timing"),
                            relocalization_ms=receipt.get("cec_relocalization_ms"),
                            navdp_timing=receipt.get("navdp_runtime_timing"),
                            depth_receipt=receipt.get("monocular_depth_receipt"),
                            fifo_updated=receipt.get("policy_fifo_updated"),
                            sealed_survey_updated=receipt.get("sealed_survey_updated"),
                            error=receipt.get("error"))
                return response
            except Exception as error:
                self.record("http_error", phase=phase, elapsed_s=time.monotonic()-start,
                            error=f"{type(error).__name__}:{error}")
                raise
        self._client.session.send = timed_send

    def phase_at(self, now):
        elapsed = now - self.bench_start
        for phase, seconds in PHASES:
            if elapsed < seconds:
                return phase
            elapsed -= seconds
        return "complete"

    def record(self, kind, **payload):
        record = dict(kind=kind, at_s=time.monotonic()-self.bench_start, **payload)
        with self.record_lock:
            self.records.append(record)
            self.trace.write(json.dumps(record, allow_nan=False) + "\n")
            self.trace.flush()

    def _on_enable(self, message):
        # Measurement does not accept arming, even on its isolated namespace.
        return

    def _set_enabled_service(self, request, response):
        response.success = False
        response.message = "motion disabled in latency measurement"
        return response

    def _on_estop(self, message):
        if message.data:
            super()._on_estop(message)

    def _publish_command(self, command, reason):
        if command.linear_x != 0 or command.angular_z != 0:
            self.nonzero_attempts += 1
            raise RuntimeError("nonzero command attempted in locked benchmark")
        super()._publish_command(command, reason)

    def _control_tick(self):
        start = time.monotonic()
        self.control_times.append((start, self.phase_at(start)))
        super()._control_tick()  # actual disabled-control callback, not moving tracking
        self.record("control", phase=self.phase_at(start), runtime_s=time.monotonic()-start,
                    enabled=self._enabled, estop=self._estop,
                    cmd_vx=self._last_command.linear_x, cmd_wz=self._last_command.angular_z)

    def _snapshot_inference_input(self):
        snapshot, reason = super()._snapshot_inference_input()
        if snapshot is not None:
            timing = snapshot[-1]
            if self.phase_at(time.monotonic()) == "geometry_only":
                # Scheduling probe only: no synthetic IMU turn or motion.
                timing["observation_only"] = True
            self.current_snapshot = dict(timing)
        return snapshot, reason

    def _publish_receipt(self, event, receipt):
        self.record("adapter_receipt", phase=self.phase_at(time.monotonic()), event=event,
                    admission_reason=receipt.get("reason"))
        super()._publish_receipt(event, receipt)

    def measure_arrival(self):
        phase = self.phase_at(time.monotonic())
        if "arrival" not in phase:
            return
        with self._lock:
            stamp = int(self._rgbd_diagnostic.get("rgb_stamp_ns") or 0)
            if self._rgb is None or stamp <= self.arrival_stamp:
                return
            self.arrival_stamp = stamp
            image = self._rgb.copy()
        start = time.monotonic()
        result = self.arrival_verifier.evaluate(image)
        self.record("arrival_compute", phase=phase, runtime_s=time.monotonic()-start,
                    rgb_age_at_finish_s=(self.get_clock().now().nanoseconds-stamp)/1e9,
                    matched=result.matched, reason=result.reason,
                    motion_or_stop_authority=False)

    def summarize(self, out, prepared):
        rows = list(self.records)
        summary = dict(schema="cec_locked_live_latency_v1", phases=PHASES,
                       safety="no motor bridge; isolated ROS command topic; enabled=false, estop=true",
                       limitation="stationary camera, no measured motion or navigation SR",
                       geometry_window="observation-only scheduling; robot did not turn",
                       arrival_workload="real matcher in an isolated timer; no arrival/STOP publisher",
                       nonzero_command_attempts=self.nonzero_attempts, preparation=prepared, metrics={})
        for phase, duration in PHASES:
            http = [r for r in rows if r["kind"] == "http" and r["phase"] == phase]
            controls = [r for r in rows if r["kind"] == "control" and r["phase"] == phase]
            arrival = [r for r in rows if r["kind"] == "arrival_compute" and r["phase"] == phase]
            pts = [t for t, p in self.control_times if p == phase]
            times = [r["at_s"] for r in http]
            summary["metrics"][phase] = dict(
                requests=len(http), endpoints=dict(Counter(r["endpoint"] for r in http)),
                http_s=quant([r["elapsed_s"] for r in http]),
                rgb_to_response_s=quant([r["rgb_to_response_s"] for r in http if r["rgb_to_response_s"] is not None]),
                request_intervals_s=quant(np.diff(times)),
                responses_over_1_5s=sum((r["rgb_to_response_s"] or 0) > 1.5 for r in http),
                http_errors=sum(r["status"] >= 400 for r in http),
                takeovers=sum(r["takeover"] is True for r in http),
                control_callback_intervals_s=quant(np.diff(pts)),
                control_callback_runtime_s=quant([r["runtime_s"] for r in controls]),
                control_callbacks=len(controls),
                all_control_callbacks_locked=all(not r["enabled"] and r["estop"] and not r["cmd_vx"] and not r["cmd_wz"] for r in controls),
                arrival_compute_s=quant([r["runtime_s"] for r in arrival]),
                arrival_finish_age_s=quant([r["rgb_age_at_finish_s"] for r in arrival]))
        summary["exceptions"] = [r for r in rows if r["kind"] == "http_error"]
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--prepared-receipt", type=Path,
                        help="reuse only this isolated benchmark's verified preparation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init(args=[])
    snapshot = Snapshot()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(snapshot)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    deadline = time.monotonic()+20
    while (snapshot.rgb is None or snapshot.intrinsic is None) and time.monotonic()<deadline:
        time.sleep(.05)
    if snapshot.rgb is None or snapshot.intrinsic is None:
        raise RuntimeError("live RGB/CameraInfo unavailable; no performance claim")
    client = NavDPClient("http://127.0.0.1:28890", 3, 180)
    client.validate_cec_contract(client.health())
    if args.prepared_receipt:
        preparation = json.loads(args.prepared_receipt.read_text())
        health = client.health()
        if (health.get("active_goal_sha256") != preparation["prepare"]["selected_goal"]["sha256"]
                or health["episodic_dataset"].get("loaded_dataset_id") != args.dataset
                or preparation["source_goal_sha256"] != hashlib.sha256(Path(args.goal).read_bytes()).hexdigest()):
            raise RuntimeError("prepared benchmark goal/history changed")
        preparation["reused_after_benchmark_startup_error"] = True
    else:
        start = time.monotonic()
        client.reset(snapshot.intrinsic)
        reset_s = time.monotonic()-start
        start = time.monotonic()
        dataset = client.load_dataset(args.dataset)
        load_s = time.monotonic()-start
        start = time.monotonic()
        prepared, _ = client.prepare_revisit_goal(load_rgb_image(args.goal), snapshot.rgb.copy())
        prepare_s = time.monotonic()-start
        preparation = dict(reset_s=reset_s, dataset_load_s=load_s, prepare_s=prepare_s,
                           dataset=dataset, prepare=prepared, intrinsic=snapshot.intrinsic.tolist(),
                           source_goal_sha256=hashlib.sha256(Path(args.goal).read_bytes()).hexdigest())
    (args.output / "preparation.json").write_text(json.dumps(preparation,indent=2)+"\n")
    print("LATENCY_PREPARED "+json.dumps({k:v for k,v in preparation.items() if k.endswith('_s')}),flush=True)
    executor.remove_node(snapshot)
    snapshot.destroy_node()
    executor.shutdown(timeout_sec=2)
    spin.join(3)
    rclpy.shutdown()
    rclpy.init(args=["--ros-args", "--params-file", args.params,
                    "-r", "__ns:=/cec_latency"])
    adapter = MeasuredAdapter(args.output, args.goal)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(adapter)
    try:
        while adapter.phase_at(time.monotonic()) != "complete":
            executor.spin_once(timeout_sec=.1)
    finally:
        adapter.stop()
        deadline = time.monotonic()+10
        while adapter._inference_busy and time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.1)
        executor.remove_node(adapter)
        executor.shutdown(timeout_sec=2)
        adapter.summarize(args.output, preparation)
        adapter.trace.close()
        adapter.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
