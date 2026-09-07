#!/usr/bin/env python3
"""Single-entry monocular real-world bridge for CEC and frozen NavDP.

The Go2 client sees the ordinary NavDP ``/navigator_reset`` and
``/imagegoal_step`` contract, extended by an explicit two-phase episode
protocol.  After reset the hub is in the memory-recording phase.  A client may
call ``/memory_step`` for a teleoperated prefix, or
``/novel_imagegoal_step`` to append the same causal RGB to the LingBot stream
while frozen native NavDP executes an independently supplied Novel goal.  At
the revisit start point the client calls ``/begin_revisit``; only then does
the CEC-routed ``/imagegoal_step`` become legal, so the first Revisit query --
which freezes MemNav's goal session and candidate ceiling -- happens after the
recorded history instead of at frame zero.

Internally the query phase advances the same causal RGB stream, performs the
frozen Certified Episodic Compass decision, and delegates control to either
monocular-native ImageGoal NavDP or the mixed image/PointGoal controller.
Client depth is accepted only for wire compatibility with the robot-side
safety stack; it is never forwarded to the navigation policy.

This process owns no actuator interface.  It is intended to listen on
loopback and be reached through an SSH local-forward from the robot computer.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
import threading
from typing import Any, Mapping

from flask import Flask, jsonify, request
import requests

from deployment.gpu.episodic_dataset import (
    DatasetContractError,
    EpisodicDatasetStore,
)
from deployment.gpu.revisit_bearing_adapter import adapt_revisit_pointgoal
from deployment.gpu.revisit_local_pose_adapter import (
    SCHEMA_VERSION as TERMINAL_HANDOFF_SCHEMA,
    decide_local_pose_handoff,
)


PROTOCOL_VERSION = 3
PHASE_RECORDING = "memory_recording"
PHASE_REVISIT = "revisit_query"
# NavDP observation-FIFO warm-up at the recording->revisit switch, mirroring
# the simulator's shared-trace replay: the sim reconstructs NavDP's queue
# from the leg-A frames at plan steps (one plan every exec_horizon=8 executed
# steps, queue depth memory_size=8).  The real analog is a stride-8 tail of
# the recorded frames, replayed oldest-first through /memory_replay_step.
NAVDP_WARMUP_MAX_FRAMES = 8
NAVDP_WARMUP_STRIDE = 8
GOAL_SCORE_STRIDE = 8
GOAL_MIN_FRAME_GAP = 16
GOAL_MIN_INLIERS = 16
GOAL_MAX_COS = 0.90


def select_warmup_frames(
    tail: list[tuple[int, bytes]], stride: int, max_frames: int
) -> list[tuple[int, bytes]]:
    """Newest-anchored strided selection, returned in chronological order."""
    picked = list(tail)[::-1][::max(1, int(stride))][: max(0, int(max_frames))]
    return picked[::-1]
POINTGOAL_UNITS = "lingbot_raw_direction_only"
PROPOSAL_ORDER = "geometry_first"
NAVIGATION_SENSOR_CONTRACT = "causal_monocular_rgb_v1"
NAVDP_DEPTH_SOURCE = "monocular_sidecar"
CLIENT_DEPTH_CONTRACT = "local_safety_only_not_forwarded"
AUTHORITY_MODES = ("cec", "native")
RETRIEVAL_TRACE_SCHEMA = "cec_online_retrieval_trace_v1"
RELOCALIZATION_TRACE_SCHEMA = "cec_online_relocalization_trace_v1"


class HybridBackendError(RuntimeError):
    """A stateful upstream failed and the session can no longer continue."""


@dataclass(frozen=True)
class UpstreamConfig:
    memnav_url: str
    navdp_url: str
    camera_height_m: float
    connect_timeout_s: float = 3.0
    request_timeout_s: float = 180.0
    navdp_depth_source: str = NAVDP_DEPTH_SOURCE
    goal_score_stride: int = GOAL_SCORE_STRIDE
    goal_min_frame_gap: int = GOAL_MIN_FRAME_GAP
    goal_min_inliers: int = GOAL_MIN_INLIERS
    goal_max_cos: float = GOAL_MAX_COS
    authority_mode: str = "cec"
    terminal_approach: str = "bearing_only"
    historical_depth_source: str = "canonical"

    def __post_init__(self) -> None:
        if self.historical_depth_source not in ("canonical", "online_history"):
            raise ValueError("unsupported historical depth source")
        if self.terminal_approach not in {"bearing_only", "height_scaled_local"}:
            raise ValueError("unsupported terminal approach")
        if self.authority_mode not in AUTHORITY_MODES:
            raise ValueError(
                f"unsupported authority mode {self.authority_mode!r}; "
                f"choose one of {AUTHORITY_MODES}"
            )

    @property
    def timeout(self) -> tuple[float, float]:
        return (self.connect_timeout_s, self.request_timeout_s)


def _json_object(response: requests.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise HybridBackendError(f"{label} returned non-object JSON")
    return payload


def _file(name: str, payload: bytes, media_type: str) -> tuple[str, io.BytesIO, str]:
    return (name, io.BytesIO(payload), media_type)


def _retrieval_trace(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the bounded online retrieval evidence that cannot be recreated exactly."""

    keys = (
        "retrieved_anchor",
        "raw_score",
        "retrieval_second_score",
        "visual_anchor",
        "visual_score",
        "visual_second_score",
        "selected_anchor_score",
        "predicted_gate",
        "forced_gate",
        "current_goal_cos",
        "candidate_count",
        "goal_start_frame",
        "candidate_ceiling",
        "certified_visual_candidates",
    )
    return {
        "schema": RETRIEVAL_TRACE_SCHEMA,
        **{key: probe[key] for key in keys if key in probe},
    }


def _relocalization_trace(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep numeric co-visibility/PnP evidence without copying image payloads."""

    keys = (
        "selected_dino_rank",
        "selected_proposal_source",
        "selected_anchor_image_sha256",
        "ranked_candidates",
        "proposal_attempts",
        "pnp",
        "reference_depth_cache",
        "reference_depth_source",
        "cached",
    )
    return {
        "schema": RELOCALIZATION_TRACE_SCHEMA,
        **{key: certificate[key] for key in keys if key in certificate},
    }


def _bind_monocular_depth_transaction(
    form: Mapping[str, str],
    append_receipt: Mapping[str, Any],
    image: bytes,
) -> dict[str, str]:
    """Bind NavDP to the exact depth materialized by this RGB append."""

    bound = dict(form)
    image_digest = hashlib.sha256(image).hexdigest()
    if append_receipt.get("image_sha256") != image_digest:
        raise HybridBackendError(
            "MemNav planning append received different JPEG bytes"
        )
    token = append_receipt.get("monocular_depth_transaction_token")
    frame_index = append_receipt.get("monocular_depth_frame_index")
    frame_idx = append_receipt.get("frame_idx")
    if (
        not isinstance(token, str)
        or len(token) != 64
        or any(character not in "0123456789abcdef" for character in token)
        or isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index != frame_idx
    ):
        raise HybridBackendError(
            "MemNav planning append returned an invalid depth transaction"
        )
    bound.update({
        "monocular_depth_transaction_token": token,
        "monocular_depth_frame_index": str(frame_index),
    })
    return bound


def _finite_intrinsic(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("intrinsic must be a finite 3x3 matrix")
        parsed = [float(item) for item in row]
        if not all(math.isfinite(item) for item in parsed):
            raise ValueError("intrinsic must be a finite 3x3 matrix")
        matrix.append(parsed)
    return matrix


class CecHybridRouter:
    """Stateful exactly-one-probe CEC router with a native safety fallback."""

    def __init__(
        self,
        config: UpstreamConfig,
        *,
        session: requests.Session | None = None,
        dataset_store: EpisodicDatasetStore | None = None,
        auto_dataset_id: str | None = None,
        auto_dataset_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.dataset_store = dataset_store
        self.auto_dataset_id = (
            None if auto_dataset_id in (None, "") else str(auto_dataset_id)
        )
        self.auto_dataset_metadata = dict(auto_dataset_metadata or {})
        self.initialized = False
        self.memory_degraded = False
        self.native_state_uncertain = False
        self.step_index = 0
        self.phase: str | None = None
        self.frames_recorded = 0
        self.revisit_started_after_frame: int | None = None
        self.goal_candidates: list[dict[str, Any]] = []
        self.active_goal: dict[str, Any] | None = None
        self.last_prepare_receipt: dict[str, Any] | None = None
        self.goal_candidate_dir: str | None = None
        self.navdp_live_recording_steps = 0
        self.navdp_live_queue_lengths: list[int] | None = None
        self.navdp_live_memory_size: int | None = None
        self.terminal_local_latched = False
        self.terminal_stop_streak = 0
        self.loaded_dataset_id: str | None = None
        self.loaded_dataset_manifest_sha256: str | None = None
        self.recorded_tail: deque[tuple[int, bytes]] = deque(
            maxlen=NAVDP_WARMUP_MAX_FRAMES * NAVDP_WARMUP_STRIDE)
        self.query_observation_count = 0

    def query_observation_step(self, image: bytes, installed_goal_sha256: str) -> dict[str, Any]:
        """One serialized query-time RGB append, with no retrieval or policy call.

        The sealed Survey and its candidate ceiling are not enlarged. These
        intermediate views only keep the causal current geometry continuous.
        The caller uses the same worker/HTTP lock as planning, never a second
        concurrent writer to LingBot.

        Continuity here means RGB ingestion, not verified pose accuracy. IMU
        yaw is not fused into LingBot; pure-rotation drift remains unvalidated.
        """
        if not self.initialized or self.phase != PHASE_REVISIT:
            raise ValueError("query observation requires an initialized revisit_query phase")
        if self.memory_degraded or self.native_state_uncertain:
            raise HybridBackendError("state is uncertain; reset is required")
        if self.active_goal is not None and installed_goal_sha256 != self.active_goal["sha256"]:
            raise ValueError("query observation goal identity changed")
        if not image:
            raise ValueError("image is required")
        try:
            payload = _json_object(self.session.post(
                f"{self.config.memnav_url}/memory_step",
                files={"image": _file("image.jpg", image, "image/jpeg")},
                data={"materialize_monocular_depth": "0"},
                timeout=self.config.timeout,
            ), "query geometry observation")
            if (payload.get("image_sha256") != hashlib.sha256(image).hexdigest()
                    or type(payload.get("frame_idx")) is not int):
                raise HybridBackendError("query geometry append did not acknowledge this RGB")
        except Exception as error:
            self.memory_degraded = True
            raise HybridBackendError(f"query geometry append failed; reset required: {error}") from error
        self.query_observation_count += 1
        return {"phase": self.phase, "frame_idx": payload.get("frame_idx"),
                "image_sha256": payload.get("image_sha256"),
                "query_observation_count": self.query_observation_count,
                "policy_fifo_updated": False, "sealed_survey_updated": False}

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.query_observation_count = 0
        resume_empty_auto_dataset = False
        if self.dataset_store is not None and self.dataset_store.recording:
            dataset_status = self.dataset_store.status()
            resume_empty_auto_dataset = (
                self.auto_dataset_id is not None
                and dataset_status.get("dataset_id") == self.auto_dataset_id
                and int(dataset_status.get("memory_frames", -1)) == 0
                and int(dataset_status.get("goal_candidates", -1)) == 0
            )
            if not resume_empty_auto_dataset:
                raise ValueError(
                    "an episodic dataset is still recording; seal it before reset"
                )
        intrinsic = _finite_intrinsic(payload.get("intrinsic"))
        navdp_payload = dict(payload)
        navdp_payload["intrinsic"] = intrinsic
        # The robot client cannot silently switch the deployed navigation
        # policy back to sensor depth.  D435i depth remains local to the
        # collision safety layer and is not part of this upstream contract.
        navdp_payload["depth_source"] = self.config.navdp_depth_source
        camera_height_m = float(self.config.camera_height_m)
        if not math.isfinite(camera_height_m) or not 0.1 <= camera_height_m <= 2.0:
            raise ValueError("camera_height_m must be finite and in [0.1, 2.0]")
        memnav_payload = {
            "camera_height": camera_height_m,
            "camera_intrinsic": intrinsic,
            "seed": payload.get("seed"),
            "episode_len": payload.get("episode_len"),
        }
        self.initialized = False
        self.memory_degraded = False
        self.native_state_uncertain = False
        self.step_index = 0
        self.phase = None
        self.frames_recorded = 0
        self.revisit_started_after_frame = None
        self.goal_candidates = []
        self.active_goal = None
        self.last_prepare_receipt = None
        self.navdp_live_recording_steps = 0
        self.navdp_live_queue_lengths = None
        self.navdp_live_memory_size = None
        self.terminal_local_latched = False
        self.terminal_stop_streak = 0
        self.loaded_dataset_id = None
        self.loaded_dataset_manifest_sha256 = None
        self.recorded_tail.clear()
        try:
            memnav = _json_object(
                self.session.post(
                    f"{self.config.memnav_url}/navigator_reset",
                    json=memnav_payload,
                    timeout=self.config.timeout,
                ),
                "MemNav reset",
            )
            navdp = _json_object(
                self.session.post(
                    f"{self.config.navdp_url}/navigator_reset",
                    json=navdp_payload,
                    timeout=self.config.timeout,
                ),
                "NavDP reset",
            )
        except Exception as error:
            raise HybridBackendError(
                f"atomic reset failed: {type(error).__name__}: {error}"
            ) from error
        certificate_status = memnav.get("certified_relocalization")
        certificate_enabled = (
            bool(certificate_status.get("enabled"))
            if isinstance(certificate_status, dict)
            else bool(certificate_status)
        )
        monocular_status = memnav.get("monocular_depth")
        monocular_enabled = (
            isinstance(monocular_status, dict)
            and monocular_status.get("enabled") is True
            and monocular_status.get("metric_depth_sensor_consumed") is False
        )
        navdp_monocular = (
            navdp.get("depth_source") == self.config.navdp_depth_source
            and navdp.get("metric_depth_sensor_consumed_by_config") is False
            and navdp.get("monocular_depth_url_configured") is True
            and navdp.get("monocular_depth_transaction_required") is True
        )
        if not certificate_enabled or not monocular_enabled or not navdp_monocular:
            self.native_state_uncertain = True
            raise HybridBackendError(
                "upstream reset did not establish the frozen monocular CEC contract"
            )
        actual_source = (
            certificate_status.get("default_reference_depth_source", "canonical")
            if isinstance(certificate_status, dict) else "canonical")
        if actual_source != self.config.historical_depth_source:
            self.native_state_uncertain = True
            raise HybridBackendError("MemNav historical depth source differs from run config")
        self.initialized = True
        self.phase = PHASE_RECORDING
        dataset_receipt = None
        if self.auto_dataset_id is not None and not resume_empty_auto_dataset:
            if self.dataset_store is None:
                self.native_state_uncertain = True
                raise HybridBackendError(
                    "auto dataset requested but episodic storage is disabled"
                )
            try:
                dataset_receipt = self.dataset_store.start(
                    self.auto_dataset_id,
                    metadata=self.auto_dataset_metadata,
                )
            except (DatasetContractError, OSError) as error:
                self.native_state_uncertain = True
                raise HybridBackendError(
                    "auto dataset start failed after upstream reset: "
                    f"{type(error).__name__}: {error}"
                ) from error
        elif resume_empty_auto_dataset:
            dataset_receipt = self.dataset_store.status()
        return {
            "algo": "cec_hybrid_navdp",
            "protocol_version": PROTOCOL_VERSION,
            "terminal_handoff_schema": TERMINAL_HANDOFF_SCHEMA,
            "query_observation_supported": True,
            "query_observation_count": self.query_observation_count,
            "phase": self.phase,
            "frames_recorded": self.frames_recorded,
            "memnav_algo": memnav.get("algo"),
            "navdp_algo": navdp.get("algo", "navdp"),
            "certificate_enabled": certificate_enabled,
            "navigation_sensor_contract": NAVIGATION_SENSOR_CONTRACT,
            "navdp_depth_source": self.config.navdp_depth_source,
            "metric_depth_sensor_consumed_by_policy": False,
            "client_depth_contract": CLIENT_DEPTH_CONTRACT,
            "camera_height_m": camera_height_m,
            "episodic_dataset": dataset_receipt,
        }

    def _validate_monocular_plan(
        self,
        result: dict[str, Any],
        *,
        image: bytes,
        form: Mapping[str, str],
    ) -> dict[str, Any]:
        receipt = result.get("monocular_depth_receipt")
        if (
            result.get("depth_source") != self.config.navdp_depth_source
            or result.get("metric_depth_sensor_consumed") is not False
            or not isinstance(receipt, dict)
            or receipt.get("image_sha256") != hashlib.sha256(image).hexdigest()
            or receipt.get("monocular_depth_transaction_token")
            != form.get("monocular_depth_transaction_token")
            or str(receipt.get("frame_index"))
            != form.get("monocular_depth_frame_index")
        ):
            self.native_state_uncertain = True
            raise HybridBackendError(
                "NavDP plan did not prove monocular depth consumption; reset is required"
            )
        result.update({
            "navigation_sensor_contract": NAVIGATION_SENSOR_CONTRACT,
            "metric_depth_sensor_consumed_by_policy": False,
            "client_metric_depth_forwarded": False,
        })
        return result

    def _native_plan(
        self,
        image: bytes,
        goal: bytes,
        form: Mapping[str, str],
    ) -> dict[str, Any]:
        try:
            result = _json_object(
                self.session.post(
                    f"{self.config.navdp_url}/imagegoal_step",
                    files={
                        "image": _file("image.jpg", image, "image/jpeg"),
                        "goal": _file("goal.jpg", goal, "image/jpeg"),
                    },
                    data=dict(form),
                    timeout=self.config.timeout,
                ),
                "native NavDP step",
            )
            return self._validate_monocular_plan(
                result, image=image, form=form
            )
        except Exception as error:
            self.native_state_uncertain = True
            raise HybridBackendError(
                "native NavDP state is uncertain; reset is required: "
                f"{type(error).__name__}: {error}"
            ) from error

    def _mixed_plan(
        self,
        image: bytes,
        goal: bytes,
        pointgoal: tuple[float, float],
        form: Mapping[str, str],
    ) -> dict[str, Any]:
        data = dict(form)
        data["goal_data"] = json.dumps({
            "goal_x": [float(pointgoal[0])],
            "goal_y": [float(pointgoal[1])],
        })
        try:
            result = _json_object(
                self.session.post(
                    f"{self.config.navdp_url}/navdp_step_ip_mixgoal",
                    files={
                        "image": _file("image.jpg", image, "image/jpeg"),
                        "image_goal": _file("goal.jpg", goal, "image/jpeg"),
                    },
                    data=data,
                    timeout=self.config.timeout,
                ),
                "mixed NavDP step",
            )
            return self._validate_monocular_plan(
                result, image=image, form=data
            )
        except Exception as error:
            self.native_state_uncertain = True
            raise HybridBackendError(
                "mixed NavDP state is uncertain; reset is required: "
                f"{type(error).__name__}: {error}"
            ) from error

    def _direct_local_pose(self, goal: bytes) -> dict[str, Any]:
        """Query the read-only current-frame PnP expert.

        Failure of this optional local expert does not corrupt the shared
        LingBot stream.  The v2 adapter grants scale-free bearing authority
        only while the proof is present, then returns to long-range CEC or
        native NavDP without treating monocular scale as an arrival signal.
        """

        try:
            return _json_object(
                self.session.post(
                    f"{self.config.memnav_url}/local_pose_query",
                    files={"goal": _file("goal.jpg", goal, "image/jpeg")},
                    timeout=self.config.timeout,
                ),
                "direct current-to-goal PnP",
            )
        except Exception as error:
            return {
                "schema_version": "lingbot_pnp_online_arrival_evidence_v2_20260818",
                "status": "arrival_endpoint_failure",
                "certificate_accepted": False,
                "metric_scale_available": False,
                "predicted_distance_m": None,
                "predicted_relative_xy_m": None,
                "error": f"{type(error).__name__}: {error}",
            }

    def memory_step(
        self,
        image: bytes,
        *,
        materialize_monocular_depth: bool = False,
        source_observation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one causal RGB frame to the shared stream, with no goal."""
        if not self.initialized:
            raise HybridBackendError("router is not initialized")
        if self.native_state_uncertain:
            raise HybridBackendError("native state is uncertain; reset is required")
        if self.memory_degraded:
            raise HybridBackendError(
                "monocular geometry stream is degraded; reset is required"
            )
        if self.phase != PHASE_RECORDING:
            raise ValueError(
                "memory_step is only valid during memory recording; "
                "reset to start a new episode"
            )
        if not image:
            raise ValueError("image is required")
        try:
            payload = _json_object(
                self.session.post(
                    f"{self.config.memnav_url}/memory_step",
                    files={"image": _file("image.jpg", image, "image/jpeg")},
                    data={
                        "materialize_monocular_depth": (
                            "1" if materialize_monocular_depth else "0"
                        )
                    },
                    timeout=self.config.timeout,
                ),
                "memory step",
            )
        except Exception as error:
            self.memory_degraded = True
            raise HybridBackendError(
                "monocular geometry stream update failed; reset is required: "
                f"{type(error).__name__}: {error}"
            ) from error
        self.frames_recorded += 1
        self.recorded_tail.append((self.frames_recorded, image))
        if self.dataset_store is not None:
            try:
                self.dataset_store.append_memory(
                    frame_index=self.frames_recorded - 1,
                    image=image,
                    upstream_sha256=payload.get("image_sha256"),
                    source_observation=source_observation,
                )
            except (DatasetContractError, OSError) as error:
                # MemNav has already advanced.  Losing the exact-byte dataset
                # receipt makes the two states irreconcilable, so do not keep
                # navigating on an unauditable stream.
                self.memory_degraded = True
                raise HybridBackendError(
                    "episodic dataset append failed after memory advance; "
                    f"reset is required: {type(error).__name__}: {error}"
                ) from error
        result = {
            "phase": self.phase,
            "frame_idx": payload.get("frame_idx"),
            "frames_recorded": self.frames_recorded,
        }
        for key in (
            "image_sha256",
            "monocular_depth_transaction_schema",
            "monocular_depth_transaction_token",
            "monocular_depth_frame_index",
            "monocular_depth_png_sha256",
        ):
            if key in payload:
                result[key] = payload[key]
        return result

    def plan_novel_and_record(
        self,
        *,
        image: bytes,
        goal: bytes,
        form: Mapping[str, str] | None = None,
        source_observation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Advance Novel navigation with an optional direct-bearing handoff.

        Both upstreams are stateful, so a partial failure cannot be rolled
        back.  Such a failure latches the existing reset-required state.  A
        successful call proves that MemNav and NavDP consumed exactly one
        shared observation and records NavDP's live FIFO receipt.  The direct
        current-to-goal proof consumes no role label and mutates no stream;
        when absent, this path is byte-for-byte native NavDP control.
        """
        if self.navdp_live_recording_steps != self.frames_recorded:
            self.native_state_uncertain = True
            raise HybridBackendError(
                "Novel navigation must begin at the first recorded frame; "
                "reset is required"
            )
        memory = self.memory_step(
            image,
            materialize_monocular_depth=True,
            source_observation=source_observation,
        )
        navdp_form = _bind_monocular_depth_transaction(
            dict(form or {}), memory, image
        )
        local_evidence = self._direct_local_pose(goal)
        local_decision = decide_local_pose_handoff(
            long_range_available=False,
            evidence=local_evidence,
            local_latched=self.terminal_local_latched,
            stop_streak=self.terminal_stop_streak,
        )
        self.terminal_local_latched = local_decision.local_latched
        self.terminal_stop_streak = local_decision.stop_streak
        if local_decision.disposition == "bearing_local":
            assert local_decision.controller_pointgoal_m is not None
            result = self._mixed_plan(
                image, goal, local_decision.controller_pointgoal_m, navdp_form
            )
            controller = "navdp_image_direct_certified_bearing_mix"
        else:
            # Native still consumes exactly one current observation for
            # native, atomic-turn, hold and STOP dispositions.  The latter
            # three are atomically overridden by the robot-side proof gate.
            result = self._native_plan(image, goal, navdp_form)
            controller = (
                f"terminal_{local_decision.disposition}"
                if local_decision.disposition in {"atomic_turn", "hold", "stop"}
                else "navdp_image_router"
            )
        queue_lengths = result.get("queue_lengths")
        memory_size = result.get("memory_size")
        expected_steps = self.navdp_live_recording_steps + 1
        if (
            not isinstance(memory_size, int)
            or memory_size <= 0
            or not isinstance(queue_lengths, list)
            or queue_lengths != [min(expected_steps, memory_size)]
        ):
            self.native_state_uncertain = True
            raise HybridBackendError(
                "Novel plan omitted a valid live FIFO receipt; reset "
                "is required"
            )
        self.navdp_live_recording_steps = expected_steps
        self.navdp_live_queue_lengths = [int(value) for value in queue_lengths]
        self.navdp_live_memory_size = int(memory_size)
        result.update(local_decision.audit_dict())
        result.update({
            "phase": self.phase,
            "frames_recorded": self.frames_recorded,
            "memory_frame_idx": memory.get("frame_idx"),
            "novel_recording": True,
            "navdp_live_recording_steps": self.navdp_live_recording_steps,
            "cec_takeover": False,
            "cec_reason": "novel_recording_no_long_range_memory",
            "cec_controller": controller,
            "terminal_localization": local_evidence,
        })
        return result

    def goal_candidate(
        self,
        image: bytes,
        *,
        validate_support: bool = False,
        evaluation_depth: bytes | None = None,
        evaluation_depth_scale_m: float | None = None,
    ) -> dict[str, Any]:
        """Register one goal-candidate photo that is NOT appended to memory.

        Mirrors the simulator's outcome-blind goal construction: the revisit
        goal must be captured during the recording walk, excluded from the
        memory stream, and later scored for weak covisibility against the
        recorded history before it may be used as the query goal.
        """
        if not self.initialized:
            raise HybridBackendError("router is not initialized")
        if self.native_state_uncertain:
            raise HybridBackendError("native state is uncertain; reset is required")
        if self.memory_degraded:
            raise HybridBackendError(
                "monocular geometry stream is degraded; reset is required"
            )
        if self.phase != PHASE_RECORDING:
            raise ValueError(
                "goal candidates must be captured during memory recording, "
                "before begin_revisit"
            )
        if not image:
            raise ValueError("image is required")
        if evaluation_depth is not None:
            if (
                evaluation_depth_scale_m is None
                or not math.isfinite(float(evaluation_depth_scale_m))
                or float(evaluation_depth_scale_m) <= 0.0
            ):
                raise ValueError(
                    "evaluation depth requires a finite positive metre scale"
                )
        elif evaluation_depth_scale_m is not None:
            raise ValueError("evaluation depth scale supplied without depth")
        digest = hashlib.sha256(image).hexdigest()
        record = {
            "candidate_id": len(self.goal_candidates),
            "captured_after_frame": self.frames_recorded,
            "sha256": digest,
            "appended_to_memory": False,
            "registered": True,
        }
        if validate_support:
            score = self._score_goal_candidate(dict(record, image=image))
            record["capture_score"] = score
            record["registered"] = (
                score["provisional_band"] == "provisional_weak_covis"
            )
            if not record["registered"]:
                return record
        candidate = dict(
            record,
            image=image,
            evaluation_depth=evaluation_depth,
            evaluation_depth_scale_m=evaluation_depth_scale_m,
        )
        if self.dataset_store is not None:
            try:
                self.dataset_store.append_candidate(
                    record=record,
                    image=image,
                    evaluation_depth=evaluation_depth,
                    evaluation_depth_scale_m=evaluation_depth_scale_m,
                )
            except (DatasetContractError, OSError) as error:
                raise HybridBackendError(
                    "episodic goal-candidate persistence failed; "
                    f"{type(error).__name__}: {error}"
                ) from error
        self.goal_candidates.append(candidate)
        if self.goal_candidate_dir is not None:
            path = (
                f"{self.goal_candidate_dir}/candidate_"
                f"{record['candidate_id']:03d}_{digest[:16]}.jpg"
            )
            with open(path, "wb") as handle:
                handle.write(image)
            record["path"] = path
        return record

    def _score_goal_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        goal = candidate.get("image")
        if not isinstance(goal, bytes) or not goal:
            raise HybridBackendError("goal candidate bytes are unavailable")
        try:
            support = _json_object(
                self.session.post(
                    f"{self.config.memnav_url}/goal_candidate_support",
                    files={"goal": _file("goal.jpg", goal, "image/jpeg")},
                    data={
                        "stride": str(max(1, self.config.goal_score_stride)),
                        "candidate_frame_idx": str(
                            int(candidate["captured_after_frame"])
                        ),
                        "min_frame_gap": str(
                            max(1, self.config.goal_min_frame_gap)
                        ),
                    },
                    timeout=self.config.timeout,
                ),
                "goal-candidate support query",
            )
            if support.get("ok") is not True:
                raise ValueError(str(support.get("reason", "support unavailable")))
        except Exception as error:
            raise HybridBackendError(
                "goal-candidate scoring failed without changing phase: "
                f"{type(error).__name__}: {error}"
            ) from error
        best_cos = float(support["max_cos"])
        if not math.isfinite(best_cos):
            raise HybridBackendError("goal-candidate support returned non-finite cosine")
        best_frame_idx = int(support["argmax_idx"])
        overlap = support.get("geometry", support.get("lightglue"))
        if not isinstance(overlap, dict):
            overlap = {}
        inliers = int(overlap.get("inliers", 0))
        inlier_ratio = float(overlap.get("inlier_ratio") or 0.0)
        if inliers < self.config.goal_min_inliers:
            band = "reject_unsupported"
        elif best_cos > self.config.goal_max_cos:
            band = "reject_near_duplicate"
        else:
            band = "provisional_weak_covis"
        raw_ceiling = support.get("eligible_anchor_ceiling")
        try:
            eligible_anchor_ceiling = int(raw_ceiling)
        except (TypeError, ValueError, OverflowError):
            eligible_anchor_ceiling = None
        if band == "provisional_weak_covis":
            latest_permitted = (
                int(candidate["captured_after_frame"])
                - max(1, self.config.goal_min_frame_gap)
            )
            if (
                eligible_anchor_ceiling is None
                or not 0 <= eligible_anchor_ceiling <= latest_permitted
            ):
                raise HybridBackendError(
                    "goal-candidate support omitted a valid frozen causal "
                    "ceiling inside the capture window"
                )
        return {
            "candidate_id": int(candidate["candidate_id"]),
            "captured_after_frame": int(candidate["captured_after_frame"]),
            "sha256": str(candidate["sha256"]),
            "max_cos": best_cos,
            "argmax_idx": best_frame_idx,
            "frames_swept": int(support.get("frames_swept", 0)),
            "frames_total": int(support.get("frames_total", self.frames_recorded)),
            "candidate_frame_idx": int(support.get(
                "candidate_frame_idx", candidate["captured_after_frame"]
            )),
            "min_frame_gap": int(support.get(
                "min_frame_gap", self.config.goal_min_frame_gap
            )),
            "eligible_anchor_ceiling": eligible_anchor_ceiling,
            "geometry": {
                "anchor": best_frame_idx,
                "matches": overlap.get("matches"),
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
            },
            "geometry_backend": support.get(
                "geometry_backend", "sift_fundamental_ransac"
            ),
            "provisional_band": band,
            "scoring_ms": support.get("scoring_ms"),
            "state_mutated": support.get("state_mutated"),
        }

    def prepare_revisit(
        self, *, query_start_image: bytes | None = None
    ) -> dict[str, Any]:
        """Select, install and freeze one recorded goal before phase switch.

        Scoring is read-only.  If no candidate satisfies the frozen support
        band, the router remains in ``memory_recording`` and no NavDP warm-up
        is attempted.  Once a candidate is selected, ``begin_revisit`` is the
        only stateful operation; its existing fail-closed semantics remain.
        """
        if (
            self.phase == PHASE_REVISIT
            and self.active_goal is not None
            and self.last_prepare_receipt is not None
        ):
            replay = {
                **self.last_prepare_receipt,
                "idempotent_replay": True,
                "goal_image_jpeg_base64": base64.b64encode(
                    self.active_goal["image"]
                ).decode("ascii"),
            }
            depth = self.active_goal.get("evaluation_depth")
            if isinstance(depth, bytes) and depth:
                replay["goal_evaluation_depth_png_base64"] = (
                    base64.b64encode(depth).decode("ascii")
                )
                replay["goal_evaluation_depth_scale_m"] = float(
                    self.active_goal["evaluation_depth_scale_m"]
                )
            return replay
        if self.phase != PHASE_RECORDING:
            raise ValueError(
                "prepare_revisit requires the memory recording phase"
            )
        if not self.goal_candidates:
            raise ValueError("prepare_revisit requires at least one goal candidate")
        scores = [
            self._score_goal_candidate(candidate)
            for candidate in self.goal_candidates
        ]
        eligible = [
            score for score in scores
            if score["provisional_band"] == "provisional_weak_covis"
        ]
        if not eligible:
            raise ValueError(
                "no goal candidate passed the frozen support band; remain in "
                "memory_recording and capture a non-trivial supported view"
            )
        # Deterministic: strongest verified geometry, then inlier ratio and
        # DINO support, then the earliest candidate id.
        selected_score = max(
            eligible,
            key=lambda score: (
                int(score["geometry"]["inliers"]),
                float(score["geometry"]["inlier_ratio"]),
                float(score["max_cos"]),
                -int(score["candidate_id"]),
            ),
        )
        selected = self.goal_candidates[int(selected_score["candidate_id"])]
        switch = self.begin_revisit(query_start_image=query_start_image)
        # The support query was frozen at candidate capture.  Carry that exact
        # upper bound into every online retrieval for this automatically
        # selected goal; otherwise a later near-adjacent history frame could
        # become a trivial anchor after the phase switch.
        candidate_ceiling_override = int(
            selected_score["eligible_anchor_ceiling"]
        )
        self.active_goal = dict(
            selected,
            candidate_ceiling_override=candidate_ceiling_override,
        )
        selected_score = dict(
            selected_score,
            candidate_ceiling_override=candidate_ceiling_override,
        )
        receipt = {
            **switch,
            "goal_selection_contract": "weak_covis_geometry_first_v1",
            "goal_score_stride": max(1, self.config.goal_score_stride),
            "goal_min_frame_gap": max(1, self.config.goal_min_frame_gap),
            "goal_min_inliers": self.config.goal_min_inliers,
            "goal_max_cos": self.config.goal_max_cos,
            "selected_goal": selected_score,
            "candidate_scores": scores,
            "goal_image_jpeg_base64": base64.b64encode(selected["image"]).decode(
                "ascii"
            ),
            "idempotent_replay": False,
        }
        evaluation_depth = selected.get("evaluation_depth")
        if isinstance(evaluation_depth, bytes) and evaluation_depth:
            receipt["goal_evaluation_depth_png_base64"] = base64.b64encode(
                evaluation_depth
            ).decode("ascii")
            receipt["goal_evaluation_depth_scale_m"] = float(
                selected["evaluation_depth_scale_m"]
            )
        # Keep a status-safe copy; the image bytes remain available via the
        # active goal and are not duplicated in /healthz.
        self.last_prepare_receipt = {
            key: value
            for key, value in receipt.items()
            if key not in {
                "goal_image_jpeg_base64",
                "goal_evaluation_depth_png_base64",
            }
        }
        return receipt

    def prepare_revisit_goal(
        self,
        goal: bytes,
        *,
        query_start_image: bytes | None = None,
    ) -> dict[str, Any]:
        """Atomically install a pre-episode frozen goal and start Revisit.

        Real-world experiments often freeze the Revisit target before motion,
        just as a simulator manifest freezes every query goal.  Such a target
        must not be registered through ``goal_candidate`` at the end of the
        Novel traversal: doing so would forge its causal capture time.  This
        endpoint records the truthful provenance while leaving the actual
        takeover decision to the ordinary per-query CEC certificate.
        """
        if not goal:
            raise ValueError("goal is required")
        digest = hashlib.sha256(goal).hexdigest()
        if self.phase == PHASE_REVISIT:
            if (
                self.active_goal is not None
                and self.last_prepare_receipt is not None
                and self.active_goal.get("goal_source")
                == "operator_frozen_external"
                and self.active_goal.get("sha256") == digest
            ):
                return {
                    **self.last_prepare_receipt,
                    "idempotent_replay": True,
                    "goal_image_jpeg_base64": base64.b64encode(
                        self.active_goal["image"]
                    ).decode("ascii"),
                }
            raise ValueError(
                "revisit already started with a different committed goal"
            )
        if self.phase != PHASE_RECORDING:
            raise ValueError(
                "prepare_revisit_goal requires the memory recording phase"
            )

        switch = self.begin_revisit(query_start_image=query_start_image)
        selected_goal = {
            "candidate_id": None,
            "captured_after_frame": None,
            "sha256": digest,
            "appended_to_memory": False,
            "goal_source": "operator_frozen_external",
        }
        self.active_goal = dict(selected_goal, image=goal)
        receipt = {
            **switch,
            "goal_selection_contract": "operator_frozen_external_v1",
            "selected_goal": selected_goal,
            "candidate_scores": [],
            "goal_image_jpeg_base64": base64.b64encode(goal).decode("ascii"),
            "idempotent_replay": False,
        }
        self.last_prepare_receipt = {
            key: value
            for key, value in receipt.items()
            if key != "goal_image_jpeg_base64"
        }
        return receipt

    def begin_revisit(
        self, *, query_start_image: bytes | None = None
    ) -> dict[str, Any]:
        """Switch to the query phase; the next goal query freezes the session."""
        if not self.initialized:
            raise HybridBackendError("router is not initialized")
        if self.native_state_uncertain:
            raise HybridBackendError("native state is uncertain; reset is required")
        if self.memory_degraded:
            raise HybridBackendError(
                "monocular geometry stream is degraded; reset is required"
            )
        if self.phase != PHASE_RECORDING:
            raise ValueError(
                "begin_revisit requires the memory recording phase; "
                "reset to start a new episode"
            )
        if self.frames_recorded < 1:
            raise ValueError(
                "begin_revisit requires at least one recorded memory frame"
            )
        warmup_frame_indices: list[int | str]
        if self.loaded_dataset_id is not None:
            # A separately recorded survey is long-term memory, not the
            # current NavDP observation FIFO.  Prime the frozen controller
            # with the physical query-start view instead of pretending that
            # the survey's last frame happened one control tick ago.
            if not query_start_image:
                raise ValueError(
                    "a loaded dataset requires the current query-start RGB"
                )
            try:
                payload = _json_object(
                    self.session.post(
                        f"{self.config.navdp_url}/memory_replay_step",
                        files={
                            "image": _file(
                                "query_start.jpg", query_start_image, "image/jpeg"
                            )
                        },
                        timeout=self.config.timeout,
                    ),
                    "NavDP independent-query warm-up",
                )
            except Exception as error:
                self.native_state_uncertain = True
                raise HybridBackendError(
                    "NavDP independent-query warm-up failed; reset is required: "
                    f"{type(error).__name__}: {error}"
                ) from error
            warmup: list[tuple[int, bytes]] = []
            queue_lengths = payload.get("queue_lengths")
            memory_size = payload.get("memory_size")
            warmup_mode = "independent_formal_query_start"
            warmup_count = 1
            warmup_frame_indices = ["query_start_current"]
        elif self.navdp_live_recording_steps:
            if self.navdp_live_recording_steps != self.frames_recorded:
                self.native_state_uncertain = True
                raise HybridBackendError(
                    "live Novel/NavDP and causal-memory frame counts diverged; "
                    "reset is required"
                )
            warmup: list[tuple[int, bytes]] = []
            queue_lengths = self.navdp_live_queue_lengths
            memory_size = self.navdp_live_memory_size
            warmup_mode = "live_novel_fifo"
            warmup_count = 0
            warmup_frame_indices = []
        else:
            # A teleoperated prefix never advanced NavDP. Reconstruct its FIFO
            # from the same strided tail used by the simulator replay.
            warmup = select_warmup_frames(
                list(self.recorded_tail),
                NAVDP_WARMUP_STRIDE,
                NAVDP_WARMUP_MAX_FRAMES,
            )
            queue_lengths = None
            memory_size = None
            for _, frame in warmup:
                try:
                    payload = _json_object(
                        self.session.post(
                            f"{self.config.navdp_url}/memory_replay_step",
                            files={"image": _file("image.jpg", frame, "image/jpeg")},
                            timeout=self.config.timeout,
                        ),
                        "NavDP warm-up replay",
                    )
                except Exception as error:
                    self.native_state_uncertain = True
                    raise HybridBackendError(
                        "NavDP observation warm-up failed; reset is required: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                queue_lengths = payload.get("queue_lengths")
                memory_size = payload.get("memory_size")
            warmup_mode = "replayed_teleop_prefix"
            warmup_count = len(warmup)
            warmup_frame_indices = [index for index, _ in warmup]
        if not isinstance(memory_size, int) or memory_size <= 0:
            self.native_state_uncertain = True
            raise HybridBackendError(
                "NavDP warm-up omitted memory size; reset is required"
            )
        expected_queue_length = min(
            self.navdp_live_recording_steps or warmup_count, memory_size
        )
        if queue_lengths != [expected_queue_length]:
            self.native_state_uncertain = True
            raise HybridBackendError(
                "NavDP warm-up queue length mismatch; reset is required"
            )
        self.phase = PHASE_REVISIT
        self.revisit_started_after_frame = self.frames_recorded
        return {
            "phase": self.phase,
            "frames_recorded": self.frames_recorded,
            "revisit_started_after_frame": self.revisit_started_after_frame,
            "goal_session_contract": "first_goal_query_after_begin_revisit",
            "navdp_warmup_mode": warmup_mode,
            "navdp_warmup_frames": warmup_count,
            "navdp_warmup_frame_indices": warmup_frame_indices,
            "navdp_queue_lengths": queue_lengths,
            "navdp_memory_size": memory_size,
            "loaded_dataset_id": self.loaded_dataset_id,
            "loaded_dataset_manifest_sha256": (
                self.loaded_dataset_manifest_sha256
            ),
        }

    def start_dataset(
        self,
        dataset_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.dataset_store is None:
            raise ValueError("episodic dataset storage is disabled")
        if not self.initialized or self.phase != PHASE_RECORDING:
            raise ValueError("dataset recording requires initialized memory_recording")
        if self.frames_recorded or self.goal_candidates:
            raise ValueError("dataset recording must start before the first frame")
        return self.dataset_store.start(dataset_id, metadata=metadata)

    def dataset_status(self, *, include_sealed: bool = True) -> dict[str, Any]:
        if self.dataset_store is None:
            return {"enabled": False, "sealed_datasets": []}
        status = {
            "enabled": True,
            **self.dataset_store.status(),
            "loaded_dataset_id": self.loaded_dataset_id,
            "loaded_dataset_manifest_sha256": (
                self.loaded_dataset_manifest_sha256
            ),
        }
        if include_sealed:
            status["sealed_datasets"] = self.dataset_store.list_sealed()
        return status

    def seal_dataset(self) -> dict[str, Any]:
        if self.dataset_store is None:
            raise ValueError("episodic dataset storage is disabled")
        if self.phase != PHASE_RECORDING:
            raise ValueError("dataset can only be sealed during memory_recording")
        return self.dataset_store.seal(protocol={
            "cec_protocol_version": PROTOCOL_VERSION,
            "navigation_sensor_contract": NAVIGATION_SENSOR_CONTRACT,
            "goal_min_frame_gap": self.config.goal_min_frame_gap,
            "goal_min_inliers": self.config.goal_min_inliers,
            "goal_max_cos": self.config.goal_max_cos,
            "metric_depth_sensor_consumed_by_policy": False,
        })

    def load_dataset(self, dataset_id: str) -> dict[str, Any]:
        if self.dataset_store is None:
            raise ValueError("episodic dataset storage is disabled")
        if self.dataset_store.recording:
            raise ValueError("seal the active dataset before loading another")
        if not self.initialized or self.phase != PHASE_RECORDING:
            raise ValueError("dataset loading requires initialized memory_recording")
        if self.frames_recorded or self.goal_candidates:
            raise ValueError("dataset loading requires a freshly reset empty stream")
        loaded = self.dataset_store.load(dataset_id)
        manifest_raw = (loaded.root / "manifest.json").read_bytes()
        for expected, image in loaded.memory_frames():
            receipt = self.memory_step(image)
            if int(receipt.get("frame_idx", -1)) != int(expected["frame_index"]):
                self.memory_degraded = True
                raise HybridBackendError(
                    "dataset replay frame identity diverged; reset is required"
                )
        restored: list[dict[str, Any]] = []
        for record, image, evaluation_depth in loaded.goal_candidates():
            restored.append(dict(
                record,
                image=image,
                evaluation_depth=evaluation_depth,
            ))
        self.goal_candidates = restored
        self.loaded_dataset_id = str(loaded.manifest["dataset_id"])
        self.loaded_dataset_manifest_sha256 = hashlib.sha256(
            manifest_raw
        ).hexdigest()
        return {
            "dataset_id": self.loaded_dataset_id,
            "manifest_sha256": self.loaded_dataset_manifest_sha256,
            "frames_replayed": self.frames_recorded,
            "goal_candidates_restored": len(self.goal_candidates),
            "phase": self.phase,
            "navdp_fifo_replayed_from_dataset": False,
            "query_start_rgb_required": True,
        }

    def plan_imagegoal(
        self,
        *,
        image: bytes,
        goal: bytes,
        depth: bytes | None = None,
        form: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self.initialized:
            raise HybridBackendError("router is not initialized")
        if self.native_state_uncertain:
            raise HybridBackendError("native state is uncertain; reset is required")
        if self.phase != PHASE_REVISIT:
            raise ValueError(
                "goal queries are forbidden during memory recording: the first "
                "goal query freezes the MemNav goal session and candidate "
                "ceiling, so call /begin_revisit at the query start point first"
            )
        if not image or not goal:
            raise ValueError("image and goal are required")
        form = dict(form or {})
        if self.active_goal is not None:
            expected_sha = str(self.active_goal["sha256"])
            if form.get("installed_goal_sha256") != expected_sha:
                raise ValueError(
                    "client has not acknowledged the atomically selected goal"
                )
            # The hub owns the selected bytes.  The client's goal upload is a
            # compatibility field and cannot replace the committed target.
            goal = self.active_goal["image"]
        self.step_index += 1

        # In the monocular system the same causal LingBot stream provides both
        # CEC proof and current-frame depth.  Missing a stream update therefore
        # cannot be disguised as an exact native fallback: fail closed and let
        # the robot-side stale-plan/watchdog layers stop motion.
        if self.memory_degraded:
            raise HybridBackendError(
                "monocular geometry stream is degraded; reset is required"
            )

        try:
            probe_form = dict(form)
            probe_form["materialize_monocular_depth"] = "1"
            candidate_ceiling_override = (
                None
                if self.active_goal is None
                else self.active_goal.get("candidate_ceiling_override")
            )
            if candidate_ceiling_override is not None:
                probe_form["candidate_ceiling_override"] = str(
                    int(candidate_ceiling_override)
                )
            probe = _json_object(
                self.session.post(
                    f"{self.config.memnav_url}/retrieval_probe_step",
                    files={
                        "image": _file("image.jpg", image, "image/jpeg"),
                        "goal": _file("goal.jpg", goal, "image/jpeg"),
                    },
                    data=probe_form,
                    timeout=self.config.timeout,
                ),
                "CEC retrieval probe",
            )
        except Exception as error:
            self.memory_degraded = True
            raise HybridBackendError(
                "monocular geometry stream update failed; reset is required: "
                f"{type(error).__name__}: {error}"
            ) from error

        navdp_form = _bind_monocular_depth_transaction(form, probe, image)
        if self.config.authority_mode == "native":
            # The paired real-robot baseline keeps the same causal monocular
            # stream and exact goal/history bytes, but grants no memory or
            # direct-pose authority.  The retrieval probe is retained because
            # it is the single append transaction that materializes current
            # LingBot depth; its proposals are deliberately not consumed.
            certificate = {
                "ok": False,
                "accepted": False,
                "reason": "authority_disabled_formal_native_arm",
            }
            decision = adapt_revisit_pointgoal(
                mode="verified_bearing_v1",
                router_active=False,
                pointgoal=None,
                source="authority_disabled_formal_native_arm",
                pointgoal_units=POINTGOAL_UNITS,
            )
            local_evidence = {
                "status": "authority_disabled_formal_native_arm",
                "certificate_accepted": False,
            }
            if self.config.terminal_approach == "height_scaled_local":
                # A new paired experiment may share this terminal adapter
                # across both arms. This is native+terminal, not the old
                # untouched-native baseline; record that distinction.
                local_evidence = self._direct_local_pose(goal)
            local_decision = decide_local_pose_handoff(
                long_range_available=False,
                evidence=local_evidence,
                local_latched=False,
                stop_streak=0,
                metric_approach=self.config.terminal_approach == "height_scaled_local",
                expected_frame_index=probe.get("frame_idx"),
            )
            if not local_decision.metric_approach_active:
                # Do not leak the direct expert's long-range bearing into the
                # memory-disabled comparator.
                local_decision = decide_local_pose_handoff(
                    long_range_available=False, evidence=None)
            self.terminal_local_latched = False
            self.terminal_stop_streak = 0
            if local_decision.metric_approach_active and local_decision.disposition == "bearing_local":
                result = self._mixed_plan(image, goal, local_decision.controller_pointgoal_m, navdp_form)
            else:
                result = self._native_plan(image, goal, navdp_form)
            result.update(decision.audit_dict())
            result.update(local_decision.audit_dict())
            result.update({
                "cec_authority_mode": self.config.authority_mode,
                "terminal_approach_mode": self.config.terminal_approach,
                "baseline_includes_shared_terminal_adapter": self.config.terminal_approach == "height_scaled_local",
                "cec_takeover": False,
                "cec_reason": certificate["reason"],
                "cec_controller": ("native_with_shared_local_approach"
                                   if local_decision.metric_approach_active
                                   else "navdp_image_authority_disabled"),
                "cec_step_index": self.step_index,
                "cec_frame_idx": probe.get("frame_idx"),
                "cec_selected_anchor": None,
                "cec_certificate": None,
                "cec_retrieval_trace": (
                    _retrieval_trace(probe) if self.step_index == 1 else None
                ),
                "cec_relocalization_trace": None,
                "cec_relocalization_ms": None,
                "cec_retrieval_probe_timing": probe.get(
                    "retrieval_probe_timing"
                ),
                "cec_add_frame_runtime_ms": probe.get(
                    "add_frame_runtime_ms"
                ),
                "cec_append_request_runtime_ms": probe.get(
                    "append_request_runtime_ms"
                ),
                "cec_monocular_depth_cache_hit": probe.get(
                    "monocular_depth_cache_hit"
                ),
                "cec_monocular_depth_prediction_runtime_ms": probe.get(
                    "monocular_depth_prediction_runtime_ms"
                ),
                "cec_monocular_depth_materialization_runtime_ms": probe.get(
                    "monocular_depth_materialization_runtime_ms"
                ),
                "cec_candidate_ceiling_override": (
                    None
                    if self.active_goal is None
                    else self.active_goal.get("candidate_ceiling_override")
                ),
                "terminal_localization": local_evidence,
            })
            return result

        candidates = probe.get("certified_visual_candidates")
        if not isinstance(candidates, list):
            candidates = []
        try:
            certificate = _json_object(
                self.session.post(
                    f"{self.config.memnav_url}/certified_relocalize",
                    files={"goal": _file("goal.jpg", goal, "image/jpeg")},
                    data={
                        "candidates": json.dumps(candidates),
                        "proposal_order": PROPOSAL_ORDER,
                        "graph_rescue": "0",
                        "learned_rescue": "0",
                    },
                    timeout=self.config.timeout,
                ),
                "CEC certificate",
            )
        except Exception as error:
            certificate = {
                "ok": False,
                "accepted": False,
                "reason": "certificate_endpoint_failure",
                "error": f"{type(error).__name__}: {error}",
            }

        active = bool(certificate.get("ok") is True and certificate.get("accepted") is True)
        units = certificate.get("pointgoal_units")
        if active and units != POINTGOAL_UNITS:
            active = False
            certificate["reason"] = "invalid_scale_free_output_contract"
        decision = adapt_revisit_pointgoal(
            mode="verified_bearing_v1",
            router_active=active,
            pointgoal=certificate.get("aux_pose"),
            source="lightglue_lingbot_pnp_v2_scale_free",
            pointgoal_units=POINTGOAL_UNITS,
        )

        # Long-range CEC answers content addressing; direct current->goal PnP
        # refines the local scale-free bearing.  The direct proof is queried
        # independently of semantic role and therefore also works when the
        # long-range certificate is absent.  Its monocular translation norm
        # has no metric-control or STOP authority; proof loss returns to the
        # preceding route.
        local_evidence = self._direct_local_pose(goal)
        local_decision = decide_local_pose_handoff(
            long_range_available=active,
            evidence=local_evidence,
            local_latched=self.terminal_local_latched,
            stop_streak=self.terminal_stop_streak,
            metric_approach=self.config.terminal_approach == "height_scaled_local",
            expected_frame_index=probe.get("frame_idx"),
        )
        self.terminal_local_latched = local_decision.local_latched
        self.terminal_stop_streak = local_decision.stop_streak

        if local_decision.disposition == "bearing_local":
            assert local_decision.controller_pointgoal_m is not None
            result = self._mixed_plan(
                image, goal, local_decision.controller_pointgoal_m, navdp_form
            )
            controller = "navdp_image_direct_certified_bearing_mix"
        elif local_decision.disposition in {"atomic_turn", "hold", "stop"}:
            # NavDP still consumes exactly one current observation so its FIFO
            # remains causal.  The robot adapter atomically overrides this
            # proposal with the audited turn/hold/STOP disposition.
            result = self._native_plan(image, goal, navdp_form)
            controller = f"terminal_{local_decision.disposition}"
        elif decision.takeover:
            assert decision.controller_pointgoal is not None
            result = self._mixed_plan(
                image, goal, decision.controller_pointgoal, navdp_form
            )
            controller = "navdp_image_point_mix"
        else:
            result = self._native_plan(image, goal, navdp_form)
            controller = "navdp_image_router"
        result.update(decision.audit_dict())
        result.update(local_decision.audit_dict())
        result.update({
            "cec_authority_mode": self.config.authority_mode,
            "terminal_approach_mode": self.config.terminal_approach,
            "cec_takeover": decision.takeover,
            "cec_reason": certificate.get("reason", decision.reason),
            "cec_controller": controller,
            "cec_step_index": self.step_index,
            "cec_frame_idx": probe.get("frame_idx"),
            "cec_selected_anchor": certificate.get("selected_anchor"),
            "cec_certificate": certificate.get("certificate"),
            "cec_retrieval_trace": (
                _retrieval_trace(probe) if self.step_index == 1 else None
            ),
            "cec_relocalization_trace": (
                _relocalization_trace(certificate)
                if self.step_index == 1 else None
            ),
            "cec_relocalization_ms": certificate.get("relocalization_ms"),
            "cec_retrieval_probe_timing": probe.get(
                "retrieval_probe_timing"
            ),
            "cec_add_frame_runtime_ms": probe.get("add_frame_runtime_ms"),
            "cec_append_request_runtime_ms": probe.get(
                "append_request_runtime_ms"
            ),
            "cec_monocular_depth_cache_hit": probe.get(
                "monocular_depth_cache_hit"
            ),
            "cec_monocular_depth_prediction_runtime_ms": probe.get(
                "monocular_depth_prediction_runtime_ms"
            ),
            "cec_monocular_depth_materialization_runtime_ms": probe.get(
                "monocular_depth_materialization_runtime_ms"
            ),
            "cec_candidate_ceiling_override": (
                None
                if self.active_goal is None
                else self.active_goal.get("candidate_ceiling_override")
            ),
            "terminal_localization": local_evidence,
        })
        if certificate.get("error"):
            result["cec_error"] = certificate["error"]
        return result


def create_app(router: CecHybridRouter) -> Flask:
    app = Flask(__name__)
    call_lock = threading.Lock()

    @app.get("/healthz")
    def healthz():
        return jsonify({
            "query_observation_supported": True,
            "query_observation_count": router.query_observation_count,
            "ok": True,
            "algo": "cec_hybrid_navdp",
            "protocol_version": PROTOCOL_VERSION,
            "terminal_handoff_schema": TERMINAL_HANDOFF_SCHEMA,
            "initialized": router.initialized,
            "phase": router.phase,
            "frames_recorded": router.frames_recorded,
            "revisit_started_after_frame": router.revisit_started_after_frame,
            "novel_recording_supported": True,
            "navdp_live_recording_steps": router.navdp_live_recording_steps,
            "navdp_live_queue_lengths": router.navdp_live_queue_lengths,
            "goal_candidates_captured": len(router.goal_candidates),
            "active_goal_id": (
                None if router.active_goal is None
                else router.active_goal.get("candidate_id")
            ),
            "active_goal_sha256": (
                None if router.active_goal is None
                else router.active_goal.get("sha256")
            ),
            "last_prepare_receipt": router.last_prepare_receipt,
            "episodic_dataset": router.dataset_status(include_sealed=False),
            "terminal_local_latched": router.terminal_local_latched,
            "terminal_stop_streak": router.terminal_stop_streak,
            "memory_degraded": router.memory_degraded,
            "native_state_uncertain": router.native_state_uncertain,
            "navigation_sensor_contract": NAVIGATION_SENSOR_CONTRACT,
            "navdp_depth_source": router.config.navdp_depth_source,
            "metric_depth_sensor_consumed_by_policy": False,
            "client_depth_contract": CLIENT_DEPTH_CONTRACT,
            "camera_height_m": float(router.config.camera_height_m),
            "cec_authority_mode": router.config.authority_mode,
            "cec_historical_depth_source": router.config.historical_depth_source,
            "terminal_approach_mode": router.config.terminal_approach,
        })

    @app.post("/navigator_reset")
    def navigator_reset():
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(router.reset(request.get_json(silent=True) or {}))
            except ValueError as error:
                return jsonify({"error": str(error)}), 400
            except HybridBackendError as error:
                return jsonify({"error": str(error), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/query_observation_step")
    def query_observation_step():
        if "image" not in request.files:
            return jsonify({"error": "missing files: image"}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(router.query_observation_step(
                    request.files["image"].read(),
                    request.form.get("installed_goal_sha256", "")))
            except ValueError as error:
                return jsonify({"error": str(error)}), 400
            except HybridBackendError as error:
                return jsonify({"error": str(error), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/memory_step")
    def memory_step():
        if "image" not in request.files:
            return jsonify({"error": "missing files: image"}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                source_observation = None
                raw_source_observation = request.form.get("source_observation")
                if raw_source_observation is not None:
                    source_observation = json.loads(raw_source_observation)
                    if not isinstance(source_observation, dict):
                        raise ValueError("source_observation must be an object")
                return jsonify(
                    router.memory_step(
                        request.files["image"].read(),
                        source_observation=source_observation,
                    )
                )
            except json.JSONDecodeError as error:
                return jsonify({
                    "error": f"source_observation is invalid JSON: {error}"
                }), 400
            except ValueError as error:
                return jsonify({"error": str(error)}), 400
            except HybridBackendError as error:
                return jsonify({"error": str(error), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/novel_imagegoal_step")
    def novel_imagegoal_step():
        missing = [name for name in ("image", "goal") if name not in request.files]
        if missing:
            return jsonify({"error": f"missing files: {', '.join(missing)}"}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                form = request.form.to_dict(flat=True)
                source_observation = None
                raw_source_observation = form.pop("source_observation", None)
                if raw_source_observation is not None:
                    source_observation = json.loads(raw_source_observation)
                    if not isinstance(source_observation, dict):
                        raise ValueError("source_observation must be an object")
                result = router.plan_novel_and_record(
                    image=request.files["image"].read(),
                    goal=request.files["goal"].read(),
                    form=form,
                    source_observation=source_observation,
                )
                app.logger.info(
                    "novel_recording_plan frames=%s queue=%s",
                    result.get("frames_recorded"),
                    result.get("queue_lengths"),
                )
                return jsonify(result)
            except json.JSONDecodeError as error:
                return jsonify({
                    "error": f"source_observation is invalid JSON: {error}",
                    "phase": router.phase,
                }), 400
            except ValueError as error:
                return jsonify({"error": str(error), "phase": router.phase}), 400
            except HybridBackendError as error:
                return jsonify({"error": str(error), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/goal_candidate")
    def goal_candidate():
        if "image" not in request.files:
            return jsonify({"error": "missing files: image"}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(
                    router.goal_candidate(
                        request.files["image"].read(),
                        validate_support=(
                            request.form.get("validate_support", "0") == "1"
                        ),
                        evaluation_depth=(
                            request.files["evaluation_depth"].read()
                            if "evaluation_depth" in request.files else None
                        ),
                        evaluation_depth_scale_m=(
                            float(request.form["evaluation_depth_scale_m"])
                            if "evaluation_depth_scale_m" in request.form
                            else None
                        ),
                    )
                )
            except ValueError as error:
                return jsonify({"error": str(error)}), 400
            except HybridBackendError as error:
                return jsonify({"error": str(error), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/begin_revisit")
    def begin_revisit():
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                result = router.begin_revisit(
                    query_start_image=(
                        request.files["query_start"].read()
                        if "query_start" in request.files else None
                    )
                )
                app.logger.info(
                    "cec_begin_revisit frames_recorded=%s",
                    result.get("frames_recorded"),
                )
                return jsonify(result)
            except ValueError as error:
                return jsonify({"error": str(error)}), 400
            except HybridBackendError as error:
                return jsonify({"error": str(error), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/prepare_revisit")
    def prepare_revisit():
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                result = router.prepare_revisit(
                    query_start_image=(
                        request.files["query_start"].read()
                        if "query_start" in request.files else None
                    )
                )
                app.logger.info(
                    "cec_prepare_revisit goal=%s frames=%s warmup=%s",
                    result.get("selected_goal", {}).get("candidate_id"),
                    result.get("frames_recorded"),
                    result.get("navdp_warmup_frames"),
                )
                return jsonify(result)
            except ValueError as error:
                return jsonify({"error": str(error), "phase": router.phase}), 400
            except HybridBackendError as error:
                return jsonify({
                    "error": str(error),
                    "reset_required": router.native_state_uncertain,
                    "phase": router.phase,
                }), 503
        finally:
            call_lock.release()

    @app.post("/prepare_revisit_goal")
    def prepare_revisit_goal():
        if "goal" not in request.files:
            return jsonify({"error": "missing files: goal"}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                result = router.prepare_revisit_goal(
                    request.files["goal"].read(),
                    query_start_image=(
                        request.files["query_start"].read()
                        if "query_start" in request.files else None
                    ),
                )
                app.logger.info(
                    "cec_prepare_external_revisit goal_sha=%s frames=%s "
                    "warmup=%s",
                    result.get("selected_goal", {}).get("sha256"),
                    result.get("frames_recorded"),
                    result.get("navdp_warmup_frames"),
                )
                return jsonify(result)
            except ValueError as error:
                return jsonify({"error": str(error), "phase": router.phase}), 400
            except HybridBackendError as error:
                return jsonify({"error": str(error), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.get("/dataset/status")
    def dataset_status():
        try:
            return jsonify(router.dataset_status())
        except (DatasetContractError, OSError) as error:
            return jsonify({"error": str(error)}), 500

    @app.post("/dataset/start")
    def dataset_start():
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            payload = request.get_json(silent=True) or {}
            metadata = payload.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                return jsonify({"error": "metadata must be an object"}), 400
            try:
                return jsonify(router.start_dataset(
                    str(payload.get("dataset_id", "")),
                    metadata=metadata,
                ))
            except (ValueError, DatasetContractError) as error:
                return jsonify({"error": str(error)}), 400
        finally:
            call_lock.release()

    @app.post("/dataset/seal")
    def dataset_seal():
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(router.seal_dataset())
            except (ValueError, DatasetContractError, OSError) as error:
                return jsonify({"error": str(error)}), 400
        finally:
            call_lock.release()

    @app.post("/dataset/load")
    def dataset_load():
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            payload = request.get_json(silent=True) or {}
            try:
                return jsonify(router.load_dataset(
                    str(payload.get("dataset_id", ""))
                ))
            except (ValueError, DatasetContractError, OSError) as error:
                return jsonify({"error": str(error)}), 400
            except HybridBackendError as error:
                return jsonify({
                    "error": str(error),
                    "reset_required": True,
                }), 503
        finally:
            call_lock.release()

    @app.post("/imagegoal_step")
    def imagegoal_step():
        missing = [name for name in ("image", "goal") if name not in request.files]
        if missing:
            return jsonify({"error": f"missing files: {', '.join(missing)}"}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                result = router.plan_imagegoal(
                    image=request.files["image"].read(),
                    goal=request.files["goal"].read(),
                    depth=(
                        request.files["depth"].read()
                        if "depth" in request.files else None
                    ),
                    form=request.form.to_dict(flat=True),
                )
                app.logger.info(
                    "cec_plan step=%s takeover=%s controller=%s reason=%s "
                    "frame=%s anchor=%s relocalization_ms=%s",
                    result.get("cec_step_index"),
                    result.get("cec_takeover"),
                    result.get("cec_controller"),
                    result.get("cec_reason"),
                    result.get("cec_frame_idx"),
                    result.get("cec_selected_anchor"),
                    result.get("cec_relocalization_ms"),
                )
                return jsonify(result)
            except ValueError as error:
                return jsonify({"error": str(error)}), 400
            except HybridBackendError as error:
                return jsonify({"error": str(error), "reset_required": True}), 503
        finally:
            call_lock.release()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18889)
    parser.add_argument("--memnav-url", default="http://127.0.0.1:18888")
    parser.add_argument("--navdp-url", default="http://127.0.0.1:8888")
    parser.add_argument("--connect-timeout-s", type=float, default=3.0)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--camera-height-m", type=float, required=True)
    parser.add_argument("--terminal-approach", choices=["bearing_only", "height_scaled_local"],
                        default="bearing_only",
                        help="opt-in local approach; use identically in both paired arms")
    parser.add_argument(
        "--historical-depth-source", choices=["canonical", "online_history"],
        default="canonical", help="must match the reset-bound MemNav depth source")
    parser.add_argument(
        "--authority-mode",
        choices=AUTHORITY_MODES,
        default="cec",
        help=("cec enables certified long/local bearing authority; native disables "
              "historical guidance. An explicit height_scaled_local adapter is "
              "shared by both arms, not the old untouched-native baseline"),
    )
    parser.add_argument(
        "--goal-candidate-dir", default=None,
        help=("directory for goal-candidate photos captured during memory "
              "recording; they are never appended to the memory stream"))
    parser.add_argument("--goal-score-stride", type=int, default=GOAL_SCORE_STRIDE)
    parser.add_argument(
        "--goal-min-frame-gap", type=int, default=GOAL_MIN_FRAME_GAP
    )
    parser.add_argument("--goal-min-inliers", type=int, default=GOAL_MIN_INLIERS)
    parser.add_argument("--goal-max-cos", type=float, default=GOAL_MAX_COS)
    parser.add_argument(
        "--episodic-dataset-root",
        default=None,
        help="root for immutable two-pass real-world Revisit datasets",
    )
    parser.add_argument(
        "--episodic-dataset-min-frames",
        type=int,
        default=40,
        help="minimum exact RGB frames required before a survey can be sealed",
    )
    parser.add_argument(
        "--auto-dataset-id",
        default=None,
        help="open this dataset atomically with the first navigator reset",
    )
    parser.add_argument(
        "--auto-dataset-metadata-json",
        default="{}",
        help="JSON object recorded in an automatically opened dataset",
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("real-world hub must bind to loopback; use an SSH tunnel")
    dataset_store = (
        None
        if args.episodic_dataset_root is None
        else EpisodicDatasetStore(
            args.episodic_dataset_root,
            minimum_frames=max(1, args.episodic_dataset_min_frames),
        )
    )
    try:
        auto_dataset_metadata = json.loads(args.auto_dataset_metadata_json)
    except json.JSONDecodeError as error:
        parser.error(f"invalid --auto-dataset-metadata-json: {error}")
    if not isinstance(auto_dataset_metadata, dict):
        parser.error("--auto-dataset-metadata-json must contain an object")
    router = CecHybridRouter(UpstreamConfig(
        memnav_url=args.memnav_url.rstrip("/"),
        navdp_url=args.navdp_url.rstrip("/"),
        connect_timeout_s=args.connect_timeout_s,
        request_timeout_s=args.request_timeout_s,
        camera_height_m=args.camera_height_m,
        goal_score_stride=max(1, args.goal_score_stride),
        goal_min_frame_gap=max(1, args.goal_min_frame_gap),
        goal_min_inliers=max(1, args.goal_min_inliers),
        goal_max_cos=float(args.goal_max_cos),
        authority_mode=args.authority_mode,
        historical_depth_source=args.historical_depth_source,
        terminal_approach=args.terminal_approach,
    ), dataset_store=dataset_store,
       auto_dataset_id=args.auto_dataset_id,
       auto_dataset_metadata=auto_dataset_metadata)
    if args.goal_candidate_dir:
        os.makedirs(args.goal_candidate_dir, exist_ok=True)
        router.goal_candidate_dir = args.goal_candidate_dir
    create_app(router).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
