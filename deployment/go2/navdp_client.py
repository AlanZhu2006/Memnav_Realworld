#!/usr/bin/env python3
"""HTTP client for the real-world NavDP ImageGoal and CEC wire formats."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from typing import Any, Mapping, Optional

import cv2
import numpy as np
import requests

from terminal_motion_override import (
    EXPECTED_HANDOFF_SCHEMA as EXPECTED_TERMINAL_HANDOFF_SCHEMA,
)

EXPECTED_CEC_PROTOCOL_VERSION = 3
NAVDP_WIRE_DEPTH_PNG_SCALE_M = 1.0e-4
EVALUATION_DEPTH_PNG_SCALE_M = 1.0e-3


class NavDPClient:
    def __init__(self, server_url: str, connect_timeout_s: float, request_timeout_s: float):
        self.server_url = server_url.rstrip("/")
        self.timeout = (float(connect_timeout_s), float(request_timeout_s))
        self.session = requests.Session()
        self.last_plan_receipt: dict[str, Any] = {}
        self.last_phase_receipt: dict[str, Any] = {}
        self.last_goal_jpeg: bytes | None = None
        self.last_goal_evaluation_depth_png: bytes | None = None
        self.last_goal_evaluation_depth_scale_m: float | None = None

    @staticmethod
    def _encode_rgb(rgb: np.ndarray) -> bytes:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"RGB image must have shape (H, W, 3), got {image.shape}")
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("failed to encode RGB image")
        return encoded.tobytes()

    @staticmethod
    def _encode_depth_at_scale(depth_m: np.ndarray, scale_m: float) -> bytes:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"depth image must have shape (H, W), got {depth.shape}")
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        encoded_depth = np.clip(
            depth / float(scale_m), 0.0, 65535.0
        ).astype(np.uint16)
        ok, encoded = cv2.imencode(".png", encoded_depth)
        if not ok:
            raise RuntimeError("failed to encode depth image")
        return encoded.tobytes()

    @classmethod
    def _encode_depth(cls, depth_m: np.ndarray) -> bytes:
        """Encode the frozen NavDP HTTP wire format (0.1 mm/unit)."""
        return cls._encode_depth_at_scale(
            depth_m, NAVDP_WIRE_DEPTH_PNG_SCALE_M
        )

    @classmethod
    def _encode_evaluation_depth(cls, depth_m: np.ndarray) -> bytes:
        """Encode the existing real-world evaluator format (1 mm/unit)."""
        return cls._encode_depth_at_scale(
            depth_m, EVALUATION_DEPTH_PNG_SCALE_M
        )

    def _post_phase_endpoint(
        self,
        route: str,
        files: Optional[dict] = None,
        data: Optional[dict] = None,
    ) -> dict:
        """POST a protocol-v3 phase endpoint, surfacing hub contract errors."""
        response = self.session.post(
            f"{self.server_url}{route}",
            files=files,
            data=data,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("error", ""))
            except Exception:
                detail = response.text[:200]
            raise RuntimeError(
                f"{route} rejected by hub ({response.status_code}): {detail}"
            )
        return response.json()

    def memory_step(
        self,
        rgb: np.ndarray,
        *,
        source_observation: Mapping[str, Any] | None = None,
    ) -> dict:
        """Protocol v3: record-only causal RGB append (memory_recording phase)."""
        data = None
        if source_observation is not None:
            data = {
                "source_observation": json.dumps(
                    dict(source_observation),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            }
        return self._post_phase_endpoint(
            "/memory_step",
            files={"image": ("image.jpg", self._encode_rgb(rgb), "image/jpeg")},
            data=data,
        )

    def novel_imagegoal_step(
        self,
        goal_rgb: np.ndarray,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        source_observation: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Record one causal frame while native NavDP executes a Novel goal."""
        files = {
            "image": ("image.jpg", self._encode_rgb(rgb), "image/jpeg"),
            "goal": ("goal.jpg", self._encode_rgb(goal_rgb), "image/jpeg"),
            # Wire compatibility only; the hub never forwards metric depth.
            "depth": ("depth.png", self._encode_depth(depth_m), "image/png"),
        }
        data = None
        if source_observation is not None:
            data = {
                "source_observation": json.dumps(
                    dict(source_observation),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            }
        response = self.session.post(
            f"{self.server_url}/novel_imagegoal_step",
            files=files,
            data=data,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("error", ""))
            except Exception:
                detail = response.text[:200]
            raise RuntimeError(
                "/novel_imagegoal_step rejected by hub "
                f"({response.status_code}): {detail}"
            )
        result = response.json()
        self.last_plan_receipt = {
            key: value for key, value in result.items()
            if key not in {"trajectory", "all_trajectory", "all_values"}
        }
        return (
            np.asarray(result["trajectory"], dtype=np.float32),
            np.asarray(result["all_trajectory"], dtype=np.float32),
            np.asarray(result["all_values"], dtype=np.float32),
        )

    def query_observation_step(self, rgb: np.ndarray, *, installed_goal_sha256: str | None) -> dict:
        """Advance query-time geometry only; never append to the policy FIFO."""
        return self._post_phase_endpoint(
            "/query_observation_step",
            files={"image": ("image.jpg", self._encode_rgb(rgb), "image/jpeg")},
            data={"installed_goal_sha256": installed_goal_sha256 or ""},
        )

    def goal_candidate(
        self,
        rgb: np.ndarray,
        *,
        validate_support: bool = False,
        evaluation_depth_m: np.ndarray | None = None,
    ) -> dict:
        """Protocol v3: register a goal-candidate photo excluded from memory."""
        files = {"image": ("image.jpg", self._encode_rgb(rgb), "image/jpeg")}
        if evaluation_depth_m is not None:
            files["evaluation_depth"] = (
                "evaluation_depth.png",
                self._encode_evaluation_depth(evaluation_depth_m),
                "image/png",
            )
        data = {"validate_support": "1" if validate_support else "0"}
        if evaluation_depth_m is not None:
            data["evaluation_depth_scale_m"] = str(
                EVALUATION_DEPTH_PNG_SCALE_M
            )
        response = self.session.post(
            f"{self.server_url}/goal_candidate",
            files=files,
            data=data,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("error", ""))
            except Exception:
                detail = response.text[:200]
            raise RuntimeError(
                f"/goal_candidate rejected by hub ({response.status_code}): {detail}"
            )
        return response.json()

    def begin_revisit(self, query_start_rgb: np.ndarray | None = None) -> dict:
        """Protocol v3: switch to revisit_query; hub warms NavDP and verifies."""
        files = None
        if query_start_rgb is not None:
            files = {"query_start": (
                "query_start.jpg", self._encode_rgb(query_start_rgb), "image/jpeg"
            )}
        receipt = self._post_phase_endpoint("/begin_revisit", files=files)
        self.last_phase_receipt = dict(receipt)
        return receipt

    def prepare_revisit(
        self, query_start_rgb: np.ndarray | None = None
    ) -> tuple[dict, np.ndarray]:
        """Atomically score/select a candidate, switch phase and install it.

        The hub owns the exact candidate JPEG.  The decoded RGB is returned for
        local display; subsequent control requests acknowledge the selected
        SHA-256 while the hub continues to use its committed bytes.
        """
        files = None
        if query_start_rgb is not None:
            files = {"query_start": (
                "query_start.jpg", self._encode_rgb(query_start_rgb), "image/jpeg"
            )}
        receipt = self._post_phase_endpoint("/prepare_revisit", files=files)
        encoded = receipt.get("goal_image_jpeg_base64")
        selected = receipt.get("selected_goal")
        if not isinstance(encoded, str) or not isinstance(selected, dict):
            raise RuntimeError("prepare_revisit omitted the selected goal payload")
        try:
            jpeg = base64.b64decode(encoded, validate=True)
        except Exception as error:
            raise RuntimeError(f"invalid selected goal encoding: {error}") from error
        expected = str(selected.get("sha256", ""))
        try:
            int(selected["candidate_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("selected goal omitted a valid candidate id") from error
        actual = hashlib.sha256(jpeg).hexdigest()
        if not expected or actual != expected:
            raise RuntimeError("selected goal SHA-256 mismatch")
        self.last_goal_jpeg = jpeg
        depth_encoded = receipt.get("goal_evaluation_depth_png_base64")
        self.last_goal_evaluation_depth_png = (
            None
            if depth_encoded is None
            else base64.b64decode(str(depth_encoded), validate=True)
        )
        depth_scale = receipt.get("goal_evaluation_depth_scale_m")
        self.last_goal_evaluation_depth_scale_m = (
            None if depth_scale is None else float(depth_scale)
        )
        if self.last_goal_evaluation_depth_png is not None and (
            self.last_goal_evaluation_depth_scale_m is None
            or not math.isfinite(self.last_goal_evaluation_depth_scale_m)
            or self.last_goal_evaluation_depth_scale_m <= 0.0
        ):
            raise RuntimeError(
                "selected evaluator depth has no valid metre scale"
            )
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("selected goal JPEG is not decodable")
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        status_receipt = {
            key: value for key, value in receipt.items()
            if key not in {
                "goal_image_jpeg_base64",
                "goal_evaluation_depth_png_base64",
            }
        }
        self.last_phase_receipt = status_receipt
        return status_receipt, rgb

    def prepare_revisit_goal(
        self,
        goal_rgb: np.ndarray,
        query_start_rgb: np.ndarray | None = None,
    ) -> tuple[dict, np.ndarray]:
        """Install one pre-episode frozen Revisit goal and switch phase."""
        files = {
            "goal": (
                "goal.jpg",
                self._encode_rgb(goal_rgb),
                "image/jpeg",
            )
        }
        if query_start_rgb is not None:
            files["query_start"] = (
                "query_start.jpg",
                self._encode_rgb(query_start_rgb),
                "image/jpeg",
            )
        response = self.session.post(
            f"{self.server_url}/prepare_revisit_goal",
            files=files,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("error", ""))
            except Exception:
                detail = response.text[:200]
            raise RuntimeError(
                "/prepare_revisit_goal rejected by hub "
                f"({response.status_code}): {detail}"
            )
        receipt = response.json()
        encoded = receipt.get("goal_image_jpeg_base64")
        selected = receipt.get("selected_goal")
        if not isinstance(encoded, str) or not isinstance(selected, dict):
            raise RuntimeError(
                "prepare_revisit_goal omitted the committed goal payload"
            )
        try:
            jpeg = base64.b64decode(encoded, validate=True)
        except Exception as error:
            raise RuntimeError(f"invalid committed goal encoding: {error}") from error
        expected = str(selected.get("sha256", ""))
        actual = hashlib.sha256(jpeg).hexdigest()
        if not expected or actual != expected:
            raise RuntimeError(
                "committed external goal SHA-256 does not match its receipt"
            )
        self.last_goal_jpeg = jpeg
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is None:
            raise RuntimeError("committed external goal JPEG is not decodable")
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        status_receipt = {
            key: value for key, value in receipt.items()
            if key != "goal_image_jpeg_base64"
        }
        self.last_goal_evaluation_depth_png = None
        self.last_goal_evaluation_depth_scale_m = None
        self.last_phase_receipt = status_receipt
        return status_receipt, rgb

    def dataset_status(self) -> dict:
        response = self.session.get(
            f"{self.server_url}/dataset/status", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def start_dataset(
        self, dataset_id: str, *, metadata: dict[str, Any] | None = None
    ) -> dict:
        response = self.session.post(
            f"{self.server_url}/dataset/start",
            json={"dataset_id": dataset_id, "metadata": dict(metadata or {})},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def seal_dataset(self) -> dict:
        response = self.session.post(
            f"{self.server_url}/dataset/seal", json={}, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def load_dataset(self, dataset_id: str) -> dict:
        timeout = (self.timeout[0], max(3600.0, self.timeout[1]))
        response = self.session.post(
            f"{self.server_url}/dataset/load",
            json={"dataset_id": dataset_id},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        response = self.session.get(f"{self.server_url}/healthz", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def validate_cec_contract(receipt: Mapping[str, Any]) -> None:
        if (receipt.get("protocol_version") != EXPECTED_CEC_PROTOCOL_VERSION
                or receipt.get("terminal_handoff_schema") != EXPECTED_TERMINAL_HANDOFF_SCHEMA
                or receipt.get("query_observation_supported") is not True):
            raise RuntimeError("CEC hub/Jetson contract mismatch: update both endpoints while motion-locked")

    def reset(self, intrinsic: np.ndarray, stop_threshold: float = -2.0) -> str:
        payload = {
            "intrinsic": np.asarray(intrinsic, dtype=float).tolist(),
            "stop_threshold": float(stop_threshold),
            "batch_size": 1,
            "sample_indices": [0],
            "scene_name": "go2_real",
        }
        response = self.session.post(
            f"{self.server_url}/navigator_reset", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        receipt = response.json()
        algorithm = str(receipt.get("algo", "unknown"))
        if algorithm == "cec_hybrid_navdp":
            self.validate_cec_contract(receipt)
        return algorithm

    def imagegoal_step(
        self,
        goal_rgb: np.ndarray,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        installed_goal_sha256: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        files = {
            "image": ("image.jpg", self._encode_rgb(rgb), "image/jpeg"),
            "goal": ("goal.jpg", self._encode_rgb(goal_rgb), "image/jpeg"),
            "depth": ("depth.png", self._encode_depth(depth_m), "image/png"),
        }
        request_args: dict[str, Any] = {
            "files": files,
            "timeout": self.timeout,
        }
        if installed_goal_sha256:
            request_args["data"] = {
                "installed_goal_sha256": str(installed_goal_sha256)
            }
        response = self.session.post(
            f"{self.server_url}/imagegoal_step", **request_args
        )
        response.raise_for_status()
        result = response.json()
        self.last_plan_receipt = {
            key: value for key, value in result.items()
            if key not in {"trajectory", "all_trajectory", "all_values"}
        }
        return (
            np.asarray(result["trajectory"], dtype=np.float32),
            np.asarray(result["all_trajectory"], dtype=np.float32),
            np.asarray(result["all_values"], dtype=np.float32),
        )
