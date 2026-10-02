#!/usr/bin/env python3
"""Shared evaluation helpers: the raw-video frame guard and the capture of the policy's live prediction heads."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO) + "/src")
if str(REPO / "src/attacca/evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "src/attacca/evaluation"))


MODEL_PREDICTION_CONTRACT = "xbench_rocket2_model_prediction/v1"


@dataclass
class ReviewFrameGuard:

    threshold: float = 25.0
    last_y: float | None = None
    kept_frames: int = 0
    dropped_frames: int = 0

    def assess(self, rgb: np.ndarray, pose: Mapping) -> tuple[int | None, str | None]:
        frame = np.asarray(rgb)
        if frame.shape != (360, 640, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"raw POV must be uint8 360x640 RGB, got {frame.dtype}{frame.shape}")
        if not bool(np.isfinite(frame).all()):
            raise ValueError("raw POV is non-finite")
        y = float(pose["y"])
        if not math.isfinite(y):
            raise ValueError("raw POV pose Y is non-finite")
        dy = 0.0 if self.last_y is None else y - float(self.last_y)
        self.last_y = y
        reason = None
        if float(frame.mean()) < float(self.threshold):
            reason = "near_black_mean_below_25"
        elif abs(dy) > 1.5:
            reason = "falling_abs_delta_y_above_1_5"
        if reason is not None:
            self.dropped_frames += 1
            return None, reason
        video_index = int(self.kept_frames)
        self.kept_frames += 1
        return video_index, None


def _prediction_vector(value: Any, *, field: str,
                       expected_size: int) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"model prediction {field} is not numeric") from exc
    if array.size != int(expected_size):
        raise RuntimeError(
            f"model prediction {field} shape drift: "
            f"expected={expected_size} observed={array.size}")
    if not bool(np.isfinite(array).all()):
        raise RuntimeError(f"model prediction {field} is non-finite")
    return np.ascontiguousarray(array)


def _prediction_digest(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _normalized_peak_yx(probability: np.ndarray) -> list[float]:
    index = int(np.argmax(np.asarray(probability).reshape(-1)))
    iy, ix = divmod(index, 14)
    return [float((iy + 0.5) / 14.0), float((ix + 0.5) / 14.0)]


def capture_model_prediction(latents: Mapping, *,
                             require_exist: bool = True) -> dict:
    if not isinstance(latents, Mapping):
        raise RuntimeError("CFGWrapper.cache_latents is unavailable")
    exist = _prediction_vector(
        latents.get("exist"), field="exist", expected_size=1)
    if require_exist and exist is None:
        raise RuntimeError("model has no predicted exist scalar")
    point = _prediction_vector(
        latents.get("point"), field="point", expected_size=2)
    bbox = _prediction_vector(
        latents.get("bbox"), field="bbox", expected_size=4)
    union_logits = _prediction_vector(
        latents.get("union_mask_logits"), field="union_mask_logits",
        expected_size=196)

    union_probability = None
    if union_logits is not None:
        clipped = np.clip(union_logits.astype(np.float64), -60.0, 60.0)
        union_probability = np.ascontiguousarray(
            1.0 / (1.0 + np.exp(-clipped)), dtype=np.float32)

    exist_logit = None if exist is None else float(exist[0])
    exist_probability = (
        None if exist_logit is None else
        float(1.0 / (1.0 + math.exp(
            -float(np.clip(exist_logit, -60.0, 60.0))))))
    point_values = None if point is None else [float(value) for value in point]
    bbox_values = None if bbox is None else [float(value) for value in bbox]
    point_valid = (
        None if point is None else
        bool(np.logical_and(point >= 0.0, point <= 1.0).all()))
    bbox_valid = (
        None if bbox is None else
        bool(np.logical_and(bbox >= 0.0, bbox <= 1.0).all()
             and float(bbox[0]) < float(bbox[2])
             and float(bbox[1]) < float(bbox[3])))

    capabilities = []
    if exist is not None:
        capabilities.append("exist")
    if point is not None:
        capabilities.append("point")
    if bbox is not None:
        capabilities.append("bbox")
    if union_probability is not None:
        capabilities.append("union_mask_14x14")
    return {
        "contract": MODEL_PREDICTION_CONTRACT,
        "source": "CFGWrapper.cache_latents_conditional_pass",
        "uses_ground_truth": False,
        "capabilities": capabilities,
        "exist_logit": exist_logit,
        "exist_probability": exist_probability,
        "point_yx_normalized": point_values,
        "point_normalized_valid": point_valid,
        "bbox_xyxy_normalized": bbox_values,
        "bbox_normalized_valid": bbox_valid,
        "mask_grid_shape": (
            None if union_probability is None else [14, 14]),
        "union_probability_14x14": (
            None if union_probability is None else
            [float(value) for value in union_probability]),
        "union_probability_float32_sha256": (
            None if union_probability is None else
            _prediction_digest(union_probability)),
        "union_peak_yx_normalized": (
            None if union_probability is None else
            _normalized_peak_yx(union_probability)),
    }
