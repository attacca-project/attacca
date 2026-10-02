#!/usr/bin/env python
"""Place human-play session: builds a command-generated cave scene with target and non-target markers, records human play, and writes exact per-frame target masks and behavioral phases after play."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO) + "/src")
sys.path.insert(0, str(REPO / "src/attacca/evaluation"))

import attacca.evaluation.geometry as G
from attacca.evaluation.policy import boot_world
from attacca.evaluation.mine_human import HumanUI
from attacca.evaluation.mine_human import ser_action
from attacca.evaluation.mine_scene import Stager
from attacca.runtime.bench_world import _fill
from attacca.worlds.human_viewmodel import RENDERER_VIEWMODEL_DEPTH_FIELD
from attacca.worlds.human_viewmodel import RENDERER_VIEWMODEL_INFO_FIELD
from attacca.worlds.human_viewmodel import decode_mask_runs_yx
from attacca.worlds.human_viewmodel import encode_mask_runs_yx
from attacca.worlds.human_viewmodel import install_renderer_viewmodel_observation
from attacca.worlds.episode_schema import VISIBLE_SURFACE_RENDER_POSE_SOURCE
from attacca.worlds.use_human_labels import MASK_SHAPE
from attacca.worlds.use_human_labels import USE_HUMAN_LABEL_CONTRACT
from attacca.worlds.use_human_labels import derive_use_training_labels
from attacca.worlds.use_human_scene import CAVE_MACRO_VARIANTS
from attacca.worlds.use_human_scene import CAVE_PANEL_CLEARANCE_HALF_WIDTH
from attacca.worlds.use_human_scene import CAVE_PANEL_CLEARANCE_HEIGHT
from attacca.worlds.use_human_scene import CAVE_PANEL_CLEARANCE_MAX_DEPTH
from attacca.worlds.use_human_scene import DEFAULT_TRAIN_MARKERS
from attacca.worlds.use_human_scene import HELD_BLOCK_ROSTER
from attacca.worlds.use_human_scene import MODE_TERMINAL_POLICY
from attacca.worlds.use_human_scene import MODE_TOOL_ID
from attacca.worlds.use_human_scene import USE_CLASS_VISIBILITY_INDEX
from attacca.worlds.use_human_scene import USE_CLASS_VISIBILITY_CENSUS_CONTRACT
from attacca.worlds.use_human_scene import USE_CLASS_VISIBILITY_ROSTER_CONTRACT
from attacca.worlds.use_human_scene import USE_INTERACTION_ID
from attacca.worlds.use_human_scene import USE_INTERACTION_IDS
from attacca.worlds.use_human_scene import USE_REACH_BLOCKS
from attacca.worlds.use_human_scene import ZERO_SHOT_MARKERS
from attacca.worlds.use_human_scene import CavePortalScene
from attacca.worlds.use_human_scene import CaveTrainingScene
from attacca.worlds.use_human_scene import Marker
from attacca.worlds.use_human_scene import MarkerRoom
from attacca.worlds.use_human_scene import bare_block
from attacca.worlds.use_human_scene import build_cave_training_scene


FPS = 20.0
PLACE_MODE = "block"
FRAME_SIZE = (640, 360)
DEPTH_NEAR = 0.05
DEPTH_FAR = 256.0
MIN_RECOGNIZABLE_AREA = 150
MIN_RECOGNIZABLE_SHORT_SIDE = 6
MAX_SECONDS = 240.0
FRAME_CONTRACT = "use_human_exact_block_frame/v2"
STORY_CONTRACT = "use_human_semantic_marker_story/v4"
CAMPAIGN_CONTRACT = "use_human_semantic_marker_campaign/v4"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(item) for item in value]
    return str(value)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(value), stream, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_live_status(path: Path | None, state: str, **fields: Any) -> None:
    if path is None:
        return
    atomic_json(Path(path), {
        "contract": "use_human_live_status/v1",
        "pid": int(os.getpid()),
        "state": str(state),
        "updated_unix_s": float(time.time()),
        **fields,
    })


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(
                json_safe(row), sort_keys=True, allow_nan=False) + "\n")


def machine_record() -> dict[str, Any]:
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=10).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu = []
    return {
        "gpu_names": gpu,
    }


def pose(world, *, render_position: Sequence[float] | None = None) -> dict[str, Any]:
    x, y, z = (
        world.get_pos() if render_position is None else render_position)
    return {
        "x": float(x), "y": float(y), "z": float(z),
        "eye_height": 1.62,
        "yaw": float(world.get_yaw()), "pitch": float(world.get_pitch()),
        "source": VISIBLE_SURFACE_RENDER_POSE_SOURCE,
    }


def held_item(world) -> str:
    raw = ((world.info.get("equipped_items") or {}).get("mainhand") or {}).get(
        "type")
    return bare_block(str(raw or ""))


def normalized_counter(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    output: Counter[str] = Counter()
    for key, raw in value.items():
        try:
            amount = int(raw)
        except (TypeError, ValueError):
            continue
        output[bare_block(str(key))] += amount
    return dict(output)


def positive_counter_delta(current: Mapping[str, int],
                           previous: Mapping[str, int]) -> dict[str, int]:
    keys = set(current) | set(previous)
    return {
        key: int(current.get(key, 0)) - int(previous.get(key, 0))
        for key in keys
        if int(current.get(key, 0)) > int(previous.get(key, 0))
    }


def strict_depth_arrays(world) -> tuple[np.ndarray, np.ndarray]:
    hand = np.asarray(world.info.get(RENDERER_VIEWMODEL_INFO_FIELD))
    depth = np.asarray(world.info.get(RENDERER_VIEWMODEL_DEPTH_FIELD))
    if hand.dtype != np.uint8 or hand.shape != MASK_SHAPE:
        raise RuntimeError(f"bad viewmodel mask {hand.dtype}{hand.shape}")
    if depth.shape != MASK_SHAPE or not np.issubdtype(depth.dtype, np.floating):
        raise RuntimeError(f"bad native depth {depth.dtype}{depth.shape}")
    return hand, depth


def recognizable(mask: np.ndarray) -> tuple[bool, list[int] | None, int]:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return False, None, 0
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
    short = min(bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
    return bool(len(xx) >= MIN_RECOGNIZABLE_AREA
                and short >= MIN_RECOGNIZABLE_SHORT_SIDE), bbox, int(short)


def aabb_distance(eye: Sequence[float], cell: Sequence[int]) -> float:
    eye_arr = np.asarray(eye, dtype=np.float64)
    lo = np.asarray(cell, dtype=np.float64)
    hi = lo + 1.0
    delta = np.maximum(np.maximum(lo - eye_arr, eye_arr - hi), 0.0)
    return float(np.linalg.norm(delta))


def _owner_pixels_from_depth(
        hand: np.ndarray, depth: np.ndarray,
        frame_pose: Mapping[str, float], markers: Sequence[Marker], *,
        lookup_scope: str, audit_per_marker_depth: bool = False,
) -> tuple[tuple[float, float, float], dict[tuple[int, int, int], np.ndarray],
           dict[str, Any]]:
    eye = (frame_pose["x"], frame_pose["y"] + frame_pose["eye_height"],
           frame_pose["z"])
    depth_started = time.perf_counter()
    G.clear_depth_frame()
    cellmap, valid = G.backproject_depth_to_cells(
        depth, eye, frame_pose["yaw"], frame_pose["pitch"],
        near=DEPTH_NEAR, far=DEPTH_FAR, pixel_exclusion_mask=hand)
    G.set_depth_frame(
        eye, frame_pose["yaw"], frame_pose["pitch"], cellmap, valid,
        near=DEPTH_NEAR, far=DEPTH_FAR,
        source=VISIBLE_SURFACE_RENDER_POSE_SOURCE,
        build_owner_index=True, audit_owner_index=False)
    try:
        owner_pixels = {
            marker.cell: G.depth_owner_pixel_indices(
                eye, frame_pose["yaw"], frame_pose["pitch"], marker.cell)
            for marker in markers
        }
        owner_stats = dict(G.active_depth_frame()["owner_stats"])
    finally:
        G.clear_depth_frame()
    owner_stats["full_frame_backproject_and_index_ms"] = round(
        (time.perf_counter() - depth_started) * 1000.0, 3)
    owner_stats["contract"] = "native_depth_full_frame_owner_index_once/v1"
    owner_stats["lookup_scope"] = str(lookup_scope)
    owner_stats["per_marker_backprojection_used"] = False
    owner_stats["marker_lookup_count"] = len(markers)
    if audit_per_marker_depth:
        per_marker = G.backproject_depth_owner_pixels_for_blocks(
            depth, eye, frame_pose["yaw"], frame_pose["pitch"],
            [row.cell for row in markers],
            near=DEPTH_NEAR, far=DEPTH_FAR,
            pixel_exclusion_mask=hand)
        mismatched = [
            list(marker.cell) for marker in markers
            if not np.array_equal(
                owner_pixels[marker.cell], per_marker[marker.cell])]
        if mismatched:
            raise RuntimeError(
                "full-frame owner index differs from the per-marker exact depth "
                f"path for marker cells: {mismatched[:8]}")
        owner_stats["per_marker_backprojection_equivalence_audited"] = True
    return eye, owner_pixels, owner_stats


def _compact_pixel_geometry(
        pixels: np.ndarray,
) -> tuple[bool, list[int] | None, int]:
    flat = np.asarray(pixels, dtype=np.int32)
    if not len(flat):
        return False, None, 0
    yy = flat // MASK_SHAPE[1]
    xx = flat % MASK_SHAPE[1]
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
    short = min(bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
    return bool(len(flat) >= MIN_RECOGNIZABLE_AREA
                and short >= MIN_RECOGNIZABLE_SHORT_SIDE), bbox, int(short)


def _compact_visibility_from_target_instances(
        instances: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [{
        key: instance[key]
        for key in (
            "instance_id", "kind", "cell", "place_cell", "role",
            "recognizable", "pixel_count", "bbox_xyxy",
            "visible_short_side_px", "nearest_surface_distance")
    } for instance in instances]


def _frame_artifacts_from_depth(
        hand: np.ndarray, depth: np.ndarray, frame_pose: Mapping[str, float],
        room: MarkerRoom, *, completed_ids: set[str],
        include_class_census: bool, audit_per_marker_depth: bool = False,
) -> dict[str, Any]:
    if tuple(room.train_markers) != tuple(
            USE_CLASS_VISIBILITY_ROSTER_CONTRACT["classes"]):
        raise RuntimeError("USE census requires the canonical seven-class roster")
    active = [row for row in room.targets
              if row.instance_id not in completed_ids]
    active_ids = {row.instance_id for row in active}
    markers = (
        [row for row in room.markers
         if row.role != "target" or row.instance_id in active_ids]
        if include_class_census else active)
    eye, owner_pixels, owner_stats = _owner_pixels_from_depth(
        hand, depth, frame_pose, markers,
        lookup_scope=(
            "all_active_markers_and_confusers"
            if include_class_census else "active_goal_markers_only"),
        audit_per_marker_depth=audit_per_marker_depth)
    union = np.zeros(MASK_SHAPE, np.uint8)
    instances = []
    census_instances = []
    census_visible_bits = 0
    centre_flat = 180 * 640 + 320
    crosshair_id = None
    crosshair_distance = None
    for marker in markers:
        pixels = np.asarray(owner_pixels.get(marker.cell, ()), dtype=np.int32)
        mask = np.zeros(MASK_SHAPE, np.uint8)
        if len(pixels):
            mask.reshape(-1)[pixels] = 1
        is_recognizable, bbox, short = recognizable(mask)
        distance = aabb_distance(eye, marker.cell)
        if marker.instance_id in active_ids and np.any(pixels == centre_flat):
            crosshair_id = marker.instance_id
            crosshair_distance = distance
        if is_recognizable and marker.instance_id in active_ids:
            union |= mask
        if len(pixels):
            surface = {
                "instance_id": marker.instance_id,
                "kind": marker.kind,
                "cell": list(marker.cell),
                "place_cell": list(marker.place_cell),
                "role": marker.role,
                "recognizable": int(is_recognizable),
                "pixel_count": int(mask.sum()),
                "bbox_xyxy": bbox,
                "visible_short_side_px": int(short),
                "nearest_surface_distance": float(distance),
                "mask_runs_yx": encode_mask_runs_yx(mask),
                "mask_semantic": "exact_native_depth_first_hit_voxel_xyz/v1",
                "recognition_gate": "full_cube_area>=150_short_side>=6/v7",
            }
            if marker.instance_id in active_ids:
                instances.append(dict(surface))
            if include_class_census and is_recognizable:
                census_visible_bits |= 1 << USE_CLASS_VISIBILITY_INDEX[
                    marker.kind]
                census_instances.append(dict(surface))
    crosshair_in_reach = bool(
        crosshair_id is not None and crosshair_distance is not None
        and crosshair_distance <= USE_REACH_BLOCKS)
    row = {
        "render_pose": dict(frame_pose),
        "target_instances": instances,
        "raw_recognizable_union_mask_runs_yx": encode_mask_runs_yx(union),
        "raw_recognizable_union_pixel_count": int(union.sum()),
        "crosshair_target_instance_id": crosshair_id,
        "crosshair_target_surface_distance": crosshair_distance,
        "crosshair_target_in_reach": int(crosshair_in_reach),
        "viewmodel_occlusion_subtracted": True,
        "depth_owner_index_stats": owner_stats,
    }
    if include_class_census:
        row.update({
            "all_train_class_depth_census": True,
            "class_visibility_census_contract": (
                USE_CLASS_VISIBILITY_CENSUS_CONTRACT),
            "class_visibility_roster": list(
                USE_CLASS_VISIBILITY_ROSTER_CONTRACT["classes"]),
            "class_visibility_roster_sha256": (
                USE_CLASS_VISIBILITY_ROSTER_CONTRACT["sha256"]),
            "class_visibility_known_bits": (
                (1 << len(USE_CLASS_VISIBILITY_ROSTER_CONTRACT["classes"])) - 1),
            "class_visibility_visible_bits": int(census_visible_bits),
            "class_visibility_geometry_valid": 1,
            "class_visible_instances": census_instances,
        })
    return {
        "union": union,
        "row": row,
    }


def _live_frame_artifacts_from_depth(
        hand: np.ndarray, depth: np.ndarray, frame_pose: Mapping[str, float],
        room: MarkerRoom, *, completed_ids: set[str],
) -> dict[str, Any]:
    if tuple(room.train_markers) != tuple(
            USE_CLASS_VISIBILITY_ROSTER_CONTRACT["classes"]):
        raise RuntimeError("USE census requires the canonical seven-class roster")
    active = [row for row in room.targets
              if row.instance_id not in completed_ids]
    eye, owner_pixels, owner_stats = _owner_pixels_from_depth(
        hand, depth, frame_pose, active,
        lookup_scope="live_active_goal_recognition_and_crosshair_only")
    centre_flat = 180 * MASK_SHAPE[1] + 320
    crosshair_id = None
    crosshair_distance = None
    visibility_stats = []
    recognizable_ids = []
    for marker in active:
        pixels = np.asarray(owner_pixels.get(marker.cell, ()), dtype=np.int32)
        is_recognizable, bbox, short = _compact_pixel_geometry(pixels)
        distance = aabb_distance(eye, marker.cell)
        if np.any(pixels == centre_flat):
            crosshair_id = marker.instance_id
            crosshair_distance = distance
        if len(pixels):
            visibility_stats.append({
                "instance_id": marker.instance_id,
                "kind": marker.kind,
                "cell": list(marker.cell),
                "place_cell": list(marker.place_cell),
                "role": marker.role,
                "recognizable": int(is_recognizable),
                "pixel_count": int(len(pixels)),
                "bbox_xyxy": bbox,
                "visible_short_side_px": int(short),
                "nearest_surface_distance": float(distance),
            })
        if is_recognizable:
            recognizable_ids.append(marker.instance_id)
    crosshair_in_reach = bool(
        crosshair_id is not None and crosshair_distance is not None
        and crosshair_distance <= USE_REACH_BLOCKS)
    return {
        "row": {
            "render_pose": dict(frame_pose),
            "live_target_visibility_stats": visibility_stats,
            "live_recognizable_target_instance_ids": recognizable_ids,
            "live_exact_mask_materialized": False,
            "live_label_timing": (
                "exact_compact_recognition_and_crosshair_before_action;"
                "dense_masks_deferred_after_human_input"),
            "crosshair_target_instance_id": crosshair_id,
            "crosshair_target_surface_distance": crosshair_distance,
            "crosshair_target_in_reach": int(crosshair_in_reach),
            "viewmodel_occlusion_subtracted": True,
            "depth_owner_index_stats": owner_stats,
        },
    }


def current_frame_artifacts(
        world, room: MarkerRoom, *, completed_ids: set[str],
        render_position: Sequence[float] | None = None,
        include_class_census: bool = True,
        audit_per_marker_depth: bool = False) -> dict[str, Any]:
    hand, depth = strict_depth_arrays(world)
    result = _frame_artifacts_from_depth(
        hand, depth, pose(world, render_position=render_position), room,
        completed_ids=completed_ids,
        include_class_census=include_class_census,
        audit_per_marker_depth=audit_per_marker_depth)
    result["_native_depth"] = depth
    result["_viewmodel_mask"] = hand
    return result


def current_live_frame_artifacts(
        world, room: MarkerRoom, *, completed_ids: set[str],
        render_position: Sequence[float] | None = None) -> dict[str, Any]:
    hand, depth = strict_depth_arrays(world)
    result = _live_frame_artifacts_from_depth(
        hand, depth, pose(world, render_position=render_position), room,
        completed_ids=completed_ids)
    result["_native_depth"] = depth
    result["_viewmodel_mask"] = hand
    return result


def apply_deferred_class_census(
        rows: Sequence[dict[str, Any]],
        label_frames: Sequence[tuple[np.ndarray, np.ndarray]],
        room: MarkerRoom, *, diagnostic_dir: Path | None = None,
) -> dict[str, Any]:
    if len(rows) != len(label_frames):
        raise RuntimeError(
            f"deferred census frame mismatch {len(rows)}!={len(label_frames)}")
    census_fields = (
        "all_train_class_depth_census",
        "class_visibility_census_contract",
        "class_visibility_roster",
        "class_visibility_roster_sha256",
        "class_visibility_known_bits",
        "class_visibility_visible_bits",
        "class_visibility_geometry_valid",
        "class_visible_instances",
    )
    exact_target_fields = (
        "target_instances",
        "raw_recognizable_union_mask_runs_yx",
        "raw_recognizable_union_pixel_count",
    )
    crosshair_fields = (
        "crosshair_target_instance_id",
        "crosshair_target_surface_distance",
        "crosshair_target_in_reach",
    )
    started = time.perf_counter()
    frame_ms = []
    buffered_bytes = 0
    for index, (row, raw_frame) in enumerate(zip(rows, label_frames)):
        hand, depth = raw_frame
        buffered_bytes += int(hand.nbytes + depth.nbytes)
        frame_started = time.perf_counter()
        produced = _frame_artifacts_from_depth(
            hand, depth, row["render_pose"], room,
            completed_ids=set(row["completed_instance_ids_before_action"]),
            include_class_census=True, audit_per_marker_depth=False)["row"]
        produced_visibility_stats = _compact_visibility_from_target_instances(
            produced["target_instances"])
        produced_recognizable_ids = [
            instance["instance_id"]
            for instance in produced["target_instances"]
            if int(instance["recognizable"]) == 1]
        live_checks = (
            ("live_target_visibility_stats",
             row.get("live_target_visibility_stats"),
             produced_visibility_stats),
            ("live_recognizable_target_instance_ids",
             row.get("live_recognizable_target_instance_ids"),
             produced_recognizable_ids),
            *((field, row.get(field), produced.get(field))
              for field in crosshair_fields),
        )
        for field, live_value, deferred_value in live_checks:
            if live_value != deferred_value:
                mismatch = {
                    "contract": "use_human_deferred_census_mismatch/v2",
                    "f": int(index),
                    "field": field,
                    "live_value": live_value,
                    "deferred_value": deferred_value,
                    "live_target_visibility_stats": row.get(
                        "live_target_visibility_stats"),
                    "deferred_target_instances": produced.get(
                        "target_instances"),
                    "live_depth_stats": row.get("depth_owner_index_stats"),
                    "deferred_depth_stats": produced.get(
                        "depth_owner_index_stats"),
                }
                if diagnostic_dir is not None:
                    diagnostic_dir.mkdir(parents=True, exist_ok=True)
                    atomic_json(
                        diagnostic_dir / "deferred_census_mismatch.json",
                        json_safe(mismatch))
                    for prefix, source in (
                            ("live", row), ("deferred", produced)):
                        mask = np.zeros(MASK_SHAPE, np.uint8)
                        for instance in source.get("target_instances") or ():
                            mask |= decode_mask_runs_yx(
                                list(instance.get("mask_runs_yx") or []),
                                list(MASK_SHAPE)).astype(np.uint8)
                        cv2.imwrite(str(
                            diagnostic_dir /
                            f"deferred_census_mismatch_f{index:05d}_{prefix}.png"),
                            mask * 255)
                raise RuntimeError(
                    "deferred full census changed live compact target proof "
                    f"at f={index} field={field}")
        for field in exact_target_fields:
            row[field] = produced[field]
        for field in census_fields:
            row[field] = produced[field]
        row["depth_owner_index_stats"] = produced[
            "depth_owner_index_stats"]
        row["class_visibility_census_timing"] = "deferred_after_human_input"
        row["exact_target_mask_timing"] = "deferred_after_human_input"
        frame_ms.append((time.perf_counter() - frame_started) * 1000.0)
    timings = np.asarray(frame_ms, dtype=np.float64)
    return {
        "contract": "use_human_deferred_train7_depth_census/v2",
        "frames": len(rows),
        "buffered_native_depth_and_viewmodel_bytes": int(buffered_bytes),
        "live_compact_target_vs_deferred_full_census_exact": True,
        "dense_target_masks_materialized_only_after_human_input": True,
        "human_input_complete_before_census": True,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "frame_ms_mean": (
            0.0 if not len(timings) else round(float(timings.mean()), 3)),
        "frame_ms_p95": (
            0.0 if not len(timings)
            else round(float(np.percentile(timings, 95)), 3)),
    }


def overlay_masks(rgb: np.ndarray, union: np.ndarray,
                  chosen: np.ndarray) -> np.ndarray:
    output = np.asarray(rgb, dtype=np.uint8).copy()
    union_bool = np.asarray(union).astype(bool)
    output[union_bool] = (
        0.62 * output[union_bool] + 0.38 * np.array([40, 230, 70])
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        union_bool.astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(output, contours, -1, (30, 255, 30), 1)
    chosen_bool = np.asarray(chosen).astype(bool)
    contours, _ = cv2.findContours(
        chosen_bool.astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(output, contours, -1, (255, 220, 0), 3)
    return output


def overlay_class_census(
        rgb: np.ndarray, instances: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, int]]:
    colors = (
        (255, 80, 80), (80, 180, 255), (255, 190, 50),
        (90, 230, 110), (190, 110, 255), (255, 110, 210),
        (80, 235, 225),
    )
    output = np.asarray(rgb, dtype=np.uint8).copy()
    pixels_by_class = Counter()
    for instance in instances:
        kind = str(instance["kind"])
        mask = decode_mask_runs_yx(
            list(instance.get("mask_runs_yx") or []), list(MASK_SHAPE))
        if not np.any(mask):
            continue
        pixels_by_class[kind] += int(mask.sum())
        color = np.asarray(
            colors[USE_CLASS_VISIBILITY_INDEX[kind]], dtype=np.float64)
        selected = mask.astype(bool)
        output[selected] = (
            0.55 * output[selected] + 0.45 * color).astype(np.uint8)
        contours, _ = cv2.findContours(
            selected.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(output, contours, -1, tuple(int(v) for v in color), 2)
    return output, {
        kind: int(pixels_by_class.get(kind, 0))
        for kind in USE_CLASS_VISIBILITY_ROSTER_CONTRACT["classes"]
    }


class UseHumanUI(HumanUI):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in tuple(self.K2A):
            remove_hotbar = name.startswith("hotbar.")
            if name == "inventory" or remove_hotbar:
                self.K2A.pop(name, None)
        self.live_union_mask: np.ndarray | None = None

    def wait_for_start(self, pov_rgb, mission, *, start_allowed=None):
        if start_allowed is None:
            return super().wait_for_start(
                pov_rgb, mission, start_allowed=start_allowed)
        frozen = np.asarray(pov_rgb).copy()
        if frozen.shape != (360, 640, 3):
            raise ValueError(
                f"ready gate requires 640x360 RGB, got {frozen.shape}")
        permit_wait_started = time.monotonic()
        self.ready_gate_active = True
        self.pressed.clear()
        self.mouse_delta = [0.0, 0.0]
        self.consume_resume_marker()
        self.grab_mouse(False)
        try:
            while not self.abort and not self.quit and not start_allowed():
                self.pump()
                self.draw(
                    frozen, mission,
                    "STAGED — waiting for this window's play permit",
                    selection_prompt=True, ready_prompt=True,
                    freeze_reason="ready")
                self.consume_resume_marker()
                time.sleep(0.1)
        finally:
            self.ready_gate_active = False
            self.pressed.clear()
            self.mouse_delta = [0.0, 0.0]
        if self.abort or self.quit:
            return {
                "started": False,
                "wait_s": max(
                    0.0, time.monotonic() - permit_wait_started),
                "simulator_steps": 0,
                "frames_recorded": 0,
                "reason": "quit" if self.quit else "aborted",
            }
        permit_wait_s = max(0.0, time.monotonic() - permit_wait_started)
        result = super().wait_for_start(
            frozen, mission, start_allowed=None)
        result["wait_s"] = float(result.get("wait_s", 0.0)) + permit_wait_s
        return result

    def draw(self, pov_rgb, mission, sub, flash=None,
             flash_color=(0, 220, 0, 255), selection_prompt=False,
             selection_overlay=None, candidate_overlays=None,
             selection_cursor_xy=None, freeze_reason=None,
             ready_prompt=False):
        display = (
            np.asarray(pov_rgb)
            if self.live_union_mask is None else
            overlay_masks(
                np.asarray(pov_rgb), self.live_union_mask,
                np.zeros(MASK_SHAPE, np.uint8)))
        window = self.win
        window.switch_to()
        window.clear()
        resized = cv2.resize(
            display, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        image = self.pyglet.image.ImageData(
            resized.shape[1], resized.shape[0], "RGB", resized.tobytes(),
            pitch=resized.shape[1] * -3)
        image.blit(0, self.INFO_H)
        self._label(mission, self.W // 2, self.INFO_H - 22, size=15,
                    anchor_x="center", color=(255, 230, 90, 255), bold=True)
        controls = ("READY | V start | ESC abort" if ready_prompt else
                    "PAUSED | V resume | ESC twice abort" if selection_prompt else
                    "WASD + mouse | RMB place/use | P pause | TAB mouse | ESC twice abort")
        if self.abort_confirmation_active():
            controls = "PAUSED — ESC again within 3s to abort | V resume"
            flash = "PRESS ESC AGAIN TO ABORT"
            flash_color = (255, 70, 70, 255)
        if flash:
            self._label(flash, self.W // 2, self.INFO_H - 52, size=12,
                        anchor_x="center", color=flash_color, bold=True)
        else:
            self._label(sub, self.W // 2, self.INFO_H - 52, size=10,
                        anchor_x="center", color=(200, 200, 200, 255))
        self._label(controls, self.W // 2, 14, size=9, anchor_x="center",
                    color=(140, 140, 140, 255))
        window.flip()


def pause_ui(ui: UseHumanUI, rgb: np.ndarray, mission: str,
             union: np.ndarray | None) -> None:
    ui.target_selection_active = True
    ui.target_selection_clickable = False
    ui.pressed.clear()
    ui.mouse_delta = [0.0, 0.0]
    ui.consume_resume_marker()
    ui.grab_mouse(False)
    ui.live_union_mask = union
    try:
        while not ui.abort and not ui.quit:
            ui.pump()
            ui.draw(rgb, mission, "frozen — no simulator step",
                    selection_prompt=True, freeze_reason="manual_pause")
            if ui.consume_resume_marker():
                break
            time.sleep(1.0 / 120.0)
    finally:
        ui.target_selection_active = False
        ui.pressed.clear()
        ui.mouse_delta = [0.0, 0.0]
        if not ui.quit and not ui.abort:
            ui.grab_mouse(True)


def stage_cave_commands(
        world, scene: CavePortalScene | CaveTrainingScene) -> None:
    x0, x1, z0, z1 = scene.walkable_bounds_xz
    y = scene.feet_y
    world.cmd("/gamerule doMobSpawning false")
    world.cmd("/gamerule doMobLoot false")
    world.cmd("/gamerule randomTickSpeed 0")
    world.cmd("/time set 6000")
    world.cmd("/weather clear")
    world.cmd("/kill @e[type=!minecraft:player]")

    _fill(world, x0 - 1, y - 2, z0 - 1,
          x1 + 1, y + 8, z1 + 1, scene.wall_block)
    _fill(world, x0, y, z0, x1, y + 6, z1, "air")
    rng = np.random.default_rng(int(scene.seed) ^ 0x43415645)
    floor_materials = tuple(scene.cave_palette[1:]) or (scene.cave_palette[0],)
    wall_materials = tuple(
        value for value in scene.cave_palette[1:] if value != "gravel")
    if not wall_materials:
        wall_materials = (scene.cave_palette[0],)
    macro = getattr(scene, "macro_variant", "oval_even")
    shoulder_outer, shoulder_inner = {
        "oval_even": (0.82, 0.68),
        "oval_low_shoulders": (0.76, 0.62),
        "oval_rocky_floor": (0.82, 0.68),
        "oval_dense_stalactites": (0.82, 0.68),
    }.get(macro, (0.82, 0.68))
    mid_x, mid_z = (x0 + x1) / 2.0, (z0 + z1) / 2.0
    for xx in range(x0, x1 + 1):
        for zz in range(z0, z1 + 1):
            radial = math.hypot(
                (float(xx) - mid_x) / 10.0,
                (float(zz) - mid_z) / 10.0)
            if radial > 1.0:
                _fill(world, xx, y, zz, xx, y + 6, zz, scene.wall_block)
            elif radial > shoulder_outer:
                _fill(world, xx, y + 5, zz, xx, y + 6, zz,
                      scene.wall_block)
            elif radial > shoulder_inner:
                world.cmd(
                    f"/setblock {xx} {y+6} {zz} minecraft:{scene.wall_block}")

    floor_veins = 26 if macro == "oval_rocky_floor" else 18
    for _ in range(floor_veins):
        xx = int(rng.integers(x0 + 2, x1 - 1))
        zz = int(rng.integers(z0 + 2, z1 - 1))
        block = floor_materials[int(rng.integers(len(floor_materials)))]
        for _ in range(int(rng.integers(2, 6))):
            world.cmd(f"/setblock {xx} {y-1} {zz} minecraft:{block}")
            xx = int(np.clip(xx + rng.integers(-1, 2), x0 + 1, x1 - 1))
            zz = int(np.clip(zz + rng.integers(-1, 2), z0 + 1, z1 - 1))

    for _ in range(30):
        side = int(rng.integers(4))
        yy = int(rng.integers(y, y + 6))
        block = wall_materials[int(rng.integers(len(wall_materials)))]
        width = int(rng.integers(1, 4))
        height = int(rng.integers(1, 3))
        if side == 0:
            zz = int(rng.integers(z0 + 1, z1 - width + 2))
            _fill(world, x0 - 1, yy, zz,
                  x0 - 1, min(y + 6, yy + height - 1), zz + width - 1, block)
        elif side == 1:
            zz = int(rng.integers(z0 + 1, z1 - width + 2))
            _fill(world, x1 + 1, yy, zz,
                  x1 + 1, min(y + 6, yy + height - 1), zz + width - 1, block)
        elif side == 2:
            xx = int(rng.integers(x0 + 1, x1 - width + 2))
            _fill(world, xx, yy, z0 - 1,
                  xx + width - 1, min(y + 6, yy + height - 1), z0 - 1, block)
        else:
            xx = int(rng.integers(x0 + 1, x1 - width + 2))
            _fill(world, xx, yy, z1 + 1,
                  xx + width - 1, min(y + 6, yy + height - 1), z1 + 1, block)

    boulder_count = 20 if macro == "oval_rocky_floor" else 12
    for _ in range(boulder_count):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        radius = float(rng.uniform(6.8, 8.2))
        xx = int(round(mid_x + math.cos(angle) * radius))
        zz = int(round(mid_z + math.sin(angle) * radius))
        block = wall_materials[int(rng.integers(len(wall_materials)))]
        world.cmd(f"/setblock {xx} {y} {zz} minecraft:{block}")
        if bool(rng.integers(0, 2)):
            world.cmd(f"/setblock {xx} {y+1} {zz} minecraft:{block}")
    stalactite_count = 30 if macro == "oval_dense_stalactites" else 16
    for _ in range(stalactite_count):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        radius = float(rng.uniform(3.0, 7.2))
        xx = int(round(mid_x + math.cos(angle) * radius))
        zz = int(round(mid_z + math.sin(angle) * radius))
        block = wall_materials[int(rng.integers(len(wall_materials)))]
        world.cmd(f"/setblock {xx} {y+6} {zz} minecraft:{block}")
        if bool(rng.integers(0, 4) == 0):
            world.cmd(f"/setblock {xx} {y+5} {zz} minecraft:{block}")

    for _panel_id, _kind, center, normal in getattr(
            scene, "panel_assignments", ()):
        if normal[2]:
            _fill(world, center[0] - 4, y, center[2],
                  center[0] + 4, y + 6, center[2], scene.wall_block)
            _fill(world, center[0] - 4, y, center[2] + normal[2],
                  center[0] + 4, y + 6, center[2] + normal[2], "air")
        else:
            _fill(world, center[0], y, center[2] - 4,
                  center[0], y + 6, center[2] + 4, scene.wall_block)
            _fill(world, center[0] + normal[0], y, center[2] - 4,
                  center[0] + normal[0], y + 6, center[2] + 4, "air")
        tangent = ((1, 0, 0) if normal[2] else (0, 0, 1))
        for depth in range(2, CAVE_PANEL_CLEARANCE_MAX_DEPTH + 1):
            for h in range(
                    -CAVE_PANEL_CLEARANCE_HALF_WIDTH,
                    CAVE_PANEL_CLEARANCE_HALF_WIDTH + 1):
                cell = (
                    center[0] + normal[0] * depth + tangent[0] * h,
                    y,
                    center[2] + normal[2] * depth + tangent[2] * h,
                )
                _fill(
                    world, cell[0], cell[1], cell[2],
                    cell[0], cell[1] + CAVE_PANEL_CLEARANCE_HEIGHT - 1,
                    cell[2], "air")

    for cell in scene.light_cells:
        world.cmd(
            f"/setblock {cell[0]} {cell[1]} {cell[2]} minecraft:glowstone")
    for cell, kind in scene.prebuilt_blocks:
        world.cmd(f"/setblock {cell[0]} {cell[1]} {cell[2]} minecraft:{kind}")
    for marker in scene.markers:
        x, yy, z = marker.cell
        px, py, pz = marker.place_cell
        world.cmd(f"/setblock {x} {yy} {z} minecraft:{marker.kind}")
        world.cmd(f"/setblock {px} {py} {pz} minecraft:air")
    for cell in scene.portal_interior_cells:
        world.cmd(f"/setblock {cell[0]} {cell[1]} {cell[2]} minecraft:air")
    world.cmd("/kill @e[type=!minecraft:player]")
    for _ in range(16):
        world.step_noop()


def audit_staged_cave(
        world, stager: Stager,
        scene: CavePortalScene | CaveTrainingScene) -> dict[str, Any]:
    if not stager.go(
            scene.center_x + 0.5, scene.feet_y, scene.center_z + 0.5,
            0.0, pitch=8.0, n=8):
        raise RuntimeError("cave centre pose was not acknowledged")
    relative = stager.converged_relative_voxels(
        (-13, 13, -3, 10, -13, 13))

    def block_at(cell):
        rel = (cell[0] - scene.center_x, cell[1] - scene.feet_y,
               cell[2] - scene.center_z)
        return bare_block(relative.get(rel, "air"))

    errors = []
    for marker in scene.markers:
        if block_at(marker.cell) != marker.kind:
            errors.append({"cell": list(marker.cell), "expected": marker.kind,
                           "actual": block_at(marker.cell)})
        if block_at(marker.place_cell) != "air":
            errors.append({"cell": list(marker.place_cell), "expected": "air",
                           "actual": block_at(marker.place_cell)})
    for cell, kind in scene.prebuilt_blocks:
        if block_at(cell) != kind:
            errors.append({"cell": list(cell), "expected": kind,
                           "actual": block_at(cell)})
    for cell in scene.light_cells:
        if block_at(cell) != "glowstone":
            errors.append({"cell": list(cell), "expected": "glowstone",
                           "actual": block_at(cell)})
    for cell in scene.portal_interior_cells:
        if block_at(cell) != "air":
            errors.append({"cell": list(cell), "expected": "air",
                           "actual": block_at(cell)})
    marker_by_cell = {row.cell: row.kind for row in scene.markers}
    prebuilt_by_cell = dict(scene.prebuilt_blocks)
    flat_panel_cells_checked = 0
    approach_clearance_cells_checked = 0
    for panel_id, _kind, center, normal in getattr(
            scene, "panel_assignments", ()):
        tangent = ((1, 0, 0) if normal[2] else (0, 0, 1))
        for h in range(-4, 5):
            for v in range(7):
                backing = (
                    center[0] + tangent[0] * h,
                    scene.feet_y + v,
                    center[2] + tangent[2] * h,
                )
                outward = tuple(backing[i] + normal[i] for i in range(3))
                expected_backing = marker_by_cell.get(
                    backing, scene.wall_block)
                expected_outward = prebuilt_by_cell.get(outward, "air")
                if block_at(backing) != expected_backing:
                    errors.append({
                        "panel_id": panel_id, "cell": list(backing),
                        "expected": expected_backing,
                        "actual": block_at(backing)})
                if block_at(outward) != expected_outward:
                    errors.append({
                        "panel_id": panel_id, "cell": list(outward),
                        "expected": expected_outward,
                        "actual": block_at(outward)})
                flat_panel_cells_checked += 2
        for depth in range(2, CAVE_PANEL_CLEARANCE_MAX_DEPTH + 1):
            for h in range(
                    -CAVE_PANEL_CLEARANCE_HALF_WIDTH,
                    CAVE_PANEL_CLEARANCE_HALF_WIDTH + 1):
                for v in range(CAVE_PANEL_CLEARANCE_HEIGHT):
                    clearance = (
                        center[0] + normal[0] * depth + tangent[0] * h,
                        scene.feet_y + v,
                        center[2] + normal[2] * depth + tangent[2] * h,
                    )
                    if block_at(clearance) != "air":
                        errors.append({
                            "panel_id": panel_id,
                            "cell": list(clearance),
                            "expected": "air",
                            "actual": block_at(clearance),
                            "scope": "panel_approach_clearance",
                        })
                    approach_clearance_cells_checked += 1
    zero_leaks = sorted({
        bare_block(value) for value in relative.values()
        if bare_block(value) in set(scene.zero_shot_markers)})
    if zero_leaks:
        errors.append({"zero_shot_leaks": zero_leaks})
    audit = {
        "contract": "use_human_cave_voxel_audit/v1",
        "accepted": not errors,
        "errors": errors,
        "walkable_interior": [20, 20],
        "queried_non_air_cells": len(relative),
        "marker_count": len(scene.markers),
        "prebuilt_blocks": len(scene.prebuilt_blocks),
        "light_count": len(scene.light_cells),
        "flat_panel_cells_checked": flat_panel_cells_checked,
        "approach_clearance_cells_checked": (
            approach_clearance_cells_checked),
        "zero_shot_leaks": zero_leaks,
    }
    if errors:
        raise RuntimeError(f"staged cave audit failed: {errors[:4]}")
    return audit


def equip_cave_training(
        world, stager: Stager, scene: CaveTrainingScene) -> None:
    stager.kit(f"minecraft:{scene.held_block} 64")
    world.cmd("/effect give @a minecraft:night_vision 999999 0 true")
    action = world.sim.noop_action()
    if "hotbar.1" in action:
        action["hotbar.1"] = 1
    world.obs, _, _, _, world.info = world.sim.step(action)
    if held_item(world) != scene.held_block:
        raise RuntimeError(
            f"held item is {held_item(world)!r}, expected {scene.held_block!r}")


def drain_toasts_camera_off(world, *, maximum_steps: int = 900,
                            required_clean_steps: int = 30,
                            minimum_wall_seconds: float = 10.0) -> dict[str, Any]:
    clean = 0
    steps = 0
    started = time.monotonic()
    for steps in range(1, int(maximum_steps) + 1):
        roi = np.asarray(world.info["pov"])[2:36, 460:638]
        beige = ((roi[..., 0] > 190) & (roi[..., 1] > 180)
                 & (roi[..., 2] > 150) & (roi[..., 2] < 230))
        clean = clean + 1 if float(beige.mean()) < 0.05 else 0
        if (clean >= int(required_clean_steps)
                and time.monotonic() - started >= float(minimum_wall_seconds)):
            break
        world.step_noop()
        time.sleep(0.02)
    if clean < int(required_clean_steps):
        raise RuntimeError("advancement/recipe toast band did not become clean")
    return {
        "contract": "camera_off_top_right_toast_drain/v1",
        "steps": int(steps),
        "required_clean_steps": int(required_clean_steps),
        "minimum_wall_seconds": float(minimum_wall_seconds),
        "wall_seconds": float(time.monotonic() - started),
        "accepted": True,
    }


def capture_visible_target_probe(world, stager: Stager, room: MarkerRoom,
                                 *, episode_dir: Path) -> dict[str, Any]:
    best = None
    for yaw in np.arange(-180, 180, 15):
        if not stager.go(
                room.center_x + 0.5, room.feet_y, room.center_z + 0.5,
                float(yaw), pitch=8.0, n=6):
            continue
        frame = current_frame_artifacts(world, room, completed_ids=set())
        exact = sum(int(row["pixel_count"])
                    for row in frame["row"]["target_instances"])
        score = (int(frame["union"].sum()), exact)
        if best is None or score > best[0]:
            best = (score, float(yaw), frame,
                    np.ascontiguousarray(world.info["pov"], dtype=np.uint8))
    if best is None or best[0][0] <= 0:
        raise RuntimeError("same-seed room has no recognizable target-mask probe")
    score, yaw, _frame, _rgb = best
    if not stager.go(
            room.center_x + 0.5, room.feet_y, room.center_z + 0.5,
            float(yaw), pitch=8.0, n=6):
        raise RuntimeError("could not restore selected target probe pose")
    frame = current_frame_artifacts(
        world, room, completed_ids=set(), audit_per_marker_depth=True)
    live_target_frame = _live_frame_artifacts_from_depth(
        frame["_viewmodel_mask"], frame["_native_depth"],
        frame["row"]["render_pose"], room, completed_ids=set())
    full_stats = _compact_visibility_from_target_instances(
        frame["row"]["target_instances"])
    full_recognizable_ids = [
        instance["instance_id"] for instance in frame["row"]["target_instances"]
        if int(instance["recognizable"]) == 1]
    for field, live_value, full_value in (
            ("live_target_visibility_stats",
             live_target_frame["row"]["live_target_visibility_stats"],
             full_stats),
            ("live_recognizable_target_instance_ids",
             live_target_frame["row"][
                 "live_recognizable_target_instance_ids"],
             full_recognizable_ids),
            *((field, live_target_frame["row"].get(field),
               frame["row"].get(field)) for field in (
                   "crosshair_target_instance_id",
                   "crosshair_target_surface_distance",
                   "crosshair_target_in_reach")),
    ):
        if live_value != full_value:
            raise RuntimeError(
                f"live-compact/deferred-census probe mismatch: {field}")
    rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    audited_score = (
        int(frame["union"].sum()),
        sum(int(row["pixel_count"])
            for row in frame["row"]["target_instances"]),
    )
    if audited_score != score:
        raise RuntimeError(
            f"restored target probe changed score {audited_score}!={score}")
    cv2.imwrite(str(episode_dir / "target_visible_probe_rgb.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(episode_dir / "target_visible_probe_union_overlay.png"),
                cv2.cvtColor(overlay_masks(
                    rgb, frame["union"], np.zeros(MASK_SHAPE, np.uint8)),
                    cv2.COLOR_RGB2BGR))
    census_overlay, census_pixels = overlay_class_census(
        rgb, frame["row"]["class_visible_instances"])
    cv2.imwrite(str(episode_dir / "target_visible_probe_census_overlay.png"),
                cv2.cvtColor(census_overlay, cv2.COLOR_RGB2BGR))
    return {
        "review_only": True,
        "yaw": yaw,
        "pitch": 8.0,
        "recognizable_union_pixel_count": int(score[0]),
        "exact_target_pixel_count": int(score[1]),
        "visible_target_instance_count": len(
            frame["row"]["target_instances"]),
        "rgb": str(episode_dir / "target_visible_probe_rgb.png"),
        "overlay": str(
            episode_dir / "target_visible_probe_union_overlay.png"),
        "census_overlay": str(
            episode_dir / "target_visible_probe_census_overlay.png"),
        "class_visibility_census_contract": (
            USE_CLASS_VISIBILITY_CENSUS_CONTRACT),
        "class_visibility_known_bits": int(
            frame["row"]["class_visibility_known_bits"]),
        "class_visibility_visible_bits": int(
            frame["row"]["class_visibility_visible_bits"]),
        "recognizable_pixels_by_class": census_pixels,
        "depth_owner_index_stats": frame["row"]["depth_owner_index_stats"],
        "live_target_depth_stats": live_target_frame["row"][
            "depth_owner_index_stats"],
        "live_compact_target_vs_deferred_census_exact": True,
        "live_dense_target_mask_materialized": False,
    }


def find_target_absent_start(world, stager: Stager, room: MarkerRoom,
                             *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    yaws = [float(value) for value in np.arange(-180, 180, 15)]
    rng.shuffle(yaws)
    attempts = []
    for yaw in yaws:
        if not stager.go(
                room.center_x + 0.5, room.feet_y, room.center_z + 0.5,
                yaw, pitch=8.0, n=6):
            continue
        frame = current_frame_artifacts(world, room, completed_ids=set())
        exact = sum(int(row["pixel_count"])
                    for row in frame["row"]["target_instances"])
        attempts.append({"yaw": yaw, "exact_target_pixels": exact})
        if exact == 0:
            return {"yaw": yaw, "pitch": 8.0, "attempts": attempts}
    raise RuntimeError("could not find a target-pixel-free start yaw")


def stage_episode(world, stager: Stager, *, world_seed: int, layout_seed: int,
                  target_kind: str,
                  background_split: str, held_block: str | None,
                  palette_index: int | None, train_markers: Sequence[str],
                  episode_dir: Path, scene_style: str = "cave",
                  macro_variant: str | None = None) -> dict[str, Any]:
    if scene_style != "cave":
        raise ValueError(f"unsupported scene style {scene_style!r}")
    stager.prep_world()
    x, y, z = world.get_pos()
    center_x, feet_y, center_z = math.floor(x), math.floor(y), math.floor(z)
    room = build_cave_training_scene(
        seed=layout_seed, center_x=center_x, feet_y=feet_y,
        center_z=center_z, target_kind=target_kind,
        background_split=background_split, held_block=held_block,
        palette_index=palette_index, macro_variant=macro_variant,
        train_markers=train_markers)
    stage_cave_commands(world, room)
    voxel_audit = audit_staged_cave(world, stager, room)
    equip_cave_training(world, stager, room)
    toast_drain = drain_toasts_camera_off(world)
    visible_probe = capture_visible_target_probe(
        world, stager, room, episode_dir=episode_dir)
    start = find_target_absent_start(
        world, stager, room, seed=layout_seed ^ 0x555345)
    first = current_frame_artifacts(world, room, completed_ids=set())
    if first["row"]["target_instances"]:
        raise RuntimeError("target pixels leaked into frame zero")
    first_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    if first_rgb.shape != (360, 640, 3) or not np.any(first_rgb):
        raise RuntimeError("frame zero is malformed or all-black")
    cv2.imwrite(str(episode_dir / "frame0_rgb.png"),
                cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(episode_dir / "frame0_union_overlay.png"),
                cv2.cvtColor(overlay_masks(
                    first_rgb, first["union"], np.zeros(MASK_SHAPE, np.uint8)),
                    cv2.COLOR_RGB2BGR))
    stage = {
        "contract": room.document()["contract"],
        "scene_style": scene_style,
        "portal_playtest": False,
        "training_eligible": room.document().get("training_eligible", True),
        "world_seed": int(world_seed),
        "layout_seed": int(layout_seed),
        "machine": machine_record(),
        "room": room.document(),
        "voxel_audit": voxel_audit,
        "toast_drain": toast_drain,
        "visible_target_probe": visible_probe,
        "start_pose_search": start,
        "frame0_target_exact_pixel_count": 0,
        "frame0_target_absent": True,
        "held_item": room.held_block,
        "camera_off_staging": True,
        "raw_pov_has_no_overlay": True,
        "first_rgb": first_rgb,
        "room_object": room,
    }
    atomic_json(
        episode_dir / "stage.json",
        {key: value for key, value in stage.items()
         if key not in {"first_rgb", "room_object"}})
    return stage


def make_writer(path: Path, size: tuple[int, int], fps: float = FPS):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer {path}")
    return writer


def query_box_for_targets(
        world, room: MarkerRoom, *,
        full_room: bool = False) -> tuple[np.ndarray, tuple[int, int, int]]:
    bx, by, bz = (math.floor(value) for value in world.get_pos())
    if full_room and isinstance(room, CavePortalScene):
        x0, x1, z0, z1 = room.walkable_bounds_xz
        xs = (x0 - 1, x1 + 1)
        ys = (room.feet_y - 1, room.feet_y + 8)
        zs = (z0 - 1, z1 + 1)
    elif full_room:
        r = room.outer_radius - 1
        xs = (room.center_x - r, room.center_x + r)
        ys = (room.feet_y, room.feet_y + room.wall_height)
        zs = (room.center_z - r, room.center_z + r)
    else:
        cells = [row.place_cell for row in room.targets]
        xs, ys, zs = zip(*cells)
    box = np.array([
        min(xs) - bx, max(xs) + 1 - bx,
        min(ys) - by, max(ys) + 1 - by,
        min(zs) - bz, max(zs) + 1 - bz,
    ], dtype=np.int32)
    return box, (bx, by, bz)


def queried_types(info: Mapping[str, Any],
                  origin: tuple[int, int, int]) -> dict[tuple[int, int, int], str]:
    output = {}
    for row in info.get("voxels") or []:
        try:
            raw = (int(row["x"]), int(row["y"]), int(row["z"]))
        except (KeyError, TypeError, ValueError):
            continue
        output[(origin[0] + raw[0], origin[1] + raw[1],
                origin[2] + raw[2])] = bare_block(row.get("type", ""))
    return output


def expected_place_type(held_block: str, value: str) -> bool:
    return bare_block(value) == bare_block(held_block)






def validate_rgb_pose(rgb: np.ndarray, current_pose: Mapping[str, float],
                      previous_y: float | None) -> None:
    if rgb.dtype != np.uint8 or rgb.shape != (360, 640, 3):
        raise RuntimeError(f"training POV changed: {rgb.dtype}{rgb.shape}")
    if not np.any(rgb):
        raise RuntimeError("training POV is all-black")
    if previous_y is not None and abs(float(current_pose["y"]) - previous_y) > 1.5:
        raise RuntimeError("training pose changed by more than 1.5 blocks")


def run_human_episode(world, staged: Mapping[str, Any], *,
                      episode_dir: Path, max_seconds: float,
                      start_permit_file: Path | None = None,
                      status_file: Path | None = None) -> dict[str, Any]:
    room: MarkerRoom = staged["room_object"]
    quota = len(room.targets)
    target_name = room.target_kind.replace("_", " ").upper()
    mission = (f"USE: place held block on all {target_name} markers "
               f"(0 / {quota})")
    window_x = os.environ.get("XBENCH_HUMAN_WINDOW_X")
    window_y = os.environ.get("XBENCH_HUMAN_WINDOW_Y")
    window_label = os.environ.get("XBENCH_HUMAN_WINDOW_LABEL")
    positioned = window_x is not None and window_y is not None
    ui = UseHumanUI(scale=1 if positioned else 2, mouse_sens=0.15,
                    prevent_initial_focus=positioned,
                    window_x=(None if window_x is None else int(window_x)),
                    window_y=(None if window_y is None else int(window_y)),
                    caption=(window_label or
                             f"USE HumanPlay — held block on {room.target_kind}"))
    writer = make_writer(episode_dir / "raw_pov.mp4", FRAME_SIZE)
    rows = []
    deferred_label_frames: list[tuple[np.ndarray, np.ndarray]] = []
    placements = []
    completed: set[str] = set()
    target_group_by_instance = {
        row.instance_id: str(row.pattern_index) for row in room.targets}
    target_group_members: dict[str, set[str]] = {}
    for identity, group_id in target_group_by_instance.items():
        target_group_members.setdefault(group_id, set()).add(identity)
    active_placement_group: str | None = None
    invalid_reason = None
    started = False
    last_use = normalized_counter(world.info.get("use_item", {}))
    announced: set[str] = set()
    flash_until = -1
    flash_message = None
    previous_y = None
    try:
        write_live_status(
            status_file, "ready", target_kind=room.target_kind,
            held_block=room.held_block, episode_dir=str(episode_dir),
            start_permit_file=(None if start_permit_file is None else
                               str(start_permit_file)))
        gate = ui.wait_for_start(
            staged["first_rgb"], mission,
            start_allowed=(
                None if start_permit_file is None else
                lambda: Path(start_permit_file).is_file()))
        if not gate.get("started"):
            return {
                "valid": False,
                "reason": gate.get("reason", "start_aborted"),
                "rows": rows, "placements": placements,
            }
        started = True
        write_live_status(
            status_file, "playing", target_kind=room.target_kind,
            held_block=room.held_block, episode_dir=str(episode_dir),
            ready_wait_s=gate.get("wait_s"))
        started_at = time.monotonic()
        deadline = started_at + float(max_seconds)
        next_frame_at = started_at
        render_position = tuple(float(value) for value in world.get_pos())
        while len(completed) < quota and time.monotonic() < deadline:
            frame_index = len(rows)
            ui.pump()
            if ui.abort or ui.quit:
                invalid_reason = "window_quit" if ui.quit else "human_abort"
                break
            frame = current_live_frame_artifacts(
                world, room, completed_ids=completed,
                render_position=render_position)
            rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
            current_pose = frame["row"]["render_pose"]
            validate_rgb_pose(rgb, current_pose, previous_y)
            previous_y = float(current_pose["y"])
            ui.live_union_mask = None
            if ui.consume_pause_marker():
                pause_ui(ui, rgb, mission, None)
                if ui.abort or ui.quit:
                    invalid_reason = "human_abort_during_pause"
                    break
                continue
            visible = set(
                str(value) for value in frame["row"][
                    "live_recognizable_target_instance_ids"])
            new_visible = sorted(visible - announced)
            if new_visible:
                announced.update(new_visible)
                flash_until = frame_index + 24
                flash_message = "TARGET FOUND"
                ui.play_completion_sound()
            mission = (
                f"CAVE: place {room.held_block.upper()} on all "
                f"{target_name} "
                f"markers ({len(completed)} / {quota})")
            sub = (
                f"interaction_id=3 (USE) | exact placements "
                f"{len(completed)}/{quota}")
            ui.draw(
                rgb, mission, sub,
                flash=(flash_message if frame_index <= flash_until else None),
                flash_color=(80, 255, 80, 255))
            action = ui.poll_action(world.sim.noop_action())
            requested_use = bool(
                np.asarray(action.get("use", 0)).reshape(-1)[0])
            crosshair_identity = frame["row"][
                "crosshair_target_instance_id"]
            crosshair_group = target_group_by_instance.get(
                str(crosshair_identity)) if crosshair_identity is not None else None
            suppress_group_switch_use = bool(
                requested_use and active_placement_group is not None
                and crosshair_group is not None
                and crosshair_group != active_placement_group)
            suppress_out_of_reach_use = bool(
                requested_use
                and frame["row"]["crosshair_target_instance_id"] is not None
                and frame["row"]["crosshair_target_in_reach"] != 1)
            if suppress_group_switch_use:
                action["use"] = 0
                flash_until = frame_index + 24
                flash_message = "FINISH CURRENT GROUP BEFORE STARTING ANOTHER"
            elif suppress_out_of_reach_use:
                action["use"] = 0
                flash_until = frame_index + 24
                flash_message = "MOVE CLOSER  |  TARGET MUST BE WITHIN 4.5 BLOCKS"
            use = bool(
                np.asarray(action.get("use", 0)).reshape(-1)[0])
            actual_held = held_item(world)
            if actual_held != room.held_block:
                invalid_reason = f"held_item_drift:{held_item(world)}"
                break
            query_box, _query_origin_before_action = query_box_for_targets(
                world, room, full_room=use)
            action["voxels"] = query_box
            row = {
                "contract": FRAME_CONTRACT,
                "f": frame_index,
                "t": frame_index,
                "goal_kind": room.target_kind,
                "target_kind": room.target_kind,
                "mode": room.mode,
                "interaction_id": int(USE_INTERACTION_ID),
                "obj_id": int(USE_INTERACTION_IDS["obj_id"]),
                "verb_id": int(USE_INTERACTION_IDS["verb_id"]),
                "tool_id": int(MODE_TOOL_ID[room.mode]),
                "interaction_ids": {
                    **USE_INTERACTION_IDS,
                    "tool_id": int(MODE_TOOL_ID[room.mode]),
                },
                "pose": current_pose,
                "action": ser_action(action),
                "use": int(use),
                "user_requested_use": int(requested_use),
                "out_of_reach_target_use_suppressed": int(
                    suppress_out_of_reach_use),
                "cross_group_target_use_suppressed": int(
                    suppress_group_switch_use),
                "held_item": actual_held,
                "playtest_phase": "place_targets",
                "completed_instance_ids_before_action": sorted(completed),
                "wall_elapsed_s": float(time.monotonic() - started_at),
                "use_item_counter_before_action": dict(last_use),
                "new_recognizable_target_instance_ids": new_visible,
                "ui_recognizable_alert": int(bool(new_visible)),
                **frame["row"],
            }
            rows.append(row)
            deferred_label_frames.append((
                np.ascontiguousarray(
                    frame["_viewmodel_mask"], dtype=np.uint8).copy(),
                np.ascontiguousarray(frame["_native_depth"]).copy(),
            ))
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            pre_action_position = tuple(
                float(value) for value in world.get_pos())
            world.obs, _, terminated, truncated, world.info = world.sim.step(action)
            render_position = pre_action_position
            if terminated or truncated:
                invalid_reason = "sim_terminated_or_truncated"
                break

            current_use = normalized_counter(world.info.get("use_item", {}))
            use_delta = positive_counter_delta(current_use, last_use)
            last_use = current_use
            row["use_item_delta_after_action"] = dict(use_delta)
            query_origin = tuple(
                math.floor(value) for value in world.get_pos())
            query = queried_types(world.info, query_origin)
            expected_locations = {row.place_cell for row in room.targets}
            allowed_existing_held_locations = {
                cell for cell, kind in getattr(room, "prebuilt_blocks", ())
                if expected_place_type(room.held_block, kind)}
            unexpected_placed = sorted(
                [cell for cell, value in query.items()
                 if expected_place_type(room.held_block, value)
                 and cell not in expected_locations
                 and cell not in allowed_existing_held_locations])
            if unexpected_placed:
                invalid_reason = (
                    f"wrong_or_off_target_placement_cells:{unexpected_placed[:8]}")
                break
            changed = []
            for marker in room.targets:
                if marker.instance_id in completed:
                    continue
                actual = query.get(marker.place_cell, "air")
                if expected_place_type(room.held_block, actual):
                    changed.append((marker, actual))
            if len(changed) > 1:
                invalid_reason = "multiple_expected_cells_changed_in_one_step"
                break
            held_delta = int(use_delta.get(room.held_block, 0))
            if changed:
                marker, actual_type = changed[0]
                exact_pre = bool(
                    frame["row"]["crosshair_target_instance_id"]
                    == marker.instance_id
                    and frame["row"]["crosshair_target_in_reach"] == 1)
                if not use or not exact_pre:
                    invalid_reason = (
                        "target_cell_changed_without_exact_use_transaction:"
                        f"use={int(use)}:delta={held_delta}:exact={int(exact_pre)}")
                    break
                marker_group = target_group_by_instance[marker.instance_id]
                if (active_placement_group is not None
                        and marker_group != active_placement_group):
                    invalid_reason = (
                        "interleaved_target_groups:active="
                        f"{active_placement_group}:clicked={marker_group}")
                    break
                if active_placement_group is None:
                    active_placement_group = marker_group
                placements.append({
                    "f": frame_index,
                    "instance_id": marker.instance_id,
                    "interaction_group_id": marker_group,
                    "marker_cell": list(marker.cell),
                    "place_cell": list(marker.place_cell),
                    "placed_block": bare_block(actual_type),
                    "crosshair_precondition": "exact_marker_cell_in_reach",
                    "crosshair_surface_distance": frame["row"][
                        "crosshair_target_surface_distance"],
                    "postcondition": "exact_expected_cell_changed",
                    "use_item_delta": held_delta,
                    "use_item_delta_required_for_success": False,
                })
                completed.add(marker.instance_id)
                if target_group_members[marker_group] <= completed:
                    active_placement_group = None
                announced.clear()
                flash_until = frame_index + 18
                flash_message = f"EXACT SUCCESS {len(completed)} / {quota}"
                ui.play_completion_sound()

            next_frame_at += 1.0 / FPS
            delay = next_frame_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_frame_at = time.monotonic()
        if invalid_reason is None:
            if len(completed) != quota:
                invalid_reason = "timeout_before_all_exact_placements"
    finally:
        writer.release()
        ui.close()

    if not started or invalid_reason is not None:
        return {
            "valid": False,
            "reason": invalid_reason or "not_started",
            "rows": rows,
            "placements": placements,
            "completed_instance_ids": sorted(completed),
            "portal_active": False,
            "training_eligible": True,
        }
    write_live_status(
        status_file, "postprocessing", target_kind=room.target_kind,
        held_block=room.held_block, episode_dir=str(episode_dir),
        raw_frame_count=len(rows), postprocess_stage="deferred_train7_census")
    deferred_census = apply_deferred_class_census(
        rows, deferred_label_frames, room, diagnostic_dir=episode_dir)
    labels = derive_use_training_labels(
        rows, placements,
        target_instance_ids=[row.instance_id for row in room.targets],
        target_group_by_instance=target_group_by_instance)
    return {
        "valid": True, "reason": None,
        "rows": rows, "placements": placements,
        "labels": labels,
        "deferred_census": deferred_census,
        "completed_instance_ids": sorted(completed),
    }




def render_review(episode_dir: Path, rows: Sequence[Mapping[str, Any]], *,
                  room: MarkerRoom) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(episode_dir / "raw_pov.mp4"))
    review_path = episode_dir / "review_union_chosen_phase.mp4"
    writer = make_writer(review_path, (640, 430))
    contact = []
    selected = {0, len(rows) - 1}
    selected.update(int(row["f"]) for row in rows
                    if row.get("binding_event")
                    or row.get("interaction_exact_success"))
    selected.update(
        min(len(rows) - 1, int(row["f"]) + 1)
        for row in rows if row.get("interaction_exact_success"))
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            row = rows[index]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            union = decode_mask_runs_yx(
                list(row.get("union_mask_runs_yx") or []), list(MASK_SHAPE))
            chosen = decode_mask_runs_yx(
                list(row.get("chosen_mask_runs_yx") or []), list(MASK_SHAPE))
            overlay = overlay_masks(rgb, union, chosen)
            canvas = np.zeros((430, 640, 3), np.uint8)
            canvas[:360] = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            line1 = (
                f"f={index} goal={room.target_kind} mode={room.mode} "
                f"USE obj=3 verb=3 phase={row['phase_name']}")
            line2 = (
                f"union={row['union_mask_pixel_count']} "
                f"chosen={row['chosen_mask_pixel_count']} "
                f"commit={row['target_committed']} "
                f"id={row.get('chosen_instance_id')} use={row.get('use', 0)}")
            cv2.putText(canvas, line1, (12, 386), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, line2, (12, 412), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (180, 255, 180), 1, cv2.LINE_AA)
            writer.write(canvas)
            if index in selected:
                path = episode_dir / f"review_f{index:05d}.png"
                cv2.imwrite(str(path), canvas)
                contact.append(cv2.resize(canvas, (480, 323)))
            index += 1
    finally:
        capture.release()
        writer.release()
    if index != len(rows):
        raise RuntimeError(f"raw/review frame mismatch {index}!={len(rows)}")
    if contact:
        cv2.imwrite(
            str(episode_dir / "review_keyframes_contact_sheet.png"),
            np.vstack(contact))
    return {
        "path": str(review_path),
        "frames": index,
        "keyframes": sorted(selected),
        "raw_training_pov_unchanged": True,
    }


def finalize_episode(episode_dir: Path, result: Mapping[str, Any], *,
                     staged: Mapping[str, Any]) -> dict[str, Any]:
    room: MarkerRoom = staged["room_object"]
    raw_rows = list(result["rows"])
    write_jsonl(episode_dir / "raw_frames.jsonl", raw_rows)
    write_jsonl(episode_dir / "raw_traj.jsonl", [
        {"f": row["f"], "action": row["action"], "pose": row["pose"],
         "interaction_id": row["interaction_id"],
         "obj_id": row["obj_id"], "verb_id": row["verb_id"],
         "tool_id": row["tool_id"]}
        for row in raw_rows])
    meta = {
        "contract": CAMPAIGN_CONTRACT,
        "valid": bool(result["valid"]),
        "reason": result.get("reason"),
        "world_seed": staged["world_seed"],
        "layout_seed": staged["layout_seed"],
        "scene_style": staged.get("scene_style", "room"),
        "target_kind": room.target_kind,
        "mode": room.mode,
        "held_block": room.held_block,
        "pattern": room.pattern,
        "target_quota": len(room.targets),
        "target_surface_counts": room.document()["target_surface_counts"],
        "interaction_id": USE_INTERACTION_ID,
        "interaction_id_compat_aliases": dict(USE_INTERACTION_IDS),
        "interaction_ids": {
            **USE_INTERACTION_IDS,
            "tool_id": MODE_TOOL_ID[room.mode],
        },
        "terminal_policy": MODE_TERMINAL_POLICY[room.mode],
        "terminal_tail_frames_recorded": 0,
        "raw_frame_count": len(raw_rows),
        "placements": result.get("placements"),
        "completed_instance_ids": result.get("completed_instance_ids"),
        "training_rgb": "raw_pov.mp4_clean_no_overlay",
        "live_target_mask_overlay": False,
        "live_target_supervision": (
            "exact_compact_recognition_and_crosshair_only"),
        "dense_target_mask_timing": "deferred_after_human_input",
        "deferred_census": result.get("deferred_census"),
    }
    if not result["valid"]:
        atomic_json(episode_dir / "meta.json", meta)
        return meta
    labels = result["labels"]
    rows = labels["rows"]
    header = {
        "version": 1,
        "contract": STORY_CONTRACT,
        "label_contract": USE_HUMAN_LABEL_CONTRACT,
        "episode_source": "human_play",
        "cell": f"place_{room.mode}",
        "biome": (
            "command_built_natural_cave"
            if isinstance(room, CaveTrainingScene)
            else "command_built_marker_room"),
        "target_kind": room.target_kind,
        "goal_kind": room.target_kind,
        "world_seed": staged["world_seed"],
        "layout_seed": staged["layout_seed"],
        "train_classes": list(room.train_markers),
        "zero_shot_classes": list(room.zero_shot_markers),
        "zero_shot_excluded_from_train": True,
        "interaction_id": USE_INTERACTION_ID,
        "interaction_id_compat_aliases": dict(USE_INTERACTION_IDS),
        "interaction_ids": {
            **USE_INTERACTION_IDS,
            "tool_id": MODE_TOOL_ID[room.mode],
        },
        "obj_id": USE_INTERACTION_IDS["obj_id"],
        "verb_id": USE_INTERACTION_IDS["verb_id"],
        "tool_id": MODE_TOOL_ID[room.mode],
        "held_item": room.held_block,
        "room": room.document(),
        "stage_audit": {
            key: value for key, value in staged.items()
            if key not in {"first_rgb", "room_object"}},
        "decision_events": [
            "commit", "switch", "seam", "seam_commit",
            "exact_success_clear"],
        "phase_semantics": (
            "EXPLORE before hindsight binding; APPROACH until a spatial "
            "marker group's first exact success; INTERACT from that first "
            "success through the same group's final exact success inclusive; "
            "then seam back to APPROACH/EXPLORE for another group"),
        "pov_storage": "sibling_raw_pov.mp4",
        "raw_pov_has_no_overlay": True,
        "live_target_mask_overlay": False,
        "live_target_supervision": (
            "exact_compact_recognition_and_crosshair_only"),
        "dense_target_mask_timing": "deferred_after_human_input",
        "inventory_and_selected_slot_preconfigured": True,
        "episode_hotbar_actions_allowed": False,
        "class_visibility_census_contract": (
            USE_CLASS_VISIBILITY_CENSUS_CONTRACT),
        "class_visibility_roster_contract": (
            USE_CLASS_VISIBILITY_ROSTER_CONTRACT),
        "all_train_class_depth_census": True,
        "deferred_census": result.get("deferred_census"),
    }
    write_jsonl(episode_dir / "story.jsonl", [header, *rows])
    write_jsonl(episode_dir / "traj.jsonl", [
        {"f": row["f"], "action": row["action"], "pose": row["pose"],
         "interaction_id": row["interaction_id"],
         "obj_id": row["obj_id"], "verb_id": row["verb_id"],
         "tool_id": row["tool_id"], "bc_valid": row["bc_valid"],
         "phase": row["phase"], "chosen_instance_id": row[
             "chosen_instance_id"],
         "interaction_group_id": row["interaction_group_id"]}
        for row in rows])
    meta.update({
        "phase_counts": labels["phase_counts"],
        "binding_events": labels["binding_events"],
        "group_binding_advances": labels["group_binding_advances"],
        "target_group_by_instance_id": labels[
            "target_group_by_instance_id"],
        "group_intervals_inclusive": labels[
            "group_intervals_inclusive"],
        "all_targets_completed": labels["all_targets_completed"],
        "story": str(episode_dir / "story.jsonl"),
    })
    review = render_review(episode_dir, rows, room=room)
    meta["review"] = review
    meta["raw_video_frame_count"] = int(review["frames"])
    meta["review_frame_count"] = int(review["frames"])
    meta["raw_review_frame_alignment"] = bool(
        int(review["frames"]) == len(raw_rows) == len(rows))
    atomic_json(episode_dir / "meta.json", meta)
    return meta




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--world-seed", type=int, default=910000)
    parser.add_argument("--layout-seed", type=int, default=920000)
    parser.add_argument("--scene-style", choices=("cave",), default="cave")
    parser.add_argument("--target-kind", default="oak_log")
    parser.add_argument("--palette-index", type=int, default=None)
    parser.add_argument("--macro-variant", choices=CAVE_MACRO_VARIANTS)
    parser.add_argument("--held-block", choices=HELD_BLOCK_ROSTER, default=None)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--human-status-file", type=Path)
    parser.add_argument("--human-start-permit-file", type=Path)
    return parser.parse_args()


def boot(seed: int, *, maximum_attempts: int = 3):
    last = None
    for attempt in range(1, int(maximum_attempts) + 1):
        try:
            return boot_world(
                int(seed), biome="plains", action_type="env",
                runtime_overlay=REPO / "configs" / "xbench_runtime_dense")
        except BaseException as exc:
            last = exc
            retryable = bool(
                isinstance(exc, BrokenPipeError)
                or "BrokenPipe" in repr(exc)
                or "Connection refused" in repr(exc)
                or "empty reply from Malmo" in repr(exc))
            if not retryable or attempt >= int(maximum_attempts):
                raise
            print(
                f"[use-human] connection error on boot attempt "
                f"{attempt}/{maximum_attempts} for world seed={seed}; restarting: "
                f"{type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"simulator boot did not complete: {last}")


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output root: {args.out}")
    args.out.mkdir(parents=True)
    episode_dir = args.out / (
        f"episode_00_{args.scene_style}_{PLACE_MODE}_"
        f"{bare_block(args.target_kind)}_"
        f"w{args.world_seed}_l{args.layout_seed}")
    episode_dir.mkdir()
    train_markers = tuple(bare_block(value) for value in DEFAULT_TRAIN_MARKERS)
    atomic_json(args.out / "run.json", {
        "contract": CAMPAIGN_CONTRACT,
        "machine": machine_record(),
        "world_seed": args.world_seed,
        "layout_seed": args.layout_seed,
        "scene_style": args.scene_style,
        "playtest_only": False,
        "training_eligible": True,
        "target_kind": bare_block(args.target_kind),
        "mode": PLACE_MODE,
        "pattern_request": "mixed",
        "background_split": "train",
        "palette_index_request": (
            args.palette_index if args.palette_index is not None
            else "layout_seed_sampled"),
        "macro_variant_request": args.macro_variant or "layout_seed_sampled",
        "held_block_request": args.held_block or "layout_seed_sampled",
        "train_markers": list(train_markers),
        "zero_shot_markers": list(ZERO_SHOT_MARKERS),
        "zero_shot_excluded_from_train": True,
        "command": sys.argv,
    })
    install_renderer_viewmodel_observation()
    world = None
    try:
        write_live_status(
            args.human_status_file, "booting",
            campaign_root=str(args.out.resolve()),
            target_kind=bare_block(args.target_kind),
            held_block_request=args.held_block or "layout_seed_sampled")
        world = boot(args.world_seed)
        stager = Stager(world)
        write_live_status(
            args.human_status_file, "staging",
            campaign_root=str(args.out.resolve()),
            target_kind=bare_block(args.target_kind),
            world_seed=args.world_seed, layout_seed=args.layout_seed)
        staged = stage_episode(
            world, stager, world_seed=args.world_seed,
            layout_seed=args.layout_seed,
            target_kind=bare_block(args.target_kind),
            background_split="train",
            held_block=args.held_block, palette_index=args.palette_index,
            train_markers=train_markers,
            episode_dir=episode_dir, scene_style=args.scene_style,
            macro_variant=args.macro_variant)
        print(
            f"[use-human] READY: activate the window and press V; "
            f"mode={PLACE_MODE} target={bare_block(args.target_kind)} "
            f"quota={len(staged['room_object'].targets)}", flush=True)
        result = run_human_episode(
            world, staged, episode_dir=episode_dir,
            max_seconds=args.max_seconds,
            start_permit_file=args.human_start_permit_file,
            status_file=args.human_status_file)
        write_live_status(
            args.human_status_file, "postprocessing",
            campaign_root=str(args.out.resolve()),
            episode_dir=str(episode_dir), valid=bool(result.get("valid")),
            reason=result.get("reason"),
            raw_frame_count=len(result.get("rows") or ()))
        meta = finalize_episode(episode_dir, result, staged=staged)
        if meta["valid"]:
            atomic_json(args.out / "release_manifest.json", {
                "contract": "use_human_release_manifest/v1",
                "episodes": [{
                    "name": episode_dir.name,
                    "episode_dir": str(episode_dir.resolve()),
                    "target_kind": staged["room_object"].target_kind,
                    "mode": staged["room_object"].mode,
                }],
            })
        atomic_json(args.out / "COMPLETE.json", {
            "valid": bool(meta["valid"]),
            "reason": meta.get("reason"),
            "episode_dir": str(episode_dir),
            "review": (meta.get("review") or {}).get("path"),
        })
        write_live_status(
            args.human_status_file,
            "completed" if meta["valid"] else "failed",
            campaign_root=str(args.out.resolve()),
            episode_dir=str(episode_dir), valid=bool(meta["valid"]),
            reason=meta.get("reason"),
            complete=str(args.out / "COMPLETE.json"))
        print(json.dumps(json_safe(meta), indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        write_live_status(
            args.human_status_file, "failed",
            campaign_root=str(args.out.resolve()),
            episode_dir=str(episode_dir), valid=False,
            reason=f"exception:{type(exc).__name__}:{exc}")
        raise
    finally:
        if world is not None:
            try:
                world.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
