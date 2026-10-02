#!/usr/bin/env python3
"""Place benchmark: in a cave scene with one target marker group and six confuser marker panels, the agent must complete the target group with its held block, given a goal image and mask."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO) + "/src")
if str(REPO / "src/attacca/evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "src/attacca/evaluation"))

os.environ.setdefault("MINESTUDIO_DIR", str(REPO / ".minestudio"))
os.environ.setdefault("MINESTUDIO_GPU_RENDER", "1")
os.environ.setdefault("RENDER_DEVICES", "0")

from attacca.evaluation.policy import Rocket2GoalRunner
from attacca.evaluation.policy import boot_world
from attacca.evaluation.place_scene import _owner_pixels_from_depth
from attacca.evaluation.place_scene import aabb_distance
from attacca.evaluation.place_scene import audit_staged_cave
from attacca.evaluation.place_scene import drain_toasts_camera_off
from attacca.evaluation.place_scene import equip_cave_training
from attacca.evaluation.place_scene import expected_place_type
from attacca.evaluation.place_scene import held_item
from attacca.evaluation.place_scene import machine_record
from attacca.evaluation.place_scene import overlay_masks
from attacca.evaluation.place_scene import pose
from attacca.evaluation.place_scene import queried_types
from attacca.evaluation.place_scene import query_box_for_targets
from attacca.evaluation.place_scene import recognizable
from attacca.evaluation.place_scene import stage_cave_commands
from attacca.evaluation.place_scene import strict_depth_arrays
from attacca.evaluation.mine_scene import Stager
from attacca.worlds.human_viewmodel import encode_mask_runs_yx
from attacca.worlds.human_viewmodel import install_renderer_viewmodel_observation
from attacca.worlds.eval_tail import POST_SUCCESS_FRAMES
from attacca.worlds.eval_tail import post_success_frames_recorded
from attacca.worlds.eval_tail import post_success_tail_complete
from attacca.worlds.use_human_scene import CAVE_TRAINING_CONTRACT
from attacca.worlds.use_human_scene import DEFAULT_TRAIN_MARKERS
from attacca.worlds.use_human_scene import HELD_BLOCK_ROSTER
from attacca.worlds.use_human_scene import MODE_TOOL_ID
from attacca.worlds.use_human_scene import USE_INTERACTION_ID
from attacca.worlds.use_human_scene import USE_INTERACTION_IDS
from attacca.worlds.use_human_scene import USE_REACH_BLOCKS
from attacca.worlds.use_human_scene import ZERO_SHOT_MARKERS
from attacca.worlds.use_human_scene import CaveTrainingScene
from attacca.worlds.use_human_scene import Marker
from attacca.worlds.use_human_scene import bare_block
from attacca.worlds.use_human_scene import build_cave_training_scene
from attacca.worlds.use_human_scene import stable_sha256


CONTRACT = "xbench_place_cave_counterfactual_benchmark/v8_fixedconf"
SCENE_CONTRACT = "xbench_place_cave_eval_scene/v5_fixedconf"
GEOMETRY_SENTINEL = "__geometry__"
GOAL_LIBRARY_CONTRACT_V1 = (
    "xbench_place_10class_plains_grass_front_goal_library/v1")
EPISODE_CONTRACT = "xbench_place_cave_eval_episode/v4"
FRAME_CONTRACT = "xbench_place_cave_eval_frame/v3"
DEFERRED_REVIEW_PAYLOAD_CONTRACT = "xbench_deferred_eval_review_payload/v1"
LAYOUT_CONTRACT = "xbench_place_cave_eval_layouts/v1"
PLACE_SNAPSHOT_PAYLOAD_CONTRACT = "xbench_place_frozen_scene_snapshot/v2"
PLACE_SNAPSHOT_BANK_CONTRACT = "xbench_place_snapshot_bank/v2"
GOAL_MODE = "plains_grass_front"
TRAIN_TARGETS = tuple(DEFAULT_TRAIN_MARKERS)
OOD_TARGETS_A = (
    "blue_ice", "red_sandstone", "gilded_blackstone",
)
OOD_TARGETS_B = (
    "chiseled_nether_bricks",
    "green_terracotta",
    "red_nether_bricks",
    "netherite_block",
    "blue_glazed_terracotta",
)
OOD_TARGETS_C = (
    "smooth_quartz",
    "dead_brain_coral_block",
    "gray_terracotta",
    "orange_concrete",
    "lime_concrete",
    "cyan_concrete",
    "pink_wool",
    "cyan_wool",
)
PLACE_EVAL_CONFUSERS = (
    "netherite_block",
    "smooth_quartz",
    "chiseled_nether_bricks",
    "gray_terracotta",
    "blue_glazed_terracotta",
)
OOD_TARGETS = tuple(
    t for t in (*OOD_TARGETS_A, *OOD_TARGETS_B,
                *OOD_TARGETS_C)
    if t not in PLACE_EVAL_CONFUSERS)
TARGETS = (*TRAIN_TARGETS, *OOD_TARGETS)
if set(PLACE_EVAL_CONFUSERS) & set(TARGETS):
    raise RuntimeError("place confuser roster may never overlap eval targets")
WORLD_COUNT = 10
START_COUNT = 1
WORLD_SEED_BASE = 1_500_000

EVAL_WORLD_SPECS = (
    {"palette": ("stone", "andesite", "cobblestone", "gravel"),
     "macro": "oval_even", "held": "black_wool",
     "quota": 3, "orient": "V"},
    {"palette": ("sandstone", "smooth_sandstone", "cut_sandstone", "gravel"),
     "macro": "oval_low_shoulders", "held": "blue_wool",
     "quota": 2, "orient": "H", "h_rel_y": 0},
    {"palette": ("granite", "polished_granite", "stone", "gravel"),
     "macro": "oval_rocky_floor", "held": "yellow_wool",
     "quota": 3, "orient": "H", "h_rel_y": 1},
    {"palette": ("diorite", "polished_diorite", "stone", "gravel"),
     "macro": "oval_dense_stalactites", "held": "purpur_block",
     "quota": 2, "orient": "V"},
    {"palette": ("andesite", "polished_andesite", "stone", "gravel"),
     "macro": "oval_even", "held": "lime_wool",
     "quota": 3, "orient": "V"},
    {"palette": ("end_stone", "end_stone_bricks", "stone", "gravel"),
     "macro": "oval_low_shoulders", "held": "obsidian",
     "quota": 2, "orient": "H", "h_rel_y": 2},
    {"palette": ("basalt", "polished_basalt", "blackstone", "gravel"),
     "macro": "oval_rocky_floor", "held": "red_wool",
     "quota": 3, "orient": "H", "h_rel_y": 1},
    {"palette": ("blackstone", "polished_blackstone", "basalt", "gravel"),
     "macro": "oval_dense_stalactites", "held": "magenta_wool",
     "quota": 2, "orient": "V"},
    {"palette": ("prismarine", "dark_prismarine", "stone", "gravel"),
     "macro": "oval_even", "held": "honeycomb_block",
     "quota": 3, "orient": "V"},
    {"palette": ("soul_soil", "basalt", "blackstone", "gravel"),
     "macro": "oval_low_shoulders", "held": "purple_wool",
     "quota": 2, "orient": "H", "h_rel_y": 0},
)
LAYOUT_SEED_BASE = 1_600_000
POLICY_SEED_BASE = 1_700_000
FPS = 20.0
DEFAULT_BUDGET = 600
DEFAULT_POST_SUCCESS_FRAMES = POST_SUCCESS_FRAMES
FRAME_SHAPE = (360, 640)
def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def checkpoint_digest(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(
                value for value in path.rglob("*") if value.is_file()):
            digest.update(str(child.relative_to(path)).encode())
            digest.update(b"\0")
            digest.update(sha256_file(child).encode())
            digest.update(b"\0")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def json_safe(value: Any) -> Any:
    if (value.__class__.__module__.startswith("torch")
            and hasattr(value, "detach") and hasattr(value, "cpu")):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_layouts() -> list[dict[str, Any]]:
    layouts = []
    for world_index in range(WORLD_COUNT):
        layouts.append({
            "contract": LAYOUT_CONTRACT,
            "world_index": world_index,
            "world_seed": WORLD_SEED_BASE + world_index,
            "layout_seed": LAYOUT_SEED_BASE + 7_919 * world_index,
            "start_ids": list(range(START_COUNT)),
            "policy_seeds": [
                POLICY_SEED_BASE + world_index * START_COUNT + start_id
                for start_id in range(START_COUNT)],
        })
    return layouts


def episode_seed(namespace: str, target: str, world_seed: int,
                 action_seed: int) -> int:
    payload = (
        f"{namespace}\0{bare_block(target)}\0{int(world_seed)}\0"
        f"{int(action_seed)}")
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


TARGET_PANEL_PLACEHOLDER = "oak_log"


def omitted_train_kind(target: str, world_index: int) -> str | None:
    del world_index
    if bare_block(target) not in TARGETS:
        raise ValueError(f"unsupported Place eval target {target!r}")
    return None


def scene_roster(target: str, world_index: int) -> tuple[str, ...]:
    del world_index
    target = bare_block(target)
    if target not in TARGETS:
        raise ValueError(f"unsupported Place eval target {target!r}")
    return (target, *PLACE_EVAL_CONFUSERS)


def scene_exclusion_roster(target: str,
                           roster: Sequence[str]) -> tuple[str, ...]:
    excluded = {
        *(bare_block(value) for value in ZERO_SHOT_MARKERS),
        *OOD_TARGETS,
    }
    excluded.discard(bare_block(target))
    excluded.difference_update(bare_block(value) for value in roster)
    return tuple(sorted(excluded))


def eval_scene_document(scene: CaveTrainingScene, *, target_split: str,
                        world_index: int) -> dict[str, Any]:
    source = scene.document()
    source_sha = source.pop("scene_sha256")
    source.update({
        "contract": SCENE_CONTRACT,
        "training_eligible": False,
        "evaluation_only": True,
        "source_geometry_contract": CAVE_TRAINING_CONTRACT,
        "source_geometry_scene_sha256": source_sha,
        "eval_targets": list(TARGETS),
        "target_split": str(target_split),
        "scene_marker_roster": list(scene.train_markers),
        "geometry_class_neutral": True,
        "fixed_confuser_roster": list(PLACE_EVAL_CONFUSERS),
        "eval_world_spec": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in EVAL_WORLD_SPECS[int(world_index)].items()},
        "eval_palette_overrides_source_palette_index": True,
        "omitted_train_confuser": omitted_train_kind(
            scene.target_kind, world_index),
        "confuser_policy": (
            "fixed_confusers: world geometry AND confuser paint are "
            "class-neutral and shared by every target, ID and OOD alike "
            "(five never-target confuser classes, one panel each); the only "
            "per-episode difference is the paint of the target cells"),
        "target_group_count": len({
            marker.pattern_index for marker in scene.targets}),
        "target_group_size": len(scene.targets),
        "target_group_sizes_allowed": [2, 3],
        "target_marker_max_relative_y": 2,
        "vertical_target_rows_relative_y": [0, 1, 2],
        "target_count": len(scene.targets),
        "confuser_count": len(scene.confusers),
        "strict_success_target_faces": len(scene.targets),
    })
    source["scene_sha256"] = stable_sha256(source)
    return source


def build_eval_scene(*, seed: int, center_x: int, feet_y: int,
                     center_z: int, target: str,
                     world_index: int) -> CaveTrainingScene:
    target = bare_block(target)
    if target not in TARGETS:
        raise ValueError(f"unsupported Place eval target {target!r}")
    spec = EVAL_WORLD_SPECS[int(world_index)]
    desired_size = int(spec["quota"])
    want_vertical = spec["orient"] == "V"
    for attempt in range(256):
        attempt_seed = episode_seed(
            "place_scene_geometry", GEOMETRY_SENTINEL,
            WORLD_SEED_BASE + int(world_index), int(seed) + attempt)
        build_roster = (TARGET_PANEL_PLACEHOLDER, *PLACE_EVAL_CONFUSERS)
        source = build_cave_training_scene(
            seed=attempt_seed, center_x=int(center_x), feet_y=int(feet_y),
            center_z=int(center_z), target_kind=TARGET_PANEL_PLACEHOLDER,
            background_split="eval", train_markers=build_roster,
            held_block=str(spec["held"]), palette_index=0,
            macro_variant=str(spec["macro"]),
            zero_shot_markers=scene_exclusion_roster(
                TARGET_PANEL_PLACEHOLDER, build_roster))
        groups = {}
        for marker in source.targets:
            groups.setdefault(marker.pattern_index, []).append(marker)
        selected = next(
            (tuple(group) for group in groups.values()
             if len(group) == desired_size), ())
        if not selected:
            raise RuntimeError("source cave lacks requested target group size")
        relative_y = [marker.cell[1] - source.feet_y for marker in selected]
        vertical = len({(marker.cell[0], marker.cell[2])
                        for marker in selected}) == 1
        if vertical != want_vertical:
            continue
        if not vertical and relative_y[0] != int(spec.get("h_rel_y", 0)):
            continue
        if vertical:
            delta_y = -min(relative_y)
            selected = tuple(replace(
                marker,
                instance_id=(
                    f"block:{marker.cell[0]}:{marker.cell[1] + delta_y}:"
                    f"{marker.cell[2]}"),
                cell=(marker.cell[0], marker.cell[1] + delta_y,
                      marker.cell[2]),
                place_cell=(marker.place_cell[0],
                            marker.place_cell[1] + delta_y,
                            marker.place_cell[2]),
            ) for marker in selected)
            relative_y = [
                marker.cell[1] - source.feet_y for marker in selected]
            if relative_y != list(range(len(selected))):
                continue
            occupied = {
                *(marker.cell for marker in source.confusers),
                *(marker.place_cell for marker in source.confusers),
                *(cell for cell, _kind in source.prebuilt_blocks),
            }
            if any(
                    marker.cell in occupied or marker.place_cell in occupied
                    for marker in selected):
                continue
        if max(relative_y) > 2:
            continue

        target_context = set(dict(
            source.context_cells_by_class).get(TARGET_PANEL_PLACEHOLDER, ()))

        selected = tuple(replace(m, kind=target) for m in selected)
        markers = selected + tuple(source.confusers)
        eval_palette = tuple(str(v) for v in spec["palette"])
        scene = replace(
            source,
            target_kind=target,
            train_markers=(target, *PLACE_EVAL_CONFUSERS),
            wall_block=eval_palette[0],
            floor_block=eval_palette[0],
            cave_palette=eval_palette,
            zero_shot_markers=tuple(
                kind for kind in source.zero_shot_markers
                if kind != target),
            markers=markers,
            prebuilt_blocks=tuple(
                row for row in source.prebuilt_blocks
                if row[0] not in target_context),
            context_group_indices_by_class=tuple(
                (target if kind == TARGET_PANEL_PLACEHOLDER else kind,
                 (() if kind == TARGET_PANEL_PLACEHOLDER else indices))
                for kind, indices in source.context_group_indices_by_class),
            context_cells_by_class=tuple(
                (target if kind == TARGET_PANEL_PLACEHOLDER else kind,
                 (() if kind == TARGET_PANEL_PLACEHOLDER else cells))
                for kind, cells in source.context_cells_by_class),
        )
        if scene.held_block == target:
            fallback = next(
                kind for kind in HELD_BLOCK_ROSTER
                if kind != target
                and kind not in {m.kind for m in scene.markers})
            scene = replace(scene, held_block=fallback)
        if len(scene.targets) not in (2, 3) or len(scene.confusers) != 30:
            raise RuntimeError("Place eval target/confuser density changed")
        if len({row.pattern_index for row in scene.targets}) != 1:
            raise RuntimeError("Place eval must contain exactly one target group")
        confuser_kinds = {row.kind for row in scene.confusers}
        if confuser_kinds != set(PLACE_EVAL_CONFUSERS):
            raise RuntimeError(
                f"Place eval confusers must be exactly the fixed roster, "
                f"got {sorted(confuser_kinds)}")
        if target in confuser_kinds:
            raise RuntimeError("target class leaked into confuser panels")
        return scene
    raise RuntimeError(
        "could not sample a Place target group satisfying the y cap")


def build_tasks(*, targets: Sequence[str] = TARGETS,
                layout_indices: Sequence[int] | None = None,
                start_indices: Sequence[int] | None = None,
                policy_seed_namespace: str | None = None
                ) -> list[dict[str, Any]]:
    layouts = build_layouts()
    selected_layouts = (
        range(WORLD_COUNT) if layout_indices is None else layout_indices)
    selected_starts = (
        range(START_COUNT) if start_indices is None else start_indices)
    tasks = []
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"unsupported target {target!r}")
        for layout_index in selected_layouts:
            layout = layouts[int(layout_index)]
            for start_id in selected_starts:
                start_id = int(start_id)
                tasks.append({
                    "task_id": (
                        f"{target}_w{int(layout_index):02d}_s{start_id}"),
                    "target": target,
                    "target_split": (
                        "train" if target in TRAIN_TARGETS else "ood"),
                    "world_index": int(layout_index),
                    "world_seed": int(layout["world_seed"]),
                    "layout_seed": episode_seed(
                        "place_layout", GEOMETRY_SENTINEL,
                        int(layout["world_seed"]), 0),
                    "start_id": start_id,
                    "policy_seed": episode_seed(
                        policy_seed_namespace if policy_seed_namespace
                        else "place_policy",
                        GEOMETRY_SENTINEL,
                        int(layout["world_seed"]), start_id),
                    "policy_seed_variant": 0,
                    "policy_seed_namespace": (
                        policy_seed_namespace or None),
                    "goal_mode": GOAL_MODE,
                    "scene_roster": list(scene_roster(
                        target, int(layout_index))),
                })
    return tasks


def place_snapshot_bundle_path(
        bank_root: Path, target: str, world_index: int) -> Path:
    return (Path(bank_root).expanduser().resolve() / "bundles" /
            f"world_{int(world_index):02d}" / str(target))


def _place_physical_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": "xbench_place_physical_scene_identity/v1",
        "target": str(task["target"]),
        "target_split": str(task["target_split"]),
        "world_index": int(task["world_index"]),
        "world_seed": int(task["world_seed"]),
        "layout_seed": int(task["layout_seed"]),
        "scene_roster": list(task["scene_roster"]),
    }


def _marker_snapshot_value(marker: Marker) -> dict[str, Any]:
    return {
        "instance_id": str(marker.instance_id),
        "kind": str(marker.kind),
        "cell": list(marker.cell),
        "outward_normal": list(marker.outward_normal),
        "place_cell": list(marker.place_cell),
        "role": str(marker.role),
        "pattern_index": marker.pattern_index,
    }


def _marker_from_snapshot(value: Mapping[str, Any]) -> Marker:
    return Marker(
        instance_id=str(value["instance_id"]), kind=str(value["kind"]),
        cell=tuple(int(v) for v in value["cell"]),
        outward_normal=tuple(int(v) for v in value["outward_normal"]),
        place_cell=tuple(int(v) for v in value["place_cell"]),
        role=str(value["role"]),
        pattern_index=(
            None if value.get("pattern_index") is None else
            int(value["pattern_index"])),
    )


def place_scene_snapshot_value(scene: CaveTrainingScene) -> dict[str, Any]:
    return {
        "contract": "xbench_place_cave_training_scene_value/v1",
        "seed": int(scene.seed),
        "center_x": int(scene.center_x), "feet_y": int(scene.feet_y),
        "center_z": int(scene.center_z),
        "background_split": str(scene.background_split),
        "palette_index": int(scene.palette_index),
        "wall_block": str(scene.wall_block), "floor_block": str(scene.floor_block),
        "held_block": str(scene.held_block), "target_kind": str(scene.target_kind),
        "train_markers": list(scene.train_markers),
        "zero_shot_markers": list(scene.zero_shot_markers),
        "mode": str(scene.mode), "pattern": str(scene.pattern),
        "markers": [_marker_snapshot_value(row) for row in scene.markers],
        "outer_radius": int(scene.outer_radius),
        "wall_height": int(scene.wall_height),
        "walkable_bounds_xz": list(scene.walkable_bounds_xz),
        "cave_palette": list(scene.cave_palette),
        "prebuilt_blocks": [
            [list(cell), str(kind)] for cell, kind in scene.prebuilt_blocks],
        "light_cells": [list(cell) for cell in scene.light_cells],
        "portal_interior_cells": [
            list(cell) for cell in scene.portal_interior_cells],
        "macro_variant": str(scene.macro_variant),
        "panel_assignments": [
            [str(panel_id), str(kind), list(center), list(normal)]
            for panel_id, kind, center, normal in scene.panel_assignments],
        "context_group_indices_by_class": [
            [str(kind), list(indices)]
            for kind, indices in scene.context_group_indices_by_class],
        "context_cells_by_class": [
            [str(kind), [list(cell) for cell in cells]]
            for kind, cells in scene.context_cells_by_class],
    }


def place_scene_from_snapshot(value: Mapping[str, Any]) -> CaveTrainingScene:
    if value.get("contract") != "xbench_place_cave_training_scene_value/v1":
        raise ValueError("unsupported Place scene snapshot value")
    return CaveTrainingScene(
        seed=int(value["seed"]), center_x=int(value["center_x"]),
        feet_y=int(value["feet_y"]), center_z=int(value["center_z"]),
        background_split=str(value["background_split"]),
        palette_index=int(value["palette_index"]),
        wall_block=str(value["wall_block"]),
        floor_block=str(value["floor_block"]),
        held_block=str(value["held_block"]),
        target_kind=str(value["target_kind"]),
        train_markers=tuple(str(v) for v in value["train_markers"]),
        zero_shot_markers=tuple(
            str(v) for v in value["zero_shot_markers"]),
        mode=str(value["mode"]), pattern=str(value["pattern"]),
        markers=tuple(_marker_from_snapshot(row) for row in value["markers"]),
        outer_radius=int(value["outer_radius"]),
        wall_height=int(value["wall_height"]),
        walkable_bounds_xz=tuple(
            int(v) for v in value["walkable_bounds_xz"]),
        cave_palette=tuple(str(v) for v in value["cave_palette"]),
        prebuilt_blocks=tuple(
            (tuple(int(v) for v in row[0]), str(row[1]))
            for row in value["prebuilt_blocks"]),
        light_cells=tuple(
            tuple(int(v) for v in row) for row in value["light_cells"]),
        portal_interior_cells=tuple(
            tuple(int(v) for v in row)
            for row in value["portal_interior_cells"]),
        macro_variant=str(value["macro_variant"]),
        panel_assignments=tuple(
            (str(row[0]), str(row[1]), tuple(int(v) for v in row[2]),
             tuple(int(v) for v in row[3]))
            for row in value["panel_assignments"]),
        context_group_indices_by_class=tuple(
            (str(row[0]), tuple(int(v) for v in row[1]))
            for row in value["context_group_indices_by_class"]),
        context_cells_by_class=tuple(
            (str(row[0]), tuple(tuple(int(v) for v in cell)
                                for cell in row[1]))
            for row in value["context_cells_by_class"]),
    )


def _validate_place_eval_snapshot_scene(scene: CaveTrainingScene) -> None:
    relative_y = sorted(
        marker.cell[1] - scene.feet_y for marker in scene.targets)
    if not relative_y or max(relative_y) > 2:
        raise ValueError("Place eval snapshot target exceeds its y cap")
    vertical = len({
        (marker.cell[0], marker.cell[2]) for marker in scene.targets}) == 1
    if vertical and relative_y != list(range(len(relative_y))):
        raise ValueError(
            "Place eval snapshot vertical target is not floor-pinned "
            f"(got rel-y {relative_y})")
    if scene.held_block not in HELD_BLOCK_ROSTER:
        raise ValueError("Place eval snapshot held block left nuisance roster")


def build_place_snapshot_payload(
        task: Mapping[str, Any], scene: CaveTrainingScene,
        stage: Mapping[str, Any]) -> dict[str, Any]:
    _validate_place_eval_snapshot_scene(scene)
    physical = _place_physical_task(task)
    scene_value = place_scene_snapshot_value(scene)
    scene_doc = dict(stage["scene"])
    return {
        "contract": PLACE_SNAPSHOT_PAYLOAD_CONTRACT,
        "physical_task": physical,
        "physical_task_sha256": stable_sha256(physical),
        "scene_value": scene_value,
        "scene_value_sha256": stable_sha256(scene_value),
        "scene_sha256": str(scene_doc["scene_sha256"]),
        "stage": dict(stage),
        "snapshot_boundary": (
            "fully_staged_static_cave_center_target_absent_before_policy/v1"),
    }


def load_place_snapshot_bundle(bank_root: Path, task: Mapping[str, Any]):
    from attacca.worlds.world_snapshot import load_world_snapshot_bundle
    path = place_snapshot_bundle_path(
        bank_root, str(task["target"]), int(task["world_index"]))
    bundle = load_world_snapshot_bundle(path)
    payload = bundle.payload
    physical = _place_physical_task(task)
    if (not isinstance(payload, dict)
            or payload.get("contract") != PLACE_SNAPSHOT_PAYLOAD_CONTRACT
            or payload.get("physical_task") != physical
            or payload.get("physical_task_sha256") != stable_sha256(physical)):
        raise ValueError(f"Place snapshot physical identity mismatch: {path}")
    scene = place_scene_from_snapshot(payload["scene_value"])
    _validate_place_eval_snapshot_scene(scene)
    current_scene_doc = eval_scene_document(
        scene, target_split=str(task["target_split"]),
        world_index=int(task["world_index"]))
    if (payload.get("scene_value_sha256")
            != stable_sha256(payload["scene_value"])
            or payload.get("scene_sha256") != current_scene_doc["scene_sha256"]):
        raise ValueError(f"Place snapshot scene digest mismatch: {path}")
    return bundle


def _goal_root(root: Path, target: str) -> Path:
    return Path(root) / f"mine_{target}" / GOAL_MODE


def load_goals(root: Path, targets: Sequence[str]) -> dict[str, dict]:
    manifest_path = Path(root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("contract") != GOAL_LIBRARY_CONTRACT_V1
            or manifest.get("complete") is not True):
        raise ValueError("Place goal library is incomplete/foreign")
    indexed = {row["target"]: dict(row) for row in manifest["entries"]}
    output = {}
    for target in targets:
        meta = indexed.get(target)
        if meta is None:
            raise ValueError(f"goal library lacks {target}")
        directory = _goal_root(root, target)
        goal_path, mask_path = directory / "goal.png", directory / "mask.png"
        if (sha256_file(goal_path) != meta["goal_rgb_sha256"]
                or sha256_file(mask_path) != meta["goal_mask_sha256"]):
            raise ValueError(f"goal digest mismatch for {target}")
        meta["goal_rgb"] = str(goal_path.resolve())
        meta["goal_mask"] = str(mask_path.resolve())
        output[target] = meta
    return output


def _circular_distance(left: float, right: float) -> float:
    delta = abs(float(left) - float(right)) % 360.0
    return min(delta, 360.0 - delta)


def target_artifacts(world, scene: CaveTrainingScene, *,
                     completed_ids: set[str],
                     render_position: Sequence[float] | None = None
                     ) -> dict[str, Any]:
    hand, depth = strict_depth_arrays(world)
    frame_pose = pose(world, render_position=render_position)
    active = [
        marker for marker in scene.targets
        if marker.instance_id not in completed_ids]
    eye, owner_pixels, owner_stats = _owner_pixels_from_depth(
        hand, depth, frame_pose, active,
        lookup_scope="place_eval_active_target_markers_only")
    union = np.zeros(FRAME_SHAPE, np.uint8)
    visible = []
    recognizable_ids = []
    centre_flat = 180 * 640 + 320
    crosshair_id = None
    crosshair_distance = None
    for marker in active:
        pixels = np.asarray(owner_pixels.get(marker.cell, ()), dtype=np.int32)
        mask = np.zeros(FRAME_SHAPE, np.uint8)
        if len(pixels):
            mask.reshape(-1)[pixels] = 1
        is_recognizable, bbox, short = recognizable(mask)
        distance = aabb_distance(eye, marker.cell)
        if np.any(pixels == centre_flat):
            crosshair_id = marker.instance_id
            crosshair_distance = float(distance)
        if is_recognizable:
            union |= mask
            recognizable_ids.append(marker.instance_id)
        if len(pixels):
            visible.append({
                "instance_id": marker.instance_id,
                "kind": marker.kind,
                "cell": list(marker.cell),
                "place_cell": list(marker.place_cell),
                "interaction_group_id": str(marker.pattern_index),
                "recognizable": int(is_recognizable),
                "pixel_count": int(len(pixels)),
                "bbox_xyxy": bbox,
                "visible_short_side_px": int(short),
                "nearest_surface_distance": float(distance),
            })
    return {
        "union": union,
        "visible_instances": visible,
        "recognizable_instance_ids": recognizable_ids,
        "exact_visible_pixel_count": int(sum(
            row["pixel_count"] for row in visible)),
        "recognizable_union_pixel_count": int(union.sum()),
        "crosshair_target_instance_id": crosshair_id,
        "crosshair_target_surface_distance": crosshair_distance,
        "crosshair_target_in_reach": int(
            crosshair_id is not None
            and crosshair_distance is not None
            and crosshair_distance <= USE_REACH_BLOCKS),
        "render_pose": frame_pose,
        "viewmodel_occlusion_subtracted": True,
        "mask_semantic": "exact_native_depth_first_hit_voxel_xyz/v1",
        "depth_owner_index_stats": owner_stats,
    }


def _select_start_yaws(candidates: Sequence[float], *, seed: int,
                       count: int = START_COUNT) -> list[float]:
    values = [float(value) for value in candidates]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(values)
    for minimum_separation in (60.0, 45.0, 30.0, 15.0, 0.0):
        selected = []
        for value in values:
            if all(_circular_distance(value, old) >= minimum_separation
                   for old in selected):
                selected.append(value)
                if len(selected) == int(count):
                    return selected
    raise RuntimeError(f"only {len(candidates)} target-free start yaws")


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    if not cv2.imwrite(
            str(path), cv2.cvtColor(
                np.ascontiguousarray(rgb, dtype=np.uint8),
                cv2.COLOR_RGB2BGR)):
        raise OSError(path)


def stage_eval_episode(world, stager: Stager, *, task: Mapping[str, Any],
                       episode_root: Path) -> tuple[CaveTrainingScene, dict]:
    stager.prep_world()
    px, py, pz = world.get_pos()
    scene = build_eval_scene(
        seed=int(task["layout_seed"]), center_x=math.floor(px),
        feet_y=math.floor(py), center_z=math.floor(pz),
        target=str(task["target"]), world_index=int(task["world_index"]))
    stage_cave_commands(world, scene)
    voxel_audit = audit_staged_cave(world, stager, scene)
    equip_cave_training(world, stager, scene)
    toast_drain = drain_toasts_camera_off(world)

    def _yaw_scan():
        scan_rows, scan_best, scan_absent = [], None, []
        for yaw in np.arange(-180.0, 180.0, 15.0):
            if not stager.go(
                    scene.center_x + 0.5, scene.feet_y,
                    scene.center_z + 0.5, float(yaw), pitch=8.0, n=5):
                continue
            artifacts = target_artifacts(world, scene, completed_ids=set())
            row = {
                "yaw": float(yaw),
                "exact_target_pixels": int(
                    artifacts["exact_visible_pixel_count"]),
                "recognizable_union_pixels": int(
                    artifacts["recognizable_union_pixel_count"]),
            }
            scan_rows.append(row)
            if row["exact_target_pixels"] == 0:
                scan_absent.append(float(yaw))
            score = (row["recognizable_union_pixels"],
                     row["exact_target_pixels"])
            if scan_best is None or score > scan_best[0]:
                scan_best = (score, float(yaw))
        return scan_rows, scan_best, scan_absent

    scans, best, absent = _yaw_scan()
    if best is None or best[0][0] <= 0:
        for _ in range(40):
            world.step_noop()
        scans, best, absent = _yaw_scan()
    if best is None or best[0][0] <= 0:
        raise RuntimeError("Place eval scene has no recognizable target probe")
    start_seed = int(task["layout_seed"])
    starts = _select_start_yaws(absent, seed=start_seed)

    if not stager.go(
            scene.center_x + 0.5, scene.feet_y, scene.center_z + 0.5,
            best[1], pitch=8.0, n=5):
        raise RuntimeError("could not restore Place eval target probe")
    probe_artifacts = target_artifacts(world, scene, completed_ids=set())
    probe_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    _save_rgb(episode_root / "target_visible_probe_rgb.png", probe_rgb)
    _save_rgb(
        episode_root / "target_visible_probe_union_overlay.png",
        overlay_masks(
            probe_rgb, probe_artifacts["union"],
            np.zeros(FRAME_SHAPE, np.uint8)))

    start_id = int(task["start_id"])
    if not 0 <= start_id < len(starts):
        raise ValueError(
            f"start_id outside the {START_COUNT}-start contract: {start_id}")
    start_yaw = starts[start_id]
    if not stager.go(
            scene.center_x + 0.5, scene.feet_y, scene.center_z + 0.5,
            start_yaw, pitch=8.0, n=6):
        raise RuntimeError("could not establish Place eval start")
    frame0 = target_artifacts(world, scene, completed_ids=set())
    if frame0["exact_visible_pixel_count"] != 0:
        raise RuntimeError("Place eval frame zero contains target pixels")
    frame0_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    if frame0_rgb.shape != (360, 640, 3) or not np.any(frame0_rgb):
        raise RuntimeError("Place eval frame zero is malformed/all-black")
    _save_rgb(episode_root / "frame0_rgb.png", frame0_rgb)

    scene_doc = eval_scene_document(
        scene, target_split=str(task["target_split"]),
        world_index=int(task["world_index"]))
    stage = {
        "contract": SCENE_CONTRACT,
        "task": dict(task), "scene": scene_doc,
        "voxel_audit": voxel_audit, "toast_drain": toast_drain,
        "start_pose": {
            "x": scene.center_x + 0.5, "y": scene.feet_y,
            "z": scene.center_z + 0.5,
            "yaw": start_yaw, "pitch": 8.0,
            "start_id": start_id,
        },
        "all_target_free_starts": [
            {"start_id": index, "yaw": yaw, "pitch": 8.0}
            for index, yaw in enumerate(starts)],
        "yaw_scan": scans,
        "visible_probe": {
            "yaw": best[1],
            "recognizable_union_pixel_count": int(
                probe_artifacts["recognizable_union_pixel_count"]),
            "exact_visible_pixel_count": int(
                probe_artifacts["exact_visible_pixel_count"]),
        },
        "frame0_target_exact_pixel_count": 0,
        "frame0_target_absent": True,
        "held_item": scene.held_block,
        "camera_off_staging": True,
    }
    atomic_json(episode_root / "stage.json", stage)
    return scene, stage


def _phase_probabilities(latents: Mapping[str, Any]) -> list[float] | None:
    from attacca.evaluation.hunt_phase import phase_probabilities
    return phase_probabilities(latents, required=False)


def _capture_prediction(runner: Rocket2GoalRunner) -> dict[str, Any]:
    from attacca.evaluation.prediction import capture_model_prediction
    prediction = capture_model_prediction(
        runner.agent.cache_latents, require_exist=True)
    probabilities = _phase_probabilities(runner.agent.cache_latents)
    prediction["phase_order"] = ["EXPLORE", "APPROACH", "INTERACT"]
    prediction["phase_probabilities"] = probabilities
    prediction["phase_argmax"] = (
        None if probabilities is None else
        prediction["phase_order"][int(np.argmax(probabilities))])
    return prediction


def _action_flag(action: Mapping[str, Any], key: str) -> bool:
    try:
        return bool(np.asarray(action.get(key, 0)).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return False


_WORKER_RUNNER: Rocket2GoalRunner | None = None
_WORKER_GOAL_ROOT: Path | None = None
_WORKER_BUDGET = DEFAULT_BUDGET
_WORKER_POST_SUCCESS_FRAMES = DEFAULT_POST_SUCCESS_FRAMES
_WORKER_MODEL_LABEL = ""
_WORKER_SNAPSHOT_ROOT: Path | None = None


def _worker_init(checkpoint: str, goal_root: str, budget: int,
                 post_success_frames: int,
                 cfg_coef: float, model_label: str,
                 snapshot_root: str) -> None:
    global _WORKER_RUNNER, _WORKER_GOAL_ROOT, _WORKER_BUDGET
    global _WORKER_POST_SUCCESS_FRAMES
    global _WORKER_MODEL_LABEL
    global _WORKER_SNAPSHOT_ROOT
    install_renderer_viewmodel_observation()
    _WORKER_RUNNER = Rocket2GoalRunner(
        checkpoint, cfg_coef=float(cfg_coef), device="cuda")
    _WORKER_RUNNER.set_obj_id(USE_INTERACTION_ID)
    _WORKER_GOAL_ROOT = Path(goal_root)
    _WORKER_BUDGET = int(budget)
    _WORKER_POST_SUCCESS_FRAMES = int(post_success_frames)
    _WORKER_MODEL_LABEL = str(model_label)
    _WORKER_SNAPSHOT_ROOT = Path(snapshot_root).resolve()


def _boot_eval(seed: int, *, maximum_attempts: int = 3):
    last = None
    for attempt in range(1, int(maximum_attempts) + 1):
        try:
            return boot_world(
                int(seed), biome="plains", action_type="env",
                runtime_overlay=REPO / "configs" / "xbench_runtime_dense")
        except BaseException as exc:
            last = exc
            retryable = (
                isinstance(exc, BrokenPipeError)
                or "BrokenPipe" in repr(exc)
                or "Connection refused" in repr(exc)
                or "empty reply from Malmo" in repr(exc))
            if not retryable or attempt == int(maximum_attempts):
                raise
            print(
                f"[place-eval] connection error on boot attempt {attempt}/{maximum_attempts}; restarting "
                f"seed={seed}: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"simulator boot did not complete: {last}")


def _boot_eval_snapshot(bundle, *, seed: int, maximum_attempts: int = 3):
    last = None
    for attempt in range(1, int(maximum_attempts) + 1):
        world = None
        try:
            world = boot_world(
                int(seed), biome="plains", action_type="env",
                runtime_overlay=REPO / "configs" / "xbench_runtime_dense",
                world_snapshot_dir=str(bundle.world.directory),
                world_snapshot_sha256=str(bundle.world.sha256),
                world_snapshot_archive=str(bundle.archive),
                world_snapshot_archive_sha256=str(
                    bundle.metadata["world_archive_sha256"]),
            )
            world.cmd("/difficulty normal")
            world.cmd("/gamerule doMobSpawning false")
            world.cmd("/gamerule randomTickSpeed 0")
            world.cmd("/gamerule doDaylightCycle false")
            world.cmd("/gamerule doWeatherCycle false")
            world.cmd("/weather clear")
            world.cmd("/time set 6000")
            world.cmd("/gamerule sendCommandFeedback false")
            return world
        except BaseException as exc:
            last = exc
            if world is not None:
                try:
                    world.close()
                except BaseException:
                    pass
            retryable = (
                isinstance(exc, BrokenPipeError)
                or "BrokenPipe" in repr(exc)
                or "Connection refused" in repr(exc)
                or "empty reply from Malmo" in repr(exc))
            if not retryable or attempt == int(maximum_attempts):
                raise
            print(
                f"[place-snapshot] connection error on boot attempt {attempt}/{maximum_attempts}; restarting "
                f"seed={seed}: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"snapshot simulator boot did not complete: {last}")


def audit_place_snapshot(
        world, stager: Stager, bundle, task: Mapping[str, Any],
        out: Path, *, full_voxel_audit: bool) -> tuple[CaveTrainingScene, dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    payload = bundle.payload
    if payload.get("physical_task") != _place_physical_task(task):
        raise ValueError("Place snapshot audit received the wrong physical task")
    scene = place_scene_from_snapshot(payload["scene_value"])
    start_id = int(task["start_id"])
    starts = list(payload["stage"]["all_target_free_starts"])
    if not 0 <= start_id < len(starts):
        raise ValueError("Place snapshot start_id outside curated start roster")
    selected = starts[start_id]
    start_pose = {
        "x": scene.center_x + 0.5, "y": scene.feet_y,
        "z": scene.center_z + 0.5,
        "yaw": float(selected["yaw"]), "pitch": float(selected["pitch"]),
        "start_id": start_id,
    }
    if not stager.go(
            start_pose["x"], start_pose["y"], start_pose["z"],
            start_pose["yaw"], pitch=start_pose["pitch"], n=6):
        raise RuntimeError("could not establish restored Place start pose")
    artifacts = target_artifacts(world, scene, completed_ids=set())
    if int(artifacts["exact_visible_pixel_count"]) != 0:
        raise RuntimeError("restored Place frame zero contains target pixels")
    restored_held = held_item(world)
    if restored_held != scene.held_block:
        raise RuntimeError(
            f"restored Place held item mismatch: {restored_held!r}")
    voxel_audit = (
        audit_staged_cave(world, stager, scene)
        if full_voxel_audit else None)
    rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    if rgb.shape != (360, 640, 3) or not np.any(rgb):
        raise RuntimeError("restored Place frame zero is malformed/all-black")
    frame_path = out / "frame0_rgb.png"
    _save_rgb(frame_path, rgb)
    audit = {
        "contract": "xbench_place_snapshot_live_audit/v1",
        "physical_task_sha256": str(payload["physical_task_sha256"]),
        "scene_value_sha256": str(payload["scene_value_sha256"]),
        "scene_sha256": str(payload["scene_sha256"]),
        "world_sha256": str(bundle.world.sha256),
        "world_archive_sha256": str(
            bundle.metadata["world_archive_sha256"]),
        "payload_sha256": str(bundle.metadata["payload_sha256"]),
        "target": str(task["target"]),
        "world_index": int(task["world_index"]),
        "start_id": start_id,
        "start_pose": start_pose,
        "frame0_target_exact_pixel_count": 0,
        "frame0_target_absent": True,
        "held_item": restored_held,
        "full_voxel_audit": voxel_audit,
        "frame0_rgb": str(frame_path.resolve()),
        "frame0_rgb_bytes_sha256": hashlib.sha256(
            rgb.tobytes()).hexdigest(),
    }
    atomic_json(out / "audit.json", audit)
    return scene, audit


def restored_place_stage(
        world, stager: Stager, bundle, task: Mapping[str, Any],
        episode_root: Path) -> tuple[CaveTrainingScene, dict]:
    scene, audit = audit_place_snapshot(
        world, stager, bundle, task,
        Path(episode_root) / "snapshot_restore_audit",
        full_voxel_audit=False)
    stage = copy.deepcopy(dict(bundle.payload["stage"]))
    stage.update({
        "contract": "xbench_place_snapshot_restore_stage/v1",
        "task": dict(task),
        "start_pose": dict(audit["start_pose"]),
        "frame0_target_exact_pixel_count": 0,
        "frame0_target_absent": True,
        "held_item": scene.held_block,
        "snapshot_restore_audit": audit,
        "snapshot_bundle": str(bundle.directory),
        "snapshot_world_sha256": str(bundle.world.sha256),
        "snapshot_archive_sha256": str(
            bundle.metadata["world_archive_sha256"]),
        "snapshot_payload_sha256": str(bundle.metadata["payload_sha256"]),
    })
    atomic_json(Path(episode_root) / "stage.json", stage)
    return scene, stage


def _video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not decode video {path}")
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _run_task(task: Mapping[str, Any], episodes_root_value: str) -> dict:
    if _WORKER_RUNNER is None or _WORKER_GOAL_ROOT is None:
        raise RuntimeError("Place eval worker was not initialized")
    import torch

    episode_started = time.monotonic()
    timings = {
        "video_writer_open_seconds": 0.0,
        "raw_video_write_seconds": 0.0,
        "review_compose_write_seconds": 0.0,
        "frame_json_write_seconds": 0.0,
    }

    episodes_root = Path(episodes_root_value)
    episode_root = episodes_root / str(task["task_id"])
    episode_root.mkdir(parents=True, exist_ok=False)
    target = str(task["target"])
    goal_meta = load_goals(_WORKER_GOAL_ROOT, [target])[target]
    goal_bgr = cv2.imread(str(goal_meta["goal_rgb"]), cv2.IMREAD_COLOR)
    goal_mask = cv2.imread(str(goal_meta["goal_mask"]), cv2.IMREAD_GRAYSCALE)
    if goal_bgr is None or goal_mask is None:
        raise RuntimeError(f"could not decode Place goal {target}")
    goal_rgb = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)
    goal_mask = np.ascontiguousarray(goal_mask > 0, dtype=np.uint8)

    policy_seed = int(task["policy_seed"])
    random.seed(policy_seed)
    np.random.seed(policy_seed & 0xFFFFFFFF)
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    runner = _WORKER_RUNNER
    runner.set_obj_id(USE_INTERACTION_ID)
    runner.reset()

    snapshot_bundle = load_place_snapshot_bundle(_WORKER_SNAPSHOT_ROOT, task)
    world = _boot_eval_snapshot(snapshot_bundle, seed=int(task["world_seed"]))
    stager = Stager(world)
    raw_path = episode_root / "raw_policy.mp4"
    raw_writer = None
    rows = []
    placements = []
    completed: set[str] = set()
    behavioral_failure = None
    off_target_total = 0
    held_drift_step = None
    first_failure_step = None
    first_failure_reason = None
    terminated = truncated = False
    first_discovery_step = None
    success_step = None
    try:
        scene, stage = restored_place_stage(
            world, stager, snapshot_bundle, task, episode_root)
        target_quota = len(scene.targets)
        expected_locations = {row.place_cell for row in scene.targets}
        allowed_existing = {
            cell for cell, kind in scene.prebuilt_blocks
            if expected_place_type(scene.held_block, kind)}
        if held_item(world) != scene.held_block:
            raise RuntimeError("Place eval staged held item mismatch")

        writer_started = time.monotonic()
        raw_writer = cv2.VideoWriter(
            str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 360))
        if not raw_writer.isOpened():
            raise RuntimeError("could not open Place eval video writers")
        timings["video_writer_open_seconds"] = float(
            time.monotonic() - writer_started)

        render_position = tuple(float(value) for value in world.get_pos())
        for step in range(
                int(_WORKER_BUDGET) + int(_WORKER_POST_SUCCESS_FRAMES)):
            if step >= int(_WORKER_BUDGET) and success_step is None:
                break
            post_success_observation = success_step is not None
            pre_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
            if pre_rgb.shape != (360, 640, 3) or not np.any(pre_rgb):
                raise RuntimeError(f"bad policy RGB at step {step}")
            artifacts = target_artifacts(
                world, scene, completed_ids=completed,
                render_position=render_position)
            if (artifacts["recognizable_instance_ids"]
                    and first_discovery_step is None):
                first_discovery_step = int(step)
            agent_action, _returned_exist = runner.act(
                world.obs, goal_rgb, goal_mask)
            prediction = _capture_prediction(runner)
            env_action = world.sim.agent_action_to_env_action(agent_action)
            use = _action_flag(env_action, "use")
            if use:
                query_box, _origin = query_box_for_targets(
                    world, scene, full_room=True)
                env_action["voxels"] = query_box

            raw_write_started = time.monotonic()
            raw_writer.write(cv2.cvtColor(pre_rgb, cv2.COLOR_RGB2BGR))
            timings["raw_video_write_seconds"] += float(
                time.monotonic() - raw_write_started)
            pre_completed = len(completed)
            pre_action_position = tuple(
                float(value) for value in world.get_pos())
            world.obs, reward, terminated, truncated, world.info = (
                world.sim.step(env_action))
            render_position = pre_action_position
            note = ""
            placement = None
            off_target_cells = []
            if (not post_success_observation
                    and held_item(world) != scene.held_block):
                if held_drift_step is None:
                    held_drift_step = int(step)
                    if first_failure_step is None:
                        first_failure_step = int(step)
                        first_failure_reason = (
                            f"held_item_drift:{held_item(world)}")
                behavioral_failure = (
                    f"held_item_drift:{held_item(world)}")
                note = "HELD DRIFT"
            elif not post_success_observation and use:
                origin = tuple(math.floor(value) for value in world.get_pos())
                query = queried_types(world.info, origin)
                off_target_cells = sorted(
                    cell for cell, value in query.items()
                    if expected_place_type(scene.held_block, value)
                    and cell not in expected_locations
                    and cell not in allowed_existing)
                if off_target_cells:
                    off_target_total += len(off_target_cells)
                    if first_failure_step is None:
                        first_failure_step = int(step)
                        first_failure_reason = (
                            "off_target_placement:"
                            f"{[list(value) for value in off_target_cells[:8]]}")
                    behavioral_failure = (
                        "off_target_placement:"
                        f"{[list(value) for value in off_target_cells[:8]]}")
                    note = "OFF-TARGET"
                changed = []
                for marker in scene.targets:
                    if marker.instance_id in completed:
                        continue
                    actual = query.get(marker.place_cell, "air")
                    if expected_place_type(scene.held_block, actual):
                        changed.append(marker)
                for marker in changed:
                    completed.add(marker.instance_id)
                    placements.append({
                        "step": int(step),
                        "instance_id": marker.instance_id,
                        "interaction_group_id": str(marker.pattern_index),
                        "marker_cell": list(marker.cell),
                        "place_cell": list(marker.place_cell),
                        "placed_block": scene.held_block,
                    })
                if len(changed) > 1:
                    note = (
                        f"PLACED +{len(changed)} "
                        f"{len(completed)}/{target_quota}")
                elif len(changed) == 1:
                    note = f"PLACED {len(completed)}/{target_quota}"

            rows.append({
                "contract": FRAME_CONTRACT,
                "step": int(step),
                "post_success_observation": bool(post_success_observation),
                "target": target,
                "interaction_id": USE_INTERACTION_ID,
                "obj_id": USE_INTERACTION_IDS["obj_id"],
                "verb_id": USE_INTERACTION_IDS["verb_id"],
                "tool_id": MODE_TOOL_ID["block"],
                "use": int(use),
                "agent_action": json_safe(agent_action),
                "executed_env_action": json_safe(env_action),
                "pose": artifacts["render_pose"],
                "held_item": held_item(world),
                "completed_before": pre_completed,
                "completed_after": len(completed),
                "recognizable_instance_ids": artifacts[
                    "recognizable_instance_ids"],
                "recognizable_union_pixel_count": artifacts[
                    "recognizable_union_pixel_count"],
                "exact_visible_pixel_count": artifacts[
                    "exact_visible_pixel_count"],
                "class_union_exact_pixels": int(np.asarray(
                    artifacts["union"], dtype=np.uint8).sum()),
                "class_union_mask_runs_yx": encode_mask_runs_yx(
                    np.ascontiguousarray(
                        artifacts["union"], dtype=np.uint8)),
                "class_union_mask_shape_hw": [360, 640],
                "crosshair_target_instance_id": artifacts[
                    "crosshair_target_instance_id"],
                "crosshair_target_surface_distance": artifacts[
                    "crosshair_target_surface_distance"],
                "crosshair_target_in_reach": artifacts[
                    "crosshair_target_in_reach"],
                "placement": placement,
                "off_target_cells": [list(value) for value in off_target_cells],
                "prediction": prediction,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "behavioral_failure": behavioral_failure,
                "review_note": note,
            })
            if success_step is None and len(completed) == target_quota:
                success_step = int(step)
                rows[-1]["success_event"] = True
            else:
                rows[-1]["success_event"] = False
            if terminated or truncated:
                behavioral_failure = behavioral_failure or (
                    "sim_terminated_or_truncated")
                break
            if post_success_tail_complete(
                    success_step, step, _WORKER_POST_SUCCESS_FRAMES):
                break

        if behavioral_failure is None and len(completed) < target_quota:
            behavioral_failure = "timeout"
            if rows:
                rows[-1]["behavioral_failure"] = behavioral_failure
        success = bool(len(completed) == target_quota)
        clean_success = bool(
            success and off_target_total == 0 and held_drift_step is None)
        strict_success = success
        json_started = time.monotonic()
        with (episode_root / "frames.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
        timings["frame_json_write_seconds"] = float(
            time.monotonic() - json_started)
        raw_writer.release()
        raw_writer = None
        raw_frames = _video_frames(raw_path)
        if raw_frames != len(rows):
            raise RuntimeError(
                "Place eval raw/frame-json alignment failed: "
                f"{raw_frames}/{len(rows)}")
        result = {
            "contract": EPISODE_CONTRACT,
            "task": dict(task),
            "model_label": _WORKER_MODEL_LABEL,
            "goal": goal_meta,
            "scene_contract": SCENE_CONTRACT,
            "scene_sha256": stage["scene"]["scene_sha256"],
            "snapshot_restore_used": True,
            "snapshot_bundle": str(snapshot_bundle.directory),
            "scoring_contract": "place_run_to_budget_voxel_success_clean/v4",
            "post_success_tail_is_non_scoring": True,
            "strict_success": int(strict_success),
            "success": int(success),
            "clean_success": int(clean_success),
            "off_target_placement_total": int(off_target_total),
            "held_item_drift_step": held_drift_step,
            "first_failure_step": first_failure_step,
            "first_failure_reason": first_failure_reason,
            "target_quota": int(target_quota),
            "all_targets_completed": int(
                len(completed) == target_quota),
            "completed_target_count": len(completed),
            "completed_instance_ids": sorted(completed),
            "placements": placements,
            "placement_groups_completed": sorted({
                row["interaction_group_id"] for row in placements}),
            "behavioral_failure": behavioral_failure,
            "first_discovery_step": first_discovery_step,
            "steps": len(rows),
            "budget": int(_WORKER_BUDGET),
            "post_success_frames_requested": int(
                _WORKER_POST_SUCCESS_FRAMES),
            "success_step": success_step,
            "post_success_frames_recorded": (
                post_success_frames_recorded(success_step, len(rows))),
            "timing": {
                **timings,
                "policy_frames": len(rows),
                "episode_preclose_wall_seconds": float(
                    time.monotonic() - episode_started),
            },
            "raw_video": str(raw_path.resolve()),
            "review_mode": "deferred",
            "review_state": "pending",
            "review_video": None,
            "deferred_review_payload_contract": (
                DEFERRED_REVIEW_PAYLOAD_CONTRACT),
            "frames_jsonl": str((episode_root / "frames.jsonl").resolve()),
            "raw_review_frame_alignment": None,
            "policy_action_modified": False,
            "observation_query_added_to_use_action": True,
        }
        atomic_json(episode_root / "result.json", result)
        return result
    finally:
        if raw_writer is not None:
            raw_writer.release()
        world.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_root).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f"fresh Place eval root required: {out}")
    out.mkdir(parents=True, exist_ok=True)
    snapshot_bank_root = Path(args.snapshot_bank_root).expanduser().resolve()
    if not snapshot_bank_root.is_dir():
        raise FileNotFoundError(snapshot_bank_root)
    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint.is_file() or checkpoint.is_dir()):
        raise FileNotFoundError(checkpoint)
    goal_root = Path(args.goal_source_root).resolve()
    load_goals(goal_root, args.targets)

    tasks = build_tasks(
        targets=args.targets, layout_indices=args.layout_indices,
        start_indices=args.start_indices,
        policy_seed_namespace=args.policy_seed_namespace)
    if not tasks:
        raise ValueError("Place eval task selection is empty")
    episodes_root = out / "episodes"
    episodes_root.mkdir(exist_ok=True)
    selected_layout_indices = (
        list(range(WORLD_COUNT)) if args.layout_indices is None else
        [int(value) for value in args.layout_indices])
    selected_start_indices = (
        list(range(START_COUNT)) if args.start_indices is None else
        [int(value) for value in args.start_indices])
    atomic_json(out / "layouts.json", {
        "contract": LAYOUT_CONTRACT,
        "layouts": build_layouts(),
        "matrix": {
            "targets": list(args.targets),
            "layout_indices": selected_layout_indices,
            "worlds_per_target": len(selected_layout_indices),
            "start_indices": selected_start_indices,
            "starts_per_world": len(selected_start_indices),
            "episodes_per_target": (
                len(selected_layout_indices) * len(selected_start_indices)),
            "full_episodes": len(tasks),
            "start_count_capacity": START_COUNT,
            "policy_seed_variant": 0,
        },
    })
    run_manifest = {
        "contract": CONTRACT, "machine": machine_record(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        "checkpoint_training_scope": "unspecified",
        "goal_root": str(goal_root), "goal_mode": GOAL_MODE,
        "targets": list(args.targets), "task_count": len(tasks),
        "layout_indices": selected_layout_indices,
        "start_indices": selected_start_indices,
        "workers": int(args.workers), "budget": int(args.budget),
        "review_mode": "deferred",
        "deferred_review_payload_contract": (
            DEFERRED_REVIEW_PAYLOAD_CONTRACT),
        "snapshot_restore_required": True,
        "snapshot_bank_root": str(snapshot_bank_root),
        "post_success_frames": int(args.post),
        "interaction_ids": {
            **USE_INTERACTION_IDS, "tool_id": MODE_TOOL_ID["block"]},
        "success_contract": (
            "voxel-certified completion of the single two/three-marker "
            "target group; clean_success additionally requires no off-target "
            "placement or held-item drift through the success frame; the "
            "fixed post-success tail is observation-only and non-scoring; "
            "no policy action rewriting"),
        "command": sys.argv,
    }
    atomic_json(out / "RUN_MANIFEST.json", run_manifest)

    reset_cap = max(
        int(args.workers),
        int(os.environ.get("MINESTUDIO_MAX_RESETTING_ENV_COUNT", "0") or 0))
    os.environ["MINESTUDIO_MAX_RESETTING_ENV_COUNT"] = str(reset_cap)
    pending = []
    results = []
    for task in tasks:
        directory = episodes_root / task["task_id"]
        result_path = directory / "result.json"
        if args.resume and result_path.is_file():
            results.append(json.loads(result_path.read_text()))
            continue
        if directory.exists():
            quarantine = out / "quarantine" / (
                f"{directory.name}_{int(time.time())}")
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(directory, quarantine)
        pending.append(task)

    failures = []
    if pending:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(args.workers), initializer=_worker_init,
                initargs=(
                    str(checkpoint), str(goal_root), int(args.budget),
                    int(args.post), float(args.cfg_coef),
                    str(args.model_label),
                    str(snapshot_bank_root))) as pool:
            future_to_task = {
                pool.submit(_run_task, task, str(episodes_root)): task
                for task in pending}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    failure = {
                        "task": task,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    print(
                        f"[place-eval] FAIL {task['task_id']}: "
                        f"{type(exc).__name__}: {exc}", flush=True)
                else:
                    results.append(result)
                    print(
                        f"[place-eval] DONE {task['task_id']} "
                        f"strict={result['strict_success']} "
                        f"completed={result['completed_target_count']}/"
                        f"{result['target_quota']}",
                        flush=True)
    return {
        "state": (
            "complete" if len(results) == len(tasks) and not failures
            else "incomplete"),
        "expected_episodes": len(tasks),
        "completed_episodes": len(results),
        "infrastructure_failures": len(failures),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal-source-root", type=Path, required=True)
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--cfg-coef", type=float, default=0.0)
    parser.add_argument("--targets", nargs="+", choices=TARGETS,
                        default=list(TARGETS))
    parser.add_argument("--layout-indices", nargs="+", type=int)
    parser.add_argument("--start-indices", nargs="+", type=int)
    parser.add_argument(
        "--policy-seed-namespace", default=None,
        help="explicit hash namespace for the rollout seed; any new string "
             "yields an independent seed set over the same scenes")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--post", type=int, default=DEFAULT_POST_SUCCESS_FRAMES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--snapshot-bank-root", type=Path, required=True,
        help="immutable Place snapshot bank; every target/world bundle is required")
    args = parser.parse_args(argv)
    if not 1 <= int(args.workers) <= 8:
        parser.error("workers must be in [1,8]")
    if int(args.budget) <= 0 or int(args.post) != 20:
        parser.error("budget must be positive and the benchmark requires --post 20")
    for name, values, limit in (
            ("layout", args.layout_indices, WORLD_COUNT),
            ("start", args.start_indices, START_COUNT)):
        if values is not None and (len(set(values)) != len(values)
                                   or any(not 0 <= value < limit
                                          for value in values)):
            parser.error(f"{name} indices must be unique in [0,{limit})")
    return args


def main() -> None:
    status = run(parse_args())
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    if status["state"] == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
