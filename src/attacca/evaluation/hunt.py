#!/usr/bin/env python3
"""Hunt benchmark: in a 21x21 Plains pen restored from a snapshot bank, the agent must kill an animal of the goal class."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO) + "/src")
if str(REPO / "src/attacca/evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "src/attacca/evaluation"))

os.environ.setdefault("MINESTUDIO_DIR", str(REPO / ".minestudio"))
os.environ.setdefault("MINESTUDIO_GPU_RENDER", "1")
os.environ.setdefault("RENDER_DEVICES", "0")

from attacca.worlds.eval_tail import POST_SUCCESS_FRAMES
from attacca.worlds.eval_tail import post_success_frames_recorded
from attacca.worlds.eval_tail import post_success_tail_complete
from attacca.worlds.human_viewmodel import encode_mask_runs_yx


CONTRACT = "xbench_hunt_center_spawn_target_scoped_21x21/v11"
LAYOUT_CONTRACT = "xbench_hunt_center_spawn_terrain_layout/v4"
FRAME_CONTRACT = "xbench_hunt_target_scoped_frame/v8"
DEFERRED_REVIEW_PAYLOAD_CONTRACT = "xbench_deferred_eval_review_payload/v1"
GOAL_CONTRACT = "xbench_hunt_cross_world_renderer_entity_id_goal/v2"
HUNT_SNAPSHOT_PAYLOAD_CONTRACT = "xbench_hunt_frozen_scene_snapshot/v2"
HUNT_SNAPSHOT_BANK_CONTRACT = "xbench_hunt_snapshot_bank/v2"
HUNT_OBJECT_ID = 0
HUNT_WEAPON_ITEM = "minecraft:iron_axe"
HUNT_WEAPON_NAME = "iron_axe"
HUNT_STAGED_HEALTH = 8.0
HUNT_REACH = 3.0
FPS = 20.0
DEFAULT_LAYOUTS = 10
DEFAULT_BUDGET = 360
DEFAULT_POST_SUCCESS_FRAMES = POST_SUCCESS_FRAMES
DEFAULT_BENCHMARK_SEED = 20260824
DEFAULT_WORLD_SEED_BASE = 940000
DEFAULT_SCENE_SEED_BASE = 2400
DEFAULT_POLICY_SEED_BASE = 2400
DEFAULT_RUNTIME_OVERLAY = REPO / "configs" / "xbench_runtime_eval"
MAX_ATTEMPTS = 3
PEN_OUTER_SIZE = 21
PEN_HALF_EXTENT = PEN_OUTER_SIZE // 2
STAGE_PAD_HALF_EXTENT = PEN_HALF_EXTENT + 2

TRAIN_TARGETS = (
    "cow", "pig", "chicken", "white_sheep", "gray_sheep",
    "light_gray_sheep", "brown_sheep",
)
ZERO_SHOT_TARGETS_A = (
    "blue_sheep", "purple_sheep", "mooshroom",
    "creamy_trader_llama", "white_trader_llama",
)
ZERO_SHOT_TARGETS_B = ("panda", "turtle", "llama", "donkey", "polar_bear")
ZERO_SHOT_TARGETS = ZERO_SHOT_TARGETS_A + ZERO_SHOT_TARGETS_B
TARGETS = TRAIN_TARGETS + ZERO_SHOT_TARGETS
MOB_SPECS = {
    "cow": {"entity": "cow", "nbt": ""},
    "pig": {"entity": "pig", "nbt": ""},
    "chicken": {"entity": "chicken", "nbt": ""},
    "white_sheep": {"entity": "sheep", "nbt": "Color:0b"},
    "gray_sheep": {"entity": "sheep", "nbt": "Color:7b"},
    "light_gray_sheep": {"entity": "sheep", "nbt": "Color:8b"},
    "brown_sheep": {"entity": "sheep", "nbt": "Color:12b"},
    "blue_sheep": {"entity": "sheep", "nbt": "Color:11b"},
    "purple_sheep": {"entity": "sheep", "nbt": "Color:10b"},
    "mooshroom": {"entity": "mooshroom", "nbt": 'Type:"red"'},
    "creamy_trader_llama": {"entity": "trader_llama", "nbt": "Variant:0"},
    "white_trader_llama": {"entity": "trader_llama", "nbt": "Variant:1"},
    "panda": {
        "entity": "panda",
        "nbt": 'MainGene:"normal",HiddenGene:"normal"',
    },
    "turtle": {"entity": "turtle", "nbt": ""},
    "llama": {"entity": "llama", "nbt": "Variant:0"},
    "donkey": {"entity": "donkey", "nbt": ""},
    "polar_bear": {"entity": "polar_bear", "nbt": ""},
}

LOCAL_START = (0.0, 0.0)
LOCAL_SLOTS = (
    (0.0, -5.0), (3.5, -3.5), (5.0, 0.0), (3.5, 3.5),
    (0.0, 5.0), (-3.5, 3.5), (-5.0, 0.0), (-3.5, -3.5),
)


def _slot_release_yaw(layout_rotation: int, slot_index: int) -> float:
    return float((int(layout_rotation) + 45 * int(slot_index)) % 360)


def _pose_inside_pen(pose: Mapping, bounds: Sequence[int]) -> bool:
    x0, x1, _, z0, z1 = [int(value) for value in bounds]
    return bool(
        float(x0 + 1) <= float(pose["x"]) <= float(x1)
        and float(z0 + 1) <= float(pose["z"]) <= float(z1))


def _base():
    import attacca.evaluation.hunt_phase as module
    return module


def atomic_json(path: Path, value: Mapping | Sequence) -> None:
    _base().atomic_json(path, value)


def sha256_file(path: Path) -> str:
    return _base().sha256_file(path)


def checkpoint_digest(path: Path) -> str:
    return _base().checkpoint_digest(path)


def json_safe(value: Any) -> Any:
    return _base().json_safe(value)


def normalize_name(value: Any) -> str:
    return _base().normalize_name(value)


def _purge_unowned_entities(world, *, preserve_scene: bool) -> None:
    selector = "@e[type=!player,tag=!xh_scene]" if preserve_scene else "@e[type=!player]"
    world.cmd(f"/execute as {selector} at @s run tp @s ~ -128 ~")
    world.cmd(f"/kill {selector}")


def _settle_frame_zero(world, *, fixed_ticks: int = 240,
                       stable_frames: int = 5,
                       maximum_seconds: float = 40.0) -> dict:
    started = time.monotonic()
    consecutive = 0
    previous_digest = None
    digest = None
    for _tick in range(int(fixed_ticks)):
        world.step_noop()
        rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        consecutive = consecutive + 1 if digest == previous_digest else 1
        previous_digest = digest
        if time.monotonic() - started >= float(maximum_seconds):
            raise RuntimeError(
                "frame-zero settle exceeded the wall-clock hang guard: "
                f"tick={_tick + 1}/{int(fixed_ticks)}")
    return {
        "ticks": int(fixed_ticks),
        "wall_seconds": float(time.monotonic() - started),
        "stable_consecutive_frames": int(consecutive),
        "requested_diagnostic_stable_frames": int(stable_frames),
        "diagnostic_stable": bool(consecutive >= int(stable_frames)),
        "final_rgb_bytes_sha256": digest,
        "settle_mode": "fixed_ticks_deterministic/v1",
    }


def normalized_counter(value: Any) -> dict[str, int]:
    return _base().normalized_counter(value)


def positive_counter_delta(current: Mapping[str, int],
                           initial: Mapping[str, int]) -> dict[str, int]:
    return _base().positive_counter_delta(current, initial)


def _rotate(x: float, z: float, degrees: int) -> tuple[float, float]:
    radians = math.radians(float(degrees))
    cosine, sine = math.cos(radians), math.sin(radians)
    return (cosine * x - sine * z, sine * x + cosine * z)


def episode_seed(namespace: str, *, target: str, world_seed: int,
                 action_seed: int) -> int:
    payload = (
        f"hunt-eval-redesign-v2\0{namespace}\0{target}\0"
        f"{int(world_seed)}\0{int(action_seed)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def episode_scene_targets(target: str) -> tuple[str, ...]:
    if target not in TARGETS:
        raise ValueError(f"unknown hunt target: {target!r}")
    roster = TRAIN_TARGETS if target in TRAIN_TARGETS else (target,) + TRAIN_TARGETS
    bystanders = tuple(value for value in roster if value != target)
    leaked = sorted(set(bystanders).intersection(ZERO_SHOT_TARGETS))
    if leaked:
        raise RuntimeError(f"OOD hunt bystander leak: {leaked}")
    return tuple(roster)


def build_layouts(*, count: int = DEFAULT_LAYOUTS,
                  benchmark_seed: int = DEFAULT_BENCHMARK_SEED,
                  world_seed_base: int = DEFAULT_WORLD_SEED_BASE,
                  policy_seed_base: int | None = None) -> list[dict]:
    if int(count) <= 0:
        raise ValueError("layout count must be positive")
    layouts = []
    for layout_index in range(int(count)):
        world_seed = int(world_seed_base + layout_index)
        rng = random.Random(episode_seed(
            "layout", target="__terrain__", world_seed=world_seed,
            action_seed=int(benchmark_seed)))
        rotation = rng.choice((0, 90, 180, 270))
        geometry = []
        for slot_index, (base_x, base_z) in enumerate(LOCAL_SLOTS):
            local_x = base_x + rng.uniform(-0.22, 0.22)
            local_z = base_z + rng.uniform(-0.18, 0.18)
            offset_x, offset_z = _rotate(local_x, local_z, rotation)
            geometry.append({
                "slot_index": int(slot_index),
                "local_xz": [float(local_x), float(local_z)],
                "rotated_offset_xz": [float(offset_x), float(offset_z)],
            })
        start_x, start_z = _rotate(*LOCAL_START, rotation)
        layouts.append({
            "contract": LAYOUT_CONTRACT,
            "layout_index": int(layout_index),
            "layout_id": f"layout_{layout_index:02d}",
            "benchmark_seed": int(benchmark_seed),
            "world_seed": world_seed,
            "rotation_degrees": int(rotation),
            "start_offset_xz": [float(start_x), float(start_z)],
            "start_yaw": float(rotation),
            "start_pitch": 4.0,
            "slot_geometry": geometry,
            "pen_outer_dimensions_blocks": [PEN_OUTER_SIZE, PEN_OUTER_SIZE],
            "episode_roster_is_target_scoped": True,
            "target_slot_index": 0,
            "target_behind_frame0_camera_by_construction": True,
        })
    return layouts


def validate_layout(layout: Mapping) -> None:
    if layout.get("contract") != LAYOUT_CONTRACT:
        raise ValueError("layout contract mismatch")
    geometry = list(layout.get("slot_geometry") or ())
    if len(geometry) != len(LOCAL_SLOTS):
        raise ValueError("layout must contain exactly eight ring slots")
    if [int(row["slot_index"]) for row in geometry] != list(range(len(LOCAL_SLOTS))):
        raise ValueError("layout ring-slot order mismatch")
    if list(layout.get("start_offset_xz") or ()) != [0.0, 0.0]:
        raise ValueError("hunt benchmark requires center spawn")
    if int(layout.get("target_slot_index", -1)) != 0:
        raise ValueError("target must use hidden ring slot zero")
    if list(layout.get("pen_outer_dimensions_blocks") or ()) != [
            PEN_OUTER_SIZE, PEN_OUTER_SIZE]:
        raise ValueError("hunt benchmark requires a 21x21 outer pen")


def episode_layout(layout: Mapping, target: str, *,
                   scene_seed: int | None = None,
                   policy_seed: int | None = None) -> dict:
    validate_layout(layout)
    scene_seed = 0 if scene_seed is None else int(scene_seed)
    policy_seed = 0 if policy_seed is None else int(policy_seed)
    roster = episode_scene_targets(target)
    geometry = list(layout["slot_geometry"])
    rng = random.Random(episode_seed(
        "bystander-order", target=target, world_seed=int(layout["world_seed"]),
        action_seed=scene_seed))
    bystanders = [value for value in TRAIN_TARGETS if value != target]
    rng.shuffle(bystanders)
    ordered = [target] + bystanders
    assignments = []
    for slot_index, kind in enumerate(ordered):
        assignments.append({
            "target": kind,
            "role": "target" if slot_index == 0 else "bystander",
            "slot_index": int(slot_index),
            "tag": f"xb_{kind}",
            "entity": MOB_SPECS[kind]["entity"],
            "nbt": MOB_SPECS[kind]["nbt"],
            "offset_xz": geometry[slot_index]["rotated_offset_xz"],
            "release_yaw": _slot_release_yaw(
                int(layout["rotation_degrees"]), slot_index),
        })
    bound = dict(layout)
    bound.update({
        "target": target,
        "scene_seed": scene_seed,
        "policy_sampling_seed": policy_seed,
        "policy_seed": episode_seed(
            "policy", target=target, world_seed=int(layout["world_seed"]),
            action_seed=policy_seed),
        "scene_targets": list(roster),
        "spawn_order": ordered,
        "assignments": assignments,
    })
    validate_episode_layout(bound, target)
    return bound


def validate_episode_layout(layout: Mapping, target: str) -> None:
    validate_layout(layout)
    expected = episode_scene_targets(target)
    assignments = list(layout.get("assignments") or ())
    actual = tuple(row.get("target") for row in assignments)
    if Counter(actual) != Counter(expected):
        raise ValueError(f"episode roster mismatch: {actual} != {expected}")
    if not assignments or assignments[0].get("target") != target:
        raise ValueError("episode target must occupy hidden slot zero")
    if int(assignments[0].get("slot_index", -1)) != 0:
        raise ValueError("episode target slot mismatch")
    leaked = set(actual[1:]).intersection(ZERO_SHOT_TARGETS)
    if leaked:
        raise ValueError(f"OOD hunt bystander leak: {sorted(leaked)}")


def write_layout_manifest(root: Path, layouts: Sequence[Mapping]) -> Path:
    path = Path(root) / "layouts.json"
    atomic_json(path, {
        "contract": LAYOUT_CONTRACT,
        "layouts_n": len(layouts),
        "targets": list(TARGETS),
        "train_targets": list(TRAIN_TARGETS),
        "zero_shot_targets": list(ZERO_SHOT_TARGETS),
        "layouts": list(layouts),
    })
    return path


def boot(seed: int):
    return _base().boot(
        int(seed), runtime_overlay=DEFAULT_RUNTIME_OVERLAY)


def _spec(target: str) -> dict:
    row = MOB_SPECS[str(target)]
    return {
        "name": str(target), "entity": str(row["entity"]),
        "tag": f"xb_{target}", "nbt": str(row["nbt"]),
    }


def _release_scene_ai(world) -> None:
    world.cmd(
        "/execute as @e[tag=xh_scene] run data merge entity @s "
        "{NoAI:0b}")


def _absolute_assignment(layout: Mapping, *, cx: int, feet_y: int,
                         cz: int) -> list[dict]:
    rows = []
    for row in layout["assignments"]:
        dx, dz = row["offset_xz"]
        rows.append({
            **dict(row),
            "position": [float(cx + float(dx)), float(feet_y),
                         float(cz + float(dz))],
        })
    return rows


def _face_pose(camera_xyz: Sequence[float], target_xyz: Sequence[float]) -> dict:
    dx = float(target_xyz[0]) - float(camera_xyz[0])
    dz = float(target_xyz[2]) - float(camera_xyz[2])
    horizontal = max(math.hypot(dx, dz), 1e-6)
    yaw = math.degrees(math.atan2(-dx, dz))
    player_eye = float(camera_xyz[1]) + 1.62
    target_eye = float(target_xyz[1]) + 0.8
    pitch = math.degrees(math.atan2(player_eye - target_eye, horizontal))
    return {
        "x": float(camera_xyz[0]), "y": float(camera_xyz[1]),
        "z": float(camera_xyz[2]), "yaw": float(yaw),
        "pitch": float(pitch),
    }


def _save_target_certification(world, target: str, out: Path) -> dict:
    import cv2
    from attacca.worlds.entity_id_mask import hunt_class_artifacts
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_VALID_FIELD
    if int(world.info.get(RENDERER_ENTITY_ID_VALID_FIELD, 0)) != 1:
        raise RuntimeError("target certification lacks renderer entity IDs")
    artifacts = hunt_class_artifacts(
        world.info[RENDERER_ENTITY_ID_FIELD], goal_kind=target)
    mask = np.asarray(artifacts["class_union_mask"], dtype=np.uint8)
    if int(artifacts["class_exist"]) != 1 or int(mask.sum()) <= 0:
        raise RuntimeError(
            f"target renderer binding/recognition failed for {target}: "
            f"pixels={int(mask.sum())}")
    rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (40, 255, 40), 2)
    _base()._save_rgb(out / "target_certification_rgb.png", rgb)
    _base()._save_mask(out / "target_certification_mask.png", mask)
    cv2.imwrite(str(out / "target_certification_overlay.png"), overlay)
    return {
        "target": target,
        "class_exist": 1,
        "exact_recognizable_pixels": int(mask.sum()),
        "mask_shape_hw": [int(mask.shape[0]), int(mask.shape[1])],
        "mask_bytes_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "rgb_bytes_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "visible_instance_ids": [
            row["instance_id"] for row in artifacts["visible_instances"]],
        "rgb": str((out / "target_certification_rgb.png").resolve()),
        "mask": str((out / "target_certification_mask.png").resolve()),
        "overlay": str((out / "target_certification_overlay.png").resolve()),
    }


def stage_layout(world, stager, layout: Mapping, target: str,
                 out: Path, *, scene_targets: Sequence[str] | None = None
                 ) -> dict:
    from attacca.worlds.entity_id_mask import ENTITY_ID_SEMANTIC_CODES
    from attacca.worlds.entity_id_mask import entity_surfaces
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_VALID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_SCENE_ALIVE_CLASSES_FIELD
    validate_episode_layout(layout, target)
    bound_scene_targets = tuple(layout["scene_targets"])
    scene_targets = bound_scene_targets if scene_targets is None else tuple(scene_targets)
    if scene_targets != bound_scene_targets:
        raise ValueError("caller scene roster differs from episode-scoped layout")
    player_x, player_y, player_z = world.get_pos()
    cx, cz = int(math.floor(player_x)), int(math.floor(player_z))
    feet_y = int(round(float(player_y)))
    x0, x1 = cx - PEN_HALF_EXTENT, cx + PEN_HALF_EXTENT
    z0, z1 = cz - PEN_HALF_EXTENT, cz + PEN_HALF_EXTENT

    world.cmd("/gamerule doMobSpawning false")
    _purge_unowned_entities(world, preserve_scene=False)
    pad = STAGE_PAD_HALF_EXTENT
    world.cmd(
        f"/fill {cx-pad} {feet_y} {cz-pad} "
        f"{cx+pad} {feet_y+8} {cz+pad} minecraft:air")
    world.cmd(
        f"/fill {cx-pad} {feet_y-1} {cz-pad} "
        f"{cx+pad} {feet_y-1} {cz+pad} minecraft:grass_block")
    world.cmd(f"/fill {x0} {feet_y} {z0} {x1} {feet_y} {z0} minecraft:oak_fence")
    world.cmd(f"/fill {x0} {feet_y} {z1} {x1} {feet_y} {z1} minecraft:oak_fence")
    world.cmd(f"/fill {x0} {feet_y} {z0+1} {x0} {feet_y} {z1-1} minecraft:oak_fence")
    world.cmd(f"/fill {x1} {feet_y} {z0+1} {x1} {feet_y} {z1-1} minecraft:oak_fence")
    cap_y = feet_y + 1
    world.cmd(f"/fill {x0} {cap_y} {z0} {x1} {cap_y} {z0} minecraft:barrier")
    world.cmd(f"/fill {x0} {cap_y} {z1} {x1} {cap_y} {z1} minecraft:barrier")
    world.cmd(f"/fill {x0} {cap_y} {z0+1} {x0} {cap_y} {z1-1} minecraft:barrier")
    world.cmd(f"/fill {x1} {cap_y} {z0+1} {x1} {cap_y} {z1-1} minecraft:barrier")
    for _ in range(8):
        world.step_noop()
    support = _base()._support_audit(world, x0, x1, feet_y - 1, z0, z1)

    stager.kit(f"{HUNT_WEAPON_ITEM} 1")
    _base().press_hotbar1(world)
    absolute = _absolute_assignment(layout, cx=cx, feet_y=feet_y, cz=cz)
    for row in absolute:
        xyz = row["position"]
        world.cmd(_base()._summon_command(
            _spec(row["target"]), xyz[0], xyz[1], xyz[2],
            health=HUNT_STAGED_HEALTH, no_ai=True,
            yaw=float(row["release_yaw"])))
    for _ in range(12):
        world.step_noop()
    ui_settle = _base().settle_wall_clock_ui(world)

    world.cmd("/gamerule sendCommandFeedback true")
    positions_by_tag = {}
    for row in absolute:
        observed = stager.entity_pos(
            str(row["entity"]), selector=f"tag={row['tag']},limit=1")
        if observed is None:
            raise RuntimeError(f"staged mob is absent: {row['target']}")
        positions_by_tag[row["tag"]] = [float(value) for value in observed]
    world.cmd("/gamerule sendCommandFeedback false")

    requested = next(row for row in absolute if row["target"] == target)
    target_xyz = requested["position"]
    toward_center_x = float(cx) - float(target_xyz[0])
    toward_center_z = float(cz) - float(target_xyz[2])
    base_bearing = math.atan2(toward_center_x, toward_center_z)
    other_positions = [
        row["position"] for row in absolute if row["target"] != target]
    certification_pose = None
    certification_candidate = None
    for bearing_off_deg in (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
        for dist in (2.6, 3.2, 2.1, 3.8):
            bearing = base_bearing + math.radians(bearing_off_deg)
            cand = [
                float(target_xyz[0]) + dist * math.sin(bearing),
                float(feet_y),
                float(target_xyz[2]) + dist * math.cos(bearing),
            ]
            if not (x0 + 1.0 <= cand[0] <= x1 - 1.0
                    and z0 + 1.0 <= cand[2] <= z1 - 1.0):
                continue
            if any(math.hypot(cand[0] - float(q[0]), cand[2] - float(q[2])) < 1.2
                   for q in other_positions):
                continue
            pose = _face_pose(cand, target_xyz)
            if stager.go(**pose, n=4):
                certification_pose = pose
                certification_candidate = {
                    "distance": float(dist),
                    "bearing_offset_deg": float(bearing_off_deg),
                    "contract": "certification_pose_search/v2",
                }
                break
        if certification_pose is not None:
            break
    if certification_pose is None:
        raise RuntimeError(f"could not set target certification pose: {target}")
    for _ in range(4):
        world.step_noop()
    target_certification = _save_target_certification(world, target, out)

    start_dx, start_dz = layout["start_offset_xz"]
    start = {
        "x": float(cx + float(start_dx)), "y": float(feet_y),
        "z": float(cz + float(start_dz)),
        "yaw": float(layout["start_yaw"]),
        "pitch": float(layout["start_pitch"]),
    }
    if not stager.go(**start, n=8):
        raise RuntimeError("could not restore paired frame-zero pose")
    frame0_purge_rounds = 0
    frame0_settle = None
    for purge_round in range(5):
        frame0_purge_rounds = purge_round + 1
        _purge_unowned_entities(world, preserve_scene=True)
        for _ in range(4 if purge_round == 0 else 2):
            world.step_noop()
        frame0_settle = _settle_frame_zero(world)
        if int(world.info.get(RENDERER_ENTITY_ID_VALID_FIELD, 0)) != 1:
            raise RuntimeError("frame zero lacks renderer entity IDs")
        surfaces = entity_surfaces(world.info[RENDERER_ENTITY_ID_FIELD])
        visible_target_surfaces = [
            {"kind": row.kind, "entity_id": int(row.entity_id),
             "pixels": int(row.area_px)}
            for row in surfaces if row.kind == target]
        if not visible_target_surfaces:
            break
    if visible_target_surfaces:
        import cv2
        evidence_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
        _base()._save_rgb(out / "frame0_violation_rgb.png", evidence_rgb)
        annotated = np.ascontiguousarray(evidence_rgb[:, :, ::-1]).copy()
        for row in surfaces:
            x0v, y0v, x1v, y1v = row.bbox_xyxy
            color = (0, 0, 255) if row.kind == target else (0, 200, 0)
            cv2.rectangle(annotated, (x0v - 2, y0v - 2), (x1v + 2, y1v + 2), color, 1)
            cv2.putText(annotated, f"{row.kind}:{int(row.entity_id)}:{int(row.area_px)}px",
                        (max(0, x0v - 2), max(10, y0v - 4)), 0, 0.38, color, 1)
        cv2.imwrite(str(out / "frame0_violation_annotated.png"), annotated)
        atomic_json(out / "frame0_violation_surfaces.json", [
            {"kind": row.kind, "entity_id": int(row.entity_id),
             "area_px": int(row.area_px), "bbox_xyxy": list(row.bbox_xyxy)}
            for row in surfaces])
        raise RuntimeError(
            "frame-zero candidate visibility contract failed: "
            f"{visible_target_surfaces}")
    held = normalize_name(
        ((world.info.get("equipped_items") or {}).get("mainhand") or {})
        .get("type"))
    if held != HUNT_WEAPON_NAME:
        raise RuntimeError(f"frame zero does not hold iron axe: {held!r}")
    expected_alive_classes_mask = sum(
        1 << (int(ENTITY_ID_SEMANTIC_CODES[row["target"]]) - 1)
        for row in absolute)
    observed_alive_classes_mask = int(
        world.info.get(RENDERER_SCENE_ALIVE_CLASSES_FIELD, -1))
    if observed_alive_classes_mask != expected_alive_classes_mask:
        raise RuntimeError(
            "frame zero scene alive-class census mismatch: "
            f"expected={expected_alive_classes_mask:#08x} "
            f"observed={observed_alive_classes_mask:#08x}")

    initial_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    _base()._save_rgb(out / "initial_policy_rgb.png", initial_rgb)
    initial_digest = hashlib.sha256(initial_rgb.tobytes()).hexdigest()
    observed_pose = _base()._pose(world)
    return {
        "contract": "xbench_hunt_stage_21x21/v5",
        "layout_id": layout["layout_id"],
        "world_seed": int(layout["world_seed"]),
        "pen_outer_dimensions_blocks": [PEN_OUTER_SIZE, PEN_OUTER_SIZE],
        "fence_bounds_inclusive": [x0, x1, feet_y, z0, z1],
        "containment_contract": "oak_fence_plus_invisible_barrier_cap/v1",
        "support_audit": support,
        "target_certification": target_certification,
        "target_certification_pose": certification_pose,
        "certification_pose_search": certification_candidate,
        "start_pose_requested": start,
        "start_pose_observed": observed_pose,
        "initial_policy_rgb": str((out / "initial_policy_rgb.png").resolve()),
        "initial_policy_rgb_bytes_sha256": initial_digest,
        "frame0_visible_target_surfaces": visible_target_surfaces,
        "scene_targets": list(scene_targets),
        "scene_mobs_n": len(scene_targets),
        "frame0_target_exactly_absent": True,
        "frame0_bystanders_may_be_visible": True,
        "held_item": held,
        "mob_health": HUNT_STAGED_HEALTH,
        "mob_max_health": HUNT_STAGED_HEALTH,
        "weapon": HUNT_WEAPON_ITEM,
        "one_hit_contract": (
            "fixed Health/max_health=8; fully cooled iron-axe direct hit kills"),
        "mob_ai": "frozen NoAI snapshot boundary; release not yet applied",
        "release_ai_applied": False,
        "wild_mob_policy": (
            "doMobSpawning=false + unowned entities teleported below world "
            "before kill; exact candidate surfaces rechecked at frame zero"),
        "frame0_purge_rounds": int(frame0_purge_rounds),
        "frame0_camera_settle": frame0_settle,
        "release_motion_xz": [0.0, 0.0],
        "release_contract": "natural_ai_no_forced_motion_slot_yaw/v1",
        "initial_scene_alive_classes_mask": int(observed_alive_classes_mask),
        "scene_alive_classes_contract": (
            "same_tick_client_entity_isAlive_semantic_class_bitmask/v1"),
        "release_yaw_by_unique_tag": {
            str(row["tag"]): float(row["release_yaw"]) for row in absolute},
        "spawn_positions_by_unique_tag": positions_by_tag,
        "absolute_assignments": absolute,
        "ui_settle": ui_settle,
    }


def _physical_scene_identity(layout: Mapping, target: str) -> dict:
    validate_episode_layout(layout, target)
    return {
        "contract": "xbench_hunt_physical_scene_identity/v1",
        "layout_contract": str(layout["contract"]),
        "layout_id": str(layout["layout_id"]),
        "layout_index": int(layout["layout_index"]),
        "benchmark_seed": int(layout["benchmark_seed"]),
        "world_seed": int(layout["world_seed"]),
        "rotation_degrees": int(layout["rotation_degrees"]),
        "start_offset_xz": list(layout["start_offset_xz"]),
        "start_yaw": float(layout["start_yaw"]),
        "start_pitch": float(layout["start_pitch"]),
        "slot_geometry": copy.deepcopy(list(layout["slot_geometry"])),
        "target_slot_index": int(layout["target_slot_index"]),
        "pen_outer_dimensions_blocks": list(
            layout["pen_outer_dimensions_blocks"]),
        "target": str(target),
        "scene_seed": int(layout["scene_seed"]),
        "scene_targets": list(layout["scene_targets"]),
        "spawn_order": list(layout["spawn_order"]),
        "assignments": copy.deepcopy(list(layout["assignments"])),
    }


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def hunt_snapshot_bundle_path(
        bank_root: Path, layout_id: str, target: str) -> Path:
    root = Path(bank_root).expanduser().resolve()
    return root / "bundles" / str(layout_id) / str(target)


def build_hunt_snapshot_payload(
        layout: Mapping, target: str, stage: Mapping) -> dict:
    if bool(stage.get("release_ai_applied")):
        raise ValueError("Hunt snapshot must be published before AI release")
    identity = _physical_scene_identity(layout, target)
    return {
        "contract": HUNT_SNAPSHOT_PAYLOAD_CONTRACT,
        "scene_identity": identity,
        "scene_identity_sha256": _canonical_sha256(identity),
        "target": str(target),
        "stage": copy.deepcopy(dict(stage)),
        "snapshot_boundary": (
            "center_frame0_certified_all_scene_mobs_NoAI1_before_release/v1"),
    }


def load_hunt_snapshot_bundle(
        bank_root: Path, layout: Mapping, target: str):
    from attacca.worlds.world_snapshot import load_world_snapshot_bundle

    path = hunt_snapshot_bundle_path(bank_root, layout["layout_id"], target)
    bundle = load_world_snapshot_bundle(path)
    payload = bundle.payload
    if not isinstance(payload, dict) or payload.get(
            "contract") != HUNT_SNAPSHOT_PAYLOAD_CONTRACT:
        raise ValueError(f"unsupported Hunt snapshot payload: {path}")
    expected = _physical_scene_identity(layout, target)
    if payload.get("scene_identity") != expected:
        raise ValueError(
            f"Hunt snapshot physical scene differs from request: {path}")
    if payload.get("scene_identity_sha256") != _canonical_sha256(expected):
        raise ValueError(f"Hunt snapshot scene identity digest mismatch: {path}")
    stage = payload.get("stage")
    if (not isinstance(stage, dict)
            or stage.get("frame0_target_exactly_absent") is not True
            or stage.get("release_ai_applied") is not False):
        raise ValueError(f"Hunt snapshot is not at the frozen frame0 boundary: {path}")
    return bundle


def _boot_hunt_snapshot(bundle, *, seed: int):
    from attacca.evaluation.policy import boot_world
    from attacca.evaluation.mine_scene import Stager
    from attacca.worlds.human_viewmodel import install_renderer_viewmodel_observation

    install_renderer_viewmodel_observation()
    for attempt in range(1, 4):
        world = None
        try:
            world = boot_world(
                int(seed), biome="plains", action_type="agent",
                runtime_overlay=DEFAULT_RUNTIME_OVERLAY,
                world_snapshot_dir=str(bundle.world.directory),
                world_snapshot_sha256=str(bundle.world.sha256),
                world_snapshot_archive=str(bundle.archive),
                world_snapshot_archive_sha256=str(
                    bundle.metadata["world_archive_sha256"]),
            )
            stager = Stager(world)
            world.cmd("/difficulty normal")
            world.cmd("/gamerule doMobSpawning false")
            world.cmd("/gamerule doMobLoot false")
            world.cmd("/gamerule randomTickSpeed 0")
            world.cmd("/gamerule doDaylightCycle false")
            world.cmd("/gamerule doWeatherCycle false")
            world.cmd("/weather clear")
            world.cmd("/time set 6000")
            world.cmd("/gamerule sendCommandFeedback false")
            return world, stager
        except BaseException as exc:
            if world is not None:
                try:
                    world.close()
                except BaseException:
                    pass
            broken_pipe = (
                isinstance(exc, BrokenPipeError)
                or "BrokenPipe" in repr(exc)
                or "Connection refused" in repr(exc))
            if not broken_pipe or attempt >= 3:
                raise
            print(
                f"[snapshot-boot] connection error on boot attempt {attempt}/3; "
                f"restarting the simulator for seed={seed}", flush=True)
    raise AssertionError("unreachable")


def _expected_scene_alive_mask(assignments: Sequence[Mapping]) -> int:
    from attacca.worlds.entity_id_mask import ENTITY_ID_SEMANTIC_CODES
    return sum(
        1 << (int(ENTITY_ID_SEMANTIC_CODES[str(row["target"])]) - 1)
        for row in assignments)


def _frozen_entity_selector(row: Mapping) -> str:
    nbt = [
        "NoAI:1b", "Silent:1b", "PersistenceRequired:1b",
        f"Health:{HUNT_STAGED_HEALTH:.1f}f",
    ]
    extra = str(row.get("nbt") or "").strip().strip(",")
    if extra:
        nbt.append(extra)
    return (
        f"type=minecraft:{row['entity']},tag=xh_scene,tag={row['tag']},"
        f"nbt={{{','.join(nbt)}}},limit=1")


def audit_frozen_hunt_snapshot(
        world, stager, bundle, layout: Mapping, target: str,
        out: Path, *, certify_target: bool) -> dict:
    from attacca.worlds.entity_id_mask import entity_surfaces
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_VALID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_SCENE_ALIVE_CLASSES_FIELD

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    payload = bundle.payload
    stage = dict(payload["stage"])
    assignments = list(stage["absolute_assignments"])
    expected_identity = _physical_scene_identity(layout, target)
    if payload.get("scene_identity") != expected_identity:
        raise ValueError("live Hunt snapshot audit received the wrong scene")

    expected_start = dict(stage["start_pose_requested"])
    observed_pose = _base()._pose(world)
    position_error = max(
        abs(float(observed_pose[key]) - float(expected_start[key]))
        for key in ("x", "y", "z"))
    yaw_error = abs(
        ((float(observed_pose["yaw"]) - float(expected_start["yaw"]) + 180.0)
         % 360.0) - 180.0)
    pitch_error = abs(
        float(observed_pose["pitch"]) - float(expected_start["pitch"]))
    if position_error > 1e-3 or yaw_error > 1e-3 or pitch_error > 1e-3:
        raise RuntimeError(
            "snapshot player pose mismatch: "
            f"position={position_error} yaw={yaw_error} pitch={pitch_error}")

    world.cmd("/gamerule sendCommandFeedback true")
    positions = {}
    for row in assignments:
        observed = None
        for _query_attempt in range(3):
            observed = stager.entity_pos(
                str(row["entity"]), selector=_frozen_entity_selector(row))
            if observed is not None:
                break
        if observed is None:
            raise RuntimeError(
                "snapshot mob/NBT audit failed after three bounded queries for "
                + str(row["target"]))
        values = [float(value) for value in observed]
        expected = stage["spawn_positions_by_unique_tag"][str(row["tag"])]
        delta = max(abs(values[index] - float(expected[index])) for index in range(3))
        if delta > 1e-6:
            raise RuntimeError(
                f"snapshot mob position mismatch for {row['target']}: {delta}")
        positions[str(row["tag"])] = values
    world.cmd("/gamerule sendCommandFeedback false")

    if int(world.info.get(RENDERER_ENTITY_ID_VALID_FIELD, 0)) != 1:
        raise RuntimeError("snapshot live audit lacks renderer entity IDs")
    expected_alive = _expected_scene_alive_mask(assignments)
    observed_alive = int(
        world.info.get(RENDERER_SCENE_ALIVE_CLASSES_FIELD, -1))
    if observed_alive != expected_alive:
        raise RuntimeError(
            "snapshot alive-class roster mismatch: "
            f"expected={expected_alive:#08x} observed={observed_alive:#08x}")
    visible_target = [
        row for row in entity_surfaces(world.info[RENDERER_ENTITY_ID_FIELD])
        if row.kind == target]
    if visible_target:
        raise RuntimeError(
            "snapshot center frame exposes target: "
            f"{[(row.kind, row.area_px) for row in visible_target]}")
    held = normalize_name(
        ((world.info.get("equipped_items") or {}).get("mainhand") or {})
        .get("type"))
    if held != HUNT_WEAPON_NAME:
        raise RuntimeError(f"snapshot does not restore iron axe: {held!r}")
    center_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    center_path = out / "center_frame0_rgb.png"
    _base()._save_rgb(center_path, center_rgb)
    center_sha = hashlib.sha256(center_rgb.tobytes()).hexdigest()

    certification = None
    if certify_target:
        pose = dict(stage["target_certification_pose"])
        if not stager.go(**pose, n=4):
            raise RuntimeError("could not restore snapshot target certification pose")
        for _ in range(4):
            world.step_noop()
        certification = _save_target_certification(world, target, out)
        reference_certification = dict(stage["target_certification"])
        if (certification.get("target") != target
                or int(certification.get("class_exist", 0)) != 1
                or certification.get("mask_shape_hw") != [360, 640]
                or len(certification.get("visible_instance_ids") or ()) != 1):
            raise RuntimeError(
                "snapshot target certification semantic gate failed: "
                f"target={certification.get('target')!r} "
                f"class_exist={certification.get('class_exist')!r} "
                f"shape={certification.get('mask_shape_hw')!r} "
                f"instances={certification.get('visible_instance_ids')!r}")
        certification["snapshot_reference_geometry"] = {
            "contract": (
                "xbench_hunt_living_model_mask_geometry_diagnostic/v1"),
            "reference_exact_recognizable_pixels": int(
                reference_certification["exact_recognizable_pixels"]),
            "observed_exact_recognizable_pixels": int(
                certification["exact_recognizable_pixels"]),
            "exact_recognizable_pixels_identical": (
                int(certification["exact_recognizable_pixels"])
                == int(reference_certification["exact_recognizable_pixels"])),
            "reference_mask_bytes_sha256": str(
                reference_certification["mask_bytes_sha256"]),
            "observed_mask_bytes_sha256": str(
                certification["mask_bytes_sha256"]),
            "mask_bytes_identical": (
                str(certification["mask_bytes_sha256"])
                == str(reference_certification["mask_bytes_sha256"])),
        }
        if not stager.go(**expected_start, n=8):
            raise RuntimeError("could not return to snapshot center pose")
        for _ in range(8):
            world.step_noop()
        if int(world.info.get(RENDERER_ENTITY_ID_VALID_FIELD, 0)) != 1:
            raise RuntimeError("snapshot returned center lacks renderer IDs")
        returned_target = [
            row for row in entity_surfaces(world.info[RENDERER_ENTITY_ID_FIELD])
            if row.kind == target]
        if returned_target:
            raise RuntimeError("snapshot target became visible after certification return")
        returned_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
        _base()._save_rgb(out / "returned_center_frame0_rgb.png", returned_rgb)

    audit = {
        "contract": "xbench_hunt_frozen_scene_live_audit/v2",
        "scene_identity_sha256": str(payload["scene_identity_sha256"]),
        "world_sha256": str(bundle.world.sha256),
        "world_archive_sha256": str(
            bundle.metadata["world_archive_sha256"]),
        "payload_sha256": str(bundle.metadata["payload_sha256"]),
        "layout_id": str(layout["layout_id"]),
        "target": str(target),
        "scene_seed": int(layout["scene_seed"]),
        "player_pose": observed_pose,
        "player_pose_max_position_error": float(position_error),
        "player_pose_yaw_error": float(yaw_error),
        "player_pose_pitch_error": float(pitch_error),
        "positions_by_unique_tag": positions,
        "expected_alive_classes_mask": int(expected_alive),
        "observed_alive_classes_mask": int(observed_alive),
        "frame0_target_exactly_absent": True,
        "held_item": held,
        "all_scene_mobs_NoAI1_nbt_verified": True,
        "center_rgb": str(center_path.resolve()),
        "center_rgb_bytes_sha256": center_sha,
        "target_certification": certification,
        "release_ai_applied": False,
    }
    atomic_json(out / "audit.json", audit)
    return audit


def cold_restore_frame0_gate(
        world, target: str, out: Path, *, expected_alive_mask: int,
        expected_held_item: str = HUNT_WEAPON_NAME) -> dict:
    from attacca.worlds.entity_id_mask import entity_surfaces
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_VALID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_SCENE_ALIVE_CLASSES_FIELD

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    final_settle = None
    surfaces = []
    visible_target_surfaces = []
    for round_index in range(5):
        _purge_unowned_entities(world, preserve_scene=True)
        for _ in range(4 if round_index == 0 else 2):
            world.step_noop()
        final_settle = _settle_frame_zero(world)
        if int(world.info.get(RENDERER_ENTITY_ID_VALID_FIELD, 0)) != 1:
            raise RuntimeError(
                "cold-restored Hunt frame zero lacks renderer entity IDs")
        surfaces = entity_surfaces(world.info[RENDERER_ENTITY_ID_FIELD])
        visible_target_surfaces = [
            {"kind": row.kind, "entity_id": int(row.entity_id),
             "pixels": int(row.area_px)}
            for row in surfaces if row.kind == target]
        if not visible_target_surfaces:
            break
    if visible_target_surfaces:
        rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
        _base()._save_rgb(out / "frame0_violation_rgb.png", rgb)
        atomic_json(out / "frame0_violation_surfaces.json", [
            {"kind": row.kind, "entity_id": int(row.entity_id),
             "area_px": int(row.area_px),
             "bbox_xyxy": list(row.bbox_xyxy)}
            for row in surfaces])
        raise RuntimeError(
            "cold-restored Hunt target-absence gate failed: "
            f"{visible_target_surfaces}")
    observed_alive = int(
        world.info.get(RENDERER_SCENE_ALIVE_CLASSES_FIELD, -1))
    if observed_alive != int(expected_alive_mask):
        raise RuntimeError(
            "cold-restored Hunt alive-class roster changed during settle: "
            f"expected={int(expected_alive_mask):#08x} "
            f"observed={observed_alive:#08x}")
    held = normalize_name(
        ((world.info.get("equipped_items") or {}).get("mainhand") or {})
        .get("type"))
    if held != str(expected_held_item):
        raise RuntimeError(
            f"cold-restored Hunt held item changed during settle: {held!r}")
    rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    frame_path = out / "ready_frame0_rgb.png"
    _base()._save_rgb(frame_path, rgb)
    result = {
        "contract": "xbench_hunt_cold_restore_frame0_gate/v1",
        "target": str(target),
        "purge_rounds": int(round_index + 1),
        "fixed_settle_ticks_per_round": 240,
        "total_fixed_settle_ticks": int((round_index + 1) * 240),
        "frame0_camera_settle": final_settle,
        "renderer_entity_ids_valid": True,
        "frame0_target_exactly_absent": True,
        "visible_target_surfaces": [],
        "expected_alive_classes_mask": int(expected_alive_mask),
        "observed_alive_classes_mask": int(observed_alive),
        "held_item": held,
        "ready_frame0_rgb": str(frame_path.resolve()),
        "ready_frame0_rgb_bytes_sha256": hashlib.sha256(
            rgb.tobytes()).hexdigest(),
        "no_hidden_noop_after_gate_before_release": True,
    }
    atomic_json(out / "cold_restore_gate.json", result)
    return result


def restored_stage_from_hunt_snapshot(
        world, stager, bundle, layout: Mapping, target: str,
        attempt_root: Path) -> dict:
    audit = audit_frozen_hunt_snapshot(
        world, stager, bundle, layout, target,
        Path(attempt_root) / "snapshot_restore_audit",
        certify_target=False)
    cold_gate = cold_restore_frame0_gate(
        world, target, Path(attempt_root) / "snapshot_cold_restore_gate",
        expected_alive_mask=int(audit["expected_alive_classes_mask"]),
        expected_held_item=str(audit["held_item"]))
    _release_scene_ai(world)
    stage = copy.deepcopy(dict(bundle.payload["stage"]))
    initial_path = Path(attempt_root) / "initial_policy_rgb.png"
    initial_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    _base()._save_rgb(initial_path, initial_rgb)
    stage.update({
        "contract": "xbench_hunt_snapshot_restore_stage/v2",
        "release_ai_applied": True,
        "mob_ai": "restored NoAI1 scene; one natural AI release before policy",
        "snapshot_restore_audit": audit,
        "snapshot_cold_restore_gate": cold_gate,
        "snapshot_bundle": str(bundle.directory),
        "snapshot_world_sha256": str(bundle.world.sha256),
        "snapshot_archive_sha256": str(
            bundle.metadata["world_archive_sha256"]),
        "snapshot_payload_sha256": str(bundle.metadata["payload_sha256"]),
        "initial_policy_rgb": str(initial_path.resolve()),
        "initial_policy_rgb_bytes_sha256": hashlib.sha256(
            initial_rgb.tobytes()).hexdigest(),
        "start_pose_observed": _base()._pose(world),
        "spawn_positions_by_unique_tag": dict(
            audit["positions_by_unique_tag"]),
        "frame0_purge_rounds": int(cold_gate["purge_rounds"]),
        "frame0_camera_settle": cold_gate["frame0_camera_settle"],
    })
    return stage


def _goal_meta(root: Path, target: str) -> dict:
    directory = Path(root) / "goals" / target
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("contract") != GOAL_CONTRACT or meta.get("target") != target:
        raise ValueError(f"goal metadata mismatch for {target}")
    for field in ("goal_rgb", "goal_mask"):
        path = Path(meta[field])
        if not path.is_file():
            path = directory / ("goal.png" if field == "goal_rgb" else "mask.png")
            meta[field] = str(path.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        digest_field = f"{field}_sha256"
        expected_digest = meta.get(digest_field)
        if (not isinstance(expected_digest, str)
                or sha256_file(path) != expected_digest):
            raise ValueError(f"goal artifact digest mismatch: {path}")
    return meta


def _selected_goal_targets(targets: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(target) for target in targets)
    if (not selected or len(set(selected)) != len(selected)
            or any(target not in TARGETS for target in selected)):
        raise ValueError(f"invalid Hunt goal target list: {selected}")
    return selected


def load_goals(
        root: Path, targets: Sequence[str] = TARGETS) -> dict[str, dict]:
    selected = _selected_goal_targets(targets)
    return {target: _goal_meta(root, target) for target in selected}


def _surface_contact(packed_ids: np.ndarray, distance: float | None,
                     attack: bool, target: str) -> dict:
    from attacca.worlds.entity_id_mask import ENTITY_ID_CODE_CLASSES
    from attacca.worlds.entity_id_mask import GOAL_ENTITY_KINDS
    value = int(np.asarray(packed_ids)[180, 320])
    semantic, entity_id = value >> 16, value & 0xffff
    kind = ENTITY_ID_CODE_CLASSES.get(semantic)
    in_reach = bool(distance is not None and float(distance) <= HUNT_REACH)
    contact = bool(attack and kind is not None and in_reach)
    target_contact = bool(
        contact and kind in set(GOAL_ENTITY_KINDS[target]))
    return {
        "crosshair_surface_kind": kind,
        "crosshair_entity_id": None if not entity_id else int(entity_id),
        "crosshair_world_surface_distance": (
            None if distance is None else float(distance)),
        "attack_contact_within_reach": bool(contact),
        "target_contact_attack": bool(target_contact),
        "wrong_contact_attack": bool(contact and not target_contact),
    }


def _target_kill_event(target: str) -> str:
    if target.endswith("sheep"):
        return "sheep"
    if target.endswith("trader_llama"):
        return "trader_llama"
    return target


def _scene_class_deaths(
        before_mask: int, after_mask: int,
        assignments: Sequence[Mapping[str, Any]]) -> dict:
    from attacca.worlds.entity_id_mask import ENTITY_ID_SEMANTIC_CODES

    before = int(before_mask)
    after = int(after_mask)
    if not (0 <= before < (1 << 32) and 0 <= after < (1 << 32)):
        raise ValueError("scene alive-class masks must be 32-bit values")
    by_code = {}
    for row in assignments:
        target = str(row["target"])
        code = int(ENTITY_ID_SEMANTIC_CODES[target])
        if code in by_code:
            raise ValueError("scene assignments contain a duplicate semantic class")
        by_code[code] = row
    if after & ~before:
        raise RuntimeError(
            "scene alive-class bit reappeared after death: "
            f"before={before:#04x} after={after:#04x}")
    known_mask = sum(1 << (code - 1) for code in by_code)
    if (before | after) & ~known_mask:
        raise RuntimeError("scene alive-class mask contains an unexpected class")
    died_mask = before & ~after
    died = [
        {
            "semantic_code": code,
            "target": str(by_code[code]["target"]),
            "role": str(by_code[code]["role"]),
        }
        for code in sorted(by_code) if died_mask & (1 << (code - 1))
    ]
    return {
        "before_mask": before,
        "after_mask": after,
        "died_mask": died_mask,
        "deaths": died,
        "target_death_event": any(
            row["role"] == "target" for row in died),
        "wrong_death_targets": [
            row["target"] for row in died if row["role"] != "target"],
        "contract": "same_tick_client_entity_isAlive_class_transition/v1",
    }


def _alive_and_positions(
        world, stager, *, scene_targets: Sequence[str]
) -> tuple[dict[str, bool], dict[str, list[float] | None]]:
    world.cmd("/gamerule sendCommandFeedback true")
    alive, positions = {}, {}
    for target in scene_targets:
        spec = _spec(target)
        exists = bool(stager.entity_exists(f"tag={spec['tag']},limit=1"))
        alive[spec["tag"]] = exists
        observed = None
        if exists:
            observed = stager.entity_pos(
                spec["entity"], selector=f"tag={spec['tag']},limit=1")
        positions[spec["tag"]] = (
            None if observed is None else [float(value) for value in observed])
    world.cmd("/gamerule sendCommandFeedback false")
    return alive, positions


def _verdict(target: str, kill_delta: Mapping[str, int],
             rows: Sequence[Mapping], *,
             scene_targets: Sequence[str],
             score_end_step: int) -> dict:
    scoring_rows = [
        row for row in rows
        if int(row["step"]) <= int(score_end_step)]
    killed_set = {
        str(death["target"])
        for row in scoring_rows
        for death in row.get("scene_class_deaths", {}).get("deaths", [])
    }
    killed = [
        candidate for candidate in scene_targets
        if candidate in killed_set]
    wrong_kills = [candidate for candidate in killed if candidate != target]
    target_killed = target in killed
    event = _target_kill_event(target)
    event_count = int(kill_delta.get(event, 0))
    contacts = [
        row for row in scoring_rows if row["attack_contact_within_reach"]]
    wrong_contacts = [row for row in contacts if row["wrong_contact_attack"]]
    target_contacts = [row for row in contacts if row["target_contact_attack"]]
    first_contact = contacts[0] if contacts else None
    success = bool(target_killed)
    clean = bool(target_killed and not wrong_kills)
    return {
        "target": target,
        "target_event": event,
        "target_event_count": event_count,
        "target_killed": bool(target_killed),
        "killed_tagged_mobs": killed,
        "wrong_kills": wrong_kills,
        "target_kill_success": int(success),
        "clean_success_no_wrong_kill": int(clean),
        "success_uses_crosshair_contact": False,
        "success_uses_kill_counter": False,
        "wrong_attack": int(bool(wrong_contacts)),
        "wrong_contact_attack_count": len(wrong_contacts),
        "target_contact_attack_count": len(target_contacts),
        "first_hit_accuracy": (
            None if first_contact is None else
            int(bool(first_contact["target_contact_attack"]))),
        "first_contact_step": (
            None if first_contact is None else int(first_contact["step"])),
        "first_contact_surface_kind": (
            None if first_contact is None else
            first_contact["crosshair_surface_kind"]),
        "kill_delta": dict(kill_delta),
    }


def run_episode(*, attempt_root: Path, layout: Mapping, target: str,
                goal_meta: Mapping, checkpoint: Path, budget: int,
                post_success_frames: int,
                checkpoint_sha256: str, cfg_coef: float,
                model_label: str, snapshot_bundle,
                scene_targets: Sequence[str] | None = None,
                runner=None):
    import cv2
    import torch
    from attacca.evaluation.policy import Rocket2GoalRunner
    from attacca.evaluation.prediction import ReviewFrameGuard
    from attacca.evaluation.prediction import capture_model_prediction
    from attacca.worlds.entity_id_mask import hunt_visibility_phase_artifacts
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_VALID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_SCENE_ALIVE_CLASSES_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_VIEWMODEL_DEPTH_FIELD

    episode_started = time.monotonic()
    timings = {
        "model_setup_seconds": 0.0,
        "boot_seconds": 0.0,
        "stage_seconds": 0.0,
        "snapshot_restore_audit_seconds": 0.0,
        "video_writer_open_seconds": 0.0,
        "policy_inference_seconds": 0.0,
        "exact_label_seconds": 0.0,
        "policy_sim_step_seconds": 0.0,
        "raw_video_write_seconds": 0.0,
        "frame_json_write_seconds": 0.0,
        "post_policy_sim_step_seconds": 0.0,
        "terminal_audit_seconds": 0.0,
    }
    validate_episode_layout(layout, target)
    bound_scene_targets = tuple(layout["scene_targets"])
    scene_targets = bound_scene_targets if scene_targets is None else tuple(scene_targets)
    if scene_targets != bound_scene_targets:
        raise ValueError("run scene roster differs from episode-scoped layout")
    attempt_root.mkdir(parents=True, exist_ok=False)
    goal_bgr = cv2.imread(str(goal_meta["goal_rgb"]), cv2.IMREAD_COLOR)
    goal_mask = cv2.imread(str(goal_meta["goal_mask"]), cv2.IMREAD_GRAYSCALE)
    if goal_bgr is None or goal_mask is None:
        raise RuntimeError(f"could not decode goal pair for {target}")
    goal_rgb = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)
    policy_seed = int(layout["policy_seed"])
    random.seed(policy_seed)
    np.random.seed(policy_seed & 0xFFFFFFFF)
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    model_started = time.monotonic()
    runner_reused = runner is not None
    if runner is None:
        runner = Rocket2GoalRunner(
            str(Path(checkpoint).resolve()), cfg_coef=float(cfg_coef),
            device="cuda")
    runner.set_obj_id(HUNT_OBJECT_ID)
    runner.reset()
    timings["model_setup_seconds"] = float(
        time.monotonic() - model_started)

    boot_started = time.monotonic()
    world, stager = _boot_hunt_snapshot(
        snapshot_bundle, seed=int(layout["world_seed"]))
    timings["boot_seconds"] = float(time.monotonic() - boot_started)
    rows = []
    try:
        stage_started = time.monotonic()
        stage = restored_stage_from_hunt_snapshot(
            world, stager, snapshot_bundle, layout, target,
            attempt_root)
        timings["snapshot_restore_audit_seconds"] = float(
            time.monotonic() - stage_started)
        timings["stage_seconds"] = float(time.monotonic() - stage_started)
        initial_kills = normalized_counter(world.info.get("kill_entity", {}))
        initial_rgb_digest = stage["initial_policy_rgb_bytes_sha256"]
        raw_path = attempt_root / "raw_policy.mp4"
        jsonl_path = attempt_root / "frames_actions_phase_kills.jsonl"
        writer_started = time.monotonic()
        raw_writer = cv2.VideoWriter(
            str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 360))
        if not raw_writer.isOpened():
            raise RuntimeError("could not open benchmark video writers")
        timings["video_writer_open_seconds"] = float(
            time.monotonic() - writer_started)
        guard = ReviewFrameGuard()
        first_discovery_step = None
        first_target_contact_step = None
        first_wrong_contact_step = None
        first_kill_event_step = None
        kill_delta: dict[str, int] = {}
        previous_kills = dict(initial_kills)
        held_item_lost_step = None
        first_player_outside_pen_step = None
        first_behavioural_failure_step = None
        first_wrong_kill_step = None
        wrong_death_targets: set[str] = set()
        target_death_step = None
        post_success_tail_start_step = None
        success_tail_cancelled_by_failure = False
        scoring_kill_delta: dict[str, int] = {}
        phase_available = None
        started = time.monotonic()
        terminated = truncated = False
        try:
            with jsonl_path.open("w", encoding="utf-8") as stream:
                for step in range(int(budget) + int(post_success_frames)):
                    if (step >= int(budget)
                            and post_success_tail_start_step is None):
                        break
                    post_success_observation = target_death_step is not None
                    pre_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
                    pre_pose = _base()._pose(world)
                    player_inside_pen = _pose_inside_pen(
                        pre_pose, stage["fence_bounds_inclusive"])
                    if (not post_success_observation
                            and not player_inside_pen
                            and first_player_outside_pen_step is None):
                        first_player_outside_pen_step = int(step)
                    if int(world.info.get(RENDERER_ENTITY_ID_VALID_FIELD, 0)) != 1:
                        raise RuntimeError("rollout frame lacks renderer entity IDs")
                    pre_alive_classes_mask = int(
                        world.info.get(RENDERER_SCENE_ALIVE_CLASSES_FIELD, -1))
                    held = normalize_name(
                        ((world.info.get("equipped_items") or {}).get("mainhand") or {})
                        .get("type"))
                    if (not post_success_observation
                            and held != HUNT_WEAPON_NAME
                            and held_item_lost_step is None):
                        held_item_lost_step = int(step)
                    inference_started = time.monotonic()
                    action, returned_exist = runner.act(
                        world.obs, goal_rgb, goal_mask)
                    prediction = capture_model_prediction(
                        runner.agent.cache_latents, require_exist=True)
                    probabilities = _base().phase_probabilities(
                        runner.agent.cache_latents, required=False)
                    available_now = isinstance(probabilities, list)
                    if phase_available is None:
                        phase_available = available_now
                    elif phase_available != available_now:
                        raise RuntimeError("phase-head availability changed")
                    prediction["phase_probabilities"] = probabilities
                    prediction["phase_order"] = list(_base().PHASE_NAMES)
                    prediction["phase_argmax"] = (
                        None if probabilities is None else
                        _base().PHASE_NAMES[int(np.argmax(probabilities))])
                    timings["policy_inference_seconds"] += float(
                        time.monotonic() - inference_started)
                    env_action = world.sim.agent_action_to_env_action(action)
                    attack = _base()._action_flag(env_action, "attack")
                    label_started = time.monotonic()
                    distance = _base().renderer_surface_distance_at_pixel(
                        world.info[RENDERER_VIEWMODEL_DEPTH_FIELD], (320, 180))
                    label = hunt_visibility_phase_artifacts(
                        world.info[RENDERER_ENTITY_ID_FIELD], goal_kind=target,
                        attack=bool(attack), target_surface_distance=distance,
                        interaction_reach=HUNT_REACH,
                        crosshair_xy=(320, 180))
                    contact = _surface_contact(
                        world.info[RENDERER_ENTITY_ID_FIELD], distance,
                        bool(attack), target)
                    timings["exact_label_seconds"] += float(
                        time.monotonic() - label_started)
                    if int(label["class_exist"]) and first_discovery_step is None:
                        first_discovery_step = int(step)
                    if contact["target_contact_attack"] and first_target_contact_step is None:
                        first_target_contact_step = int(step)
                    if contact["wrong_contact_attack"] and first_wrong_contact_step is None:
                        first_wrong_contact_step = int(step)

                    sim_step_started = time.monotonic()
                    world.obs, reward, terminated, truncated, world.info = (
                        world.sim.step(action))
                    timings["policy_sim_step_seconds"] += float(
                        time.monotonic() - sim_step_started)
                    current_kills = normalized_counter(
                        world.info.get("kill_entity", {}))
                    step_kill_delta = positive_counter_delta(
                        current_kills, previous_kills)
                    previous_kills = current_kills
                    kill_delta = positive_counter_delta(
                        current_kills, initial_kills)
                    if not post_success_observation:
                        scoring_kill_delta = dict(kill_delta)
                    after_alive_classes_mask = int(
                        world.info.get(RENDERER_SCENE_ALIVE_CLASSES_FIELD, -1))
                    class_deaths = _scene_class_deaths(
                        pre_alive_classes_mask, after_alive_classes_mask,
                        stage["absolute_assignments"])
                    target_death_event = bool(
                        class_deaths["target_death_event"])
                    newly_wrong_dead = list(
                        class_deaths["wrong_death_targets"])
                    if not post_success_observation:
                        wrong_death_targets.update(newly_wrong_dead)
                        if newly_wrong_dead and first_wrong_kill_step is None:
                            first_wrong_kill_step = int(step)
                        behavioural_failure_now = bool(
                            newly_wrong_dead
                            or held_item_lost_step is not None
                            or first_player_outside_pen_step is not None)
                        if (behavioural_failure_now
                                and first_behavioural_failure_step is None):
                            first_behavioural_failure_step = int(step)
                    if target_death_event:
                        if target_death_step is not None:
                            raise RuntimeError("target scene slot died more than once")
                        target_death_step = int(step)
                        post_success_tail_start_step = int(step)
                    if (class_deaths["deaths"]
                            and first_kill_event_step is None):
                        first_kill_event_step = int(step)
                    video_index, drop_reason = guard.assess(pre_rgb, pre_pose)
                    note = None
                    if contact["wrong_contact_attack"]:
                        note = f"WRONG CONTACT: {contact['crosshair_surface_kind']}"
                    elif contact["target_contact_attack"]:
                        note = "TARGET CONTACT ATTACK"
                    if video_index is not None:
                        raw_write_started = time.monotonic()
                        raw_writer.write(cv2.cvtColor(pre_rgb, cv2.COLOR_RGB2BGR))
                        timings["raw_video_write_seconds"] += float(
                            time.monotonic() - raw_write_started)
                    row = {
                        "contract": FRAME_CONTRACT,
                        "layout_id": layout["layout_id"],
                        "target": target,
                        "step": int(step),
                        "post_success_observation": bool(
                            post_success_observation),
                        "video_frame_index": video_index,
                        "video_drop_reason": drop_reason,
                        "pose": pre_pose,
                        "player_inside_pen": bool(player_inside_pen),
                        "policy_action": json_safe(action),
                        "env_action": json_safe(env_action),
                        "attack": bool(attack),
                        "held_item_before_action": held,
                        "returned_exist_probability": float(returned_exist),
                        "prediction": json_safe(prediction),
                        "phase_probabilities": probabilities,
                        "phase_label": int(label["phase"]),
                        "phase_name": _base().PHASE_NAMES[int(label["phase"])],
                        "class_exist": int(label["class_exist"]),
                        "class_union_exact_pixels": int(
                            np.asarray(label["class_union_mask"]).sum()),
                        "class_union_mask_runs_yx": encode_mask_runs_yx(
                            np.ascontiguousarray(
                                label["class_union_mask"], dtype=np.uint8)),
                        "class_union_mask_shape_hw": [360, 640],
                        "target_surface_distance": label["target_surface_distance"],
                        **contact,
                        "scene_alive_classes_before_action": int(
                            pre_alive_classes_mask),
                        "scene_alive_classes_after_action": int(
                            after_alive_classes_mask),
                        "scene_class_deaths": class_deaths,
                        "step_kill_counter_delta": step_kill_delta,
                        "kill_delta_after_action": kill_delta,
                        "review_note": note,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                    rows.append(row)
                    row["behavioural_failure_latched"] = bool(
                        first_behavioural_failure_step is not None)
                    row["success_event"] = bool(target_death_event)
                    row["clean_success_event"] = bool(
                        target_death_event
                        and post_success_tail_start_step == int(step))
                    json_started = time.monotonic()
                    stream.write(json.dumps(
                        row, sort_keys=True, separators=(",", ":"),
                        allow_nan=False) + "\n")
                    stream.flush()
                    timings["frame_json_write_seconds"] += float(
                        time.monotonic() - json_started)
                    if terminated or truncated:
                        break
                    if post_success_tail_complete(
                            post_success_tail_start_step, step,
                            post_success_frames):
                        break
            for _ in range(30 if kill_delta else 4):
                post_step_started = time.monotonic()
                world.step_noop()
                timings["post_policy_sim_step_seconds"] += float(
                    time.monotonic() - post_step_started)
        finally:
            raw_writer.release()

        terminal_audit_started = time.monotonic()
        alive, terminal_positions = _alive_and_positions(
            world, stager, scene_targets=scene_targets)
        verdict = _verdict(
            target, scoring_kill_delta, rows,
            scene_targets=scene_targets,
            score_end_step=(
                target_death_step if target_death_step is not None
                else int(budget) - 1))
        verdict["arena_escape"] = bool(
            first_player_outside_pen_step is not None)
        terminal_target_killed = bool(
            alive.get(f"xb_{target}") is False)
        telemetry_target_killed = target_death_step is not None
        terminal_target_state_disagreement = bool(
            terminal_target_killed != telemetry_target_killed)
        verdict["terminal_target_state_disagreement"] = bool(
            terminal_target_state_disagreement)
        timings["terminal_audit_seconds"] = float(
            time.monotonic() - terminal_audit_started)
        displacement = {}
        for tag, initial in stage["spawn_positions_by_unique_tag"].items():
            final = terminal_positions.get(tag)
            displacement[tag] = (
                None if final is None else
                float(math.hypot(final[0] - initial[0], final[2] - initial[2])))
        result = {
            "contract": CONTRACT,
            "state": "completed",
            "layout_id": layout["layout_id"],
            "layout_index": int(layout["layout_index"]),
            "world_seed": int(layout["world_seed"]),
            "scene_seed": int(layout["scene_seed"]),
            "policy_sampling_seed": int(layout["policy_sampling_seed"]),
            "policy_seed": int(layout["policy_seed"]),
            "snapshot_restore_used": True,
            "snapshot_bundle": str(snapshot_bundle.directory),
            "target": target,
            "split": "train" if target in TRAIN_TARGETS else "zero_shot",
            "hunt_object_id": HUNT_OBJECT_ID,
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": str(checkpoint_sha256),
            "model_label": str(model_label),
            "cfg_coef": float(cfg_coef),
            "budget": int(budget),
            "post_success_frames_requested": int(post_success_frames),
            "post_success_tail_is_non_scoring": True,
            "success_step": target_death_step,
            "target_death_step": target_death_step,
            "post_success_tail_start_step": post_success_tail_start_step,
            "success_tail_cancelled_by_failure": bool(
                success_tail_cancelled_by_failure),
            "post_success_frames_recorded": (
                post_success_frames_recorded(
                    target_death_step, len(rows))),
            "steps_executed": len(rows),
            "end_reason": (
                "target_death_plus_post_tail" if (
                    post_success_tail_complete(
                        post_success_tail_start_step, len(rows) - 1,
                        post_success_frames)) else
                "sim_terminated" if terminated or truncated else
                "budget_exhausted"),
            "first_target_recognizable_step": first_discovery_step,
            "time_to_first_target_frames": first_discovery_step,
            "first_target_contact_step": first_target_contact_step,
            "first_wrong_contact_step": first_wrong_contact_step,
            "first_kill_event_step": first_kill_event_step,
            "first_wrong_kill_step": first_wrong_kill_step,
            "time_to_kill_frames": target_death_step,
            "target_death_contract": (
                "same_tick_client_entity_isAlive_target_class_transition/v1"),
            "kill_counter_is_diagnostic_only": True,
            "terminal_target_state_disagreement": bool(
                terminal_target_state_disagreement),
            "policy_loop_hidden_sim_steps": 0,
            "timing": {
                **timings,
                "runner_reused": bool(runner_reused),
                "policy_frames": len(rows),
                "episode_preclose_wall_seconds": float(
                    time.monotonic() - episode_started),
            },
            "held_item_lost_step": held_item_lost_step,
            "first_player_outside_pen_step": first_player_outside_pen_step,
            "first_behavioural_failure_step": first_behavioural_failure_step,
            "verdict": verdict,
            "target_kill_success": int(bool(telemetry_target_killed)),
            "strict_success": int(bool(
                telemetry_target_killed
                and not verdict["wrong_kills"]
                and held_item_lost_step is None
                and first_player_outside_pen_step is None)),
            "wrong_attack": verdict["wrong_attack"],
            "wrong_kill": int(bool(verdict["wrong_kills"])),
            "wrong_death_targets_during_policy": sorted(
                wrong_death_targets),
            "first_hit_accuracy": verdict["first_hit_accuracy"],
            "alive_by_tag_after_settle": alive,
            "terminal_positions_by_tag": terminal_positions,
            "horizontal_displacement_from_spawn_by_tag": displacement,
            "live_mob_movement_observed": bool(any(
                value is not None and value >= 0.10
                for value in displacement.values())),
            "phase_head_available": bool(phase_available),
            "phase": _base().phase_summary(rows),
            "stage": stage,
            "initial_policy_rgb_bytes_sha256": initial_rgb_digest,
            "goal": dict(goal_meta),
            "raw_video": str(raw_path.resolve()),
            "review_mode": "deferred",
            "review_state": "pending",
            "review_video": None,
            "deferred_review_payload_contract": (
                DEFERRED_REVIEW_PAYLOAD_CONTRACT),
            "post_policy_raw_video": None,
            "post_policy_frame_jsonl": None,
            "frame_jsonl": str(jsonl_path.resolve()),
            "kept_video_frames": int(guard.kept_frames),
            "dropped_video_frames": int(guard.dropped_frames),
            "rollout_wall_seconds": float(time.monotonic() - started),
        }
        atomic_json(attempt_root / "stage_audit.json", stage)
        atomic_json(attempt_root / "result.json", result)
        return result, runner
    finally:
        world.close()


def _next_attempt_indices(task_root: Path, count: int) -> tuple[int, ...]:
    if int(count) <= 0:
        raise ValueError("attempt count must be positive")
    used = []
    if Path(task_root).is_dir():
        for entry in Path(task_root).iterdir():
            if not entry.is_dir() or not entry.name.startswith("attempt_"):
                continue
            suffix = entry.name[len("attempt_"):]
            if suffix.isdigit():
                used.append(int(suffix))
    start = max(used, default=-1) + 1
    return tuple(range(start, start + int(count)))


_PARALLEL_RUNNER = None
_PARALLEL_CHECKPOINT: Path | None = None
_PARALLEL_CHECKPOINT_SHA256 = ""
_PARALLEL_BUDGET = DEFAULT_BUDGET
_PARALLEL_POST = DEFAULT_POST_SUCCESS_FRAMES
_PARALLEL_CFG_COEF = 0.0
_PARALLEL_MODEL_LABEL = ""
_PARALLEL_SNAPSHOT_ROOT: Path | None = None


def _parallel_worker_init(
        checkpoint: str, checkpoint_sha256: str, budget: int,
        post_success_frames: int, cfg_coef: float, model_label: str,
        snapshot_root: str) -> None:
    global _PARALLEL_RUNNER, _PARALLEL_CHECKPOINT
    global _PARALLEL_CHECKPOINT_SHA256, _PARALLEL_BUDGET, _PARALLEL_POST
    global _PARALLEL_CFG_COEF, _PARALLEL_MODEL_LABEL
    global _PARALLEL_SNAPSHOT_ROOT
    from attacca.evaluation.policy import Rocket2GoalRunner

    _PARALLEL_CHECKPOINT = Path(checkpoint).resolve()
    _PARALLEL_CHECKPOINT_SHA256 = str(checkpoint_sha256)
    _PARALLEL_BUDGET = int(budget)
    _PARALLEL_POST = int(post_success_frames)
    _PARALLEL_CFG_COEF = float(cfg_coef)
    _PARALLEL_MODEL_LABEL = str(model_label)
    _PARALLEL_SNAPSHOT_ROOT = Path(snapshot_root).resolve()
    _PARALLEL_RUNNER = Rocket2GoalRunner(
        str(_PARALLEL_CHECKPOINT), cfg_coef=_PARALLEL_CFG_COEF,
        device="cuda")


def _parallel_run_task(arguments) -> dict:
    global _PARALLEL_RUNNER
    layout, target, goal_meta, root, max_attempts = arguments
    layout = dict(layout)
    target = str(target)
    task_root = Path(root) / "rollouts" / str(layout["layout_id"]) / target
    task_root.mkdir(parents=True, exist_ok=True)
    task_result_path = task_root / "task_result.json"
    last_failure = None
    attempt_indices = _next_attempt_indices(task_root, int(max_attempts))
    for attempt_index in attempt_indices:
        attempt_root = task_root / f"attempt_{attempt_index:02d}"
        try:
            snapshot_bundle = load_hunt_snapshot_bundle(
                _PARALLEL_SNAPSHOT_ROOT, layout, target)
            result, _PARALLEL_RUNNER = run_episode(
                attempt_root=attempt_root, layout=layout, target=target,
                goal_meta=goal_meta, checkpoint=_PARALLEL_CHECKPOINT,
                checkpoint_sha256=_PARALLEL_CHECKPOINT_SHA256,
                budget=_PARALLEL_BUDGET,
                post_success_frames=_PARALLEL_POST,
                cfg_coef=_PARALLEL_CFG_COEF,
                model_label=_PARALLEL_MODEL_LABEL,
                scene_targets=tuple(layout["scene_targets"]),
                runner=_PARALLEL_RUNNER,
                snapshot_bundle=snapshot_bundle)
            atomic_json(task_result_path, result)
            return {"result": result, "failure": None}
        except BaseException as exc:
            last_failure = {
                "layout_id": layout["layout_id"], "target": target,
                "attempt": int(attempt_index), "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_json(attempt_root / "FAILED.json", last_failure)
            _PARALLEL_RUNNER = None
    return {
        "result": None,
        "failure": last_failure or {
            "layout_id": layout["layout_id"], "target": target,
            "error": "all newly allocated attempt slots failed",
            "allocated_attempts": list(attempt_indices),
        },
    }


def run(args) -> dict:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not (checkpoint.is_file() or checkpoint.is_dir()):
        raise FileNotFoundError(checkpoint)
    root = Path(args.out_root).expanduser().resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(f"refusing existing output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    snapshot_bank_root = Path(args.snapshot_bank_root).expanduser().resolve()
    if not snapshot_bank_root.is_dir():
        raise FileNotFoundError(snapshot_bank_root)
    layouts = build_layouts(
        count=int(args.layouts), benchmark_seed=int(args.benchmark_seed),
        world_seed_base=int(args.world_seed_base))
    for layout in layouts:
        validate_layout(layout)
    layout_path = root / "layouts.json"
    if layout_path.exists():
        existing = json.loads(layout_path.read_text(encoding="utf-8"))
        if existing.get("layouts") != layouts:
            raise ValueError("resume layout manifest differs from requested benchmark")
    else:
        write_layout_manifest(root, layouts)
    selected_targets = tuple(args.targets)
    if (not selected_targets or len(set(selected_targets)) != len(selected_targets)
            or any(target not in TARGETS for target in selected_targets)):
        raise ValueError("invalid selected target list")
    selected_layout_indices = tuple(range(len(layouts)))
    selected_layouts = [layouts[index] for index in selected_layout_indices]
    manifest = {
        "contract": CONTRACT,
        "created_unix_s": time.time(),
        "machine": _base().machine_record(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        "model_label": str(args.model_label),
        "cfg_coef": float(args.cfg_coef),
        "layouts_n": len(selected_layouts),
        "layout_indices": list(selected_layout_indices),
        "full_layout_manifest_layouts_n": len(layouts),
        "targets": list(selected_targets),
        "full_benchmark_targets": list(TARGETS),
        "train_targets": list(TRAIN_TARGETS),
        "zero_shot_targets": list(ZERO_SHOT_TARGETS),
        "expected_episodes": len(selected_layouts) * len(selected_targets),
        "scene_seed_base": int(args.scene_seed_base),
        "policy_seed_base": int(args.policy_seed_base),
        "workers": int(args.workers),
        "seed_separation_contract": (
            "scene_seed_controls_physical_roster; "
            "policy_seed_controls_action_sampling_only/v1"),
        "snapshot_restore_required": True,
        "snapshot_bank_root": str(snapshot_bank_root),
        "policy_budget": int(args.budget),
        "post_success_frames": int(args.post),
        "review_mode": "deferred",
        "deferred_review_payload_contract": (
            DEFERRED_REVIEW_PAYLOAD_CONTRACT),
        "hunt_object_id": HUNT_OBJECT_ID,
        "runtime_overlay": str(DEFAULT_RUNTIME_OVERLAY.resolve()),
        "runtime_overlay_options_sha256": sha256_file(
            DEFAULT_RUNTIME_OVERLAY / "options.txt"),
        "same_layout_controls": [
            "world_seed", "arena_commands", "ring_geometry",
            "center_agent_pose", "iron_axe", "mob_health_8",
            "policy_memory_reset", "hunt_obj_id_0",
            "live_ai_release_command", "clouds_disabled",
            "oak_fence_plus_invisible_barrier_cap",
        ],
        "episode_scene_contract": (
            "ID target with train7 roster; OOD target with itself plus train7; "
            "OOD never appears as bystander"),
        "frame0_contract": "episode target exact renderer surface absent",
        "strict_success_contract": (
            "target_class_dead_and_no_bystander_class_dead_no_drift_no_escape_"
            "through_target_death;post_success_tail_non_scoring"),
        "target_success_contract": (
            "target_class_Entity.isAlive_true_to_false; "
            "crosshair_and_kill_counter_diagnostic_only"),
        "layout_manifest": str(layout_path.resolve()),
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        volatile = ("created_unix_s", "machine", "workers")
        stable = {key: value for key, value in manifest.items()
                  if key not in volatile}
        old_stable = {key: value for key, value in old.items()
                      if key not in volatile}
        if stable != old_stable:
            raise ValueError("resume run manifest differs from requested benchmark")
        manifest = old
    else:
        atomic_json(manifest_path, manifest)

    goals = load_goals(
        Path(args.goal_source_root).expanduser().resolve(), selected_targets)

    results, failures = [], []
    tasks = [(episode_layout(
                layout, target,
                scene_seed=int(args.scene_seed_base),
                policy_seed=int(args.policy_seed_base)), target)
             for layout in selected_layouts for target in selected_targets]
    pending = []
    for task_index, (layout, target) in enumerate(tasks):
        task_root = root / "rollouts" / layout["layout_id"] / target
        task_root.mkdir(parents=True, exist_ok=True)
        task_result_path = task_root / "task_result.json"
        if task_result_path.is_file():
            result = json.loads(task_result_path.read_text(encoding="utf-8"))
            if (result.get("contract") != CONTRACT
                    or result.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
                    or result.get("layout_id") != layout["layout_id"]
                    or result.get("target") != target):
                raise ValueError(f"resume task result mismatch: {task_root}")
            results.append(result)
            continue
        pending.append((task_index, layout, target))

    if pending:
        reset_cap = max(
            int(args.workers), int(os.environ.get(
                "MINESTUDIO_MAX_RESETTING_ENV_COUNT", "0") or 0))
        os.environ["MINESTUDIO_MAX_RESETTING_ENV_COUNT"] = str(reset_cap)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(args.workers),
                initializer=_parallel_worker_init,
                initargs=(
                    str(checkpoint), str(manifest["checkpoint_sha256"]),
                    int(args.budget), int(args.post), float(args.cfg_coef),
                    str(args.model_label),
                    str(snapshot_bank_root))) as pool:
            future_to_task = {
                pool.submit(_parallel_run_task, (
                    layout, target, goals[target], str(root),
                    MAX_ATTEMPTS)): (task_index, layout, target)
                for task_index, layout, target in pending}
            for future in concurrent.futures.as_completed(future_to_task):
                task_index, layout, target = future_to_task[future]
                try:
                    outcome = future.result()
                except BaseException as exc:
                    failure = {
                        "layout_id": layout["layout_id"], "target": target,
                        "error": repr(exc), "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                else:
                    if outcome.get("result") is not None:
                        results.append(outcome["result"])
                    else:
                        failures.append(outcome["failure"])
                atomic_json(root / "status.json", {
                    "contract": CONTRACT,
                    "state": "running",
                    "task_index": int(task_index),
                    "tasks_total": len(tasks),
                    "completed": len(results),
                    "failed": len(failures),
                    "current_layout": layout["layout_id"],
                    "current_target": target,
                    "updated_unix_s": time.time(),
                })

    expected = int(manifest["expected_episodes"])
    status = {
        "contract": CONTRACT,
        "state": ("finished" if len(results) == expected and not failures
                  else "incomplete"),
        "expected_episodes": expected,
        "completed_episodes": len(results),
        "failed_tasks_n": len(failures),
        "failures": list(failures),
        "updated_unix_s": time.time(),
    }
    atomic_json(root / "status.json", status)
    return status


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal-source-root", type=Path, required=True)
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--cfg-coef", type=float, default=0.0)
    parser.add_argument(
        "--layouts", type=int, default=DEFAULT_LAYOUTS,
        help="number of deterministic class-neutral terrain layouts")
    parser.add_argument("--targets", nargs="+", choices=TARGETS,
                        default=list(TARGETS))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--post", type=int, default=DEFAULT_POST_SUCCESS_FRAMES)
    parser.add_argument("--benchmark-seed", type=int,
                        default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--world-seed-base", type=int,
                        default=DEFAULT_WORLD_SEED_BASE)
    parser.add_argument(
        "--scene-seed-base", type=int, default=DEFAULT_SCENE_SEED_BASE,
        help=("physical bystander permutation seed; keep fixed across "
              "checkpoints and policy-sampling repeats"))
    parser.add_argument("--policy-seed-base", type=int,
                        default=DEFAULT_POLICY_SEED_BASE)
    parser.add_argument(
        "--snapshot-bank-root", type=Path, required=True,
        help="immutable Hunt snapshot bank; every layout/target bundle is required")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if (int(args.budget) <= 0
            or not 1 <= int(args.workers) <= 8
            or int(args.post) != 20):
        raise ValueError(
            "budget must be positive, workers must be in [1,8], "
            "and the benchmark requires --post 20")
    status = run(args)
    print(json.dumps({
        key: status[key] for key in (
            "state", "expected_episodes", "completed_episodes",
            "failed_tasks_n")
    }, indent=2, sort_keys=True), flush=True)
    if status["state"] == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
