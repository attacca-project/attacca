#!/usr/bin/env python3
"""CCFW long-horizon chain (mine coal ore, hunt a cow, open a furnace and cook, feed a wolf) in one continuous world, with one goal-conditioned policy rollout per stage."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO) + "/src")

from attacca.evaluation import chain_policy as CHAIN
from attacca.evaluation import dpx as DPX
from attacca.evaluation import chain_engine as ENGINE
from attacca.evaluation import place as PLACE
from attacca.evaluation.cooking_adapters import AuditedGuiOpenStageAdapter
from attacca.evaluation.cooking_adapters import CowHuntStageAdapter
from attacca.evaluation.chain_adapters import MineStageAdapter
from attacca.evaluation.wolf import WolfFeedStageAdapter
from attacca.evaluation.wolf import audit_staged_wolf_camera_off
from attacca.evaluation.wolf import load_wolf_goal
from attacca.evaluation.wolf import prepare_cooked_beef_for_wolf
from attacca.evaluation.wolf import runtime_player_uuid_camera_off
from attacca.evaluation.chain_policy import build_runner
from attacca.evaluation.mine_scene import Stager
from attacca.worlds.coal_cow_furnace_chain_scene import COW_TAG
from attacca.worlds.coal_cow_furnace_chain_scene import CoalCowFurnaceScene
from attacca.worlds import coal_cow_furnace_wolf_scene as WOLF_SCENE
from attacca.worlds.human_viewmodel import install_renderer_viewmodel_observation


CONTRACT = "xbench_coal_cow_furnace_wolf_policy_chain/v5"
FRAME_CONTRACT = "xbench_coal_cow_furnace_wolf_policy_frame/v5"
WOLF_POST_SUCCESS_FRAMES = 20
STAGES = (
    {"key": "coal_ore_mine", "label": "Mine COAL ORE", "mode": "mine",
     "target_kind": "coal_ore", "pickup_kind": "coal", "quota": 1,
     "interaction_id": 2},
    {"key": "cow_hunt", "label": "Find / Hunt COW", "mode": "hunt",
     "target_kind": "cow", "quota": 1, "interaction_id": 0},
    {"key": "furnace_find", "label": "Find and open furnace", "mode": "open",
     "target_kind": "furnace", "quota": 1, "interaction_id": 3},
)
WOLF_STAGE = {
    "key": WOLF_SCENE.WOLF_STAGE_KEY, "label": "Feed cooked beef to WOLF",
    "mode": "feed", "target_kind": "wolf", "quota": 1,
    "interaction_id": WOLF_SCENE.WOLF_INTERACTION_ID,
}


def _copy_goal_files(source: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    for name in ("goal.png", "mask.png", "meta.json"):
        shutil.copy2(source / name, out / name)


def load_cow_goal(source: Path, out: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    source = source.expanduser().resolve()
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    if (meta.get("contract") != "xbench_hunt_cross_world_renderer_entity_id_goal/v2"
            or meta.get("target") != "cow"):
        raise ValueError(f"invalid cow goal contract: {source}")
    rgb_path, mask_path = source / "goal.png", source / "mask.png"
    if (DPX._sha256(rgb_path) != meta.get("goal_rgb_sha256")
            or DPX._sha256(mask_path) != meta.get("goal_mask_sha256")):
        raise ValueError(f"cow goal SHA mismatch: {source}")
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or raw_mask is None or raw_mask.shape != bgr.shape[:2]:
        raise ValueError(f"could not decode cow goal: {source}")
    mask = np.ascontiguousarray(raw_mask > 0, dtype=np.uint8)
    if not int(mask.sum()) or set(np.unique(raw_mask).tolist()) - {0, 255}:
        raise ValueError(f"cow goal mask is empty/non-binary: {source}")
    _copy_goal_files(source, out)
    copied = {**meta, "source": str(source),
              "goal": str((out / "goal.png").resolve()),
              "mask": str((out / "mask.png").resolve()),
              "goal_sha256": meta["goal_rgb_sha256"],
              "mask_sha256": meta["goal_mask_sha256"],
              "goal_role": "whole_exemplar_variant"}
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask, copied


def load_block_goal(source: Path, out: Path, source_task: str,
                    *, stage_task: str) -> tuple[np.ndarray, np.ndarray, dict]:
    rgb, mask, meta = DPX.load_goal(source, out, source_task)
    return rgb, mask, {**meta, "source_task_key": source_task,
                       "policy_stage_key": stage_task,
                       "stage_alias_preserves_pixels": True}


def _query(world, cells):
    action = world.sim.noop_action()
    action["voxels"] = ENGINE.query_box(world, cells)
    world.obs, _reward, terminated, truncated, world.info = world.sim.step(action)
    if terminated or truncated:
        raise RuntimeError("sim ended during camera-off voxel verification")
    return ENGINE.current_query(world)


def _face_pose(stance, point):
    px, py, pz = map(float, stance)
    dx, dy, dz = (float(point[0]) - px,
                  float(point[1]) - (py + 1.62),
                  float(point[2]) - pz)
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = -math.degrees(math.atan2(dy, max(1e-9, math.hypot(dx, dz))))
    return px, py, pz, yaw, pitch


def stage_scene(world, stager, scene: CoalCowFurnaceScene,
                out: Path) -> tuple[Any, dict[str, Any]]:
    from attacca.worlds.entity_id_mask import ENTITY_ID_SEMANTIC_CODES
    from attacca.worlds.entity_id_mask import entity_surfaces
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_VALID_FIELD
    from attacca.worlds.human_viewmodel import RENDERER_SCENE_ALIVE_CLASSES_FIELD
    player_uuid = runtime_player_uuid_camera_off(world, stager)
    built = WOLF_SCENE.build_scene_commands(
        scene, owner_uuid=player_uuid)
    pure = WOLF_SCENE.validate_scene_manifest(built, raise_on_error=True)
    for command in built.commands:
        world.cmd(command)
    for _ in range(45):
        world.step_noop()
    world.cmd("/kill @e[type=minecraft:item]")
    for _ in range(6):
        world.step_noop()
    inventory = DPX._inventory_counts(world)
    expected_inventory = {"stone_pickaxe": 1, "iron_axe": 1}
    if inventory != expected_inventory or ENGINE.held_item(world) != "stone_pickaxe":
        raise RuntimeError(f"initial kit mismatch: {inventory}/{ENGINE.held_item(world)}")

    support_rows = []
    support_cells = [tuple(c) for c in built.manifest["placed_object_supports"]]
    support_query = _query(world, support_cells)
    for cell in support_cells:
        actual = DPX._kind(support_query.get(cell, "missing"))
        expected = DPX._kind(built.final_blocks.get(cell, "missing"))
        if actual != expected or actual in {"air", "water", "lava", "missing"}:
            raise RuntimeError(f"structure support mismatch {cell}: {actual}/{expected}")
        support_rows.append({"cell": list(cell), "kind": actual})

    block_rows = []
    for cell, kind in ((scene.coal_ore_cell, "coal_ore"),
                       (scene.furnace_cell, "furnace")):
        support = cell[0], cell[1] - 1, cell[2]
        query = _query(world, (cell, support))
        actual = DPX._kind(query.get(cell, "missing"))
        under = DPX._kind(query.get(support, "missing"))
        if actual != kind or under in {"air", "water", "lava", "missing"}:
            raise RuntimeError(f"target/support mismatch {cell}: {actual}/{under}")
        block_rows.append({"cell": list(cell), "kind": actual,
                           "support_cell": list(support), "support_kind": under})

    boundary = scene.pen_boundary_cells
    pen_query = _query(world, (*boundary, (scene.cow_cell[0], scene.cow_cell[1] - 1,
                                           scene.cow_cell[2])))
    pen_rows = []
    for cell in boundary:
        actual = DPX._kind(pen_query.get(cell, "missing"))
        expected = "oak_fence_gate" if cell == scene.pen_gate_cell else "oak_fence"
        if actual != expected:
            raise RuntimeError(f"pen boundary mismatch {cell}: {actual}/{expected}")
        pen_rows.append({"cell": list(cell), "kind": actual})
    cow_support = (scene.cow_cell[0], scene.cow_cell[1] - 1, scene.cow_cell[2])
    if DPX._kind(pen_query.get(cow_support, "missing")) != "grass_block":
        raise RuntimeError("cow interior support is not grass")

    stand_rows = []
    for coords in (scene.start_navigation_cell,
                   *(cell for stands in built.manifest["stage_stance_cells"].values()
                     for cell in stands)):
        x, y, z = map(int, coords)
        points = ((x, y - 1, z), (x, y, z), (x, y + 1, z))
        query = _query(world, points)
        kinds = [DPX._kind(query.get(point, "air")) for point in points]
        if kinds[0] in {"air", "water", "lava", "missing"} or kinds[1:] != ["air", "air"]:
            raise RuntimeError(f"unsafe stand {coords}: {kinds}")
        stand_rows.append({"feet": list(coords), "kinds": kinds})

    world.cmd("/gamerule sendCommandFeedback true")
    cow_pos = stager.entity_pos(
        "cow", selector=f"type=minecraft:cow,tag={COW_TAG},limit=1")
    unowned = stager.entity_exists(
        f"type=minecraft:cow,tag=!{COW_TAG},limit=1")
    if cow_pos is None or unowned:
        raise RuntimeError(f"unique cow audit failed: pos={cow_pos}, unowned={unowned}")
    if math.dist(tuple(map(float, cow_pos)), scene.cow_position) > .2:
        raise RuntimeError(f"cow escaped one-cell pen during staging: {cow_pos}")
    expected_alive = 1 << (int(ENTITY_ID_SEMANTIC_CODES["cow"]) - 1)
    alive = int(world.info.get(RENDERER_SCENE_ALIVE_CLASSES_FIELD, -1))
    if alive != expected_alive:
        raise RuntimeError(f"renderer cow census mismatch: {alive:#x}/{expected_alive:#x}")
    wolf_manifest = built.manifest["wolf"]
    wolf_live = audit_staged_wolf_camera_off(
        world, stager,
        expected_position=wolf_manifest["spawn_position"],
        expected_owner_uuid=player_uuid,
        expected_invulnerable=bool(wolf_manifest["Invulnerable"]),
        expected_max_health=float(wolf_manifest["max_health"]))
    world.cmd("/gamerule sendCommandFeedback false")

    views = {}
    for task in ("coal_ore_mine", "cow_hunt", "furnace_find"):
        stance = built.manifest["stage_stance_cells"][task][0]
        target = (scene.cow_position[0], scene.cow_position[1] + .95,
                  scene.cow_position[2]) if task == "cow_hunt" else tuple(
                      value + .5 for value in built.manifest["target_cells"][task][0])
        pose = _face_pose((stance[0] + .5, stance[1], stance[2] + .5), target)
        if not stager.go(*pose[:4], pitch=pose[4], n=8):
            raise RuntimeError(f"could not establish stage review pose: {task}")
        for _ in range(3):
            world.step_noop()
        if int(world.info.get(RENDERER_ENTITY_ID_VALID_FIELD, 0)) != 1:
            raise RuntimeError("stage review lacks renderer entity IDs")
        if task == "cow_hunt":
            mask = ENGINE.entity_union(world, ("cow",))
            cow_surfaces = [row for row in entity_surfaces(
                world.info[RENDERER_ENTITY_ID_FIELD]) if row.kind == "cow"]
            if len(cow_surfaces) != 1 or int(mask.sum()) <= 0:
                raise RuntimeError(
                    f"cow certification failed: surfaces={len(cow_surfaces)}, px={int(mask.sum())}")
        else:
            cell = tuple(built.manifest["target_cells"][task][0])
            mask = ENGINE.block_union(
                world, (cell,), render_position=tuple(map(float, world.get_pos())))
            if int(mask.sum()) <= 0:
                raise RuntimeError(f"block certification failed: {task}")
        rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
        views[task] = {"pose": list(map(float, pose)), "mask_pixels": int(mask.sum()),
                       "rgb_mean": float(rgb.mean())}

    task = WOLF_SCENE.WOLF_STAGE_KEY
    stance = built.manifest["stage_stance_cells"][task][0]
    target = (wolf_manifest["spawn_position"][0],
              wolf_manifest["spawn_position"][1] + .7,
              wolf_manifest["spawn_position"][2])
    pose = _face_pose((stance[0] + .5, stance[1], stance[2] + .5), target)
    if not stager.go(*pose[:4], pitch=pose[4], n=8):
        raise RuntimeError("could not establish wolf_feed review pose")
    for _ in range(3):
        world.step_noop()
    mask = ENGINE.entity_union(world, ("wolf",))
    wolf_surfaces = [row for row in entity_surfaces(
        world.info[RENDERER_ENTITY_ID_FIELD]) if row.kind == "wolf"]
    if len(wolf_surfaces) != 1 or int(mask.sum()) <= 0:
        raise RuntimeError(
            f"wolf certification failed: surfaces={len(wolf_surfaces)}, "
            f"px={int(mask.sum())}")
    rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    views[task] = {"pose": list(map(float, pose)),
                   "mask_pixels": int(mask.sum()),
                   "rgb_mean": float(rgb.mean()),
                   "sitting": True}

    if not stager.go(*scene.start_pose, float(built.manifest["start_yaw"]),
                     pitch=float(built.manifest["start_pitch"]), n=8):
        raise RuntimeError("could not restore chain start pose")
    for _ in range(6):
        world.step_noop()
    toast_drain = ENGINE.drain_toasts_camera_off(world)
    position = tuple(float(v) for v in world.get_pos())
    frame0 = {
        "coal_ore_mine": int(ENGINE.block_union(
            world, (scene.coal_ore_cell,), render_position=position).sum()),
        "cow_hunt": int(ENGINE.entity_union(world, ("cow",)).sum()),
        "furnace_find": int(ENGINE.block_union(
            world, (scene.furnace_cell,), render_position=position).sum()),
    }
    frame0[WOLF_SCENE.WOLF_STAGE_KEY] = int(
        ENGINE.entity_union(world, ("wolf",)).sum())
    if any(frame0[key] != 0 for key in (
            "coal_ore_mine", "cow_hunt", "furnace_find",
            WOLF_SCENE.WOLF_STAGE_KEY)):
        raise RuntimeError(f"frame-zero exploration contract failed: {frame0}")
    ENGINE.save_rgb(out / "initial_scene.png",
                   np.ascontiguousarray(world.info["pov"], dtype=np.uint8))
    audit = {
        "pure_scene_audit": pure,
        "initial_inventory": inventory,
        "initial_held_item": ENGINE.held_item(world),
        "placed_structure_support_live_audit": support_rows,
        "block_target_support_live_audit": block_rows,
        "pen_boundary_live_audit": pen_rows,
        "pen_interior_support": {"cell": list(cow_support), "kind": "grass_block"},
        "unique_cow_live_audit": {
            "tag": COW_TAG, "position": list(map(float, cow_pos)),
            "expected_position": list(scene.cow_position),
            "unowned_cow_exists": False,
            "renderer_alive_mask": alive,
        },
        "unique_wolf_live_audit": wolf_live,
        "verified_stands_live": stand_rows,
        "stage_view_certification": views,
        "frame0_exact_target_pixels": frame0,
        "frame0_wolf_visibility_permitted": False,
        "frame0_all_four_targets_hidden": True,
        "start_pose_actual": list(position),
        "toast_drain": toast_drain,
        "camera_off_staging": True,
    }
    ENGINE.atomic_json(out / "scene_manifest.json", {
        **built.manifest, "live_scene_audit": audit})
    return built, audit


def _goals(args, out: Path) -> dict[str, tuple[np.ndarray, np.ndarray, dict]]:
    paths = {
        "coal_ore_mine": (args.coal_goal, "coal_ore_mine"),
        "furnace_find": (args.furnace_goal, "furnace_open"),
    }
    goals = {}
    for stage in STAGES:
        task = stage["key"]
        goal_out = out / task / "goal"
        if task == "cow_hunt":
            goals[task] = load_cow_goal(args.cow_goal, goal_out)
        else:
            source, source_task = paths[task]
            goals[task] = load_block_goal(
                source, goal_out, source_task, stage_task=task)
    goals[WOLF_SCENE.WOLF_STAGE_KEY] = load_wolf_goal(
        args.wolf_goal, out / WOLF_SCENE.WOLF_STAGE_KEY / "goal")
    return goals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--coal-goal", type=Path)
    parser.add_argument("--cow-goal", type=Path)
    parser.add_argument("--furnace-goal", type=Path)
    parser.add_argument("--wolf-goal", type=Path)
    parser.add_argument("--start-yaw-offset", choices=(90,),
                        type=int, default=90)
    parser.add_argument("--stage-budgets", nargs=3, type=int,
                        default=(400, 1200, 1600),
                        metavar=("COAL", "COW", "FURNACE_OPEN"))
    parser.add_argument("--wolf-budget", type=int, default=900,
                        help="wolf_feed step budget")
    parser.add_argument("--completion-mode", choices=("full",),
                        default="full")
    parser.add_argument("--model-label", default="ours")
    parser.add_argument("--cfg-coef", type=float, default=0.)
    CHAIN.add_policy_arguments(parser)
    args = parser.parse_args(argv)
    if min(args.stage_budgets) < 1 or int(args.wolf_budget) < 1:
        raise ValueError("stage budgets must be positive")
    if args.policy_backend not in CHAIN.ROCKET_BACKENDS:
        raise ValueError(f"unsupported chain policy backend: {args.policy_backend}")
    if any(value is None for value in (
            args.checkpoint, args.coal_goal, args.cow_goal, args.furnace_goal,
            args.wolf_goal)):
        parser.error("policy evaluation requires checkpoint and all four goals")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    install_renderer_viewmodel_observation()
    ENGINE.CONTRACT = CONTRACT
    ENGINE.FRAME_CONTRACT = FRAME_CONTRACT
    DPX.FRAME_CONTRACT = FRAME_CONTRACT
    ENGINE.WORLD_SEEDS.update({stage["key"]: 1903000 for stage in (*STAGES, WOLF_STAGE)})
    started = time.monotonic()
    runner = build_runner(args)
    world = PLACE._boot_eval(1903000)
    try:
        stager = Stager(world)
        origin = tuple(math.floor(v) for v in world.get_pos())
        scene = CoalCowFurnaceScene(
            *origin, start_yaw_offset_degrees=int(args.start_yaw_offset))
        built, live = stage_scene(world, stager, scene, out)

        for stage in STAGES:
            (out / stage["key"]).mkdir(parents=True, exist_ok=False)
        (out / WOLF_SCENE.WOLF_STAGE_KEY).mkdir(parents=True, exist_ok=False)
        goals = _goals(args, out)
        results: dict[str, dict[str, Any]] = {}
        stage_adapters: dict[str, Any] = {}
        transitions = []
        raw_segments: list[Path] = []
        system_macros: dict[str, Any] = {}
        inventory_ledger = [{"point": "initial",
                             "counts": DPX._inventory_counts(world)}]
        global_step = 0
        completion_error = None
        for index, stage in enumerate(STAGES):
            task = stage["key"]
            transition = None
            if index:
                prev = STAGES[index - 1]["key"]
                if not results[prev]["success"]:
                    break
                if prev == "coal_ore_mine":
                    coal_count = ENGINE.inventory_count(world, "coal")
                    if coal_count < 1:
                        completion_error = "coal was not collected"
                        system_macros["natural_coal_pickup"] = {
                            "success": False, "error": completion_error}
                        break
                    system_macros["natural_coal_pickup"] = {
                        "success": True,
                        "mechanism": "inventory_observation",
                        "coal_inventory_at_transition": int(coal_count),
                    }
                    inventory_ledger.append({
                        "point": "post_in_stage_coal_credit",
                        "counts": DPX._inventory_counts(world)})
                transition = DPX._transition(
                    prev, task, "scored_stage_complete")
                if prev == "cow_hunt":
                    transition.update({
                        "contract": "cow_death_goal_only_transition/v1",
                        "performed": False,
                        "environment_steps": 0,
                        "system_actions": 0,
                        "camera_actions": 0,
                        "wait_frames": 0,
                    })
                transitions.append(transition)

            cells = built.manifest["target_cells"][task]
            if stage["mode"] == "mine":
                adapter = MineStageAdapter(
                    world, stager, scene, task, cells,
                    target_kind=stage["target_kind"],
                    pickup_kind=stage["pickup_kind"], quota=stage["quota"],
                    interaction_id=stage["interaction_id"])
            elif stage["mode"] == "hunt":
                adapter = CowHuntStageAdapter(
                    world, stager, scene, task,
                    interaction_id=stage["interaction_id"])
            else:
                adapter = AuditedGuiOpenStageAdapter(
                    world, stager, scene, task, cells[0],
                    target_kind=stage["target_kind"],
                    interaction_id=stage["interaction_id"])
            stage_adapters[task] = adapter
            ENGINE.initial_artifacts(world, out / task)
            results[task] = ENGINE.rollout(
                world=world, runner=runner, adapter=adapter, out=out / task,
                goal_rgb=goals[task][0], goal_mask=goals[task][1],
                goal_meta=goals[task][2], goal_label=stage["label"],
                budget=int(args.stage_budgets[index]), post=0,
                post_action_mode="observe_noop",
                observation_wait_override=0, guard_frames=True)
            global_step, _ = DPX._enrich_policy_rows(
                out / task, adapter, goals[task][2],
                global_start=global_step)
            if results[task].get("raw_video"):
                raw_segments.append(Path(results[task]["raw_video"]))
            inventory_ledger.append({
                "point": f"post_{task}", "counts": DPX._inventory_counts(world)})

        pre_wolf_policy_success = all(
            results.get(stage["key"], {}).get("success", False) for stage in STAGES)
        if pre_wolf_policy_success:
            try:
                cooking, equip = prepare_cooked_beef_for_wolf(
                    world,
                    cook_out=out / "system_furnace_cooking",
                    equip_out=out / "system_cooked_beef_hotbar_inventory")
                system_macros["furnace_cooking"] = cooking
                raw_segments.append(Path(cooking["raw_video"]))
                inventory_ledger.append({
                    "point": "post_system_furnace_cooking",
                    "counts": DPX._inventory_counts(world)})
                if ENGINE.inventory_count(world, "cooked_beef") != 1:
                    raise RuntimeError(
                        "furnace handoff requires exactly one cooked_beef")
                system_macros["cooked_beef_hotbar_inventory"] = equip
                if equip.get("raw_video") and equip["raw_video"] != cooking["raw_video"]:
                    raw_segments.append(Path(equip["raw_video"]))
                if (ENGINE.inventory_count(world, "cooked_beef") != 1
                        or ENGINE.held_item(world) != "cooked_beef"):
                    raise RuntimeError(
                        "wolf stage must start with cooked_beef equipped")
                inventory_ledger.append({
                    "point": "post_cooked_beef_hotbar_inventory",
                    "counts": DPX._inventory_counts(world),
                    "held_item": ENGINE.held_item(world),
                    "hotbar_key": equip["hotbar_key"],
                })

                wolf_task = WOLF_SCENE.WOLF_STAGE_KEY
                transition = DPX._transition(
                    "furnace_find", wolf_task,
                    "real_gui_cooking_complete_goal_changed_to_wolf")
                transition.update({
                        "contract": "xbench_ccfw_cooking_goal_transition/v1",
                        "performed": True,
                        "environment_steps": int(
                            cooking["raw_video_receipt"]["decoded_frames"]
                            + (equip.get("raw_video_receipt") or {}).get(
                                "decoded_frames", 0)),
                        "world_camera_actions": 0,
                        "wait_frames_after_goal_change": 0,
                        "teleports": 0,
                        "cooked_beef_count": 1,
                        "held_item_after": ENGINE.held_item(world),
                        "cooked_beef_hotbar_key": equip["hotbar_key"],
                    })
                transitions.append(transition)
                wolf_manifest = built.manifest["wolf"]
                wolf_live = live["unique_wolf_live_audit"]
                wolf_adapter = WolfFeedStageAdapter(
                    world, stager, scene, wolf_task,
                    wolf_position=wolf_manifest["spawn_position"],
                    wolf_uuid=wolf_live["mobs_id"],
                    initial_health=float(wolf_live["Health"]),
                    interaction_id=WOLF_SCENE.WOLF_INTERACTION_ID)
                stage_adapters[wolf_task] = wolf_adapter
                ENGINE.initial_artifacts(world, out / wolf_task)
                results[wolf_task] = ENGINE.rollout(
                    world=world, runner=runner, adapter=wolf_adapter,
                    out=out / wolf_task,
                    goal_rgb=goals[wolf_task][0],
                    goal_mask=goals[wolf_task][1],
                    goal_meta=goals[wolf_task][2],
                    goal_label=WOLF_STAGE["label"],
                    budget=int(args.wolf_budget),
                    post=WOLF_POST_SUCCESS_FRAMES,
                    post_action_mode="observe_noop",
                    observation_wait_override=0, guard_frames=True)
                global_step, _ = DPX._enrich_policy_rows(
                    out / wolf_task, wolf_adapter, goals[wolf_task][2],
                    global_start=global_step)
                if results[wolf_task].get("raw_video"):
                    raw_segments.append(Path(results[wolf_task]["raw_video"]))
                inventory_ledger.append({
                    "point": "post_wolf_feed",
                    "counts": DPX._inventory_counts(world),
                    "held_item": ENGINE.held_item(world),
                })
            except Exception as exc:
                completion_error = f"{type(exc).__name__}: {exc}"
                if "furnace_cooking" not in system_macros:
                    system_macros["furnace_cooking"] = {
                        "success": False, "error": completion_error}
                results.setdefault(WOLF_SCENE.WOLF_STAGE_KEY, {
                    "state": "not_run", "success": 0,
                    "failure_mode": "cooked_beef_unavailable",
                    "error": completion_error,
                })
        else:
            results[WOLF_SCENE.WOLF_STAGE_KEY] = {
                "state": "not_run", "success": 0,
                "failure_mode": "predecessor_stage_failed",
            }
        wolf_success = bool(
            results.get(WOLF_SCENE.WOLF_STAGE_KEY, {}).get("success", False))
        policy_success = bool(pre_wolf_policy_success and wolf_success)
        raw = DPX._concat_videos(
            raw_segments, out / "chain_raw_combined.mp4", (640, 360))
        demonstration_complete = bool(
            policy_success
            and system_macros.get("natural_coal_pickup", {}).get("success")
            and system_macros.get("furnace_cooking", {}).get("success")
            and system_macros.get(
                "cooked_beef_hotbar_inventory", {}).get("success"))
        result = {
            "contract": CONTRACT, "state": "complete",
            "world_seed": 1903000, "distance_band": scene.distance_band,
            "start_yaw_offset_degrees": int(args.start_yaw_offset),
            "scene_version": built.manifest["contract"],
            "artifact_mode": "deferred",
            "completion_mode": args.completion_mode,
            "model_label": args.model_label,
            "policy_backend": args.policy_backend,
            "policy_rollout_seed": args.policy_rollout_seed,
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": runner.model_contract["checkpoint_sha256"],
            "model_contract": runner.model_contract,
            "stage_budgets": {
                **dict(zip((stage["key"] for stage in STAGES),
                           args.stage_budgets)),
                WOLF_SCENE.WOLF_STAGE_KEY: int(args.wolf_budget),
            },
            "goals": {key: value[2] for key, value in goals.items()},
            "system_fsm": [stage["key"] for stage in STAGES] + [
                WOLF_SCENE.WOLF_STAGE_KEY],
            "interaction_id_sequence": [stage["interaction_id"] for stage in STAGES] + [
                WOLF_SCENE.WOLF_INTERACTION_ID],
            "results": results,
            "transitions": transitions,
            "pre_wolf_policy_success": pre_wolf_policy_success,
            "wolf_feed_success": wolf_success,
            "policy_chain_success": policy_success,
            "mission_terminal_stage": WOLF_SCENE.WOLF_STAGE_KEY,
            "system_macros": system_macros,
            "demonstration_complete": demonstration_complete,
            "demonstration_completion_reason": (
                "policy fed the sitting tamed wolf after recorded real-GUI cooking"
                if demonstration_complete else completion_error or
                "the policy stages did not complete"),
            "commands_during_scored_policy_or_completion": sum(
                int(row.get("commands_issued", 0))
                for row in system_macros.values()),
            "inventory_ledger": inventory_ledger,
            "final_inventory": DPX._inventory_counts(world),
            "world_continuous": True,
            "world_reset_between_stages": False,
            "teleports_during_rollout_or_completion": 0,
            "combined_raw_video": str(out / "chain_raw_combined.mp4") if raw else None,
            "combined_review_overlay": None,
            "wall_seconds": time.monotonic() - started,
        }
        ENGINE.atomic_json(out / "result.json", result)
        print(json.dumps({
            "state": "complete", "policy_success": policy_success,
            "demonstration_complete": demonstration_complete,
            "stages": {key: value["success"] for key, value in results.items()},
            "completion_error": completion_error, "out": str(out)},
            sort_keys=True), flush=True)
        return 0
    finally:
        world.close()


if __name__ == "__main__":
    raise SystemExit(main())
