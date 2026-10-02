#!/usr/bin/env python3
"""WLO long-horizon chain (scoop water, pour it onto lava, mine the obsidian, build and ignite a portal frame) in one continuous world, with one goal-conditioned policy rollout per stage."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
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

from attacca.evaluation import chain_policy as CHAIN
from attacca.evaluation import chain_engine as ENGINE
from attacca.evaluation import place as PLACE
from attacca.evaluation.chain_policy import build_runner
from attacca.evaluation.mine_scene import Stager
from attacca.evaluation.water_lava_obsidian import LAVA_GOAL_SOURCE
from attacca.evaluation.water_lava_obsidian import OBSIDIAN_GOAL_SOURCE
from attacca.evaluation.water_lava_obsidian import WATER_GOAL_SOURCE
from attacca.evaluation.water_lava_obsidian import capture_goals
from attacca.worlds.human_viewmodel import install_renderer_viewmodel_observation
from attacca.worlds.water_lava_obsidian_portal_scene import POLICY_STAGE_KEYS
from attacca.worlds.water_lava_obsidian_portal_scene import PORTAL_PLACE_MARKER_KIND
from attacca.worlds.water_lava_obsidian_portal_scene import PORTAL_PLACE_MARKER_KINDS
from attacca.worlds.water_lava_obsidian_portal_scene import WORLD_SEED
from attacca.worlds.water_lava_obsidian_portal_scene import WaterLavaObsidianPortalScene
from attacca.worlds.water_lava_obsidian_portal_scene import build_scene_commands
from attacca.worlds.water_lava_obsidian_portal_scene import validate_scene_manifest


CONTRACT = "xbench_water_lava_obsidian_portal_policy_chain/v9"
FRAME_CONTRACT = "xbench_water_lava_obsidian_portal_policy_frame/v5"
PORTAL_GOAL_ROOT = REPO / "assets" / "goals" / "wlo" / "portal_markers"
PORTAL_GOAL_SWITCH_DISTANCE = 7.0


def load_portal_goal(root: Path, name: str, out: Path, semantic: str):
    source = root / name
    meta_path = source / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not bool(source_meta.get("world_disjoint_from_evaluation")):
        raise RuntimeError(f"{name} goal is not world-disjoint")
    rgb, mask, meta = ENGINE.load_copied_goal(
        source, out, semantic=semantic)
    meta.update({
        "source_contract": source_meta.get("contract"),
        "donor_world_seed": source_meta.get("donor_world_seed"),
        "evaluation_world_seed": source_meta.get("evaluation_world_seed"),
        "world_disjoint_from_evaluation": True,
        "target": source_meta.get("target"),
    })
    return rgb, mask, meta


def _kind(value: object) -> str:
    return str(value).removeprefix("minecraft:").split("[", 1)[0]


class PortalBuildGoalProvider:
    goal_conditioning = "portal_frame_then_placement_marker"

    def __init__(self, world, frame_cells, frame_goal, marker_goal):
        self.world = world
        self.frame_cells = tuple(tuple(cell) for cell in frame_cells)
        if not self.frame_cells:
            raise ValueError("portal goal switching requires frame cells")
        self.frame_goal = frame_goal
        self.marker_goal = marker_goal
        self.switch_step: int | None = None
        self.switch_distance: float | None = None

    def resolve(self, step, pre_rgb, observation_rgb):
        del pre_rgb, observation_rgb
        x, y, z = (float(value) for value in self.world.get_pos())
        eye = (x, y + 1.62, z)
        if not all(math.isfinite(value) for value in eye):
            raise ValueError("portal goal switching requires a finite position")
        distance = min(PLACE.aabb_distance(eye, cell)
                       for cell in self.frame_cells)
        if self.switch_step is None and distance <= PORTAL_GOAL_SWITCH_DISTANCE:
            self.switch_step = int(step)
            self.switch_distance = float(distance)
        marker_selected = self.switch_step is not None
        goal = self.marker_goal if marker_selected else self.frame_goal
        return goal[0], goal[1], {
            "goal_kind": "sea_lantern" if marker_selected else "portal_frame",
            "distance_to_frame": float(distance),
            "switch_step": self.switch_step,
            "goal": dict(goal[2]),
        }

    def summary(self):
        return {
            "switch_distance_blocks": PORTAL_GOAL_SWITCH_DISTANCE,
            "switch_step": self.switch_step,
            "distance_at_switch": self.switch_distance,
            "frame_goal": dict(self.frame_goal[2]),
            "marker_goal": dict(self.marker_goal[2]),
        }


class LavaPourAdapter(ENGINE.Adapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task != "lava_pour" or len(self.scene.lava_cells) != 2:
            raise ValueError("LavaPourAdapter requires two lava cells")
        self.pour_step: int | None = None
        self.conversion_step: int | None = None
        self.converted_cells: list[tuple[int, int, int]] = []

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        if (self.pour_step is None
                and use
                and _kind(context.get("held_before", "")) == "water_bucket"
                and int(context.get("water_bucket_before", 0)) > 0
                and ENGINE.inventory_count(self.world, "water_bucket") == 0
                and ENGINE.inventory_count(self.world, "bucket")
                > int(context.get("bucket_before", 0))):
            self.pour_step = int(step)
        if self.pour_step is not None and self.conversion_step is None:
            queried = ENGINE.current_query(self.world)
            cells = self.scene.lava_cells
            if (all(cell in queried for cell in cells)
                    and any(_kind(queried[cell]) in {"water", "obsidian"}
                            for cell in cells)):
                self.converted_cells = [
                    cell for cell in cells
                    if _kind(queried[cell]) != "obsidian"]
                if self.converted_cells:
                    previous_action = copy.deepcopy(
                        self.world.obs["env_prev_action"])
                    for x, y, z in self.converted_cells:
                        self.world.cmd(
                            f"/setblock {x} {y} {z} minecraft:obsidian")
                    for _ in range(8):
                        action = self.world.sim.noop_action()
                        self.prepare_action(action)
                        self.world.obs, _, terminated, truncated, self.world.info = (
                            self.world.sim.step(action))
                        if terminated or truncated:
                            raise RuntimeError("sim ended during lava conversion")
                        queried = ENGINE.current_query(self.world)
                        if all(_kind(queried.get(cell, "air")) == "obsidian"
                               for cell in cells):
                            break
                    self.world.obs["env_prev_action"] = previous_action
                if not all(_kind(queried.get(cell, "air")) == "obsidian"
                           for cell in cells):
                    raise RuntimeError(
                        "lava conversion did not produce two obsidian blocks: "
                        f"{[(cell, queried.get(cell)) for cell in cells]}")
                self.conversion_step = int(step)
        event = super().after_step(
            step=step, use=use, attack=attack, context=context)
        if self.success_evidence is not None:
            self.success_evidence["resource_conversion"] = {
                "pour_step": self.pour_step,
                "conversion_step": self.conversion_step,
                "converted_cells": [list(cell) for cell in self.converted_cells],
            }
        return event


class ExactMineAdapter(ENGINE.Adapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task != "obsidian_mine" or not self.scene.lava_cells:
            raise ValueError("ExactMineAdapter requires obsidian cells")
        self.target_cells = tuple(
            tuple(cell) for cell in self.scene.lava_cells)
        self.latest_query = {cell: "obsidian" for cell in self.target_cells}
        self.last_attack_step: int | None = None
        self.target_removed_step: int | None = None
        self.target_removal_attack_step: int | None = None
        self.target_removed_cells: tuple[tuple[int, int, int], ...] = ()
        self.target_removal_causal = False
        self.target_removal_events: list[dict[str, Any]] = []
        self.causal_removed_cells: set[tuple[int, int, int]] = set()
        self.initial_obsidian_inventory = ENGINE.inventory_count(
            self.world, "obsidian")
        self.last_obsidian_inventory = self.initial_obsidian_inventory
        self.pickup_events: list[dict[str, Any]] = []

    @property
    def quota(self) -> int:
        return int(self.scene.resource_obsidian_quota)

    @property
    def progress(self) -> int:
        inventory = ENGINE.inventory_count(self.world, "obsidian")
        collected = max(0, inventory - self.initial_obsidian_inventory)
        return min(len(self.causal_removed_cells), collected, self.quota)

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        del use, context
        if attack:
            self.last_attack_step = int(step)
        previous_remaining = {
            cell for cell in self.target_cells
            if _kind(self.latest_query.get(cell, "air")) == "obsidian"}
        self.latest_query = ENGINE.current_query(self.world)
        current_remaining = {
            cell for cell in self.target_cells
            if _kind(self.latest_query.get(cell, "air")) == "obsidian"}
        newly_removed = tuple(sorted(
            previous_remaining - current_remaining
            - set(self.target_removed_cells)))
        removed = tuple(sorted(
            cell for cell in self.target_cells
            if cell not in current_remaining))
        self.target_removed_cells = removed
        for cell in newly_removed:
            causal = bool(
                self.last_attack_step is not None
                and int(step) - self.last_attack_step <= 6)
            removal_event = {
                "cell": list(cell),
                "removal_step": int(step),
                "causal_attack_step": self.last_attack_step,
                "causal_attack_max_lag": 6,
                "causal": causal,
            }
            self.target_removal_events.append(removal_event)
            if self.target_removed_step is None:
                self.target_removed_step = int(step)
                self.target_removal_attack_step = self.last_attack_step
                self.target_removal_causal = causal
            if causal:
                self.causal_removed_cells.add(cell)
        inventory = ENGINE.inventory_count(self.world, "obsidian")
        if inventory > self.last_obsidian_inventory:
            self.pickup_events.append({
                "step": int(step),
                "item": "obsidian",
                "before": int(self.last_obsidian_inventory),
                "after": int(inventory),
                "delta": int(inventory - self.last_obsidian_inventory),
            })
        self.last_obsidian_inventory = int(inventory)
        success_now = bool(
            self.progress >= self.quota and self.success_step is None)
        note = ""
        if attack or self.target_removed_step is not None:
            note = (
                f"MINE exact_removed={len(self.target_removed_cells)} "
                f"causal_removed={len(self.causal_removed_cells)} "
                f"inventory={inventory}/{self.quota}")
        if success_now:
            self.success_step = int(step)
            self.success_evidence = {
                "step": int(step),
                "task": self.task,
                "target_cells": [list(cell) for cell in self.target_cells],
                "target_removed_cells": [
                    list(cell) for cell in self.target_removed_cells],
                "target_removed_step": self.target_removed_step,
                "causal_attack_step": self.target_removal_attack_step,
                "causal_attack_max_lag": 6,
                "target_removal_events": copy.deepcopy(
                    self.target_removal_events),
                "causal_removed_cells": [
                    list(cell) for cell in sorted(self.causal_removed_cells)],
                "obsidian_inventory_at_success": int(inventory),
            }
        return {
            "success_now": success_now,
            "control_complete_now": False,
            "note": note,
        }

    def finish_audit(self) -> bool:
        inventory = ENGINE.inventory_count(self.world, "obsidian")
        passed = bool(
            self.success_step is not None
            and self.target_removed_step is not None
            and len(self.causal_removed_cells) >= self.quota
            and inventory == self.initial_obsidian_inventory + self.quota)
        self.terminal_audit = {
            "performed": self.success_step is not None,
            "passed": passed,
            "success_requires_exact_authored_cell_removal": True,
            "success_requires_policy_attack_causal_fifo": True,
            "causal_attack_max_lag": 6,
            "target_removed_cells": [
                list(cell) for cell in self.target_removed_cells],
            "target_removed_step": self.target_removed_step,
            "causal_attack_step": self.target_removal_attack_step,
            "target_removal_events": copy.deepcopy(
                self.target_removal_events),
            "causal_removed_cells": [
                list(cell) for cell in sorted(self.causal_removed_cells)],
            "pickup_item": "obsidian",
            "initial_pickup_inventory": int(self.initial_obsidian_inventory),
            "pickup_inventory_count": int(inventory),
            "pickup_events": copy.deepcopy(self.pickup_events),
            "success_evidence": copy.deepcopy(self.success_evidence),
        }
        return passed


class MultiBlockPortalAdapter(ENGINE.Adapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_use_step: int | None = None
        self.placement_events: list[dict[str, Any]] = []

    @property
    def quota(self) -> int:
        return (len(self.scene.build_frame_cells)
                if self.task == "portal_build" else 1)

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        if use:
            self.last_use_step = int(step)
        completed_before = set(self.completed)
        event = super().after_step(
            step=step, use=use, attack=attack, context=context)
        newly_completed = (
            sorted(self.completed - completed_before)
            if self.task == "portal_build" else [])
        causal = bool(
            self.last_use_step is not None
            and int(step) - self.last_use_step <= 6)
        if newly_completed and not causal:
            raise RuntimeError(
                "portal cell completed without a causal policy USE")
        for cell in newly_completed:
            self.placement_events.append({
                "cell": list(cell),
                "observed_step": int(step),
                "causal_use_step": self.last_use_step,
                "causal_use_max_lag": 6,
            })
        if (self.task == "portal_ignite" and event["success_now"]
                and not causal):
            raise RuntimeError(
                "portal activation completed without a causal policy USE")
        if (self.task == "portal_build" and event["success_now"]
                and len(self.completed) != self.quota):
            raise RuntimeError("portal transition completed at wrong quota")
        return event

    def finish_audit(self) -> bool:
        passed = super().finish_audit()
        if self.task != "portal_build":
            return passed
        event_cells = {
            tuple(row["cell"]) for row in self.placement_events}
        expected = set(self.scene.build_frame_cells)
        causal_passed = bool(passed and event_cells == expected)
        self.terminal_audit.update({
            "passed": causal_passed,
            "success_requires_all_policy_causal_placements": True,
            "required_policy_causal_placement_count": int(self.quota),
            "placement_events": copy.deepcopy(self.placement_events),
            "expected_build_frame_cells": [
                list(cell) for cell in self.scene.build_frame_cells],
        })
        return causal_passed


def _query_authored_volume(world, scene) -> dict[tuple[int, int, int], str]:
    bounds = scene.reset_bounds
    first = (bounds["x_min"], scene.feet_y - 1, bounds["z_min"])
    last = (bounds["x_max"], scene.feet_y + 5, bounds["z_max"])
    action = world.sim.noop_action()
    action["voxels"] = ENGINE.query_box(world, (first, last))
    world.obs, _reward, terminated, truncated, world.info = world.sim.step(action)
    if terminated or truncated:
        raise RuntimeError("sim ended during authored-volume audit")
    return {
        cell: _kind(value) for cell, value in ENGINE.current_query(world).items()
        if all(first[index] <= cell[index] <= last[index]
               for index in range(3))
    }


def _live_scene_audit(world, scene, manifest: dict) -> dict[str, Any]:
    queried = _query_authored_volume(world, scene)
    water = sorted(cell for cell, kind in queried.items() if kind == "water")
    lava = sorted(cell for cell, kind in queried.items() if kind == "lava")
    obsidian = sorted(
        cell for cell, kind in queried.items() if kind == "obsidian")
    backing_stone_cells = (
        set(scene.portal_backing_wall_cells)
        - set(scene.portal_place_marker_cells)
        - {scene.portal_ignite_netherrack_cell})
    marker_faces_occluded = all(
        queried.get(cell, "air") not in {"air", "water", "lava"}
        for cell in scene.portal_place_marker_occluders)
    ignite_faces_occluded = all(
        queried.get(cell, "air") not in {"air", "water", "lava"}
        for cell in scene.portal_ignite_target_occluders)
    mossy_floor_distractors = sum(
        kind == "mossy_stone_bricks" for kind in queried.values())
    stands = []
    for raw in manifest["verified_stand_cells"]:
        x, y, z = (int(value) for value in raw)
        row = {
            "cell": [x, y, z],
            "support": queried.get((x, y - 1, z), "air"),
            "feet": queried.get((x, y, z), "air"),
            "head": queried.get((x, y + 1, z), "air"),
        }
        row["passed"] = bool(
            row["support"] not in {"air", "water", "lava"}
            and row["feet"] == "air" and row["head"] == "air")
        stands.append(row)
    passed = bool(
        set(water) == set(scene.water_cells)
        and set(lava) == set(scene.lava_cells)
        and set(obsidian) == set(scene.portal_prebuilt_obsidian_cells)
        and all(queried.get(cell) == scene.portal_place_marker_kind
                for cell in scene.portal_place_marker_cells)
        and queried.get(
            scene.portal_ignite_netherrack_cell) == "netherrack"
        and all(queried.get(cell) == "stone" for cell in backing_stone_cells)
        and marker_faces_occluded
        and ignite_faces_occluded
        and mossy_floor_distractors == 0
        and all(queried.get(cell, "air") == "air"
                for cell in scene.portal_missing_cells)
        and all(queried.get(cell, "air") == "air"
                for cell in scene.portal_interior_cells)
        and all(row["passed"] for row in stands))
    return {
        "contract": "xbench_wlo_portal_policy_scene_live_audit/v1",
        "passed": passed,
        "water_cells": [list(cell) for cell in water],
        "lava_cells": [list(cell) for cell in lava],
        "prebuilt_obsidian_count": len(obsidian),
        "portal_missing_kinds": [
            queried.get(cell, "air") for cell in scene.portal_missing_cells],
        "portal_marker_kinds": [
            queried.get(cell, "air")
            for cell in scene.portal_place_marker_cells],
        "portal_ignite_target_kind": queried.get(
            scene.portal_ignite_netherrack_cell, "air"),
        "portal_backing_stone_count": sum(
            queried.get(cell) == "stone" for cell in backing_stone_cells),
        "portal_backing_expected_stone_count": len(backing_stone_cells),
        "portal_marker_non_answer_faces_occluded": marker_faces_occluded,
        "portal_ignite_non_answer_faces_occluded": ignite_faces_occluded,
        "mossy_floor_distractors": mossy_floor_distractors,
        "portal_interior_air_count": sum(
            queried.get(cell, "air") == "air"
            for cell in scene.portal_interior_cells),
        "verified_stands": stands,
    }


def _stage_scene(world, stager: Stager,
                 scene: WaterLavaObsidianPortalScene) -> tuple[dict, dict]:
    built = build_scene_commands(scene)
    pure = validate_scene_manifest(built)
    if not pure["passed"]:
        raise RuntimeError(f"pure scene audit failed: {pure}")
    for command in built.commands:
        world.cmd(command)
    for _ in range(12):
        world.step_noop()
    stager.kit("minecraft:bucket 1", fire_res=True)
    world.cmd(
        f"/replaceitem entity @a hotbar.1 minecraft:{scene.mine_tool} 1")
    world.cmd(
        "/replaceitem entity @a hotbar.2 minecraft:flint_and_steel 1")
    world.cmd("/recipe give @a *")
    for _ in range(6):
        world.step_noop()
    live = _live_scene_audit(world, scene, built.manifest)
    if not live["passed"]:
        raise RuntimeError(f"live scene audit failed: {live}")
    return {**built.manifest, "live_audit": live}, pure


def _run_stage(*, world, runner, adapter, out: Path, goal,
               goal_label: str, budget: int, post: int = 0,
               dynamic_goal_provider=None) -> dict:
    ENGINE.initial_artifacts(world, out)
    result = ENGINE.rollout(
        world=world, runner=runner, adapter=adapter, out=out,
        goal_rgb=goal[0], goal_mask=goal[1], goal_meta=goal[2],
        goal_label=goal_label, budget=int(budget), post=int(post),
        post_action_mode="policy",
        observation_wait_override=getattr(
            adapter, "observation_wait_steps", None),
        dynamic_goal_provider=dynamic_goal_provider)
    ENGINE.atomic_json(out / "result.json", result)
    return result


def _equip_hotbar_item(world, kind: str) -> None:
    control = f"hotbar.{ENGINE.hotbar_slot_for_item(world, kind)}"
    action = world.sim.noop_action()
    if control not in action:
        raise RuntimeError(f"sim action lacks {control}")
    action[control] = 1
    world.obs, _, terminated, truncated, world.info = world.sim.step(action)
    if terminated or truncated:
        raise RuntimeError(f"sim ended while equipping {kind}")
    if ENGINE.held_item(world) != kind:
        raise RuntimeError(f"failed to equip {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--cfg-coef", type=float, default=0.0)
    parser.add_argument("--stage-budget", type=int, default=1000)
    parser.add_argument("--scoop-budget", type=int, required=True)
    parser.add_argument("--post", type=int, default=20)
    parser.add_argument("--water-goal-source", type=Path, default=WATER_GOAL_SOURCE)
    parser.add_argument("--lava-goal-source", type=Path, default=LAVA_GOAL_SOURCE)
    parser.add_argument(
        "--obsidian-goal-source", type=Path, default=OBSIDIAN_GOAL_SOURCE)
    parser.add_argument(
        "--portal-obsidian-goal-source", type=Path, required=True,
        help="goal bundle for the portal stage's navigation phase")
    parser.add_argument("--portal-goal-root", type=Path, default=PORTAL_GOAL_ROOT)
    parser.add_argument(
        "--portal-marker-kind", choices=PORTAL_PLACE_MARKER_KINDS,
        default=PORTAL_PLACE_MARKER_KIND)
    CHAIN.add_policy_arguments(parser)
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if out.exists():
        raise FileExistsError(out)
    if not checkpoint.is_file() and not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    scoop_budget = int(args.scoop_budget)
    if args.stage_budget <= 0 or scoop_budget <= 0 or args.post < 0:
        raise ValueError(
            "stage-budget and scoop-budget must be positive and post non-negative")
    out.mkdir(parents=True)
    (out / "portal").mkdir()
    (out / "portal_marker").mkdir()
    (out / "ignite").mkdir()

    ENGINE.CONTRACT = CONTRACT
    ENGINE.FRAME_CONTRACT = FRAME_CONTRACT
    ENGINE.WORLD_SEEDS.update({key: WORLD_SEED for key in POLICY_STAGE_KEYS})
    ENGINE.WORLD_SEEDS["portal"] = WORLD_SEED
    install_renderer_viewmodel_observation()
    runner = build_runner(args)
    world = PLACE._boot_eval(WORLD_SEED)
    try:
        stager = Stager(world)
        px, py, pz = world.get_pos()
        scene = WaterLavaObsidianPortalScene(
            math.floor(px), math.floor(py), math.floor(pz),
            portal_place_marker_kind=args.portal_marker_kind)
        scene_manifest, pure_audit = _stage_scene(world, stager, scene)
        toast_drain = ENGINE.drain_toasts_camera_off(world)
        goals = capture_goals(
            out, args.water_goal_source.expanduser().resolve(),
            args.lava_goal_source.expanduser().resolve(),
            args.obsidian_goal_source.expanduser().resolve())
        goals["ignite"] = load_portal_goal(
            args.portal_goal_root.expanduser().resolve(),
            "netherrack", out / "ignite",
            "netherrack_negative_z_front_face_only")
        goals["portal_marker"] = load_portal_goal(
            args.portal_goal_root.expanduser().resolve(),
            "sea_lantern", out / "portal_marker",
            "sea_lantern_front_face_placement_target")
        (out / "portal_obsidian").mkdir()
        goals["portal_obsidian"] = ENGINE.load_copied_goal(
            args.portal_obsidian_goal_source.expanduser().resolve(),
            out / "portal_obsidian", semantic="portal_complete_frame_obsidian_goal")

        if not stager.go(
                *scene.start_pose, scene.start_yaw,
                pitch=scene.start_pitch, n=8):
            raise RuntimeError("could not establish exploration start")
        for _ in range(6):
            world.step_noop()
        render_position = tuple(float(value) for value in world.get_pos())
        frame0 = {
            "water_scoop": int(ENGINE.block_union(
                world, scene.water_cells,
                render_position=render_position).sum()),
            "lava_pour": int(ENGINE.block_union(
                world, scene.lava_cells,
                render_position=render_position).sum()),
            "obsidian_mine": int(ENGINE.block_union(
                world, scene.lava_cells,
                render_position=render_position).sum()),
            "portal_place": int(ENGINE.block_union(
                world, scene.portal_place_marker_cells,
                render_position=render_position).sum()),
            "portal_ignite": int(ENGINE.block_union(
                world, (scene.portal_ignite_netherrack_cell,),
                render_position=render_position).sum()),
        }
        if any(frame0.values()):
            raise RuntimeError(f"frame-zero target leak: {frame0}")
        if ENGINE.held_item(world) != "bucket":
            raise RuntimeError(f"initial held item is {ENGINE.held_item(world)!r}")
        ENGINE.save_rgb(
            out / "frame0_rgb.png",
            np.ascontiguousarray(world.info["pov"], dtype=np.uint8))
        ENGINE.atomic_json(out / "frame0.json", {
            "exact_target_pixels": frame0,
            "all_targets_hidden": True,
            "held_item": ENGINE.held_item(world),
        })

        results: dict[str, dict] = {}
        scoop_adapter = ENGINE.Adapter(
            world, stager, scene, "water_scoop")
        results["scoop"] = _run_stage(
            world=world, runner=runner, adapter=scoop_adapter,
            out=out / "scoop", goal=goals["scoop"],
            goal_label="water pool", budget=scoop_budget)

        if results["scoop"]["success"]:
            if ENGINE.inventory_count(world, "water_bucket") != 1:
                raise RuntimeError("scoop succeeded without one water bucket")
            _equip_hotbar_item(world, "water_bucket")
            pour_adapter = LavaPourAdapter(
                world, stager, scene, "lava_pour")
            results["pour"] = _run_stage(
                world=world, runner=runner, adapter=pour_adapter,
                out=out / "pour", goal=goals["pour"],
                goal_label="2x1 lava", budget=args.stage_budget)

        if results.get("pour", {}).get("success"):
            _equip_hotbar_item(world, scene.mine_tool)
            mine_adapter = ExactMineAdapter(
                world, stager, scene, "obsidian_mine")
            results["mine"] = _run_stage(
                world=world, runner=runner, adapter=mine_adapter,
                out=out / "mine", goal=goals["mine"],
                goal_label="obsidian block", budget=args.stage_budget)

        if results.get("mine", {}).get("success"):
            if (ENGINE.inventory_count(world, "obsidian")
                    < scene.resource_obsidian_quota):
                raise RuntimeError(
                    "mine succeeded without the required obsidian inventory")
            _equip_hotbar_item(world, "obsidian")
            portal_scene = SimpleNamespace(
                build_frame_cells=scene.portal_missing_cells,
                build_marker_cells=scene.portal_place_marker_cells,
                complete_frame_cells=scene.portal_outer_frame_cells,
                portal_interior_cells=scene.portal_interior_cells,
                ignite_marker_cell=scene.portal_ignite_netherrack_cell,
            )
            portal_adapter = MultiBlockPortalAdapter(
                world, stager, portal_scene, "portal_build")
            portal_goal_provider = PortalBuildGoalProvider(
                world, scene.portal_outer_frame_cells,
                goals["portal_obsidian"], goals["portal_marker"])
            results["portal"] = _run_stage(
                world=world, runner=runner, adapter=portal_adapter,
                out=out / "portal", goal=goals["portal_obsidian"],
                goal_label="complete nether portal frame",
                budget=args.stage_budget,
                dynamic_goal_provider=portal_goal_provider)

        if results.get("portal", {}).get("success"):
            _equip_hotbar_item(world, "flint_and_steel")
            ignite_adapter = MultiBlockPortalAdapter(
                world, stager, portal_scene, "portal_ignite")
            results["ignite"] = _run_stage(
                world=world, runner=runner, adapter=ignite_adapter,
                out=out / "ignite", goal=goals["ignite"],
                goal_label="netherrack ignition target",
                budget=args.stage_budget, post=args.post)

        stage_success = {
            key: int(results.get(key, {}).get("success", 0))
            for key in ("scoop", "pour", "mine", "portal", "ignite")}
        result = {
            "contract": CONTRACT,
            "state": "complete",
            "world_seed": WORLD_SEED,
            "policy_rollout_seed": int(args.policy_rollout_seed),
            "policy_backend": str(args.policy_backend),
            "model_contract": runner.model_contract,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": PLACE.checkpoint_digest(checkpoint),
            "model_label": str(args.model_label),
            "cfg_coef": float(args.cfg_coef),
            "stage_budget_each": int(args.stage_budget),
            "stage_budgets": {
                "scoop": int(scoop_budget),
                "pour": int(args.stage_budget),
                "mine": int(args.stage_budget),
                "portal": int(args.stage_budget),
                "ignite": int(args.stage_budget),
            },
            "post_non_scoring": int(args.post),
            "scene": scene_manifest,
            "pure_scene_audit": pure_audit,
            "toast_drain": toast_drain,
            "frame0_exact_target_pixels": frame0,
            "system_fsm": list(POLICY_STAGE_KEYS),
            "stage_success": stage_success,
            "results": results,
            "full_episode_goal_overlay": None,
            "world_continuous": True,
            "world_reset_after_frame0": False,
            "player_teleported_after_frame0": False,
            "scoop_or_use_guard_enabled": False,
            "policy_use_or_attack_suppressed": False,
            "policy_action_replayed": False,
            "post_build_terrain_mutation": False,
            "natural_obsidian_tile_drops_enabled": True,
            "success": int(all(stage_success.values())),
        }
        ENGINE.atomic_json(out / "scene_manifest.json", scene_manifest)
        ENGINE.atomic_json(out / "result.json", result)
        print(json.dumps({
            "state": "complete",
            "success": result["success"],
            "stages": stage_success,
            "steps": {
                key: row["steps"] for key, row in results.items()},
            "result": str((out / "result.json").resolve()),
        }, sort_keys=True), flush=True)
    finally:
        world.close()


if __name__ == "__main__":
    main()
