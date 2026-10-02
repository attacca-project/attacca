#!/usr/bin/env python3
"""DPX long-horizon chain (mine an oak log, mine diamond ore, open a crafting table) in one continuous world, with goal-conditioned policy rollouts for the scored stages and crafting-GUI macros between them."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO) + "/src")
if str(REPO / "src/attacca/evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "src/attacca/evaluation"))

from attacca.evaluation import chain_policy as CHAIN
from attacca.evaluation import crafting as CRAFT
from attacca.evaluation import chain_engine as ENGINE
from attacca.evaluation import place as PLACE
from attacca.evaluation.chain_policy import build_runner
from attacca.evaluation.mine_scene import Stager
from attacca.worlds.diamond_pickaxe_chain_scene import DiamondPickaxeChainScene
from attacca.worlds.diamond_pickaxe_chain_scene import SceneBuildResult
from attacca.worlds.diamond_pickaxe_chain_scene import build_scene_commands as build_v1_scene_commands
from attacca.worlds.diamond_pickaxe_chain_scene import validate_scene_manifest as validate_v1_scene_manifest


WORLD_SEED = 1_903_000
CONTRACT = "xbench_diamond_pickaxe_chain/v2"
FRAME_CONTRACT = "xbench_diamond_pickaxe_chain_frame/v1"
TRANSITION_CONTRACT = "xbench_diamond_pickaxe_chain_transition/v1"
STAGES = ("oak_log_mine", "diamond_ore_mine3", "crafting_table_open")
GOAL_LABELS = {
    "oak_log_mine": "Oak Log",
    "diamond_ore_mine3": "Diamond Ore vein",
    "crafting_table_open": "Crafting Table",
}
TARGET_KINDS = {
    "oak_log_mine": "oak_log",
    "diamond_ore_mine3": "diamond_ore",
    "crafting_table_open": "crafting_table",
}
INTERACTION_IDS = {"oak_log_mine": 2, "diamond_ore_mine3": 2,
                   "crafting_table_open": 3}


def configure_runtime_defaults() -> None:
    os.environ.setdefault("MINESTUDIO_DIR", str(REPO / ".minestudio"))
    os.environ.setdefault("MINESTUDIO_GPU_RENDER", "1")
    os.environ.setdefault("RENDER_DEVICES", "0")
QUOTAS = {"oak_log_mine": 1, "diamond_ore_mine3": 3,
          "crafting_table_open": 1}
DEFAULT_GOAL_ROOT = REPO / "assets/goals/dpx"
CAUSAL_FIFO_STEPS = 8


def _kind(value: object) -> str:
    return str(value).removeprefix("minecraft:").split("[", 1)[0]


def _scalar(value: object) -> int:
    try:
        return int(float(np.asarray(value).reshape(-1)[0]))
    except (TypeError, ValueError, IndexError):
        return 0


def _inventory_slots(world) -> dict[str, dict[str, Any]]:
    inventory = (world.info or {}).get("inventory") or {}
    items = inventory.items() if isinstance(inventory, Mapping) else enumerate(inventory)
    return {
        str(slot): {
            "type": _kind(row.get("type", "air")),
            "quantity": _scalar(row.get("quantity", 0)),
        }
        for slot, row in items if isinstance(row, Mapping)
    }


def _inventory_counts(world) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _inventory_slots(world).values():
        kind, count = str(row["type"]), int(row["quantity"])
        if kind not in ("", "air", "none") and count > 0:
            counts[kind] = counts.get(kind, 0) + count
    return dict(sorted(counts.items()))


def _stat_delta(info: Mapping[str, Any], initial: Mapping[str, Any],
                family: str, kind: str) -> int:
    wanted = _kind(kind)
    current = (info or {}).get(family, {}) or {}
    before = (initial or {}).get(family, {}) or {}
    total = 0
    for key, value in current.items():
        if _kind(key) == wanted:
            total += _scalar(value) - _scalar(before.get(key, 0))
    return max(0, int(total))


def _wrong_mine_deltas(info: Mapping[str, Any], initial: Mapping[str, Any],
                       expected: str) -> dict[str, int]:
    current = (info or {}).get("mine_block", {}) or {}
    before = (initial or {}).get("mine_block", {}) or {}
    output: dict[str, int] = {}
    for key, value in current.items():
        kind = _kind(key)
        delta = _scalar(value) - _scalar(before.get(key, 0))
        if delta > 0 and kind != _kind(expected):
            output[kind] = output.get(kind, 0) + int(delta)
    return dict(sorted(output.items()))


def _center_hit(mask: np.ndarray) -> bool:
    height, width = mask.shape
    return bool(mask[max(0, height // 2 - 1):height // 2 + 1,
                     max(0, width // 2 - 1):width // 2 + 1].any())


def _crosshair_target_cell(world, cells: Sequence[Sequence[int]]) -> tuple[int, int, int] | None:
    render_position = tuple(float(value) for value in world.get_pos())
    hits = []
    for raw_cell in cells:
        cell = tuple(int(value) for value in raw_cell)
        mask = ENGINE.block_union(
            world, (cell,), render_position=render_position)
        if _center_hit(mask):
            hits.append(cell)
    return hits[0] if len(hits) == 1 else None


class MineStageAdapter(ENGINE.Adapter):

    def __init__(self, world, stager: Stager, scene: DiamondPickaxeChainScene,
                 task: str, target_cells: Sequence[Sequence[int]]):
        if task not in ("oak_log_mine", "diamond_ore_mine3"):
            raise ValueError(task)
        super().__init__(world, stager, scene, task)
        self.target_cells = tuple(
            tuple(int(value) for value in cell) for cell in target_cells)
        self.target_kind = TARGET_KINDS[task]
        self.pickup_kind = "oak_log" if task == "oak_log_mine" else "diamond"
        self.policy_attribution_mode = (
            "policy_attack_then_exact_designated_cell_transition_fifo")
        self.latest_query = {cell: self.target_kind for cell in self.target_cells}
        self.initial_event_info = {
            "mine_block": copy.deepcopy((world.info or {}).get("mine_block", {}) or {})}
        self.pending_attacks: list[dict[str, Any]] = []
        self.removed_cells: set[tuple[int, int, int]] = set()
        self.attributed_cells: set[tuple[int, int, int]] = set()
        self.removal_events: list[dict[str, Any]] = []
        self.pickup_events: list[dict[str, Any]] = []
        self.initial_pickup_inventory = ENGINE.inventory_count(
            world, self.pickup_kind)
        self.last_inventory = self.initial_pickup_inventory
        self.step_events: dict[int, dict[str, Any]] = {}

    @property
    def interaction_id(self) -> int:
        return INTERACTION_IDS[self.task]

    @property
    def quota(self) -> int:
        return QUOTAS[self.task]

    @property
    def tracking_quota(self) -> int:
        return len(self.target_cells)

    @property
    def progress(self) -> int:
        return min(len(self.attributed_cells), self.quota)

    @property
    def mine_event_delta(self) -> int:
        return _stat_delta(
            self.world.info, self.initial_event_info,
            "mine_block", self.target_kind)

    def pre_context(self) -> dict[str, Any]:
        return {
            "crosshair_target_cell": None,
            "policy_attribution_mode": self.policy_attribution_mode,
            "target_states_before": {
                str(list(cell)): _kind(self.latest_query.get(cell, self.target_kind))
                for cell in self.target_cells},
            "inventory_before": _inventory_counts(self.world),
            "inventory_slots_before": _inventory_slots(self.world),
            "pickup_count_before": ENGINE.inventory_count(
                self.world, self.pickup_kind),
            "mine_event_delta_before": int(self.mine_event_delta),
            "held_before": ENGINE.held_item(self.world),
        }

    def prepare_action(self, action: dict[str, Any]) -> None:
        action["voxels"] = ENGINE.query_box(self.world, self.target_cells)

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        if attack:
            self.pending_attacks.append({"step": int(step)})
        self.pending_attacks = [
            row for row in self.pending_attacks
            if int(step) - int(row["step"]) <= CAUSAL_FIFO_STEPS]
        self.latest_query = ENGINE.current_query(self.world)
        newly_removed = []
        newly_attributed = []
        for cell in self.target_cells:
            if (cell not in self.removed_cells
                    and _kind(self.latest_query.get(cell, "air")) != self.target_kind):
                self.removed_cells.add(cell)
                newly_removed.append(cell)
                causal = self.pending_attacks[-1] if self.pending_attacks else None
                attributed = causal is not None
                if attributed:
                    self.attributed_cells.add(cell)
                    newly_attributed.append(cell)
                self.removal_events.append({
                    "removal_step": int(step), "cell": list(cell),
                    "policy_attack_attributed": attributed,
                    "attack_step": None if causal is None else int(causal["step"]),
                    "confirmation_delay_steps": (
                        None if causal is None else int(step) - int(causal["step"])),
                    "mine_block_stat_delta": int(self.mine_event_delta),
                    "policy_attribution_mode": self.policy_attribution_mode,
                })
        inventory = ENGINE.inventory_count(self.world, self.pickup_kind)
        if inventory > self.last_inventory:
            delta = int(inventory - self.last_inventory)
            self.pickup_events.append({
                "step": int(step), "item": self.pickup_kind,
                "before": int(self.last_inventory), "after": int(inventory),
                "delta": delta,
            })
        self.last_inventory = inventory
        success_now = bool(
            len(self.attributed_cells) >= self.quota
            and inventory >= self.initial_pickup_inventory + self.quota
            and self.success_step is None)
        if success_now:
            self.success_step = int(step)
            self.success_evidence = {
                "step": int(step),
                "initial_target_cells": [list(cell) for cell in self.target_cells],
                "removed_cells": [list(cell) for cell in sorted(self.removed_cells)],
                "policy_attack_attributed_cells": [
                    list(cell) for cell in sorted(self.attributed_cells)],
                "inventory_item": self.pickup_kind,
                "inventory_count": int(inventory),
            }
        wrong = _wrong_mine_deltas(
            self.world.info, self.initial_event_info, self.target_kind)
        event = {
            "step": int(step),
            "target_states_after": {
                str(list(cell)): _kind(self.latest_query.get(cell, "air"))
                for cell in self.target_cells},
            "inventory_after": _inventory_counts(self.world),
            "inventory_slots_after": _inventory_slots(self.world),
            "newly_removed_cells": [list(cell) for cell in newly_removed],
            "newly_attributed_cells": [list(cell) for cell in newly_attributed],
            "pickup_count_after": int(inventory),
            "wrong_target_blocks_removed": int(sum(wrong.values())),
            "wrong_block_deltas": wrong,
        }
        self.step_events[int(step)] = event
        note = ""
        if attack or newly_removed or inventory != int(context["pickup_count_before"]):
            note = (f"MINE attributed={len(self.attributed_cells)}/{self.quota} "
                    f"pickup={inventory}/{self.quota} wrong={sum(wrong.values())}")
        return {"success_now": success_now, "control_complete_now": False,
                "note": note}

    def finish_audit(self) -> bool:
        inventory = ENGINE.inventory_count(self.world, self.pickup_kind)
        wrong = _wrong_mine_deltas(
            self.world.info, self.initial_event_info, self.target_kind)
        passed = bool(
            self.success_step is not None
            and len(self.attributed_cells) >= self.quota
            and inventory >= self.initial_pickup_inventory + self.quota)
        self.terminal_audit = {
            "performed": self.success_step is not None,
            "passed": passed,
            "success_requires_exact_initial_cell_removal": True,
            "success_requires_policy_attack_causal_fifo": True,
            "policy_attribution_mode": self.policy_attribution_mode,
            "pre_action_dense_target_mask_materialized": False,
            "causal_fifo_steps": CAUSAL_FIFO_STEPS,
            "removed_cells": [list(cell) for cell in sorted(self.removed_cells)],
            "policy_attack_attributed_cells": [
                list(cell) for cell in sorted(self.attributed_cells)],
            "pickup_item": self.pickup_kind,
            "pickup_inventory_count": int(inventory),
            "removal_events": self.removal_events,
            "pickup_events": self.pickup_events,
            "wrong_block_deltas": wrong,
            "wrong_target_blocks_removed": int(sum(wrong.values())),
            "clean_success": bool(passed and not wrong),
            "success_evidence": self.success_evidence,
        }
        return passed


class CraftingTableOpenAdapter(ENGINE.Adapter):

    def __init__(self, world, stager: Stager, scene: DiamondPickaxeChainScene):
        super().__init__(world, stager, scene, "crafting_table_open")
        self.policy_attribution_mode = (
            "policy_use_unique_table_interaction_stat_gui_fifo")
        self.target_cell = tuple(scene.crafting_table_cell)
        self.latest_query = {self.target_cell: "crafting_table"}
        self.initial_event_info = {"custom": copy.deepcopy(
            (world.info or {}).get("custom", {}) or {})}
        self.pending_uses: list[dict[str, Any]] = []
        self.step_events: dict[int, dict[str, Any]] = {}

    @property
    def interaction_id(self) -> int:
        return 3

    @property
    def quota(self) -> int:
        return 1

    @property
    def progress(self) -> int:
        return int(self.success_step is not None)

    @property
    def interact_stat_delta(self) -> int:
        return _stat_delta(
            self.world.info, self.initial_event_info,
            "custom", "interact_with_crafting_table")

    def pre_context(self) -> dict[str, Any]:
        return {
            "crosshair_target_cell": None,
            "crosshair_exact_target": False,
            "policy_attribution_mode": self.policy_attribution_mode,
            "target_before": _kind(
                self.latest_query.get(self.target_cell, "crafting_table")),
            "gui_open_before": bool(self.world.info.get("is_gui_open", False)),
            "interact_stat_delta_before": int(self.interact_stat_delta),
            "inventory_before": _inventory_counts(self.world),
            "inventory_slots_before": _inventory_slots(self.world),
            "held_before": ENGINE.held_item(self.world),
        }

    def prepare_action(self, action: dict[str, Any]) -> None:
        action["voxels"] = ENGINE.query_box(self.world, (self.target_cell,))

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        del attack
        if use and context.get("target_before") == "crafting_table":
            self.pending_uses.append({"step": int(step), "cell": list(self.target_cell)})
        self.pending_uses = [
            row for row in self.pending_uses
            if int(step) - int(row["step"]) <= CAUSAL_FIFO_STEPS]
        self.latest_query = ENGINE.current_query(self.world)
        gui = bool(self.world.info.get("is_gui_open", False))
        stat = int(self.interact_stat_delta)
        causal = self.pending_uses[-1] if self.pending_uses else None
        success_now = bool(
            causal is not None and stat > 0 and gui
            and self.success_step is None)
        if success_now:
            self.success_step = int(step)
            self.success_evidence = {
                "step": int(step), "policy_use_step": int(causal["step"]),
                "confirmation_delay_steps": int(step) - int(causal["step"]),
                "pre_action_crosshair_cell": None,
                "policy_attribution_mode": self.policy_attribution_mode,
                "interact_with_crafting_table_stat_delta": stat,
                "is_gui_open_after": gui,
                "gui_semantic": (
                    "crafting_table_container_from_exact_target_use_stat"),
            }
        self.step_events[int(step)] = {
            "inventory_after": _inventory_counts(self.world),
            "inventory_slots_after": _inventory_slots(self.world),
            "gui_open_after": gui,
            "interact_with_crafting_table_stat_delta_after": stat,
            "policy_use_causal": causal is not None,
        }
        note = (f"OPEN table_crosshair={int(bool(context.get('crosshair_exact_target')))} "
                f"stat={stat} gui={int(gui)}") if use or success_now else ""
        return {"success_now": success_now, "control_complete_now": False,
                "note": note}

    def finish_audit(self) -> bool:
        passed = bool(
            self.success_step is not None and self.interact_stat_delta > 0
            and self.world.info.get("is_gui_open", False))
        self.terminal_audit = {
            "performed": self.success_step is not None,
            "passed": passed,
            "target_cell": list(self.target_cell),
            "target_type_after": _kind(
                self.latest_query.get(self.target_cell, "air")),
            "interact_with_crafting_table_stat_delta": int(
                self.interact_stat_delta),
            "is_gui_open_after": bool(self.world.info.get("is_gui_open", False)),
            "policy_attribution_mode": self.policy_attribution_mode,
            "pre_action_dense_target_mask_materialized": False,
            "success_evidence": self.success_evidence,
        }
        return passed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_goal(source: Path, out: Path, task: str
              ) -> tuple[np.ndarray, np.ndarray, dict]:
    source = source.expanduser().resolve()
    meta_path = source / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("task_key") != task
            or meta.get("world_disjoint_from_evaluation") is not True
            or meta.get("mask_backend") != "renderer_depth_owner_exact/v1"):
        raise ValueError(f"invalid {task} donor metadata: {source}")
    expected_cells = 3 if task == "oak_log_mine" else 1
    if int(meta.get("target_cell_count", 0)) != expected_cells:
        raise ValueError(f"{task} donor target count mismatch")
    goal_path, mask_path = source / "goal.png", source / "mask.png"
    contour_path = source / "goal_contour.png"
    for path, key in ((goal_path, "goal_sha256"), (mask_path, "mask_sha256"),
                      (contour_path, "contour_sha256")):
        if not path.is_file() or _sha256(path) != meta.get(key):
            raise ValueError(f"{task} donor SHA mismatch: {path}")
    goal_bgr = cv2.imread(str(goal_path), cv2.IMREAD_COLOR)
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if goal_bgr is None or mask_raw is None:
        raise ValueError(f"could not decode {task} goal")
    mask = np.ascontiguousarray(mask_raw > 0, dtype=np.uint8)
    if not int(mask.sum()) or set(np.unique(mask_raw).tolist()) - {0, 255}:
        raise ValueError(f"{task} goal mask is empty or non-binary")
    out.mkdir(parents=True)
    for path in (goal_path, mask_path, contour_path, meta_path):
        shutil.copy2(path, out / path.name)
    copied_meta = {
        **meta,
        "source": str(source),
        "goal": str((out / "goal.png").resolve()),
        "mask": str((out / "mask.png").resolve()),
        "contour": str((out / "goal_contour.png").resolve()),
    }
    return cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB), mask, copied_meta


_CLEAN_RADIUS_XZ = 80
_CLEAN_Y_BELOW = 4
_CLEAN_Y_ABOVE = 80
_CLEAN_TILE_X = 21
_CLEAN_TILE_Z = 18
_FILL_BLOCK_LIMIT = 32_768
_CLEAN_REPLACE_TAGS = ("#minecraft:logs", "#minecraft:leaves")


def stage_clean_world(world, cx: int, cy: int, cz: int) -> dict[str, Any]:
    x0, x1 = cx - _CLEAN_RADIUS_XZ, cx + _CLEAN_RADIUS_XZ
    y0, y1 = cy - _CLEAN_Y_BELOW, cy + _CLEAN_Y_ABOVE
    z0, z1 = cz - _CLEAN_RADIUS_XZ, cz + _CLEAN_RADIUS_XZ
    fills = 0
    for tx0 in range(x0, x1 + 1, _CLEAN_TILE_X):
        tx1 = min(tx0 + _CLEAN_TILE_X - 1, x1)
        for tz0 in range(z0, z1 + 1, _CLEAN_TILE_Z):
            tz1 = min(tz0 + _CLEAN_TILE_Z - 1, z1)
            volume = (tx1 - tx0 + 1) * (y1 - y0 + 1) * (tz1 - tz0 + 1)
            if volume > _FILL_BLOCK_LIMIT:
                raise AssertionError(f"v_clean /fill tile too large: {volume}")
            coords = f"{tx0} {y0} {tz0} {tx1} {y1} {tz1}"
            for tag in _CLEAN_REPLACE_TAGS:
                world.cmd(f"/fill {coords} minecraft:air replace {tag}")
                fills += 1
    for _ in range(6):
        world.step_noop()
    return {
        "world_variant": "v_clean",
        "center": [cx, cy, cz],
        "radius_xz": _CLEAN_RADIUS_XZ,
        "y_span": [y0, y1],
        "replace_tags": list(_CLEAN_REPLACE_TAGS),
        "fill_commands": fills,
    }


def stage_scene(world, stager: Stager,
                scene: DiamondPickaxeChainScene, *,
                start_pose_override: Sequence[float]
                ) -> tuple[SceneBuildResult, dict[str, Any]]:
    built = build_v1_scene_commands(scene)
    pure_audit = validate_v1_scene_manifest(built, raise_on_error=True)
    for command in built.commands:
        world.cmd(command)
    for _ in range(16):
        world.step_noop()
    world.cmd("/kill @e[type=minecraft:item]")
    for _ in range(4):
        world.step_noop()
    if (_inventory_counts(world) != {"iron_pickaxe": 1}
            or ENGINE.held_item(world) != "iron_pickaxe"):
        raise RuntimeError(
            f"initial inventory mismatch: {_inventory_counts(world)} "
            f"held={ENGINE.held_item(world)}")
    ox, oy, oz, oyaw, opitch = (float(v) for v in start_pose_override)
    if not stager.go(ox, oy, oz, oyaw, pitch=opitch, n=8):
        raise RuntimeError(
            "could not establish overridden Diamond chain start pose")
    for _ in range(6):
        world.step_noop()
    live = {
        "pure_scene_audit": pure_audit,
        "initial_inventory": _inventory_counts(world),
        "initial_held_item": ENGINE.held_item(world),
        "frame0_exact_target_pixels": None,
        "live_exact_mask_materialized": False,
        "oak_start_hidden": None,
        "start_pose_actual": [float(value) for value in world.get_pos()],
        "do_tile_drops": True,
    }
    return built, live


def _enrich_policy_rows(out: Path, adapter: Any, goal_meta: Mapping[str, Any],
                        *, global_start: int) -> tuple[int, list[dict[str, Any]]]:
    path = out / "frames.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for offset, row in enumerate(rows):
        step = int(row["step"])
        row.update({
            "contract": FRAME_CONTRACT,
            "global_chain_step": int(global_start + offset),
            "stage_local_step": step,
            "stage_task_key": adapter.task,
            "goal_asset_sha256": goal_meta["goal_sha256"],
            "target_visible_pixels": (
                None if row["exact_actionable_pixels"] is None
                else int(row["exact_actionable_pixels"])),
            "stage_event": adapter.step_events.get(step),
        })
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(PLACE.json_safe(row), sort_keys=True) + "\n")
    return global_start + len(rows), rows


def _concat_videos(sources: Sequence[Path], out: Path,
                   expected_size: tuple[int, int]) -> dict[str, Any] | None:
    sources = [Path(path) for path in sources if path is not None and Path(path).is_file()]
    if not sources:
        return None
    writer = ENGINE.make_writer(out, expected_size)
    count = 0
    try:
        for source in sources:
            capture = cv2.VideoCapture(str(source))
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if (frame.shape[1], frame.shape[0]) != expected_size:
                        raise RuntimeError(
                            f"video size mismatch {source}: {frame.shape[1]}x{frame.shape[0]}")
                    writer.write(frame)
                    count += 1
            finally:
                capture.release()
    finally:
        writer.release()
    return ENGINE.verify_video(out, expected_frames=count, expected_size=expected_size)


def _run_gui_macro(world, out: Path, method_name: str) -> dict[str, Any]:
    recorder = CRAFT.MacroArtifactRecorder(out)
    try:
        macro = CRAFT.DiamondPickaxeCraftingMacro(
            world, recorder=recorder, settle_frames=2)
        result = getattr(macro, method_name)()
    finally:
        recorder.close()
    return {
        **result,
        **recorder.artifact_manifest(),
        "passed": bool(result.get("success")),
        "item_injection_count": int(result.get("inventory_injection_count", 0)),
    }


def _transition(from_task: str, to_task: str,
                trigger: str) -> dict[str, Any]:
    return {
        "contract": TRANSITION_CONTRACT,
        "from": from_task, "to": to_task, "trigger": trigger,
        "performed": True, "world_reset": False, "player_teleported": False,
        "camera_forced": False, "transition_card_frames": 0,
        "goal_token_changed": True,
    }

def main(argv: Sequence[str] | None = None) -> None:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal-root", type=Path, default=DEFAULT_GOAL_ROOT)
    parser.add_argument("--model-label", default="ours")
    parser.add_argument("--cfg-coef", type=float, default=0.0)
    parser.add_argument("--oak-budget", type=int, default=300)
    parser.add_argument("--diamond-budget", type=int, default=600)
    parser.add_argument("--table-budget", type=int, default=300)
    parser.add_argument(
        "--scene-version",
        choices=("v1_rock_bend",),
        default="v1_rock_bend")
    parser.add_argument(
        "--clean-world", action="store_true",
        help=(
            "use the cleaned evaluation world variant: strip natural logs and "
            "leaves in a finite box around the arena, camera-off, before "
            "staging rebuilds the arena and its oak canopy."))
    parser.add_argument(
        "--start-pose", required=True,
        help="start pose as 'x,y,z,yaw,pitch'")
    CHAIN.add_policy_arguments(parser)
    args = parser.parse_args(cli_args)
    out = args.out.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    goal_root = args.goal_root.expanduser().resolve()
    if out.exists():
        raise FileExistsError(out)
    if not checkpoint.is_file() and not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if min(args.oak_budget, args.diamond_budget, args.table_budget) <= 0:
        raise ValueError("all policy budgets must be positive")
    out.mkdir(parents=True)

    ENGINE.CONTRACT = CONTRACT
    ENGINE.FRAME_CONTRACT = FRAME_CONTRACT
    ENGINE.WORLD_SEEDS.update({task: WORLD_SEED for task in STAGES})
    runner = build_runner(args)
    start_pose_override = [float(v) for v in str(args.start_pose).split(",")]
    if len(start_pose_override) != 5:
        raise ValueError(
            "--start-pose must be 'x,y,z,yaw,pitch' (5 comma values)")
    world = PLACE._boot_eval(WORLD_SEED)
    try:
        stager = Stager(world)
        px, py, pz = world.get_pos()
        clean_world_audit = None
        if args.clean_world:
            clean_world_audit = stage_clean_world(
                world, math.floor(px), math.floor(py), math.floor(pz))
        scene = DiamondPickaxeChainScene(
            math.floor(px), math.floor(py), math.floor(pz))
        built, live_scene_audit = stage_scene(
            world, stager, scene, start_pose_override=start_pose_override)
        if clean_world_audit is not None:
            live_scene_audit["clean_world"] = clean_world_audit
        live_scene_audit["start_pose_override"] = list(start_pose_override)
        toast_drain = ENGINE.drain_toasts_camera_off(world)
        ENGINE.save_rgb(
            out / "initial_scene.png",
            np.ascontiguousarray(world.info["pov"], dtype=np.uint8))
        ENGINE.atomic_json(out / "scene_manifest.json", {
            **built.manifest,
            "live_scene_audit": live_scene_audit,
            "toast_drain": toast_drain,
        })

        goals = {}
        for task in STAGES:
            stage_dir = out / task
            stage_dir.mkdir()
            goals[task] = load_goal(
                goal_root / task, stage_dir / "goal", task)

        results: dict[str, Any] = {}
        transitions: list[dict[str, Any]] = []
        macros: dict[str, Any] = {}
        inventory_ledger = [{"point": "initial", "counts": _inventory_counts(world)}]
        global_step = 0
        raw_segments: list[Path] = []

        oak = MineStageAdapter(
            world, stager, scene, "oak_log_mine", scene.oak_target_cells)
        ENGINE.initial_artifacts(world, out / "oak_log_mine")
        results["oak_log_mine"] = ENGINE.rollout(
            world=world, runner=runner, adapter=oak, out=out / "oak_log_mine",
            goal_rgb=goals["oak_log_mine"][0], goal_mask=goals["oak_log_mine"][1],
            goal_meta=goals["oak_log_mine"][2], goal_label=GOAL_LABELS["oak_log_mine"],
            budget=int(args.oak_budget), post=0,
            post_action_mode="policy")
        global_step, _ = _enrich_policy_rows(
            out / "oak_log_mine", oak, goals["oak_log_mine"][2],
            global_start=global_step)
        raw_segments.append(Path(results["oak_log_mine"]["raw_video"]))
        inventory_ledger.append({"point": "post_oak", "counts": _inventory_counts(world)})

        transition_a = None
        if results["oak_log_mine"]["success"]:
            world.cmd("/kill @e[type=minecraft:item]")
            macro_a_dir = out / "macro_a_sticks"
            macros["sticks"] = _run_gui_macro(
                world, macro_a_dir, "run_macro_a")
            inventory_ledger.append({
                "point": "post_macro_a", "counts": _inventory_counts(world)})
            transition_a = _transition(
                "oak_log_mine", "diamond_ore_mine3",
                "oak_removed_and_collected_then_real_gui_sticks_crafted")
            transitions.append(transition_a)
            raw_segments.append(Path(macros["sticks"]["raw_video"]))
            diamond = MineStageAdapter(
                world, stager, scene, "diamond_ore_mine3", scene.diamond_ore_cells)
            ENGINE.initial_artifacts(world, out / "diamond_ore_mine3")
            results["diamond_ore_mine3"] = ENGINE.rollout(
                world=world, runner=runner, adapter=diamond,
                out=out / "diamond_ore_mine3",
                goal_rgb=goals["diamond_ore_mine3"][0],
                goal_mask=goals["diamond_ore_mine3"][1],
                goal_meta=goals["diamond_ore_mine3"][2],
                goal_label=GOAL_LABELS["diamond_ore_mine3"],
                budget=int(args.diamond_budget), post=0,
                post_action_mode="policy")
            global_step, _ = _enrich_policy_rows(
                out / "diamond_ore_mine3", diamond,
                goals["diamond_ore_mine3"][2], global_start=global_step)
            raw_segments.append(Path(results["diamond_ore_mine3"]["raw_video"]))
            inventory_ledger.append({
                "point": "post_diamond", "counts": _inventory_counts(world)})

        transition_b = None
        if results.get("diamond_ore_mine3", {}).get("success"):
            world.cmd("/kill @e[type=minecraft:item]")
            transition_b = _transition(
                "diamond_ore_mine3", "crafting_table_open",
                "three_exact_diamond_ores_removed_and_collected")
            transitions.append(transition_b)
            table = CraftingTableOpenAdapter(world, stager, scene)
            ENGINE.initial_artifacts(world, out / "crafting_table_open")
            results["crafting_table_open"] = ENGINE.rollout(
                world=world, runner=runner, adapter=table,
                out=out / "crafting_table_open",
                goal_rgb=goals["crafting_table_open"][0],
                goal_mask=goals["crafting_table_open"][1],
                goal_meta=goals["crafting_table_open"][2],
                goal_label=GOAL_LABELS["crafting_table_open"],
                budget=int(args.table_budget), post=0,
                post_action_mode="observe_noop")
            global_step, _ = _enrich_policy_rows(
                out / "crafting_table_open", table,
                goals["crafting_table_open"][2], global_start=global_step)
            raw_segments.append(Path(results["crafting_table_open"]["raw_video"]))

        policy_chain_success = bool(
            results.get("crafting_table_open", {}).get("success"))
        if policy_chain_success:
            macros["diamond_pickaxe"] = _run_gui_macro(
                world, out / "macro_b_diamond_pickaxe", "run_macro_b")
            inventory_ledger.append({
                "point": f"post_macro_b_and_noop{CRAFT.EQUIPPED_PROOF_FRAMES}",
                "counts": _inventory_counts(world),
                "held_item": ENGINE.held_item(world),
            })
            raw_segments.append(Path(macros["diamond_pickaxe"]["raw_video"]))

        combined_raw = _concat_videos(
            raw_segments, out / "diamond_pickaxe_chain_raw_combined.mp4",
            (640, 360))
        demonstration_complete = bool(
            policy_chain_success
            and macros.get("diamond_pickaxe", {}).get("passed"))
        macro_item_injection_count = int(sum(
            int(row.get("item_injection_count", 0)) for row in macros.values()))
        final_counts = _inventory_counts(world)
        conservation = {
            "oak_log_mined": int(
                results.get("oak_log_mine", {}).get("success", 0)),
            "oak_log_consumed_exactly_1": bool(
                macros.get("sticks", {}).get("item_conservation", {}).get(
                    "oak_log_consumed") == 1),
            "oak_planks_produced_4_consumed_2_remaining_2": final_counts.get(
                "oak_planks", 0) == 2 if "sticks" in macros else False,
            "sticks_produced_4_consumed_2_remaining_2": final_counts.get(
                "stick", 0) == 2 if "diamond_pickaxe" in macros else False,
            "diamonds_mined_3_consumed_3": bool(
                "diamond_pickaxe" in macros and final_counts.get("diamond", 0) == 0),
            "diamond_pickaxe_crafted_not_injected": bool(
                final_counts.get("diamond_pickaxe", 0) >= 1
                and macros.get("diamond_pickaxe", {}).get("item_injection_count", 1) == 0),
        }
        result = {
            "contract": CONTRACT,
            "state": "complete",
            "world_seed": WORLD_SEED,
            "fixed_world_single_layout": True,
            "policy_rollout_seed": int(args.policy_rollout_seed),
            "policy_backend": str(args.policy_backend),
            "model_label": str(args.model_label),
            "scene_version": str(args.scene_version),
            "artifact_mode": "deferred",
            "equipped_proof_frames_requested": CRAFT.EQUIPPED_PROOF_FRAMES,
            "live_exact_masks_materialized": False,
            "inline_prediction_overlays_rendered": False,
            "model_contract": runner.model_contract,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": PLACE.checkpoint_digest(checkpoint),
            "scene_manifest": str((out / "scene_manifest.json").resolve()),
            "goals": {task: goal[2] for task, goal in goals.items()},
            "system_fsm": list(STAGES),
            "interaction_id_sequence": [2, 2, 3],
            "results": results,
            "transitions": transitions,
            "system_macros": macros,
            "inventory_ledger": inventory_ledger,
            "item_conservation": conservation,
            "policy_chain_success_step": (
                results.get("crafting_table_open", {}).get("success_step")),
            "policy_chain_success": policy_chain_success,
            "demonstration_complete": demonstration_complete,
            "macro_item_injection_count": macro_item_injection_count,
            "item_injection_count": macro_item_injection_count,
            "world_continuous": True,
            "combined_raw_video": str(
                (out / "diamond_pickaxe_chain_raw_combined.mp4").resolve())
                if combined_raw else None,
            "combined_review_overlay": None,
            "combined_raw_receipt": combined_raw,
            "combined_review_receipt": None,
            "final_inventory": final_counts,
            "final_held_item": ENGINE.held_item(world),
        }
        ENGINE.atomic_json(out / "result.json", result)
        print(json.dumps({
            "state": "complete",
            "policy_chain_success": int(policy_chain_success),
            "demonstration_complete": int(demonstration_complete),
            "stages": {name: row["success"] for name, row in results.items()},
            "result": str((out / "result.json").resolve()),
        }, sort_keys=True), flush=True)
    finally:
        world.close()


if __name__ == "__main__":
    configure_runtime_defaults()
    main()
