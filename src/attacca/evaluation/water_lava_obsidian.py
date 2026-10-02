#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO) + "/src")
if str(REPO / "src/attacca/evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "src/attacca/evaluation"))

from attacca.evaluation import chain_engine as ENGINE


WATER_GOAL_SOURCE = REPO / "assets" / "goals" / "wlo" / "water"
LAVA_GOAL_SOURCE = REPO / "assets" / "goals" / "wlo" / "lava"
OBSIDIAN_GOAL_SOURCE = REPO / "assets" / "goals" / "wlo" / "obsidian"


def capture_goals(root: Path, water_goal_source: Path,
                  lava_goal_source: Path, obsidian_goal_source: Path):
    outputs = {name: root / name for name in ("scoop", "pour", "mine")}
    for out in outputs.values():
        out.mkdir()
    scoop_rgb, scoop_mask, scoop_meta = ENGINE.load_copied_goal(
        water_goal_source, outputs["scoop"],
        semantic="world_disjoint_stone_floor_1x1_water_goal")
    source_meta_path = water_goal_source / "meta.json"
    if not source_meta_path.is_file():
        raise FileNotFoundError(source_meta_path)
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    if not bool(source_meta.get("world_disjoint_from_evaluation")):
        raise RuntimeError("water goal donor must be world-disjoint")
    if source_meta.get("mask_semantic") not in {
            "all_visible_owner_pixels_of_entire_exposed_water_component",
            "all_visible_owner_pixels_of_staged_2x2_source_pool/v1",
            "all_visible_owner_pixels_of_staged_1x1_source_pool/v1"}:
        raise RuntimeError("water goal does not carry the full-component mask")
    if (source_meta.get("pool_shape") != [1, 1]
            or source_meta.get("background") != "plain_stone_floor"):
        raise RuntimeError("water goal must be the stone-floor 1x1 asset")
    scoop_meta.update({
        "source_contract": source_meta.get("contract"),
        "donor_world_seed": source_meta.get("donor_world_seed"),
        "evaluation_world_seed": source_meta.get("evaluation_world_seed"),
        "world_disjoint_from_evaluation": True,
        "component_cell_count": source_meta.get("component_cell_count"),
        "visible_mask_pixels": source_meta.get("visible_mask_pixels"),
        "mask_semantic": source_meta.get("mask_semantic"),
    })
    scoop = scoop_rgb, scoop_mask, scoop_meta

    pour_rgb, pour_mask, pour_meta = ENGINE.load_copied_goal(
        lava_goal_source, outputs["pour"],
        semantic="world_disjoint_stone_floor_1x1_lava_goal")
    lava_source_meta = json.loads(
        (lava_goal_source / "meta.json").read_text(encoding="utf-8"))
    if not bool(lava_source_meta.get("world_disjoint_from_evaluation")):
        raise RuntimeError("lava goal donor must be world-disjoint")
    if lava_source_meta.get("mask_semantic") not in {
            "all_visible_owner_pixels_of_entire_exposed_lava_component",
            "all_visible_owner_pixels_of_staged_2x2_source_pool/v1",
            "all_visible_owner_pixels_of_staged_1x1_source_pool/v1"}:
        raise RuntimeError(
            "lava goal does not carry the full natural-component mask")
    if (lava_source_meta.get("pool_shape") != [1, 1]
            or lava_source_meta.get("background") != "plain_stone_floor"):
        raise RuntimeError("lava goal must be the stone-floor 1x1 asset")
    pour_meta.update({
        "source_contract": lava_source_meta.get("contract"),
        "donor_world_seed": lava_source_meta.get("donor_world_seed"),
        "evaluation_world_seed": lava_source_meta.get("evaluation_world_seed"),
        "world_disjoint_from_evaluation": True,
        "component_cell_count": lava_source_meta.get("component_cell_count"),
        "visible_mask_pixels": lava_source_meta.get("visible_mask_pixels"),
        "mask_semantic": lava_source_meta.get("mask_semantic"),
    })
    pour = pour_rgb, pour_mask, pour_meta

    mine_rgb, mine_mask, mine_meta = ENGINE.load_copied_goal(
        obsidian_goal_source, outputs["mine"],
        semantic="world_disjoint_obsidian_on_plains_grass")
    obsidian_source_meta = json.loads(
        (obsidian_goal_source / "meta.json").read_text(encoding="utf-8"))
    if not bool(obsidian_source_meta.get("world_disjoint_from_evaluation")):
        raise RuntimeError("obsidian goal donor must be world-disjoint")
    if obsidian_source_meta.get("mask_semantic") != (
            "one_exact_visible_obsidian_block_surface/v1"):
        raise RuntimeError("obsidian goal is not the standard grass block goal")
    mine_meta.update({
        "source_contract": obsidian_source_meta.get("source_contract"),
        "donor_world_seed": obsidian_source_meta.get("donor_world_seed"),
        "evaluation_world_seed": obsidian_source_meta.get(
            "evaluation_world_seed"),
        "world_disjoint_from_evaluation": True,
        "background": obsidian_source_meta.get("background"),
        "mask_semantic": obsidian_source_meta.get("mask_semantic"),
        "visible_mask_pixels": obsidian_source_meta.get(
            "native_geometry", {}).get("area"),
    })
    mine = mine_rgb, mine_mask, mine_meta
    return {"scoop": scoop, "pour": pour, "mine": mine}
