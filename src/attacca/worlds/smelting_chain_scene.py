from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from attacca.worlds.diamond_pickaxe_chain_scene import Cell
from attacca.worlds.diamond_pickaxe_chain_scene import SceneBuildResult
from attacca.worlds.diamond_pickaxe_chain_scene import _CommandBuilder
from attacca.worlds.diamond_pickaxe_chain_scene import _base_kind
from attacca.worlds.diamond_pickaxe_chain_scene import _neighbors4
from attacca.worlds.diamond_pickaxe_chain_scene import _shortest_path


SCENE_CONTRACT = "xbench_smelting_chain_scene/v1"
DISTANCE_BLOCKS = {"S0": 20}
ORE_SPECS = {
    "iron_ore": {"quota": 3, "tool": "stone_pickaxe",
                 "title": "Smelt Iron", "stage": "iron_ore_mine3"},
}
INTERACTION_IDS = (2, 2, 3)


@dataclass(frozen=True)
class SmeltingChainScene:
    center_x: int
    feet_y: int
    center_z: int
    ore_kind: str = "iron_ore"
    distance_band: str = "S0"

    def __post_init__(self) -> None:
        if self.ore_kind not in ORE_SPECS:
            raise ValueError(f"unsupported smelting ore: {self.ore_kind}")
        if self.distance_band not in DISTANCE_BLOCKS:
            raise ValueError(f"unsupported distance band: {self.distance_band}")

    @property
    def origin(self) -> Cell:
        return self.center_x, self.feet_y, self.center_z

    @property
    def distance_blocks(self) -> int:
        return DISTANCE_BLOCKS[self.distance_band]

    @property
    def ore_spec(self) -> dict:
        return ORE_SPECS[self.ore_kind]

    @property
    def policy_stage_keys(self) -> tuple[str, str, str]:
        return "coal_ore_mine", self.ore_spec["stage"], "furnace_open"

    @property
    def coal_ore_cells(self) -> tuple[Cell, ...]:
        return ((self.center_x - self.distance_blocks // 2,
                 self.feet_y + 1, self.center_z),)

    @property
    def ore_cells(self) -> tuple[Cell, ...]:
        x = self.center_x + self.distance_blocks // 2
        return tuple((x, self.feet_y + 1, self.center_z + dz)
                     for dz in (0, -1, 1)[:self.ore_spec["quota"]])

    @property
    def furnace_cell(self) -> Cell:
        return (self.center_x + self.distance_blocks // 2,
                self.feet_y, self.center_z - self.distance_blocks)

    @property
    def start_navigation_cell(self) -> Cell:
        coal = self.coal_ore_cells[0]
        return coal[0] + 4, self.feet_y, coal[2] + 4

    @property
    def start_pose(self) -> tuple[float, float, float]:
        x, y, z = self.start_navigation_cell
        return x + 0.5, float(y), z + 0.5

    @property
    def target_cells(self) -> dict[str, tuple[Cell, ...]]:
        coal, ore, furnace = self.policy_stage_keys
        return {coal: self.coal_ore_cells, ore: self.ore_cells,
                furnace: (self.furnace_cell,)}

    @property
    def stage_stance_cells(self) -> dict[str, tuple[Cell, ...]]:
        coal, ore, furnace = self.policy_stage_keys
        cx, _cy, cz = self.coal_ore_cells[0]
        ox, _oy, oz = self.ore_cells[0]
        fx, fy, fz = self.furnace_cell
        return {
            coal: ((cx + 1, self.feet_y, cz),
                   (cx + 1, self.feet_y, cz + 1)),
            ore: tuple((ox - 1, self.feet_y, oz + dz) for dz in (-1, 0, 1)),
            furnace: ((fx, fy, fz + 1), (fx - 1, fy, fz),
                      (fx + 1, fy, fz)),
        }


def _fill_tiled(builder: _CommandBuilder, first: Cell, last: Cell,
                block: str) -> None:
    x0, y0, z0 = first
    x1, y1, z1 = last
    for x in range(x0, x1 + 1, 24):
        for z in range(z0, z1 + 1, 24):
            for y in range(y0, y1 + 1, 24):
                builder.fill((x, y, z),
                             (min(x + 23, x1), min(y + 23, y1),
                              min(z + 23, z1)), block)


def _occluding_cells(blocks: Mapping[Cell, str], eye: Sequence[float],
                     target: Cell) -> tuple[Cell, ...]:
    end = (target[0] + 0.5, target[1] + 0.5, target[2] + 0.5)
    length = math.dist(eye, end)
    seen: set[Cell] = set()
    for step in range(1, max(2, math.ceil(length * 8))):
        fraction = step / max(2, math.ceil(length * 8))
        cell = tuple(math.floor(eye[i] + fraction * (end[i] - eye[i]))
                     for i in range(3))
        if cell != target and _base_kind(blocks.get(cell, "air")) != "air":
            seen.add(cell)
    return tuple(sorted(seen))


def build_scene_commands(scene: SmeltingChainScene) -> SceneBuildResult:
    cx, y, cz = scene.origin
    distance = scene.distance_blocks
    coal_x = scene.coal_ore_cells[0][0]
    ore_x = scene.ore_cells[0][0]
    x0, x1 = coal_x - 10, ore_x + 12
    z0, z1 = cz - distance - 10, cz + 12
    builder = _CommandBuilder()
    for command in (
        "/gamerule doMobSpawning false", "/gamerule doDaylightCycle false",
        "/gamerule doWeatherCycle false", "/gamerule doTileDrops true",
        "/gamerule randomTickSpeed 0",
        "/gamerule fallDamage false", "/gamerule showDeathMessages false",
        "/gamerule sendCommandFeedback false", "/gamerule announceAdvancements false",
        "/time set day", "/weather clear",
    ):
        builder.command(command)

    _fill_tiled(builder, (x0, y, z0), (x1, y + 20, z1), "air")
    _fill_tiled(builder, (x0, y - 4, z0), (x1, y - 2, z1), "stone")
    _fill_tiled(builder, (x0, y - 1, z0), (x1, y - 1, z1), "grass_block")

    for first, last in (
        ((x0, y, z0), (x1, y + 2, z0 + 2)),
        ((x0, y, z1 - 2), (x1, y + 2, z1)),
        ((x0, y, z0 + 3), (x0 + 2, y + 2, z1 - 3)),
        ((x1 - 2, y, z0 + 3), (x1, y + 2, z1 - 3)),
    ):
        _fill_tiled(builder, first, last, "dirt")
        _fill_tiled(builder, (first[0], y + 3, first[2]),
                    (last[0], y + 3, last[2]), "grass_block")

    host_boxes = (
        ((coal_x - 2, y, cz - 2), (coal_x, y + 2, cz + 2)),
        ((coal_x - 2, y + 3, cz - 1), (coal_x - 1, y + 3, cz + 1)),
        ((ore_x, y, cz - 3), (ore_x + 3, y + 2, cz + 3)),
        ((ore_x + 1, y + 3, cz - 2), (ore_x + 3, y + 3, cz + 2)),
        ((cx - 2, y, cz - 3), (cx + 2, y + 2, cz + 5)),
        ((cx - 1, y + 3, cz - 2), (cx + 1, y + 3, cz + 3)),
        ((ore_x - 8, y, cz - distance + 8),
         (ore_x + 3, y + 2, cz - distance + 12)),
        ((ore_x - 6, y + 3, cz - distance + 9),
         (ore_x + 1, y + 3, cz - distance + 11)),
    )
    for first, last in host_boxes:
        builder.fill(first, last, "stone")

    fx, fy, fz = scene.furnace_cell
    for dx, dz in ((-2, 0), (-2, -1), (0, -2), (1, -2)):
        builder.setblock((fx + dx, fy - 1, fz + dz), "cobblestone")
    for cell in scene.coal_ore_cells:
        builder.setblock(cell, "coal_ore")
    for cell in scene.ore_cells:
        builder.setblock(cell, scene.ore_kind)
    builder.setblock(scene.furnace_cell, "furnace[facing=south,lit=false]")

    for command in ("/kill @e[type=!minecraft:player]",
                    "/kill @e[type=minecraft:item]", "/clear @a",
                    "/effect clear @a", "/time set day"):
        builder.command(command)
    tool = scene.ore_spec["tool"]
    builder.command(f"/replaceitem entity @p hotbar.0 minecraft:{tool} 1")

    blocks = builder.blocks
    navigation = {
        (x, y, z)
        for x in range(x0 + 3, x1 - 2)
        for z in range(z0 + 3, z1 - 2)
        if _base_kind(blocks[(x, y, z)]) == "air"
        and _base_kind(blocks[(x, y + 1, z)]) == "air"
        and _base_kind(blocks[(x, y - 1, z)]) not in {"air", "water", "lava"}
    }
    paths = {}
    departure = scene.start_navigation_cell
    for stage in scene.policy_stage_keys:
        path = _shortest_path(navigation, departure, scene.stage_stance_cells[stage])
        paths[stage] = path
        if path:
            departure = path[-1]

    eye = (scene.start_pose[0], scene.start_pose[1] + 1.62, scene.start_pose[2])
    witnesses = {
        stage: [list(cell) for cell in _occluding_cells(blocks, eye, targets[0])]
        for stage, targets in scene.target_cells.items()
    }
    tx, ty, tz = scene.coal_ore_cells[0]
    yaw = math.degrees(math.atan2(-(tx + 0.5 - eye[0]), tz + 0.5 - eye[2]))
    pitch = -math.degrees(math.atan2(ty + 0.5 - eye[1],
                                   math.hypot(tx + 0.5 - eye[0], tz + 0.5 - eye[2])))
    target_supports = {
        stage: [[x, yy - 1, z] for x, yy, z in targets]
        for stage, targets in scene.target_cells.items()
    }
    object_columns = sorted({(x, z) for (x, yy, z), block in blocks.items()
                             if yy >= y and _base_kind(block) != "air"})
    column_supports = [(x, y - 1, z) for x, z in object_columns]
    verified_stands = {scene.start_navigation_cell}
    for stands in scene.stage_stance_cells.values():
        verified_stands.update(stands)
    counts = Counter(_base_kind(block) for block in blocks.values())
    stages = scene.policy_stage_keys
    manifest = {
        "contract": SCENE_CONTRACT,
        "scene_name": "smelting_quarry_chain",
        "layout_id": f"smelting_quarry_shared_v1_{scene.distance_band}",
        "chain_title": scene.ore_spec["title"],
        "ore_kind": scene.ore_kind,
        "distance_band": scene.distance_band,
        "distance_blocks": distance,
        "world_seed": 1903000,
        "fixed_world_single_layout": True,
        "origin": list(scene.origin),
        "reset_bounds": {"x_min": x0, "x_max": x1, "y_min": y - 4,
                         "y_max": y + 20, "z_min": z0, "z_max": z1},
        "start_pose": list(scene.start_pose),
        "start_navigation_cell": list(scene.start_navigation_cell),
        "start_yaw": yaw, "start_pitch": pitch,
        "policy_stage_keys": list(stages),
        "target_kinds": ["coal_ore", scene.ore_kind, "furnace"],
        "success_quotas": [1, scene.ore_spec["quota"], 1],
        "interaction_ids": list(INTERACTION_IDS),
        "macros_enabled": False,
        "causal_narrative": {
            stages[0]: "Mine coal to fuel the furnace.",
            stages[1]: f"Mine {scene.ore_spec['quota']} {scene.ore_kind} blocks as smelting input.",
            stages[2]: "Open the furnace with the mined fuel and ore to begin smelting.",
            "scored_endpoint": "furnace GUI open",
        },
        "target_cells": {stage: [list(c) for c in cells]
                         for stage, cells in scene.target_cells.items()},
        "stage_stance_cells": {stage: [list(c) for c in cells]
                               for stage, cells in scene.stage_stance_cells.items()},
        "target_support_cells": target_supports,
        "placed_object_supports": [list(c) for c in column_supports],
        "verified_stand_cells": [list(c) for c in sorted(verified_stands)],
        "navigation_cells": [list(c) for c in sorted(navigation)],
        "navigation_paths": {stage: [list(c) for c in path]
                             for stage, path in paths.items()},
        "navigation_distances_blocks": {
            "start_to_coal_target": math.dist(scene.start_pose[::2],
                                               (tx + 0.5, tz + 0.5)),
            "coal_to_ore_target": float(distance),
            "ore_to_furnace_target": float(distance),
        },
        "start_visibility_contract": {
            "coal_center_ray_clear": not witnesses[stages[0]],
            "hidden_stage_keys": list(stages[1:]),
            "center_ray_occluder_cells": witnesses,
            "live_exact_mask_required": True,
        },
        "initial_inventory": {
            "clear_before_equip": True,
            "slots": {"hotbar.0": {"item": f"minecraft:{tool}", "count": 1,
                                    "enchanted": False}},
            "selected_hotbar_slot": 1,
            "selection_control": "camera_off_hotbar.1",
            "held_item_after_selection": f"minecraft:{tool}",
            "all_other_slots_empty": True,
        },
        "item_provenance": {
            "do_tile_drops": True,
            "intermediate_item_injection_allowed": False,
            "injected_materials": [],
            "initial_kit_only": [tool],
        },
        "staging": {
            "camera_off": True, "mobs_purged_after_build": True,
            "dropped_items_purged_after_build": True,
            "fluid_purge": "entire reset volume cleared and solid substrate rebuilt",
            "weather": "clear_locked", "daylight": "day_locked",
            "per_column_grounding": "authored ground; live voxel probes required",
            "teleport_stand_probes_required": True,
            "frame_guard": {"min_mean_rgb": 25.0, "max_abs_dy": 1.5},
        },
        "final_block_counts": dict(sorted(counts.items())),
    }
    result = SceneBuildResult(tuple(builder.commands), manifest, dict(blocks))
    validate_scene_manifest(result, raise_on_error=True)
    return result


def validate_scene_manifest(result: SceneBuildResult, *,
                            raise_on_error: bool = False) -> dict:
    manifest, blocks = result.manifest, result.final_blocks
    issues = []
    solid = lambda cell: _base_kind(blocks.get(tuple(cell), "air")) not in {
        "air", "water", "lava"}
    if manifest.get("contract") != SCENE_CONTRACT:
        issues.append("scene contract mismatch")
    ore_kind = manifest.get("ore_kind")
    ore_spec = ORE_SPECS.get(ore_kind, {})
    stages = tuple(manifest.get("policy_stage_keys", ()))
    expected_stages = ("coal_ore_mine", ore_spec.get("stage"), "furnace_open")
    if stages != expected_stages:
        issues.append("stage keys mismatch")
    if manifest.get("success_quotas") != [1, ore_spec.get("quota"), 1]:
        issues.append("stage quotas mismatch")
    if manifest.get("target_kinds") != ["coal_ore", ore_kind, "furnace"]:
        issues.append("target kinds mismatch")
    if manifest.get("interaction_ids") != [2, 2, 3]:
        issues.append("interaction IDs mismatch")
    if manifest.get("macros_enabled") is not False:
        issues.append("smelting scene must use three policy rollouts without macros")
    counts = Counter(_base_kind(block) for block in blocks.values())
    for stage, kind, quota in zip(stages, manifest.get("target_kinds", ()),
                                  manifest.get("success_quotas", ())):
        targets = manifest.get("target_cells", {}).get(stage, [])
        if len(targets) != quota or counts[kind] != quota:
            issues.append(f"{stage}: target count differs from exact quota")
        for cell in targets:
            if _base_kind(blocks.get(tuple(cell), "air")) != kind:
                issues.append(f"{stage}: target voxel kind mismatch {cell}")
            support = (cell[0], cell[1] - 1, cell[2])
            if not solid(support):
                issues.append(f"{stage}: unsupported target {cell}")
        if not manifest.get("navigation_paths", {}).get(stage):
            issues.append(f"{stage}: no reachable verified stance")
    for support in manifest.get("placed_object_supports", []):
        if not solid(support):
            issues.append(f"unsupported placed-object column {support}")
    navigation = {tuple(c) for c in manifest.get("navigation_cells", [])}
    for stand in manifest.get("verified_stand_cells", []):
        x, y, z = stand
        if (not solid((x, y - 1, z)) or
                _base_kind(blocks.get((x, y, z), "air")) != "air" or
                _base_kind(blocks.get((x, y + 1, z), "air")) != "air"):
            issues.append(f"unsafe stand {stand}")
        if tuple(stand) not in navigation:
            issues.append(f"stand outside navigation {stand}")
    for stage, path in manifest.get("navigation_paths", {}).items():
        for first, last in zip(path, path[1:]):
            if tuple(last) not in _neighbors4(tuple(first)):
                issues.append(f"{stage}: discontinuous route")
        if any(tuple(c) not in navigation for c in path):
            issues.append(f"{stage}: route leaves safe navigation")
    visibility = manifest.get("start_visibility_contract", {})
    witnesses = visibility.get("center_ray_occluder_cells", {})
    if not visibility.get("coal_center_ray_clear"):
        issues.append("coal is not initially visible by center ray")
    for stage in visibility.get("hidden_stage_keys", []):
        if not witnesses.get(stage):
            issues.append(f"{stage}: missing start occlusion witness")
    for command in result.commands:
        if command.startswith("/fill "):
            fields = command.split()
            values = [int(v) for v in fields[1:7]]
            if math.prod(abs(values[i + 3] - values[i]) + 1 for i in range(3)) > 32768:
                issues.append("fill command exceeds Minecraft volume limit")
    expected_tool = f"minecraft:{ore_spec.get('tool')}"
    if manifest.get("initial_inventory", {}).get("held_item_after_selection") != expected_tool:
        issues.append("initial tool mismatch")
    audit = {"passed": not issues, "issues": issues,
             "support_columns": len(manifest.get("placed_object_supports", [])),
             "verified_stands": len(manifest.get("verified_stand_cells", [])),
             "target_counts": {kind: counts[kind]
                               for kind in manifest.get("target_kinds", [])},
             "pure_only": True, "live_visual_verification_performed": False}
    if issues and raise_on_error:
        raise RuntimeError("invalid smelting scene: " + "; ".join(issues))
    return audit
