from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from attacca.worlds.diamond_pickaxe_chain_scene import Cell
from attacca.worlds.diamond_pickaxe_chain_scene import SceneBuildResult
from attacca.worlds.diamond_pickaxe_chain_scene import _base_kind
from attacca.worlds.diamond_pickaxe_chain_scene import _shortest_path
from attacca.worlds.smelting_chain_scene import DISTANCE_BLOCKS
from attacca.worlds.smelting_chain_scene import SmeltingChainScene
from attacca.worlds.smelting_chain_scene import build_scene_commands as build_smelting_scene_commands


SCENE_CONTRACT = "xbench_coal_cow_furnace_chain_scene/v2"
WORLD_SEED = 1903000
POLICY_STAGE_KEYS = (
    "coal_ore_mine", "cow_hunt", "furnace_find", "beef_cook")
INTERACTION_IDS = (2, 0, 6, 3)
COW_TAG = "xb_chain_cow"


@dataclass(frozen=True)
class CoalCowFurnaceScene:
    center_x: int
    feet_y: int
    center_z: int
    distance_band: str = "S0"
    start_yaw_offset_degrees: int = 90

    def __post_init__(self) -> None:
        if self.distance_band not in DISTANCE_BLOCKS:
            raise ValueError(f"unsupported distance band: {self.distance_band}")
        if int(self.start_yaw_offset_degrees) != 90:
            raise ValueError(
                f"unsupported hidden start yaw: {self.start_yaw_offset_degrees}")

    @property
    def origin(self) -> Cell:
        return self.center_x, self.feet_y, self.center_z

    @property
    def distance_blocks(self) -> int:
        return DISTANCE_BLOCKS[self.distance_band]

    @property
    def coal_ore_cell(self) -> Cell:
        return (self.center_x - self.distance_blocks // 2,
                self.feet_y + 1, self.center_z)

    @property
    def cow_cell(self) -> Cell:
        return (self.center_x + self.distance_blocks // 2,
                self.feet_y, self.center_z)

    @property
    def cow_position(self) -> tuple[float, float, float]:
        x, y, z = self.cow_cell
        return x + 0.5, float(y), z + 0.5

    @property
    def pen_fence_cells(self) -> tuple[Cell, ...]:
        x, y, z = self.cow_cell
        return tuple(
            (x + dx, y, z + dz)
            for dx in (-1, 0, 1) for dz in (-1, 0, 1)
            if (dx, dz) not in ((0, 0), (-1, 0)))

    @property
    def pen_gate_cell(self) -> Cell:
        x, y, z = self.cow_cell
        return x - 1, y, z

    @property
    def pen_boundary_cells(self) -> tuple[Cell, ...]:
        return tuple(sorted((*self.pen_fence_cells, self.pen_gate_cell)))

    @property
    def furnace_cell(self) -> Cell:
        return (self.center_x + self.distance_blocks // 2,
                self.feet_y, self.center_z - self.distance_blocks)

    @property
    def start_navigation_cell(self) -> Cell:
        x, _y, z = self.coal_ore_cell
        return x + 4, self.feet_y, z + 4

    @property
    def start_pose(self) -> tuple[float, float, float]:
        x, y, z = self.start_navigation_cell
        return x + 0.5, float(y), z + 0.5

    @property
    def target_cells(self) -> dict[str, tuple[Cell, ...]]:
        return {
            "coal_ore_mine": (self.coal_ore_cell,),
            "cow_hunt": (self.cow_cell,),
            "furnace_find": (self.furnace_cell,),
            "beef_cook": (self.furnace_cell,),
        }

    @property
    def stage_stance_cells(self) -> dict[str, tuple[Cell, ...]]:
        cx, _cy, cz = self.coal_ore_cell
        ox, _oy, oz = self.cow_cell
        fx, fy, fz = self.furnace_cell
        return {
            "coal_ore_mine": ((cx + 1, self.feet_y, cz),
                              (cx + 1, self.feet_y, cz + 1)),
            "cow_hunt": ((ox - 2, self.feet_y, oz),
                         (ox, self.feet_y, oz + 2),
                         (ox, self.feet_y, oz - 2)),
            "furnace_find": ((fx, fy, fz + 2), (fx - 2, fy, fz),
                             (fx + 2, fy, fz)),
            "beef_cook": ((fx, fy, fz + 1), (fx - 1, fy, fz),
                          (fx + 1, fy, fz)),
        }


def _set_box(blocks: dict[Cell, str], first: Cell, last: Cell,
             block: str) -> None:
    for x in range(first[0], last[0] + 1):
        for y in range(first[1], last[1] + 1):
            for z in range(first[2], last[2] + 1):
                blocks[(x, y, z)] = block


def build_scene_commands(scene: CoalCowFurnaceScene) -> SceneBuildResult:
    base_scene = SmeltingChainScene(
        *scene.origin, ore_kind="iron_ore", distance_band=scene.distance_band)
    base = build_smelting_scene_commands(base_scene)
    commands = list(base.commands)
    blocks = dict(base.final_blocks)
    cow_x, y, cow_z = scene.cow_cell

    clear_first = (cow_x - 2, y, cow_z - 4)
    clear_last = (cow_x + 4, y + 4, cow_z + 4)
    commands.extend((
        "/gamerule doMobLoot true",
        f"/fill {clear_first[0]} {clear_first[1]} {clear_first[2]} "
        f"{clear_last[0]} {clear_last[1]} {clear_last[2]} minecraft:air",
        f"/fill {clear_first[0]} {y-1} {clear_first[2]} "
        f"{clear_last[0]} {y-1} {clear_last[2]} minecraft:grass_block",
    ))
    _set_box(blocks, clear_first, clear_last, "air")
    _set_box(blocks, (clear_first[0], y - 1, clear_first[2]),
             (clear_last[0], y - 1, clear_last[2]), "grass_block")
    for cell in scene.pen_fence_cells:
        commands.append(
            f"/setblock {cell[0]} {cell[1]} {cell[2]} minecraft:oak_fence")
        blocks[cell] = "oak_fence"
    gate = scene.pen_gate_cell
    commands.append(
        f"/setblock {gate[0]} {gate[1]} {gate[2]} "
        "minecraft:oak_fence_gate[facing=east,open=false,powered=false]")
    blocks[gate] = "oak_fence_gate[facing=east,open=false,powered=false]"
    commands.extend((
        "/replaceitem entity @p hotbar.0 minecraft:stone_pickaxe 1",
        "/replaceitem entity @p hotbar.1 minecraft:iron_axe 1",
        f"/summon minecraft:cow {cow_x + .5:.1f} {y:.1f} {cow_z + .5:.1f} "
        "{Health:8.0f,NoAI:1b,Silent:1b,PersistenceRequired:1b,"
        "Attributes:[{Name:\"minecraft:generic.max_health\",Base:8.0d}],"
        f"Tags:[\"xb_coal_cow_chain\",\"{COW_TAG}\"]}}",
    ))

    bounds = base.manifest["reset_bounds"]
    navigation = {
        (x, y, z)
        for x in range(int(bounds["x_min"]) + 3, int(bounds["x_max"]) - 1)
        for z in range(int(bounds["z_min"]) + 3, int(bounds["z_max"]) - 1)
        if _base_kind(blocks.get((x, y, z), "air")) == "air"
        and _base_kind(blocks.get((x, y + 1, z), "air")) == "air"
        and _base_kind(blocks.get((x, y - 1, z), "air"))
        not in {"air", "water", "lava"}
    }
    paths: dict[str, tuple[Cell, ...]] = {}
    departure = scene.start_navigation_cell
    for stage in POLICY_STAGE_KEYS:
        path = _shortest_path(navigation, departure, scene.stage_stance_cells[stage])
        paths[stage] = path
        if path:
            departure = path[-1]

    eye = (scene.start_pose[0], scene.start_pose[1] + 1.62, scene.start_pose[2])
    tx, ty, tz = scene.coal_ore_cell
    coal_yaw = math.degrees(math.atan2(
        -(tx + .5 - eye[0]), tz + .5 - eye[2]))
    yaw = ((coal_yaw + float(scene.start_yaw_offset_degrees) + 180.0)
           % 360.0 - 180.0)
    pitch = -math.degrees(math.atan2(
        ty + .5 - eye[1], math.hypot(tx + .5 - eye[0], tz + .5 - eye[2])))
    target_supports = {
        key: [[cell[0], cell[1] - 1, cell[2]] for cell in cells]
        for key, cells in scene.target_cells.items()
    }
    object_columns = sorted({
        (x, z) for (x, yy, z), block in blocks.items()
        if yy >= y and _base_kind(block) != "air"})
    verified_stands = {scene.start_navigation_cell}
    for stands in scene.stage_stance_cells.values():
        verified_stands.update(stands)
    counts = Counter(_base_kind(block) for block in blocks.values())
    manifest = {
        "contract": SCENE_CONTRACT,
        "scene_name": "coal_cow_furnace_quarry",
        "layout_id": (
            "coal_cow_furnace_one_cell_pen_hidden_start_v2_"
            f"{scene.distance_band}_yaw{scene.start_yaw_offset_degrees}"),
        "world_seed": WORLD_SEED,
        "fixed_world_single_layout": True,
        "origin": list(scene.origin),
        "distance_band": scene.distance_band,
        "distance_blocks": scene.distance_blocks,
        "reset_bounds": dict(bounds),
        "start_pose": list(scene.start_pose),
        "start_navigation_cell": list(scene.start_navigation_cell),
        "start_yaw": yaw,
        "start_pitch": pitch,
        "start_camera_contract": "coal_behind_camera_all_targets_hidden/v1",
        "start_yaw_offset_degrees": int(scene.start_yaw_offset_degrees),
        "policy_stage_keys": list(POLICY_STAGE_KEYS),
        "target_kinds": ["coal_ore", "cow", "furnace", "furnace"],
        "success_quotas": [1, 1, 1, 1],
        "interaction_ids": list(INTERACTION_IDS),
        "target_cells": {key: [list(c) for c in cells]
                         for key, cells in scene.target_cells.items()},
        "stage_stance_cells": {key: [list(c) for c in cells]
                               for key, cells in scene.stage_stance_cells.items()},
        "target_support_cells": target_supports,
        "placed_object_supports": [[x, y - 1, z] for x, z in object_columns],
        "verified_stand_cells": [list(c) for c in sorted(verified_stands)],
        "navigation_cells": [list(c) for c in sorted(navigation)],
        "navigation_paths": {key: [list(c) for c in path]
                             for key, path in paths.items()},
        "navigation_distances_blocks": {
            "start_to_coal_target": math.dist(
                scene.start_pose[::2], (tx + .5, tz + .5)),
            "coal_to_cow_pen": float(scene.distance_blocks),
            "cow_pen_to_furnace": float(scene.distance_blocks),
        },
        "start_visibility_contract": {
            "coal_must_be_visible": True,
            "hidden_stage_keys": ["cow_hunt", "furnace_find", "beef_cook"],
            "live_exact_block_and_entity_masks_required": True,
        },
        "cow_pen": {
            "interior_dimensions_blocks": [1, 1],
            "interior_cell": list(scene.cow_cell),
            "fence_cells": [list(c) for c in scene.pen_fence_cells],
            "fence_gate_cell": list(scene.pen_gate_cell),
            "boundary_cells": [list(c) for c in scene.pen_boundary_cells],
            "fence_cell_count": 7,
            "closed_fence_gate_count": 1,
            "entity": "minecraft:cow",
            "unique_tag": COW_TAG,
            "spawn_position": list(scene.cow_position),
            "NoAI": True,
            "health": 8.0,
            "max_health": 8.0,
            "expected_scene_cows": 1,
            "containment": (
                "one interior block enclosed by seven oak fences and one closed "
                "oak fence gate; NoAI"),
        },
        "initial_inventory": {
            "clear_before_equip": True,
            "slots": {
                "hotbar.0": {"item": "minecraft:stone_pickaxe", "count": 1},
                "hotbar.1": {"item": "minecraft:iron_axe", "count": 1},
            },
            "selected_hotbar_slot": 1,
            "all_other_slots_empty": True,
        },
        "item_provenance": {
            "do_tile_drops": True,
            "do_mob_loot": True,
            "initial_kit_only": ["stone_pickaxe", "iron_axe"],
            "resource_injection_allowed": False,
            "required_natural_pickups": ["coal", "beef"],
        },
        "system_controls": {
            "hotbar_transition": "camera-off selection of pre-existing iron axe",
            "furnace_completion": (
                "unscored real GUI clicks and vanilla cooking; no command/NBT/item injection"),
        },
        "causal_narrative": {
            "coal_ore_mine": "Mine and naturally pick up coal.",
            "cow_hunt": "Find the sole penned cow, kill it, and naturally pick up raw beef.",
            "furnace_find": "Find the furnace and approach until it is visibly in range.",
            "beef_cook": "Use the exact furnace; a disclosed system GUI macro completes vanilla cooking.",
            "scored_endpoint": "policy exact furnace USE and GUI-open transition",
        },
        "staging": {
            "camera_off": True,
            "mobs_purged_before_unique_cow_summon": True,
            "dropped_items_purged_before_policy": True,
            "per_column_grounding": "authored ground; live voxel probes required",
            "teleport_stand_probes_required": True,
        },
        "final_block_counts": dict(sorted(counts.items())),
    }
    result = SceneBuildResult(tuple(commands), manifest, blocks)
    validate_scene_manifest(result, raise_on_error=True)
    return result


def validate_scene_manifest(result: SceneBuildResult, *,
                            raise_on_error: bool = False) -> dict:
    manifest, blocks = result.manifest, result.final_blocks
    issues: list[str] = []
    if manifest.get("contract") != SCENE_CONTRACT:
        issues.append("scene contract mismatch")
    if tuple(manifest.get("policy_stage_keys", ())) != POLICY_STAGE_KEYS:
        issues.append("stage sequence mismatch")
    pen = manifest.get("cow_pen", {})
    fences = tuple(tuple(c) for c in pen.get("fence_cells", ()))
    gate = tuple(pen.get("fence_gate_cell", ()))
    interior = tuple(pen.get("interior_cell", ()))
    if (pen.get("interior_dimensions_blocks") != [1, 1] or len(fences) != 7
            or len(gate) != 3):
        issues.append("cow pen is not one interior block with a closed eight-cell boundary")
    if len(interior) != 3 or _base_kind(blocks.get(interior, "air")) != "air":
        issues.append("cow interior cell is not air")
    if any(_base_kind(blocks.get(cell, "air")) != "oak_fence" for cell in fences):
        issues.append("cow pen fence identity mismatch")
    if _base_kind(blocks.get(gate, "air")) != "oak_fence_gate":
        issues.append("cow pen gate identity mismatch")
    for key, path in manifest.get("navigation_paths", {}).items():
        if not path:
            issues.append(f"unreachable stage stance: {key}")
    for key, cells in manifest.get("target_support_cells", {}).items():
        for cell in cells:
            if _base_kind(blocks.get(tuple(cell), "air")) in {
                    "air", "water", "lava"}:
                issues.append(f"unsupported target column: {key}:{cell}")
    report = {"passed": not issues, "issues": issues,
              "contract": "coal_cow_furnace_scene_pure_audit/v1"}
    if raise_on_error and issues:
        raise ValueError("; ".join(issues))
    return report
