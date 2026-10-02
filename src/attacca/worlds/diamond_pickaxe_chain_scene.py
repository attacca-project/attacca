from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping


Cell = tuple[int, int, int]
SCENE_CONTRACT = "xbench_diamond_pickaxe_chain_scene/v1"
POLICY_STAGE_KEYS = (
    "oak_log_mine",
    "diamond_ore_mine3",
    "crafting_table_open",
)
INTERACTION_IDS = (2, 2, 3)


def _base_kind(value: object) -> str:
    return str(value).removeprefix("minecraft:").split("[", 1)[0]


def _neighbors6(cell: Cell) -> tuple[Cell, ...]:
    x, y, z = cell
    return (
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1),
    )


def _neighbors4(cell: Cell) -> tuple[Cell, ...]:
    x, y, z = cell
    return (
        (x + 1, y, z), (x - 1, y, z),
        (x, y, z + 1), (x, y, z - 1),
    )


def _is_connected(cells: Iterable[Cell], *, horizontal: bool = False) -> bool:
    remaining = set(cells)
    if not remaining:
        return False
    seen = {next(iter(remaining))}
    frontier = list(seen)
    while frontier:
        cell = frontier.pop()
        neighbors = _neighbors4(cell) if horizontal else _neighbors6(cell)
        for neighbor in neighbors:
            if neighbor in remaining and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return seen == remaining


def _shortest_path(
        allowed: set[Cell], start: Cell, goals: Iterable[Cell]
) -> tuple[Cell, ...]:
    goals = set(goals)
    if start not in allowed or not goals:
        return ()
    queue = deque([start])
    parent: dict[Cell, Cell | None] = {start: None}
    end = None
    while queue:
        current = queue.popleft()
        if current in goals:
            end = current
            break
        for neighbor in _neighbors4(current):
            if neighbor in allowed and neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)
    if end is None:
        return ()
    path = []
    current: Cell | None = end
    while current is not None:
        path.append(current)
        current = parent[current]
    return tuple(reversed(path))


def _grid_line_xz(start: Cell, end: Cell) -> tuple[Cell, ...]:
    x0, y, z0 = start
    x1, _unused_y, z1 = end
    dx, dz = abs(x1 - x0), abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1
    error = dx - dz
    cells = []
    while True:
        cells.append((x0, y, z0))
        if x0 == x1 and z0 == z1:
            return tuple(cells)
        twice = 2 * error
        if twice > -dz:
            error -= dz
            x0 += sx
        if twice < dx:
            error += dx
            z0 += sz


@dataclass(frozen=True)
class DiamondPickaxeChainScene:

    center_x: int
    feet_y: int
    center_z: int

    @property
    def origin(self) -> Cell:
        return self.center_x, self.feet_y, self.center_z

    @property
    def start_pose(self) -> tuple[float, float, float]:
        return self.center_x + 1.5, float(self.feet_y), self.center_z + 4.5

    @property
    def start_navigation_cell(self) -> Cell:
        return self.center_x + 1, self.feet_y, self.center_z + 4

    @property
    def oak_log_cells(self) -> tuple[Cell, ...]:
        return tuple(
            (self.center_x - 4, self.feet_y + dy, self.center_z + 3)
            for dy in range(3))

    @property
    def oak_target_cells(self) -> tuple[Cell, ...]:
        return self.oak_log_cells[:2]

    @property
    def oak_mine_stance_cells(self) -> tuple[Cell, ...]:
        x, y, z = self.oak_log_cells[0]
        return ((x, y, z + 1), (x - 1, y, z), (x + 1, y, z))

    @property
    def diamond_ore_cells(self) -> tuple[Cell, ...]:
        return (
            (self.center_x + 3, self.feet_y + 1, self.center_z - 3),
            (self.center_x + 3, self.feet_y + 1, self.center_z - 4),
            (self.center_x + 3, self.feet_y + 2, self.center_z - 4),
        )

    @property
    def diamond_mine_stance_cells(self) -> tuple[Cell, ...]:
        return (
            (self.center_x + 2, self.feet_y, self.center_z - 3),
            (self.center_x + 2, self.feet_y, self.center_z - 4),
        )

    @property
    def crafting_table_cell(self) -> Cell:
        return self.center_x + 6, self.feet_y, self.center_z - 8

    @property
    def crafting_table_stance_cells(self) -> tuple[Cell, ...]:
        x, y, z = self.crafting_table_cell
        return ((x - 1, y, z), (x, y, z + 1))

    @property
    def planned_route_cells(self) -> tuple[Cell, ...]:
        cx, y, cz = self.origin
        route: list[Cell] = []

        def add(cell: Cell) -> None:
            if not route or route[-1] != cell:
                route.append(cell)

        for x in range(cx + 1, cx - 5, -1):
            add((x, y, cz + 4))
        add((cx - 3, y, cz + 4))
        add((cx - 3, y, cz + 3))
        add((cx - 3, y, cz + 2))
        for x in range(cx - 2, cx + 3):
            add((x, y, cz + 2))
        for z in range(cz + 1, cz - 9, -1):
            add((cx + 2, y, z))
        for x in range(cx + 3, cx + 6):
            add((x, y, cz - 8))
        return tuple(route)


@dataclass(frozen=True)
class SceneBuildResult:
    commands: tuple[str, ...]
    manifest: dict
    final_blocks: Mapping[Cell, str] = field(repr=False)


class _CommandBuilder:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.blocks: dict[Cell, str] = {}

    def command(self, value: str) -> None:
        self.commands.append(value)

    def fill(self, first: Cell, last: Cell, block: str) -> None:
        x0, y0, z0 = first
        x1, y1, z1 = last
        lo = (min(x0, x1), min(y0, y1), min(z0, z1))
        hi = (max(x0, x1), max(y0, y1), max(z0, z1))
        value = block if block.startswith("minecraft:") else f"minecraft:{block}"
        self.commands.append(
            f"/fill {lo[0]} {lo[1]} {lo[2]} {hi[0]} {hi[1]} {hi[2]} "
            f"{value}")
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    self.blocks[(x, y, z)] = value

    def setblock(self, cell: Cell, block: str) -> None:
        value = block if block.startswith("minecraft:") else f"minecraft:{block}"
        self.commands.append(
            f"/setblock {cell[0]} {cell[1]} {cell[2]} {value}")
        self.blocks[cell] = value


def _leaf_cells(scene: DiamondPickaxeChainScene) -> tuple[Cell, ...]:
    tx, y, tz = scene.oak_log_cells[-1]
    leaves = set()
    for dy, radius, max_manhattan in ((0, 2, 3), (1, 2, 3), (2, 1, 2)):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if abs(dx) + abs(dz) <= max_manhattan:
                    leaves.add((tx + dx, y + dy, tz + dz))
    leaves.difference_update(scene.oak_log_cells)
    return tuple(sorted(leaves))


def _distance_xz(first: Cell, second: Cell) -> float:
    return math.hypot(first[0] - second[0], first[2] - second[2])


def build_scene_commands(scene: DiamondPickaxeChainScene) -> SceneBuildResult:
    cx, y, cz = scene.origin
    builder = _CommandBuilder()
    builder.command("/gamerule doMobSpawning false")
    builder.command("/gamerule doDaylightCycle false")
    builder.command("/gamerule doWeatherCycle false")
    builder.command("/gamerule doTileDrops true")
    builder.command("/gamerule fallDamage false")
    builder.command("/gamerule showDeathMessages false")
    builder.command("/gamerule sendCommandFeedback false")
    builder.command("/gamerule announceAdvancements false")
    builder.command("/time set day")
    builder.command("/weather clear")

    reset_bounds = {
        "x_min": cx - 17, "x_max": cx + 17,
        "y_min": y - 4, "y_max": y + 7,
        "z_min": cz - 17, "z_max": cz + 17,
    }
    builder.fill(
        (reset_bounds["x_min"], reset_bounds["y_min"], reset_bounds["z_min"]),
        (reset_bounds["x_max"], reset_bounds["y_max"], reset_bounds["z_max"]),
        "air")

    builder.fill((cx - 16, y - 4, cz - 16),
                 (cx + 16, y - 3, cz + 16), "stone")
    builder.fill((cx - 16, y - 2, cz - 16),
                 (cx + 16, y - 1, cz + 16), "dirt")
    builder.fill((cx - 16, y - 1, cz - 16),
                 (cx + 16, y - 1, cz + 16), "grass_block")

    hill_boxes = (
        ((cx - 16, y, cz - 16), (cx + 16, y + 2, cz - 14)),
        ((cx - 16, y, cz + 14), (cx + 16, y + 2, cz + 16)),
        ((cx - 16, y, cz - 13), (cx - 14, y + 2, cz + 13)),
        ((cx + 14, y, cz - 13), (cx + 16, y + 2, cz + 13)),
        ((cx - 13, y, cz - 13), (cx - 12, y + 1, cz - 8)),
        ((cx - 13, y, cz + 7), (cx - 11, y, cz + 12)),
        ((cx + 7, y, cz + 11), (cx + 13, y, cz + 13)),
    )
    for first, last in hill_boxes:
        builder.fill(first, last, "dirt")
        builder.fill((first[0], last[1] + 1, first[2]),
                     (last[0], last[1] + 1, last[2]), "grass_block")

    outcrop_boxes = (
        ((cx + 3, y, cz - 6), (cx + 9, y + 3, cz + 1)),
        ((cx + 5, y + 4, cz - 5), (cx + 9, y + 4, cz)),
        ((cx + 7, y + 5, cz - 3), (cx + 9, y + 5, cz - 1)),
        ((cx + 9, y, cz - 6), (cx + 14, y + 2, cz - 2)),
    )
    for first, last in outcrop_boxes:
        builder.fill(first, last, "stone")
    cave_mouth = ((cx + 3, y, cz), (cx + 5, y + 2, cz + 1))
    builder.fill(*cave_mouth, "air")
    builder.fill((cx + 3, y - 1, cz), (cx + 5, y - 1, cz + 1), "stone")

    oak_leaves = _leaf_cells(scene)
    gravel_patch = (
        (cx, y - 1, cz - 6),
        (cx + 1, y - 1, cz - 6),
        (cx + 2, y - 1, cz - 6),
        (cx + 1, y - 1, cz - 7),
        (cx + 2, y - 1, cz - 7),
    )
    workspot_floor = (
        (cx + 5, y - 1, cz - 9),
        (cx + 5, y - 1, cz - 8),
        (cx + 6, y - 1, cz - 9),
        (cx + 6, y - 1, cz - 7),
        (cx + 7, y - 1, cz - 9),
        (cx + 7, y - 1, cz - 8),
        (cx + 7, y - 1, cz - 7),
    )
    workspot_objects: dict[Cell, str] = {
        (cx + 8, y, cz - 8): "minecraft:cobblestone",
        (cx + 8, y, cz - 9): "minecraft:cobblestone",
        (cx + 8, y + 1, cz - 8): "minecraft:torch",
        (cx + 7, y, cz - 9): "minecraft:stripped_oak_log[axis=x]",
    }
    decorative: dict[Cell, str] = {
        **{cell: "minecraft:oak_leaves[persistent=true]"
           for cell in oak_leaves},
        **{cell: "minecraft:cobblestone" for cell in workspot_floor},
        **workspot_objects,
    }
    for cell, block in sorted(decorative.items()):
        builder.setblock(cell, block)

    coal_cells = (
        (cx + 3, y + 1, cz - 2),
        (cx + 3, y + 2, cz - 2),
    )
    iron_cells = (
        (cx + 3, y + 1, cz - 5),
        (cx + 3, y + 2, cz - 5),
    )
    andesite_cells = (
        (cx + 3, y, cz - 1),
        (cx + 3, y + 1, cz - 1),
        (cx + 3, y + 2, cz - 1),
    )
    for cell in gravel_patch:
        builder.setblock(cell, "gravel")
    for cell in coal_cells:
        builder.setblock(cell, "coal_ore")
    for cell in iron_cells:
        builder.setblock(cell, "iron_ore")
    for cell in andesite_cells:
        builder.setblock(cell, "andesite")

    for cell in scene.oak_log_cells:
        builder.setblock(cell, "oak_log[axis=y]")
    for cell in scene.diamond_ore_cells:
        builder.setblock(cell, "diamond_ore")
    builder.setblock(scene.crafting_table_cell, "crafting_table")

    builder.command("/kill @e[type=!minecraft:player]")
    builder.command("/kill @e[type=minecraft:item]")
    builder.command("/clear @a")
    builder.command("/effect clear @a")
    builder.command(
        "/replaceitem entity @p hotbar.0 minecraft:iron_pickaxe 1")

    blocks = builder.blocks
    interior_x = range(cx - 13, cx + 14)
    interior_z = range(cz - 13, cz + 14)
    candidates = {
        (x, y, z)
        for x in interior_x for z in interior_z
        if _base_kind(blocks.get((x, y, z), "air")) == "air"
        and _base_kind(blocks.get((x, y + 1, z), "air")) == "air"
        and _base_kind(blocks.get((x, y - 1, z), "air")) != "air"
    }
    start = scene.start_navigation_cell
    navigation = {start} if start in candidates else set()
    frontier = list(navigation)
    while frontier:
        current = frontier.pop()
        for neighbor in _neighbors4(current):
            if neighbor in candidates and neighbor not in navigation:
                navigation.add(neighbor)
                frontier.append(neighbor)

    oak_path = _shortest_path(
        navigation, start, scene.oak_mine_stance_cells)
    oak_departure = oak_path[-1] if oak_path else scene.oak_mine_stance_cells[0]
    diamond_path = _shortest_path(
        navigation, oak_departure, scene.diamond_mine_stance_cells)
    diamond_departure = (
        diamond_path[-1] if diamond_path else scene.diamond_mine_stance_cells[0])
    table_path = _shortest_path(
        navigation, diamond_departure, scene.crafting_table_stance_cells)

    outcrop_region = {
        (x, yy, z)
        for x in range(cx + 3, cx + 15)
        for yy in range(y, y + 6)
        for z in range(cz - 6, cz + 2)
    }
    exposed_stone = {
        cell for cell in outcrop_region
        if _base_kind(blocks.get(cell, "air")) == "stone"
        and any(_base_kind(blocks.get(neighbor, "air")) == "air"
                for neighbor in _neighbors6(cell))
    }
    confuser_cells = {
        "coal_ore": tuple(coal_cells),
        "iron_ore": tuple(iron_cells),
        "andesite": tuple(andesite_cells),
        "gravel": tuple(gravel_patch),
        "stone_context": tuple(sorted(exposed_stone)),
    }
    all_confusers = set().union(*map(set, confuser_cells.values()))

    escape_barrier = {
        (x, yy, z)
        for yy in (y, y + 1)
        for x in range(cx - 15, cx + 16)
        for z in range(cz - 15, cz + 16)
        if x in (cx - 15, cx + 15) or z in (cz - 15, cz + 15)
    }
    all_targets = (
        set(scene.oak_target_cells) | set(scene.diamond_ore_cells)
        | {scene.crafting_table_cell})
    reserved = set(all_targets) | all_confusers | escape_barrier
    for cell in navigation:
        reserved.add(cell)
        reserved.add((cell[0], cell[1] + 1, cell[2]))

    decoration_cells = set(decorative)
    decoration_collision = decoration_cells & reserved
    if decoration_collision:
        raise RuntimeError(
            "fixed decoration intersects reserved cells: "
            f"{sorted(decoration_collision)}")

    table_line = _grid_line_xz(
        scene.diamond_mine_stance_cells[-1], scene.crafting_table_cell)
    table_occluders = tuple(
        cell for cell in table_line[1:-1]
        if (_base_kind(blocks.get(cell, "air")) != "air"
            or _base_kind(blocks.get(
                (cell[0], cell[1] + 1, cell[2]), "air")) != "air"))

    substrate_cells = {
        (x, yy, z)
        for x in range(cx - 16, cx + 17)
        for yy in (y - 4, y - 3)
        for z in range(cz - 16, cz + 17)
        if _base_kind(blocks.get((x, yy, z), "air")) == "stone"
    }
    hill_natural_cells = {
        cell for cell, value in blocks.items()
        if cell[1] >= y
        and cx - 16 <= cell[0] <= cx + 16
        and cz - 16 <= cell[2] <= cz + 16
        and _base_kind(value) in {"dirt", "grass_block"}
    }
    outcrop_solid_cells = {
        cell for cell in outcrop_region
        if _base_kind(blocks.get(cell, "air")) != "air"
    }
    outcrop_touches_east_boundary = any(
        cell[0] == cx + 14 for cell in outcrop_solid_cells)

    counts = Counter(_base_kind(value) for value in blocks.values())
    manifest = {
        "contract": SCENE_CONTRACT,
        "scene_name": "natural_diamond_pickaxe_chain",
        "layout_id": "forest_clearing_rock_bend_workspot_v1",
        "fixed_world_single_layout": True,
        "origin": list(scene.origin),
        "reset_bounds": reset_bounds,
        "start_pose": list(scene.start_pose),
        "start_yaw": -164.0,
        "start_pitch": 4.0,
        "start_navigation_cell": list(start),
        "policy_stage_keys": list(POLICY_STAGE_KEYS),
        "interaction_ids": list(INTERACTION_IDS),
        "success_quotas": [1, 3, 1],
        "target_cells": {
            "oak_log_mine": [list(cell) for cell in scene.oak_target_cells],
            "diamond_ore_mine3": [
                list(cell) for cell in scene.diamond_ore_cells],
            "crafting_table_open": [list(scene.crafting_table_cell)],
        },
        "oak_tree": {
            "all_log_cells_bottom_to_top": [
                list(cell) for cell in scene.oak_log_cells],
            "policy_target_cells": [
                list(cell) for cell in scene.oak_target_cells],
            "mine_stance_cells": [
                list(cell) for cell in scene.oak_mine_stance_cells],
            "log_blockstate": "minecraft:oak_log[axis=y]",
            "canopy_cells": [list(cell) for cell in oak_leaves],
            "canopy_blockstate": "minecraft:oak_leaves[persistent=true]",
            "leaf_decay_confound_disabled": True,
        },
        "diamond_vein": {
            "cells": [list(cell) for cell in scene.diamond_ore_cells],
            "mine_stance_cells": [
                list(cell) for cell in scene.diamond_mine_stance_cells],
            "connected_6_neighbor": _is_connected(scene.diamond_ore_cells),
            "exposed_face_direction": "-X",
            "exact_scene_count": counts["diamond_ore"],
        },
        "crafting_table": {
            "cell": list(scene.crafting_table_cell),
            "use_stance_cells": [
                list(cell) for cell in scene.crafting_table_stance_cells],
            "exact_scene_count": counts["crafting_table"],
            "diamond_stance_line_cells": [list(cell) for cell in table_line],
            "terrain_occluder_cells": [list(cell) for cell in table_occluders],
            "hidden_behind_outcrop_bend": bool(table_occluders),
        },
        "confuser_cells": {
            name: [list(cell) for cell in cells]
            for name, cells in confuser_cells.items()
        },
        "all_confuser_cells": [list(cell) for cell in sorted(all_confusers)],
        "reserved_cells": [list(cell) for cell in sorted(reserved)],
        "navigation_cells": [list(cell) for cell in sorted(navigation)],
        "planned_route_cells": [
            list(cell) for cell in scene.planned_route_cells],
        "navigation_paths": {
            "start_to_oak": [list(cell) for cell in oak_path],
            "oak_to_diamond": [list(cell) for cell in diamond_path],
            "diamond_to_table": [list(cell) for cell in table_path],
        },
        "navigation_distances_blocks": {
            "start_to_oak_target": _distance_xz(
                scene.start_navigation_cell, scene.oak_target_cells[0]),
            "oak_to_diamond_target": _distance_xz(
                scene.oak_target_cells[0], scene.diamond_ore_cells[1]),
            "diamond_to_table_target": _distance_xz(
                scene.diamond_ore_cells[1], scene.crafting_table_cell),
        },
        "escape_barrier_cells": [
            list(cell) for cell in sorted(escape_barrier)],
        "decoration_cells": [list(cell) for cell in sorted(decoration_cells)],
        "decoration_reserved_collision_count": 0,
        "natural_context": {
            "connected_stone_substrate": _is_connected(substrate_cells),
            "connected_grass_dirt_boundary": _is_connected(
                hill_natural_cells),
            "connected_stone_outcrop": _is_connected(outcrop_solid_cells),
            "outcrop_connected_to_east_boundary": (
                outcrop_touches_east_boundary),
            "cave_mouth_bounds": [list(cave_mouth[0]), list(cave_mouth[1])],
            "workspot_floor_cells": [
                list(cell) for cell in workspot_floor],
            "interactive_decorations": [],
        },
        "start_visibility_contract": {
            "camera_faces": "mostly_-Z_slightly_-X",
            "oak_is_behind_camera": True,
            "oak_turn_degrees": abs((
                math.degrees(math.atan2(
                    -(scene.oak_target_cells[0][0] + 0.5
                      - scene.start_pose[0]),
                    scene.oak_target_cells[0][2] + 0.5
                    - scene.start_pose[2]))
                - (-164.0) + 180.0) % 360.0 - 180.0),
        },
        "final_block_counts": dict(sorted(counts.items())),
        "initial_inventory": {
            "clear_before_equip": True,
            "slots": {
                "hotbar.0": {
                    "item": "minecraft:iron_pickaxe",
                    "count": 1,
                    "enchanted": False,
                },
            },
            "selected_hotbar_slot": 1,
            "selection_control": "camera_off_hotbar.1",
            "held_item_after_selection": "minecraft:iron_pickaxe",
            "all_other_slots_empty": True,
        },
        "item_provenance": {
            "do_tile_drops": True,
            "intermediate_item_injection_allowed": False,
            "injected_materials": [],
            "crafted_or_mined_only": [
                "minecraft:oak_log", "minecraft:oak_planks",
                "minecraft:stick", "minecraft:diamond",
                "minecraft:diamond_pickaxe",
            ],
        },
        "staging": {
            "camera_off": True,
            "mobs_purged_after_build": True,
            "dropped_items_purged_after_build": True,
            "weather": "clear_locked",
            "daylight": "day_locked",
        },
    }
    result = SceneBuildResult(
        commands=tuple(builder.commands), manifest=manifest,
        final_blocks=dict(blocks))
    validate_scene_manifest(result, raise_on_error=True)
    return result


def validate_scene_manifest(
        result: SceneBuildResult, *, raise_on_error: bool = False
) -> dict:
    manifest = result.manifest
    blocks = result.final_blocks
    issues: list[str] = []

    def cells(path: object) -> set[Cell]:
        return {tuple(int(value) for value in row) for row in path or ()}

    targets = manifest.get("target_cells") or {}
    oak_targets = cells(targets.get("oak_log_mine"))
    diamond_targets = cells(targets.get("diamond_ore_mine3"))
    table_targets = cells(targets.get("crafting_table_open"))
    decorations = cells(manifest.get("decoration_cells"))
    reserved = cells(manifest.get("reserved_cells"))
    navigation = cells(manifest.get("navigation_cells"))
    barrier = cells(manifest.get("escape_barrier_cells"))
    planned_route = cells(manifest.get("planned_route_cells"))

    counts = Counter(_base_kind(value) for value in blocks.values())
    if manifest.get("contract") != SCENE_CONTRACT:
        issues.append("scene contract mismatch")
    if tuple(manifest.get("policy_stage_keys") or ()) != POLICY_STAGE_KEYS:
        issues.append("policy stage sequence mismatch")
    if tuple(manifest.get("interaction_ids") or ()) != INTERACTION_IDS:
        issues.append("interaction ID sequence mismatch")
    if tuple(manifest.get("success_quotas") or ()) != (1, 3, 1):
        issues.append("success quota sequence mismatch")

    if len(oak_targets) != 2 or any(
            _base_kind(blocks.get(cell, "air")) != "oak_log"
            for cell in oak_targets):
        issues.append("Oak target set must contain two lower live Oak Logs")
    all_logs = cells((manifest.get("oak_tree") or {}).get(
        "all_log_cells_bottom_to_top"))
    if len(all_logs) != 3 or counts["oak_log"] != 3:
        issues.append("scene must contain exactly one three-block Oak trunk")
    canopy = cells((manifest.get("oak_tree") or {}).get("canopy_cells"))
    if not canopy or any(
            blocks.get(cell) != "minecraft:oak_leaves[persistent=true]"
            for cell in canopy):
        issues.append("every canopy cell must be persistent Oak Leaves")

    if len(diamond_targets) != 3 or counts["diamond_ore"] != 3:
        issues.append("scene must contain exactly three target Diamond Ores")
    if not _is_connected(diamond_targets):
        issues.append("Diamond vein is not 6-neighbor connected")
    for cell in diamond_targets:
        west = (cell[0] - 1, cell[1], cell[2])
        if _base_kind(blocks.get(west, "air")) != "air":
            issues.append(f"Diamond Ore lacks exposed west face: {cell}")

    if len(table_targets) != 1 or counts["crafting_table"] != 1:
        issues.append("scene must contain exactly one Crafting Table")
    for forbidden in ("chest", "furnace", "barrel"):
        if counts[forbidden]:
            issues.append(f"interactive confuser present: {forbidden}")
    table_info = manifest.get("crafting_table") or {}
    if not table_info.get("hidden_behind_outcrop_bend"):
        issues.append("Crafting Table lacks the required terrain bend")
    if not cells(table_info.get("terrain_occluder_cells")):
        issues.append("Crafting Table occlusion proof is empty")

    confusers = manifest.get("confuser_cells") or {}
    required_confusers = {
        "coal_ore": 2, "iron_ore": 2, "andesite": 3, "gravel": 5}
    for kind, count in required_confusers.items():
        declared = cells(confusers.get(kind))
        if len(declared) != count or any(
                _base_kind(blocks.get(cell, "air")) != kind
                for cell in declared):
            issues.append(f"{kind} confuser patch mismatch")
        if not _is_connected(declared):
            issues.append(f"{kind} confuser patch is disconnected")
    if not cells(confusers.get("stone_context")):
        issues.append("exposed connected Stone context is missing")

    try:
        cx, feet_y, cz = (
            int(value) for value in manifest.get("origin") or ())
    except (TypeError, ValueError):
        cx = feet_y = cz = 0
        issues.append("scene origin is missing or malformed")
    substrate_cells = {
        (x, yy, z)
        for x in range(cx - 16, cx + 17)
        for yy in (feet_y - 4, feet_y - 3)
        for z in range(cz - 16, cz + 17)
        if _base_kind(blocks.get((x, yy, z), "air")) == "stone"
    }
    hill_natural_cells = {
        cell for cell, value in blocks.items()
        if cell[1] >= feet_y
        and cx - 16 <= cell[0] <= cx + 16
        and cz - 16 <= cell[2] <= cz + 16
        and _base_kind(value) in {"dirt", "grass_block"}
    }
    outcrop_solid_cells = {
        (x, yy, z)
        for x in range(cx + 3, cx + 15)
        for yy in range(feet_y, feet_y + 6)
        for z in range(cz - 6, cz + 2)
        if _base_kind(blocks.get((x, yy, z), "air")) != "air"
    }
    natural = manifest.get("natural_context") or {}
    natural_checks = {
        "connected_stone_substrate": _is_connected(substrate_cells),
        "connected_grass_dirt_boundary": _is_connected(hill_natural_cells),
        "connected_stone_outcrop": _is_connected(outcrop_solid_cells),
        "outcrop_connected_to_east_boundary": any(
            cell[0] == cx + 14 for cell in outcrop_solid_cells),
    }
    for name, observed in natural_checks.items():
        if not observed or natural.get(name) is not observed:
            issues.append(f"natural connected-mass audit failed: {name}")

    collision = decorations & reserved
    if collision:
        issues.append(
            f"decoration intersects reserved cells: {sorted(collision)}")
    all_targets = oak_targets | diamond_targets | table_targets
    all_confusers = cells(manifest.get("all_confuser_cells"))
    if not all_targets <= reserved:
        issues.append("target cells are not all reserved")
    if not all_confusers <= reserved:
        issues.append("confuser cells are not all reserved")

    if not navigation or not _is_connected(navigation, horizontal=True):
        issues.append("navigation component is empty or disconnected")
    if not planned_route <= navigation:
        issues.append("protected planned route left the navigation component")
    for cell in navigation:
        below = (cell[0], cell[1] - 1, cell[2])
        head = (cell[0], cell[1] + 1, cell[2])
        if (_base_kind(blocks.get(below, "air")) == "air"
                or _base_kind(blocks.get(cell, "air")) != "air"
                or _base_kind(blocks.get(head, "air")) != "air"):
            issues.append(f"navigation cell lacks floor/clearance: {cell}")
            break
        if cell not in reserved or head not in reserved:
            issues.append(f"navigation body/head not reserved: {cell}")
            break

    paths = manifest.get("navigation_paths") or {}
    for name in ("start_to_oak", "oak_to_diamond", "diamond_to_table"):
        path = [tuple(int(value) for value in row)
                for row in paths.get(name) or ()]
        if not path or any(cell not in navigation for cell in path):
            issues.append(f"missing or invalid navigation path: {name}")
            continue
        if any(second not in _neighbors4(first)
               for first, second in zip(path, path[1:])):
            issues.append(f"non-contiguous navigation path: {name}")

    stance_groups = (
        cells((manifest.get("oak_tree") or {}).get("mine_stance_cells")),
        cells((manifest.get("diamond_vein") or {}).get("mine_stance_cells")),
        cells(table_info.get("use_stance_cells")),
    )
    if any(not (stances & navigation) for stances in stance_groups):
        issues.append("one or more task targets has no reachable stance")
    for label, target_group, stances in zip(
            ("oak", "diamond", "crafting_table"),
            (oak_targets, diamond_targets, table_targets), stance_groups):
        reachable_stances = stances & navigation
        for target in target_group:
            within_reach = any(math.dist(
                (stance[0] + 0.5, stance[1] + 1.62, stance[2] + 0.5),
                (target[0] + 0.5, target[1] + 0.5, target[2] + 0.5),
            ) <= 4.45 for stance in reachable_stances)
            if not within_reach:
                issues.append(f"{label} target has no in-reach stance: {target}")

    if not barrier or any(
            _base_kind(blocks.get(cell, "air")) == "air" for cell in barrier):
        issues.append("natural boundary does not seal the navigation arena")

    inventory = manifest.get("initial_inventory") or {}
    slots = inventory.get("slots") or {}
    expected_slots = {
        "hotbar.0": {
            "item": "minecraft:iron_pickaxe", "count": 1,
            "enchanted": False}}
    if (slots != expected_slots or not inventory.get("all_other_slots_empty")
            or inventory.get("selected_hotbar_slot") != 1):
        issues.append("initial inventory is not exactly one selected Iron Pickaxe")
    replace_commands = [
        command for command in result.commands
        if command.startswith("/replaceitem") or command.startswith("/give")]
    if replace_commands != [
            "/replaceitem entity @p hotbar.0 minecraft:iron_pickaxe 1"]:
        issues.append("material/tool injection command contract mismatch")
    if "/gamerule doTileDrops true" not in result.commands:
        issues.append("doTileDrops=true is missing")
    if "/gamerule doTileDrops false" in result.commands:
        issues.append("doTileDrops was disabled")
    for command in (
            "/kill @e[type=!minecraft:player]",
            "/kill @e[type=minecraft:item]", "/clear @a"):
        if command not in result.commands:
            issues.append(f"staging cleanup command missing: {command}")

    audit = {
        "passed": not issues,
        "issues": issues,
        "target_counts": {
            "oak_policy_targets": len(oak_targets),
            "oak_logs_in_scene": counts["oak_log"],
            "diamond_ore": counts["diamond_ore"],
            "crafting_table": counts["crafting_table"],
        },
        "navigation_cell_count": len(navigation),
        "reserved_cell_count": len(reserved),
        "decoration_collision_count": len(collision),
        "diamond_connected": _is_connected(diamond_targets),
        "boundary_sealed": bool(barrier) and all(
            _base_kind(blocks.get(cell, "air")) != "air" for cell in barrier),
    }
    if issues and raise_on_error:
        raise ValueError("invalid Diamond Pickaxe scene: " + "; ".join(issues))
    return audit
