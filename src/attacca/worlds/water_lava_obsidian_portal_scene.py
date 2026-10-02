from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from attacca.worlds.diamond_pickaxe_chain_scene import Cell
from attacca.worlds.diamond_pickaxe_chain_scene import SceneBuildResult
from attacca.worlds.diamond_pickaxe_chain_scene import _CommandBuilder
from attacca.worlds.diamond_pickaxe_chain_scene import _base_kind
from attacca.worlds.diamond_pickaxe_chain_scene import _shortest_path


SCENE_CONTRACT = "xbench_water_lava_obsidian_portal_exploration_scene/v9"
WORLD_SEED = 1_834_000
PORTAL_PLACE_MARKER_KIND = "sea_lantern"
PORTAL_PLACE_MARKER_KINDS = ("sea_lantern",)
POLICY_STAGE_KEYS = (
    "water_scoop", "lava_pour", "obsidian_mine", "portal_place",
    "portal_ignite")
INTERACTION_IDS = (3, 3, 2, 3, 3)


@dataclass(frozen=True)
class WaterLavaObsidianPortalScene:

    center_x: int
    feet_y: int
    center_z: int
    portal_place_marker_kind: str = PORTAL_PLACE_MARKER_KIND

    @property
    def origin(self) -> Cell:
        return self.center_x, self.feet_y, self.center_z

    @property
    def reset_bounds(self) -> dict[str, int]:
        return {
            "x_min": self.center_x - 22,
            "x_max": self.center_x + 24,
            "y_min": self.feet_y - 4,
            "y_max": self.feet_y + 18,
            "z_min": self.center_z - 32,
            "z_max": self.center_z + 14,
        }

    @property
    def start_navigation_cell(self) -> Cell:
        return self.center_x - 6, self.feet_y, self.center_z + 5

    @property
    def start_pose(self) -> tuple[float, float, float]:
        x, y, z = self.start_navigation_cell
        return x + 0.5, float(y), z + 0.5

    @property
    def start_yaw(self) -> float:
        return 0.0

    @property
    def start_pitch(self) -> float:
        return 4.0

    @property
    def water_cells(self) -> tuple[Cell, ...]:
        return tuple(
            (self.center_x + dx, self.feet_y - 1, self.center_z + dz)
            for dz in (0, 1) for dx in (-11, -10))

    @property
    def lava_cells(self) -> tuple[Cell, ...]:
        return tuple(
            (self.center_x + dx, self.feet_y - 1, self.center_z + dz)
            for dz in (0,) for dx in (9, 10))

    @property
    def lava_cleanup_bounds(self) -> tuple[Cell, Cell]:
        cx, y, cz = self.center_x, self.feet_y, self.center_z
        return (cx + 2, y - 1, cz - 7), (cx + 17, y + 1, cz + 8)

    @property
    def resource_obsidian_quota(self) -> int:
        return 2

    @property
    def mine_tool(self) -> str:
        return "netherite_pickaxe"

    @property
    def portal_frame_origin(self) -> Cell:
        return self.center_x + 8, self.feet_y, self.center_z - 29

    @property
    def portal_outer_frame_cells(self) -> tuple[Cell, ...]:
        fx, fy, fz = self.portal_frame_origin
        bottom = tuple((fx + dx, fy, fz) for dx in range(4))
        top = tuple((fx + dx, fy + 4, fz) for dx in range(4))
        left = tuple((fx, fy + dy, fz) for dy in range(1, 4))
        right = tuple((fx + 3, fy + dy, fz) for dy in range(1, 4))
        return tuple(dict.fromkeys(bottom + top + left + right))

    @property
    def portal_missing_cells(self) -> tuple[Cell, ...]:
        fx, fy, fz = self.portal_frame_origin
        return (
            (fx + 1, fy, fz),
            (fx, fy + 1, fz),
        )

    @property
    def portal_prebuilt_obsidian_cells(self) -> tuple[Cell, ...]:
        missing = set(self.portal_missing_cells)
        return tuple(
            cell for cell in self.portal_outer_frame_cells
            if cell not in missing)

    @property
    def portal_interior_cells(self) -> tuple[Cell, ...]:
        fx, fy, fz = self.portal_frame_origin
        return tuple(
            (fx + dx, fy + dy, fz)
            for dy in range(1, 4) for dx in (1, 2))

    @property
    def portal_place_marker_cells(self) -> tuple[Cell, ...]:
        return tuple((x, y, z - 1) for x, y, z in self.portal_missing_cells)

    @property
    def portal_ignite_netherrack_cell(self) -> Cell:
        fx, fy, fz = self.portal_frame_origin
        return fx + 1, fy + 2, fz - 1

    @property
    def portal_ignite_fire_cell(self) -> Cell:
        x, y, z = self.portal_ignite_netherrack_cell
        return x, y, z + 1

    @property
    def portal_backing_wall_cells(self) -> tuple[Cell, ...]:
        fx, fy, fz = self.portal_frame_origin
        return tuple(
            (fx + dx, fy + dy, fz - 1)
            for dy in range(5) for dx in range(4))

    @property
    def portal_place_marker_occluders(self) -> tuple[Cell, ...]:
        cells = []
        for x, y, z in self.portal_place_marker_cells:
            cells.extend((
                (x - 1, y, z), (x + 1, y, z),
                (x, y - 1, z), (x, y + 1, z), (x, y, z - 1)))
        return tuple(dict.fromkeys(cells))

    @property
    def portal_ignite_target_occluders(self) -> tuple[Cell, ...]:
        x, y, z = self.portal_ignite_netherrack_cell
        return (
            (x - 1, y, z), (x + 1, y, z),
            (x, y - 1, z), (x, y + 1, z), (x, y, z - 1))

    @property
    def stage_stance_cells(self) -> dict[str, tuple[Cell, ...]]:
        cx, y, cz = self.origin
        fx, _fy, fz = self.portal_frame_origin
        return {
            "water_scoop": ((cx - 9, y, cz),),
            "lava_pour": ((cx + 10, y, cz + 3),),
            "obsidian_mine": ((cx + 10, y, cz + 3),),
            "portal_place": ((fx + 2, y, fz + 2),),
            "portal_ignite": ((fx + 2, y, fz + 2),),
        }

    @property
    def target_cells(self) -> dict[str, tuple[Cell, ...]]:
        return {
            "water_scoop": self.water_cells,
            "lava_pour": self.lava_cells,
            "obsidian_mine": self.lava_cells,
            "portal_place": self.portal_place_marker_cells,
            "portal_ignite": (self.portal_ignite_netherrack_cell,),
        }


def _fill_tiled(builder: _CommandBuilder, first: Cell, last: Cell,
                block: str) -> None:
    x0, y0, z0 = first
    x1, y1, z1 = last
    for x in range(x0, x1 + 1, 24):
        for z in range(z0, z1 + 1, 24):
            builder.fill(
                (x, y0, z),
                (min(x + 23, x1), y1, min(z + 23, z1)), block)


def _tree(builder: _CommandBuilder, x: int, y: int, z: int,
          height: int) -> None:
    for dy in range(height):
        builder.setblock((x, y + dy, z), "oak_log")
    crown_y = y + height - 2
    for dy, radius in ((0, 2), (1, 2), (2, 1)):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if abs(dx) + abs(dz) > radius + 1:
                    continue
                cell = (x + dx, crown_y + dy, z + dz)
                if _base_kind(builder.blocks.get(cell, "air")) == "air":
                    builder.setblock(
                        cell, "oak_leaves[persistent=true]")


def _navigation_cells(builder: _CommandBuilder,
                      scene: WaterLavaObsidianPortalScene) -> set[Cell]:
    bounds = scene.reset_bounds
    y = scene.feet_y
    return {
        (x, y, z)
        for x in range(bounds["x_min"] + 3, bounds["x_max"] - 2)
        for z in range(bounds["z_min"] + 3, bounds["z_max"] - 2)
        if _base_kind(builder.blocks.get((x, y, z), "air")) == "air"
        and _base_kind(builder.blocks.get((x, y + 1, z), "air")) == "air"
        and _base_kind(builder.blocks.get((x, y - 1, z), "air"))
        not in {"air", "water", "lava"}
    }


def build_scene_commands(
        scene: WaterLavaObsidianPortalScene) -> SceneBuildResult:
    cx, y, cz = scene.origin
    bounds = scene.reset_bounds
    builder = _CommandBuilder()
    for command in (
            "/gamerule doMobSpawning false",
            "/gamerule doDaylightCycle false",
            "/gamerule doWeatherCycle false",
            "/gamerule doFireTick false",
            "/gamerule doTileDrops true",
            "/gamerule randomTickSpeed 0",
            "/gamerule fallDamage false",
            "/gamerule sendCommandFeedback false",
            "/gamerule announceAdvancements false",
            "/time set day", "/weather clear"):
        builder.command(command)

    _fill_tiled(
        builder,
        (bounds["x_min"], y, bounds["z_min"]),
        (bounds["x_max"], bounds["y_max"], bounds["z_max"]), "air")
    _fill_tiled(
        builder,
        (bounds["x_min"], y - 4, bounds["z_min"]),
        (bounds["x_max"], y - 2, bounds["z_max"]), "stone")
    _fill_tiled(
        builder,
        (bounds["x_min"], y - 1, bounds["z_min"]),
        (bounds["x_max"], y - 1, bounds["z_max"]), "grass_block")

    rim_boxes = (
        ((bounds["x_min"], y, bounds["z_min"]),
         (bounds["x_max"], y + 2, bounds["z_min"] + 2)),
        ((bounds["x_min"], y, bounds["z_max"] - 2),
         (bounds["x_max"], y + 2, bounds["z_max"])),
        ((bounds["x_min"], y, bounds["z_min"] + 3),
         (bounds["x_min"] + 2, y + 2, bounds["z_max"] - 3)),
        ((bounds["x_max"] - 2, y, bounds["z_min"] + 3),
         (bounds["x_max"], y + 2, bounds["z_max"] - 3)),
    )
    for first, last in rim_boxes:
        builder.fill(first, last, "dirt")
        builder.fill(
            (first[0], y + 3, first[2]),
            (last[0], y + 3, last[2]), "grass_block")

    outcrops = (
        ((cx - 2, y, cz - 4), (cx + 2, y + 2, cz + 7)),
        ((cx - 1, y + 3, cz - 3), (cx + 1, y + 3, cz + 6)),
        ((cx + 2, y, cz - 15), (cx + 13, y + 2, cz - 10)),
        ((cx + 4, y + 3, cz - 14), (cx + 11, y + 3, cz - 11)),
    )
    for first, last in outcrops:
        builder.fill(first, last, "stone")

    for dx, dz, height in (
            (-17, 5, 5), (-16, -7, 4), (18, 5, 5),
            (18, -20, 4), (-12, -24, 5)):
        _tree(builder, cx + dx, y, cz + dz, height)

    fx, fy, fz = scene.portal_frame_origin
    builder.fill((fx - 2, y - 1, fz - 2),
                 (fx + 5, y - 1, fz + 3), "stone_bricks")

    for cell in scene.portal_backing_wall_cells:
        builder.setblock(cell, "stone")

    for cell in scene.water_cells:
        builder.setblock(cell, "water[level=0]")
    for cell in scene.lava_cells:
        builder.setblock(cell, "lava[level=0]")
    for cell in scene.portal_prebuilt_obsidian_cells:
        builder.setblock(cell, "obsidian")
    for cell in scene.portal_missing_cells:
        builder.setblock(cell, "air")
    for cell in scene.portal_interior_cells:
        builder.setblock(cell, "air")
    if scene.portal_place_marker_kind not in PORTAL_PLACE_MARKER_KINDS:
        raise ValueError(
            f"unsupported portal marker: {scene.portal_place_marker_kind!r}")
    for cell in scene.portal_place_marker_cells:
        builder.setblock(cell, scene.portal_place_marker_kind)
    builder.setblock(scene.portal_ignite_netherrack_cell, "netherrack")

    for command in (
            "/kill @e[type=!minecraft:player]",
            "/kill @e[type=minecraft:item]",
            "/clear @a", "/effect clear @a", "/time set day"):
        builder.command(command)

    navigation = _navigation_cells(builder, scene)
    paths: dict[str, tuple[Cell, ...]] = {}
    departure = scene.start_navigation_cell
    for stage in ("water_scoop", "lava_pour", "portal_place"):
        path = _shortest_path(
            navigation, departure, scene.stage_stance_cells[stage])
        paths[stage] = path
        if path:
            departure = path[-1]
    paths["obsidian_mine"] = paths["lava_pour"][-1:]
    paths["portal_ignite"] = paths["portal_place"][-1:]

    verified_stands = {scene.start_navigation_cell}
    for stands in scene.stage_stance_cells.values():
        verified_stands.update(stands)
    counts = Counter(_base_kind(block) for block in builder.blocks.values())
    manifest = {
        "contract": SCENE_CONTRACT,
        "scene_name": "water_lava_obsidian_portal_bent_quarry",
        "layout_id": "wlo_portal_ccf_bends_water2x2_lava2x1_wall_portal_sealantern2_v6",
        "world_seed": WORLD_SEED,
        "fixed_world_single_layout": True,
        "origin": list(scene.origin),
        "reset_bounds": dict(bounds),
        "start_pose": list(scene.start_pose),
        "start_navigation_cell": list(scene.start_navigation_cell),
        "start_yaw": scene.start_yaw,
        "start_pitch": scene.start_pitch,
        "start_camera_contract": "all_four_stage_targets_zero_visible_pixels/v1",
        "policy_stage_keys": list(POLICY_STAGE_KEYS),
        "interaction_ids": list(INTERACTION_IDS),
        "target_cells": {
            key: [list(cell) for cell in cells]
            for key, cells in scene.target_cells.items()},
        "stage_stance_cells": {
            key: [list(cell) for cell in cells]
            for key, cells in scene.stage_stance_cells.items()},
        "water": {
            "cells": [list(cell) for cell in scene.water_cells],
            "source_cell_count": len(scene.water_cells),
            "shape": "2x2_infinite_source_flush_grass_cove",
            "natural_refill_behavior": "asynchronous_not_frame0_guaranteed",
        },
        "lava": {
            "cells": [list(cell) for cell in scene.lava_cells],
            "source_cell_count": len(scene.lava_cells),
            "shape": "2x1_flush_grass_cove",
            "conversion_semantic": (
                "first_causal_obsidian_finalizes_both_then_clears_water"),
            "post_conversion_water_cleanup_bounds": [
                list(cell) for cell in scene.lava_cleanup_bounds],
        },
        "portal": {
            "frame_origin": list(scene.portal_frame_origin),
            "outer_frame_positions": [
                list(cell) for cell in scene.portal_outer_frame_cells],
            "prebuilt_obsidian_positions": [
                list(cell) for cell in scene.portal_prebuilt_obsidian_cells],
            "missing_obsidian_positions": [
                list(cell) for cell in scene.portal_missing_cells],
            "place_marker_positions": [
                list(cell) for cell in scene.portal_place_marker_cells],
            "place_marker_count": len(scene.portal_place_marker_cells),
            "place_marker_kind": scene.portal_place_marker_kind,
            "ignite_netherrack_pos": list(
                scene.portal_ignite_netherrack_cell),
            "ignite_fire_pos": list(scene.portal_ignite_fire_cell),
            "ignite_item": "flint_and_steel",
            "attached_to_existing_north_dirt_wall": True,
            "backing_wall_positions": [
                list(cell) for cell in scene.portal_backing_wall_cells],
            "backing_wall_material": "stone",
            "backing_wall_thickness_blocks": 1,
            "place_marker_occluder_positions": [
                list(cell) for cell in scene.portal_place_marker_occluders],
            "place_marker_exposed_face": "+Z_into_missing_obsidian_cell",
            "ignite_target_occluder_positions": [
                list(cell) for cell in scene.portal_ignite_target_occluders],
            "ignite_target_exposed_face": "+Z_into_portal_interior",
            "ignite_target_offset_from_bottom_place_marker": [0, 2, 0],
            "raised_mossy_floor_distractors": 0,
            "interior_positions": [
                list(cell) for cell in scene.portal_interior_cells],
            "front_view_bottom_to_top": [
                "oloo", "lsso", "onso", "osso", "oooo"],
            "front_view_legend": {
                "o": "obsidian", "l": scene.portal_place_marker_kind,
                "s": "stone_backing", "n": "netherrack"},
            "initial_frame_state": "12_of_14_obsidian_two_cells_missing",
            "completion_semantic": (
                "place_two_missing_obsidian_then_ignite_netherrack_front_face"),
        },
        "navigation_cells": [list(cell) for cell in sorted(navigation)],
        "navigation_paths": {
            key: [list(cell) for cell in path]
            for key, path in paths.items()},
        "navigation_distances_blocks": {
            "start_to_water_stance": max(0, len(paths["water_scoop"]) - 1),
            "water_to_lava_stance": max(0, len(paths["lava_pour"]) - 1),
            "lava_to_portal_stance": max(0, len(paths["portal_place"]) - 1),
            "portal_markers_max_from_stance_euclidean": max(
                math.dist(
                    tuple(float(v) for v in scene.stage_stance_cells[
                        "portal_place"][0]),
                    tuple(float(v) for v in cell))
                for cell in scene.portal_place_marker_cells),
            "portal_ignite_from_stance_euclidean": math.dist(
                tuple(float(v) for v in scene.stage_stance_cells[
                    "portal_ignite"][0]),
                tuple(float(v) for v in scene.portal_ignite_netherrack_cell)),
        },
        "verified_stand_cells": [
            list(cell) for cell in sorted(verified_stands)],
        "initial_inventory_plan": {
            "hotbar.0": "bucket",
            "hotbar.1": "netherite_pickaxe",
            "hotbar.2": "flint_and_steel",
        },
        "staging": {
            "camera_off": True,
            "full_authored_reset": True,
            "live_voxel_and_verified_stand_audits_required": True,
            "frame0_live_exact_masks_required": True,
        },
        "final_block_counts": dict(sorted(counts.items())),
    }
    result = SceneBuildResult(tuple(builder.commands), manifest, builder.blocks)
    validate_scene_manifest(result, raise_on_error=True)
    return result


def validate_scene_manifest(result: SceneBuildResult, *,
                            raise_on_error: bool = False) -> dict:
    manifest = result.manifest
    blocks = result.final_blocks
    issues: list[str] = []
    if manifest.get("contract") != SCENE_CONTRACT:
        issues.append("scene contract mismatch")
    if tuple(manifest.get("policy_stage_keys", ())) != POLICY_STAGE_KEYS:
        issues.append("stage sequence mismatch")
    if "/gamerule doTileDrops true" not in result.commands:
        issues.append("natural obsidian pickup requires doTileDrops=true")
    if "/gamerule doTileDrops false" in result.commands:
        issues.append("natural obsidian tile drops were disabled")

    water = tuple(tuple(cell) for cell in manifest["water"]["cells"])
    lava = tuple(tuple(cell) for cell in manifest["lava"]["cells"])
    cleanup_bounds = tuple(
        tuple(cell) for cell in
        manifest["lava"]["post_conversion_water_cleanup_bounds"])
    if len(water) != 4 or any(
            _base_kind(blocks.get(cell, "air")) != "water" for cell in water):
        issues.append("water pool is not exactly four declared source cells")
    if len(lava) != 2 or any(
            _base_kind(blocks.get(cell, "air")) != "lava" for cell in lava):
        issues.append("lava pool is not exactly two declared source cells")
    fluid_counts = Counter(
        _base_kind(value) for value in blocks.values()
        if _base_kind(value) in {"water", "lava"})
    if fluid_counts != Counter({"water": 4, "lava": 2}):
        issues.append(f"undeclared fluid exists: {dict(fluid_counts)}")
    if len(cleanup_bounds) != 2:
        issues.append("lava spill-water cleanup bounds are malformed")
    else:
        cleanup_first, cleanup_last = cleanup_bounds

        def inside_cleanup(cell: Cell) -> bool:
            return all(
                min(cleanup_first[index], cleanup_last[index])
                <= cell[index]
                <= max(cleanup_first[index], cleanup_last[index])
                for index in range(3))

        if any(inside_cleanup(cell) for cell in water):
            issues.append("lava cleanup overlaps the authored 2x2 water pool")
        if any(not inside_cleanup(cell) for cell in lava):
            issues.append("lava cleanup does not cover every authored lava cell")

    portal = manifest["portal"]
    outer = {tuple(cell) for cell in portal["outer_frame_positions"]}
    prebuilt = {tuple(cell) for cell in portal["prebuilt_obsidian_positions"]}
    missing = {
        tuple(cell) for cell in portal["missing_obsidian_positions"]}
    markers = tuple(
        tuple(cell) for cell in portal["place_marker_positions"])
    netherrack = tuple(portal["ignite_netherrack_pos"])
    fire_cell = tuple(portal["ignite_fire_pos"])
    interior = {tuple(cell) for cell in portal["interior_positions"]}
    backing = {tuple(cell) for cell in portal["backing_wall_positions"]}
    marker_occluders = {
        tuple(cell) for cell in portal["place_marker_occluder_positions"]}
    ignite_occluders = {
        tuple(cell) for cell in portal["ignite_target_occluder_positions"]}
    fx, fy, fz = (int(value) for value in portal["frame_origin"])
    expected_backing = {
        (fx + dx, fy + dy, fz - 1)
        for dy in range(5) for dx in range(4)}
    expected_marker_occluders = {
        cell
        for marker in markers
        for cell in (
            (marker[0] - 1, marker[1], marker[2]),
            (marker[0] + 1, marker[1], marker[2]),
            (marker[0], marker[1] - 1, marker[2]),
            (marker[0], marker[1] + 1, marker[2]),
            (marker[0], marker[1], marker[2] - 1))
    }
    expected_ignite_occluders = {
        (netherrack[0] - 1, netherrack[1], netherrack[2]),
        (netherrack[0] + 1, netherrack[1], netherrack[2]),
        (netherrack[0], netherrack[1] - 1, netherrack[2]),
        (netherrack[0], netherrack[1] + 1, netherrack[2]),
        (netherrack[0], netherrack[1], netherrack[2] - 1),
    }
    if len(outer) != 14 or len(prebuilt) != 12 or outer - missing != prebuilt:
        issues.append("portal is not exactly 12/14 with two missing cells")
    if any(_base_kind(blocks.get(cell, "air")) != "obsidian"
           for cell in prebuilt):
        issues.append("prebuilt portal frame contains a non-obsidian cell")
    if len(missing) != 2 or any(
            _base_kind(blocks.get(cell, "air")) != "air"
            for cell in missing):
        issues.append("portal missing cells are not exactly two air cells")
    expected_markers = {
        (cell[0], cell[1], cell[2] - 1) for cell in missing}
    if len(markers) != 2 or set(markers) != expected_markers:
        issues.append("portal markers are not directly behind all missing cells")
    marker_kind = str(portal.get("place_marker_kind", ""))
    if marker_kind not in PORTAL_PLACE_MARKER_KINDS:
        issues.append("portal placement marker kind is unsupported")
    if any(_base_kind(blocks.get(marker, "air")) != marker_kind
           for marker in markers):
        issues.append("portal placement marker kinds mismatch")
    bottom_marker = min(markers, key=lambda cell: cell[1])
    if netherrack != (
            bottom_marker[0], bottom_marker[1] + 2,
            bottom_marker[2]):
        issues.append(
            "portal ignition netherrack is not two blocks above bottom marker")
    if _base_kind(blocks.get(netherrack, "air")) != "netherrack":
        issues.append("portal ignition target is not netherrack")
    if backing != expected_backing:
        issues.append("portal stone backing cells differ from the expected layout")
    if int(portal.get("backing_wall_thickness_blocks", 0)) != 1:
        issues.append("portal backing wall is not exactly one block thick")
    if any(_base_kind(blocks.get(cell, "air")) != "stone"
           for cell in backing - set(markers) - {netherrack}):
        issues.append("portal backing wall contains non-stone cells")
    if marker_occluders != expected_marker_occluders or any(
            _base_kind(blocks.get(cell, "air")) in {"air", "water", "lava"}
            for cell in marker_occluders):
        issues.append("portal marker has an exposed non-answer face")
    if ignite_occluders != expected_ignite_occluders or any(
            _base_kind(blocks.get(cell, "air")) in {"air", "water", "lava"}
            for cell in ignite_occluders):
        issues.append("portal ignition target has an exposed non-answer face")
    if int(portal.get("raised_mossy_floor_distractors", -1)) != 0 or any(
            _base_kind(value) == "mossy_stone_bricks"
            for value in blocks.values()):
        issues.append("portal floor contains mossy brick distractors")
    if fire_cell != (netherrack[0], netherrack[1], netherrack[2] + 1):
        issues.append("portal ignition fire cell is not in front of netherrack")
    if fire_cell not in interior:
        issues.append("portal ignition fire cell is outside portal interior")
    if len(interior) != 6 or any(
            _base_kind(blocks.get(cell, "air")) != "air"
            for cell in interior):
        issues.append("portal interior is not six air cells")

    navigation = {tuple(cell) for cell in manifest["navigation_cells"]}
    paths = manifest["navigation_paths"]
    for stage in ("water_scoop", "lava_pour", "portal_place"):
        path = tuple(tuple(cell) for cell in paths.get(stage, ()))
        if not path:
            issues.append(f"unreachable stage stance: {stage}")
            continue
        if any(cell not in navigation for cell in path):
            issues.append(f"path leaves audited navigation cells: {stage}")
    distances = manifest["navigation_distances_blocks"]
    if int(distances.get("water_to_lava_stance", 0)) < 20:
        issues.append("water-to-lava route is too short to expose exploration")
    if int(distances.get("lava_to_portal_stance", 0)) < 20:
        issues.append("lava-to-portal route is too short to expose exploration")
    if float(distances.get(
            "portal_markers_max_from_stance_euclidean", 99.0)) > 4.0:
        issues.append("a portal marker stance is outside the 4-block reach gate")
    if float(distances.get(
            "portal_ignite_from_stance_euclidean", 99.0)) > 4.0:
        issues.append("portal ignite stance is outside the 4-block reach gate")

    report = {
        "contract": "xbench_water_lava_obsidian_portal_scene_pure_audit/v1",
        "passed": not issues,
        "issues": issues,
    }
    if raise_on_error and issues:
        raise ValueError("; ".join(issues))
    return report
