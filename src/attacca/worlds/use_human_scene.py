"""Command-built cave and marker scenes for Place human play and evaluation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from attacca.worlds.episode_schema import class_visibility_roster_contract


USE_INTERACTION_ID = 3
USE_OBJ_ID = USE_INTERACTION_ID
USE_VERB_ID = USE_INTERACTION_ID
USE_INTERACTION_IDS = {"obj_id": USE_OBJ_ID, "verb_id": USE_VERB_ID}
USE_REACH_BLOCKS = 4.5

DEFAULT_TRAIN_MARKERS = (
    "oak_log",
    "birch_log",
    "pumpkin",
    "melon",
    "mossy_cobblestone",
    "hay_block",
    "bookshelf",
)
USE_CLASS_VISIBILITY_ROSTER_CONTRACT = class_visibility_roster_contract(
    DEFAULT_TRAIN_MARKERS)
USE_CLASS_VISIBILITY_CENSUS_CONTRACT = (
    "use_human_train7_native_depth_visibility_census/v1")
USE_CLASS_VISIBILITY_INDEX = {
    kind: index for index, kind in enumerate(DEFAULT_TRAIN_MARKERS)}
ZERO_SHOT_MARKERS = (
    "netherrack",
    "blue_ice",
    "magma_block",
    "prismarine_bricks",
)

MINE8_EXCLUDED_BLOCKS = (
    "coal_ore", "iron_ore", "gold_ore", "lapis_ore",
    "diamond_ore", "emerald_ore", "redstone_ore", "gilded_blackstone",
)

HELD_BLOCK_ROSTER = (
    "obsidian",
    "quartz_block",
    "purpur_block",
    "honeycomb_block",
    "red_wool",
    "orange_wool",
    "yellow_wool",
    "lime_wool",
    "blue_wool",
    "purple_wool",
    "magenta_wool",
    "black_wool",
)

MODE_TOOL_ID = {"block": 10}
MODE_TERMINAL_POLICY = {
    "block": "finish_all_expected_marker_faces_without_training_tail",
}

CAVE_TRAINING_CONTRACT = "use_human_natural_cave_training/v4"
CAVE_TRAINING_PATTERN = (
    "cave_wall_groups_1_2_3_sparse_partial_context_max_row5")
CAVE_PALETTE = (
    "stone", "andesite", "diorite", "granite", "gravel", "cobblestone",
)
CAVE_TRAIN_PALETTES = (
    ("stone", "andesite", "cobblestone", "gravel"),
    ("stone", "granite", "andesite", "gravel"),
    ("stone", "diorite", "andesite", "cobblestone"),
    ("stone", "granite", "diorite", "gravel"),
    ("stone", "cobblestone", "andesite", "granite"),
    ("stone", "diorite", "cobblestone", "gravel"),
)
CAVE_MACRO_VARIANTS = (
    "oval_even", "oval_low_shoulders", "oval_rocky_floor",
    "oval_dense_stalactites",
)
CAVE_PANEL_CLEARANCE_HALF_WIDTH = 3
CAVE_PANEL_CLEARANCE_MAX_DEPTH = 4
CAVE_PANEL_CLEARANCE_HEIGHT = 6
CAVE_MARKER_MIN_RELATIVE_Y = 0
CAVE_MARKER_MAX_RELATIVE_Y = 4


def bare_block(value: str) -> str:
    return str(value).strip().split(":")[-1]


def stable_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Marker:
    instance_id: str
    kind: str
    cell: tuple[int, int, int]
    outward_normal: tuple[int, int, int]
    place_cell: tuple[int, int, int]
    role: str
    pattern_index: int | None = None

    @property
    def surface(self) -> str:
        return "floor" if self.outward_normal == (0, 1, 0) else "wall"

    def document(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "kind": self.kind,
            "cell": list(self.cell),
            "outward_normal": list(self.outward_normal),
            "place_cell": list(self.place_cell),
            "surface": self.surface,
            "role": self.role,
            "pattern_index": self.pattern_index,
        }


@dataclass(frozen=True)
class MarkerRoom:
    seed: int
    center_x: int
    feet_y: int
    center_z: int
    background_split: str
    palette_index: int
    wall_block: str
    floor_block: str
    held_block: str
    target_kind: str
    train_markers: tuple[str, ...]
    zero_shot_markers: tuple[str, ...]
    mode: str
    pattern: str
    markers: tuple[Marker, ...]
    outer_radius: int = 10
    wall_height: int = 5

    @property
    def targets(self) -> tuple[Marker, ...]:
        return tuple(row for row in self.markers if row.role == "target")

    @property
    def confusers(self) -> tuple[Marker, ...]:
        return tuple(row for row in self.markers if row.role == "confuser")


@dataclass(frozen=True)
class CavePortalScene(MarkerRoom):

    walkable_bounds_xz: tuple[int, int, int, int] = (0, 0, 0, 0)
    cave_palette: tuple[str, ...] = CAVE_PALETTE
    prebuilt_blocks: tuple[tuple[tuple[int, int, int], str], ...] = ()
    light_cells: tuple[tuple[int, int, int], ...] = ()
    portal_interior_cells: tuple[tuple[int, int, int], ...] = ()

    @property
    def room_bounds(self) -> tuple[int, int, int, int, int, int]:
        x0, x1, z0, z1 = self.walkable_bounds_xz
        return (x0 - 1, self.feet_y - 2, z0 - 1,
                x1 + 1, self.feet_y + 8, z1 + 1)


@dataclass(frozen=True)
class CaveTrainingScene(CavePortalScene):

    macro_variant: str = CAVE_MACRO_VARIANTS[0]
    panel_assignments: tuple[
        tuple[str, str, tuple[int, int, int], tuple[int, int, int]], ...
    ] = ()
    context_group_indices_by_class: tuple[
        tuple[str, tuple[int, ...]], ...
    ] = ()
    context_cells_by_class: tuple[
        tuple[str, tuple[tuple[int, int, int], ...]], ...
    ] = ()

    def document(self) -> dict:
        x0, x1, z0, z1 = self.walkable_bounds_xz
        by_class = {
            kind: sorted(Counter(
                row.pattern_index for row in self.markers
                if row.kind == kind).values())
            for kind in self.train_markers
        }
        payload = {
            "contract": CAVE_TRAINING_CONTRACT,
            "training_eligible": True,
            "playtest_only": False,
            "seed": int(self.seed),
            "center_xyz": [self.center_x, self.feet_y, self.center_z],
            "cave_bounds_inclusive": list(self.room_bounds),
            "walkable_bounds_xz_inclusive": [x0, x1, z0, z1],
            "footprint_envelope": [x1 - x0 + 1, z1 - z0 + 1],
            "walkable_shape": "irregular_oval_within_20x20_envelope",
            "interior_height": 7,
            "marker_wall_rows_1based_inclusive": [
                CAVE_MARKER_MIN_RELATIVE_Y + 1,
                CAVE_MARKER_MAX_RELATIVE_Y + 1,
            ],
            "flat_panel_approach_clearance": {
                "tangent_half_width_blocks": CAVE_PANEL_CLEARANCE_HALF_WIDTH,
                "depths_from_backing_inclusive": [
                    2, CAVE_PANEL_CLEARANCE_MAX_DEPTH],
                "height_blocks": CAVE_PANEL_CLEARANCE_HEIGHT,
                "required_block": "air",
            },
            "macro_variant": self.macro_variant,
            "background_split": self.background_split,
            "palette_index": int(self.palette_index),
            "cave_palette": list(self.cave_palette),
            "lighting": {
                "ceiling_block": "glowstone",
                "ceiling_cells": [list(cell) for cell in self.light_cells],
                "night_vision_effect": True,
            },
            "target_kind": self.target_kind,
            "train_markers": list(self.train_markers),
            "zero_shot_markers": list(self.zero_shot_markers),
            "zero_shot_excluded_from_scene": True,
            "mode": self.mode,
            "held_block": self.held_block,
            "held_item": f"minecraft:{self.held_block} 64",
            "held_item_semantics": "random_nuisance_and_partial_shape_material",
            "hotbar_policy": (
                "preconfigured_before_episode_no_hotbar_actions"),
            "tool_id": MODE_TOOL_ID[self.mode],
            "interaction_id": USE_INTERACTION_ID,
            "interaction_id_compat_aliases": dict(USE_INTERACTION_IDS),
            "interaction_ids": dict(USE_INTERACTION_IDS),
            "terminal_policy": MODE_TERMINAL_POLICY[self.mode],
            "pattern": self.pattern,
            "group_sizes_by_class": by_class,
            "all_classes_equal_marker_count": len({
                sum(row.kind == kind for row in self.markers)
                for kind in self.train_markers}) == 1,
            "target_quota": len(self.targets),
            "target_surface_counts": {
                surface: sum(row.surface == surface for row in self.targets)
                for surface in ("wall", "floor")
            },
            "confuser_surface_counts": {
                surface: sum(row.surface == surface for row in self.confusers)
                for surface in ("wall", "floor")
            },
            "flat_host_panels": [
                {"panel_id": panel_id, "kind": kind,
                 "center_cell": list(center), "outward_normal": list(normal)}
                for panel_id, kind, center, normal in self.panel_assignments
            ],
            "partial_shape_context": (
                "one_or_two_seeded_role_neutral_groups_per_class; other_groups_marker_only; "
                "no_outward_context_below_any_marker_in_same_panel_column"),
            "context_group_indices_by_class": {
                kind: list(indices)
                for kind, indices in self.context_group_indices_by_class
            },
            "context_cells_by_class": {
                kind: [list(cell) for cell in cells]
                for kind, cells in self.context_cells_by_class
            },
            "prebuilt_blocks": [
                {"cell": list(cell), "kind": kind}
                for cell, kind in self.prebuilt_blocks
            ],
            "markers": [row.document() for row in self.markers],
        }
        payload["scene_sha256"] = stable_sha256(payload)
        return payload


def validate_train_roster(
        train_markers: Sequence[str],
        zero_shot_markers: Sequence[str] = ZERO_SHOT_MARKERS) -> tuple[str, ...]:
    train = tuple(bare_block(value) for value in train_markers)
    zero = {bare_block(value) for value in zero_shot_markers}
    if len(train) < 2 or len(set(train)) != len(train):
        raise ValueError("train marker roster must contain distinct classes")
    leaked = sorted(set(train) & zero)
    if leaked:
        raise ValueError(f"zero-shot markers leaked into train roster: {leaked}")
    if any(not value for value in train):
        raise ValueError("marker names must be non-empty")
    return train


def _marker(kind: str, cell: tuple[int, int, int], normal: tuple[int, int, int],
            role: str, index: int | None) -> Marker:
    place = tuple(int(cell[axis] + normal[axis]) for axis in range(3))
    identity = f"block:{cell[0]}:{cell[1]}:{cell[2]}"
    return Marker(identity, bare_block(kind), tuple(cell), tuple(normal),
                  place, role, index)


def _sample_cave_panel_layout(
        rng: np.random.Generator,
) -> tuple[
        tuple[tuple[tuple[int, int], ...], ...],
        tuple[int, ...],
        tuple[tuple[int, int], ...],
]:
    marker_h = tuple(range(-3, 4))
    marker_v = tuple(range(
        CAVE_MARKER_MIN_RELATIVE_Y, CAVE_MARKER_MAX_RELATIVE_Y + 1))

    def line(size: int) -> tuple[tuple[int, int], ...]:
        horizontal = bool(int(rng.integers(2)))
        if horizontal:
            h0 = int(rng.integers(marker_h[0], marker_h[-1] - size + 2))
            v0 = int(rng.choice(marker_v))
            return tuple((h0 + offset, v0) for offset in range(size))
        h0 = int(rng.choice(marker_h))
        v0 = int(rng.integers(marker_v[0], marker_v[-1] - size + 2))
        return tuple((h0, v0 + offset) for offset in range(size))

    def groups_separated(groups: Sequence[Sequence[tuple[int, int]]]) -> bool:
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1:]:
                if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
                       for a in left for b in right):
                    return False
        return True

    for _attempt in range(4096):
        groups = (
            ((int(rng.choice(marker_h)), int(rng.choice(marker_v))),),
            line(2),
            line(3),
        )
        flat_markers = {cell for group in groups for cell in group}
        if sum(len(group) for group in groups) != len(flat_markers):
            continue
        if not groups_separated(groups):
            continue

        context_count = int(rng.integers(1, 3))
        context_groups = tuple(sorted(
            int(value) for value in rng.choice(
                np.arange(3), size=context_count, replace=False)))
        context: list[tuple[int, int]] = []
        safe = True
        for group_index in context_groups:
            candidates = set()
            for h, v in groups[group_index]:
                candidates.update(((h - 1, v), (h + 1, v),
                                   (h, v - 1), (h, v + 1)))
            candidates = {
                cell for cell in candidates
                if -4 <= cell[0] <= 4
                and CAVE_MARKER_MIN_RELATIVE_Y <= cell[1] <= (
                    CAVE_MARKER_MAX_RELATIVE_Y)
                and cell not in flat_markers and cell not in context
                and not any(
                    cell[0] == marker_h_value and cell[1] < marker_v_value
                    for marker_h_value, marker_v_value in flat_markers)
            }
            ordered = sorted(candidates)
            rng.shuffle(ordered)
            if len(ordered) < 2:
                safe = False
                break
            context.extend(ordered[:2])
        if safe:
            return groups, context_groups, tuple(context)
    raise RuntimeError("could not sample a safe separated cave panel layout")


def build_cave_training_scene(
        *, seed: int, center_x: int, feet_y: int, center_z: int,
        target_kind: str, background_split: str = "train",
        held_block: str | None = None, palette_index: int | None = None,
        macro_variant: str | None = None,
        train_markers: Sequence[str] = DEFAULT_TRAIN_MARKERS,
        zero_shot_markers: Sequence[str] = ZERO_SHOT_MARKERS,
) -> CaveTrainingScene:
    train = validate_train_roster(train_markers, zero_shot_markers)
    target = bare_block(target_kind)
    if target not in train:
        raise ValueError(f"training target {target!r} is outside train roster")
    if background_split not in {"train", "eval"}:
        raise ValueError(f"unsupported cave background split {background_split!r}")
    rng = np.random.default_rng(int(seed))
    if palette_index is None:
        resolved_palette_index = int(rng.integers(len(CAVE_TRAIN_PALETTES)))
    else:
        resolved_palette_index = int(palette_index)
        if not 0 <= resolved_palette_index < len(CAVE_TRAIN_PALETTES):
            raise ValueError("cave palette index is outside the supported range")
    palette = tuple(CAVE_TRAIN_PALETTES[resolved_palette_index])
    if macro_variant is None:
        resolved_macro = CAVE_MACRO_VARIANTS[
            int(rng.integers(len(CAVE_MACRO_VARIANTS)))]
    else:
        resolved_macro = str(macro_variant)
        if resolved_macro not in CAVE_MACRO_VARIANTS:
            raise ValueError(f"unsupported cave macro variant {resolved_macro!r}")
    if held_block is None:
        resolved_held = HELD_BLOCK_ROSTER[
            int(rng.integers(len(HELD_BLOCK_ROSTER)))]
    else:
        resolved_held = bare_block(held_block)
        if resolved_held not in HELD_BLOCK_ROSTER:
            raise ValueError("cave held block is outside the safe full-cube roster")

    cx, y, cz = int(center_x), int(feet_y), int(center_z)
    x0, x1, z0, z1 = cx - 10, cx + 9, cz - 10, cz + 9
    panel_slots = [
        ("north_w", (cx - 5, y, z0 - 1), (0, 0, 1)),
        ("north_e", (cx + 5, y, z0 - 1), (0, 0, 1)),
        ("south_w", (cx - 5, y, z1 + 1), (0, 0, -1)),
        ("south_e", (cx + 5, y, z1 + 1), (0, 0, -1)),
        ("west_n", (x0 - 1, y, cz - 5), (1, 0, 0)),
        ("west_s", (x0 - 1, y, cz + 5), (1, 0, 0)),
        ("east_n", (x1 + 1, y, cz - 5), (-1, 0, 0)),
        ("east_s", (x1 + 1, y, cz + 5), (-1, 0, 0)),
    ]
    rng.shuffle(panel_slots)
    assignments = tuple(
        (panel_id, kind, center, normal)
        for kind, (panel_id, center, normal) in zip(train, panel_slots))

    markers: list[Marker] = []
    prebuilt_cells: list[tuple[int, int, int]] = []
    context_groups_by_class: list[tuple[str, tuple[int, ...]]] = []
    context_cells_by_class: list[
        tuple[str, tuple[tuple[int, int, int], ...]]] = []
    for panel_id, kind, center, normal in assignments:
        del panel_id
        tangent = ((1, 0, 0) if normal[2] else (0, 0, 1))

        def backing(h: int, v: int) -> tuple[int, int, int]:
            return (
                int(center[0] + tangent[0] * h),
                int(y + v),
                int(center[2] + tangent[2] * h),
            )

        def outward(cell: tuple[int, int, int]) -> tuple[int, int, int]:
            return tuple(int(cell[i] + normal[i]) for i in range(3))

        for _panel_attempt in range(256):
            local_groups, context_group_indices, local_context = (
                _sample_cave_panel_layout(rng))
            groups = tuple(
                tuple(backing(h, v) for h, v in group)
                for group in local_groups)
            proposed_markers = {cell for group in groups for cell in group}
            proposed_places = {outward(cell) for cell in proposed_markers}
            panel_context_cells = tuple(
                outward(backing(h, v)) for h, v in local_context)
            existing_markers = {row.cell for row in markers}
            existing_places = {row.place_cell for row in markers}
            existing_context = set(prebuilt_cells)
            if (proposed_markers & (
                    existing_markers | existing_places | existing_context)
                    or proposed_places & (
                        existing_markers | existing_places | existing_context)
                    or set(panel_context_cells) & (
                        existing_markers | existing_places | existing_context
                        | proposed_markers | proposed_places)):
                continue
            break
        else:
            raise RuntimeError(
                f"could not place collision-free cave panel for {kind}")
        role = "target" if kind == target else "confuser"
        for group_index, group in enumerate(groups):
            markers.extend(
                _marker(kind, cell, normal, role, group_index)
                for cell in group)
        prebuilt_cells.extend(panel_context_cells)
        context_groups_by_class.append((kind, context_group_indices))
        context_cells_by_class.append((kind, panel_context_cells))

    light_cells = tuple(
        (cx + dx, y + 7, cz + dz)
        for dx, dz in ((-5, -5), (5, -5), (-5, 5), (5, 5), (0, 0)))
    scene = CaveTrainingScene(
        seed=int(seed), center_x=cx, feet_y=y, center_z=cz,
        background_split=background_split,
        palette_index=resolved_palette_index,
        wall_block=palette[0], floor_block=palette[0],
        held_block=resolved_held, target_kind=target,
        train_markers=train,
        zero_shot_markers=tuple(bare_block(v) for v in zero_shot_markers),
        mode="block", pattern=CAVE_TRAINING_PATTERN,
        markers=tuple(markers), outer_radius=11, wall_height=8,
        walkable_bounds_xz=(x0, x1, z0, z1), cave_palette=palette,
        prebuilt_blocks=tuple(
            (tuple(cell), resolved_held) for cell in prebuilt_cells),
        light_cells=light_cells, portal_interior_cells=(),
        macro_variant=resolved_macro, panel_assignments=assignments,
        context_group_indices_by_class=tuple(context_groups_by_class),
        context_cells_by_class=tuple(context_cells_by_class))
    validate_cave_training_scene(scene)
    return scene


def validate_cave_training_scene(scene: CaveTrainingScene) -> None:
    if scene.pattern != CAVE_TRAINING_PATTERN:
        raise ValueError("cave training pattern changed")
    if scene.macro_variant not in CAVE_MACRO_VARIANTS:
        raise ValueError("unknown cave macro variant")
    x0, x1, z0, z1 = scene.walkable_bounds_xz
    if (x1 - x0 + 1, z1 - z0 + 1) != (20, 20):
        raise ValueError("cave training envelope must be exactly 20x20")
    if scene.palette_index not in range(len(CAVE_TRAIN_PALETTES)):
        raise ValueError("cave training palette index is invalid")
    if tuple(scene.cave_palette) != CAVE_TRAIN_PALETTES[scene.palette_index]:
        raise ValueError("cave training palette provenance disagrees")
    if scene.held_block not in HELD_BLOCK_ROSTER:
        raise ValueError("cave training held block is unsafe")
    if scene.held_block in set(scene.cave_palette) | set(scene.train_markers):
        raise ValueError("held block collides with cave or marker identity")
    if scene.held_block in set(MINE8_EXCLUDED_BLOCKS):
        raise ValueError("Mine-8 class leaked into cave held block")
    zero = set(scene.zero_shot_markers)
    staged_kinds = {
        *scene.cave_palette,
        *(row.kind for row in scene.markers),
        *(kind for _, kind in scene.prebuilt_blocks),
        "glowstone",
    }
    leaked = sorted(staged_kinds & zero)
    if leaked:
        raise ValueError(f"zero-shot marker leaked into cave training: {leaked}")
    if scene.target_kind not in scene.train_markers:
        raise ValueError("cave target is outside train roster")
    if len(scene.panel_assignments) != len(scene.train_markers):
        raise ValueError("every cave class needs one flat panel")
    if {row[1] for row in scene.panel_assignments} != set(scene.train_markers):
        raise ValueError("flat panel class assignment is incomplete")
    if len({row[0] for row in scene.panel_assignments}) != len(
            scene.panel_assignments):
        raise ValueError("flat host panel was assigned twice")

    counts = Counter(row.kind for row in scene.markers)
    if set(counts) != set(scene.train_markers) or set(counts.values()) != {6}:
        raise ValueError("every cave class must have exactly six markers")
    for kind in scene.train_markers:
        groups = Counter(
            row.pattern_index for row in scene.markers if row.kind == kind)
        if sorted(groups.values()) != [1, 2, 3]:
            raise ValueError(f"cave class {kind} lacks 1/2/3 groups")
    expected_confusers = 6 * (len(scene.train_markers) - 1)
    if len(scene.targets) != 6 or any(row.surface != "wall" for row in scene.targets):
        raise ValueError("all six cave targets must use flat wall faces")
    if len(scene.confusers) != expected_confusers or any(
            row.surface != "wall" for row in scene.confusers):
        raise ValueError(
            f"all {expected_confusers} cave confusers must use flat wall faces")
    marker_relative_y = [
        row.cell[1] - scene.feet_y for row in scene.markers]
    if (min(marker_relative_y) < CAVE_MARKER_MIN_RELATIVE_Y
            or max(marker_relative_y) > CAVE_MARKER_MAX_RELATIVE_Y):
        raise ValueError("cave marker is outside first-through-fifth wall rows")

    cells = [row.cell for row in scene.markers]
    places = [row.place_cell for row in scene.markers]
    prebuilt = [cell for cell, kind in scene.prebuilt_blocks
                if kind == scene.held_block]
    if len(cells) != len(set(cells)):
        raise ValueError("cave training marker cells collide")
    if len(places) != len(set(places)):
        raise ValueError("cave training placement cells collide")
    context_groups = dict(scene.context_group_indices_by_class)
    context_cells = dict(scene.context_cells_by_class)
    if set(context_groups) != set(scene.train_markers):
        raise ValueError("partial-context group provenance is incomplete")
    if set(context_cells) != set(scene.train_markers):
        raise ValueError("partial-context cell provenance is incomplete")
    if any(len(indices) not in (1, 2)
           or len(set(indices)) != len(indices)
           or not set(indices) <= {0, 1, 2}
           for indices in context_groups.values()):
        raise ValueError("each class needs one or two distinct context groups")
    if len(prebuilt) != 2 * sum(
            len(indices) for indices in context_groups.values()):
        raise ValueError("partial-context block count disagrees with groups")
    if len(prebuilt) != len(set(prebuilt)):
        raise ValueError("partial-shape blocks collide")
    if set(cells) & set(places):
        raise ValueError("a cave marker occupies another placement cell")
    if set(prebuilt) & (set(cells) | set(places)):
        raise ValueError("partial-shape context collides with marker transaction")
    accounted_prebuilt = set()
    for _panel_id, kind, center, normal in scene.panel_assignments:
        tangent = ((1, 0, 0) if normal[2] else (0, 0, 1))

        def local(cell: tuple[int, int, int]) -> tuple[int, int, int]:
            delta = tuple(cell[index] - center[index] for index in range(3))
            return (
                sum(delta[index] * tangent[index] for index in range(3)),
                cell[1] - center[1],
                sum(delta[index] * normal[index] for index in range(3)),
            )

        panel_markers = [row for row in scene.markers if row.kind == kind]
        local_markers = [(row.pattern_index, *local(row.cell)[:2])
                         for row in panel_markers]
        for left_index, left_h, left_v in local_markers:
            for right_index, right_h, right_v in local_markers:
                if (left_index != right_index
                        and max(abs(left_h - right_h),
                                abs(left_v - right_v)) <= 1):
                    raise ValueError("separate cave groups touch on a panel")
        panel_prebuilt = list(context_cells[kind])
        if len(panel_prebuilt) != 2 * len(context_groups[kind]):
            raise ValueError("panel context count disagrees with provenance")
        if any(local(cell)[2] != 1 or not -4 <= local(cell)[0] <= 4
               or not CAVE_MARKER_MIN_RELATIVE_Y <= local(cell)[1] <= (
                   CAVE_MARKER_MAX_RELATIVE_Y)
               for cell in panel_prebuilt):
            raise ValueError("panel context cell is outside its safe host patch")
        accounted_prebuilt.update(panel_prebuilt)
        selected_markers = {
            (h, v) for group_index, h, v in local_markers
            if group_index in context_groups[kind]}
        all_markers = {(h, v) for _group_index, h, v in local_markers}
        for cell in panel_prebuilt:
            h, v, _depth = local(cell)
            if any(h == marker_h and v < marker_v
                   for marker_h, marker_v in all_markers):
                raise ValueError("outward context sits below a cave marker")
            if not any(abs(h - marker_h) + abs(v - marker_v) == 1
                       for marker_h, marker_v in selected_markers):
                raise ValueError("partial context is detached from its group")
    if accounted_prebuilt != set(prebuilt):
        raise ValueError("partial context is outside all assigned panels")
    for row in scene.markers:
        if sum(abs(value) for value in row.outward_normal) != 1:
            raise ValueError("cave marker normal must be cardinal")
        expected = tuple(
            row.cell[i] + row.outward_normal[i] for i in range(3))
        if row.place_cell != expected:
            raise ValueError("cave placement cell does not match marker normal")
