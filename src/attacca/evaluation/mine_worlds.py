#!/usr/bin/env python3
"""Offline contracts for the Mine class-swap benchmark: the staged target and confuser cells of a world snapshot and the class reassignment applied at restore time."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence


REGRESSION_TARGETS = (
    "coal_ore",
    "iron_ore",
    "gold_ore",
    "lapis_ore",
    "diamond_ore",
    "emerald_ore",
    "redstone_ore",
)
GENERALIZATION_TARGETS = (
    "tube_coral_block",
    "brain_coral_block",
    "bubble_coral_block",
    "fire_coral_block",
    "horn_coral_block",
)
EVAL_TARGETS = REGRESSION_TARGETS + GENERALIZATION_TARGETS
CORAL_HYDRATION_BLOCK = "stone_brick_stairs[waterlogged=true]"

SCENE_MANIFEST_CONTRACT = (
    "xbench_mine_shared16_classswap_scene_manifest/v2"
)
SOURCE_PAYLOAD_CONTRACT = "xbench-v2-demo-staged-state/v1"

_IDENTITY_FIELDS = (
    "world_seed",
    "site_seed",
    "layout_seed",
    "pose_seed",
    "action_seed",
    "cell",
    "setting",
    "biome",
    "census_scrub",
    "mine_worldgen_profile",
)
_START_CONTEXT_FIELDS = (
    "exact_start",
    "fixed_start_yaw",
    "start_from",
    "mine_survey_route",
    "mine_survey_checkpoints",
)
_SITE_GEOMETRY_FIELDS = (
    "original",
    "source_block_kind",
    "source",
    "placement_source",
    "placement_template",
    "placement_geometry",
    "screened_placement_geometry",
    "staging_provenance",
    "pair_lane_id",
    "accessible",
    "support",
    "contacts",
    "exposed",
    "side_exposed",
    "primary_pocket_ok",
    "start_surface_pos",
    "start_surface_branches",
    "start_access_path_steps",
    "survey_checkpoint_step",
    "survey_view_pos",
    "survey_incoming_yaw",
    "survey_face_normal",
    "access_pos",
    "access_dist",
    "access_pitch",
)


def _canonical_json(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def norm_kind(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid block kind: {value!r}")
    kind = value.strip()
    if kind.startswith("minecraft:"):
        kind = kind.split(":", 1)[1]
    if not kind or ":" in kind:
        raise ValueError(f"invalid Minecraft block kind: {value!r}")
    return kind


def _cell(site: Mapping, field: str = "mine_sites") -> tuple[int, int, int]:
    output = []
    for axis in ("x", "y", "z"):
        value = site.get(axis)
        if isinstance(value, bool):
            raise ValueError(f"{field}.{axis} is not an integer coordinate")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field}.{axis} is not an integer coordinate") from exc
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{field}.{axis} is not an integer coordinate")
        output.append(integer)
    return tuple(output)


def _mine_sites(payload: Mapping, *, require_stone: bool = True) -> list[dict]:
    ctx = payload.get("ctx")
    if not isinstance(ctx, Mapping):
        raise ValueError("snapshot payload lacks ctx")
    raw_sites = ctx.get("mine_sites")
    if not isinstance(raw_sites, Sequence) or isinstance(raw_sites, (str, bytes)):
        raise ValueError("snapshot ctx.mine_sites is not a sequence")
    sites = []
    for index, raw in enumerate(raw_sites):
        if not isinstance(raw, Mapping):
            raise ValueError(f"mine_sites[{index}] is not an object")
        site = dict(raw)
        _cell(site, f"mine_sites[{index}]")
        if require_stone:
            if norm_kind(site.get("original")) != "stone":
                raise ValueError(
                    f"mine_sites[{index}] lacks original stone provenance")
            if ("source_block_kind" in site
                    and norm_kind(site["source_block_kind"]) != "stone"):
                raise ValueError(
                    f"mine_sites[{index}] source_block_kind is not stone")
        sites.append(site)
    if len(sites) != 6:
        raise ValueError(
            f"source snapshot must contain exactly six mine_sites, got {len(sites)}")
    cells = [_cell(site) for site in sites]
    if len(set(cells)) != 6:
        raise ValueError("source snapshot mine_sites cells are not all unique")
    return sites


def extract_source_layout(
        payload: Mapping, *,
        source_target_kind: str | None = None) -> dict:
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("snapshot payload lacks identity")
    original_source = norm_kind(
        identity.get("target_kind") if source_target_kind is None
        else source_target_kind)
    sites = _mine_sites(payload)
    targets = [site for site in sites
               if norm_kind(site.get("kind")) == original_source]
    distractors = [site for site in sites if site not in targets]
    if len(targets) != 3 or len(distractors) != 3:
        raise ValueError(
            "source snapshot is not exact source-target3 + distractor3")

    world_seed = int(identity.get("world_seed"))
    designated_site = min(targets, key=_cell)
    raw_normal = designated_site.get("survey_face_normal")
    if (not isinstance(raw_normal, Sequence)
            or isinstance(raw_normal, (str, bytes))
            or len(raw_normal) != 3):
        raise ValueError("designated target lacks a survey face normal")
    try:
        target_face_normal = tuple(int(value) for value in raw_normal)
    except (TypeError, ValueError) as exc:
        raise ValueError("designated target face normal is not integral") from exc
    if (sum(abs(value) for value in target_face_normal) != 1
            or any(value not in (-1, 0, 1) for value in target_face_normal)):
        raise ValueError("designated target face normal is not cardinal")
    designated_cell = _cell(designated_site)
    hydration_cell = tuple(
        designated_cell[index] - target_face_normal[index]
        for index in range(3))
    if hydration_cell in {_cell(site) for site in sites}:
        raise ValueError("hidden coral hydration cell overlaps a staged cell")
    if original_source in REGRESSION_TARGETS:
        base_target = original_source
    else:
        digest = hashlib.sha256(
            f"mine-shared16-base-target\0{world_seed}".encode()).digest()
        base_target = REGRESSION_TARGETS[
            int.from_bytes(digest[:4], "big") % len(REGRESSION_TARGETS)]

    ordered_sites = [designated_site] + sorted(
        (site for site in sites if site is not designated_site), key=_cell)
    used = {base_target}
    provisional = [base_target]
    for site in ordered_sites[1:]:
        raw_kind = norm_kind(site.get("kind"))
        keep = raw_kind if raw_kind in REGRESSION_TARGETS and raw_kind not in used else None
        provisional.append(keep)
        if keep is not None:
            used.add(keep)
    missing = [kind for kind in REGRESSION_TARGETS if kind not in used]
    normalized_rows = []
    for index, (site, proposed) in enumerate(zip(ordered_sites, provisional)):
        raw_kind = norm_kind(site.get("kind"))
        normalized = proposed
        if normalized is None:
            normalized = missing.pop(0)
        role = "target" if index == 0 else "confuser"
        normalized_rows.append({
            "cell": list(_cell(site)),
            "kind": raw_kind,
            "base_kind": normalized,
            "original": "stone",
            "role": role,
        })
    if (len(normalized_rows) != 6
            or len({row["base_kind"] for row in normalized_rows}) != 6
            or any(row["base_kind"] not in REGRESSION_TARGETS
                   for row in normalized_rows)):
        raise RuntimeError("ID-only six-cell source normalization failed")
    return {
        "world_seed": world_seed,
        "original_source_target_kind": original_source,
        "source_target_kind": base_target,
        "designated_target_cell": list(designated_cell),
        "target_face_normal": list(target_face_normal),
        "coral_hydration_cell": list(hydration_cell),
        "coral_hydration_block": CORAL_HYDRATION_BLOCK,
        "normalized_cells": normalized_rows,
    }


def deterministic_assignment(source_layout: Mapping,
                             target_kind: str) -> dict:
    requested = norm_kind(target_kind)
    if requested not in EVAL_TARGETS:
        raise ValueError(f"unsupported selected-world target: {requested}")
    source_kind = norm_kind(source_layout.get("source_target_kind"))
    rows = list(source_layout.get("normalized_cells") or ())
    designated = tuple(_cell_from_record({
        "cell": source_layout.get("designated_target_cell")},
        "designated_target_cell"))
    if (len(rows) != 6 or source_kind not in REGRESSION_TARGETS
            or sum(tuple(row.get("cell") or ()) == designated for row in rows) != 1):
        raise ValueError("source layout lacks one designated target among six cells")
    rotations = []
    staged = []
    for row in rows:
        cell = tuple(_cell_from_record(row, "normalized_cells"))
        raw_kind = norm_kind(row.get("kind"))
        base_kind = norm_kind(row.get("base_kind"))
        role = "target" if cell == designated else "confuser"
        if role == "target":
            after = requested
        elif base_kind == requested:
            after = source_kind
            rotations.append({
                "cell": list(cell),
                "before": base_kind,
                "after": after,
                "reason": "requested_target_confuser_collision",
            })
        else:
            after = base_kind
        staged.append({
            "cell": list(cell),
            "role": role,
            "source_kind": raw_kind,
            "base_kind": base_kind,
            "after": after,
        })
    base_confusers = [row["base_kind"] for row in staged
                      if row["role"] == "confuser"]
    if len(rotations) != int(requested in base_confusers):
        raise AssertionError("collision rotation is not one-for-one")

    staged.sort(key=lambda row: tuple(row["cell"]))
    targets = [row for row in staged if row["role"] == "target"]
    confusers = [row for row in staged if row["role"] == "confuser"]
    final_confuser_kinds = [row["after"] for row in confusers]
    if len(targets) != 1 or targets[0]["after"] != requested:
        raise AssertionError("assignment does not contain exactly one target")
    if (len(confusers) != 5 or len(set(final_confuser_kinds)) != 5
            or requested in final_confuser_kinds
            or any(kind not in REGRESSION_TARGETS
                   for kind in final_confuser_kinds)):
        raise ValueError(
            "assignment must contain five distinct ID-only confusers")

    result = {
        "target_kind": requested,
        "source_target_kind": source_kind,
        "original_source_target_kind": source_layout.get(
            "original_source_target_kind"),
        "staged_cells": staged,
        "target_cells": [row["cell"] for row in targets],
        "confuser_cells": [row["cell"] for row in confusers],
        "confuser_classes": final_confuser_kinds,
        "collision_rotations": rotations,
        "target_face_normal": list(source_layout.get("target_face_normal") or ()),
        "coral_hydration_cell": list(
            source_layout.get("coral_hydration_cell") or ()),
        "coral_hydration_block": str(
            source_layout.get("coral_hydration_block") or ""),
    }
    result["assignment_sha256"] = assignment_sha256(result)
    return result


def _cell_from_record(row: Mapping, field: str) -> tuple[int, int, int]:
    raw = row.get("cell")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) \
            or len(raw) != 3:
        raise ValueError(f"{field} row has invalid cell")
    return _cell(dict(zip(("x", "y", "z"), raw)), field)


def assignment_sha256(assignment: Mapping) -> str:
    rows = assignment.get("staged_cells")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("assignment lacks staged_cells")
    canonical = []
    for row in rows:
        canonical.append({
            "cell": list(_cell_from_record(row, "staged_cells")),
            "role": str(row.get("role")),
            "source_kind": norm_kind(row.get("source_kind")),
            "after": norm_kind(row.get("after")),
        })
    canonical.sort(key=lambda row: tuple(row["cell"]))
    hydration_cell = _cell_from_record({
        "cell": assignment.get("coral_hydration_cell")},
        "coral_hydration_cell")
    hydration_block = str(assignment.get("coral_hydration_block") or "")
    if hydration_block != CORAL_HYDRATION_BLOCK:
        raise ValueError("coral hydration block contract differs")
    return _sha256_json({
        "staged_cells": canonical,
        "coral_hydration_cell": list(hydration_cell),
        "coral_hydration_block": hydration_block,
    })


def expected_source_to_variant_diff(assignment: Mapping) -> list[dict]:
    output = []
    for row in assignment.get("staged_cells") or ():
        before = norm_kind(row.get("source_kind"))
        after = norm_kind(row.get("after"))
        if before != after:
            output.append({
                "cell": list(_cell_from_record(row, "staged_cells")),
                "before": before,
                "after": after,
            })
    return sorted(output, key=lambda row: tuple(row["cell"]))


def _canonical_diff(rows: Iterable[Mapping]) -> list[dict]:
    output = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("voxel diff row is not an object")
        output.append({
            "cell": list(_cell_from_record(row, "voxel_diff")),
            "before": norm_kind(row.get("before")),
            "after": norm_kind(row.get("after")),
        })
    output.sort(key=lambda row: tuple(row["cell"]))
    if len({tuple(row["cell"]) for row in output}) != len(output):
        raise ValueError("voxel diff contains duplicate cells")
    return output


def assert_exact_diff(actual: Iterable[Mapping],
                      expected: Iterable[Mapping]) -> None:
    actual_rows = _canonical_diff(actual)
    expected_rows = _canonical_diff(expected)
    if actual_rows != expected_rows:
        raise RuntimeError(
            "class-swap voxel diff mismatch: "
            f"expected={expected_rows!r} actual={actual_rows!r}")


def _json_safe(value):
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry/start identity is not strict JSON") from exc


def geometry_start_projection(payload: Mapping) -> dict:
    if payload.get("contract") != SOURCE_PAYLOAD_CONTRACT:
        raise ValueError("unsupported source snapshot payload contract")
    sites = _mine_sites(payload)
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("snapshot payload lacks identity")
    start = payload.get("accepted_start_pose")
    if not isinstance(start, Mapping):
        raise ValueError("snapshot payload lacks accepted_start_pose")
    for field in ("x", "y", "z", "yaw", "pitch"):
        try:
            value = float(start.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"accepted_start_pose.{field} is not numeric") from exc
        if not math.isfinite(value):
            raise ValueError(
                f"accepted_start_pose.{field} is not finite")

    site_projection = []
    for site in sites:
        row = {"cell": list(_cell(site))}
        for field in _SITE_GEOMETRY_FIELDS:
            if field in site:
                row[field] = site[field]
        site_projection.append(row)
    site_projection.sort(key=lambda row: tuple(row["cell"]))

    ctx = payload["ctx"]
    output = {
        "source_payload_contract": SOURCE_PAYLOAD_CONTRACT,
        "identity": {
            field: identity[field]
            for field in _IDENTITY_FIELDS if field in identity
        },
        "accepted_start_pose": dict(start),
        "world_origin": payload.get("world_origin"),
        "start_context": {
            field: ctx[field]
            for field in _START_CONTEXT_FIELDS if field in ctx
        },
        "sites": site_projection,
    }
    return _json_safe(output)


def geometry_start_sha256(payload: Mapping) -> str:
    return _sha256_json(geometry_start_projection(payload))


def _require_sha256(value, field: str) -> str:
    digest = str(value)
    if (len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError(f"{field} is not a lowercase SHA256")
    return digest


def source_snapshot_hash_pins(metadata: Mapping,
                              payload: Mapping) -> dict:
    if not isinstance(metadata, Mapping):
        raise ValueError("snapshot metadata is not an object")
    return {
        "source_world_sha256": _require_sha256(
            metadata.get("world_sha256"), "world_sha256"),
        "source_world_archive_sha256": _require_sha256(
            metadata.get("world_archive_sha256"), "world_archive_sha256"),
        "source_payload_sha256": _require_sha256(
            metadata.get("payload_sha256"), "payload_sha256"),
        "source_staged_rgb_sha256": _require_sha256(
            payload.get("staged_rgb_sha256"), "staged_rgb_sha256"),
    }



def _grid_norm_kind(value: str) -> str:
    value = str(value).lower().replace("minecraft:", "")
    return value.split("/")[-1]


def canonical_grid_bytes(grid: dict) -> bytes:
    rows = []
    for cell, block in sorted(grid.items()):
        row = [int(cell[0]), int(cell[1]), int(cell[2]), _grid_norm_kind(block)]
        rows.append(json.dumps(row, separators=(",", ":")))
    return ("\n".join(rows) + "\n").encode("utf-8")


def canonical_grid_sha256(grid: dict) -> str:
    return hashlib.sha256(canonical_grid_bytes(grid)).hexdigest()


def grid_diff(before: dict, after: dict):
    rows = []
    for cell in sorted(set(before) | set(after)):
        old = _grid_norm_kind(before.get(cell, "air"))
        new = _grid_norm_kind(after.get(cell, "air"))
        if old != new:
            rows.append({
                "cell": [int(q) for q in cell],
                "before": old,
                "after": new,
            })
    return rows


def pixel_exclusion(info: dict):
    import numpy as np
    from attacca.evaluation import geometry as G
    from attacca.worlds.human_viewmodel import RENDERER_VIEWMODEL_INFO_FIELD

    excluded = G.native_ui_pixel_exclusion_mask().copy()
    renderer = (info or {}).get(RENDERER_VIEWMODEL_INFO_FIELD)
    if (isinstance(renderer, np.ndarray)
            and renderer.dtype == np.uint8
            and renderer.shape == excluded.shape
            and not np.any((renderer != 0) & (renderer != 1))):
        x0, y0, x1, y1 = G.NATIVE_UI_STATIC_RECTS_XYXY[
            "held_item_conservative"]
        excluded[y0:y1, x0:x1] = 0
        excluded |= renderer
    return excluded
