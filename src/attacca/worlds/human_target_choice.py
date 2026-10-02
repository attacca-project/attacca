from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from attacca.worlds.structural_decision_merge import CLASS_CENSUS_FIELDS
from attacca.worlds.structural_decision_merge import structural_rgb_identity


VISIBLE_SURFACE_MASK_SEMANTIC = "chosen_instance_actual_visible_surface/v1"
HUMAN_CHOICE_SOURCE = "human_current_rgb_interaction_raycast/v1"
HUMAN_MARKER_SOURCE = "human_frozen_current_rgb_pixel_marker/v2"


class HumanTargetChoiceError(ValueError):
    pass


def _clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_clone(item) for item in value]
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _exact_cell(raw: Any, *, label: str) -> tuple[int, int, int]:
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        raise HumanTargetChoiceError(f"{label} must be an xyz cell")
    if any(isinstance(value, bool) for value in raw):
        raise HumanTargetChoiceError(f"{label} contains a boolean coordinate")
    try:
        cell = tuple(int(value) for value in raw)
        if any(float(raw[index]) != float(cell[index]) for index in range(3)):
            raise HumanTargetChoiceError(f"{label} is not an exact integer cell")
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, HumanTargetChoiceError):
            raise
        raise HumanTargetChoiceError(f"{label} is not an exact integer cell") from exc
    return cell


def _finite_vector(raw: Any, *, label: str, integer: bool) -> list[int] | list[float]:
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        raise HumanTargetChoiceError(f"{label} must have three coordinates")
    if any(isinstance(value, bool) for value in raw):
        raise HumanTargetChoiceError(f"{label} contains a boolean coordinate")
    try:
        if integer:
            out = [int(value) for value in raw]
            if any(float(raw[index]) != float(out[index]) for index in range(3)):
                raise HumanTargetChoiceError(f"{label} is not integral")
            return out
        out = [float(value) for value in raw]
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, HumanTargetChoiceError):
            raise
        raise HumanTargetChoiceError(f"{label} is malformed") from exc
    if not all(math.isfinite(value) for value in out):
        raise HumanTargetChoiceError(f"{label} is not finite")
    return out


def _instance_cells(instance: Mapping[str, Any]) -> frozenset[tuple[int, int, int]]:
    raw_cells = instance.get("component_cells")
    if raw_cells is None:
        proof = instance.get("proof")
        if isinstance(proof, Mapping):
            raw_cells = proof.get("visible_member_cells")
    if raw_cells is None:
        raw_cells = [instance.get("world_position")]
    if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
        raise HumanTargetChoiceError("visible instance component_cells is malformed")
    cells = frozenset(
        _exact_cell(raw, label="visible instance component cell")
        for raw in raw_cells)
    if not cells:
        raise HumanTargetChoiceError("visible instance has no component cells")
    return cells


def _validate_dense_instance(instance: Mapping[str, Any]) -> tuple[list[float], list[int], dict]:
    if int(instance.get("visible", 0)) != 1:
        raise HumanTargetChoiceError("human choice candidate is not currently visible")
    identity = instance.get("instance_id")
    if identity is None or not str(identity):
        raise HumanTargetChoiceError("human choice candidate lacks instance_id")
    point = _finite_vector(
        instance.get("visible_point"), label="visible_point", integer=False)
    normal = _finite_vector(
        instance.get("face_normal"), label="face_normal", integer=True)
    proof = instance.get("proof")
    if not isinstance(proof, Mapping):
        raise HumanTargetChoiceError("human choice candidate lacks dense proof")
    if (proof.get("dense_segmentation_supervision") is not True
            or proof.get("certified_mask_semantics")
               != VISIBLE_SURFACE_MASK_SEMANTIC
            or proof.get("oracle_recognizable") is not True):
        raise HumanTargetChoiceError(
            "human choice candidate is not an exact recognizable visible surface")
    runs = proof.get("certified_pixel_runs_yx")
    if not isinstance(runs, list) or not runs:
        raise HumanTargetChoiceError("human choice candidate has no certified pixels")
    proof_point = _finite_vector(
        proof.get("visible_point"), label="proof.visible_point", integer=False)
    proof_normal = _finite_vector(
        proof.get("face_normal"), label="proof.face_normal", integer=True)
    if proof_point != point or proof_normal != normal:
        raise HumanTargetChoiceError(
            "instance point/normal disagree with the certified surface proof")
    return point, normal, _clone(proof)


def _snapshot_aux_matches(rows: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any]) -> None:
    confusers = snapshot.get("confuser_instances")
    census = snapshot.get("class_census")
    if not isinstance(confusers, list):
        raise HumanTargetChoiceError("predecision auxiliary snapshot is malformed")
    for row in rows:
        if row.get("confuser_instances") != confusers:
            raise HumanTargetChoiceError(
                "same-RGB confuser evidence changed before human target rebind")
        if census is None:
            if any(field in row for field in CLASS_CENSUS_FIELDS):
                raise HumanTargetChoiceError(
                    "same-RGB class census appeared after an absent snapshot")
        else:
            if not isinstance(census, Mapping):
                raise HumanTargetChoiceError("predecision class census is malformed")
            if any(row.get(field) != census.get(field)
                   for field in CLASS_CENSUS_FIELDS):
                raise HumanTargetChoiceError(
                    "same-RGB class census changed before human target rebind")


def _human_raycast_target_choice(
        pending_row: Mapping[str, Any] | None,
        story_row: Mapping[str, Any] | None,
        ray_cell: Sequence[int],
        *,
        predecision_aux: Mapping[str, Any] | None,
        source: str,
        ray_label: str,
) -> dict[str, Any]:
    if not isinstance(pending_row, Mapping) or not isinstance(story_row, Mapping):
        raise HumanTargetChoiceError("human choice has no pending traj/story pair")
    if not isinstance(predecision_aux, Mapping):
        raise HumanTargetChoiceError(
            "human choice lacks a predecision same-RGB auxiliary snapshot")
    identity = structural_rgb_identity(pending_row, story_row)
    if identity is None or predecision_aux.get("frame_identity") != identity:
        raise HumanTargetChoiceError(
            "human choice rows do not match the predecision rendered RGB")
    _snapshot_aux_matches((pending_row, story_row), predecision_aux)

    pending_instances = pending_row.get("visible_instances")
    story_instances = story_row.get("visible_instances")
    if (not isinstance(pending_instances, list)
            or pending_instances != story_instances):
        raise HumanTargetChoiceError(
            "pending traj/story visible instances differ on the current RGB")
    ray = _exact_cell(ray_cell, label=ray_label)
    identities: set[str] = set()
    matches: list[tuple[Mapping[str, Any], frozenset[tuple[int, int, int]]]] = []
    for instance in pending_instances:
        if not isinstance(instance, Mapping):
            raise HumanTargetChoiceError("visible_instances contains a non-record")
        identity9 = str(instance.get("instance_id") or "")
        if not identity9 or identity9 in identities:
            raise HumanTargetChoiceError("visible instance IDs are missing or duplicated")
        identities.add(identity9)
        cells = _instance_cells(instance)
        if ray in cells:
            matches.append((instance, cells))
    if len(matches) != 1:
        raise HumanTargetChoiceError(
            f"human ray must identify exactly one visible instance, got {len(matches)}")

    selected, component_cells = matches[0]
    point, normal, proof = _validate_dense_instance(selected)
    chosen_id = str(selected["instance_id"])
    rebound = []
    chosen_count = 0
    for raw in pending_instances:
        instance = _clone(raw)
        selected9 = str(instance.get("instance_id")) == chosen_id
        instance["chosen"] = bool(selected9)
        chosen_count += int(selected9)
        rebound.append(instance)
    if chosen_count != 1:
        raise HumanTargetChoiceError("human rebind did not produce one chosen instance")

    previous = pending_row.get("chosen_instance_id")
    return {
        "version": 1,
        "source": str(source),
        "frame_identity": _clone(identity),
        "ray_cell": list(ray),
        "chosen_instance_id": chosen_id,
        "previous_chosen_instance_id": (
            None if previous is None else str(previous)),
        "chosen_changed": previous is None or str(previous) != chosen_id,
        "component_cells": [list(cell) for cell in sorted(component_cells)],
        "visible_point": point,
        "face_normal": normal,
        "proof": proof,
        "visible_instances": rebound,
        "same_rgb_aux": {
            "confuser_instances": _clone(predecision_aux["confuser_instances"]),
            "class_census": _clone(predecision_aux.get("class_census")),
        },
    }


def human_raycast_target_choice(
        pending_row: Mapping[str, Any] | None,
        story_row: Mapping[str, Any] | None,
        ray_cell: Sequence[int],
        *,
        predecision_aux: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _human_raycast_target_choice(
        pending_row, story_row, ray_cell,
        predecision_aux=predecision_aux,
        source=HUMAN_CHOICE_SOURCE,
        ray_label="human interaction ray cell")


def _marker_pixel(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise HumanTargetChoiceError("human marker pixel must be [x,y]")
    if any(isinstance(value, bool) for value in raw):
        raise HumanTargetChoiceError("human marker pixel contains a boolean")
    try:
        pixel = tuple(int(value) for value in raw)
        if any(float(raw[index]) != float(pixel[index]) for index in range(2)):
            raise HumanTargetChoiceError("human marker pixel is not integral")
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, HumanTargetChoiceError):
            raise
        raise HumanTargetChoiceError("human marker pixel is malformed") from exc
    if not (0 <= pixel[0] < 640 and 0 <= pixel[1] < 360):
        raise HumanTargetChoiceError("human marker pixel is outside raw RGB")
    return pixel


def _proof_contains_pixel(proof: Mapping[str, Any], pixel: tuple[int, int]) -> bool:
    shape = proof.get("certified_mask_shape")
    if shape != [360, 640] and shape != (360, 640):
        raise HumanTargetChoiceError("human marker mask shape is not native RGB")
    xx, yy = pixel
    runs = proof.get("certified_pixel_runs_yx")
    if not isinstance(runs, list) or not runs:
        raise HumanTargetChoiceError("human marker candidate has no certified pixels")
    for raw in runs:
        if (not isinstance(raw, (tuple, list)) or len(raw) != 3
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in raw)):
            raise HumanTargetChoiceError("human marker mask run is malformed")
        row, x0, x1 = raw
        if not (0 <= row < 360 and 0 <= x0 <= x1 < 640):
            raise HumanTargetChoiceError("human marker mask run is outside raw RGB")
        if row == yy and x0 <= xx <= x1:
            return True
        if row > yy:
            break
    return False


def human_marker_pixel_match_ids(
        visible_instances: Sequence[Mapping[str, Any]],
        pixel_xy: Sequence[int],
) -> list[str]:
    pixel = _marker_pixel(pixel_xy)
    if (not isinstance(visible_instances, Sequence)
            or isinstance(visible_instances, (str, bytes))):
        raise HumanTargetChoiceError("visible instances are malformed")
    identities: set[str] = set()
    matches: list[str] = []
    for instance in visible_instances:
        if not isinstance(instance, Mapping):
            raise HumanTargetChoiceError("visible_instances contains a non-record")
        identity = str(instance.get("instance_id") or "")
        if not identity or identity in identities:
            raise HumanTargetChoiceError("visible instance IDs are missing or duplicated")
        identities.add(identity)
        _point, _normal, proof = _validate_dense_instance(instance)
        if _proof_contains_pixel(proof, pixel):
            matches.append(identity)
    return matches


def human_marker_pixel_target_choice(
        pending_row: Mapping[str, Any] | None,
        story_row: Mapping[str, Any] | None,
        pixel_xy: Sequence[int],
        *,
        predecision_aux: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(pending_row, Mapping) or not isinstance(story_row, Mapping):
        raise HumanTargetChoiceError("human marker has no pending traj/story pair")
    if not isinstance(predecision_aux, Mapping):
        raise HumanTargetChoiceError(
            "human marker lacks a predecision same-RGB auxiliary snapshot")
    identity = structural_rgb_identity(pending_row, story_row)
    if identity is None or predecision_aux.get("frame_identity") != identity:
        raise HumanTargetChoiceError(
            "human marker rows do not match the frozen rendered RGB")
    _snapshot_aux_matches((pending_row, story_row), predecision_aux)
    pending_instances = pending_row.get("visible_instances")
    story_instances = story_row.get("visible_instances")
    if (not isinstance(pending_instances, list)
            or pending_instances != story_instances):
        raise HumanTargetChoiceError(
            "pending traj/story visible instances differ on the frozen RGB")
    pixel = _marker_pixel(pixel_xy)
    match_ids = human_marker_pixel_match_ids(pending_instances, pixel)
    if len(match_ids) != 1:
        raise HumanTargetChoiceError(
            f"human marker pixel must identify exactly one visible instance, "
            f"got {len(match_ids)}")
    chosen_id = match_ids[0]
    selected = next(instance for instance in pending_instances
                    if str(instance.get("instance_id")) == chosen_id)
    point, normal, proof = _validate_dense_instance(selected)
    component_cells = _instance_cells(selected)
    rebound = []
    for raw in pending_instances:
        instance = _clone(raw)
        instance["chosen"] = str(instance.get("instance_id")) == chosen_id
        rebound.append(instance)
    previous = pending_row.get("chosen_instance_id")
    return {
        "version": 2,
        "source": HUMAN_MARKER_SOURCE,
        "frame_identity": _clone(identity),
        "pixel_xy": list(pixel),
        "chosen_instance_id": chosen_id,
        "previous_chosen_instance_id": (
            None if previous is None else str(previous)),
        "chosen_changed": previous is None or str(previous) != chosen_id,
        "component_cells": [list(cell) for cell in sorted(component_cells)],
        "visible_point": point,
        "face_normal": normal,
        "proof": proof,
        "visible_instances": rebound,
        "same_rgb_aux": {
            "confuser_instances": _clone(predecision_aux["confuser_instances"]),
            "class_census": _clone(predecision_aux.get("class_census")),
        },
    }
