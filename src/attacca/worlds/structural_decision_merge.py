from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


RENDER_POSE_FLOAT_FIELDS = (
    "x", "y", "z", "yaw", "pitch", "eye_height",
)
CLASS_CENSUS_FIELDS = (
    "class_visibility_known_bits",
    "class_visibility_visible_bits",
    "class_visible_instances",
    "class_visibility_scan_witness",
    "class_census_runtime",
)
OPTIONAL_CLASS_CENSUS_FIELDS = ("class_union_masks",)
FULL_CURRENT_CAMERA_SOURCE = "current_camera+known_geometry"
FRAME_IDENTITY_FIELD = "structural_visibility_frame_identity"


def _clone_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone_jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_clone_jsonish(item) for item in value]
    return value


def _normalise_render_pose(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    pose: dict[str, Any] = {}
    for field in RENDER_POSE_FLOAT_FIELDS:
        value = raw.get(field)
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        pose[field] = number
    source = raw.get("source")
    if not isinstance(source, str) or not source:
        return None
    pose["source"] = source
    return pose


def structural_rgb_identity(
        pending_row: Mapping[str, Any] | None,
        story_row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(pending_row, Mapping) or not isinstance(story_row, Mapping):
        return None
    pending_t = pending_row.get("t")
    story_t = story_row.get("traj_t")
    video_f = story_row.get("f")
    if any(isinstance(value, bool) for value in (pending_t, story_t, video_f)):
        return None
    raw_indices = (pending_t, story_t, video_f)
    try:
        pending_t = int(pending_t)
        story_t = int(story_t)
        video_f = int(video_f)
    except (TypeError, ValueError):
        return None
    for raw, normalised in zip(raw_indices, (pending_t, story_t, video_f)):
        try:
            if float(raw) != float(normalised):
                return None
        except (TypeError, ValueError):
            return None
    if pending_t < 0 or video_f < 0 or pending_t != story_t:
        return None
    pending_pose = _normalise_render_pose(pending_row.get("render_pose"))
    story_pose = _normalise_render_pose(story_row.get("render_pose"))
    if pending_pose is None or story_pose is None or pending_pose != story_pose:
        return None
    return {
        "version": 1,
        "traj_t": pending_t,
        "video_f": video_f,
        "render_pose": pending_pose,
    }


def same_rgb_structural_aux(
        pending_row: Mapping[str, Any] | None,
        story_row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    identity = structural_rgb_identity(pending_row, story_row)
    if identity is None:
        return None
    assert pending_row is not None and story_row is not None
    for row in (pending_row, story_row):
        if row.get(FRAME_IDENTITY_FIELD) != identity:
            return None
        if row.get("structural_visibility_source") != FULL_CURRENT_CAMERA_SOURCE:
            return None

    compared_fields = (
        "confuser_instances", "confuser_exist", "confuser_classes_visible",
        *CLASS_CENSUS_FIELDS, *OPTIONAL_CLASS_CENSUS_FIELDS,
    )
    for field in compared_fields:
        if (field in pending_row) != (field in story_row):
            return None
        if field in pending_row and pending_row[field] != story_row[field]:
            return None

    confusers = pending_row.get("confuser_instances")
    classes = pending_row.get("confuser_classes_visible")
    if not isinstance(confusers, list) or not isinstance(classes, list):
        return None
    try:
        confuser_exist = int(pending_row.get("confuser_exist"))
    except (TypeError, ValueError):
        return None
    if confuser_exist not in (0, 1) or confuser_exist != int(bool(confusers)):
        return None
    expected_classes = sorted({
        str(instance.get("kind"))
        for instance in confusers if isinstance(instance, Mapping)
    })
    if classes != expected_classes:
        return None

    census_present = [field in pending_row for field in CLASS_CENSUS_FIELDS]
    if any(census_present) and not all(census_present):
        return None
    census = None
    if all(census_present):
        census = {
            field: _clone_jsonish(pending_row[field])
            for field in CLASS_CENSUS_FIELDS
        }
        for field in OPTIONAL_CLASS_CENSUS_FIELDS:
            if field in pending_row:
                census[field] = _clone_jsonish(pending_row[field])
    return {
        "frame_identity": _clone_jsonish(identity),
        "confuser_instances": _clone_jsonish(confusers),
        "class_census": census,
        "positive_scan_complete": True,
        "source": FULL_CURRENT_CAMERA_SOURCE,
    }
