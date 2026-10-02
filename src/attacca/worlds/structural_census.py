from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from attacca.worlds import ore_classes as OC


SUPPORTED_CLASSES = tuple(OC.MINE_TARGET_POOL)
EVAL_HELDOUT_CLASSES = tuple(OC.MINE_HELDOUT_EVAL_POOL)
FACE_NEIGHBOURS = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0),
    (0, -1, 0), (0, 0, 1), (0, 0, -1),
)

RECOGNITION_HORIZONTAL_CELL_RADIUS = 192
RECOGNITION_VERTICAL_CELL_RADIUS = 192

CENSUS_SCRUB_VERSION = 1
CENSUS_SCRUB_CLASSES = tuple(OC.MINE_POC_ORE_POOL)
CENSUS_SCRUB_MIN_FULL_CUBE_VISIBLE_AREA_PX = 150
CENSUS_SCRUB_VIEWPORT_PX = (640, 360)
CENSUS_SCRUB_FOV_DEG = 70.0
_SCRUB_FOCAL_PX = (CENSUS_SCRUB_VIEWPORT_PX[0] / 2.0) / math.tan(
    math.radians(CENSUS_SCRUB_FOV_DEG / 2.0))
_SCRUB_CORNER_SEC = math.sqrt(
    1.0 + (CENSUS_SCRUB_VIEWPORT_PX[0] / 2.0 / _SCRUB_FOCAL_PX) ** 2
    + (CENSUS_SCRUB_VIEWPORT_PX[1] / 2.0 / _SCRUB_FOCAL_PX) ** 2)
CENSUS_SCRUB_SINGLE_FACE_RADIUS = _SCRUB_FOCAL_PX / math.sqrt(
    CENSUS_SCRUB_MIN_FULL_CUBE_VISIBLE_AREA_PX)
CENSUS_SCRUB_RECOGNITION_RADIUS = int(math.ceil(
    1.0 + _SCRUB_FOCAL_PX * math.sqrt(
        (_SCRUB_CORNER_SEC ** 3) * math.sqrt(3.0)
        / CENSUS_SCRUB_MIN_FULL_CUBE_VISIBLE_AREA_PX)))
CENSUS_SCRUB_MIN_HALF_EXTENT = 48
CENSUS_SCRUB_FILL_VOLUME_LIMIT = 32768


def census_scrub_region(
        roam_box: Sequence[int], extra_cells: Iterable[Sequence[int]] = (), *,
        world_min_y: int = 0, world_max_y_exclusive: int = 256,
) -> dict[str, Any]:
    box9 = _normalise_box(roam_box, label9="roam_box")
    cells9 = [tuple(int(v9) for v9 in c9) for c9 in extra_cells]
    if any(len(c9) != 3 for c9 in cells9):
        raise ValueError("census scrub extra cell must be xyz")
    x09 = min([box9[0]] + [c9[0] for c9 in cells9])
    x19 = max([box9[1]] + [c9[0] + 1 for c9 in cells9])
    y09 = min([box9[2]] + [c9[1] for c9 in cells9])
    y19 = max([box9[3]] + [c9[1] + 2 for c9 in cells9])
    z09 = min([box9[4]] + [c9[2] for c9 in cells9])
    z19 = max([box9[5]] + [c9[2] + 1 for c9 in cells9])
    radius9 = int(CENSUS_SCRUB_RECOGNITION_RADIUS)
    pad9 = radius9 + 3
    def _floor9(lo9, hi9):
        half9 = (hi9 - lo9) / 2.0
        need9 = CENSUS_SCRUB_MIN_HALF_EXTENT - (half9 + pad9)
        return int(math.ceil(need9)) if need9 > 0 else 0
    grow_x9 = _floor9(x09, x19)
    grow_z9 = _floor9(z09, z19)
    region9 = (
        x09 - pad9 - grow_x9, x19 + pad9 + grow_x9,
        max(int(world_min_y), y09 - pad9),
        min(int(world_max_y_exclusive), y19 + pad9),
        z09 - pad9 - grow_z9, z19 + pad9 + grow_z9,
    )
    if region9[2] >= region9[3]:
        raise ValueError("census scrub region does not intersect world height")
    return {
        "version": int(CENSUS_SCRUB_VERSION),
        "region_box": list(region9),
        "coordinate_system": "absolute_xyz_half_open",
        "roam_box": list(box9),
        "recognition_radius_cells": radius9,
        "world_y_limits": [int(world_min_y), int(world_max_y_exclusive)],
        "radius_justification": {
            "recognition_gate": "v7_full_cube_150_6_no_fill",
            "min_full_cube_visible_area_px": int(
                CENSUS_SCRUB_MIN_FULL_CUBE_VISIBLE_AREA_PX),
            "viewport_px": list(CENSUS_SCRUB_VIEWPORT_PX),
            "fov_deg": float(CENSUS_SCRUB_FOV_DEG),
            "focal_px_upper_bound": round(float(_SCRUB_FOCAL_PX), 3),
            "single_face_on_axis_radius": round(
                float(CENSUS_SCRUB_SINGLE_FACE_RADIUS), 3),
            "corner_silhouette_factor": round(math.sqrt(3.0), 6),
            "viewport_corner_area_magnification": round(
                float(_SCRUB_CORNER_SEC ** 3), 6),
            "min_half_extent_floor": int(CENSUS_SCRUB_MIN_HALF_EXTENT),
        },
    }


def fill_slab_tiling(
        box: Sequence[int], *,
        volume_limit: int = CENSUS_SCRUB_FILL_VOLUME_LIMIT,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    box9 = _normalise_box(box, label9="fill box")
    limit9 = int(volume_limit)
    if limit9 < 1:
        raise ValueError("fill volume limit must be positive")
    w9 = box9[1] - box9[0]
    h9 = box9[3] - box9[2]
    d9 = box9[5] - box9[4]
    best9 = None
    for nx9 in range(1, 9):
        for nz9 in range(1, 9):
            wx9 = -(-w9 // nx9)
            wz9 = -(-d9 // nz9)
            if wx9 * wz9 > limit9:
                continue
            slab9 = max(1, limit9 // (wx9 * wz9))
            count9 = nx9 * nz9 * (-(-h9 // slab9))
            key9 = (count9, nx9 * nz9)
            if best9 is None or key9 < best9[0]:
                best9 = (key9, nx9, nz9, slab9)
    if best9 is None:
        raise ValueError(
            "fill tiling cannot satisfy the volume limit with an 8x8 xz split")
    _, nx9, nz9, slab9 = best9
    x_edges9 = [box9[0] + (w9 * i9) // nx9 for i9 in range(nx9 + 1)]
    z_edges9 = [box9[4] + (d9 * k9) // nz9 for k9 in range(nz9 + 1)]
    y_edges9 = list(range(box9[2], box9[3], slab9)) + [box9[3]]
    out9 = []
    for xa9, xb9 in zip(x_edges9, x_edges9[1:]):
        for za9, zb9 in zip(z_edges9, z_edges9[1:]):
            for ya9, yb9 in zip(y_edges9, y_edges9[1:]):
                if xa9 < xb9 and ya9 < yb9 and za9 < zb9:
                    if (xb9 - xa9) * (yb9 - ya9) * (zb9 - za9) > limit9:
                        raise AssertionError("fill tiling exceeded volume limit")
                    out9.append((xa9, xb9, ya9, yb9, za9, zb9))
    return tuple(out9)


def validate_census_scrub_witness(witness: Mapping[str, Any]) -> None:
    if not isinstance(witness, Mapping):
        raise ValueError("census scrub witness must be a mapping")
    if int(witness.get("version", -1)) != CENSUS_SCRUB_VERSION:
        raise ValueError("census scrub witness has an unsupported version")
    classes9 = tuple(str(kind9) for kind9 in (witness.get("classes") or ()))
    if not classes9 or not set(classes9) <= set(
            SUPPORTED_CLASSES + EVAL_HELDOUT_CLASSES):
        raise ValueError("census scrub classes must be a nonempty roster subset")
    if len(classes9) != len(set(classes9)):
        raise ValueError("census scrub classes must be unique")
    _normalise_box(witness.get("region_box") or (), label9="scrub region_box")
    radius9 = int(witness.get("recognition_radius_cells", 0))
    if radius9 < CENSUS_SCRUB_RECOGNITION_RADIUS:
        raise ValueError(
            "census scrub radius below the certified recognition bound")
    if int(witness.get("fill_command_count", 0)) < 1:
        raise ValueError("census scrub witness lacks executed fill commands")
    central9 = witness.get("post_scrub_central_audit")
    if (not isinstance(central9, Mapping)
            or int(central9.get("remaining_natural_roster_cells", -1)) != 0):
        raise ValueError(
            "census scrub central audit missing or found surviving natural ore")
    samples9 = witness.get("outer_audit_samples")
    ok_samples9 = [s9 for s9 in (samples9 or ())
                   if isinstance(s9, Mapping) and s9.get("converged") is True]
    if len(ok_samples9) < 2:
        raise ValueError("census scrub outer audit needs >=2 converged samples")
    if any(int(s9.get("roster_ore_cells", -1)) != 0 for s9 in ok_samples9):
        raise ValueError("census scrub outer audit found surviving natural ore")


def _cell_beyond_recognition(eye9, cell9, radius9):
    dx9 = max(cell9[0] - eye9[0], eye9[0] - (cell9[0] + 1), 0.0)
    dy9 = max(cell9[1] - eye9[1], eye9[1] - (cell9[1] + 1), 0.0)
    dz9 = max(cell9[2] - eye9[2], eye9[2] - (cell9[2] + 1), 0.0)
    return dx9 * dx9 + dy9 * dy9 + dz9 * dz9 > float(radius9) ** 2


def scrub_certified_absent_kinds(
        witness: Mapping[str, Any], eye: Sequence[float], *,
        staged_cells_by_kind: Mapping[str, Sequence[Sequence[int]]],
        cell_known: Callable[[tuple[int, int, int]], bool],
) -> tuple[frozenset, dict[str, Any]]:
    audit9 = {"scrub_active": True, "eye_ball_contained": False,
              "blocked_kinds": {}}
    try:
        validate_census_scrub_witness(witness)
    except ValueError as exc9:
        audit9["witness_invalid"] = str(exc9)
        return frozenset(), audit9
    eye9 = tuple(float(v9) for v9 in eye)
    if len(eye9) != 3 or not all(math.isfinite(v9) for v9 in eye9):
        audit9["witness_invalid"] = "camera eye must be finite xyz"
        return frozenset(), audit9
    region9 = tuple(int(v9) for v9 in witness["region_box"])
    radius9 = float(witness["recognition_radius_cells"])
    wy09, wy19 = (int(v9) for v9 in (
        witness.get("world_y_limits") or (0, 256)))
    contained9 = (
        region9[0] <= eye9[0] - radius9 and eye9[0] + radius9 <= region9[1]
        and (region9[2] <= eye9[1] - radius9 or region9[2] <= wy09)
        and (eye9[1] + radius9 <= region9[3] or region9[3] >= wy19)
        and region9[4] <= eye9[2] - radius9 and eye9[2] + radius9 <= region9[5])
    audit9["eye_ball_contained"] = bool(contained9)
    if not contained9:
        return frozenset(), audit9
    certified9 = set()
    for kind9 in witness["classes"]:
        blocked9 = None
        for raw9 in (staged_cells_by_kind.get(kind9) or ()):
            cell9 = tuple(int(v9) for v9 in raw9)
            if _cell_beyond_recognition(eye9, cell9, radius9):
                continue
            if not cell_known(cell9):
                blocked9 = f"staged_cell_unknown:{cell9[0]}:{cell9[1]}:{cell9[2]}"
                break
        if blocked9 is not None:
            audit9["blocked_kinds"][kind9] = blocked9
            continue
        certified9.add(str(kind9))
    return frozenset(certified9), audit9


def bare_kind(raw: Any) -> str:
    return str(raw).split(":")[-1].split(" ")[0].strip()


def validate_roster(roster: Sequence[str] = SUPPORTED_CLASSES) -> tuple[str, ...]:
    roster9 = tuple(str(kind9) for kind9 in roster)
    if not roster9 or len(roster9) != len(set(roster9)):
        raise ValueError("structural census roster must be nonempty and unique")
    if roster9 not in (
            SUPPORTED_CLASSES,
            SUPPORTED_CLASSES + EVAL_HELDOUT_CLASSES):
        raise ValueError(
            "structural census roster/order must equal OC.MINE_TARGET_POOL or "
            "its explicit held-out evaluation extension")
    return roster9


def index_supported_cells(
        grid: Mapping[tuple[int, int, int], Any],
        roster: Sequence[str] = SUPPORTED_CLASSES,
) -> dict[str, tuple[tuple[int, int, int], ...]]:
    roster9 = validate_roster(roster)
    mutable9: dict[str, list[tuple[int, int, int]]] = {
        kind9: [] for kind9 in roster9}
    supported9 = frozenset(roster9)
    for raw_cell9, raw_kind9 in grid.items():
        kind9 = bare_kind(raw_kind9)
        if kind9 not in supported9:
            continue
        if not isinstance(raw_cell9, (tuple, list)) or len(raw_cell9) != 3:
            raise ValueError("occupancy grid cell must be an xyz triple")
        mutable9[kind9].append(tuple(int(value9) for value9 in raw_cell9))
    return {
        kind9: tuple(sorted(mutable9[kind9]))
        for kind9 in roster9
    }


def index_supported_cells_cached(
        occupancy: Any,
        roster: Sequence[str] = SUPPORTED_CLASSES,
) -> tuple[dict[str, tuple[tuple[int, int, int], ...]], dict[str, int]]:
    roster9 = validate_roster(roster)
    grid9 = getattr(occupancy, "grid", None)
    if not isinstance(grid9, Mapping):
        raise TypeError("cached structural index requires occupancy.grid mapping")
    raw_revision9 = getattr(occupancy, "revision", None)
    if raw_revision9 is None:
        raise TypeError("cached structural index requires tracked occupancy revision")
    revision9 = int(raw_revision9)
    cache_key9 = (id(grid9), revision9, roster9)
    cache9 = getattr(occupancy, "_structural_class_index_cache", None)
    cache_hit9 = bool(
        isinstance(cache9, dict) and cache9.get("key") == cache_key9)
    if not cache_hit9:
        index9 = index_supported_cells(grid9, roster9)
        cache9 = {"key": cache_key9, "index": dict(index9)}
        occupancy._structural_class_index_cache = cache9
    return dict(cache9["index"]), {
        "occupancy_grid_index_passes": int(not cache_hit9),
        "index_cache_hit": int(cache_hit9),
        "occupancy_revision": revision9,
    }


def has_known_potentially_visible_face(
        cell: Sequence[int], *, known: Callable[[tuple[int, int, int]], bool],
        solid: Callable[[tuple[int, int, int]], bool],
        visual_extra_solid: Iterable[tuple[int, int, int]] = (),
) -> bool:
    cell9 = tuple(int(value9) for value9 in cell)
    if len(cell9) != 3:
        raise ValueError("structural candidate cell must be xyz")
    extra9 = frozenset(tuple(int(value9) for value9 in raw9)
                       for raw9 in visual_extra_solid)
    for dx9, dy9, dz9 in FACE_NEIGHBOURS:
        neighbour9 = (cell9[0] + dx9, cell9[1] + dy9, cell9[2] + dz9)
        if (known(neighbour9) and not solid(neighbour9)
                and neighbour9 not in extra9):
            return True
    return False


def recognition_domain_box(
        player_position: Sequence[float], *, world_min_y: int = 0,
        world_max_y_exclusive: int = 256,
) -> tuple[int, int, int, int, int, int]:
    if len(player_position) != 3:
        raise ValueError("player_position must be xyz")
    position9 = tuple(float(value9) for value9 in player_position)
    if not all(math.isfinite(value9) for value9 in position9):
        raise ValueError("player_position must be finite")
    px9, py9, pz9 = (int(math.floor(value9)) for value9 in position9)
    hr9 = RECOGNITION_HORIZONTAL_CELL_RADIUS
    vr9 = RECOGNITION_VERTICAL_CELL_RADIUS
    y09 = max(int(world_min_y), py9 - vr9)
    y19 = min(int(world_max_y_exclusive), py9 + vr9 + 1)
    if y19 <= y09:
        raise ValueError("recognition domain does not intersect world height")
    return (px9 - hr9, px9 + hr9 + 1,
            y09, y19,
            pz9 - hr9, pz9 + hr9 + 1)


def _normalise_box(raw9: Sequence[int], *, label9: str) -> tuple[int, ...]:
    if not isinstance(raw9, (tuple, list)) or len(raw9) != 6:
        raise ValueError(f"{label9} must be an xyz half-open six-tuple")
    box9 = tuple(int(value9) for value9 in raw9)
    if not (box9[0] < box9[1] and box9[2] < box9[3]
            and box9[4] < box9[5]):
        raise ValueError(f"{label9} must have positive extent")
    return box9


def _clip_box(box9: tuple[int, ...], domain9: tuple[int, ...]):
    clipped9 = (
        max(box9[0], domain9[0]), min(box9[1], domain9[1]),
        max(box9[2], domain9[2]), min(box9[3], domain9[3]),
        max(box9[4], domain9[4]), min(box9[5], domain9[5]),
    )
    if (clipped9[0] >= clipped9[1] or clipped9[2] >= clipped9[3]
            or clipped9[4] >= clipped9[5]):
        return None
    return clipped9


def exact_box_union_covers(
        domain_box: Sequence[int], known_boxes: Iterable[Sequence[int]]) -> bool:
    domain9 = _normalise_box(domain_box, label9="domain_box")
    clipped9 = []
    for index9, raw9 in enumerate(known_boxes):
        box9 = _normalise_box(raw9, label9=f"known_boxes[{index9}]")
        value9 = _clip_box(box9, domain9)
        if value9 is not None:
            clipped9.append(value9)
    if not clipped9:
        return False
    if any(
            box9[0] <= domain9[0] and box9[1] >= domain9[1]
            and box9[2] <= domain9[2] and box9[3] >= domain9[3]
            and box9[4] <= domain9[4] and box9[5] >= domain9[5]
            for box9 in clipped9):
        return True

    x_edges9 = sorted({domain9[0], domain9[1], *(
        value9 for box9 in clipped9 for value9 in box9[:2])})
    for x09, x19 in zip(x_edges9, x_edges9[1:]):
        if x19 <= x09:
            continue
        active_x9 = [box9 for box9 in clipped9
                     if box9[0] <= x09 and box9[1] >= x19]
        if not active_x9:
            return False
        y_edges9 = sorted({domain9[2], domain9[3], *(
            value9 for box9 in active_x9 for value9 in box9[2:4])})
        for y09, y19 in zip(y_edges9, y_edges9[1:]):
            if y19 <= y09:
                continue
            intervals9 = sorted(
                (box9[4], box9[5]) for box9 in active_x9
                if box9[2] <= y09 and box9[3] >= y19)
            if not intervals9:
                return False
            cursor9 = domain9[4]
            for z09, z19 in intervals9:
                if z19 <= cursor9:
                    continue
                if z09 > cursor9:
                    return False
                cursor9 = max(cursor9, z19)
                if cursor9 >= domain9[5]:
                    break
            if cursor9 < domain9[5]:
                return False
    return True


def exact_scan_coverage_witness(
        player_position: Sequence[float], known_boxes: Iterable[Sequence[int]],
) -> dict[str, Any]:
    domain9 = recognition_domain_box(player_position)
    boxes9 = [tuple(int(value9) for value9 in box9) for box9 in known_boxes]
    intersecting9 = [box9 for box9 in boxes9
                     if _clip_box(_normalise_box(
                         box9, label9="known_box"), domain9) is not None]
    complete9 = exact_box_union_covers(domain9, intersecting9)
    return {
        "version": 1,
        "complete": bool(complete9),
        "negative_supervision_available": bool(complete9),
        "method": "exact_x_slab_y_slab_merged_z_interval_union",
        "coordinate_system": "absolute_xyz_half_open",
        "recognition_domain_box": list(domain9),
        "recognition_horizontal_cell_radius": int(
            RECOGNITION_HORIZONTAL_CELL_RADIUS),
        "recognition_vertical_cell_radius": int(
            RECOGNITION_VERTICAL_CELL_RADIUS),
        "known_box_count": int(len(boxes9)),
        "intersecting_known_box_count": int(len(intersecting9)),
        "incomplete_reason": (None if complete9 else
                              "known_box_union_has_holes_or_insufficient_extent"),
        "corners_are_not_a_coverage_witness": True,
        "absence_policy": (
            "known_negative_only_if_complete_else_unknown"),
    }


def visibility_bits(
        known_kinds: Iterable[str], visible_kinds: Iterable[str],
        roster: Sequence[str] = SUPPORTED_CLASSES,
) -> tuple[int, int]:
    roster9 = validate_roster(roster)
    known9, visible9 = set(known_kinds), set(visible_kinds)
    supported9 = set(roster9)
    if not known9 <= supported9 or not visible9 <= supported9:
        raise ValueError("visibility bitsets contain a class outside the roster")
    if not visible9 <= known9:
        raise ValueError("visible classes must be a subset of known classes")
    known_bits9 = visible_bits9 = 0
    for index9, kind9 in enumerate(roster9):
        if kind9 in known9:
            known_bits9 |= 1 << index9
        if kind9 in visible9:
            visible_bits9 |= 1 << index9
    return int(known_bits9), int(visible_bits9)


def class_state(
        kind: str, known_bits: int, visible_bits: int,
        roster: Sequence[str] = SUPPORTED_CLASSES,
) -> tuple[int, int]:
    roster9 = validate_roster(roster)
    if str(kind) not in roster9:
        raise ValueError(f"class outside structural census roster: {kind!r}")
    mask9 = 1 << roster9.index(str(kind))
    known9 = int(bool(int(known_bits) & mask9))
    visible9 = int(bool(int(visible_bits) & mask9))
    if visible9 and not known9:
        raise ValueError("forbidden UNKNOWN+visible structural census state")
    return known9, visible9


def _recognizable_instance(instance9: Mapping[str, Any]) -> bool:
    proof9 = instance9.get("proof")
    return bool(
        int(instance9.get("visible", 0)) == 1
        and isinstance(proof9, Mapping)
        and proof9.get("oracle_recognizable") is True
        and isinstance(instance9.get("visible_point"), list)
        and len(instance9["visible_point"]) == 3)


def build_supported_class_census(
        kind_to_cells: Mapping[str, Sequence[tuple[int, int, int]]], *,
        recognize_instance: Callable[[str, tuple[int, int, int]],
                                     Mapping[str, Any] | None],
        candidate_filter: Callable[[str, tuple[int, int, int]], bool] | None = None,
        rank_key: Callable[[tuple[int, int, int]], Any] | None = None,
        preconfirmed_instances: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        exhaustive_kinds: Iterable[str] = (),
        recognizable_instance_sink: Callable[
            [str, Mapping[str, Any]], None] | None = None,
        coverage_witness: Mapping[str, Any] | None = None,
        certified_absent_kinds: Iterable[str] = (),
        roster: Sequence[str] = SUPPORTED_CLASSES,
) -> dict[str, Any]:
    started9 = time.perf_counter()
    roster9 = validate_roster(roster)
    if set(kind_to_cells) != set(roster9):
        raise ValueError("kind_to_cells must have exactly the supported roster keys")
    preconfirmed9 = dict(preconfirmed_instances or {})
    if not set(preconfirmed9) <= set(roster9):
        raise ValueError("preconfirmed instance class outside roster")
    certified9 = {str(kind9) for kind9 in (certified_absent_kinds or ())}
    if not certified9 <= set(roster9):
        raise ValueError("certified-absent class outside structural census roster")
    exhaustive9 = {str(kind9) for kind9 in (exhaustive_kinds or ())}
    if not exhaustive9 <= set(roster9):
        raise ValueError("exhaustive class outside structural census roster")
    witness9 = dict(coverage_witness or {})
    if certified9:
        witness9["scrub_certified_absent_classes"] = sorted(certified9)
    complete9 = witness9.get("complete") is True
    rank9 = rank_key or (lambda cell9: cell9)
    records9: list[dict[str, Any]] = []
    visible_kinds9: set[str] = set()
    rasterized9 = 0
    candidate9 = 0
    examined9 = 0
    filtered9 = 0
    unrecognizable9 = 0
    recognizable9 = 0
    exhaustive_recognizable9 = 0
    scanned_classes9 = 0
    preconfirmed_classes9 = 0
    preconfirmed_instances9 = 0
    empty_preconfirmation_fallback_classes9 = 0
    ordering_ns9 = 0
    filter_ns9 = 0
    recognize_ns9 = 0

    for kind9 in roster9:
        confirmed9 = preconfirmed9.get(kind9)
        if confirmed9:
            preconfirmed_classes9 += 1
            for raw9 in confirmed9:
                instance9 = dict(raw9)
                instance9["kind"] = kind9
                if not _recognizable_instance(instance9):
                    raise ValueError(
                        f"preconfirmed {kind9} instance lacks strict recognizable proof")
                records9.append(instance9)
                preconfirmed_instances9 += 1
                if recognizable_instance_sink is not None:
                    recognizable_instance_sink(kind9, instance9)
            visible_kinds9.add(kind9)
            continue
        if confirmed9 is not None:
            empty_preconfirmation_fallback_classes9 += 1

        scanned_classes9 += 1
        ordering_started9 = time.perf_counter_ns()
        ordered9 = sorted(
            (tuple(int(value9) for value9 in cell9)
             for cell9 in kind_to_cells[kind9]),
            key=lambda cell9: (rank9(cell9), cell9))
        ordering_ns9 += time.perf_counter_ns() - ordering_started9
        candidate9 += len(ordered9)
        stored_nearest9 = False
        for cell9 in ordered9:
            examined9 += 1
            if candidate_filter is not None:
                filter_started9 = time.perf_counter_ns()
                accepted9 = bool(candidate_filter(kind9, cell9))
                filter_ns9 += time.perf_counter_ns() - filter_started9
                if not accepted9:
                    filtered9 += 1
                    continue
            rasterized9 += 1
            recognize_started9 = time.perf_counter_ns()
            raw9 = recognize_instance(kind9, cell9)
            recognize_ns9 += time.perf_counter_ns() - recognize_started9
            if raw9 is None:
                unrecognizable9 += 1
                continue
            instance9 = dict(raw9)
            instance9["kind"] = kind9
            if not _recognizable_instance(instance9):
                unrecognizable9 += 1
                continue
            instance9["chosen"] = False
            if not stored_nearest9:
                records9.append(instance9)
                stored_nearest9 = True
            visible_kinds9.add(kind9)
            recognizable9 += 1
            if recognizable_instance_sink is not None:
                recognizable_instance_sink(kind9, instance9)
            if kind9 not in exhaustive9:
                break
            exhaustive_recognizable9 += 1

    known_kinds9 = (set(roster9) if complete9
                    else set(visible_kinds9) | certified9)
    known_bits9, visible_bits9 = visibility_bits(
        known_kinds9, visible_kinds9, roster9)
    return {
        "class_visibility_known_bits": int(known_bits9),
        "class_visibility_visible_bits": int(visible_bits9),
        "class_visible_instances": records9,
        "class_visibility_scan_witness": witness9,
        "class_census_runtime": {
            "census_ms": round(1000.0 * (time.perf_counter() - started9), 3),
            "indexed_cell_count": int(sum(
                len(kind_to_cells[kind9]) for kind9 in roster9)),
            "candidate_cell_count": int(candidate9),
            "examined_candidate_count": int(examined9),
            "candidate_filter_rejected_count": int(filtered9),
            "rasterized_candidate_count": int(rasterized9),
            "unrecognizable_rasterized_candidate_count": int(unrecognizable9),
            "recognizable_candidate_count": int(recognizable9),
            "exhaustive_recognizable_candidate_count": int(
                exhaustive_recognizable9),
            "exhaustive_class_count": int(len(exhaustive9)),
            "scanned_class_count": int(scanned_classes9),
            "preconfirmed_class_count": int(preconfirmed_classes9),
            "preconfirmed_instance_count": int(preconfirmed_instances9),
            "empty_preconfirmation_fallback_class_count": int(
                empty_preconfirmation_fallback_classes9),
            "candidate_ordering_ms": round(ordering_ns9 / 1_000_000.0, 3),
            "candidate_filter_ms": round(filter_ns9 / 1_000_000.0, 3),
            "recognize_callback_ms": round(recognize_ns9 / 1_000_000.0, 3),
            "sparse_instance_policy": (
                "all_preconfirmed_goal_plus_nearest_per_other_visible_class;"
                "exhaustive_recognizable_sink_for_configured_classes"
                if exhaustive9 else
                "all_preconfirmed_goal_plus_nearest_per_other_visible_class"),
            "occupancy_grid_index_passes": 1,
        },
    }


def validate_census(
        census: Mapping[str, Any], *, goal_kind: str | None = None,
        goal_class_exist: bool | None = None,
        roster: Sequence[str] = SUPPORTED_CLASSES,
) -> None:
    roster9 = validate_roster(roster)
    mask9 = (1 << len(roster9)) - 1
    known9 = int(census.get("class_visibility_known_bits", 0))
    visible9 = int(census.get("class_visibility_visible_bits", 0))
    if known9 < 0 or visible9 < 0 or known9 & ~mask9 or visible9 & ~mask9:
        raise ValueError("structural census bitset is outside roster width")
    if visible9 & ~known9:
        raise ValueError("visible structural census bit is not known")
    instances9 = census.get("class_visible_instances")
    if not isinstance(instances9, list):
        raise ValueError("class_visible_instances must be a list")
    instance_kinds9 = set()
    ids9 = set()
    for index9, raw9 in enumerate(instances9):
        if not isinstance(raw9, Mapping):
            raise ValueError(f"class_visible_instances[{index9}] is not a mapping")
        kind9 = str(raw9.get("kind") or "")
        if kind9 not in roster9 or not _recognizable_instance(raw9):
            raise ValueError(f"class_visible_instances[{index9}] is not recognizable")
        instance_id9 = str(raw9.get("instance_id") or "")
        if not instance_id9 or instance_id9 in ids9:
            raise ValueError("class census instance IDs must be nonempty and unique")
        ids9.add(instance_id9)
        instance_kinds9.add(kind9)
    _, instance_visible_bits9 = visibility_bits(
        instance_kinds9, instance_kinds9, roster9)
    if instance_visible_bits9 != visible9:
        raise ValueError("class visible bits disagree with sparse instance kinds")
    unions9 = census.get("class_union_masks")
    if unions9 is not None:
        if not isinstance(unions9, list):
            raise ValueError("class_union_masks must be a list")
        union_kinds9 = set()
        for index9, raw9 in enumerate(unions9):
            if not isinstance(raw9, Mapping):
                raise ValueError(f"class_union_masks[{index9}] is not a mapping")
            kind9 = str(raw9.get("kind") or "")
            if kind9 not in roster9 or kind9 in union_kinds9:
                raise ValueError("class union kinds must be unique roster classes")
            if (int(raw9.get("visible", 0)) != 1
                    or not isinstance(raw9.get("visible_point"), list)
                    or len(raw9["visible_point"]) != 3):
                raise ValueError(
                    f"class_union_masks[{index9}] lacks visible support")
            member_count9 = raw9.get("member_instance_count")
            if (isinstance(member_count9, bool)
                    or not isinstance(member_count9, int)
                    or member_count9 < 1):
                raise ValueError("class union member count must be a positive integer")
            proof9 = raw9.get("proof")
            proof_member_count9 = (
                None if not isinstance(proof9, Mapping) else
                proof9.get("class_union_member_count"))
            if (not isinstance(proof9, Mapping)
                    or proof9.get("class_union_complete_enumeration") is not True
                    or isinstance(proof_member_count9, bool)
                    or not isinstance(proof_member_count9, int)
                    or proof_member_count9 != member_count9):
                raise ValueError("class union lacks exhaustive enumeration proof")
            union_kinds9.add(kind9)
        if union_kinds9 != instance_kinds9:
            raise ValueError(
                "class union kinds disagree with visible sparse instance kinds")
    witness9 = census.get("class_visibility_scan_witness")
    if not isinstance(witness9, Mapping):
        raise ValueError("class visibility scan witness must be a mapping")
    if witness9.get("complete") is not True and known9 != visible9:
        certified9 = witness9.get("scrub_certified_absent_classes")
        if (not isinstance(certified9, (list, tuple))
                or not set(map(str, certified9)) <= set(roster9)):
            raise ValueError("partial census may know positives only")
        certified_bits9, _ = visibility_bits(
            set(map(str, certified9)), (), roster9)
        if known9 & ~(visible9 | certified_bits9):
            raise ValueError("partial census may know positives only")
    if goal_kind is not None:
        goal9 = str(goal_kind)
        known_goal9, visible_goal9 = class_state(
            goal9, known9, visible9, roster9)
        del known_goal9
        if goal_class_exist is not None and visible_goal9 != int(bool(goal_class_exist)):
            raise ValueError("goal class_exist disagrees with structural census visible bit")
