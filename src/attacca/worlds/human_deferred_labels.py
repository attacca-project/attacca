"""Simulator-free helpers that attach exact labels to buffered human-play frames after the play window closes."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSequence
import copy
import math
from typing import Any, Callable


HUMAN_DEFERRED_LABEL_CONTRACT = "human_postplay_exact_visible_surface/v1"
HUMAN_PIXEL_MARKER_CONTRACT = "human_frozen_rgb_pixel_marker/v1"


class DeferredHumanLabelError(ValueError):
    pass


def isolated_frozen_marker_instances(
        *, ctx: dict[str, Any], struct: dict[str, Any], traj: dict[str, Any],
        replay_occupancy: Any, source_story: dict[str, Any],
        refresh: Callable[[], None],
) -> list[dict[str, Any]]:
    if not isinstance(source_story, dict):
        raise DeferredHumanLabelError(
            "frozen marker source story must be a mutable mapping")
    grid = getattr(replay_occupancy, "grid", None)
    boxes = getattr(replay_occupancy, "boxes", None)
    if grid is None or boxes is None:
        raise DeferredHumanLabelError(
            "frozen marker replay occupancy is malformed")
    before_grid = dict(grid)
    before_boxes = copy.deepcopy(list(boxes))
    before_revision = getattr(replay_occupancy, "revision", None)
    original_occ = ctx.get("_struct_occ")
    original_offline = struct.get("offline_human_label_row")
    last_y_present = "_last_labeled_y" in traj
    original_last_y = traj.get("_last_labeled_y")
    try:
        ctx["_struct_occ"] = replay_occupancy
        struct["offline_human_label_row"] = source_story
        struct["visibility_cache"] = {}
        refresh()
        instances = copy.deepcopy(source_story.get("visible_instances", []))
        if not isinstance(instances, list) or any(
                not isinstance(record, dict) for record in instances):
            raise DeferredHumanLabelError(
                "frozen marker exact refresh emitted malformed instances")
    finally:
        ctx["_struct_occ"] = original_occ
        struct["offline_human_label_row"] = original_offline
        if last_y_present:
            traj["_last_labeled_y"] = original_last_y
        else:
            traj.pop("_last_labeled_y", None)
    if (dict(grid) != before_grid
            or list(boxes) != before_boxes
            or (before_revision is not None
                and getattr(replay_occupancy, "revision", None)
                    != before_revision)):
        raise DeferredHumanLabelError(
            "frozen marker refresh mutated frozen replay occupancy")
    return instances


class RisingEdgeLatch:

    def __init__(self) -> None:
        self._held = False
        self._pending = 0

    @property
    def pending(self) -> bool:
        return bool(self._pending > 0)

    def press(self) -> bool:
        if self._held:
            return False
        self._held = True
        self._pending += 1
        return True

    def release(self) -> None:
        self._held = False

    def consume(self) -> bool:
        if self._pending <= 0:
            return False
        self._pending -= 1
        return True


def window_to_raw_pixel(
        window_x: int | float,
        window_y: int | float,
        *,
        viewport_x: int = 0,
        viewport_y: int = 90,
        viewport_width: int = 640,
        viewport_height: int = 360,
        raw_width: int = 640,
        raw_height: int = 360,
) -> tuple[int, int] | None:
    numeric = (window_x, window_y, viewport_x, viewport_y,
               viewport_width, viewport_height, raw_width, raw_height)
    if any(isinstance(value, bool) for value in numeric):
        raise DeferredHumanLabelError("pixel transform contains a boolean")
    try:
        wx, wy = float(window_x), float(window_y)
        vx, vy = int(viewport_x), int(viewport_y)
        vw, vh = int(viewport_width), int(viewport_height)
        rw, rh = int(raw_width), int(raw_height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeferredHumanLabelError("pixel transform is malformed") from exc
    if vw <= 0 or vh <= 0 or rw <= 0 or rh <= 0:
        raise DeferredHumanLabelError("pixel transform dimensions must be positive")
    if not (vx <= wx < vx + vw and vy <= wy < vy + vh):
        return None
    display_x = int(wx - vx)
    display_y_from_top = vh - 1 - int(wy - vy)
    raw_x = min(rw - 1, (display_x * rw) // vw)
    raw_y = min(rh - 1, (display_y_from_top * rh) // vh)
    return int(raw_x), int(raw_y)


def selected_component_removed(
        selected_cells: Iterable[tuple[int, int, int]],
        remaining_goal_cells: Iterable[tuple[int, int, int]],
) -> bool:
    selected = frozenset(tuple(int(q) for q in cell)
                         for cell in selected_cells)
    if not selected:
        return False
    remaining = frozenset(tuple(int(q) for q in cell)
                          for cell in remaining_goal_cells)
    return selected.isdisjoint(remaining)


def persistent_nearest_choice(
        previous_id: str | None,
        previous_anchor_xz: tuple[float, float] | None,
        visible_candidates: Iterable[tuple[str, tuple[float, float]]],
        *,
        player_xz: tuple[float, float],
        closer_margin: float = 1.5,
) -> dict[str, Any]:
    try:
        px, pz = (float(q) for q in player_xz)
        margin = float(closer_margin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeferredHumanLabelError("persistent choice geometry is malformed") from exc
    if not (math.isfinite(px) and math.isfinite(pz)
            and math.isfinite(margin) and margin >= 0.0):
        raise DeferredHumanLabelError("persistent choice geometry is non-finite")
    candidates = []
    identities = set()
    for raw_id, raw_anchor in visible_candidates:
        identity = str(raw_id or "")
        if not identity or identity in identities:
            raise DeferredHumanLabelError(
                "persistent choice candidate IDs are missing or duplicated")
        identities.add(identity)
        try:
            ax, az = (float(q) for q in raw_anchor)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DeferredHumanLabelError(
                "persistent choice candidate anchor is malformed") from exc
        if not (math.isfinite(ax) and math.isfinite(az)):
            raise DeferredHumanLabelError(
                "persistent choice candidate anchor is non-finite")
        candidates.append((math.hypot(ax - px, az - pz), identity, (ax, az)))
    candidates.sort(key=lambda row: (row[0], row[1]))

    previous = None if previous_id is None else str(previous_id)
    previous_anchor = None
    if previous_anchor_xz is not None:
        try:
            previous_anchor = tuple(float(q) for q in previous_anchor_xz)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DeferredHumanLabelError(
                "persistent previous anchor is malformed") from exc
        if (len(previous_anchor) != 2
                or not all(math.isfinite(q) for q in previous_anchor)):
            raise DeferredHumanLabelError(
                "persistent previous anchor is non-finite")

    if previous is None:
        if not candidates:
            return {"chosen_id": None, "anchor_xz": None,
                    "transition": "", "chosen_visible": False}
        _distance, identity, anchor = candidates[0]
        return {"chosen_id": identity, "anchor_xz": anchor,
                "transition": "commit", "chosen_visible": True}

    same = next((row for row in candidates if row[1] == previous), None)
    if same is not None:
        previous_anchor = same[2]
    if previous_anchor is None:
        raise DeferredHumanLabelError(
            "persistent chosen ID has no remembered anchor")
    previous_distance = math.hypot(
        previous_anchor[0] - px, previous_anchor[1] - pz)
    if candidates:
        candidate_distance, candidate_id, candidate_anchor = candidates[0]
        if (candidate_id != previous
                and candidate_distance < previous_distance - margin):
            return {"chosen_id": candidate_id, "anchor_xz": candidate_anchor,
                    "transition": "switch", "chosen_visible": True}
    return {
        "chosen_id": previous,
        "anchor_xz": previous_anchor,
        "transition": "",
        "chosen_visible": same is not None,
    }


def human_marker_transition(
        previous_chosen_id: str | None,
        marked_chosen_id: str,
) -> str:
    chosen = str(marked_chosen_id or "")
    if not chosen:
        raise DeferredHumanLabelError("human marker chosen id is empty")
    if previous_chosen_id is None:
        return "human_commit"
    previous = str(previous_chosen_id)
    return "human_reaffirm" if previous == chosen else "human_switch"


def buffer_or_write_row(
        row: Mapping[str, Any] | None,
        *,
        deferred: bool,
        start_t: int | None,
        buffer: MutableSequence[dict[str, Any]],
        write: Callable[[Mapping[str, Any]], None],
) -> None:
    if row is None:
        return
    if not isinstance(row, Mapping) or isinstance(row.get("t"), bool):
        raise DeferredHumanLabelError("trajectory row is malformed")
    try:
        row_t = int(row["t"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DeferredHumanLabelError("trajectory row lacks an integer t") from exc
    if deferred:
        if start_t is None:
            raise DeferredHumanLabelError("deferred segment has no source start_t")
        if row_t >= int(start_t):
            if buffer and int(buffer[-1]["t"]) >= row_t:
                raise DeferredHumanLabelError("deferred trajectory order is not strict")
            buffer.append(dict(row))
            return
    write(row)


def source_action_index(rows: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    last_t: int | None = None
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise DeferredHumanLabelError("deferred trajectory contains a non-row")
        try:
            row_t = int(raw["t"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise DeferredHumanLabelError("deferred trajectory row lacks t") from exc
        if last_t is not None and row_t <= last_t:
            raise DeferredHumanLabelError("deferred trajectory order is not strict")
        last_t = row_t
        evidence = raw.get("human_preaction")
        if evidence is None:
            continue
        if not isinstance(evidence, Mapping):
            raise DeferredHumanLabelError("human_preaction is malformed")
        try:
            source_t = int(evidence["source_traj_t"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise DeferredHumanLabelError(
                "human_preaction lacks source_traj_t") from exc
        if source_t in out:
            raise DeferredHumanLabelError(
                f"two controls consume source RGB traj_t={source_t}")
        record = dict(evidence)
        record["action_row_t"] = row_t
        action = raw.get("action")
        if not isinstance(action, Mapping):
            raise DeferredHumanLabelError("human action row has no action mapping")
        record["action"] = dict(action)
        out[source_t] = record
    return out


def interaction_pressed(action: Mapping[str, Any]) -> bool:
    if not isinstance(action, Mapping):
        raise DeferredHumanLabelError("interaction action must be a mapping")
    for key in ("attack", "use"):
        value = action.get(key, 0)
        if isinstance(value, bool):
            value = int(value)
        try:
            if int(value):
                return True
        except (TypeError, ValueError, OverflowError) as exc:
            raise DeferredHumanLabelError(
                f"human action {key} is not scalar") from exc
    return False
