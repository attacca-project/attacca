"""Place human-play labels: identity binding of the used marker and three-way behavioral phases."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from attacca.worlds.human_viewmodel import decode_mask_runs_yx
from attacca.worlds.human_viewmodel import encode_mask_runs_yx


USE_HUMAN_LABEL_CONTRACT = "use_human_group_transaction_binding/v2"
USE_PHASE_NAMES = {0: "EXPLORE", 1: "APPROACH", 2: "INTERACT"}
MASK_SHAPE = (360, 640)


class UseHumanLabelError(ValueError):
    pass


def _binary(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    raise UseHumanLabelError(f"{label} must be binary")


def _instance_masks(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for index, raw in enumerate(row.get("target_instances") or []):
        if not isinstance(raw, Mapping):
            raise UseHumanLabelError(
                f"target_instances[{index}] must be a mapping")
        identity = raw.get("instance_id")
        if not isinstance(identity, str) or not identity:
            raise UseHumanLabelError(
                f"target_instances[{index}] lacks an identity")
        if identity in output:
            raise UseHumanLabelError(f"duplicate target identity {identity}")
        mask = decode_mask_runs_yx(
            list(raw.get("mask_runs_yx") or []), list(MASK_SHAPE))
        count = raw.get("pixel_count")
        if (isinstance(count, bool) or not isinstance(count, int)
                or int(count) != int(mask.sum())):
            raise UseHumanLabelError(
                f"target {identity} pixel count disagrees with its mask")
        output[identity] = {
            **dict(raw),
            "recognizable": _binary(
                raw.get("recognizable", 0),
                label=f"target {identity}.recognizable"),
            "mask": np.ascontiguousarray(mask, dtype=np.uint8),
        }
    return output


def _validate_placements(
        rows: Sequence[Mapping[str, Any]],
        placements: Sequence[Mapping[str, Any]],
        target_ids: set[str]) -> list[dict[str, Any]]:
    normalized = []
    seen_f = set()
    seen_ids = set()
    for index, raw in enumerate(placements):
        if not isinstance(raw, Mapping):
            raise UseHumanLabelError(f"placement[{index}] must be a mapping")
        try:
            frame = int(raw["f"])
            identity = str(raw["instance_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UseHumanLabelError(
                f"placement[{index}] is malformed") from exc
        if not 0 <= frame < len(rows) or frame in seen_f:
            raise UseHumanLabelError(
                f"placement[{index}] frame is invalid or duplicated")
        if identity not in target_ids or identity in seen_ids:
            raise UseHumanLabelError(
                f"placement[{index}] target identity is invalid or repeated")
        if raw.get("postcondition") != "exact_expected_cell_changed":
            raise UseHumanLabelError(
                f"placement[{index}] lacks the exact block postcondition")
        if raw.get("crosshair_precondition") != "exact_marker_cell_in_reach":
            raise UseHumanLabelError(
                f"placement[{index}] lacks the exact crosshair precondition")
        action = rows[frame].get("action") or {}
        if int(action.get("use", 0)) != 1:
            raise UseHumanLabelError(
                f"placement[{index}] is not aligned to a USE action")
        seen_f.add(frame)
        seen_ids.add(identity)
        normalized.append({**dict(raw), "f": frame, "instance_id": identity})
    normalized.sort(key=lambda value: value["f"])
    if [row["f"] for row in normalized] != sorted(seen_f):
        raise UseHumanLabelError("placement frames are not strictly ordered")
    return normalized


def derive_use_training_labels(
        raw_rows: Sequence[Mapping[str, Any]],
        placements: Sequence[Mapping[str, Any]], *,
        target_instance_ids: Sequence[str],
        target_group_by_instance: Mapping[str, int | str],
) -> dict[str, Any]:
    rows = [deepcopy(dict(row)) for row in raw_rows]
    target_ids = tuple(str(value) for value in target_instance_ids)
    if not rows:
        raise UseHumanLabelError("episode contains no recorded frames")
    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise UseHumanLabelError("target identities must be distinct")
    if set(target_group_by_instance) != set(target_ids):
        raise UseHumanLabelError(
            "target group mapping must cover every target identity exactly")
    group_by_id = {
        identity: str(target_group_by_instance[identity])
        for identity in target_ids}
    group_members: dict[str, set[str]] = {}
    for identity, group_id in group_by_id.items():
        group_members.setdefault(group_id, set()).add(identity)
    placement_rows = _validate_placements(rows, placements, set(target_ids))
    if {row["instance_id"] for row in placement_rows} != set(target_ids):
        raise UseHumanLabelError("not every target marker received an exact placement")
    if int(placement_rows[-1]["f"]) != len(rows) - 1:
        raise UseHumanLabelError("training rows continue after the final success")

    completed_by_group: dict[str, set[str]] = {
        group_id: set() for group_id in group_members}
    active_ledger_group = None
    placement_groups = []
    for placement in placement_rows:
        identity = placement["instance_id"]
        group_id = group_by_id[identity]
        if active_ledger_group is None:
            active_ledger_group = group_id
            placement_groups.append(group_id)
        elif group_id != active_ledger_group:
            if completed_by_group[active_ledger_group] != group_members[
                    active_ledger_group]:
                raise UseHumanLabelError(
                    "placement switched groups before completing the active group")
            if group_id in placement_groups:
                raise UseHumanLabelError(
                    "placement returned to an already-started group")
            active_ledger_group = group_id
            placement_groups.append(group_id)
        completed_by_group[group_id].add(identity)

    placement_f = {row["instance_id"]: int(row["f"])
                   for row in placement_rows}
    placement_at = {int(row["f"]): row for row in placement_rows}
    group_intervals = {
        group_id: (
            min(placement_f[identity] for identity in members),
            max(placement_f[identity] for identity in members),
        )
        for group_id, members in group_members.items()
        if members <= set(placement_f)
    }
    completed = set()
    committed = None
    switches = []
    group_binding_advances = []
    first_commit_f = {}
    output = []

    for frame, row in enumerate(rows):
        if int(row.get("f", frame)) != frame:
            raise UseHumanLabelError("raw rows must use contiguous zero-based f")
        instances = _instance_masks(row)
        unknown = set(instances) - set(target_ids)
        if unknown:
            raise UseHumanLabelError(
                f"frame {frame} contains unknown target identities {sorted(unknown)}")
        active = set(target_ids) - completed
        visible_recognizable = {
            identity for identity, value in instances.items()
            if identity in active and value["recognizable"]
            and int(value["mask"].sum()) > 0
        }
        future = {
            identity: success_f for identity, success_f in placement_f.items()
            if identity in active and success_f >= frame
        }
        interaction_group = next((
            group_id for group_id, (start_f, end_f) in group_intervals.items()
            if start_f <= frame <= end_f), None)
        candidates = visible_recognizable & set(future)
        best = (min(candidates, key=lambda identity: (future[identity], identity))
                if candidates else None)
        event_kind = None
        previous = committed
        group_advance = False
        if interaction_group is not None:
            start_f, _end_f = group_intervals[interaction_group]
            if (frame == start_f and (
                    committed is None
                    or group_by_id.get(committed) != interaction_group)):
                raise UseHumanLabelError(
                    f"group {interaction_group} reached its first success "
                    "without a preceding group-level APPROACH commitment")
            group_future = {
                identity: success_f for identity, success_f in future.items()
                if group_by_id[identity] == interaction_group}
            if not group_future:
                raise UseHumanLabelError(
                    f"group {interaction_group} has no remaining success")
            desired = min(
                group_future, key=lambda identity: (group_future[identity], identity))
            if committed != desired:
                committed = desired
                group_advance = True
                first_commit_f.setdefault(committed, frame)
                group_binding_advances.append({
                    "f": frame, "previous_instance_id": previous,
                    "instance_id": committed,
                    "interaction_group_id": interaction_group,
                })
        else:
            if committed is None and best is not None:
                committed = best
                event_kind = "seam_commit" if (
                    output and int(output[-1]["phase"]) == 2) else "commit"
                first_commit_f.setdefault(committed, frame)
            elif (committed is not None and best is not None
                  and future.get(best, 10 ** 12) < future.get(committed, 10 ** 12)):
                committed = best
                event_kind = "switch"
                first_commit_f.setdefault(committed, frame)
        if event_kind:
            switches.append({
                "f": frame, "event_kind": event_kind,
                "previous_instance_id": previous,
                "instance_id": committed,
            })

        union = np.zeros(MASK_SHAPE, np.uint8)
        for identity in visible_recognizable:
            union |= instances[identity]["mask"]
        fresh_class_recognizable = int(bool(union.any()))
        chosen = np.zeros(MASK_SHAPE, np.uint8)
        if committed is not None and committed in instances:
            chosen = instances[committed]["mask"].copy()
            union |= chosen
        if union.any() and committed is None:
            raise UseHumanLabelError(
                f"frame {frame} has union pixels without hindsight binding")
        if np.any((chosen > 0) & (union == 0)):
            raise UseHumanLabelError("chosen is not a subset of union")

        success = placement_at.get(frame)
        if interaction_group is not None:
            phase = 2
        elif success is not None:
            raise UseHumanLabelError("success lies outside its group interval")
        elif committed is not None or union.any():
            phase = 1
        else:
            phase = 0
        if output and int(output[-1]["phase"]) == 2 and phase == 0:
            event_kind = "seam"
            switches.append({
                "f": frame, "event_kind": event_kind,
                "previous_instance_id": previous,
                "instance_id": None,
            })
        if success is not None:
            if committed != success["instance_id"]:
                raise UseHumanLabelError(
                    f"frame {frame} success {success['instance_id']} differs "
                    f"from committed {committed}")

        normalized_instances = []
        for identity in sorted(instances):
            value = {key: item for key, item in instances[identity].items()
                     if key != "mask"}
            value["chosen"] = bool(identity == committed)
            normalized_instances.append(value)
        row.update({
            "contract": USE_HUMAN_LABEL_CONTRACT,
            "f": frame,
            "t": frame,
            "class_exist": fresh_class_recognizable,
            "class_recognizable": fresh_class_recognizable,
            "chosen_visible": int(bool(chosen.any())),
            "target_committed": int(committed is not None),
            "chosen_instance_id": committed,
            "union_mask_runs_yx": encode_mask_runs_yx(union),
            "union_mask_pixel_count": int(union.sum()),
            "chosen_mask_runs_yx": encode_mask_runs_yx(chosen),
            "chosen_mask_pixel_count": int(chosen.sum()),
            "phase": int(phase),
            "phase_id": int(phase),
            "phase_name": USE_PHASE_NAMES[phase],
            "interaction_exact_success": int(success is not None),
            "interaction_group_id": (
                group_by_id[committed] if committed is not None else None),
            "group_interaction_active": int(interaction_group is not None),
            "group_binding_advance": int(group_advance),
            "binding_event": event_kind,
            "visible_instances": normalized_instances,
            "bc_valid": 1,
            "post_success": 0,
            "terminal_tail": 0,
        })
        output.append(row)

        if success is not None:
            completed.add(success["instance_id"])
            committed = None

    phase_counts = {
        USE_PHASE_NAMES[value]: sum(int(row["phase"] == value) for row in output)
        for value in sorted(USE_PHASE_NAMES)
    }
    return {
        "contract": USE_HUMAN_LABEL_CONTRACT,
        "rows": output,
        "placements": placement_rows,
        "binding_events": switches,
        "group_binding_advances": group_binding_advances,
        "target_group_by_instance_id": dict(group_by_id),
        "group_intervals_inclusive": {
            group_id: [int(bounds[0]), int(bounds[1])]
            for group_id, bounds in group_intervals.items()},
        "first_commit_f_by_instance_id": first_commit_f,
        "phase_counts": phase_counts,
        "completed_instance_ids": sorted(completed),
        "all_targets_completed": completed == set(target_ids),
        "terminal_tail_frames_recorded": 0,
    }
