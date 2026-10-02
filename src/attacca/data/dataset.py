"""Training data from the human-play packs: schema-13 LMDB episode windows with actions, per-frame target masks, behavioral phases and goal images, and the multi-pack Lightning data module."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)

try:
    import lmdb
except ImportError:
    lmdb = None

from attacca.worlds import ore_classes
from attacca.worlds.entity_id_mask import HUNT_CLASS_VISIBILITY_ROSTER_CONTRACT
from attacca.worlds.use_human_scene import USE_CLASS_VISIBILITY_ROSTER_CONTRACT
from attacca.worlds.episode_schema import COMMITTED_CLASS_UNION_SUPPORT_CONTRACT
from attacca.worlds.episode_schema import COMMITTED_CLASS_VISIBLE_SURFACE_UNION_SEMANTIC
from attacca.worlds.episode_schema import COMMITTED_SURFACE_TRACKING_CONTRACT
from attacca.worlds.episode_schema import FRESH_SHORT_SIDE_6_SYNTHETIC_LMDB_SCHEMA_VERSION
from attacca.worlds.episode_schema import VISIBLE_SURFACE_RENDER_POSE_SOURCE
from attacca.worlds.episode_schema import class_visibility_roster_contract
from attacca.worlds.episode_schema import RECOGNITION_GATE_VERSION
from attacca.worlds.episode_schema import normalize_recognition_gate_version
from attacca.worlds.episode_schema import recognition_contract_gate_version
from attacca.worlds.episode_schema import require_matching_recognition_gate_version
from attacca.worlds.episode_schema import structural_surface_recognition_fields
from attacca.worlds.episode_schema import validate_positive_pair_episode_groups
from attacca.worlds.episode_schema import viewmodel_occlusion_contract


_LABEL_FRAME_FIELDS = frozenset({
    "phase", "class_exist", "chosen_visible", "centroid_valid",
    "centroid_phase_weight", "decision_weight", "event_kind", "bc_valid",
})
_BASE_FRAME_FIELDS = frozenset({
    "pov_jpeg", "buttons", "camera", "env_prev_action", "chosen_box",
    "chosen_mask_png", "chosen_mask_kind", "chosen_point", "chosen_instance_id",
    "verb_id", "tool_id", "visible_instances",
    "visible_instance_count", "class_visible_mask_png",
    "class_visible_mask_kind", "certified_mask_instance_count",
    "visibility_supervision_valid", "visibility_scan_scope",
})
_COMMITTED_SURFACE_FRAME_FIELDS = frozenset({
    "class_recognizable", "chosen_recognizable", "chosen_surface_visible",
    "target_committed", "committed_instance_id", "committed_surface_instance",
})
_COMMITTED_UNION_FRAME_FIELDS = frozenset({
    "class_union_support_contract",
    "class_union_committed_surface_added_pixels",
})
_CLASS_VISIBILITY_FRAME_FIELDS = frozenset({
    "class_visibility_known_bits", "class_visibility_visible_bits",
    "class_visible_instances", "class_visibility_geometry_valid",
})
_EVENT_CODE = {
    "": 0, "none": 0, "commit": 1, "switch": 2, "seam": 3,
    "completion": 4, "terminal": 5, "seam_commit": 6,
}
_TIMESTAMP_LOCAL_BITS = 21
_TIMESTAMP_LOCAL_LIMIT = 1 << _TIMESTAMP_LOCAL_BITS
_TIMESTAMP_EPISODE_BITS = 40
VISIBLE_SURFACE_SEMANTIC = "chosen_instance_actual_visible_surface/v1"
CLASS_VISIBLE_SURFACE_UNION_SEMANTIC = (
    "goal_class_actual_visible_surface_union/v1")
CLASS_VISIBILITY_ROSTER_CONTRACT = class_visibility_roster_contract(
    ore_classes.MINE_TARGET_POOL)
CLASS_VISIBILITY_ROSTER = tuple(CLASS_VISIBILITY_ROSTER_CONTRACT["classes"])
CLASS_VISIBILITY_INDEX = {
    kind: index for index, kind in enumerate(CLASS_VISIBILITY_ROSTER)}


def validation_donor_pool_widens(*, hunt_adapter, use_adapter, episode_split):
    return bool((hunt_adapter or use_adapter)
                and str(episode_split) == "validation")


def split_episode_metadata_by_world_and_pair(episodes, *,
                                             validation_fraction,
                                             seed=0):
    rows = list(episodes)
    fraction = float(validation_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("validation_fraction must be in [0,1]")
    ids = [str(row.get("episode", "")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("episode split requires unique non-empty episode ids")

    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    owners = {}
    row_keys = []
    for index, row in enumerate(rows):
        keys = []
        world_seed = row.get("world_seed")
        if world_seed is not None:
            if isinstance(world_seed, bool) or not isinstance(world_seed, int):
                raise ValueError(
                    f"episode {ids[index]} has non-integer world_seed")
            keys.append(("world", int(world_seed)))
        pair_group = row.get("positive_pair_group_id")
        if pair_group is not None:
            if not isinstance(pair_group, str) or not pair_group.strip():
                raise ValueError(
                    f"episode {ids[index]} has invalid positive_pair_group_id")
            keys.append(("pair", pair_group))
        if not keys:
            keys.append(("episode", ids[index]))
        row_keys.append(tuple(keys))
        for key in keys:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index

    components = collections.defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    assignment = {}
    for indices in components.values():
        component_keys = sorted({
            f"{kind}:{value}" for index in indices
            for kind, value in row_keys[index]})
        digest = hashlib.sha256(
            f"positive-pair-split/v1/{int(seed)}/".encode("utf-8")
            + "\n".join(component_keys).encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / float(1 << 64)
        side = "validation" if score < fraction else "train"
        assignment.update({ids[index]: side for index in indices})
    return {
        "train": [row for row in rows if assignment[row["episode"]] == "train"],
        "validation": [
            row for row in rows if assignment[row["episode"]] == "validation"],
        "assignment": assignment,
        "component_count": len(components),
        "validation_fraction": fraction,
        "seed": int(seed),
    }


def episode_world_seed_for_goal_donor_exclusion(episode):
    value = episode.get("world_seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"episode {episode.get('episode')!r} lacks integer world_seed for "
            "cross-world goal sampling")
    return int(value)


def _strict_class_visibility_bits(
        value, *, label: str,
        class_visibility_roster=CLASS_VISIBILITY_ROSTER) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer bitset")
    if value >> len(class_visibility_roster):
        raise ValueError(f"{label} sets bits outside the class roster")
    return int(value)


def _validate_class_visibility_row(
        row, *, target_kind, label: str,
        expected_recognition_gate_version=None,
        class_visibility_roster=CLASS_VISIBILITY_ROSTER,
        class_visibility_index=None) -> None:
    if class_visibility_index is None:
        class_visibility_index = {
            kind: index for index, kind in enumerate(class_visibility_roster)}
    known = _strict_class_visibility_bits(
        row.get("class_visibility_known_bits"),
        label=f"{label}.class_visibility_known_bits",
        class_visibility_roster=class_visibility_roster)
    visible = _strict_class_visibility_bits(
        row.get("class_visibility_visible_bits"),
        label=f"{label}.class_visibility_visible_bits",
        class_visibility_roster=class_visibility_roster)
    if visible & ~known:
        raise ValueError(f"{label} visible bits are not a subset of known bits")
    records = row.get("class_visible_instances")
    if not isinstance(records, list):
        raise ValueError(f"{label}.class_visible_instances is not a list")
    _strict_binary(
        row.get("class_visibility_geometry_valid"),
        label=f"{label}.class_visibility_geometry_valid")
    recorded = 0
    identities = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{label}.class_visible_instances[{index}] is not a mapping")
        kind = str(record.get("kind", "")).split(":")[-1]
        if kind not in class_visibility_index:
            raise ValueError(f"{label} census kind {kind!r} is outside the roster")
        bit = 1 << class_visibility_index[kind]
        if not visible & bit:
            raise ValueError(f"{label} census record {kind!r} has no visible bit")
        recorded |= bit
        identity = (kind, record.get("instance_id"))
        if (not isinstance(identity[1], str) or not identity[1].strip()
                or identity in identities):
            raise ValueError(f"{label} census record has bad/duplicate instance id")
        identities.add(identity)
        point = record.get("grounding_point")
        box = record.get("grounding_box")
        distance = record.get("grounding_distance")
        if (not isinstance(point, (tuple, list)) or len(point) != 2
                or any(isinstance(q, bool) or not isinstance(q, (int, float))
                       or not np.isfinite(float(q)) or not 0 <= float(q) < 1
                       for q in point)):
            raise ValueError(f"{label} census grounding_point is invalid")
        if (not isinstance(box, (tuple, list)) or len(box) != 4
                or any(isinstance(q, bool) or not isinstance(q, (int, float))
                       or not np.isfinite(float(q)) for q in box)):
            raise ValueError(f"{label} census grounding_box is invalid")
        if (isinstance(distance, bool) or not isinstance(distance, (int, float))
                or not np.isfinite(float(distance)) or float(distance) < 0):
            raise ValueError(f"{label} census grounding_distance is invalid")
        mask_png = record.get("grounding_mask_png")
        if not isinstance(mask_png, (bytes, bytearray, memoryview)):
            raise ValueError(f"{label} census grounding_mask_png is invalid")
        proof_summary = record.get("proof_summary")
        if not isinstance(proof_summary, dict):
            raise ValueError(f"{label} census proof_summary is missing")
        class_union = (
            record.get("instance_scope")
            == "all_recognizable_instances_class_union")
        if class_union:
            member_count = record.get("member_instance_count")
            representative_id = record.get("representative_instance_id")
            if (record.get("instance_id") != f"class_union:{kind}"
                    or isinstance(member_count, bool)
                    or not isinstance(member_count, int) or member_count < 1
                    or not isinstance(representative_id, str)
                    or not representative_id
                    or record.get("grounding_mask_kind")
                    != CLASS_VISIBLE_SURFACE_UNION_SEMANTIC):
                raise ValueError(f"{label} census class-union contract is invalid")
        bbox = proof_summary.get("visible_bbox_px")
        area = proof_summary.get("visible_area_px")
        short_side = proof_summary.get("visible_short_side_px")
        try:
            summary_gate_version = normalize_recognition_gate_version(
                proof_summary.get("recognition_gate_version"))
            if expected_recognition_gate_version is not None:
                summary_gate_version = require_matching_recognition_gate_version(
                    expected_recognition_gate_version,
                    proof_summary.get("recognition_gate_version"),
                    label=f"{label} census proof")
        except ValueError as exc:
            raise ValueError(f"{label} census recognition gate stamp: {exc}")
        expected_recognition = structural_surface_recognition_fields(
            area, short_side, visible_bbox_px=bbox,
            require_full_cube_fill=not class_union,
            gate_version=summary_gate_version)
        for key, expected_value in expected_recognition.items():
            observed = proof_summary.get(key)
            matches = (observed is expected_value
                       if isinstance(expected_value, bool)
                       else observed == expected_value)
            if not matches:
                raise ValueError(
                    f"{label} census recognition proof {key} mismatch: "
                    f"{observed!r}!={expected_value!r}")
        if (not class_union
                and expected_recognition["oracle_recognizable"] is not True):
            raise ValueError(
                f"{label} census record is below recognition gate")
    target_kind = str(target_kind).split(":")[-1]
    expected_sparse = visible
    if target_kind in class_visibility_index:
        expected_sparse &= ~(1 << class_visibility_index[target_kind])
    if recorded != expected_sparse:
        raise ValueError(
            f"{label} sparse non-goal record bits mismatch: "
            f"{recorded:#x}!={expected_sparse:#x}")
    if target_kind in class_visibility_index:
        bit = 1 << class_visibility_index[target_kind]
        if int(row.get("visibility_supervision_valid", 0)) == 1:
            if not known & bit:
                raise ValueError(f"{label} valid goal visibility is UNKNOWN")
            if int(bool(visible & bit)) != int(row.get("class_exist", -1)):
                raise ValueError(f"{label} goal visibility bit/class_exist mismatch")
def _episode_namespace(episode) -> int:
    digest = hashlib.blake2b(str(episode).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << _TIMESTAMP_EPISODE_BITS) - 1)


def _validate_episode_namespaces(episodes, *, label: str) -> None:
    episode_keys = [str(episode) for episode in episodes]
    duplicate_keys = sorted(key for key, count in collections.Counter(episode_keys).items()
                            if count > 1)
    if duplicate_keys:
        raise ValueError(f"duplicate {label} episode ids alias keys: {duplicate_keys[:10]}")
    namespace_to_episode = {}
    for episode_key in episode_keys:
        namespace = _episode_namespace(episode_key)
        previous = namespace_to_episode.setdefault(namespace, episode_key)
        if previous != episode_key:
            raise ValueError(
                f"{label} episode timestamp namespace collision: "
                f"{previous!r} and {episode_key!r}")


def _namespaced_timestamp(timestamp, *, source: int, episode) -> torch.Tensor:
    source = int(source)
    if source not in (0, 1):
        raise ValueError(f"timestamp source must be existing=0 or synthetic=1, got {source}")
    local = torch.as_tensor(timestamp, dtype=torch.long)
    if local.numel() and (int(local.min()) < 0 or int(local.max()) >= _TIMESTAMP_LOCAL_LIMIT):
        raise ValueError(
            f"local timestamp must be in [0,{_TIMESTAMP_LOCAL_LIMIT}); "
            f"got [{int(local.min())},{int(local.max())}]"
        )
    episode_bits = _episode_namespace(episode)
    base = (source << 61) | (episode_bits << _TIMESTAMP_LOCAL_BITS)
    return local + base


def _event_code(value) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _EVENT_CODE:
            raise ValueError(f"unknown event_kind {value!r}")
        return _EVENT_CODE[key]
    code = int(value)
    if code < 0:
        raise ValueError(f"event_kind must be nonnegative, got {code}")
    return code


def _strict_binary(value, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"{label} must be an integer 0/1, got {value!r}")
    return value


def temporal_binding_bc_valid(
        rows, *, identity_change_is_boundary=False,
        commit_identity_change_is_boundary=False,
        interaction_group_identity_change_is_boundary=False):
    out = np.zeros(len(rows), dtype=np.float32)
    bound_id = None
    previous_id = None
    previous_group_id = None
    for index, row in enumerate(rows):
        phase = row.get("phase")
        if isinstance(phase, bool) or not isinstance(phase, int) or phase not in (0, 1, 2):
            raise ValueError(f"temporal binding row {index} has invalid phase {phase!r}")
        event = str(row.get("event_kind", ""))
        committed = _strict_binary(
            row.get("target_committed"),
            label=f"temporal binding row {index}.target_committed")
        recognizable = _strict_binary(
            row.get("chosen_recognizable"),
            label=f"temporal binding row {index}.chosen_recognizable")
        visibility_valid = _strict_binary(
            row.get("visibility_supervision_valid"),
            label=(f"temporal binding row {index}."
                   "visibility_supervision_valid"))
        committed_id = row.get("committed_instance_id")
        if bool(committed_id) != bool(committed):
            raise ValueError(
                f"temporal binding row {index} committed identity mismatch")
        interaction_group_id = row.get("interaction_group_id")
        if interaction_group_identity_change_is_boundary:
            if bool(interaction_group_id) != bool(committed):
                raise ValueError(
                    f"temporal binding row {index} committed group mismatch")
            if interaction_group_id is not None:
                interaction_group_id = str(interaction_group_id)
        if phase == 0:
            if committed or recognizable:
                raise ValueError(
                    f"temporal binding row {index} EXPLORE carries target state")
            bound_id = None
            previous_id = None
            previous_group_id = None
            out[index] = 1.0
            continue
        if not committed:
            removed = row.get("target_removed")
            removed_id = (
                "block:%d:%d:%d" % tuple(int(q) for q in removed[:3])
                if isinstance(removed, (tuple, list)) and len(removed) >= 3
                else None)
            post_break = _strict_binary(
                row.get("post_break", 0),
                label=f"temporal binding row {index}.post_break")
            drop_collect = str(row.get("motion_phase") or "") == "drop_collect"
            if phase == 2 and (removed_id is not None or post_break
                               or drop_collect):
                out[index] = float(
                    bool(post_break or drop_collect)
                    or bound_id == removed_id)
                bound_id = None
                previous_id = None
                continue
            raise ValueError(
                f"temporal binding row {index} committed phase lacks identity")
        identity_changed = (
            previous_id is not None and committed_id != previous_id)
        marked_commit_change = bool(
            identity_changed and event == "commit"
            and commit_identity_change_is_boundary)
        within_group_change = bool(
            identity_changed
            and interaction_group_identity_change_is_boundary
            and previous_group_id is not None
            and str(interaction_group_id) == str(previous_group_id))
        if (identity_changed and event not in {"switch", "seam_commit"}
                and not identity_change_is_boundary
                and not marked_commit_change and not within_group_change):
            raise ValueError(
                f"temporal binding row {index} identity changed without boundary")
        if event in {"switch", "seam_commit"} or (
                identity_changed and identity_change_is_boundary
                ) or marked_commit_change or within_group_change:
            bound_id = None
        if recognizable and visibility_valid:
            bound_id = str(committed_id)
        out[index] = float(bound_id == str(committed_id))
        previous_id = str(committed_id)
        previous_group_id = interaction_group_id
    return out


def goal_query_bc_valid(rows):
    return np.asarray([
        _strict_binary(row.get("bc_valid"), label=f"query row {index}.bc_valid")
        for index, row in enumerate(rows)
    ], dtype=np.float32)


def _goal_donor_row_eligible(row):
    return bool(
        int(row.get("phase", -1)) == 1
        and int(row.get("chosen_recognizable", 0)) == 1
        and int(row.get("dense_mask_valid", 0)) == 1
        and row.get("chosen_mask_png") is not None)


def _goal_donor_resized_mask_quality(mask, *, min_pixels, min_short_side):
    for name, value in (("min_pixels", min_pixels),
                        ("min_short_side", min_short_side)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"goal donor {name} must be a non-negative integer, got {value!r}")
    support = np.asarray(mask) > 0
    if support.ndim != 2:
        raise ValueError(
            f"goal donor resized mask must be 2-D, got {support.shape}")
    area = int(support.sum())
    if area <= 0:
        return False, 0, 0
    ys, xs = np.nonzero(support)
    short_side = int(min(xs.max() - xs.min() + 1,
                         ys.max() - ys.min() + 1))
    return bool(area >= min_pixels and short_side >= min_short_side), area, short_side


def _decode_native_support_png(value, *, label: str):
    if value is None:
        raise ValueError(f"{label} is missing")
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{label} must be PNG bytes")
    mask = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (360, 640):
        raise ValueError(
            f"{label} must decode to native 640x360 support, got "
            f"{None if mask is None else mask.shape}")
    values = np.unique(mask)
    if not set(map(int, values)).issubset({0, 255}):
        raise ValueError(f"{label} is not a binary native support mask")
    return (mask > 0).astype(np.uint8)


def _bound_pack_recognition_gate_version(schema) -> str:
    normalize_recognition_gate_version(schema.get("recognition_gate_version"))
    contract = schema.get("recognition_contract")
    if contract is None:
        raise ValueError("dense pack lacks structural recognition contract")
    contract_gate_version = recognition_contract_gate_version(contract)
    return require_matching_recognition_gate_version(
        contract_gate_version, schema.get("recognition_gate_version"),
        label="synthetic pack contract/top-level stamp")


def _resize_dense_visible_surface(mask, size):
    src = (np.asarray(mask) > 0).astype(np.float32)
    if src.ndim != 2:
        raise ValueError(f"dense visible-surface mask must be 2-D, got {src.shape}")
    dst_w, dst_h = map(int, size)
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"bad dense visible-surface resize {src.shape}->{size}")
    interpolation = (cv2.INTER_AREA
                     if dst_w <= src.shape[1] and dst_h <= src.shape[0]
                     else cv2.INTER_NEAREST)
    resized = cv2.resize(src, (dst_w, dst_h), interpolation=interpolation)
    return (resized > 0.0).astype(np.uint8)


_LMDB_ENV_CACHE = {}


def _open_pack_env(path):
    key = str(path)
    cached = _LMDB_ENV_CACHE.get(key)
    if cached is not None:
        pid, env = cached
        if pid == os.getpid():
            return env
        try:
            env.close()
        except lmdb.Error:
            pass
        del _LMDB_ENV_CACHE[key]
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False,
                    max_readers=512, subdir=True)
    _LMDB_ENV_CACHE[key] = (os.getpid(), env)
    return env


class SyntheticWindowDataset(Dataset):
    def __init__(self, lmdb_dir, window=128, seed=0,
                 pad_short_min_length=48,
                 require_visible_surface_masks=False,
                 episode_split="all",
                 episode_validation_fraction=0.0,
                 episode_split_seed=0,
                 timestamp_namespace_prefix="",
                 donor_infeasible_policy="error",
                 goal_donor_min_resized_pixels=0,
                 goal_donor_min_resized_short_side=0,
                 goal_donor_max_attempts=24,
                 camera_quantization="packed",
                 require_chosen_surface_masks=None):
        if lmdb is None:
            raise RuntimeError(
                "SyntheticWindowDataset requires the optional 'lmdb' package; "
                "activate the minestudio training environment")
        self.path = str(Path(lmdb_dir).resolve())
        for name, value in (
                ("goal_donor_min_resized_pixels",
                 goal_donor_min_resized_pixels),
                ("goal_donor_min_resized_short_side",
                 goal_donor_min_resized_short_side)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        self.goal_donor_min_resized_pixels = int(
            goal_donor_min_resized_pixels)
        self.goal_donor_min_resized_short_side = int(
            goal_donor_min_resized_short_side)
        if (isinstance(goal_donor_max_attempts, bool)
                or not isinstance(goal_donor_max_attempts, int)
                or goal_donor_max_attempts < 1):
            raise ValueError(
                "goal_donor_max_attempts must be a positive integer, got "
                f"{goal_donor_max_attempts!r}")
        self.goal_donor_max_attempts = int(goal_donor_max_attempts)
        self.camera_quantization = str(camera_quantization)
        if self.camera_quantization not in ("packed", "mu_law"):
            raise ValueError(
                "camera_quantization must be packed|mu_law, "
                f"got {camera_quantization!r}")
        self._camera_transformer = None
        if self.camera_quantization == "mu_law":
            from minestudio.utils.vpt_lib.actions import ActionTransformer
            self._camera_transformer = ActionTransformer(
                camera_maxval=10, camera_binsize=2,
                camera_quantization_scheme="mu_law", camera_mu=10.0)
        self.timestamp_namespace_prefix = str(timestamp_namespace_prefix)
        self.donor_infeasible_policy = str(donor_infeasible_policy)
        if self.donor_infeasible_policy not in ("error", "drop_episodes"):
            raise ValueError(
                f"donor_infeasible_policy must be 'error' or 'drop_episodes', "
                f"got {donor_infeasible_policy!r}")
        self.window = int(window)
        if self.window != 128:
            raise ValueError(f"the packed schema requires window=128, got {self.window}")
        self.require_visible_surface_masks = bool(require_visible_surface_masks)
        self.require_chosen_surface_masks = bool(
            self.require_visible_surface_masks
            or (require_chosen_surface_masks is True))
        if not self.require_chosen_surface_masks:
            raise ValueError(
                "the loader reads packs with exact chosen visible-surface masks "
                "(require_visible_surface_masks or require_chosen_surface_masks)")
        self.episode_split = str(episode_split)
        if self.episode_split not in {"all", "train", "validation"}:
            raise ValueError(
                "episode_split must be one of {'all','train','validation'}")
        self.episode_validation_fraction = float(
            episode_validation_fraction)
        self.episode_split_seed = int(episode_split_seed)
        if self.episode_split != "all" and not (
                0.0 < self.episode_validation_fraction < 1.0):
            raise ValueError(
                "train/validation episode_split requires "
                "episode_validation_fraction in (0,1)")
        if (pad_short_min_length is None
                or not 0 < int(pad_short_min_length) <= self.window):
            raise ValueError(
                f"pad_short_min_length must be in (0, {self.window}], "
                f"got {pad_short_min_length}")
        self.pad_short_min_length = int(pad_short_min_length)
        self.min_length = self.pad_short_min_length
        self.image_size = 224
        self.seed = int(seed)
        env = _open_pack_env(self.path)
        with env.begin() as txn:
            raw_schema = txn.get(b"__schema__")
            if raw_schema is None:
                raise ValueError(f"missing __schema__ in synthetic LMDB {self.path}")
            try:
                schema = json.loads(raw_schema.decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"invalid JSON __schema__ in {self.path}") from exc
            packed_roster_contract = schema.get(
                "class_visibility_roster_contract")
            if packed_roster_contract is None:
                packed_roster_contract = CLASS_VISIBILITY_ROSTER_CONTRACT
            try:
                canonical_packed_roster = class_visibility_roster_contract(
                    packed_roster_contract["classes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "invalid packed class visibility roster contract") from exc
            if canonical_packed_roster != packed_roster_contract:
                raise ValueError(
                    "packed class visibility roster contract is not canonical")
            self.class_visibility_roster_contract = canonical_packed_roster
            self.class_visibility_roster = tuple(
                canonical_packed_roster["classes"])
            self.class_visibility_index = {
                kind: index for index, kind in enumerate(
                    self.class_visibility_roster)}
            observed_schema_version = int(schema.get("version", -1))
            try:
                pack_recognition_gate_version = (
                    _bound_pack_recognition_gate_version(schema))
            except ValueError as exc:
                raise ValueError(
                    f"synthetic pack recognition gate binding is invalid: "
                    f"{exc}") from exc
            if (schema.get("positive_pair_contracts")
                    or schema.get("positive_pair_contract")):
                raise ValueError(
                    f"paired-episode packs are not supported: {self.path}")
            required_frame_fields = (
                _BASE_FRAME_FIELDS | _LABEL_FRAME_FIELDS
                | _COMMITTED_SURFACE_FRAME_FIELDS | _COMMITTED_UNION_FRAME_FIELDS)
            distance_extension9 = dict(
                schema.get("per_frame_label_extensions") or {}).get(
                    "chosen_distance")
            chosen_distance_available9 = bool(
                isinstance(distance_extension9, dict)
                and distance_extension9.get("present") is True)
            if chosen_distance_available9:
                required_frame_fields |= {"chosen_distance", "post_break"}
            target_removed_extension9 = dict(
                schema.get("per_frame_label_extensions") or {}).get(
                    "target_removed")
            if (isinstance(target_removed_extension9, dict)
                    and target_removed_extension9.get("present") is True):
                required_frame_fields |= {"target_removed", "motion_phase"}
            if (observed_schema_version
                    != FRESH_SHORT_SIDE_6_SYNTHETIC_LMDB_SCHEMA_VERSION
                    or int(schema.get("window", -1)) != 128
                    or schema.get("per_episode_target_kind") is not True
                    or schema.get("strict_current_visibility") is not True
                    or int(schema.get("visible_instance_contract", -1)) != 1
                    or int(schema.get(
                        "per_frame_visibility_scan_scope_contract", -1)) != 1
                    or schema.get("chosen_mask_contract") != VISIBLE_SURFACE_SEMANTIC
                    or schema.get("committed_surface_tracking_contract")
                        != COMMITTED_SURFACE_TRACKING_CONTRACT
                    or pack_recognition_gate_version != RECOGNITION_GATE_VERSION
                    or schema.get("dense_segmentation_supervision") is not True
                    or schema.get("chosen_visible_surface_masks") is not True
                    or schema.get("render_pose_contract")
                        != VISIBLE_SURFACE_RENDER_POSE_SOURCE
                    or schema.get("class_visible_mask_contract")
                        != COMMITTED_CLASS_VISIBLE_SURFACE_UNION_SEMANTIC
                    or schema.get("class_union_support_contract")
                        != COMMITTED_CLASS_UNION_SUPPORT_CONTRACT
                    or schema.get("viewmodel_occlusion_contract")
                        != viewmodel_occlusion_contract()
                    or schema.get("dense_mask_valid_contract") != (
                        "chosen_surface_visible AND visibility_supervision_valid AND "
                        "viewmodel_pixel_supervision_valid")
                    or (self.require_visible_surface_masks
                        and schema.get("all_class_visibility_census") is not True)
                    or (self.require_visible_surface_masks
                        and schema.get("class_visibility_roster_contract")
                            != self.class_visibility_roster_contract)
                    or (self.require_visible_surface_masks
                        and not isinstance(
                            schema.get("class_visibility_frame_contract"), dict))):
                raise ValueError(
                    f"synthetic loader requires strict schema version "
                    f"{FRESH_SHORT_SIDE_6_SYNTHETIC_LMDB_SCHEMA_VERSION}, window=128, "
                    "per_episode_target_kind, and certified current-visibility capabilities; "
                    f"got {schema}"
                )
            raw_episodes = txn.get(b"__episodes__")
            if raw_episodes is None:
                raise ValueError(f"missing __episodes__ in synthetic LMDB {self.path}")
            episodes = pickle.loads(raw_episodes)
            if self.require_visible_surface_masks:
                census_cells = {str(ep.get("cell", "")) for ep in episodes}
                expected_roster_by_cell = {
                    "mine": CLASS_VISIBILITY_ROSTER_CONTRACT,
                    "hunt": HUNT_CLASS_VISIBILITY_ROSTER_CONTRACT,
                    "place_block": USE_CLASS_VISIBILITY_ROSTER_CONTRACT,
                }
                if len(census_cells) != 1 or next(iter(census_cells)) not in (
                        expected_roster_by_cell):
                    raise ValueError(
                        "all-class census pack must declare one supported cell: "
                        f"{sorted(census_cells)}")
                census_cell = next(iter(census_cells))
                if (self.class_visibility_roster_contract
                        != expected_roster_by_cell[census_cell]):
                    raise ValueError(
                        f"{census_cell} census roster does not match its canonical "
                        "class contract")
            required_meta = {"episode", "length", "cell", "verb_id", "tool_id",
                             "target_kind", "commits", "switches", "seams"}
            _validate_episode_namespaces((ep.get("episode") for ep in episodes),
                                         label="synthetic")
            for ep in episodes:
                missing = required_meta.difference(ep)
                if missing:
                    raise ValueError(f"episode metadata missing {sorted(missing)}: {ep.get('episode')}")
                try:
                    require_matching_recognition_gate_version(
                        pack_recognition_gate_version,
                        ep.get("recognition_gate_version"),
                        label=f"episode {ep.get('episode')} pack/episode stamp")
                except ValueError as exc:
                    raise ValueError(
                        f"episode recognition gate binding is invalid: {exc}") from exc
                if int(ep["length"]) <= 0:
                    raise ValueError(f"invalid episode length: {ep.get('episode')}={ep.get('length')}")
                if int(ep["length"]) >= _TIMESTAMP_LOCAL_LIMIT:
                    raise ValueError(f"episode exceeds timestamp namespace: {ep.get('episode')}")
                for field in ("commits", "switches", "seams"):
                    vals = [int(v) for v in ep[field]]
                    if vals != sorted(set(vals)) or any(v < 0 or v >= int(ep["length"]) for v in vals):
                        raise ValueError(f"invalid {field} in {ep.get('episode')}: {ep[field]}")
                first = txn.get(f"frame/{ep['episode']}/000000".encode())
                if first is None:
                    raise ValueError(f"missing first frame for episode {ep['episode']}")
                first_row = pickle.loads(first)
                missing = required_frame_fields.difference(first_row)
                if self.require_visible_surface_masks:
                    missing |= _CLASS_VISIBILITY_FRAME_FIELDS.difference(first_row)
                if missing:
                    raise ValueError(f"frame {ep['episode']}/0 missing {sorted(missing)}")
                if self.require_visible_surface_masks and (
                        ep.get("class_visibility_roster")
                            != list(self.class_visibility_roster)
                        or ep.get("class_visibility_roster_sha256")
                            != self.class_visibility_roster_contract["sha256"]):
                    raise ValueError(
                        f"episode {ep['episode']} class visibility roster/hash "
                        "does not match the loader")
        self.schema = schema
        self.schema_version = observed_schema_version
        self.binding_identity_change_is_boundary = bool(
            schema.get("hunt_native_adapter_contract"))
        self.binding_commit_identity_change_is_boundary = bool(
            schema.get("use_human_native_adapter_contract"))
        self.binding_group_identity_change_is_boundary = bool(
            schema.get("use_group_transaction_contract"))
        self.pack_recognition_gate_version = pack_recognition_gate_version
        self.chosen_distance_available = chosen_distance_available9
        validate_positive_pair_episode_groups(
            episodes, require_all=False, label="synthetic LMDB")
        if self.episode_split == "all":
            split_episodes = list(episodes)
            self.episode_split_summary = None
        else:
            split = split_episode_metadata_by_world_and_pair(
                episodes,
                validation_fraction=self.episode_validation_fraction,
                seed=self.episode_split_seed)
            split_episodes = list(split[self.episode_split])
            self.episode_split_summary = {
                key: value for key, value in split.items()
                if key not in {"train", "validation"}}
            self.episode_split_summary.update(
                train_episodes=len(split["train"]),
                validation_episodes=len(split["validation"]),
                selected=self.episode_split)
            if not split_episodes:
                raise ValueError(
                    f"episode split {self.episode_split!r} is empty: "
                    f"{self.episode_split_summary}")
        self.episodes = self._filter_short_episodes(split_episodes)
        self._goal_donor_pool_episodes = (
            self._filter_short_episodes(list(episodes))
            if validation_donor_pool_widens(
                hunt_adapter=self.binding_identity_change_is_boundary,
                use_adapter=self.binding_commit_identity_change_is_boundary,
                episode_split=self.episode_split)
            else self.episodes)
        self._goal_donor_pool = self._resolve_goal_donors()
        self._build_sequential_items()

    def _filter_short_episodes(self, episodes):
        episodes = list(episodes)
        kept = [e for e in episodes if int(e["length"]) >= self.min_length]
        self.dropped_short_episodes = [
            (str(e["episode"]), int(e["length"])) for e in episodes
            if int(e["length"]) < self.min_length]
        if self.dropped_short_episodes:
            lost = sum(length for _, length in self.dropped_short_episodes)
            print(f"[synthetic-sampler] DROPPED {len(self.dropped_short_episodes)}/"
                  f"{len(episodes)} episodes shorter than min_length={self.min_length} "
                  f"({lost} frames unreachable by any window): "
                  + ", ".join(f"{name}:{length}"
                              for name, length in self.dropped_short_episodes[:12]))
        if not kept:
            raise ValueError(f"no synthetic episodes >= {self.min_length} frames in {self.path}")
        return kept

    @property
    def env(self):
        return _open_pack_env(self.path)

    def _build_sequential_items(self):
        episodes_with_items = []
        index_map = []
        bias = 0
        for ep_idx, ep in enumerate(self.episodes):
            length = int(ep["length"])
            n = max(1, (length + self.window - 1) // self.window)
            episodes_with_items.append(
                (f"{self.timestamp_namespace_prefix}{ep['episode']}", n, bias))
            index_map.extend((ep_idx, k) for k in range(n))
            bias += n
        self.episodes_with_items = episodes_with_items
        self._sequential_index_map = tuple(index_map)
        n_tail_padded = sum(
            1 for ep_idx, k in index_map
            if (k + 1) * self.window > int(self.episodes[ep_idx]["length"]))
        n_padded_frames = sum(
            max(0, (k + 1) * self.window - int(self.episodes[ep_idx]["length"]))
            for ep_idx, k in index_map)
        print(
            f"[synthetic-sequential] episodes={len(self.episodes)} "
            f"items={len(index_map)} tail_padded={n_tail_padded} "
            f"padded_frames={n_padded_frames} window={self.window} "
            "source_frame_coverage=all",
            flush=True)

    def __len__(self):
        return len(self._sequential_index_map)

    def _resolve_goal_donors(self):
        pool = {}
        donor_episodes = getattr(
            self, "_goal_donor_pool_episodes", self.episodes)
        if donor_episodes is not self.episodes:
            print(
                "[synthetic-goal-donor] validation donor pool widened to "
                f"the full pack (native Hunt/USE adapter): "
                f"{len(donor_episodes)} episodes", flush=True)
        for ep in sorted(donor_episodes, key=lambda e: str(e["episode"])):
            commits = tuple(int(c) for c in ep.get("commits") or ())
            if not commits:
                continue
            key = ore_classes.goal_key(ep["cell"], ep.get("target_kind"))
            pool.setdefault(key, []).append(
                (str(ep["episode"]),
                 episode_world_seed_for_goal_donor_exclusion(ep),
                 int(ep["length"]), commits))
        needed = sorted({
            ore_classes.goal_key(ep["cell"], ep.get("target_kind"))
            for ep in self.episodes})
        problems = []
        for key in needed:
            worlds = {entry[1] for entry in pool.get(key, ())}
            if len(worlds) < 2:
                problems.append(f"{key}(worlds={len(worlds)})")
        if problems:
            if self.donor_infeasible_policy == "drop_episodes":
                bad_keys = {p.split("(")[0] for p in problems}
                before = len(self.episodes)
                self.episodes = [
                    ep for ep in self.episodes
                    if ore_classes.goal_key(ep["cell"], ep.get("target_kind"))
                    not in bad_keys]
                print(
                    f"[synthetic-goal-donor] drop_episodes: {problems} -> "
                    f"dropped {before - len(self.episodes)} episodes",
                    flush=True)
                if not self.episodes:
                    raise ValueError(
                        "donor_infeasible_policy=drop_episodes removed every "
                        "episode in this pack")
            else:
                raise ValueError(
                    "cross-world goal sampling needs committed donor episodes in "
                    f">=2 distinct worlds per goal key; infeasible: {problems}")
        donors = {key: tuple(entries) for key, entries in pool.items()}
        print("[synthetic-goal-donor] per-key donor episodes: "
              + ", ".join(f"{key}={len(entries)}"
                          for key, entries in sorted(donors.items())),
              flush=True)
        return donors

    GOAL_DONOR_BAND_FRAMES = 40

    def _goal_donor_exemplar(self, goal_key, episode, start, draw_id,
                             exclude_world_seed):
        if exclude_world_seed is None:
            raise ValueError(
                "episode_donor goals are cross-world by definition; the "
                "calling episode must supply its world seed for exclusion")
        entries = [entry for entry in self._goal_donor_pool.get(goal_key, ())
                   if entry[1] != int(exclude_world_seed)]
        if not entries:
            raise ValueError(
                f"no cross-world donor episode for {goal_key!r} outside "
                f"world_seed={int(exclude_world_seed)}")
        digest_key = (f"{self.seed}/{episode}/{int(start)}/{int(draw_id)}"
                      f"/donor/exclude={int(exclude_world_seed)}")
        digest = hashlib.blake2b(
            digest_key.encode("utf-8"), digest_size=8).digest()
        draw_rng = np.random.default_rng(int.from_bytes(digest, "little"))
        with self.env.begin() as txn:
            for _ in range(self.goal_donor_max_attempts):
                donor_ep, _, length, commits = (
                    entries[int(draw_rng.integers(0, len(entries)))])
                anchor = commits[int(draw_rng.integers(0, len(commits)))]
                t = anchor + 1 + int(draw_rng.integers(
                    0, self.GOAL_DONOR_BAND_FRAMES))
                if t >= length:
                    continue
                value = txn.get(f"frame/{donor_ep}/{t:06d}".encode())
                if value is None:
                    raise KeyError((donor_ep, t))
                row = pickle.loads(value)
                if not _goal_donor_row_eligible(row):
                    continue
                bgr = cv2.imdecode(
                    np.frombuffer(row["pov_jpeg"], np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"goal donor {donor_ep}/{t}: bad pov_jpeg")
                native_mask = _decode_native_support_png(
                    row["chosen_mask_png"], label=f"goal donor {donor_ep}/{t}")
                img, mask = self._finalize_goal_pair(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), native_mask)
                quality_ok, _, _ = _goal_donor_resized_mask_quality(
                    mask,
                    min_pixels=self.goal_donor_min_resized_pixels,
                    min_short_side=(
                        self.goal_donor_min_resized_short_side))
                if not quality_ok:
                    continue
                return img, mask, f"donor:{donor_ep}/{t}"
        raise ValueError(
            f"goal donor draw exhausted {self.goal_donor_max_attempts} attempts "
            f"for {goal_key!r} (episode={episode} start={int(start)} "
            f"draw={int(draw_id)}); donor band lacks certified frames with "
            f"resized area>={self.goal_donor_min_resized_pixels} and "
            f"short-side>={self.goal_donor_min_resized_short_side}")

    def _finalize_goal_pair(self, img_rgb, mask):
        size = (self.image_size, self.image_size)
        img = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
        mask = _resize_dense_visible_surface(mask, size)
        return img.astype(np.uint8), (mask > 0).astype(np.uint8)

    def _camera_labels(self, rows):
        if self._camera_transformer is None:
            return np.asarray([r["camera"] for r in rows], np.int64)
        out = np.empty(len(rows), np.int64)
        for i, r in enumerate(rows):
            deg = np.asarray((r.get("env_action") or {}).get("camera", (0.0, 0.0)),
                             np.float64).reshape(1, 2)
            if not np.all(np.isfinite(deg)):
                raise ValueError("packed env_action.camera is not finite")
            pq = self._camera_transformer.discretize_camera(deg)[0]
            out[i] = int(pq[0]) * 11 + int(pq[1])
        return out

    def __getitem__(self, index):
        item_index = int(index)
        rng = np.random.default_rng((self.seed + item_index * 1000003) & 0xFFFFFFFF)
        ep_idx, chunk = self._sequential_index_map[item_index]
        ep = self.episodes[ep_idx]
        start = chunk * self.window
        real_len = min(self.window, int(ep["length"]) - start)
        rows = []
        with self.env.begin() as txn:
            for t in range(start, start + real_len):
                value = txn.get(f"frame/{ep['episode']}/{t:06d}".encode())
                if value is None:
                    raise KeyError((ep["episode"], t))
                row = pickle.loads(value)
                missing = (
                    _BASE_FRAME_FIELDS | _LABEL_FRAME_FIELDS
                    | _COMMITTED_SURFACE_FRAME_FIELDS
                    | _COMMITTED_UNION_FRAME_FIELDS).difference(row)
                if self.chosen_distance_available:
                    missing |= {"chosen_distance", "post_break"}.difference(row)
                if missing:
                    raise ValueError(f"frame {ep['episode']}/{t} missing {sorted(missing)}")
                if self.require_visible_surface_masks:
                    _validate_class_visibility_row(
                        row, target_kind=ep["target_kind"],
                        label=f"frame {ep['episode']}/{t}",
                        expected_recognition_gate_version=(
                            self.pack_recognition_gate_version),
                        class_visibility_roster=self.class_visibility_roster,
                        class_visibility_index=self.class_visibility_index)
                rows.append(row)
        temporal_binding = temporal_binding_bc_valid(
            rows,
            identity_change_is_boundary=(
                self.binding_identity_change_is_boundary),
            commit_identity_change_is_boundary=(
                self.binding_commit_identity_change_is_boundary),
            interaction_group_identity_change_is_boundary=(
                self.binding_group_identity_change_is_boundary))
        pad_len = self.window - real_len
        if pad_len:
            template = dict(rows[-1])
            template.update(bc_valid=0, centroid_valid=0, decision_weight=0.0,
                            class_exist=0, class_recognizable=0,
                            chosen_visible=0, chosen_surface_visible=0,
                            chosen_recognizable=0, target_committed=0,
                            committed_instance_id=None,
                            committed_surface_instance=None,
                            chosen_instance_id=None, chosen_mask_png=None,
                            chosen_point=None, chosen_box=None,
                            visible_instances=[], visible_instance_count=0,
                            certified_mask_instance_count=0,
                            class_visibility_known_bits=0,
                            class_visibility_visible_bits=0,
                            class_visibility_geometry_valid=0,
                            class_visible_instances=[],
                            class_visible_mask_png=None,
                            class_visible_mask_kind="none",
                            class_union_support_contract=(
                                COMMITTED_CLASS_UNION_SUPPORT_CONTRACT),
                            class_union_committed_surface_added_pixels=0,
                            visibility_supervision_valid=0,
                            dense_mask_valid=0, bbox_valid=0,
                            chosen_distance=-1.0, post_break=1)
            rows.extend(dict(template) for _ in range(pad_len))
            temporal_binding = np.pad(
                temporal_binding, (0, pad_len), constant_values=0.0)
        verb_ids = {int(r["verb_id"]) for r in rows}
        if verb_ids != {int(ep["verb_id"])}:
            raise AssertionError(f"episode mixes interaction ids: {ep['episode']} {verb_ids}")
        tool_ids = {int(r["tool_id"]) for r in rows}
        if tool_ids != {int(ep["tool_id"])}:
            raise AssertionError(
                f"episode mixes tool ids: {ep['episode']} {tool_ids}")

        frames, union_masks = [], []
        for row in rows:
            bgr = cv2.imdecode(np.frombuffer(row["pov_jpeg"], np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (360, 640, 3):
                raise ValueError(
                    f"strict synthetic POV must decode to 640x360x3: "
                    f"{ep['episode']}/{row.get('traj_t')} -> "
                    f"{None if bgr is None else bgr.shape}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (self.image_size, self.image_size),
                                     interpolation=cv2.INTER_LINEAR))
            mask_png = row["chosen_mask_png"]
            expected_visible = bool(row["chosen_surface_visible"])
            if mask_png is None:
                mask = np.zeros((self.image_size, self.image_size), np.uint8)
            else:
                native_mask = _decode_native_support_png(
                    mask_png, label="chosen_mask_png")
                mask = _resize_dense_visible_surface(
                    native_mask, (self.image_size, self.image_size))
            if bool(mask.any()) != expected_visible:
                raise ValueError(
                    f"strict synthetic grounding mask/visibility mismatch: "
                    f"{ep['episode']}/{row.get('traj_t')}")
            union_png9 = row.get("class_visible_mask_png")
            if union_png9 is None:
                union_mask9 = np.zeros(
                    (self.image_size, self.image_size), np.uint8)
            else:
                union_native9 = _decode_native_support_png(
                    union_png9, label="class_visible_mask_png")
                union_mask9 = _resize_dense_visible_surface(
                    union_native9, (self.image_size, self.image_size))
            committed_union9 = bool(
                row.get("class_union_support_contract")
                == COMMITTED_CLASS_UNION_SUPPORT_CONTRACT)
            expected_union9 = bool(
                row.get("class_recognizable", row["class_exist"])
                or (committed_union9
                    and int(row.get("chosen_surface_visible", 0)) == 1))
            if bool(union_mask9.any()) != expected_union9:
                raise ValueError(
                    "strict synthetic class-union support "
                    f"mismatch: {ep['episode']}/{row.get('traj_t')}")
            if (committed_union9 and mask.any()
                    and np.any((mask > 0) & (union_mask9 == 0))):
                raise ValueError(
                    "strict synthetic chosen support escaped committed "
                    f"class union: {ep['episode']}/{row.get('traj_t')}")
            union_masks.append(
                (union_mask9 > 0).astype(np.uint8))
        frames = np.stack(frames)
        union_masks = np.stack(union_masks)
        class_exist = np.asarray([r["class_exist"] for r in rows], np.float32)
        target_committed = np.asarray(
            [r["target_committed"] for r in rows], np.float32)
        supervision_valid = np.asarray(
            [r["visibility_supervision_valid"] for r in rows], np.float32)
        for r in rows:
            _event_code(r["event_kind"])
        union_mask_valid = supervision_valid.copy()
        sample_bc_valid = goal_query_bc_valid(rows)
        rng.random()
        goal_key = ore_classes.goal_key(ep["cell"], ep.get("target_kind"))
        goal_img, goal_mask, exemplar_id = self._goal_donor_exemplar(
            goal_key, ep["episode"], start, draw_id=item_index,
            exclude_world_seed=episode_world_seed_for_goal_donor_exclusion(ep))
        L = self.window
        obj_id = np.full(L, int(ep["verb_id"]), np.int64)
        action_keys = list(rows[0]["env_prev_action"])
        env_prev_action = {
            k: torch.from_numpy(np.stack([np.asarray(r["env_prev_action"][k]) for r in rows]))
            for k in action_keys
        }
        return {
            "image": torch.from_numpy(np.ascontiguousarray(frames)),
            "timestamp": _namespaced_timestamp(
                torch.arange(start, start + L, dtype=torch.long),
                source=1,
                episode=f"{self.timestamp_namespace_prefix}{ep['episode']}"),
            "env_prev_action": env_prev_action,
            "agent_action": {
                "buttons": torch.from_numpy(np.asarray([r["buttons"] for r in rows], np.int64)[:, None]),
                "camera": torch.from_numpy(self._camera_labels(rows)[:, None]),
            },
            "segmentation": {
                "union_mask": torch.from_numpy(union_masks),
            },
            "cross_view": {
                "cross_view_image": torch.from_numpy(np.repeat(goal_img[None], L, axis=0).copy()),
                "cross_view_obj_mask": torch.from_numpy(np.repeat(goal_mask[None], L, axis=0).copy()),
                "cross_view_obj_id": torch.from_numpy(obj_id.copy()),
            },
            "mask": torch.from_numpy(
                (np.arange(L) < real_len).astype(np.float32)),
            "prev_action_dropout": torch.from_numpy(
                (rng.random(L) > 0.75).astype(np.float32)),
            "synthetic": torch.ones(L, dtype=torch.float32),
            "phase": torch.from_numpy(np.asarray([r["phase"] for r in rows], np.int64)),
            "class_exist": torch.from_numpy(class_exist),
            "target_committed": torch.from_numpy(target_committed),
            "visibility_supervision_valid": torch.from_numpy(
                supervision_valid),
            "decision_weight": torch.from_numpy(
                np.asarray([r["decision_weight"] for r in rows], np.float32)),
            "bc_valid": torch.from_numpy(
                sample_bc_valid),
            "temporal_binding_bc_valid": torch.from_numpy(
                np.asarray(temporal_binding, np.float32)),
            "union_mask_valid": torch.from_numpy(union_mask_valid),
            "episode": ep["episode"],
            "exemplar_id": exemplar_id,
        }


class SyntheticOnlyDataset(Dataset):

    def __init__(self, synthetic):
        self.synthetic = synthetic

    def __len__(self):
        return len(self.synthetic)

    def __getitem__(self, index):
        local_index = int(index) % len(self.synthetic)
        return self._contract(self.synthetic[local_index])

    @staticmethod
    def _contract(item):
        mask = item.get("mask", torch.ones(128))
        synthetic = torch.ones_like(mask).float()
        timestamp = torch.as_tensor(item["timestamp"], dtype=torch.long)
        phase = torch.as_tensor(item["phase"], dtype=torch.long)
        class_exist = item["class_exist"].float()
        target_committed = item.get(
            "target_committed", torch.zeros_like(mask)).float()
        decision_weight = item["decision_weight"].float()
        bc_valid = item["bc_valid"].float()
        temporal_binding_bc_valid = item.get(
            "temporal_binding_bc_valid", torch.ones_like(mask)).float()
        supervision_valid = item["visibility_supervision_valid"].float()
        union_mask_valid = item.get(
            "union_mask_valid", torch.zeros_like(mask)).float()
        return {
            "image": item["image"],
            "timestamp": timestamp,
            "env_prev_action": item["env_prev_action"],
            "agent_action": item["agent_action"],
            "segmentation": {"union_mask": item["segmentation"]["union_mask"]},
            "cross_view": {k: item["cross_view"][k]
                           for k in ("cross_view_image", "cross_view_obj_mask",
                                     "cross_view_obj_id")},
            "mask": mask,
            "prev_action_dropout": item.get("prev_action_dropout", torch.ones_like(mask)),
            "synthetic": synthetic,
            "phase": phase,
            "class_exist": class_exist,
            "target_committed": target_committed,
            "visibility_supervision_valid": supervision_valid,
            "decision_weight": decision_weight,
            "bc_valid": bc_valid,
            "temporal_binding_bc_valid": temporal_binding_bc_valid,
            "union_mask_valid": union_mask_valid,
        }


def parse_synthetic_pack_dirs(value):
    if value is None:
        raise ValueError("synthetic_lmdb is required (no pack directories)")
    if isinstance(value, (str, Path)):
        dirs = [part.strip() for part in re.split(r"[,:]", str(value))
                if part.strip()]
    else:
        try:
            dirs = [str(entry) for entry in value]
        except TypeError:
            raise ValueError(
                f"synthetic_lmdb must be a directory, a comma/colon-separated "
                f"string, or a list of directories, got {value!r}")
    if not dirs:
        raise ValueError(
            f"synthetic_lmdb resolved to zero pack directories: {value!r}")
    resolved = [str(Path(entry).resolve()) for entry in dirs]
    duplicates = sorted(path for path, count
                        in collections.Counter(resolved).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"synthetic_lmdb lists the same pack more than once: {duplicates}")
    return dirs


def synthetic_pack_capabilities(lmdb_dir):
    if lmdb is None:
        raise RuntimeError(
            "synthetic pack capability probing requires the optional 'lmdb' "
            "package; activate the minestudio training environment")
    path = str(Path(lmdb_dir).resolve())
    env = _open_pack_env(path)
    with env.begin() as txn:
        raw_schema = txn.get(b"__schema__")
        if raw_schema is None:
            raise ValueError(f"missing __schema__ in synthetic LMDB {path}")
        try:
            schema = json.loads(raw_schema.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid JSON __schema__ in {path}") from exc
    return {
        "path": path,
        "schema_version": int(schema.get("version", -1)),
        "all_class_visibility_census": (
            schema.get("all_class_visibility_census") is True),
        "chosen_surface_masks": bool(
            schema.get("chosen_visible_surface_masks") is True
            and schema.get("dense_segmentation_supervision") is True),
        "recognition_gate_version": schema.get("recognition_gate_version"),
    }


class MultiPackSyntheticDataset(Dataset):

    def __init__(self, lmdb_dirs, *, seed=0,
                 require_visible_surface_masks=False,
                 **pack_kwargs):
        dirs = parse_synthetic_pack_dirs(lmdb_dirs)
        self.pack_dirs = tuple(str(Path(entry).resolve()) for entry in dirs)
        self.seed = int(seed)
        requested_visible_surface = bool(require_visible_surface_masks)
        capabilities = [synthetic_pack_capabilities(path)
                        for path in self.pack_dirs]
        if requested_visible_surface and not any(
                cap["all_class_visibility_census"] for cap in capabilities):
            raise ValueError(
                "require_visible_surface_masks=True but no pack declares "
                "all_class_visibility_census; refusing a silent downgrade: "
                f"{[cap['path'] for cap in capabilities]}")
        self.packs = []
        self.pack_capabilities = []
        for pack_index, (path, cap) in enumerate(
                zip(self.pack_dirs, capabilities)):
            pack_visible_surface = (
                requested_visible_surface and cap["all_class_visibility_census"])
            pack_chosen_surface = bool(
                requested_visible_surface and cap.get("chosen_surface_masks"))
            prefix = f"pack{pack_index}/" if len(self.pack_dirs) > 1 else ""
            pack = SyntheticWindowDataset(
                path, seed=self.seed,
                require_visible_surface_masks=pack_visible_surface,
                require_chosen_surface_masks=pack_chosen_surface,
                timestamp_namespace_prefix=prefix,
                **pack_kwargs)
            self.packs.append(pack)
            self.pack_capabilities.append(dict(
                cap,
                require_visible_surface_masks=pack_visible_surface,
                require_chosen_surface_masks=pack_chosen_surface,
                timestamp_namespace_prefix=prefix))
        if len(self.packs) > 1:
            _validate_episode_namespaces(
                (cap["timestamp_namespace_prefix"] + str(ep["episode"])
                 for cap, pack in zip(self.pack_capabilities, self.packs)
                 for ep in pack.episodes),
                label="multi-pack synthetic")
        self._env_action_union_templates = None

        self.pack_bc_frames = tuple(
            sum(int(ep["length"]) for ep in pack.episodes)
            for pack in self.packs)
        raw_weights = [float(count) for count in self.pack_bc_frames]
        total = float(sum(raw_weights))
        if not (np.isfinite(total) and total > 0):
            raise ValueError(
                f"multi-pack bc-frame counts must sum > 0: {raw_weights}")
        self.pack_weights_source = "bc_frames"
        self.pack_probabilities = tuple(
            weight / total for weight in raw_weights)
        print("[synthetic-multipack] packs=" + str(len(self.packs))
              + f" weights_source={self.pack_weights_source}"
              + " p=" + str([round(p, 4) for p in self.pack_probabilities])
              + " bc_frames=" + str(list(self.pack_bc_frames))
              + " caps=" + str([
                  {"path": cap["path"],
                   "schema_version": cap["schema_version"],
                   "gate": cap["recognition_gate_version"],
                   "census": cap["all_class_visibility_census"],
                   "visible_surface": cap["require_visible_surface_masks"],
                   "chosen_surface": cap["require_chosen_surface_masks"]}
                  for cap in self.pack_capabilities]),
              flush=True)

    @property
    def episodes_with_items(self):
        if getattr(self, "_sequential_table", None) is None:
            merged = []
            for pack_index, pack in enumerate(self.packs):
                for episode_key, n_items, item_bias in pack.episodes_with_items:
                    merged.append((pack_index, episode_key, n_items, item_bias))
            order_rng = np.random.default_rng(self.seed + 0x5E0)
            order = order_rng.permutation(len(merged))
            table, index_map, bias = [], [], 0
            for merged_idx in order:
                pack_index, episode_key, n_items, item_bias = merged[int(merged_idx)]
                table.append((episode_key, n_items, bias))
                index_map.extend(
                    (pack_index, item_bias + k) for k in range(n_items))
                bias += n_items
            self._sequential_table = tuple(table)
            self._sequential_index_map = tuple(index_map)
            self._sequential_total = bias
        return self._sequential_table

    def __len__(self):
        _ = self.episodes_with_items
        return self._sequential_total

    def _build_env_action_union_templates(self):
        templates = {}
        if len(self.packs) > 1:
            probe_items = [pack[0] for pack in self.packs]
            for field in ("env_prev_action", "env_action"):
                union = {}
                for cap, item in zip(self.pack_capabilities, probe_items):
                    value = item.get(field)
                    if not isinstance(value, dict):
                        continue
                    for key, tensor in value.items():
                        spec = (tensor.dtype, tuple(tensor.shape[1:]))
                        if key in union and union[key] != spec:
                            raise ValueError(
                                f"multi-pack {field}[{key!r}] disagrees in "
                                f"dtype/shape across packs: {union[key]} vs "
                                f"{spec} ({cap['path']})")
                        union.setdefault(key, spec)
                per_pack_missing = [
                    sorted(set(union) - set(item.get(field) or {}))
                    for item in probe_items]
                if any(per_pack_missing):
                    templates[field] = union
                    print(f"[synthetic-multipack] {field} key union fill: "
                          + "; ".join(
                              f"pack{i} += {missing}"
                              for i, missing in enumerate(per_pack_missing)
                              if missing), flush=True)
        return templates

    def _fill_env_action_key_union(self, item):
        import torch as _torch
        if self._env_action_union_templates is None:
            self._env_action_union_templates = (
                self._build_env_action_union_templates())
        for field, union in self._env_action_union_templates.items():
            value = item.get(field)
            if not isinstance(value, dict):
                continue
            missing = [key for key in union if key not in value]
            if not missing:
                continue
            frames = next(iter(value.values())).shape[0]
            for key in missing:
                dtype, tail = union[key]
                value[key] = _torch.zeros((frames, *tail), dtype=dtype)
        return item

    def __getitem__(self, index):
        _ = self.episodes_with_items
        pack_index, local_index = self._sequential_index_map[int(index)]
        return self._fill_env_action_key_union(
            self.packs[pack_index][local_index])


def build_mixed_datamodule(base_cls):
    class SyntheticMixedDataModule(base_cls):
        def __init__(self, *args, synthetic_lmdb,
                     synthetic_seed=0,
                     synthetic_only=False,
                     synthetic_pad_short_min_length=48,
                     synthetic_require_visible_surface_masks=False,
                     synthetic_episode_validation_fraction=0.0,
                     synthetic_episode_split_seed=0,
                     synthetic_donor_infeasible_policy="error",
                     synthetic_goal_donor_min_resized_pixels=0,
                     synthetic_goal_donor_min_resized_short_side=0,
                     synthetic_goal_donor_max_attempts=24,
                     synthetic_camera_quantization="packed",
                     **kwargs):
            super().__init__(*args, **kwargs)
            self.synthetic_lmdb = synthetic_lmdb
            self.synthetic_seed = int(synthetic_seed)
            self.synthetic_only = bool(synthetic_only)
            self.synthetic_pad_short_min_length = (
                None if synthetic_pad_short_min_length is None
                else int(synthetic_pad_short_min_length))
            self.synthetic_require_visible_surface_masks = bool(
                synthetic_require_visible_surface_masks)
            self.synthetic_episode_validation_fraction = float(
                synthetic_episode_validation_fraction)
            self.synthetic_episode_split_seed = int(
                synthetic_episode_split_seed)
            self.synthetic_donor_infeasible_policy = str(
                synthetic_donor_infeasible_policy)
            self.synthetic_goal_donor_min_resized_pixels = int(
                synthetic_goal_donor_min_resized_pixels)
            self.synthetic_goal_donor_min_resized_short_side = int(
                synthetic_goal_donor_min_resized_short_side)
            self.synthetic_goal_donor_max_attempts = int(
                synthetic_goal_donor_max_attempts)
            self.synthetic_camera_quantization = str(synthetic_camera_quantization)

        def _synthetic_windows(self, seed, *, episode_split="all"):
            pack_dirs = parse_synthetic_pack_dirs(self.synthetic_lmdb)
            shared_kwargs = dict(
                window=int(self.data_params["win_len"]),
                seed=seed,
                pad_short_min_length=self.synthetic_pad_short_min_length,
                require_visible_surface_masks=(
                    self.synthetic_require_visible_surface_masks),
                episode_split=episode_split,
                episode_validation_fraction=(
                    self.synthetic_episode_validation_fraction),
                episode_split_seed=self.synthetic_episode_split_seed,
                donor_infeasible_policy=self.synthetic_donor_infeasible_policy,
                goal_donor_min_resized_pixels=(
                    self.synthetic_goal_donor_min_resized_pixels),
                goal_donor_min_resized_short_side=(
                    self.synthetic_goal_donor_min_resized_short_side),
                goal_donor_max_attempts=(
                    self.synthetic_goal_donor_max_attempts),
                camera_quantization=self.synthetic_camera_quantization)
            if len(pack_dirs) == 1:
                return SyntheticWindowDataset(pack_dirs[0], **shared_kwargs)
            return MultiPackSyntheticDataset(pack_dirs, **shared_kwargs)

        def setup(self, stage=None):
            if not self.synthetic_only:
                raise ValueError("training requires synthetic_only=true")
            use_split = self.synthetic_episode_validation_fraction > 0
            self.train_dataset = SyntheticOnlyDataset(self._synthetic_windows(
                self.synthetic_seed,
                episode_split=("train" if use_split else "all")))
            self.val_dataset = SyntheticOnlyDataset(self._synthetic_windows(
                self.synthetic_seed + 7919,
                episode_split=("validation" if use_split else "all")))
    return SyntheticMixedDataModule
