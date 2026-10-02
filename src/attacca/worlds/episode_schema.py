"""Schema constants and validation helpers shared by episode recording, packing and loading."""
from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping, Sequence
from numbers import Integral


STRUCTURAL_ENTITY_OCCLUSION_POLICY = (
    "current_mobs_packet_aabb_outer_voxel_union_guarded_query_coverage")
STRUCTURAL_ENTITY_QUERY_GUARD = (2.25, 4.25, 0.25, 2.25)

LEGACY_SYNTHETIC_LMDB_SCHEMA_VERSION = 5
DENSE_SYNTHETIC_LMDB_SCHEMA_VERSION = 9
COMMITTED_UNION_SYNTHETIC_LMDB_SCHEMA_VERSION = 12
FRESH_SHORT_SIDE_6_SYNTHETIC_LMDB_SCHEMA_VERSION = 13

COMMITTED_CLASS_UNION_SUPPORT_CONTRACT = (
    "recognizable_class_union_plus_committed_exact_visible_surface/v1")
COMMITTED_CLASS_VISIBLE_SURFACE_UNION_SEMANTIC = (
    "goal_class_recognizable_plus_committed_visible_surface_union/v2")

LEGACY_COMMITTED_SURFACE_TRACKING_CONTRACT_V1 = {
    "version": 1,
    "fresh_recognition": "class_exist_and_chosen_recognizable_use_v6_150px_10side",
    "identity_memory": "target_committed_plus_committed_instance_id",
    "post_commit_surface": (
        "chosen_surface_visible_is_any_exact_current_rgb_first_hit_pixel_of_"
        "committed_instance; no_area_or_short_side_gate"),
    "fully_occluded": (
        "target_committed_remains_1; chosen_surface_visible_mask_point_are_zero"),
    "fresh_commit_and_switch": (
        "same_frame_chosen_recognizable_required; below_threshold_surface_cannot_bind"),
    "model_input": "tracking_fields_are_labels_and_audit_only_never_policy_inputs",
}

COMMITTED_SURFACE_TRACKING_CONTRACT = {
    **LEGACY_COMMITTED_SURFACE_TRACKING_CONTRACT_V1,
    "version": 2,
    "fresh_recognition": "class_exist_and_chosen_recognizable_use_v7_150px_6side",
}

POSITIVE_PAIR_EPISODE_FIELDS = (
    "positive_pair_contract",
    "positive_pair_group_id",
    "positive_pair_member_goal_kind",
    "positive_pair_counterpart_goal_kind",
    "positive_pair_counterpart_episode_id",
    "positive_pair_eligible",
    "positive_pair_scene_sha256",
    "positive_pair_start_sha256",
    "positive_pair_frame0_rgb_sha256",
    "positive_pair_interaction_sha256",
    "positive_pair_census_sha256",
    "positive_pair_world_seed",
    "positive_pair_site_seed",
    "positive_pair_layout_seed",
    "positive_pair_pose_seed",
    "positive_pair_structural_scene_sha256",
    "positive_pair_route_sha256",
    "positive_pair_targeted_visibility_sha256",
    "positive_pair_strict_scene_sha256",
    "positive_pair_terrain_capture_sha256",
    "positive_pair_world_snapshot_sha256",
)


def all_visible_instances_navigation_inadmissible(
        instances, navigation_dead_ids=()):
    if not isinstance(instances, list) or not instances:
        return False
    dead_ids = frozenset(str(item) for item in (navigation_dead_ids or ()))
    for instance in instances:
        if not isinstance(instance, Mapping):
            return False
        if str(instance.get("instance_id")) in dead_ids:
            continue
        proof = instance.get("proof")
        if (not isinstance(proof, Mapping)
                or proof.get("navigation_admissible") is not False):
            return False
    return True


def _strict_nonempty_string(value, *, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_positive_pair_episode_groups(episodes, *, require_all=False,
                                          label="positive paired BC") -> dict:
    rows = list(episodes)
    episode_ids = []
    for index, episode in enumerate(rows):
        if not isinstance(episode, Mapping):
            raise ValueError(f"{label}[{index}] must be a mapping")
        episode_id = _strict_nonempty_string(
            episode.get("episode"), label=f"{label}[{index}].episode")
        episode_ids.append(episode_id)
        present = [field for field in POSITIVE_PAIR_EPISODE_FIELDS
                   if episode.get(field) is not None]
        if present:
            raise ValueError(
                f"{label} episode {episode_id} carries positive-pair metadata "
                f"{present[:3]}; paired episodes are not supported")
        if require_all:
            raise ValueError(
                f"{label} requires pair metadata for episode {episode_id}")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError(f"{label} episode ids are not unique")
    return {
        "contracts": [],
        "groups": {},
        "group_count": 0,
        "paired_episode_count": 0,
        "unpaired_episode_count": len(rows),
        "audit_tier_counts": {},
    }


def class_visibility_roster_contract(roster) -> dict:
    normalized = tuple(str(value).split(":")[-1].strip() for value in roster)
    if (not normalized or any(not value for value in normalized)
            or len(set(normalized)) != len(normalized)):
        raise ValueError("class visibility roster must be nonempty and unique")
    payload = json.dumps(list(normalized), separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return {
        "version": 1,
        "classes": list(normalized),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bit_order": "classes[i] maps to bit (1 << i), least-significant first",
        "unknown_semantics": "known_bit=0; exclude exist and point losses",
        "visible_semantics": (
            "known_bit=1 and recognizable actually-visible surface exists"),
        "absent_semantics": (
            "known_bit=1 and visible_bit=0; certified absent within finite census"),
    }

VISIBLE_SURFACE_RENDER_POSE_SOURCE = (
    "sync_partial_ticks_0_pre_position_post_rotation")

STRUCTURAL_MIN_VISIBLE_AREA_PX = 64
STRUCTURAL_MIN_VISIBLE_SHORT_SIDE_PX = 10
STRUCTURAL_MIN_FULL_CUBE_VISIBLE_AREA_PX = 150
STRUCTURAL_MIN_FULL_CUBE_VISIBLE_SHORT_SIDE_PX = 6
LEGACY_V6_MIN_FULL_CUBE_VISIBLE_SHORT_SIDE_PX = 10
LEGACY_V4_V5_MIN_FULL_CUBE_VISIBLE_SHORT_SIDE_PX = 12
STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_NUMERATOR = 1
STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_DENOMINATOR = 3

RECOGNITION_GATE_VERSION = "v7_full_cube_150_6_no_fill"
RECOGNITION_GATE_VERSION_LEGACY_V6 = "v6_full_cube_150_10_no_fill"
RECOGNITION_GATE_VERSION_LEGACY_V5 = "v5_full_cube_150_12_no_fill"
RECOGNITION_GATE_VERSION_LEGACY_V4 = "v4_full_cube_150_12"
RECOGNITION_GATE_VERSION_LEGACY_V3 = "recognition_gate_v3_180_21"
LEGACY_V3_MIN_FULL_CUBE_VISIBLE_AREA_PX = 180
LEGACY_V3_MIN_FULL_CUBE_VISIBLE_SHORT_SIDE_PX = 21


def normalize_recognition_gate_version(stamp):
    if stamp is None or stamp == 3 or stamp == RECOGNITION_GATE_VERSION_LEGACY_V3:
        return RECOGNITION_GATE_VERSION_LEGACY_V3
    if stamp == RECOGNITION_GATE_VERSION:
        return RECOGNITION_GATE_VERSION
    if stamp == RECOGNITION_GATE_VERSION_LEGACY_V6:
        return RECOGNITION_GATE_VERSION_LEGACY_V6
    if stamp == RECOGNITION_GATE_VERSION_LEGACY_V5:
        return RECOGNITION_GATE_VERSION_LEGACY_V5
    if stamp == RECOGNITION_GATE_VERSION_LEGACY_V4:
        return RECOGNITION_GATE_VERSION_LEGACY_V4
    raise ValueError(f"unknown recognition gate version stamp: {stamp!r}")


def committed_surface_tracking_contract_for_gate(gate_version):
    gate_version = normalize_recognition_gate_version(gate_version)
    if gate_version == RECOGNITION_GATE_VERSION:
        return COMMITTED_SURFACE_TRACKING_CONTRACT
    if gate_version == RECOGNITION_GATE_VERSION_LEGACY_V6:
        return LEGACY_COMMITTED_SURFACE_TRACKING_CONTRACT_V1
    raise ValueError(
        "committed-surface tracking is only defined for v6/v7 fresh gates: "
        f"{gate_version!r}")


def require_matching_recognition_gate_version(expected, observed, *, label):
    expected_version = normalize_recognition_gate_version(expected)
    observed_version = normalize_recognition_gate_version(observed)
    if observed_version != expected_version:
        raise ValueError(
            f"{label} recognition gate version mismatch: "
            f"{observed_version!r}!={expected_version!r}")
    return expected_version


def full_cube_recognition_thresholds(
        gate_version=RECOGNITION_GATE_VERSION):
    gate_version = normalize_recognition_gate_version(gate_version)
    if gate_version == RECOGNITION_GATE_VERSION:
        return (STRUCTURAL_MIN_FULL_CUBE_VISIBLE_AREA_PX,
                STRUCTURAL_MIN_FULL_CUBE_VISIBLE_SHORT_SIDE_PX)
    if gate_version == RECOGNITION_GATE_VERSION_LEGACY_V6:
        return (STRUCTURAL_MIN_FULL_CUBE_VISIBLE_AREA_PX,
                LEGACY_V6_MIN_FULL_CUBE_VISIBLE_SHORT_SIDE_PX)
    if gate_version in (RECOGNITION_GATE_VERSION_LEGACY_V5,
                        RECOGNITION_GATE_VERSION_LEGACY_V4):
        return (STRUCTURAL_MIN_FULL_CUBE_VISIBLE_AREA_PX,
                LEGACY_V4_V5_MIN_FULL_CUBE_VISIBLE_SHORT_SIDE_PX)
    return (LEGACY_V3_MIN_FULL_CUBE_VISIBLE_AREA_PX,
            LEGACY_V3_MIN_FULL_CUBE_VISIBLE_SHORT_SIDE_PX)


def structural_surface_recognizable(visible_area_px: int,
                                    visible_short_side_px: int) -> bool:
    for value, label in (
            (visible_area_px, "visible_area_px"),
            (visible_short_side_px, "visible_short_side_px")):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    return bool(
        int(visible_area_px) >= STRUCTURAL_MIN_VISIBLE_AREA_PX
        and int(visible_short_side_px) >= STRUCTURAL_MIN_VISIBLE_SHORT_SIDE_PX)


def _structural_visible_bbox_area(visible_bbox_px) -> tuple[int, int]:
    if (not isinstance(visible_bbox_px, (tuple, list))
            or len(visible_bbox_px) != 4
            or any(isinstance(value, bool) or not isinstance(value, Integral)
                   for value in visible_bbox_px)):
        raise ValueError("visible_bbox_px must be four inclusive integer coordinates")
    x0, y0, x1, y1 = (int(value) for value in visible_bbox_px)
    if x0 < 0 or y0 < 0 or x1 < x0 or y1 < y0:
        raise ValueError("visible_bbox_px must be a nonempty nonnegative bbox")
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    return width * height, min(width, height)


def structural_full_cube_surface_recognizable(
        visible_area_px: int, visible_short_side_px: int,
        visible_bbox_px, *,
        gate_version=RECOGNITION_GATE_VERSION) -> bool:
    gate_version = normalize_recognition_gate_version(gate_version)
    min_area, min_short = full_cube_recognition_thresholds(gate_version)
    structural_surface_recognizable(visible_area_px, visible_short_side_px)
    bbox_area, bbox_short = _structural_visible_bbox_area(visible_bbox_px)
    if int(visible_short_side_px) != bbox_short:
        raise ValueError(
            "visible_short_side_px disagrees with visible_bbox_px")
    if int(visible_area_px) > bbox_area:
        raise ValueError("visible_area_px exceeds visible_bbox_px area")
    fill_pass = bool(
        STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_DENOMINATOR
        * int(visible_area_px)
        >= STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_NUMERATOR * bbox_area)
    fill_is_behavior_gate = gate_version in (
        RECOGNITION_GATE_VERSION_LEGACY_V3,
        RECOGNITION_GATE_VERSION_LEGACY_V4,
    )
    return bool(
        int(visible_area_px) >= min_area
        and int(visible_short_side_px) >= min_short
        and (fill_pass or not fill_is_behavior_gate))


def structural_surface_recognition_fields(visible_area_px: int,
                                          visible_short_side_px: int, *,
                                          visible_bbox_px=None,
                                          require_full_cube_fill: bool = False,
                                          gate_version=RECOGNITION_GATE_VERSION
                                          ) -> dict:
    if not isinstance(require_full_cube_fill, bool):
        raise ValueError("require_full_cube_fill must be boolean")
    gate_version = normalize_recognition_gate_version(gate_version)
    legacy_v3 = gate_version == RECOGNITION_GATE_VERSION_LEGACY_V3
    legacy_fill_gate = gate_version in (
        RECOGNITION_GATE_VERSION_LEGACY_V3,
        RECOGNITION_GATE_VERSION_LEGACY_V4,
    )
    min_area, min_short = full_cube_recognition_thresholds(gate_version)
    base_recognizable = structural_surface_recognizable(
        visible_area_px, visible_short_side_px)
    full_cube_result = "not_applicable"
    bbox_area = None
    full_cube_recognizable = True
    full_cube_area_recognizable = True
    full_cube_short_side_recognizable = True
    full_cube_fill_recognizable = True
    if require_full_cube_fill:
        bbox_area, _bbox_short = _structural_visible_bbox_area(visible_bbox_px)
        full_cube_area_recognizable = bool(
            int(visible_area_px) >= min_area)
        full_cube_short_side_recognizable = bool(
            int(visible_short_side_px) >= min_short)
        full_cube_fill_recognizable = bool(
            STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_DENOMINATOR
            * int(visible_area_px)
            >= STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_NUMERATOR * bbox_area)
        full_cube_recognizable = structural_full_cube_surface_recognizable(
            visible_area_px, visible_short_side_px, visible_bbox_px,
            gate_version=gate_version)
        if legacy_fill_gate:
            full_cube_result = "pass" if full_cube_recognizable else "fail"
        else:
            full_cube_result = (
                "audit_only_pass" if full_cube_fill_recognizable
                else "audit_only_fail")
    recognizable = bool(
        full_cube_recognizable if require_full_cube_fill else base_recognizable)
    perceptibility_contract = "actual_visible_surface_area>=64_and_short_side>=10"
    if require_full_cube_fill:
        if gate_version == RECOGNITION_GATE_VERSION:
            perceptibility_contract = (
                f"actual_visible_surface_area>={min_area}_and_short_side>="
                f"{min_short};bbox_fill_audit_only")
        else:
            perceptibility_contract += (
                f"_and_full_cube_area>={min_area}_and_short_side>={min_short}")
            perceptibility_contract += (
                "_and_bbox_fill>=1/3" if legacy_fill_gate
                else ";bbox_fill_audit_only")
    fields = {
        "visible": True,
        "surface_visible": True,
        "surface_visibility_contract": "any_actual_visible_surface_pixel",
        "perceptible": recognizable,
        "oracle_recognizable": recognizable,
        "perceptibility_contract": perceptibility_contract,
        "recognition_threshold_64px_10side": base_recognizable,
        "recognition_gate_version": (3 if legacy_v3 else gate_version),
        "recognition_full_cube_bbox_fill_gate_applied": bool(
            require_full_cube_fill and legacy_fill_gate),
        "recognition_full_cube_bbox_fill_gate_result": full_cube_result,
    }
    if require_full_cube_fill:
        fields.update({
            "visible_bbox_area_px": int(bbox_area),
            "visible_bbox_fill_numerator_px": int(visible_area_px),
            "visible_bbox_fill_denominator_px": int(bbox_area),
            "recognition_full_cube_min_bbox_fill_numerator": (
                STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_NUMERATOR),
            "recognition_full_cube_min_bbox_fill_denominator": (
                STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_DENOMINATOR),
            "recognition_full_cube_min_visible_area_px": int(min_area),
            "recognition_full_cube_min_visible_short_side_px": int(min_short),
        })
        if legacy_v3:
            fields.update({
                "recognition_threshold_full_cube_area_180px": bool(
                    full_cube_area_recognizable),
                "recognition_threshold_full_cube_short_side_21px": bool(
                    full_cube_short_side_recognizable),
                "recognition_threshold_full_cube_bbox_fill_one_third": bool(
                    full_cube_fill_recognizable),
            })
        else:
            fields.update({
                "recognition_full_cube_area_pass": bool(
                    full_cube_area_recognizable),
                "recognition_full_cube_short_side_pass": bool(
                    full_cube_short_side_recognizable),
                "recognition_full_cube_bbox_fill_pass": bool(
                    full_cube_fill_recognizable),
            })
    return fields


def structural_recognition_contract(
        gate_version=RECOGNITION_GATE_VERSION) -> dict:
    gate_version = normalize_recognition_gate_version(gate_version)
    legacy_v3 = gate_version == RECOGNITION_GATE_VERSION_LEGACY_V3
    legacy_v4 = gate_version == RECOGNITION_GATE_VERSION_LEGACY_V4
    legacy_v5 = gate_version == RECOGNITION_GATE_VERSION_LEGACY_V5
    legacy_v6 = gate_version == RECOGNITION_GATE_VERSION_LEGACY_V6
    legacy_fill_gate = legacy_v3 or legacy_v4
    min_area, min_short = full_cube_recognition_thresholds(gate_version)
    contract = {
        "version": (3 if legacy_v3 else 4 if legacy_v4 else
                    5 if legacy_v5 else 6 if legacy_v6 else 7),
        "surface_mask_support": "any_actual_visible_surface_pixel",
        "behavior_instance_gate": (
            "mine_full_cube_versioned_area_short_side; shaped_objects_base_"
            "area_short_side; full_cube_bbox_fill_audit_only"
            if gate_version == RECOGNITION_GATE_VERSION else
            "per_instance_area_short_side_plus_full_cube_bbox_fill"
            if legacy_fill_gate else
            "per_instance_area_short_side; full_cube_bbox_fill_audit_only"),
        "min_visible_area_px": STRUCTURAL_MIN_VISIBLE_AREA_PX,
        "min_visible_short_side_px": STRUCTURAL_MIN_VISIBLE_SHORT_SIDE_PX,
        "min_full_cube_visible_area_px": int(min_area),
        "min_full_cube_visible_short_side_px": int(min_short),
        "full_cube_scope": (
            "mine_target_roster_only; exclude_shaped_use_mobs_water_lava"),
        "below_threshold_policy": (
            "retain_in_geometry_audit_only; exclude_from_visible_instances_"
            "class_exist_chosen_and_commit"),
    }
    fill_contract = {
        "numerator": STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_NUMERATOR,
        "denominator": STRUCTURAL_MIN_FULL_CUBE_BBOX_FILL_DENOMINATOR,
        "comparison": "denominator*visible_area_px>=numerator*bbox_area_px",
    }
    if legacy_fill_gate:
        contract["min_full_cube_bbox_fill"] = fill_contract
    else:
        contract["full_cube_bbox_fill_audit"] = {
            **fill_contract,
            "behavior_gate": False,
        }
    if not legacy_v3:
        contract["recognition_gate_version"] = gate_version
        contract["phase_purity"] = (
            "class_exist_visible_instances_chosen_use_the_same_current_rgb_"
            "rule_in_every_phase; no_sticky_positive; no_post_commit_hysteresis")
    return contract


def recognition_contract_gate_version(contract):
    if contract is None:
        return RECOGNITION_GATE_VERSION_LEGACY_V3
    for candidate in (RECOGNITION_GATE_VERSION,
                      RECOGNITION_GATE_VERSION_LEGACY_V6,
                      RECOGNITION_GATE_VERSION_LEGACY_V5,
                      RECOGNITION_GATE_VERSION_LEGACY_V4,
                      RECOGNITION_GATE_VERSION_LEGACY_V3):
        if contract == structural_recognition_contract(candidate):
            return candidate
    raise ValueError(
        f"unrecognized structural recognition contract: {contract!r}")

VIEWMODEL_POST_ACTION_CONTROL_TICKS = 6
VIEWMODEL_PIXEL_SUPERVISION_FIELD = "viewmodel_pixel_supervision_valid"
VIEWMODEL_OCCLUSION_AUDIT_FIELD = "viewmodel_occlusion_audit"
VIEWMODEL_CONTROL_CLOCK = (
    "step_kind_control_only; sensor_ticks_do_not_decrement_conservative")
RENDERER_VIEWMODEL_ACTION_MARKER = (
    "_xbench_renderer_viewmodel_twopass_mask_v1")
RENDERER_VIEWMODEL_MASK_CONTRACT = (
    "minecraft_java_1_16_5_same_tick_hand_only_alpha/v1")


def viewmodel_occlusion_contract() -> dict:
    return {
        "version": 1,
        "per_frame_valid_field": VIEWMODEL_PIXEL_SUPERVISION_FIELD,
        "per_frame_audit_field": VIEWMODEL_OCCLUSION_AUDIT_FIELD,
        "invalid_current_actions": ["attack", "use"],
        "post_action_control_ticks": VIEWMODEL_POST_ACTION_CONTROL_TICKS,
        "clock": VIEWMODEL_CONTROL_CLOCK,
        "rationale": (
            "vanilla_1_16_5_default_arm_swing_is_6_ticks; partialTicks=0 "
            "and unavailable framebuffer depth require invalidating the action "
            "RGB plus six following logged control observations"),
        "scope": (
            "chosen pixel-mask and centroid only; class_exist and BC remain valid"),
    }


def _action_pressed(action, key: str) -> bool:
    if not isinstance(action, Mapping):
        raise ValueError("viewmodel action must be a mapping")
    value = action.get(key, 0)
    if hasattr(value, "reshape"):
        flat = value.reshape(-1)
        if getattr(flat, "size", 0) != 1:
            raise ValueError(f"viewmodel action {key!r} is not one scalar")
        value = flat[0]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(f"viewmodel action {key!r} is not one scalar")
        value = value[0]
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or int(value) not in (0, 1)):
        raise ValueError(
            f"viewmodel action {key!r} must be an integer 0/1")
    return int(value) == 1


def viewmodel_supervision_step(action, step_kind, remaining_controls: int):
    if (isinstance(remaining_controls, bool)
            or not isinstance(remaining_controls, int)
            or not 0 <= remaining_controls <= VIEWMODEL_POST_ACTION_CONTROL_TICKS):
        raise ValueError("viewmodel remaining_controls is outside its horizon")
    if step_kind not in ("control", "sensor"):
        raise ValueError(f"unsupported viewmodel step_kind {step_kind!r}")
    is_control = step_kind == "control"
    attack = bool(is_control and _action_pressed(action, "attack"))
    use = bool(is_control and _action_pressed(action, "use"))
    active = bool(attack or use)
    renderer_masked = _action_pressed(
        action, RENDERER_VIEWMODEL_ACTION_MARKER)
    if renderer_masked:
        audit = {
            "version": 2,
            "active_attack": attack,
            "active_use": use,
            "post_action_control_ticks_before": int(remaining_controls),
            "post_action_control_ticks": VIEWMODEL_POST_ACTION_CONTROL_TICKS,
            "control_clock": VIEWMODEL_CONTROL_CLOCK,
            "control_tick_advanced": bool(is_control),
            "renderer_viewmodel_mask_contract": (
                RENDERER_VIEWMODEL_MASK_CONTRACT),
        }
        return 1, audit, 0
    valid = int(not active and remaining_controls == 0)
    audit = {
        "version": 1,
        "active_attack": attack,
        "active_use": use,
        "post_action_control_ticks_before": int(remaining_controls),
        "post_action_control_ticks": VIEWMODEL_POST_ACTION_CONTROL_TICKS,
        "control_clock": VIEWMODEL_CONTROL_CLOCK,
        "control_tick_advanced": bool(is_control),
    }
    next_remaining = int(remaining_controls)
    if is_control:
        next_remaining = (VIEWMODEL_POST_ACTION_CONTROL_TICKS if active else
                          max(0, next_remaining - 1))
    return valid, audit, next_remaining


def validate_viewmodel_supervision_record(
        row, action, step_kind, remaining_controls: int, *, label: str) -> int:
    expected_valid, expected_audit, next_remaining = viewmodel_supervision_step(
        action, step_kind, remaining_controls)
    observed_valid = row.get(VIEWMODEL_PIXEL_SUPERVISION_FIELD)
    if (isinstance(observed_valid, bool) or not isinstance(observed_valid, int)
            or observed_valid not in (0, 1)
            or observed_valid != expected_valid):
        raise ValueError(
            f"{label} {VIEWMODEL_PIXEL_SUPERVISION_FIELD} mismatch: "
            f"{observed_valid!r}!={expected_valid}")
    observed_audit = row.get(VIEWMODEL_OCCLUSION_AUDIT_FIELD)
    if observed_audit != expected_audit:
        raise ValueError(
            f"{label} {VIEWMODEL_OCCLUSION_AUDIT_FIELD} mismatch: "
            f"{observed_audit!r}!={expected_audit!r}")
    return next_remaining


def _finite_numbers(value, length: int, *, label: str) -> list[float]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or len(value) != length):
        raise ValueError(f"{label} must contain {length} finite numbers")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} must contain {length} finite numbers")
        item = float(item)
        if not math.isfinite(item):
            raise ValueError(f"{label} must contain {length} finite numbers")
        out.append(item)
    return out


def _strict_count(value, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def structural_mob_aabb_voxel_cover(kind, position):
    px, py, pz = _finite_numbers(position, 3, label="structural mob position")
    bare = str(kind).lower().split(":")[-1].replace("entity", "")
    fish = ("cod", "salmon", "pufferfish", "tropicalfish")
    small = ("chicken", "rabbit", "bee", "bat", "parrot")
    medium = (
        "sheep", "cow", "pig", "mooshroom", "wolf", "cat", "fox",
        "ocelot", "goat", "panda", "polar_bear")
    riding = ("horse", "donkey", "mule", "llama", "traderllama")
    humanoid = (
        "zombie", "skeleton", "stray", "husk", "drowned", "creeper",
        "villager", "witch", "pillager", "vindicator", "piglin")
    if any(token in bare for token in fish):
        half, below, height = 0.70, 0.25, 1.00
    elif any(token in bare for token in small):
        half, below, height = 0.65, 0.25, 1.35
    elif "squid" in bare or "dolphin" in bare:
        half, below, height = 1.00, 0.25, 1.35
    elif any(token in bare for token in riding):
        half, below, height = 1.00, 0.25, 2.35
    elif any(token in bare for token in humanoid):
        half, below, height = 0.70, 0.25, 2.35
    elif "spider" in bare:
        half, below, height = 1.00, 0.25, 1.35
    elif any(token in bare for token in medium):
        half, below, height = 0.80, 0.25, 1.85
    elif any(token in bare for token in ("enderman", "irongolem", "ravager")):
        half, below, height = 1.25, 0.25, 3.50
    else:
        half, below, height = 2.25, 0.25, 4.25
    lo = (px - half, py - below, pz - half)
    hi = (px + half, py + height, pz + half)
    starts = tuple(int(math.floor(value)) for value in lo)
    stops = tuple(int(math.ceil(value)) for value in hi)
    cells = frozenset(
        (xx, yy, zz)
        for xx in range(starts[0], stops[0])
        for yy in range(starts[1], stops[1])
        for zz in range(starts[2], stops[2]))
    return cells, lo, hi


def validate_structural_entity_occlusion_scope(
        entity, player_position) -> dict:
    if not isinstance(entity, Mapping):
        raise ValueError("structural entity occlusion scope must be a mapping")
    player = _finite_numbers(player_position, 3, label="player_position")
    if entity.get("policy") != STRUCTURAL_ENTITY_OCCLUSION_POLICY:
        raise ValueError("structural entity occlusion policy mismatch")
    for field in ("mobs_packet_present", "mobs_packet_parse_valid",
                  "mob_query_scope_available",
                  "certified_target_ray_domain_covered"):
        if entity.get(field) is not True:
            raise ValueError(f"{field} must be true")

    reported = _strict_count(
        entity.get("reported_mob_count"), label="reported_mob_count")
    normalized = _strict_count(
        entity.get("normalized_mob_count"), label="normalized_mob_count")
    aabb_count = _strict_count(
        entity.get("conservative_aabb_count"),
        label="conservative_aabb_count")
    if normalized != reported or aabb_count != reported:
        raise ValueError("reported/normalized/AABB mob counts disagree")

    relative_raw = entity.get("mob_query_relative_xyz_half_open")
    if (not isinstance(relative_raw, Sequence)
            or isinstance(relative_raw, (str, bytes))
            or len(relative_raw) != 6
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in relative_raw)):
        raise ValueError("structural mob query relative box must be six integers")
    relative = [int(item) for item in relative_raw]
    absolute = _finite_numbers(
        entity.get("mob_query_absolute_xyz_half_open"), 6,
        label="structural mob query absolute box")
    if not (relative[0] < relative[1] and relative[2] < relative[3]
            and relative[4] < relative[5]):
        raise ValueError("structural mob query box is empty/reversed")
    expected_absolute = [
        player[0] + relative[0], player[0] + relative[1],
        player[1] + relative[2], player[1] + relative[3],
        player[2] + relative[4], player[2] + relative[5],
    ]
    if any(abs(absolute[i] - expected_absolute[i]) > 1e-5
           for i in range(6)):
        raise ValueError("structural mob absolute box disagrees with player+relative")

    guard = entity.get("query_guard")
    if not isinstance(guard, Mapping):
        raise ValueError("structural entity query guard is missing")
    observed_guard = (
        guard.get("horizontal_each_side"),
        guard.get("lower_y_for_outside_body_top"),
        guard.get("upper_y_for_outside_body_bottom"),
    )
    observed_guard = _finite_numbers(
        observed_guard, 3, label="structural entity query guard")
    expected_guard = STRUCTURAL_ENTITY_QUERY_GUARD[:3]
    if any(abs(observed_guard[i] - expected_guard[i]) > 1e-9
           for i in range(3)):
        raise ValueError("structural entity query guard mismatch")

    guarded = _finite_numbers(
        entity.get("guarded_mob_query_absolute_xyz_half_open"), 6,
        label="guarded structural mob query box")
    horizontal, lower_y, upper_y, horizontal_z = (
        STRUCTURAL_ENTITY_QUERY_GUARD)
    expected_guarded = [
        absolute[0] + horizontal, absolute[1] - horizontal,
        absolute[2] + lower_y, absolute[3] - upper_y,
        absolute[4] + horizontal_z, absolute[5] - horizontal_z,
    ]
    if (not (guarded[0] < guarded[1] and guarded[2] < guarded[3]
             and guarded[4] < guarded[5])
            or any(abs(guarded[i] - expected_guarded[i]) > 1e-5
                   for i in range(6))):
        raise ValueError("guarded structural mob query box mismatch")

    raw_aabbs = entity.get("conservative_aabbs")
    if not isinstance(raw_aabbs, list) or len(raw_aabbs) != aabb_count:
        raise ValueError("conservative AABB list/count mismatch")
    observation_indices = set()
    exact_union = set()
    normalized_aabbs = []
    for index, raw in enumerate(raw_aabbs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"conservative_aabbs[{index}] is not a mapping")
        observation_index = raw.get("observation_index")
        if (isinstance(observation_index, bool)
                or not isinstance(observation_index, int)
                or observation_index < 0
                or observation_index in observation_indices):
            raise ValueError("conservative AABB observation indices are invalid")
        observation_indices.add(observation_index)
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"conservative_aabbs[{index}] kind is missing")
        position = _finite_numbers(
            raw.get("position"), 3, label=f"AABB {index} position")
        if not (absolute[0] <= position[0] < absolute[1]
                and absolute[2] <= position[1] < absolute[3]
                and absolute[4] <= position[2] < absolute[5]):
            raise ValueError(f"conservative_aabbs[{index}] base is outside query")
        lo = _finite_numbers(raw.get("lo"), 3, label=f"AABB {index} lo")
        hi = _finite_numbers(raw.get("hi"), 3, label=f"AABB {index} hi")
        if any(lo[axis] >= hi[axis] for axis in range(3)):
            raise ValueError(f"conservative_aabbs[{index}] is empty/reversed")
        outer = _strict_count(
            raw.get("outer_voxel_count"), label=f"AABB {index} voxel count")
        expected_cells, expected_lo, expected_hi = (
            structural_mob_aabb_voxel_cover(kind, position))
        if (any(abs(lo[axis] - expected_lo[axis]) > 1e-6
                or abs(hi[axis] - expected_hi[axis]) > 1e-6
                for axis in range(3))
                or outer <= 0 or outer != len(expected_cells)):
            raise ValueError(f"conservative_aabbs[{index}] voxel count mismatch")
        exact_union.update(expected_cells)
        normalized_aabbs.append({
            "observation_index": int(observation_index),
            "kind": kind,
            "position": position,
            "lo": lo,
            "hi": hi,
            "outer_voxel_count": outer,
        })
    if observation_indices != set(range(reported)):
        raise ValueError("conservative AABBs do not cover every reported mob row")

    voxel_count = _strict_count(
        entity.get("occluder_voxel_count"), label="occluder_voxel_count")
    if voxel_count != len(exact_union):
        raise ValueError("occluder union count is not the exact AABB voxel union")

    normalized_entity = dict(entity)
    normalized_entity.update({
        "mob_query_relative_xyz_half_open": relative,
        "mob_query_absolute_xyz_half_open": absolute,
        "guarded_mob_query_absolute_xyz_half_open": guarded,
        "conservative_aabbs": normalized_aabbs,
    })
    return normalized_entity
