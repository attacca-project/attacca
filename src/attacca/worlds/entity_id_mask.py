"""Exact per-instance entity surfaces and per-class visible unions computed from the renderer entity-ID image."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from attacca.worlds.episode_schema import RECOGNITION_GATE_VERSION
from attacca.worlds.episode_schema import VISIBLE_SURFACE_RENDER_POSE_SOURCE
from attacca.worlds.episode_schema import class_visibility_roster_contract
from attacca.worlds.episode_schema import structural_full_cube_surface_recognizable
from attacca.worlds.episode_schema import structural_surface_recognition_fields


HUNT_ENTITY_CLASSES = ("sheep", "cow", "pig", "chicken", "horse")
HUNT_TRAIN_ENTITY_CLASSES = (
    "cow", "pig", "chicken", "white_sheep", "gray_sheep",
    "light_gray_sheep", "brown_sheep",
)
HUNT_ZERO_SHOT_ENTITY_CLASSES = (
    "blue_sheep", "purple_sheep", "mooshroom",
    "creamy_trader_llama", "white_trader_llama",
    "panda", "turtle", "llama", "donkey", "fox", "cat", "ocelot",
    "polar_bear", "rabbit", "parrot",
)
HUNT_HUMAN_ENTITY_CLASSES = HUNT_TRAIN_ENTITY_CLASSES
HUNT_CLASS_VISIBILITY_ROSTER_CONTRACT = class_visibility_roster_contract(
    HUNT_TRAIN_ENTITY_CLASSES)
HUNT_CLASS_VISIBILITY_INDEX = {
    kind: index for index, kind in enumerate(HUNT_TRAIN_ENTITY_CLASSES)}
ENTITY_ID_CLASSES = (
    *HUNT_ENTITY_CLASSES, "panda", "mooshroom", "black_sheep",
    "white_sheep", "gray_sheep", "light_gray_sheep", "brown_sheep",
    "blue_sheep", "purple_sheep",
    "creamy_trader_llama", "white_trader_llama",
    "red_sheep", "yellow_sheep", "orange_sheep",
    "turtle", "llama", "donkey", "fox", "cat", "ocelot",
    "polar_bear", "rabbit", "parrot", "wolf",
)
ENTITY_ID_SEMANTIC_CODES = {
    name: index + 1 for index, name in enumerate(ENTITY_ID_CLASSES)
}
ENTITY_ID_CODE_CLASSES = {
    code: name for name, code in ENTITY_ID_SEMANTIC_CODES.items()
}
GOAL_ENTITY_KINDS = {
    **{kind: (kind,) for kind in ENTITY_ID_CLASSES},
    "sheep": (
        "sheep", "black_sheep", "white_sheep", "gray_sheep",
        "light_gray_sheep", "brown_sheep", "blue_sheep", "purple_sheep",
        "red_sheep", "yellow_sheep", "orange_sheep",
    ),
}
ENTITY_ID_MASK_CONTRACT = (
    "minecraft_java_1_16_5_same_tick_textured_entity_id_depth_occluded/v1")
ENTITY_ID_OCCLUSION_CONTRACT = (
    "same_tick_world_depth_then_alpha_textured_entity_id_geometry")
ENTITY_ID_PROJECTION_CONTRACT = (
    "renderer_model_alpha_textured_entity_id_no_aabb_or_bbox_fill")
VISIBLE_SURFACE_MASK_SEMANTIC = "chosen_instance_actual_visible_surface/v1"
HUNT_RECOGNITION_GATE_VERSION = RECOGNITION_GATE_VERSION
HUNT_RECOGNITION_CONTRACT = (
    "exact_visible_entity_surface_area>=150_short_side>=6_"
    "mine_v7_bbox_fill_audit_only/v1")
HUNT_ATTACK_REACH = 3.0
HUNT_VISIBILITY_PHASE_CONTRACT = (
    "explore_absent_approach_recognizable_interact_exact_target_surface_"
    "attack_within_3_blocks/v3")


def hunt_entity_surface_recognizable(
        visible_area_px: int, visible_short_side_px: int,
        visible_bbox_px) -> bool:
    return structural_full_cube_surface_recognizable(
        visible_area_px, visible_short_side_px, visible_bbox_px,
        gate_version=HUNT_RECOGNITION_GATE_VERSION)


def _certified_pixel_runs_yx(mask: np.ndarray) -> list[list[int]]:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or np.any((binary != 0) & (binary != 1)):
        raise ValueError("certified entity mask must be binary and 2-D")
    padded = np.pad(binary, ((0, 0), (1, 1)), constant_values=0)
    edges = np.diff(padded.astype(np.int8), axis=1)
    starts = np.argwhere(edges == 1)
    ends = np.argwhere(edges == -1)
    if (starts.shape != ends.shape
            or not np.array_equal(starts[:, 0], ends[:, 0])):
        raise RuntimeError("entity mask run boundaries are unbalanced")
    return np.column_stack((
        starts[:, 0], starts[:, 1], ends[:, 1] - 1,
    )).astype(np.int64).tolist()


@dataclass(frozen=True)
class EntitySurface:
    kind: str
    entity_id: int
    instance_id: str
    mask: np.ndarray
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    short_side_px: int
    recognizable: bool

    def proof(self) -> dict:
        yy, xx = np.nonzero(self.mask)
        center_x, center_y = 0.5 * (self.mask.shape[1] - 1), 0.5 * (
            self.mask.shape[0] - 1)
        nearest = int(np.argmin(
            (xx.astype(np.float64) - center_x) ** 2
            + (yy.astype(np.float64) - center_y) ** 2))
        aim_pixel = [int(xx[nearest]), int(yy[nearest])]
        proof = {
            "surface_visible": True,
            "visible_area_px": int(self.area_px),
            "visible_short_side_px": int(self.short_side_px),
            "visible_bbox_px": list(self.bbox_xyxy),
            "certified_mask_shape": list(self.mask.shape),
            "certified_pixel_runs_yx": _certified_pixel_runs_yx(self.mask),
            "certified_pixel_count": int(self.area_px),
            "dense_segmentation_supervision": True,
            "certified_mask_semantics": VISIBLE_SURFACE_MASK_SEMANTIC,
            "recognition_shape": "renderer_textured_entity",
            "mask_contract": ENTITY_ID_MASK_CONTRACT,
            "visibility_measure": "same_tick_renderer_entity_id_visible_surface",
            "render_pose_source": VISIBLE_SURFACE_RENDER_POSE_SOURCE,
            "projection_contract": ENTITY_ID_PROJECTION_CONTRACT,
            "occlusion_contract": ENTITY_ID_OCCLUSION_CONTRACT,
            "tested_ray_count": 0,
            "aim_pixel": aim_pixel,
            "sample_pixel": aim_pixel,
        }
        proof.update(structural_surface_recognition_fields(
            int(self.area_px), int(self.short_side_px),
            visible_bbox_px=list(self.bbox_xyxy),
            require_full_cube_fill=True,
            gate_version=HUNT_RECOGNITION_GATE_VERSION))
        proof["hunt_recognition_contract"] = HUNT_RECOGNITION_CONTRACT
        if bool(proof["oracle_recognizable"]) != bool(self.recognizable):
            raise RuntimeError("entity recognition predicate/proof disagree")
        return proof

    def record(self, *, chosen=False) -> dict:
        return {
            "instance_id": self.instance_id,
            "kind": self.kind,
            "type": self.kind,
            "entity_id": int(self.entity_id),
            "semantic_code": int(ENTITY_ID_SEMANTIC_CODES[self.kind]),
            "visible": 1,
            "chosen": bool(chosen),
            "identity_scope": "stable_client_entity_id_within_episode",
            "proof": self.proof(),
        }


def _strict_packed_ids(packed_ids) -> np.ndarray:
    packed = np.asarray(packed_ids)
    if packed.ndim != 2 or packed.dtype != np.uint32:
        raise ValueError("packed entity IDs must be a 2-D uint32 array")
    semantic = packed >> 16
    entity_id = packed & np.uint32(0xffff)
    supported = np.asarray(tuple(ENTITY_ID_CODE_CLASSES), dtype=np.uint32)
    if np.any((semantic != 0) & ~np.isin(semantic, supported)):
        raise ValueError("packed entity IDs contain an unsupported semantic code")
    if np.any((semantic == 0) != (entity_id == 0)):
        raise ValueError("packed entity background/identity disagree")
    return packed


def occlude_entity_ids_by_viewmodel(packed_ids, viewmodel_mask) -> np.ndarray:
    packed = _strict_packed_ids(packed_ids)
    mask = np.asarray(viewmodel_mask)
    if mask.shape != packed.shape or mask.dtype not in (np.uint8, np.bool_):
        raise ValueError(
            "viewmodel mask must be binary uint8/bool with entity-ID shape")
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("viewmodel mask must be binary")
    visible = np.ascontiguousarray(packed.copy())
    visible[mask.astype(bool)] = np.uint32(0)
    return visible


def entity_surfaces(packed_ids) -> list[EntitySurface]:
    packed = _strict_packed_ids(packed_ids)
    surfaces: list[EntitySurface] = []
    for packed_id in np.unique(packed):
        value = int(packed_id)
        if value == 0:
            continue
        semantic = value >> 16
        entity_id = value & 0xffff
        kind = ENTITY_ID_CODE_CLASSES[semantic]
        mask = np.ascontiguousarray(packed == packed_id, dtype=np.uint8)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise RuntimeError("visible packed entity unexpectedly has no pixels")
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        short_side = min(bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
        area = int(mask.sum())
        surfaces.append(EntitySurface(
            kind=kind,
            entity_id=entity_id,
            instance_id=f"entity:{entity_id}",
            mask=mask,
            area_px=area,
            bbox_xyxy=bbox,
            short_side_px=short_side,
            recognizable=hunt_entity_surface_recognizable(
                area, short_side, bbox),
        ))
    return surfaces


def hunt_train_class_census_from_surfaces(
        surfaces: list[EntitySurface]) -> dict:
    unexpected = sorted({
        surface.kind for surface in surfaces
        if surface.kind not in HUNT_CLASS_VISIBILITY_INDEX})
    if unexpected:
        raise RuntimeError(
            "non-train renderer class leaked into Hunt train census: "
            f"{unexpected}")
    recognizable = [surface for surface in surfaces if surface.recognizable]
    visible_bits = 0
    for surface in recognizable:
        visible_bits |= 1 << HUNT_CLASS_VISIBILITY_INDEX[surface.kind]
    known_bits = (1 << len(HUNT_TRAIN_ENTITY_CLASSES)) - 1
    return {
        "class_visibility_roster": list(HUNT_TRAIN_ENTITY_CLASSES),
        "class_visibility_roster_sha256": (
            HUNT_CLASS_VISIBILITY_ROSTER_CONTRACT["sha256"]),
        "class_visibility_known_bits": int(known_bits),
        "class_visibility_visible_bits": int(visible_bits),
        "class_visibility_geometry_valid": 1,
        "recognizable_surfaces": recognizable,
        "class_visible_instances": [
            surface.record(chosen=False) for surface in recognizable],
    }


def hunt_class_artifacts(
        packed_ids, *, goal_kind: str,
        committed_instance_id: str | None = None) -> dict:
    if goal_kind not in GOAL_ENTITY_KINDS:
        raise ValueError(f"unsupported Hunt goal class: {goal_kind!r}")
    surfaces = entity_surfaces(packed_ids)
    target_kinds = GOAL_ENTITY_KINDS[goal_kind]
    target = [surface for surface in surfaces if surface.kind in target_kinds]
    recognizable = [surface for surface in target if surface.recognizable]
    union = np.zeros(np.asarray(packed_ids).shape, dtype=np.uint8)
    for surface in recognizable:
        union |= surface.mask
    chosen_surface = next(
        (surface for surface in target
         if surface.instance_id == committed_instance_id), None)
    return {
        "goal_kind": goal_kind,
        "class_exist": int(bool(recognizable)),
        "class_recognizable": int(bool(recognizable)),
        "class_union_mask": union,
        "visible_instances": [
            surface.record(chosen=(
                surface.instance_id == committed_instance_id))
            for surface in recognizable
        ],
        "all_exact_surfaces": surfaces,
        "chosen_surface_visible": int(chosen_surface is not None),
        "chosen_recognizable": int(
            chosen_surface is not None and chosen_surface.recognizable),
        "committed_surface_instance": (
            None if chosen_surface is None or chosen_surface.recognizable
            else {
                **chosen_surface.record(chosen=False),
                "surface_for_committed_target": True,
            }),
    }


def hunt_visibility_phase_artifacts(
        packed_ids, *, goal_kind: str, attack: bool,
        target_surface_distance: float | None = None,
        interaction_reach: float = HUNT_ATTACK_REACH,
        crosshair_xy: tuple[int, int] = (320, 180)) -> dict:
    packed = _strict_packed_ids(packed_ids)
    if not isinstance(attack, bool):
        raise ValueError("attack must be boolean")
    if (isinstance(interaction_reach, bool)
            or not isinstance(interaction_reach, (int, float))
            or not math.isfinite(float(interaction_reach))
            or float(interaction_reach) <= 0.0):
        raise ValueError("interaction_reach must be a positive finite number")
    if (target_surface_distance is not None
            and (isinstance(target_surface_distance, bool)
                 or not isinstance(target_surface_distance, (int, float))
                 or not math.isfinite(float(target_surface_distance))
                 or float(target_surface_distance) < 0.0)):
        raise ValueError(
            "target_surface_distance must be a nonnegative finite number or None")
    if (not isinstance(crosshair_xy, tuple) or len(crosshair_xy) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in crosshair_xy)):
        raise ValueError("crosshair_xy must be an integer (x,y) tuple")
    x, y = crosshair_xy
    if not (0 <= x < packed.shape[1] and 0 <= y < packed.shape[0]):
        raise ValueError("crosshair_xy is outside the entity-ID image")
    artifacts = hunt_class_artifacts(packed, goal_kind=goal_kind)
    packed_hit = int(packed[y, x])
    semantic_hit = packed_hit >> 16
    entity_hit = packed_hit & 0xffff
    goal_codes = {
        ENTITY_ID_SEMANTIC_CODES[kind]
        for kind in GOAL_ENTITY_KINDS[goal_kind]
    }
    crosshair_on_goal_surface = bool(semantic_hit in goal_codes)
    attack_on_goal_surface = bool(attack and crosshair_on_goal_surface)
    effective_target_distance = (
        target_surface_distance if crosshair_on_goal_surface else None)
    within_reach = bool(
        effective_target_distance is not None
        and float(effective_target_distance) <= float(interaction_reach))
    interaction_on_goal = bool(attack_on_goal_surface and within_reach)
    phase = 2 if interaction_on_goal else 1 if artifacts["class_exist"] else 0
    return {
        **artifacts,
        "phase": int(phase),
        "phase_contract": HUNT_VISIBILITY_PHASE_CONTRACT,
        "crosshair_on_goal_surface": int(crosshair_on_goal_surface),
        "attack_on_goal_surface": int(attack_on_goal_surface),
        "interaction_on_goal_within_reach": int(interaction_on_goal),
        "target_surface_distance": (
            None if effective_target_distance is None
            else float(effective_target_distance)),
        "crosshair_world_surface_distance": (
            None if target_surface_distance is None
            else float(target_surface_distance)),
        "interaction_reach": float(interaction_reach),
        "target_within_interaction_reach": int(within_reach),
        "interacted_instance_id": (
            f"entity:{entity_hit}" if interaction_on_goal else None),
        "crosshair_xy": [int(x), int(y)],
    }
