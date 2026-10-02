#!/usr/bin/env python
"""Camera-off world staging: site search, block and entity placement, kit reset, verified teleports and the frame-guarded POV recorder."""
import os, sys, json, math, hashlib, itertools, copy, time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import numpy as np
import cv2
try:
    import attacca.evaluation.geometry as G
except ModuleNotFoundError:
    from attacca.evaluation import geometry as G

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, str(REPO) + "/src")
from attacca.worlds import ore_classes as OC
from attacca.worlds import episode_schema as EPISODE_SCHEMA
from attacca.worlds import structural_census as SC

CELLS = ["mine"]
MINE_REACH = 4.35
MINE_CHAIN_RADIUS = 18.0
MINE_CHAIN_MAX_DY = 6.0
MINE_PRIMARY_MIN_HORIZONTAL = 6.0
MINE_PRIMARY_MAX_DISCOVERY_STEP = 24
MINE_PRIMARY_MAX_ACCESS_PATH_STEPS = 40
MINE_PRIMARY_MIN_START_BRANCHES = 2
MINE_SURVEY_ROUTE_STEPS = 40
MINE_SURVEY_CHECKPOINT_STRIDE = 4
MINE_COUNTERFACTUAL_LANE_MIN_GAP = 1.1
MINE_SITE_SCAN_SPAN = 18
V12_INDEPENDENT_MULTICLASS6_CONTRACT = (
    "independent_v12_exposed_stone_multiclass6/v1")
V12_INDEPENDENT_MULTICLASS6_PROFILE = "v12_independent_multiclass6"
V12_INDEPENDENT_TARGET_MAX_SPAN_BLOCKS = MINE_CHAIN_RADIUS

_MINE_NEIGHBORS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                   (0, 0, 1), (0, 0, -1))
_MINE_HORIZONTAL = ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))

ORE_SIDE_PLACEMENT_GEOMETRY = "exposed_stone_full_cube/v1"
ORE_TOP_PLACEMENT_GEOMETRY = "embedded_rock_top_face/v1"


def ore_replacement_placement_geometry(geometry):
    geom9 = dict(geometry or {})
    if not bool(geom9.get("ok")):
        return None
    if int(geom9.get("side_exposed", 0)) >= 1:
        return ORE_SIDE_PLACEMENT_GEOMETRY
    if int(geom9.get("exposed", 0)) >= 1 and bool(geom9.get("support")):
        return ORE_TOP_PLACEMENT_GEOMETRY
    return None

ENTITY_QUERY_POSITION = "position"
ENTITY_QUERY_ABSENT = "absent"
ENTITY_QUERY_UNKNOWN = "unknown"


def _natural_registry_kind(raw):
    value9 = str(raw).strip().split("[", 1)[0]
    if not value9:
        raise ValueError("voxel registry kind must be non-empty")
    return value9 if ":" in value9 else f"minecraft:{value9}"


def _format_reject_census(rejects):
    return json.dumps(
        {str(key9): int(value9) for key9, value9 in sorted(rejects.items())},
        sort_keys=True)


def terrain_only_survey_route(center, reachable, max_steps=MINE_SURVEY_ROUTE_STEPS,
                              checkpoint_stride=MINE_SURVEY_CHECKPOINT_STRIDE):
    states9 = frozenset(tuple(int(v9) for v9 in q9) for q9 in reachable)
    if not states9:
        return {"route": [], "checkpoints": [], "checkpoint_steps": []}
    cx9, cy9, cz9 = (float(v9) for v9 in center)
    sx9, sz9 = int(math.floor(cx9)), int(math.floor(cz9))
    starts9 = [
        (abs(q9[1] - cy9) + math.hypot(q9[0] - sx9, q9[2] - sz9), q9)
        for q9 in states9
        if abs(q9[0] - sx9) <= 1 and abs(q9[2] - sz9) <= 1
    ]
    if not starts9:
        return {"route": [], "checkpoints": [], "checkpoint_steps": []}
    start9 = min(starts9, key=lambda q9: (q9[0], q9[1]))[1]
    columns9 = defaultdict(list)
    for q9 in states9:
        columns9[(q9[0], q9[2])].append(q9)
    for values9 in columns9.values():
        values9.sort()

    directions9 = list(_MINE_HORIZONTAL)
    rotate9 = abs(start9[0] * 73 + start9[2] * 151) % len(directions9)
    directions9 = directions9[rotate9:] + directions9[:rotate9]

    def neighbours9(q9):
        x9, fy9, z9 = q9
        out9 = []
        for order9, (dx9, _dy9, dz9) in enumerate(directions9):
            for next9 in columns9.get((x9 + dx9, z9 + dz9), ()):
                if abs(next9[1] - fy9) <= 1:
                    radial9 = math.hypot(next9[0] - start9[0], next9[2] - start9[2])
                    out9.append((-radial9, order9, abs(next9[1] - fy9), next9))
        return [item9[-1] for item9 in sorted(out9)]

    limit9 = max(0, int(max_steps))
    route9 = [start9]
    visited9 = {start9}
    stack9 = [[start9, neighbours9(start9), 0]]
    while stack9 and len(route9) - 1 < limit9:
        current9, options9, option_i9 = stack9[-1]
        while option_i9 < len(options9) and options9[option_i9] in visited9:
            option_i9 += 1
        stack9[-1][2] = option_i9
        if option_i9 < len(options9):
            next9 = options9[option_i9]
            stack9[-1][2] += 1
            visited9.add(next9)
            route9.append(next9)
            stack9.append([next9, neighbours9(next9), 0])
            continue
        stack9.pop()
        if stack9 and len(route9) - 1 < limit9:
            route9.append(stack9[-1][0])

    stride9 = max(1, int(checkpoint_stride))
    checkpoint_steps9 = list(range(stride9, len(route9), stride9))
    if len(route9) > 1 and (not checkpoint_steps9 or checkpoint_steps9[-1] != len(route9) - 1):
        checkpoint_steps9.append(len(route9) - 1)
    checkpoints9 = [route9[i9] for i9 in checkpoint_steps9]
    return {
        "route": [[int(v9) for v9 in q9] for q9 in route9],
        "checkpoints": [[int(v9) for v9 in q9] for q9 in checkpoints9],
        "checkpoint_steps": [int(i9) for i9 in checkpoint_steps9],
    }


BIOME_TERRAIN_TOPOLOGY_CONTRACT = "v13_biome_terrain_topology/v1"
BIOME_TERRAIN_ROUTE_ARCHETYPES = (
    "level_bank", "winding_level", "ascent", "descent", "rolling")


def mine_candidate_topology_witness(grid, survey, cell, geom):
    survey9, geom9 = dict(survey or {}), dict(geom or {})
    route9 = [tuple(int(v9) for v9 in q9)
              for q9 in survey9.get("route") or ()]
    step9 = int(geom9.get("survey_checkpoint_step", 0))
    if not route9 or step9 < 0 or step9 >= len(route9):
        raise ValueError("terrain topology needs a valid discovery route prefix")
    prefix9 = route9[:step9 + 1]
    deltas9 = [
        (right9[0] - left9[0], right9[1] - left9[1],
         right9[2] - left9[2])
        for left9, right9 in zip(prefix9[:-1], prefix9[1:])]
    ascent9 = sum(delta9[1] > 0 for delta9 in deltas9)
    descent9 = sum(delta9[1] < 0 for delta9 in deltas9)
    ys9 = [q9[1] for q9 in prefix9]
    relief9 = max(ys9) - min(ys9)
    net9 = prefix9[-1][1] - prefix9[0][1]
    horizontal9 = [(q9[0], q9[2]) for q9 in deltas9]
    turn_90_9 = 0
    backtrack9 = 0
    for left9, right9 in zip(horizontal9[:-1], horizontal9[1:]):
        dot9 = left9[0] * right9[0] + left9[1] * right9[1]
        if dot9 == -1:
            backtrack9 += 1
        elif dot9 == 0 and left9 != (0, 0) and right9 != (0, 0):
            turn_90_9 += 1
    displacement9 = math.hypot(
        prefix9[-1][0] - prefix9[0][0],
        prefix9[-1][2] - prefix9[0][2])
    route_steps9 = max(0, len(prefix9) - 1)
    if ascent9 and descent9 and relief9 >= 2:
        route_archetype9 = "rolling"
    elif net9 > 0:
        route_archetype9 = "ascent"
    elif net9 < 0:
        route_archetype9 = "descent"
    elif turn_90_9 >= 2:
        route_archetype9 = "winding_level"
    else:
        route_archetype9 = "level_bank"
    cell9 = tuple(int(v9) for v9 in cell)
    open_normals9 = [
        [int(dx9), int(dy9), int(dz9)]
        for dx9, dy9, dz9 in _MINE_HORIZONTAL
        if grid.get((cell9[0] + dx9, cell9[1] + dy9,
                     cell9[2] + dz9)) is None]
    side_exposed9 = int(geom9.get("side_exposed", len(open_normals9)))
    return {
        "contract": BIOME_TERRAIN_TOPOLOGY_CONTRACT,
        "route_archetype": route_archetype9,
        "host_archetype": (
            "multi_face_outcrop" if side_exposed9 >= 2 else
            "single_face_bank"),
        "discovery_step": step9,
        "route_steps": route_steps9,
        "route_relief": int(relief9),
        "net_elevation": int(net9),
        "ascent_steps": int(ascent9),
        "descent_steps": int(descent9),
        "turn_90_count": int(turn_90_9),
        "backtrack_count": int(backtrack9),
        "straight_displacement": round(float(displacement9), 3),
        "tortuosity": round(
            float(route_steps9) / max(1.0, float(displacement9)), 3),
        "unique_route_columns": len({(q9[0], q9[2]) for q9 in prefix9}),
        "side_exposed": side_exposed9,
        "open_side_normals": open_normals9,
        "start_access_path_steps": geom9.get("start_access_path_steps"),
        "start_surface_branches": geom9.get("start_surface_branches"),
        "access_dist": geom9.get("access_dist"),
        "access_pitch": geom9.get("access_pitch"),
    }


def selected_biome_terrain_route_archetype(center):
    x9, _y9, z9 = (float(v9) for v9 in center)
    payload9 = (f"{BIOME_TERRAIN_TOPOLOGY_CONTRACT}:"
                f"{math.floor(x9)}:{math.floor(z9)}")
    digest9 = hashlib.sha256(payload9.encode("ascii")).digest()
    return BIOME_TERRAIN_ROUTE_ARCHETYPES[
        int.from_bytes(digest9[:8], "big")
        % len(BIOME_TERRAIN_ROUTE_ARCHETYPES)]


def survey_checkpoint_stance(survey, step, *, max_step=MINE_SURVEY_ROUTE_STEPS):
    route9 = [tuple(int(v9) for v9 in q9) for q9 in (survey or {}).get("route", ())]
    step9 = int(step)
    if not (1 <= step9 < len(route9)) or step9 > int(max_step):
        return None
    stance9 = route9[step9]
    previous9 = route9[step9 - 1]
    delta_xz9 = (stance9[0] - previous9[0], stance9[2] - previous9[2])
    if delta_xz9 == (0, 0):
        return None
    return stance9, math.degrees(math.atan2(-delta_xz9[0], delta_xz9[1]))


def stance_perceptibility(stance, yaw, target, occ, *,
                          max_dist=MINE_CHAIN_RADIUS, pitch=8.0):
    stance9 = tuple(int(v9) for v9 in stance)
    target9 = tuple(int(v9) for v9 in target)
    yaw9, pitch9, max_dist9 = float(yaw), float(pitch), float(max_dist)
    dx9 = target9[0] + 0.5 - (stance9[0] + 0.5)
    dz9 = target9[2] + 0.5 - (stance9[2] + 0.5)
    horizontal9 = math.hypot(dx9, dz9)
    if horizontal9 < 1.2 or horizontal9 > max_dist9:
        return None
    eye9 = (stance9[0] + 0.5, stance9[1] + G.EYE, stance9[2] + 0.5)
    mask9, point9, bbox9, meta9 = G.visible_surface_mask_blocks(
        eye9, yaw9, pitch9, (target9,), occ,
        extra_visual_solid=G.visual_occluders(occ, eye9),
        alpha_cutout_visual=G.visual_alpha_cutout_cells(occ, eye9),
        unknown_is_solid=True)
    if point9 is None or bbox9 is None or not np.asarray(mask9).any():
        return None
    area9 = int(np.asarray(mask9, dtype=np.uint8).sum())
    short9 = int(min(
        int(bbox9[2]) - int(bbox9[0]) + 1,
        int(bbox9[3]) - int(bbox9[1]) + 1))
    recognition9 = EPISODE_SCHEMA.structural_surface_recognition_fields(
        area9, short9, visible_bbox_px=list(bbox9),
        require_full_cube_fill=True)
    if not recognition9["oracle_recognizable"]:
        return None
    normal9 = tuple(int(v9) for v9 in meta9["representative_surface_normal"])
    proof9 = {
        **recognition9,
        "visible_area_px": area9,
        "visible_short_side_px": short9,
        "visible_bbox_px": [int(v9) for v9 in bbox9],
        "visibility_measure": "dense_native_visible_surface_first_hit",
        "mask_semantic": str(meta9["mask_semantic"]),
        "tested_ray_count": int(meta9["tested_ray_count"]),
        "candidate_pixel_count": int(meta9["candidate_pixel_count"]),
        "visible_point": [float(v9) for v9 in point9],
        "face_normal": [int(v9) for v9 in normal9],
        "aim_pixel": [int(v9) for v9 in meta9["representative_pixel_xy"]],
    }
    point_dx9 = point9[0] - (stance9[0] + 0.5)
    point_dz9 = point9[2] - (stance9[2] + 0.5)
    face_yaw9 = math.degrees(math.atan2(-point_dx9, point_dz9))
    heading_error9 = abs((face_yaw9 - yaw9 + 180.0) % 360.0 - 180.0)
    face_pitch9 = math.degrees(math.atan2(
        eye9[1] - point9[1], max(0.5, horizontal9)))
    pitch_error9 = abs(face_pitch9 - pitch9)
    return {
        "survey_heading_error": round(float(heading_error9), 3),
        "survey_incoming_yaw": round(float(yaw9), 3),
        "survey_view_pos": [int(v9) for v9 in stance9],
        "survey_pitch": round(float(pitch9), 3),
        "survey_pitch_error": round(float(pitch_error9), 3),
        "survey_target_horizontal": round(float(horizontal9), 3),
        "survey_visible_point": [float(v9) for v9 in point9],
        "survey_face_normal": [int(v9) for v9 in normal9],
        "survey_perceptibility": dict(proof9),
    }


def survey_checkpoint_perceptibility(survey, target, occ, step, *,
                                     max_step=MINE_SURVEY_ROUTE_STEPS,
                                     max_dist=MINE_CHAIN_RADIUS,
                                     pitch=8.0):
    pose9 = survey_checkpoint_stance(survey, step, max_step=max_step)
    if pose9 is None:
        return None
    stance9, yaw9 = pose9
    proof9 = stance_perceptibility(stance9, yaw9, target, occ,
                                   max_dist=max_dist, pitch=pitch)
    if proof9 is None:
        return None
    return {"survey_checkpoint_step": int(step), **proof9}


def first_survey_checkpoint_perceptibility(survey, target, occ, *,
                                            max_step=MINE_SURVEY_ROUTE_STEPS,
                                            max_dist=MINE_CHAIN_RADIUS,
                                            pitch=8.0):
    steps9 = sorted({int(v9) for v9 in (survey or {}).get("checkpoint_steps", ())})
    for step9 in steps9:
        proof9 = survey_checkpoint_perceptibility(
            survey, target, occ, step9,
            max_step=max_step, max_dist=max_dist, pitch=pitch)
        if proof9 is not None:
            return proof9
    return None


def _select_ordered_mine_chain(specs, count=3, *, min_gap=0.0):
    count9 = max(1, int(count))
    ordered9 = sorted(specs, key=lambda q9: (
        int(q9["survey_step"]),
        0 if q9["source"] in {
            "natural", "natural_exact_class",
            getattr(OC, "NATURAL_EXACT_EXPOSED_BY_STONE_CARVE_SOURCE",
                    "natural_exact_exposed_by_stone_carve")} else 1,
        tuple(q9["slot"])))
    for first9 in ordered9:
        if (not bool(first9.get("primary_eligible"))
                or int(first9["survey_step"]) > MINE_PRIMARY_MAX_DISCOVERY_STEP):
            continue
        chain9 = [first9]
        used9 = {tuple(first9["slot"])}
        for next9 in ordered9:
            slot9 = tuple(next9["slot"])
            if (slot9 in used9
                    or int(next9["survey_step"]) < int(first9["survey_step"])):
                continue
            if float(min_gap) > 0.0 and any(
                    math.dist(slot9, tuple(chosen9["slot"]))
                    < float(min_gap) for chosen9 in chain9):
                continue
            chain9.append(next9)
            used9.add(slot9)
            if len(chain9) == count9:
                return chain9
        if len(chain9) == count9:
            return chain9
    return None


def select_diverse_ore_lane_candidates(
        specs, *, max_candidates=72,
        min_gap=MINE_COUNTERFACTUAL_LANE_MIN_GAP,
        max_pair_probes=4_500_000):
    limit9 = max(2, int(max_candidates))
    gap9 = float(min_gap)
    ordered9 = sorted(
        list(specs),
        key=lambda spec9: (int(spec9.get("survey_step", 0)), tuple(spec9["slot"])))
    n9 = len(ordered9)
    if n9 < 3:
        return []

    conflict9 = []
    for spec9 in ordered9:
        a9 = tuple(spec9["slot"])
        mask9 = 0
        for j9, other9 in enumerate(ordered9):
            if math.dist(a9, tuple(other9["slot"])) < gap9:
                mask9 |= 1 << j9
        conflict9.append(mask9)

    records9 = []
    for indices9 in itertools.combinations(range(n9), 3):
        if not any(bool(ordered9[i9].get("primary_eligible")) for i9 in indices9):
            continue
        if any(conflict9[i9] & (1 << j9)
               for i9, j9 in itertools.combinations(indices9, 2)):
            continue
        lane_mask9 = sum(1 << i9 for i9 in indices9)
        forbidden9 = 0
        for i9 in indices9:
            forbidden9 |= conflict9[i9]
        records9.append((lane_mask9, forbidden9, indices9))
    if len(records9) <= limit9:
        return [tuple(ordered9[i9] for i9 in record9[2]) for record9 in records9]

    spread9 = [i9 * (len(records9) - 1) // (limit9 - 1)
               for i9 in range(limit9)]
    pair9 = None
    probes9 = 0
    probe_cap9 = max(1, int(max_pair_probes))
    for left_i9 in spread9:
        _left_mask9, left_forbidden9, _left_indices9 = records9[left_i9]
        for right_i9, (right_mask9, _right_forbidden9, _right_indices9) \
                in enumerate(records9):
            probes9 += 1
            if not (right_mask9 & left_forbidden9):
                pair9 = (left_i9, right_i9)
                break
            if probes9 >= probe_cap9:
                break
        if pair9 is not None or probes9 >= probe_cap9:
            break

    chosen9 = []
    if pair9 is not None:
        chosen9.extend(pair9)
    for index9 in spread9:
        if index9 not in chosen9:
            chosen9.append(index9)
        if len(chosen9) >= limit9:
            break
    if len(chosen9) < limit9:
        for index9 in range(len(records9)):
            if index9 not in chosen9:
                chosen9.append(index9)
            if len(chosen9) >= limit9:
                break
    return [tuple(ordered9[i9] for i9 in records9[index9][2])
            for index9 in chosen9]


def mine_slot_reachable_stance(occ, spec, *, reach=MINE_REACH):
    return G.reachable_mine_stance(
        occ, tuple(int(v9) for v9 in spec["survey_view_pos"]),
        tuple(int(v9) for v9 in spec["slot"]), reach=float(reach))


def select_stanceable_mine_chain(
        specs, occ, *, count=3, reach=MINE_REACH, min_gap=0.0):
    specs9 = list(specs)
    proofs9 = {}
    unstanceable9 = []
    while True:
        chain9 = _select_ordered_mine_chain(
            specs9, count=count, min_gap=min_gap)
        if chain9 is None:
            return None, proofs9, unstanceable9
        failed9 = []
        for spec9 in chain9:
            slot9 = tuple(spec9["slot"])
            if slot9 not in proofs9:
                proofs9[slot9] = mine_slot_reachable_stance(
                    occ, spec9, reach=reach)
            if proofs9[slot9] is None:
                failed9.append(slot9)
        if not failed9:
            return chain9, proofs9, unstanceable9
        unstanceable9 += failed9
        dropped9 = set(unstanceable9)
        specs9 = [q9 for q9 in specs9 if tuple(q9["slot"]) not in dropped9]


@dataclass(frozen=True)
class EntityQueryResult:

    selector: str
    status: str
    position: object = None
    fresh: bool = False
    pending: bool = False
    reason: str = "unknown"
    age_calls: object = None

    @property
    def exists(self):
        if self.status == ENTITY_QUERY_POSITION:
            return True
        if self.status == ENTITY_QUERY_ABSENT:
            return False
        return None


class Rec:

    def __init__(self, w, out_dir, tag, thresh=25.0):
        self.w, self.out_dir = w, out_dir
        self.tag, self.frames, self.last_y = tag, [], None
        self.thresh = thresh

    def save(self):
        if not self.frames:
            return
        h, wd = self.frames[0].shape[:2]
        vw = cv2.VideoWriter(os.path.join(self.out_dir, f"{self.tag}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 20, (wd, h))
        for f in self.frames:
            vw.write(f)
        vw.release()


class CleanPOVRec(Rec):

    def __init__(self, w, out_dir, tag, thresh=25.0, *, allow_pose_jumps=False):
        super().__init__(w, out_dir, tag, thresh)
        self.allow_pose_jumps = bool(allow_pose_jumps)

    def grab(self, note=""):
        raw9 = np.asarray(self.w.info.get("pov"))
        if raw9.shape != (360, 640, 3):
            raise RuntimeError(
                f"policy-source POV shape changed: {raw9.shape}")
        if (not np.issubdtype(raw9.dtype, np.number)
                or not bool(np.isfinite(raw9).all())):
            raise RuntimeError("policy-source POV is non-finite or non-numeric")
        if raw9.dtype != np.uint8:
            raise RuntimeError(
                f"policy-source POV dtype changed: {raw9.dtype}")
        if not bool(np.any(raw9)):
            raise RuntimeError("policy-source POV is exact all-black")

        try:
            y9 = float(self.w.get_pos()[1])
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "policy-source pose is malformed") from exc
        if not math.isfinite(y9):
            raise RuntimeError("policy-source pose is non-finite")
        dy9 = 0.0 if self.last_y is None else y9 - float(self.last_y)
        if abs(dy9) > 1.5 and not self.allow_pose_jumps:
            raise RuntimeError(
                f"policy-source pose jump exceeds 1.5 blocks: dy={dy9:.6f}")
        self.last_y = y9
        self.frames.append(cv2.cvtColor(
            raw9, cv2.COLOR_RGB2BGR).copy())


class Stager:

    def __init__(self, w, repo=REPO):
        self.w, self.repo = w, repo
        self.PLACED_UNDO, self.SITES = [], set()
        self.LOGSTATE = {}
        self.ENTITY_QUERY_STATE = {
            "call": 0,
            "cache": {},
            "last_issued": {},
            "pending": None,
        }
        self.last_pose_spec = None
        self._last_mine_survey_contract = None
        self._last_mine_slot_diagnostics = {}
        self._last_mine_occupancy = None
        self.mine_class_pool = tuple(OC.validate_mine_pool())
        self.mine_scrub_classes = tuple(SC.CENSUS_SCRUB_CLASSES)
        self.plan_mode = "pair"
        self.mine_chain_count = 3
        self.world_origin = tuple(float(v) for v in w.get_pos())

    def prep_world(self):
        w = self.w
        self.bind_entity_log_camera_off()
        for g in ("showDeathMessages", "fallDamage", "doDaylightCycle", "doMobSpawning",
                  "doInsomnia", "doFireTick", "doWeatherCycle", "announceAdvancements", "mobGriefing"):
            w.cmd(f"/gamerule {g} false")
        w.cmd("/difficulty normal")
        w.cmd("/weather clear")
        w.cmd("/gamerule randomTickSpeed 0")
        w.cmd("/gamerule doFireTick false")
        w.cmd("/time set 6000")
        for adv in ("adventure/kill_a_mob", "adventure/shoot_arrow", "adventure/root",
                    "story/root", "story/mine_stone", "husbandry/root"):
            w.cmd(f"/advancement grant @a only minecraft:{adv}")
        w.cmd("/recipe give @a *")
        for _ in range(60):
            w.step_noop()
        for _ in range(4):
            w.step_noop()
        self.bind_entity_log_camera_off()

    def settle_feet(self, x, z):
        cx = math.floor(float(x)) + 0.5
        cz = math.floor(float(z)) + 0.5
        current_y = float(self.w.get_pos()[1])
        ref_y = min(202.0, max(70.0, current_y))
        feet_y = self.w._ground_y(cx, cz, ref_y, lift=48, max_steps=100)
        if feet_y >= ref_y + 47.0 and ref_y < 202.0:
            feet_y = self.w._ground_y(cx, cz, 202.0, lift=48, max_steps=100)
        return feet_y

    def place_cell_y(self, x, z):
        return int(round(self.settle_feet(x, z)))

    def go(self, x, y, z, yaw, pitch=8, n=6):
        return self._teleport_stable(x, y, z, yaw, pitch, max_ticks=max(16, n + 8))

    def kit(self, item, extra=None, fire_res=False):
        w = self.w
        w.cmd("/effect clear @a")
        w.cmd("/effect give @a minecraft:resistance 999999 4 true")
        if fire_res:
            w.cmd("/effect give @a minecraft:fire_resistance 999999 1 true")
        w.cmd("/clear @a")
        w.step_noop()
        w.cmd(f"/replaceitem entity @a hotbar.0 {item}")
        if extra:
            w.cmd(extra)
        for _ in range(3):
            w.step_noop()

    @staticmethod
    def _mine_solid(block):
        if block is None:
            return False
        bare = str(block).split(":")[-1]
        if bare in ("air", "cave_air", "void_air", "water", "lava", "bubble_column"):
            return False
        if bare.endswith(("_leaves", "_sapling", "_flower", "_tulip", "_mushroom")):
            return False
        return bare not in {
            "grass", "tall_grass", "fern", "large_fern", "dead_bush", "sweet_berry_bush",
            "seagrass", "tall_seagrass", "kelp", "kelp_plant", "sugar_cane", "vine",
            "poppy", "dandelion", "oxeye_daisy", "azure_bluet", "cornflower", "allium",
            "blue_orchid", "lily_of_the_valley", "sunflower", "lilac", "rose_bush", "peony",
            "snow", "torch", "wall_torch",
        }

    @classmethod
    def _mine_walk_support(cls, block):
        if not cls._mine_solid(block):
            return False
        bare9 = str(block).split(":")[-1]
        return not bare9.endswith(("_leaves", "_log"))

    @classmethod
    def _mine_walk_blocked(cls, block):
        if block is None:
            return False
        bare9 = str(block).split(":")[-1]
        return (cls._mine_solid(block)
                or bare9 in ("water", "lava", "bubble_column")
                or bare9.endswith(("_leaves", "_log")))

    @classmethod
    def nearby_safe_mine_start_offsets(cls, rel_voxels, *, radius=4, max_dy=2):
        rel9 = {tuple(int(v9) for v9 in key9): value9
                for key9, value9 in dict(rel_voxels or {}).items()}
        rows9 = []
        radius9, max_dy9 = max(0, int(radius)), max(0, int(max_dy))
        feet_offsets9 = sorted(range(-max_dy9, max_dy9 + 1),
                               key=lambda value9: (abs(value9), value9))
        for dx9 in range(-radius9, radius9 + 1):
            for dz9 in range(-radius9, radius9 + 1):
                if dx9 == 0 and dz9 == 0:
                    continue
                radial9 = math.hypot(dx9, dz9)
                if radial9 > float(radius9):
                    continue
                for feet_dy9 in feet_offsets9:
                    support9 = rel9.get((dx9, feet_dy9 - 1, dz9))
                    foot9 = rel9.get((dx9, feet_dy9, dz9))
                    head9 = rel9.get((dx9, feet_dy9 + 1, dz9))
                    if (cls._mine_walk_support(support9)
                            and not cls._mine_walk_blocked(foot9)
                            and not cls._mine_walk_blocked(head9)):
                        rows9.append((radial9, abs(feet_dy9), feet_dy9,
                                      dx9, dz9))
                        break
        rows9.sort()
        return tuple((int(dx9), int(feet_dy9), int(dz9))
                     for _radial9, _abs_dy9, feet_dy9, dx9, dz9 in rows9)

    def converged_relative_voxels(self, box, *, max_passes=6):
        box9 = np.asarray(tuple(int(v9) for v9 in box), dtype=np.int32)
        previous9 = None
        for _pass9 in range(max(2, int(max_passes))):
            action9 = self.w.sim.noop_action()
            action9["voxels"] = box9.copy()
            self.w.obs, _, _, _, self.w.info = self.w.sim.step(action9)
            current9 = {
                (int(row9["x"]), int(row9["y"]), int(row9["z"])):
                str(row9.get("type", ""))
                for row9 in (self.w.info.get("voxels") or [])}
            if previous9 is not None and current9 == previous9:
                return current9
            previous9 = current9
        raise RuntimeError("relative voxel reply did not converge")

    @staticmethod
    def _scan_contains(bounds, cell, margin=0):
        if bounds is None:
            return False
        x, y, z = (int(v) for v in cell)
        margin = int(margin)
        return (bounds[0] + margin <= x < bounds[1] - margin
                and bounds[2] + margin <= y < bounds[3] - margin
                and bounds[4] + margin <= z < bounds[5] - margin)

    def _mine_scan_contains(self, cell, margin=0):
        return self._scan_contains(
            getattr(self, "_last_mine_scan_bounds", None), cell, margin)

    @staticmethod
    def mine_occupancy(grid, bounds):
        occ9 = G.OccupancyMap(leaves_occlude=True)
        occ9.grid = dict(grid)
        occ9.boxes = [tuple(int(v9) for v9 in bounds)]
        return occ9

    @classmethod
    def _ore_geometry(cls, grid, cell):
        x, y, z = cell
        contacts = sum(cls._mine_solid(grid.get((x + dx, y + dy, z + dz)))
                       for dx, dy, dz in _MINE_NEIGHBORS)
        exposed = sum(grid.get((x + dx, y + dy, z + dz)) is None
                      for dx, dy, dz in _MINE_NEIGHBORS)
        side_exposed = sum(grid.get((x + dx, y + dy, z + dz)) is None
                           for dx, dy, dz in _MINE_HORIZONTAL)
        support = cls._mine_solid(grid.get((x, y - 1, z)))
        return dict(contacts=int(contacts), exposed=int(exposed),
                    side_exposed=int(side_exposed), support=bool(support),
                    ok=bool(contacts >= 3 and exposed >= 1 and support))

    def scan_mine_terrain(self, span=24, y_down=12, y_up=12):
        w = self.w
        span, y_down, y_up = int(span), int(y_down), int(y_up)
        ax, ay, az = w.get_pos()
        bx, by, bz = int(math.floor(ax)), int(math.floor(ay)), int(math.floor(az))
        self._last_mine_scan_bounds = (
            bx - span, bx + span + 1, by - y_down, by + y_up + 1,
            bz - span, bz + span + 1)
        x_ranges = [(lo, min(span + 1, lo + 41))
                    for lo in range(-span, span + 1, 41)]
        z_ranges = [(lo, min(span + 1, lo + 41))
                    for lo in range(-span, span + 1, 41)]
        y_ranges = [(lo, min(y_up + 1, lo + 13))
                    for lo in range(-y_down, y_up + 1, 13)]
        previous = None
        for _pass9 in range(10):
            grid = {}
            for xlo9, xhi9 in x_ranges:
                for zlo9, zhi9 in z_ranges:
                    for ylo9, yhi9 in y_ranges:
                        qx9, qy9, qz9 = w.get_pos()
                        qbx9, qby9, qbz9 = (int(math.floor(qx9)),
                                             int(math.floor(qy9)),
                                             int(math.floor(qz9)))
                        box9 = [bx + xlo9 - qbx9, bx + xhi9 - qbx9,
                                by + ylo9 - qby9, by + yhi9 - qby9,
                                bz + zlo9 - qbz9, bz + zhi9 - qbz9]
                        a9 = w.sim.noop_action()
                        a9["voxels"] = np.array(box9, np.int32)
                        w.obs, _, _, _, w.info = w.sim.step(a9)
                        rx9, ry9, rz9 = w.get_pos()
                        rbx9, rby9, rbz9 = (int(math.floor(rx9)),
                                             int(math.floor(ry9)),
                                             int(math.floor(rz9)))
                        for c9 in (w.info.get("voxels") or []):
                            grid[(rbx9 + int(c9["x"]), rby9 + int(c9["y"]),
                                  rbz9 + int(c9["z"]))] = str(c9.get("type", ""))
            if previous is not None and grid == previous:
                return grid
            previous = grid
            for _ in range(12):
                w.step_noop()
        raise RuntimeError("mine terrain voxel scan did not converge")


    def _census_scrub_voxel_sample(self, half_xz=8, y_down=6, y_up=6):
        w = self.w
        previous9 = None
        for _pass9 in range(5):
            qx9, qy9, qz9 = w.get_pos()
            box9 = [-int(half_xz), int(half_xz) + 1, -int(y_down),
                    int(y_up) + 1, -int(half_xz), int(half_xz) + 1]
            a9 = w.sim.noop_action()
            a9["voxels"] = np.array(box9, np.int32)
            w.obs, _, _, _, w.info = w.sim.step(a9)
            rx9, ry9, rz9 = w.get_pos()
            rbx9, rby9, rbz9 = (int(math.floor(rx9)), int(math.floor(ry9)),
                                int(math.floor(rz9)))
            grid9 = {}
            for c9 in (w.info.get("voxels") or []):
                grid9[(rbx9 + int(c9["x"]), rby9 + int(c9["y"]),
                       rbz9 + int(c9["z"]))] = str(c9.get("type", ""))
            if previous9 is not None and grid9 == previous9:
                return grid9
            previous9 = grid9
            w.step_noop()
        return None

    @staticmethod
    def _census_scrub_roster_cells(grid, classes):
        counts9 = {kind9: 0 for kind9 in classes}
        cells9 = []
        for cell9, raw9 in grid.items():
            kind9 = OC.bare_block(raw9)
            if kind9 in counts9:
                counts9[kind9] += 1
                cells9.append((tuple(int(v9) for v9 in cell9), kind9))
        return counts9, cells9

    def census_scrub_natural_ores(self, free, *, staged_cells):
        w = self.w
        started9 = time.monotonic()
        bounds9 = [int(v9) for v9 in (free.get("mine_scan_bounds") or ())]
        if len(bounds9) != 6:
            raise RuntimeError(
                "census scrub requires the screened mine_scan_bounds envelope")
        region_record9 = SC.census_scrub_region(
            bounds9, extra_cells=staged_cells)
        region9 = tuple(int(v9) for v9 in region_record9["region_box"])
        classes9 = tuple(getattr(
            self, "mine_scrub_classes", SC.CENSUS_SCRUB_CLASSES))
        scan_started9 = time.monotonic()
        pre_grid9 = self.scan_mine_terrain(span=MINE_SITE_SCAN_SPAN)
        pre_counts9, _ = self._census_scrub_roster_cells(pre_grid9, classes9)
        central_box9 = [int(v9) for v9 in self._last_mine_scan_bounds]
        pre_scan_wall9 = time.monotonic() - scan_started9
        fill_started9 = time.monotonic()
        w.cmd(f"/forceload add {region9[0]} {region9[4]} "
              f"{region9[1] - 1} {region9[5] - 1}")
        for _ in range(10):
            w.step_noop()
        slabs9 = SC.fill_slab_tiling(region9)
        fill_count9 = 0
        for kind9 in classes9:
            for x09, x19, y09, y19, z09, z19 in slabs9:
                w.cmd(f"/fill {x09} {y09} {z09} {x19 - 1} {y19 - 1} {z19 - 1} "
                      f"minecraft:stone replace minecraft:{kind9}")
                fill_count9 += 1
                if fill_count9 % 64 == 0:
                    w.step_noop()
        w.cmd(f"/forceload remove {region9[0]} {region9[4]} "
              f"{region9[1] - 1} {region9[5] - 1}")
        for _ in range(3):
            w.step_noop()
        fill_wall9 = time.monotonic() - fill_started9
        audit_started9 = time.monotonic()
        cx9 = (region9[0] + region9[1]) / 2.0
        cz9 = (region9[4] + region9[5]) / 2.0
        reach_x9 = (region9[1] - region9[0]) / 2.0 - 12.0
        reach_z9 = (region9[5] - region9[4]) / 2.0 - 12.0
        samples9 = []
        for sx9, sz9 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            px9 = math.floor(cx9 + sx9 * reach_x9) + 0.5
            pz9 = math.floor(cz9 + sz9 * reach_z9) + 0.5
            sample9 = {"probe_xz": [px9, pz9], "converged": False,
                       "roster_ore_cells": None, "known_cells": 0}
            w.cmd(f"/tp @a {px9:.1f} 150 {pz9:.1f} 0 30")
            for _ in range(45):
                w.step_noop()
            try:
                feet9 = self.settle_feet(px9, pz9)
            except RuntimeError as exc9:
                sample9["skip_reason"] = f"unstable_ground:{exc9}"
                samples9.append(sample9)
                continue
            grid9 = self._census_scrub_voxel_sample()
            if grid9 is None:
                sample9["skip_reason"] = "voxel_sample_not_converged"
                samples9.append(sample9)
                continue
            counts9, found9 = self._census_scrub_roster_cells(grid9, classes9)
            staged_set9 = {tuple(int(v9) for v9 in c9) for c9 in staged_cells}
            natural9 = [(cell9, kind9) for cell9, kind9 in found9
                        if cell9 not in staged_set9]
            sample9.update(
                converged=True, feet_y=float(feet9),
                known_cells=int(len(grid9)),
                roster_ore_cells=int(len(natural9)),
                per_class_counts={k9: int(v9) for k9, v9 in counts9.items()
                                  if v9})
            if natural9:
                sample9["surviving_cells"] = [
                    [int(cell9[0]), int(cell9[1]), int(cell9[2]), kind9]
                    for cell9, kind9 in natural9[:8]]
            samples9.append(sample9)
        converged9 = [s9 for s9 in samples9 if s9.get("converged") is True]
        if len(converged9) < 2:
            raise RuntimeError(
                f"census scrub outer audit got {len(converged9)}/4 converged "
                "samples; refusing to certify absence")
        survivors9 = sum(int(s9["roster_ore_cells"]) for s9 in converged9)
        if survivors9:
            raise RuntimeError(
                f"census scrub outer audit found {survivors9} surviving "
                f"natural roster-ore cells: {samples9}")
        if not self.go(free["x"], free["y"], free["z"], 0, 8, n=4):
            raise RuntimeError(
                "census scrub could not return to the accepted site centre")
        audit_wall9 = time.monotonic() - audit_started9
        witness9 = dict(region_record9)
        witness9.update({
            "classes": list(classes9),
            "fill_command_count": int(fill_count9),
            "fill_slab_count": int(len(slabs9)),
            "fill_volume_limit": int(SC.CENSUS_SCRUB_FILL_VOLUME_LIMIT),
            "forceload_used": True,
            "central_envelope_box": central_box9,
            "pre_scrub_central_counts": {
                k9: int(v9) for k9, v9 in pre_counts9.items()},
            "outer_audit_samples": samples9,
            "wall_seconds": {
                "pre_scan": round(pre_scan_wall9, 3),
                "fill": round(fill_wall9, 3),
                "outer_audit": round(audit_wall9, 3),
                "total": round(time.monotonic() - started9, 3),
            },
        })
        return witness9

    def _census_scrub_central_audit(self, witness, grid, staged_cells):
        classes9 = tuple(str(k9) for k9 in witness["classes"])
        _, found9 = self._census_scrub_roster_cells(grid, classes9)
        staged_set9 = {tuple(int(v9) for v9 in c9) for c9 in staged_cells}
        survivors9 = [(cell9, kind9) for cell9, kind9 in found9
                      if cell9 not in staged_set9]
        witness9 = {
            "scanned_cells": int(len(grid)),
            "staged_roster_cells": int(len(found9) - len(survivors9)),
            "remaining_natural_roster_cells": int(len(survivors9)),
        }
        if survivors9:
            witness9["surviving_cells"] = [
                [int(c9[0]), int(c9[1]), int(c9[2]), k9]
                for (c9, k9) in survivors9[:16]]
        witness["post_scrub_central_audit"] = witness9
        if survivors9:
            raise RuntimeError(
                "census scrub central audit found surviving natural roster "
                f"ore after placement: {witness9['surviving_cells']}")
        total9 = witness.get("wall_seconds") or {}
        total9["central_audit_appended"] = True
        witness["wall_seconds"] = total9

    def _mine_reachable_surface(self, grid, center):
        levels = defaultdict(list)
        for (x, y, z), block in sorted(grid.items()):
            if not self._mine_walk_support(block):
                continue
            fy = y + 1
            if not (self._mine_scan_contains((x, y, z))
                    and self._mine_scan_contains((x, fy, z))
                    and self._mine_scan_contains((x, fy + 1, z))):
                continue
            if (not self._mine_walk_blocked(grid.get((x, fy, z)))
                    and not self._mine_walk_blocked(grid.get((x, fy + 1, z)))):
                levels[(x, z)].append(fy)
        for values9 in levels.values():
            values9.sort()
        if not levels:
            return frozenset()
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        sx, sz = int(math.floor(cx)), int(math.floor(cz))
        starts = [(abs(fy - cy), sx, fy, sz) for fy in levels.get((sx, sz), [])]
        if not starts:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for fy in levels.get((sx + dx, sz + dz), []):
                        starts.append((abs(fy - cy) + math.hypot(dx, dz),
                                       sx + dx, fy, sz + dz))
        if not starts:
            return frozenset()
        _, x0, y0, z0 = min(starts)
        seen = {(x0, y0, z0)}
        queue = deque([(x0, y0, z0)])
        while queue:
            x, fy, z = queue.popleft()
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, nz = x + dx, z + dz
                for nfy in levels.get((nx, nz), []):
                    if abs(nfy - fy) > 1:
                        continue
                    state = (nx, nfy, nz)
                    if state not in seen:
                        seen.add(state)
                        queue.append(state)
        return frozenset(seen)

    def _validate_frozen_mine_survey(self, grid, route, checkpoint_steps):
        route9 = [tuple(int(v9) for v9 in q9) for q9 in (route or ())]
        if len(route9) < 2:
            raise RuntimeError("screened mine survey route is empty after placement")
        for i9, q9 in enumerate(route9):
            x9, fy9, z9 = q9
            support9, feet9, head9 = (
                (x9, fy9 - 1, z9), (x9, fy9, z9), (x9, fy9 + 1, z9))
            if not all(self._mine_scan_contains(cell9) for cell9 in
                       (support9, feet9, head9)):
                raise RuntimeError(
                    f"screened mine survey step left verified scan at {i9}: {q9}")
            if (not self._mine_walk_support(grid.get(support9))
                    or self._mine_walk_blocked(grid.get(feet9))
                    or self._mine_walk_blocked(grid.get(head9))):
                raise RuntimeError(
                    f"screened mine survey step lost support/headroom at {i9}: {q9}")
            if i9:
                prev9 = route9[i9 - 1]
                cardinal9 = abs(q9[0] - prev9[0]) + abs(q9[2] - prev9[2])
                if cardinal9 != 1 or abs(q9[1] - prev9[1]) > 1:
                    raise RuntimeError(
                        f"screened mine survey has illegal transition {prev9}->{q9}")
        steps9 = [int(v9) for v9 in (checkpoint_steps or ())]
        if (not steps9 or steps9 != sorted(set(steps9))
                or any(v9 < 1 or v9 >= len(route9) for v9 in steps9)):
            raise RuntimeError(
                f"screened mine survey checkpoints changed after placement: {steps9}")
        return {
            "route": [[int(v9) for v9 in q9] for q9 in route9],
            "checkpoints": [[int(v9) for v9 in route9[i9]] for i9 in steps9],
            "checkpoint_steps": steps9,
        }

    def _mine_access_audit(self, grid, cell, reachable):
        if not reachable:
            return dict(accessible=False, access_dist=None, access_pos=None)
        x, y, z = cell
        bounds = getattr(self, "_last_mine_scan_bounds", None)

        def inside(c):
            if bounds is None:
                return False
            return (bounds[0] <= c[0] < bounds[1] and bounds[2] <= c[1] < bounds[3]
                    and bounds[4] <= c[2] < bounds[5])

        best = None
        nearby = [(rx, rfy, rz)
                  for rx in range(x - 5, x + 6) for rz in range(z - 5, z + 6)
                  for rfy in reachable.get((rx, rz), ())]
        for rx, rfy, rz in nearby:
            eye = np.asarray((rx + .5, rfy + 1.62, rz + .5), dtype=float)
            for oy in (.22, .50, .78):
                target = np.asarray((x + .5, y + oy, z + .5), dtype=float)
                ray = target - eye
                dist = float(np.linalg.norm(ray))
                if dist < 1.2 or dist > MINE_REACH:
                    continue
                pitch = math.degrees(math.atan2(eye[1] - target[1],
                                                 max(.25, math.hypot(ray[0], ray[2]))))
                if abs(pitch) > 48.0:
                    continue
                clear = True
                for tt in np.arange(.08, max(.09, dist - .42), .22):
                    p = eye + ray * (tt / dist)
                    q = tuple(int(math.floor(v)) for v in p)
                    if q == cell:
                        continue
                    if not inside(q) or self._mine_solid(grid.get(q)):
                        clear = False
                        break
                rank9 = (dist, rx, rfy, rz, oy)
                if clear and (best is None or rank9 < best[0]):
                    best = (rank9, (rx, rfy, rz), pitch)
        if best is None:
            return dict(accessible=False, access_dist=None, access_pos=None)
        return dict(accessible=True, access_dist=round(best[0][0], 3),
                    access_pos=[int(v) for v in best[1]], access_pitch=round(best[2], 2))

    @staticmethod
    def _mine_primary_pocket_profile(center, reachable):
        cx, cy, cz = (float(v9) for v9 in center)
        states9 = frozenset(tuple(int(v9) for v9 in q9) for q9 in reachable)
        if not states9:
            return None
        sx9, sz9 = int(math.floor(cx)), int(math.floor(cz))
        starts9 = [
            (abs(q9[1] - cy) + math.hypot(q9[0] - sx9, q9[2] - sz9), q9)
            for q9 in states9
            if abs(q9[0] - sx9) <= 1 and abs(q9[2] - sz9) <= 1
        ]
        if not starts9:
            return None
        start9 = min(starts9, key=lambda q9: (q9[0], q9[1]))[1]
        mutable_columns9 = defaultdict(list)
        for q9 in states9:
            mutable_columns9[(q9[0], q9[2])].append(q9)
        columns9 = {
            key9: tuple(sorted(values9))
            for key9, values9 in mutable_columns9.items()
        }

        def neighbours9(q9):
            x9, fy9, z9 = q9
            for dx9, _dy9, dz9 in _MINE_HORIZONTAL:
                for next9 in columns9.get((x9 + dx9, z9 + dz9), ()):
                    if abs(next9[1] - fy9) <= 1:
                        yield next9

        branches9 = len({(q9[0], q9[2]) for q9 in neighbours9(start9)})
        distance9 = {start9: 0}
        queue9 = deque([start9])
        while queue9:
            current9 = queue9.popleft()
            next_distance9 = distance9[current9] + 1
            if next_distance9 > MINE_PRIMARY_MAX_ACCESS_PATH_STEPS:
                continue
            for next9 in neighbours9(current9):
                if next9 not in distance9:
                    distance9[next9] = next_distance9
                    queue9.append(next9)
        survey9 = terrain_only_survey_route((cx, cy, cz), states9)
        return {
            "states": states9,
            "start": start9,
            "branches": int(branches9),
            "distance": distance9,
            "survey": survey9,
        }

    @classmethod
    def _mine_primary_pocket_audit(cls, center, cell, reachable, access_pos,
                                   profile=None):
        cx, _cy, cz = (float(v9) for v9 in center)
        tx, _ty, tz = (int(v9) for v9 in cell)
        horizontal9 = math.hypot(tx + 0.5 - cx, tz + 0.5 - cz)
        result9 = {
            "primary_pocket_ok": False,
            "start_target_horizontal": round(horizontal9, 3),
            "start_access_path_steps": None,
            "start_surface_branches": 0,
            "start_surface_pos": None,
        }
        if horizontal9 < MINE_PRIMARY_MIN_HORIZONTAL or not reachable or access_pos is None:
            return result9

        profile9 = profile or cls._mine_primary_pocket_profile(center, reachable)
        if profile9 is None:
            return result9
        states9 = profile9["states"]
        start9 = profile9["start"]
        goal9 = tuple(int(v9) for v9 in access_pos)
        if goal9 not in states9:
            return result9

        branches9 = int(profile9["branches"])
        result9["start_surface_branches"] = int(branches9)
        result9["start_surface_pos"] = [int(v9) for v9 in start9]
        if branches9 < MINE_PRIMARY_MIN_START_BRANCHES:
            return result9
        steps9 = profile9["distance"].get(goal9)
        result9["start_access_path_steps"] = steps9
        result9["primary_pocket_ok"] = bool(
            steps9 is not None and steps9 <= MINE_PRIMARY_MAX_ACCESS_PATH_STEPS)
        return result9


    @staticmethod
    def v12_independent_multiclass6_layout(pair_plan, target_kind,
                                           distractor_kinds, layout_seed,
                                           roster=None):
        plan9 = copy.deepcopy(dict(pair_plan or {}))
        if plan9.get("contract") != getattr(
                OC, "COUNTERFACTUAL_PAIRED_SCENE_CONTRACT",
                "same_physical_scene_goal_role_only/v1"):
            raise ValueError("multiclass6 needs a certified two-lane geometry")
        target9 = OC.bare_block(target_kind)
        distractors9 = tuple(OC.bare_block(q9) for q9 in distractor_kinds)
        if (len(distractors9) != 3 or len(set(distractors9)) != 3
                or target9 in distractors9):
            raise ValueError(
                "multiclass6 requires three distinct non-goal classes")
        roster9 = set(OC.validate_mine_pool(
            OC.MINE_TARGET_POOL if roster is None else roster))
        if target9 not in roster9 or not set(distractors9) <= roster9:
            raise ValueError("multiclass6 classes escaped the mine roster")
        lanes9 = [copy.deepcopy(dict(q9)) for q9 in plan9.get("lanes", ())]
        if len(lanes9) != 2 or any(len(q9.get("specs", ())) != 3 for q9 in lanes9):
            raise ValueError("multiclass6 geometry must have two three-cell lanes")
        def lane_span9(lane9):
            cells9 = [tuple(float(v9) for v9 in q9["slot"])
                      for q9 in lane9["specs"]]
            return max(
                math.sqrt(sum((a9[i9] - b9[i9]) ** 2 for i9 in range(3)))
                for index9, a9 in enumerate(cells9)
                for b9 in cells9[index9 + 1:])

        lane_spans9 = {int(q9["lane_id"]): lane_span9(q9) for q9 in lanes9}
        eligible_target_lanes9 = [
            q9 for q9 in lanes9
            if lane_spans9[int(q9["lane_id"])]
               <= float(V12_INDEPENDENT_TARGET_MAX_SPAN_BLOCKS) + 1e-9]
        if not eligible_target_lanes9:
            raise ValueError(
                "multiclass6 has no target lane inside the local span bound")
        target_lane9 = min(
            eligible_target_lanes9,
            key=lambda q9: hashlib.sha256(
                (f"v12-bounded-random-lane:{int(layout_seed)}:"
                 f"{int(q9['lane_id'])}").encode()).digest())
        target_lane_id9 = int(target_lane9["lane_id"])
        distractor_lane9 = next(
            q9 for q9 in lanes9 if int(q9["lane_id"]) != target_lane_id9)

        ordered_distractors9 = sorted(
            distractors9,
            key=lambda kind9: hashlib.sha256(
                f"v12-multiclass6:{int(layout_seed)}:{kind9}".encode()).digest())
        target_specs9 = []
        for raw9 in target_lane9["specs"]:
            spec9 = copy.deepcopy(dict(raw9))
            spec9.update(kind=target9, scene_role="target")
            target_specs9.append(spec9)
        distractor_specs9 = []
        for raw9, kind9 in zip(distractor_lane9["specs"], ordered_distractors9):
            spec9 = copy.deepcopy(dict(raw9))
            spec9.update(kind=kind9, scene_role="distractor")
            distractor_specs9.append(spec9)
        all_specs9 = target_specs9 + distractor_specs9
        digest_payload9 = {
            "contract": V12_INDEPENDENT_MULTICLASS6_CONTRACT,
            "selection_profile": "v12_974f505",
            "target_kind": target9,
            "distractor_kinds": list(ordered_distractors9),
            "target_lane_id": target_lane_id9,
            "target_lane_selection": "bounded_random_lane_action_neutral",
            "target_max_span_blocks": float(
                V12_INDEPENDENT_TARGET_MAX_SPAN_BLOCKS),
            "eligible_target_lane_ids": sorted(
                int(q9["lane_id"]) for q9 in eligible_target_lanes9),
            "lane_spans_blocks": {
                str(key9): round(float(value9), 6)
                for key9, value9 in sorted(lane_spans9.items())},
            "cells": [{"kind": str(q9["kind"]),
                       "role": str(q9["scene_role"]),
                       "slot": [int(v9) for v9 in q9["slot"]]}
                      for q9 in all_specs9],
            "shared_decision_checkpoint": copy.deepcopy(
                plan9.get("shared_decision_checkpoint")),
            "terrain_route_sha256": str(plan9.get("terrain_route_sha256") or ""),
        }
        digest9 = hashlib.sha256(json.dumps(
            digest_payload9, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        return {
            **digest_payload9,
            "layout_sha256": digest9,
            "canonical_classes": sorted({target9, *ordered_distractors9}),
            "target_specs": target_specs9,
            "distractor_specs": distractor_specs9,
            "all_specs": all_specs9,
            "scan_bounds": copy.deepcopy(plan9.get("scan_bounds")),
            "placement_source": plan9.get("placement_source"),
            "internal_geometry_contract": plan9.get("contract"),
        }

    def exposed_mine_slots(
            self, center=None, span=24, max_dist=24.0, max_dy=12.0):
        grid = self.scan_mine_terrain(span=span)
        if center is None:
            center = self.w.get_pos()
        cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))

        def usable(cell):
            x, y, z = cell
            return not (math.hypot(x + 0.5 - cx, z + 0.5 - cz) > max_dist
                        or abs(y - cy) > max_dy)

        natural = {"coal_ore": [], "iron_ore": []}
        stone = []
        diag9 = Counter({"grid_cells": len(grid)})
        raw_candidates9 = []
        for cell, block in grid.items():
            if not usable(cell):
                continue
            diag9["within_search"] += 1
            if not self._mine_scan_contains(cell, margin=1):
                continue
            diag9["scan_interior"] += 1
            bare = block.split(":")[-1]
            if bare not in natural and bare != "stone":
                continue
            diag9["resource_material"] += 1
            geom = self._ore_geometry(grid, cell)
            if not geom["ok"]:
                continue
            diag9["safe_geometry"] += 1
            x, y, z = cell
            if bare == "stone":
                if geom["side_exposed"] < 1:
                    continue
                geom["placement_geometry"] = ORE_SIDE_PLACEMENT_GEOMETRY
            if bare in natural and not any(
                    str(grid.get((x + dx, y + dy, z + dz), "")).split(":")[-1] == "stone"
                    for dx, dy, dz in _MINE_NEIGHBORS):
                continue
            diag9["natural_or_replaceable"] += 1
            raw_candidates9.append((cell, bare, geom))

        reachable_states = self._mine_reachable_surface(grid, (cx, cy, cz))
        reachable = defaultdict(list)
        for rx, rfy, rz in reachable_states:
            reachable[(rx, rz)].append(rfy)
        primary_occ9 = self.mine_occupancy(grid, self._last_mine_scan_bounds)
        primary_profile9 = self._mine_primary_pocket_profile(
            (cx, cy, cz), reachable_states)
        self._last_mine_survey_contract = None
        if primary_profile9 is not None:
            self._last_mine_survey_contract = dict(primary_profile9["survey"])
        self._last_mine_occupancy = primary_occ9
        diag9.update({
            "reachable_states": len(reachable_states),
            "survey_checkpoints": (0 if primary_profile9 is None else
                                   len(primary_profile9["survey"]["checkpoint_steps"])),
        })
        for cell, bare, geom in raw_candidates9:
            x, y, z = cell
            access = self._mine_access_audit(
                grid, cell, reachable)
            if not access["accessible"]:
                continue
            diag9["interaction_access"] += 1
            geom.update(access)
            if primary_profile9 is None:
                continue
            pocket9 = self._mine_primary_pocket_audit(
                (cx, cy, cz), cell, reachable_states, access["access_pos"],
                profile=primary_profile9)
            geom.update(pocket9)
            if pocket9["primary_pocket_ok"]:
                diag9["primary_eligible_pocket"] += 1
            discovery9 = first_survey_checkpoint_perceptibility(
                primary_profile9["survey"], cell, primary_occ9,
                max_step=MINE_SURVEY_ROUTE_STEPS,
                max_dist=MINE_CHAIN_RADIUS)
            if discovery9 is None:
                continue
            diag9["checkpoint_perceptible"] += 1
            geom.update(discovery9)
            geom["terrain_topology"] = mine_candidate_topology_witness(
                grid, self._last_mine_survey_contract, cell, geom)
            geom["start_perceptibility"] = dict(discovery9["survey_perceptibility"])
            dist = math.hypot(x + 0.5 - cx, z + 0.5 - cz)
            rank = (geom.get("survey_checkpoint_step", 0),
                    0 if geom["side_exposed"] else 1, geom["access_dist"],
                    -geom["contacts"], dist, abs(y - cy), cell)
            if bare in natural:
                natural[bare].append((rank, cell, geom))
                diag9[f"accepted_{bare}"] += 1
            elif bare == "stone":
                stone.append((rank, cell, geom))
                diag9["accepted_stone"] += 1
        for vals in natural.values():
            vals.sort(key=lambda v: v[0])
        stone.sort(key=lambda v: v[0])
        self._last_mine_slot_diagnostics = dict(sorted(diag9.items()))
        return grid, natural, stone

    @classmethod
    def exhaustive_pair_class_pose_census(cls, grid, bounds, classes, stance, yaw, *,
                                           max_dist=MINE_CHAIN_RADIUS, pitch=8.0):
        classes9 = tuple(sorted({OC.bare_block(q9) for q9 in classes}))
        occ9 = cls.mine_occupancy(grid, tuple(int(v9) for v9 in bounds))
        indexed9 = {kind9: [] for kind9 in classes9}
        for cell9, raw9 in sorted(grid.items()):
            kind9 = _natural_registry_kind(raw9).split(":", 1)[-1]
            if kind9 in indexed9:
                indexed9[kind9].append(tuple(int(v9) for v9 in cell9))
        result9 = {}
        for kind9 in classes9:
            visible9 = []
            for cell9 in indexed9[kind9]:
                proof9 = stance_perceptibility(
                    stance, yaw, cell9, occ9, max_dist=max_dist, pitch=pitch)
                if proof9 is not None:
                    visible9.append({"cell": list(cell9), "proof": dict(proof9)})
            result9[kind9] = {
                "indexed_cells": [list(q9) for q9 in indexed9[kind9]],
                "recognizable_cells": visible9,
                "recognizable": bool(visible9),
            }
        return result9

    def biome_plausible_pair_slots(self, target_kind, confuser_kind, center=None,
                                    span=MINE_SITE_SCAN_SPAN, *, max_lane_candidates=72):
        target9, confuser9 = OC.bare_block(target_kind), OC.bare_block(confuser_kind)
        if target9 == confuser9:
            raise ValueError("counterfactual pair needs two distinct classes")
        families9 = {str(OC.harvest_rule(q9)["target_family"])
                     for q9 in (target9, confuser9)}
        if families9 != {"ore"}:
            raise ValueError(
                f"biome placement pair must be two ore classes, got "
                f"{target9}/{confuser9} families={sorted(families9)}")
        return self.biome_plausible_ore_pair_slots(
            target9, confuser9, center=center, span=span,
            max_lane_candidates=max_lane_candidates)

    def biome_plausible_ore_pair_slots(self, target_kind, confuser_kind, center=None,
                                        span=MINE_SITE_SCAN_SPAN, *,
                                        max_lane_candidates=72):
        target9, confuser9 = OC.bare_block(target_kind), OC.bare_block(confuser_kind)
        classes9 = tuple(sorted((target9, confuser9)))
        if (target9 == confuser9
                or any(OC.harvest_rule(q9)["target_family"] != "ore" for q9 in classes9)):
            raise ValueError("ore counterfactual pair needs two distinct ore classes")
        center9 = tuple(float(v9) for v9 in (center or self.w.get_pos()))
        grid9, _natural9, stone9 = self.exposed_mine_slots(
            center=center9, span=span, max_dist=MINE_CHAIN_RADIUS,
            max_dy=MINE_CHAIN_MAX_DY)
        bounds9 = tuple(int(v9) for v9 in self._last_mine_scan_bounds)
        survey9 = dict(self._last_mine_survey_contract or {})
        occ9 = self._last_mine_occupancy
        diag9 = Counter(dict(self._last_mine_slot_diagnostics or {}))
        if not survey9.get("route") or not survey9.get("checkpoint_steps") or occ9 is None:
            diag9["missing_terrain_survey"] += 1
            self._last_mine_slot_diagnostics = dict(sorted(diag9.items()))
            return grid9, None
        route_supports9 = {
            (int(q9[0]), int(q9[1]) - 1, int(q9[2])) for q9 in survey9["route"]}
        specs9 = []
        placement_source9 = getattr(
            OC, "EPISODE_BIOME_PLACEMENT_SOURCE", "biome_plausible_placement")
        for _rank9, cell9, geom9 in stone9:
            cell9 = tuple(int(v9) for v9 in cell9)
            if cell9 in route_supports9:
                continue
            spec9 = {
                "slot": cell9, "source": placement_source9, "original": "stone",
                "survey_step": int(geom9["survey_checkpoint_step"]),
                "survey_view_pos": list(geom9["survey_view_pos"]),
                "primary_eligible": bool(geom9["primary_pocket_ok"]),
                "terrain_topology": copy.deepcopy(
                    geom9.get("terrain_topology") or {}),
                "geometry": {
                    **dict(geom9),
                    "placement_geometry": str(
                        geom9.get("placement_geometry")
                        or ORE_SIDE_PLACEMENT_GEOMETRY),
                },
            }
            if mine_slot_reachable_stance(occ9, spec9) is not None:
                specs9.append(spec9)
        specs9.sort(key=lambda q9: (q9["survey_step"], q9["slot"]))
        diag9["ore_pair_target_grade_candidates"] = len(specs9)
        specs9 = [q9 for q9 in specs9 if bool(q9.get("primary_eligible"))]
        diag9["ore_pair_primary_access_candidates"] = len(specs9)
        selection_profile9 = "v12_974f505"
        preferred_requested9 = selected_biome_terrain_route_archetype(center9)
        preferred_count9 = sum(
            dict(q9.get("terrain_topology") or {}).get("route_archetype")
            == preferred_requested9 for q9 in specs9)
        diag9["ore_pair_preferred_topology_candidates"] = int(preferred_count9)
        specs9 = specs9[:max(6, int(max_lane_candidates))]
        if (int(diag9["ore_pair_target_grade_candidates"]) < 6
                or int(diag9["ore_pair_primary_access_candidates"]) < 6):
            self._last_mine_slot_diagnostics = dict(sorted(diag9.items()))
            return grid9, None

        lanes9 = select_diverse_ore_lane_candidates(
            specs9, max_candidates=max_lane_candidates,
            min_gap=MINE_COUNTERFACTUAL_LANE_MIN_GAP)
        diag9["ore_class_independent_lane_candidates"] = len(lanes9)
        assignment_order9 = [
            (i9, j9) for i9 in range(len(lanes9))
            for j9 in range(i9 + 1, len(lanes9))]
        for lane0_i9, lane1_i9 in assignment_order9:
            lane0_specs9 = lanes9[lane0_i9]
            lane1_specs9 = lanes9[lane1_i9]
            lane0_cells9 = {q9["slot"] for q9 in lane0_specs9}
            lane1_cells9 = {q9["slot"] for q9 in lane1_specs9}
            if lane0_cells9 & lane1_cells9:
                continue
            if min(math.dist(a9, b9) for a9 in lane0_cells9
                   for b9 in lane1_cells9) < MINE_COUNTERFACTUAL_LANE_MIN_GAP:
                diag9["ore_lanes_too_close"] += 1
                continue
            lane_records9 = []
            for lane_i9, raw_specs9 in enumerate((lane0_specs9, lane1_specs9)):
                with_lane9 = []
                for raw9 in raw_specs9:
                    row9 = dict(raw9)
                    row9.update(kind=classes9[lane_i9], pair_lane_id=int(lane_i9))
                    with_lane9.append(row9)
                chain9, _proofs9, _bad9 = select_stanceable_mine_chain(
                    with_lane9, occ9, count=3)
                if (chain9 is None
                        or any(not bool(q9.get("primary_eligible"))
                               for q9 in chain9)):
                    break
                lane_records9.append({
                    "lane_id": int(lane_i9), "kind": classes9[lane_i9],
                    "cells": [list(q9["slot"]) for q9 in chain9],
                    "specs": chain9,
                })
            if len(lane_records9) != 2:
                continue
            lane_spans9 = []
            for lane9 in lane_records9:
                cells9 = [tuple(float(v9) for v9 in q9["slot"])
                          for q9 in lane9["specs"]]
                lane_spans9.append(max(
                    math.dist(a9, b9)
                    for index9, a9 in enumerate(cells9)
                    for b9 in cells9[index9 + 1:]))
            if min(lane_spans9) > (
                    float(V12_INDEPENDENT_TARGET_MAX_SPAN_BLOCKS) + 1e-9):
                diag9["ore_v12_no_bounded_target_lane"] += 1
                continue
            shared9 = None
            for step9 in survey9["checkpoint_steps"]:
                pose9 = survey_checkpoint_stance(survey9, step9)
                if pose9 is None:
                    continue
                stance9, yaw9 = pose9
                per_lane9 = []
                for lane9 in lane_records9:
                    visible9 = []
                    for spec9 in lane9["specs"]:
                        proof9 = stance_perceptibility(
                            stance9, yaw9, tuple(spec9["slot"]), occ9)
                        if proof9 is not None:
                            visible9.append((spec9, proof9))
                    per_lane9.append(visible9)
                if all(per_lane9):
                    shared9 = {
                        "step": int(step9), "view_pos": list(stance9),
                        "yaw": round(float(yaw9), 3),
                        "lane_cells": [list(q9[0][0]["slot"]) for q9 in per_lane9],
                        "lane_perceptibility": [
                            dict(q9[0][1]["survey_perceptibility"])
                            for q9 in per_lane9],
                    }
                    break
            if shared9 is None:
                diag9["ore_no_shared_decision_checkpoint"] += 1
                continue
            class_to_lane9 = {kind9: i9 for i9, kind9 in enumerate(classes9)}
            digest_payload9 = {
                "contract": getattr(
                    OC, "COUNTERFACTUAL_PAIRED_SCENE_CONTRACT",
                    "same_physical_scene_goal_role_only/v1"),
                "family": "ore", "classes": list(classes9),
                "lanes": [{"lane_id": q9["lane_id"], "kind": q9["kind"],
                           "cells": q9["cells"]} for q9 in lane_records9],
                "start": [round(v9, 4) for v9 in center9],
                "route": survey9["route"],
                "checkpoints": survey9["checkpoint_steps"],
            }
            digest9 = hashlib.sha256(json.dumps(
                digest_payload9, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            proposed9 = dict(grid9)
            for lane9 in lane_records9:
                for cell9 in lane9["cells"]:
                    proposed9[tuple(cell9)] = f"minecraft:{lane9['kind']}"
            plan9 = {
                **digest_payload9,
                "placement_source": placement_source9,
                "canonical_classes": list(classes9),
                "class_to_lane": class_to_lane9,
                "lanes": lane_records9,
                "pair_class_assignment_fixed": True,
                "terrain_route_sha256": digest9,
                "shared_decision_checkpoint": shared9,
                "all_cells": [cell9 for lane9 in lane_records9
                              for cell9 in lane9["cells"]],
                "scan_bounds": list(bounds9),
                "local_f0_class_census": None,
                "biome_terrain_topology": {
                    "contract": BIOME_TERRAIN_TOPOLOGY_CONTRACT,
                    "preference_source": "sha256_integer_site_xz/v1",
                    "requested_route_archetype": preferred_requested9,
                    "preference_enabled": False,
                    "preferred_candidate_count": int(preferred_count9),
                    "lane_route_archetypes": [
                        [dict(spec9.get("terrain_topology") or {}).get(
                            "route_archetype") for spec9 in lane9["specs"]]
                        for lane9 in lane_records9],
                    "lane_host_archetypes": [
                        [dict(spec9.get("terrain_topology") or {}).get(
                            "host_archetype") for spec9 in lane9["specs"]]
                        for lane9 in lane_records9],
                },
                "selection_profile": selection_profile9,
            }
            self._last_biome_pair_planned_grid = proposed9
            self._last_mine_slot_diagnostics = dict(sorted(diag9.items()))
            return grid9, plan9
        self._last_biome_pair_planned_grid = None
        self._last_mine_slot_diagnostics = dict(sorted(diag9.items()))
        return grid9, None

    def fade(self):
        for _ in range(210):
            self.w.step_noop()

    @staticmethod
    def _entity_selector_key(etype, selector=None):
        if selector is None:
            kind = str(etype).strip()
            if not kind:
                raise ValueError("entity type must be non-empty")
            if ":" not in kind:
                kind = f"minecraft:{kind}"
            body = f"type={kind},limit=1,sort=nearest"
        else:
            body = str(selector).strip()
            if body.startswith("@e[") and body.endswith("]"):
                body = body[3:-1].strip()
        if not body or any(ch in body for ch in ("\r", "\n", "]")):
            raise ValueError(f"unsafe entity selector: {body!r}")
        return f"@e[{body}]"

    def bind_entity_log_camera_off(self):
        import glob as _g

        LOGSTATE = self.LOGSTATE
        bound = LOGSTATE.get("log")
        if bound and os.path.isfile(bound):
            return True
        if bound:
            LOGSTATE.pop("log", None)
            LOGSTATE.pop("entity_bind_mark", None)

        mark = LOGSTATE.get("entity_bind_mark")
        if mark is None:
            mark = f"XV2STAGE_{os.getpid()}_{id(self):x}"
            LOGSTATE["entity_bind_mark"] = mark
            self.w.cmd(f"/me {mark}")
            return False

        for path in sorted(_g.glob(os.path.join(self.repo, "logs", "mc_*.log")),
                           key=os.path.getmtime, reverse=True):
            try:
                with open(path, errors="ignore") as handle:
                    if mark not in handle.read():
                        continue
            except OSError:
                continue
            LOGSTATE["log"] = path
            LOGSTATE["entity_feedback"] = True
            self.w.cmd("/gamerule sendCommandFeedback true")
            return True
        return False

    @staticmethod
    def _parse_entity_query_reply(tail):
        import re as _re

        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        pos_re = _re.compile(
            rf"entity data:\s*\[({number})d,\s*({number})d,\s*({number})d\]",
            _re.IGNORECASE,
        )
        absent_re = _re.compile(r"No entity was found|Found no elements", _re.IGNORECASE)
        pos = pos_re.search(tail)
        absent = absent_re.search(tail)
        if pos is not None and (absent is None or pos.start() < absent.start()):
            return ENTITY_QUERY_POSITION, tuple(float(v) for v in pos.groups())
        if absent is not None:
            return ENTITY_QUERY_ABSENT, None
        return None, None

    def _entity_query_result(self, selector, *, fresh=False, reason=None):
        state = self.ENTITY_QUERY_STATE
        now = int(state["call"])
        cached = state["cache"].get(selector)
        if cached is None:
            status, position, updated = ENTITY_QUERY_UNKNOWN, None, None
        else:
            status = cached.get("status", ENTITY_QUERY_UNKNOWN)
            position = cached.get("position")
            updated = cached.get("updated_call")
        pending = state.get("pending")
        pending_here = bool(pending and pending.get("selector") == selector)
        if reason is None:
            if pending_here:
                reason = "pending"
            elif pending:
                reason = "blocked_by_other_selector"
            elif cached is not None:
                reason = cached.get("reason", "cached")
            else:
                reason = "unknown"
        return EntityQueryResult(
            selector=selector,
            status=status,
            position=position,
            fresh=bool(fresh),
            pending=pending_here,
            reason=str(reason),
            age_calls=None if updated is None else max(0, now - int(updated)),
        )

    def entity_pos_nonblocking(self, etype, selector=None, *, refresh_calls=4,
                               timeout_calls=3, issue_if_due=True):
        selector_key = self._entity_selector_key(etype, selector)
        state = self.ENTITY_QUERY_STATE
        state["call"] = int(state.get("call", 0)) + 1
        now = int(state["call"])
        timeout_key = None
        fresh_key = None
        fresh_reason = None

        pending = state.get("pending")
        if pending is not None:
            log_path = self.LOGSTATE.get("log")
            tail = None
            try:
                if log_path and os.path.getsize(log_path) >= int(pending["offset"]):
                    with open(log_path, errors="ignore") as handle:
                        handle.seek(int(pending["offset"]))
                        tail = handle.read()
            except OSError:
                tail = None

            if tail is None:
                timeout_key = pending["selector"]
                state["pending"] = None
                self.LOGSTATE.pop("log", None)
                self.LOGSTATE.pop("entity_bind_mark", None)
                reason = "log_unavailable"
                entry = state["cache"].setdefault(
                    timeout_key,
                    {"status": ENTITY_QUERY_UNKNOWN, "position": None,
                     "updated_call": None},
                )
                entry["reason"] = reason
                fresh_reason = reason
            else:
                status, position = self._parse_entity_query_reply(tail)
                if status is not None:
                    fresh_key = pending["selector"]
                    state["cache"][fresh_key] = {
                        "status": status,
                        "position": position,
                        "updated_call": now,
                        "reason": "reply",
                    }
                    state["pending"] = None
                    fresh_reason = "reply"
                else:
                    pending["poll_calls"] = int(pending.get("poll_calls", 0)) + 1
                    if pending["poll_calls"] >= int(pending["timeout_calls"]):
                        timeout_key = pending["selector"]
                        state["pending"] = None
                        entry = state["cache"].setdefault(
                            timeout_key,
                            {"status": ENTITY_QUERY_UNKNOWN, "position": None,
                             "updated_call": None},
                        )
                        entry["reason"] = "timeout"
                        fresh_reason = "timeout"

        if timeout_key is not None:
            return self._entity_query_result(
                selector_key,
                reason=fresh_reason if timeout_key == selector_key else None,
            )

        if not self.LOGSTATE.get("log"):
            if not bool(issue_if_due):
                return self._entity_query_result(
                    selector_key, reason="poll_only_log_unbound")
            self.bind_entity_log_camera_off()
            return self._entity_query_result(selector_key, reason="log_unbound")

        pending = state.get("pending")
        if pending is not None:
            return self._entity_query_result(selector_key)

        if fresh_key == selector_key:
            return self._entity_query_result(
                selector_key, fresh=True, reason=fresh_reason or "reply")

        if not bool(issue_if_due):
            return self._entity_query_result(
                selector_key, reason="poll_only_no_pending")

        refresh_calls = max(1, int(refresh_calls))
        last_issued = state["last_issued"].get(selector_key)
        due = last_issued is None or now - int(last_issued) >= refresh_calls
        if due:
            try:
                offset = os.path.getsize(self.LOGSTATE["log"])
            except OSError:
                self.LOGSTATE.pop("log", None)
                self.LOGSTATE.pop("entity_bind_mark", None)
                return self._entity_query_result(selector_key, reason="log_unavailable")
            self.w.cmd(f"/data get entity {selector_key} Pos")
            state["last_issued"][selector_key] = now
            state["pending"] = {
                "selector": selector_key,
                "offset": int(offset),
                "poll_calls": 0,
                "timeout_calls": max(1, int(timeout_calls)),
            }
            return self._entity_query_result(selector_key, reason="pending")

        return self._entity_query_result(selector_key, reason="cached")

    def entity_pos(self, etype, selector=None):
        import glob as _g, re as _re
        w, LOGSTATE = self.w, self.LOGSTATE
        if "log" not in LOGSTATE:
            mark = f"XV2DEMO_{os.getpid()}"
            w.cmd(f"/me {mark}")
            for _ in range(2):
                w.step_noop()
            for _ in range(12):
                for f in sorted(_g.glob(os.path.join(self.repo, "logs", "mc_*.log")),
                                key=os.path.getmtime, reverse=True):
                    try:
                        if mark in open(f, errors="ignore").read():
                            LOGSTATE["log"] = f
                            break
                    except OSError:
                        pass
                if "log" in LOGSTATE:
                    break
                w.step_noop()
            w.cmd("/gamerule sendCommandFeedback true")
            w.step_noop()
        if "log" not in LOGSTATE:
            return None
        n0 = os.path.getsize(LOGSTATE["log"])
        sel = selector or f"type=minecraft:{etype},limit=1,sort=nearest"
        w.cmd(f"/data get entity @e[{sel}] Pos")
        pat = _re.compile(r"\[([-\d.]+)d,\s*([-\d.]+)d,\s*([-\d.]+)d\]")
        for _ in range(6):
            w.step_noop()
            with open(LOGSTATE["log"], errors="ignore") as log_f9:
                m = pat.search(log_f9.read()[n0:])
            if m:
                LOGSTATE["miss"] = 0
                return tuple(float(v) for v in m.groups())
        LOGSTATE["miss"] = LOGSTATE.get("miss", 0) + 1
        if LOGSTATE["miss"] >= 2:
            LOGSTATE.pop("log", None)
            LOGSTATE["miss"] = 0
        return None

    def entity_exists(self, selector):
        import re as _re
        w, LOGSTATE = self.w, self.LOGSTATE
        if "log" not in LOGSTATE:
            self.entity_pos("sheep")
        if "log" not in LOGSTATE:
            return True
        n0 = os.path.getsize(LOGSTATE["log"])
        w.cmd(f"/data get entity @e[{selector}] Pos")
        for _ in range(6):
            w.step_noop()
            tail = open(LOGSTATE["log"], errors="ignore").read()[n0:]
            if _re.search(r"\[[-\d.]+d,", tail):
                return True
            if "No entity was found" in tail or "Found no elements" in tail:
                return False
        return True

    def _wait_pose_stable(self, stable_ticks=4, max_ticks=32,
                          expected_xyz=None, expected_yaw=None, expected_pitch=None):
        w = self.w
        stable9 = 0
        prev9 = (*w.get_pos(), w.get_yaw(), w.get_pitch())
        for _ in range(int(max_ticks)):
            w.step_noop()
            cur9 = (*w.get_pos(), w.get_yaw(), w.get_pitch())
            angle9 = max(abs((cur9[3] - prev9[3] + 180.0) % 360.0 - 180.0),
                         abs(cur9[4] - prev9[4]))
            moved9 = max(abs(cur9[i] - prev9[i]) for i in range(3))
            arrived9 = True
            if expected_xyz is not None:
                arrived9 = (math.hypot(cur9[0] - float(expected_xyz[0]),
                                       cur9[2] - float(expected_xyz[2])) <= 0.38
                            and abs(cur9[1] - float(expected_xyz[1])) <= 1.05)
            if expected_yaw is not None:
                arrived9 = arrived9 and abs(
                    (cur9[3] - float(expected_yaw) + 180.0) % 360.0 - 180.0) <= 1.0
            if expected_pitch is not None:
                arrived9 = arrived9 and abs(cur9[4] - float(expected_pitch)) <= 1.0
            stable9 = stable9 + 1 if (arrived9 and moved9 < 0.005 and angle9 < 0.05) else 0
            prev9 = cur9
            if stable9 >= int(stable_ticks):
                return prev9
        return None if (expected_xyz is not None or expected_yaw is not None) else prev9

    def _teleport_stable(self, x, y, z, yaw, pitch=8, max_ticks=32):
        for attempt9 in range(2):
            self.w.cmd(f"/tp @a {x:.2f} {y:.2f} {z:.2f} {yaw:.2f} {pitch:.2f}")
            got9 = self._wait_pose_stable(
                stable_ticks=4, max_ticks=max_ticks,
                expected_xyz=(x, y, z), expected_yaw=yaw, expected_pitch=pitch)
            if got9 is not None:
                return True
            cur9 = (*self.w.get_pos(), self.w.get_yaw(), self.w.get_pitch())
            print(f"[pose] teleport acknowledgement miss attempt={attempt9 + 1} "
                  f"target=({x:.2f},{y:.2f},{z:.2f},{yaw:.1f},{pitch:.1f}) "
                  f"observed=({cur9[0]:.2f},{cur9[1]:.2f},{cur9[2]:.2f},"
                  f"{cur9[3]:.1f},{cur9[4]:.1f})",
                  flush=True)
        return False

    def stand_ok(self, min_bright=20.0, forward_only=False):
        w = self.w
        self._wait_pose_stable()
        a = w.sim.noop_action()
        a["voxels"] = np.array([-2, 3, -3, 4, -2, 3], np.int32)
        w.obs, _, _, _, w.info = w.sim.step(a)
        cells = {(c["x"], c["y"], c["z"]): str(c.get("type", ""))
                 for c in (w.info.get("voxels") or [])}
        below = cells.get((0, -1, 0))
        veg = ("grass", "fern", "flower", "poppy", "dandelion", "bluet", "daisy",
               "cornflower", "tulip", "snow")
        clear = all(t is None or any(k in t for k in veg)
                    for t in (cells.get((0, 0, 0)), cells.get((0, 1, 0))))
        solid = below is not None and "water" not in below and "lava" not in below
        bright_samples9 = [float(np.asarray(w.info["pov"]).mean())]
        for _ in range(2):
            w.step_noop()
            bright_samples9.append(float(np.asarray(w.info["pov"]).mean()))
        bright = float(np.median(bright_samples9))
        if not (solid and clear and bright >= min_bright):
            return False

        yaw9 = math.radians(w.get_yaw())
        headings9 = ((yaw9,) if forward_only else
                     (yaw9, yaw9 + math.radians(35), yaw9 - math.radians(35)))
        for heading9 in headings9:
            dx9 = int(round(-math.sin(heading9)))
            dz9 = int(round(math.cos(heading9)))
            if dx9 == 0 and dz9 == 0:
                continue
            for support_y9 in (0, -1, -2):
                support9 = cells.get((dx9, support_y9, dz9))
                if not self._mine_solid(support9):
                    continue
                foot9 = cells.get((dx9, support_y9 + 1, dz9))
                head9 = cells.get((dx9, support_y9 + 2, dz9))
                if (not self._mine_solid(foot9) and not self._mine_solid(head9)
                        and not any(q9 in str(foot9) or q9 in str(head9)
                                    for q9 in ("water", "lava"))):
                    return True
        return False

    def fixed_start_yaw_ok(self, x, y, z, yaw, min_bright=25.0):
        if not self._teleport_stable(float(x), float(y), float(z), float(yaw), 8):
            return False
        if not self.stand_ok(min_bright=float(min_bright), forward_only=True):
            return False
        frame9 = np.asarray(self.w.info["pov"])
        if float(frame9.mean()) < float(min_bright):
            return False
        top9 = frame9[:100].astype(int)
        sky_mask9 = ((top9[..., 2] > 130)
                     & (top9[..., 2] > top9[..., 0] + 18))
        sky9 = float(max(sky_mask9[:, :213].mean(),
                         sky_mask9[:, 213:426].mean(),
                         sky_mask9[:, 426:].mean()))
        action9 = self.w.sim.noop_action()
        action9["voxels"] = np.array([-6, 7, -3, 4, -6, 7], np.int32)
        self.w.obs, _, _, _, self.w.info = self.w.sim.step(action9)
        solids9 = {
            (int(b9["x"]), int(b9["y"]), int(b9["z"]))
            for b9 in (self.w.info.get("voxels") or [])
            if (self._mine_solid(str(b9.get("type", "")))
                or str(b9.get("type", "")).split(":")[-1].endswith("_leaves"))
        }
        open_rays9 = 0
        for offset9 in (-35.0, 0.0, 35.0):
            theta9 = math.radians(float(yaw) + offset9)
            ray_open9 = True
            for radius9 in np.linspace(0.75, 5.0, 18):
                qx9 = int(math.floor(-math.sin(theta9) * radius9))
                qz9 = int(math.floor(math.cos(theta9) * radius9))
                if any((qx9, qy9, qz9) in solids9 for qy9 in (1, 2)):
                    ray_open9 = False
                    break
            open_rays9 += int(ray_open9)
        return bool(sky9 > 0.10 or open_rays9 >= 1)

    def start_pose(self, tx, tz, dist, setting, jitter, max_retreat=34,
                   require_water_free=False, allow_night=False,
                   fixed_start=None, require_forward_clear=False,
                   fixed_yaw=None, allow_enclosed=False):
        if fixed_start is None:
            raise ValueError("start_pose requires the certified exact start of a screened site")
        if allow_enclosed:
            raise ValueError("enclosed exact starts are not supported")
        frame_ok = self._pose_once(
            tx, tz, dist, setting, jitter,
            require_water_free=require_water_free, allow_night=allow_night,
            fixed_start=fixed_start, fixed_yaw=fixed_yaw)
        verified9 = bool(frame_ok and self.stand_ok(
            min_bright=20.0 if allow_night else 25.0,
            forward_only=require_forward_clear))
        px9, py9, pz9 = self.w.get_pos()
        self.last_pose_spec = {
            "x": round(float(px9), 3), "y": round(float(py9), 3),
            "z": round(float(pz9), 3),
            "yaw": round(float(self.w.get_yaw()), 3),
            "pitch": round(float(self.w.get_pitch()), 3),
            "requested_jitter": round(float(jitter), 6),
            "candidate_offset": 0, "verified": verified9,
            "fixed_start": True,
            "fixed_yaw": None if fixed_yaw is None else round(float(fixed_yaw), 3),
        }
        status9 = "accepted" if verified9 else "REJECTED"
        print(f"[pose-fixed] {status9} spec={self.last_pose_spec}", flush=True)
        return verified9

    def _pose_once(self, tx, tz, dist, setting, jitter,
                   require_water_free=False, allow_night=False,
                   fixed_start=None, fixed_yaw=None):
        w, settle_feet, go = self.w, self.settle_feet, self.go
        b = jitter
        sx, expected_y9, sz = (float(v9) for v9 in fixed_start)
        sy = settle_feet(sx, sz)
        if abs(float(sy) - expected_y9) > 0.38:
            print(f"[pose-fixed] floor changed expected={expected_y9:.2f} "
                  f"observed={float(sy):.2f} at ({sx:.2f},{sz:.2f})", flush=True)
            return False
        if not go(sx, sy, sz, b, 8, n=10):
            return False
        ax, ay, az = w.get_pos()
        yb = math.degrees(math.atan2(-(tx + 0.5 - ax), (tz + 0.5 - az)))
        if setting == "vis":
            yaw0 = yb
        else:
            if fixed_yaw is None:
                raise ValueError("an invis exact start requires its screened start yaw")
            fixed_yaw9 = float(fixed_yaw)
            min_bright9 = 20 if allow_night else 25
            if require_water_free or not self.fixed_start_yaw_ok(
                    sx, sy, sz, fixed_yaw9, min_bright=min_bright9):
                print(f"[pose-fixed] stored yaw contract changed yaw={fixed_yaw9:.1f}",
                      flush=True)
                return False
            bx9, bz9 = int(math.floor(sx)), int(math.floor(sz))
            by9 = int(math.floor(sy))
            for gname9 in ("grass", "tall_grass"):
                w.cmd(f"/fill {bx9-1} {by9} {bz9-1} {bx9+1} {by9+1} {bz9+1} "
                      f"minecraft:air replace minecraft:{gname9}")
            return self._teleport_stable(sx, sy, sz, fixed_yaw9, 8)
        bx9, bz9 = int(math.floor(ax)), int(math.floor(az))
        by9 = int(math.floor(ay))
        for gname in ("grass", "tall_grass"):
            w.cmd(f"/fill {bx9-1} {by9} {bz9-1} {bx9+1} {by9+1} {bz9+1} "
                  f"minecraft:air replace minecraft:{gname}")
        return self._teleport_stable(ax, ay, az, yaw0, 8)

    def reset_world(self):
        w, PLACED_UNDO, SITES = self.w, self.PLACED_UNDO, self.SITES
        place_cell_y = self.place_cell_y
        w.cmd("/kill @e[type=!player,distance=..96]")
        w.cmd("/kill @e[type=item]")
        w.cmd("/difficulty peaceful")
        w.cmd("/time set 6000")
        for c in PLACED_UNDO:
            w.cmd(c)
        del PLACED_UNDO[:]
        for (sx0, sz0) in sorted(set(SITES)):
            sy0 = place_cell_y(sx0, sz0)
            for y0 in range(int(sy0) - 5, int(sy0) + 8, 4):
                for liq in ("fire", "obsidian"):
                    w.cmd(f"/fill {int(sx0)-20} {y0} {int(sz0)-20} {int(sx0)+20} {y0+3} {int(sz0)+20} "
                          f"minecraft:air replace minecraft:{liq}")
        SITES.clear()

    def screen_free_site(self, cell, rng, setting="invis", attempts=10,
                         mine_target_kind=None, mine_confuser_kind=None):
        if setting not in {"vis", "invis"}:
            raise ValueError(f"unknown free-site setting: {setting}")
        if cell != "mine":
            raise ValueError(
                f"free-site screening supports only the mine cell, got {cell!r}")
        mine_worldgen_profile9 = str(getattr(
            self, "mine_worldgen_profile", "current"))
        if mine_worldgen_profile9 != V12_INDEPENDENT_MULTICLASS6_PROFILE:
            raise ValueError(
                f"unsupported mine worldgen profile {mine_worldgen_profile9!r}")
        if mine_target_kind is None:
            raise ValueError("mine screening requires mine_target_kind")
        mine_class_pool9 = tuple(OC.validate_mine_pool(
            getattr(self, "mine_class_pool", OC.MINE_TARGET_POOL)))
        mine_target_kind9 = OC.bare_block(mine_target_kind)
        if mine_target_kind9 not in mine_class_pool9:
            raise ValueError(
                f"mine target {mine_target_kind9!r} is not in "
                f"the active mine roster {mine_class_pool9}")
        mine_source_policy9 = OC.episode_source_policy(mine_target_kind9)
        mine_target_family9 = str(
            OC.harvest_rule(mine_target_kind9)["target_family"])
        if not OC.episode_target_generation_enabled(mine_target_kind9):
            print(
                f"[freesite-mine] target={mine_target_kind9} blocked by episode "
                f"source policy {mine_source_policy9}: no truthful route",
                flush=True)
            return None
        if (mine_target_family9 != "ore"
                or mine_source_policy9 != OC.EPISODE_BIOME_PLACEMENT_SOURCE):
            raise ValueError(
                "mine screening supports biome-placed ore targets only, got "
                f"{mine_target_kind9!r}/{mine_target_family9!r}")
        mine_confuser_kind9 = None
        if mine_confuser_kind is not None:
            mine_confuser_kind9 = OC.bare_block(mine_confuser_kind)
            if mine_confuser_kind9 not in mine_class_pool9:
                raise ValueError(
                    f"mine confuser {mine_confuser_kind9!r} is not in "
                    f"the active mine roster {mine_class_pool9}")
            mine_confuser_family9 = str(
                OC.harvest_rule(mine_confuser_kind9)["target_family"])
            if mine_confuser_family9 != mine_target_family9:
                raise ValueError(
                    "mine target/confuser must share a biome placement family: "
                    f"{mine_target_kind9}/{mine_target_family9} versus "
                    f"{mine_confuser_kind9}/{mine_confuser_family9}")
        mine_chain_count9 = int(getattr(self, "mine_chain_count", 3))
        if mine_chain_count9 != 3:
            raise ValueError(
                "independent multiclass6 screening stages exactly three target cells")
        w = self.w
        ox0, _, oz0 = self.world_origin
        mine_rejects9 = Counter()
        for att9 in range(int(attempts)):
            attempt_started9 = time.monotonic()
            bearing9 = float(rng.uniform(0, 360))
            radius9 = float(rng.uniform(50, 130))
            cx9 = math.floor(ox0 - radius9 * math.sin(math.radians(bearing9))) + 0.5
            cz9 = math.floor(oz0 + radius9 * math.cos(math.radians(bearing9))) + 0.5
            candidate_load_started9 = time.monotonic()
            w.cmd(f"/tp @a {cx9:.1f} 150 {cz9:.1f} 0 30")
            for _ in range(45):
                w.step_noop()
            try:
                sy9 = self.settle_feet(cx9, cz9)
                ys9 = [sy9]
            except RuntimeError as exc9:
                print(f"[freesite] {cell} reject unstable ground at "
                      f"({cx9:.1f},{cz9:.1f}): {exc9} "
                      f"candidate_load_wall_s="
                      f"{time.monotonic() - candidate_load_started9:.3f} "
                      f"attempt_wall_s={time.monotonic() - attempt_started9:.3f}",
                      flush=True)
                mine_rejects9["unstable_ground"] += 1
                continue

            try:
                if not self.go(cx9, sy9, cz9, 0, 8, n=8):
                    mine_rejects9["site_teleport_unacknowledged"] += 1
                    continue
                rel9 = self.converged_relative_voxels(
                    (-24, 25, -6, 8, -24, 25))
            except RuntimeError as exc9:
                print(f"[freesite-mine] attempt={att9 + 1} reject="
                      f"unstable_local_voxels error={exc9}", flush=True)
                mine_rejects9["unstable_local_voxels"] += 1
                continue
            under9 = rel9.get((0, -1, 0))
            feet9, head9 = rel9.get((0, 0, 0)), rel9.get((0, 1, 0))
            unsafe_start9 = (not self._mine_solid(under9)
                    or any(v9 is not None and self._mine_solid(v9) for v9 in (feet9, head9))
                    or any("water" in str(v9) for v9 in (under9, feet9, head9))
                    or any(q9 in str(under9) for q9 in ("leaves", "log")))
            start_relocation9 = None
            if unsafe_start9:
                origin9 = (float(cx9), float(sy9), float(cz9))
                origin_floor9 = tuple(int(math.floor(v9)) for v9 in origin9)
                for dx9, feet_dy9, dz9 in self.nearby_safe_mine_start_offsets(
                        rel9, radius=4, max_dy=2)[:8]:
                    nx9 = origin_floor9[0] + dx9 + 0.5
                    ny9 = origin_floor9[1] + feet_dy9
                    nz9 = origin_floor9[2] + dz9 + 0.5
                    if not self.go(nx9, ny9, nz9, 0, 8, n=8):
                        continue
                    try:
                        candidate_rel9 = self.converged_relative_voxels(
                            (-2, 3, -3, 4, -2, 3))
                    except RuntimeError:
                        continue
                    candidate_under9 = candidate_rel9.get((0, -1, 0))
                    candidate_feet9 = candidate_rel9.get((0, 0, 0))
                    candidate_head9 = candidate_rel9.get((0, 1, 0))
                    if (not self._mine_walk_support(candidate_under9)
                            or self._mine_walk_blocked(candidate_feet9)
                            or self._mine_walk_blocked(candidate_head9)):
                        continue
                    live_x9, live_y9, live_z9 = w.get_pos()
                    cx9, sy9, cz9 = float(live_x9), float(live_y9), float(live_z9)
                    ys9 = [sy9]
                    rel9 = candidate_rel9
                    under9, feet9, head9 = (
                        candidate_under9, candidate_feet9, candidate_head9)
                    start_relocation9 = {
                        "from": [round(v9, 3) for v9 in origin9],
                        "to": [round(cx9, 3), round(sy9, 3), round(cz9, 3)],
                        "offset": [int(dx9), int(feet_dy9), int(dz9)],
                    }
                    unsafe_start9 = False
                    print(f"[freesite-mine] relocated unsafe random anchor "
                          f"{start_relocation9}", flush=True)
                    break
            if unsafe_start9:
                mine_rejects9["unsafe_start_column"] += 1
                continue
            if all(self._mine_solid(rel9.get((0, dy9, 0)))
                   for dy9 in range(2, 7)):
                mine_rejects9["enclosed_start"] += 1
                continue
            if float(np.asarray(w.info["pov"]).mean()) < 25.0:
                mine_rejects9["dark_start"] += 1
                continue

            waters9 = [(int(b9["x"]), int(b9["y"]), int(b9["z"]))
                       for b9 in (w.info.get("voxels") or [])
                       if "water" in str(b9.get("type", ""))]
            near9 = [math.hypot(wx9, wz9) for wx9, _, wz9 in waters9]
            if mine_confuser_kind9 is None:
                mine_rejects9["missing_counterfactual_pair_class"] += 1
                continue
            mine_grid9, biome_pair_plan9 = self.biome_plausible_pair_slots(
                mine_target_kind9, mine_confuser_kind9,
                center=(cx9, sy9, cz9), span=MINE_SITE_SCAN_SPAN)
            slot_diag9 = dict(self._last_mine_slot_diagnostics or {})
            if biome_pair_plan9 is None:
                mine_rejects9["no_checkpoint_perceptible_slot"] += 1
                print(f"[freesite-mine] attempt={att9 + 1} reject=no_slot "
                      f"slot_gates={json.dumps(slot_diag9, sort_keys=True)}",
                      flush=True)
                continue
            biome_terrain_topology9 = copy.deepcopy(
                biome_pair_plan9.get("biome_terrain_topology") or {})
            mine_survey9 = dict(self._last_mine_survey_contract or {})
            if not mine_survey9.get("route") or not mine_survey9.get("checkpoints"):
                mine_rejects9["missing_survey"] += 1
                print(f"[freesite-mine] attempt={att9 + 1} reject=missing_survey "
                      f"slot_gates={json.dumps(slot_diag9, sort_keys=True)}", flush=True)
                continue

            yaw_base9 = float(abs(int(math.floor(cx9)) * 73
                                  + int(math.floor(cz9)) * 151) % 360)
            safe_yaws9 = []
            for yaw_i9 in range(12):
                yaw9 = ((yaw_base9 + 30.0 * yaw_i9 + 180.0)
                        % 360.0 - 180.0)
                if self.fixed_start_yaw_ok(
                        cx9, sy9, cz9, yaw9, min_bright=25.0):
                    safe_yaws9.append(float(yaw9))
            if not safe_yaws9:
                mine_rejects9["no_safe_start_yaw"] += 1
                print(f"[freesite-mine] attempt={att9 + 1} reject=no_safe_yaw "
                      f"slot_gates={json.dumps(slot_diag9, sort_keys=True)}",
                      flush=True)
                continue

            target_lane9 = int(
                biome_pair_plan9["class_to_lane"][mine_target_kind9])
            target_lane_record9 = next(
                q9 for q9 in biome_pair_plan9["lanes"]
                if int(q9["lane_id"]) == target_lane9)
            choices9 = [
                ((
                    (int(spec9["survey_step"]),),
                    tuple(spec9["slot"]),
                    dict(spec9["geometry"]),
                ), str(spec9["source"]), str(spec9["original"]))
                for spec9 in target_lane_record9["specs"]]
            specs9 = [dict(slot=tuple(entry9[1]), source=source9,
                           original=original9,
                           survey_step=int(entry9[2]["survey_checkpoint_step"]),
                           survey_view_pos=list(entry9[2]["survey_view_pos"]),
                           primary_eligible=bool(
                               entry9[2]["primary_pocket_ok"]),
                           natural_instance_cells=list(
                               entry9[2].get("natural_instance_cells") or ()),
                           natural_camera_stance_witness=dict(
                               entry9[2].get("natural_camera_stance_witness") or {}),
                           terrain_topology=copy.deepcopy(
                               entry9[2].get("terrain_topology") or {}),
                           geometry={"terrain_topology": copy.deepcopy(
                               entry9[2].get("terrain_topology") or {})})
                      for entry9, source9, original9 in choices9]
            for spec9, source_entry9 in zip(specs9, target_lane_record9["specs"]):
                spec9["pair_lane_id"] = int(source_entry9["pair_lane_id"])
                spec9["kind"] = str(source_entry9["kind"])
                spec9["geometry"] = dict(source_entry9["geometry"])
            specs9.sort(key=lambda q9: (q9["survey_step"], q9["slot"]))
            route_supports9 = {
                (int(q9[0]), int(q9[1]) - 1, int(q9[2]))
                for q9 in mine_survey9["route"]
            }
            specs9 = [spec9 for spec9 in specs9
                      if spec9["slot"] not in route_supports9]
            pre_target_cone_specs9 = len(specs9)
            hidden_census_by_yaw9 = {}
            if setting == "invis":
                stance0_9 = (int(math.floor(cx9)), int(math.floor(sy9)),
                             int(math.floor(cz9)))
                hidden_yaws9 = []
                f0_grid9 = self._last_biome_pair_planned_grid
                f0_classes9 = biome_pair_plan9["canonical_classes"]
                for yaw9 in safe_yaws9:
                    census9 = self.exhaustive_pair_class_pose_census(
                        f0_grid9,
                        self._last_mine_scan_bounds,
                        f0_classes9,
                        stance0_9, yaw9)
                    if not any(row9["recognizable"] for row9 in census9.values()):
                        hidden_yaws9.append(yaw9)
                        hidden_census_by_yaw9[float(yaw9)] = census9
                safe_yaws9 = hidden_yaws9
                if not safe_yaws9:
                    mine_rejects9["target_class_visible_at_invis_f0"] += 1
                    continue
            mine_start_yaw9 = float(safe_yaws9[0])

            if hidden_census_by_yaw9:
                biome_pair_plan9["local_f0_class_census"] = hidden_census_by_yaw9[
                    mine_start_yaw9]
            post_target_cone_specs9 = len(specs9)
            filtered_specs9 = len(specs9)
            primary_specs9 = sum(bool(q9.get("primary_eligible")) for q9 in specs9)
            stance_occ9 = (
                self._last_mine_occupancy
                if self._last_mine_occupancy is not None
                else self.mine_occupancy(
                    mine_grid9, self._last_mine_scan_bounds))
            mine_chain9, _stance_proofs9, unstanceable9 = (
                select_stanceable_mine_chain(
                    specs9, stance_occ9, count=mine_chain_count9))
            if mine_chain9 is None:
                if unstanceable9:
                    mine_rejects9["no_stanceable_chain"] += 1
                    print(f"[freesite-mine] attempt={att9 + 1} "
                          f"reject=no_stanceable_chain "
                          f"unstanceable={sorted(set(unstanceable9))} "
                          f"slots={len(choices9)} filtered={filtered_specs9} "
                          f"start_yaw={mine_start_yaw9:.3f} "
                          f"cone_counts={json.dumps({'pre': pre_target_cone_specs9, 'post': post_target_cone_specs9}, sort_keys=True)} "
                          f"slot_gates={json.dumps(slot_diag9, sort_keys=True)}",
                          flush=True)
                else:
                    mine_rejects9[
                        f"no_ordered_{mine_chain_count9}_chain"] += 1
                    print(f"[freesite-mine] attempt={att9 + 1} reject=no_chain "
                          f"slots={len(choices9)} filtered={filtered_specs9} "
                          f"primary={primary_specs9} "
                          f"steps={[q9['survey_step'] for q9 in specs9]} "
                          f"start_yaw={mine_start_yaw9:.3f} "
                          f"cone_counts={json.dumps({'pre': pre_target_cone_specs9, 'post': post_target_cone_specs9}, sort_keys=True)} "
                          f"slot_gates={json.dumps(slot_diag9, sort_keys=True)}",
                          flush=True)
                continue
            if unstanceable9:
                print(f"[freesite-mine] attempt={att9 + 1} "
                      f"replaced_unstanceable={sorted(set(unstanceable9))} "
                      f"chain={[list(q9['slot']) for q9 in mine_chain9]}",
                      flush=True)
            mine_slot9 = (mine_chain9[0]["slot"], mine_chain9[0]["source"])

            other_lane9 = int(
                biome_pair_plan9["class_to_lane"][mine_confuser_kind9])
            other_record9 = next(
                q9 for q9 in biome_pair_plan9["lanes"]
                if int(q9["lane_id"]) == other_lane9)
            shared9 = dict(biome_pair_plan9["shared_decision_checkpoint"])
            pair_index9 = list(biome_pair_plan9["canonical_classes"]).index(
                mine_confuser_kind9)
            confuser9 = {
                "required": True, "satisfied": True, "verified": False,
                "reason": "counterfactual_pair_shared_checkpoint",
                "tier": 1, "require_nearer": False,
                "confuser_class": mine_confuser_kind9,
                "slots": [list(q9) for q9 in other_record9["cells"]],
                "certified_slot": list(shared9["lane_cells"][pair_index9]),
                "commit_view_pos": list(shared9["view_pos"]),
                "commit_yaw": float(shared9["yaw"]),
                "commit_pose_source": "shared_survey_checkpoint",
                "commit_step": int(shared9["step"]),
                "perceptibility": dict(
                    shared9["lane_perceptibility"][pair_index9]),
                "counterfactual_pair": True,
                "target_horizontal": None,
                "confuser_horizontal": None,
                "lateral_offset": None,
            }
            print(f"[freesite-mine] attempt={att9 + 1} confuser="
                  f"{confuser9['slots']} pose={confuser9['commit_pose_source']}"
                  f"@{confuser9['commit_step']} "
                  f"target_d={confuser9['target_horizontal']} "
                  f"confuser_d={confuser9['confuser_horizontal']} "
                  f"lateral={confuser9['lateral_offset']}", flush=True)

            ax9, ay9, az9 = w.get_pos()
            absolute_waters9 = [(int(math.floor(ax9)) + wx9,
                                 int(math.floor(ay9)) + wy9,
                                 int(math.floor(az9)) + wz9)
                                for wx9, wy9, wz9 in waters9]
            site9 = dict(x=float(cx9), y=float(sy9), z=float(cz9),
                         attempt=att9 + 1, flatness=float(max(ys9) - min(ys9)),
                         waters=absolute_waters9,
                         near_water_min_distance=(
                             None if not near9 else float(min(near9))),
                         near_water_within_16=bool(
                             any(float(d9) <= 16.0 for d9 in near9)))
            if start_relocation9 is not None:
                site9["mine_start_relocation"] = dict(start_relocation9)
            site9["mine_slot"], site9["mine_source"] = mine_slot9
            site9["mine_target_kind_screened"] = mine_target_kind9
            site9["mine_worldgen_profile"] = mine_worldgen_profile9
            site9["mine_stage_policy"] = mine_source_policy9
            site9["mine_chain_slots"] = [
                {"slot": list(spec9["slot"]), "source": spec9["source"],
                 "original": str(spec9["original"]),
                 "survey_step": int(spec9["survey_step"]),
                 "survey_view_pos": list(spec9["survey_view_pos"]),
                 "pair_lane_id": spec9.get("pair_lane_id"),
                 "kind": spec9.get("kind"),
                 "geometry": dict(spec9.get("geometry") or {}),
                 "natural_instance_cells": list(
                     spec9.get("natural_instance_cells") or ()),
                 "natural_camera_stance_witness": dict(
                     spec9.get("natural_camera_stance_witness") or {})}
                for spec9 in mine_chain9]
            site9["mine_chain_count"] = int(mine_chain_count9)
            site9["mine_human_chain_count"] = int(mine_chain_count9)
            site9["mine_start_yaw"] = float(mine_start_yaw9)
            visible_pose9 = survey_checkpoint_stance(
                mine_survey9, int(mine_chain9[0]["survey_step"]))
            if visible_pose9 is None:
                raise RuntimeError(
                    "accepted mine chain lost its certified visible checkpoint")
            visible_stance9, visible_yaw9 = visible_pose9
            site9["mine_visible_start"] = {
                "source": "first_certified_survey_visible_checkpoint",
                "survey_step": int(mine_chain9[0]["survey_step"]),
                "position": [
                    float(visible_stance9[0]) + 0.5,
                    float(visible_stance9[1]),
                    float(visible_stance9[2]) + 0.5],
                "proof_yaw": round(float(visible_yaw9), 3),
                "target_slot": [int(v9) for v9 in mine_chain9[0]["slot"]],
            }
            site9["mine_survey_route"] = mine_survey9["route"]
            site9["mine_survey_checkpoints"] = mine_survey9["checkpoint_steps"]
            site9["mine_scan_bounds"] = [
                int(v9) for v9 in (self._last_mine_scan_bounds or ())]
            site9["mine_confuser"] = confuser9
            biome_pair_plan9["frozen_start"] = [
                float(cx9), float(sy9), float(cz9), float(mine_start_yaw9)]
            site9["mine_counterfactual_pair"] = biome_pair_plan9
            if biome_terrain_topology9:
                site9["biome_terrain_topology"] = copy.deepcopy(
                    biome_terrain_topology9)
            print(f"[freesite] {cell} site=({cx9:.0f},{cz9:.0f}) y={sy9:.0f} "
                  f"attempt={att9 + 1} flatness={site9['flatness']:.1f}", flush=True)
            return site9
        print(f"[freesite-mine] exhausted={int(attempts)} "
              f"rejects={_format_reject_census(mine_rejects9)}",
              flush=True)
        return None

    def stage(self, cell, setting, free_site=False, ore_class=None,
              confuser_class=None, confuser_classes=None, census_scrub=False):
        w, PLACED_UNDO, kit = self.w, self.PLACED_UNDO, self.kit
        free = free_site if isinstance(free_site, dict) else None
        if cell != "mine" or free is None:
            raise ValueError(
                "staging supports only the mine cell with a screened free site")
        self.reset_world()

        mine_class_pool9 = tuple(OC.validate_mine_pool(
            getattr(self, "mine_class_pool", OC.MINE_TARGET_POOL)))
        ore_class9 = mine_class_pool9[0] if ore_class is None else OC.bare_block(ore_class)
        if ore_class9 not in mine_class_pool9:
            raise ValueError(
                f"mine target class {ore_class9!r} is not in active roster "
                f"{mine_class_pool9}")
        mine_worldgen_profile9 = str(free.get(
            "mine_worldgen_profile",
            getattr(self, "mine_worldgen_profile", "current")))
        if mine_worldgen_profile9 != V12_INDEPENDENT_MULTICLASS6_PROFILE:
            raise ValueError(
                f"unsupported mine worldgen profile {mine_worldgen_profile9!r}")
        distinct_confuser_classes9 = tuple(
            OC.bare_block(q9) for q9 in (confuser_classes or ()))
        if (getattr(self, "plan_mode", "pair") != "solo"
                or len(distinct_confuser_classes9) != 3
                or len(set(distinct_confuser_classes9)) != 3
                or ore_class9 in distinct_confuser_classes9):
            raise ValueError(
                "independent multiclass6 staging requires solo target x3 "
                "plus three distinct non-goal classes")
        target_harvest9 = OC.harvest_rule(ore_class9)
        episode_source_policy9 = OC.episode_source_policy(ore_class9)
        if not OC.episode_target_generation_enabled(ore_class9):
            raise RuntimeError(
                f"mine target {ore_class9!r} has no truthful episode source route: "
                f"{episode_source_policy9}")
        if (target_harvest9["target_family"] != "ore"
                or episode_source_policy9 != OC.EPISODE_BIOME_PLACEMENT_SOURCE):
            raise ValueError(
                "mine staging supports biome-placed ore targets only, got "
                f"{ore_class9!r}")
        requested_chain_count9 = int(free.get(
            "mine_chain_count", free.get("mine_human_chain_count", 3)))
        kit(f"{OC.MINE_KIT_ITEM} 1")
        if not self.go(free["x"], free["y"], free["z"], 0, 8, n=4):
            raise RuntimeError("screened mine site teleport was not acknowledged")
        screened_kind9 = free.get("mine_target_kind_screened")
        if screened_kind9 is None or OC.bare_block(screened_kind9) != ore_class9:
            raise RuntimeError(
                "mine target changed between site screening and staging: "
                f"screened={screened_kind9!r} staged={ore_class9!r}")
        screened_policy9 = free.get("mine_stage_policy")
        if (screened_policy9 is not None
                and str(screened_policy9) != episode_source_policy9):
            raise RuntimeError(
                "mine source policy changed between screening and staging: "
                f"screened={screened_policy9!r} staged={episode_source_policy9!r}")
        chain_specs9 = list(free.get("mine_chain_slots") or ())
        if len(chain_specs9) < requested_chain_count9:
            raise RuntimeError(
                "screened mine site lost its requested ore quota chain: "
                f"{len(chain_specs9)}/{requested_chain_count9}")
        pair_record9 = dict(free.get("mine_counterfactual_pair") or {})
        pair_specs9 = [dict(spec9)
                       for lane9 in pair_record9.get("lanes", ())
                       for spec9 in lane9.get("specs", ())]
        if (pair_record9.get("contract") != getattr(
                OC, "COUNTERFACTUAL_PAIRED_SCENE_CONTRACT",
                "same_physical_scene_goal_role_only/v1")
                or len(pair_specs9) != 6
                or len(pair_record9.get("lanes", ())) != 2):
            raise RuntimeError(
                "screened biome pair lost its two fixed three-cell lanes")
        assigned9 = dict(pair_record9.get("class_to_lane") or {})
        if assigned9.get(ore_class9) is None:
            raise RuntimeError(
                f"screened biome pair has no lane for target {ore_class9!r}")
        if requested_chain_count9 != 3:
            raise RuntimeError(
                "independent multiclass6 lost its screened 3+3 lanes")
        mine_scene_layout9 = self.v12_independent_multiclass6_layout(
            pair_record9, ore_class9, distinct_confuser_classes9,
            free.get("layout_seed", 0),
            roster=mine_class_pool9)
        chain_specs9 = [dict(q9) for q9 in
                        mine_scene_layout9["target_specs"]]
        placement_specs9 = [dict(q9) for q9 in
                            mine_scene_layout9["all_specs"]]
        census_scrub_witness9 = None
        census_scrub_cells9 = ()
        if census_scrub:
            sources9 = {str(s9.get("source")) for s9 in placement_specs9}
            if sources9 & {
                    "natural", "natural_exact_class",
                    OC.NATURAL_EXACT_EXPOSED_BY_STONE_CARVE_SOURCE}:
                raise RuntimeError(
                    "census scrub refused: natural-provenance mine "
                    f"slots present (sources={sorted(sources9)})")
            census_scrub_cells9 = tuple(
                tuple(int(v9) for v9 in s9["slot"])
                for s9 in placement_specs9)
            census_scrub_witness9 = self.census_scrub_natural_ores(
                free, staged_cells=census_scrub_cells9)
            print(f"[census-scrub] region={census_scrub_witness9['region_box']} "
                  f"fills={census_scrub_witness9['fill_command_count']} "
                  f"pre_counts={census_scrub_witness9['pre_scrub_central_counts']} "
                  f"wall={census_scrub_witness9['wall_seconds']}", flush=True)
        for spec9 in placement_specs9:
            slot9 = tuple(int(v9) for v9 in spec9["slot"])
            source9 = str(spec9["source"])
            original9 = str(spec9.get("original") or "stone")
            placed_kind9 = OC.bare_block(spec9.get("kind"))
            OC.episode_staging_provenance(
                placed_kind9, source9, original9)
            if source9 != OC.EPISODE_BIOME_PLACEMENT_SOURCE:
                raise RuntimeError(
                    f"biome placement target {placed_kind9!r} received "
                    f"source={source9!r}")
            w.cmd(
                f"/setblock {slot9[0]} {slot9[1]} {slot9[2]} "
                f"minecraft:{placed_kind9}")
            PLACED_UNDO.append(
                f"/setblock {slot9[0]} {slot9[1]} {slot9[2]} "
                f"minecraft:{original9}")
        confuser_plan9 = dict(free.get("mine_confuser") or
                              {"required": False, "satisfied": False,
                               "reason": "not_screened"})
        confuser_class9 = (OC.bare_block(confuser_class) if confuser_class
                           else next(c9 for c9 in mine_class_pool9
                                     if c9 != ore_class9))
        if confuser_class9 not in mine_class_pool9:
            raise ValueError(
                f"mine confuser class {confuser_class9!r} is not in "
                f"the active mine roster {mine_class_pool9}")
        if confuser_class9 == ore_class9:
            raise ValueError(
                f"confuser class {confuser_class9!r} must differ from the target "
                "class: a same-class cluster is another instance of the goal, not "
                "a discrimination decision")
        placed_slots9 = [
            (tuple(int(v9) for v9 in s9["slot"]), OC.bare_block(s9.get("kind")))
            for s9 in placement_specs9]
        grid9 = None
        for _settle9 in range(5):
            for _ in range(3):
                w.step_noop()
            grid9 = self.scan_mine_terrain(span=MINE_SITE_SCAN_SPAN)
            if all(str(grid9.get(sl9, "")).split(":")[-1] == k9
                   for sl9, k9 in placed_slots9):
                break
        screened_bounds9 = [int(v9) for v9 in free.get("mine_scan_bounds", ())]
        verified_bounds9 = [
            int(v9) for v9 in (self._last_mine_scan_bounds or ())]
        if screened_bounds9 and verified_bounds9 != screened_bounds9:
            raise RuntimeError(
                "screened mine scan envelope changed after placement: "
                f"screened={screened_bounds9} verified={verified_bounds9}")
        if census_scrub_witness9 is not None:
            self._census_scrub_central_audit(
                census_scrub_witness9, grid9, census_scrub_cells9)
        reachable_states9 = self._mine_reachable_surface(
            grid9, (free["x"], free["y"], free["z"]))
        route9 = [list(q9) for q9 in free.get("mine_survey_route", ())]
        checkpoints9 = [
            int(v9) for v9 in free.get("mine_survey_checkpoints", ())]
        frozen_survey9 = self._validate_frozen_mine_survey(
            grid9, route9, checkpoints9)
        missing_route_states9 = sorted(
            set(map(tuple, route9)) - set(reachable_states9))
        if missing_route_states9:
            raise RuntimeError(
                "screened mine survey disconnected after placement: "
                f"{missing_route_states9[:3]}")
        profile9 = self._mine_primary_pocket_profile(
            (free["x"], free["y"], free["z"]), reachable_states9)
        if profile9 is None:
            raise RuntimeError("screened mine survey component disappeared")
        profile9["survey"] = frozen_survey9
        reachable9 = defaultdict(list)
        for rx9, rfy9, rz9 in reachable_states9:
            reachable9[(rx9, rz9)].append(rfy9)
        route9 = frozen_survey9["route"]
        checkpoints9 = frozen_survey9["checkpoint_steps"]
        occ9 = G.OccupancyMap(leaves_occlude=True)
        occ9.grid = dict(grid9)
        occ9.boxes = [tuple(int(v9) for v9 in self._last_mine_scan_bounds)]
        route_supports9 = {
            (int(q9[0]), int(q9[1]) - 1, int(q9[2])) for q9 in route9
        }
        ores = []
        for spec_i9, spec9 in enumerate(chain_specs9):
            slot9 = tuple(int(v9) for v9 in spec9["slot"])
            source9 = str(spec9["source"])
            original9 = str(spec9.get("original") or "stone")
            provenance9 = OC.episode_staging_provenance(
                ore_class9, source9, original9)
            got9 = str(grid9.get(slot9, "")).split(":")[-1]
            if not self._mine_scan_contains(slot9, margin=1):
                raise RuntimeError(f"screened mine slot lies at scan boundary: {slot9}")
            geom9 = self._ore_geometry(grid9, slot9)
            screened_geometry9 = dict(
                spec9.get("geometry") or {})
            screened_placement_geometry9 = str(
                screened_geometry9.get("placement_geometry")
                or ORE_SIDE_PLACEMENT_GEOMETRY)
            measured_placement_geometry9 = (
                ore_replacement_placement_geometry(geom9))
            geom9.update(
                placement_geometry=measured_placement_geometry9,
                screened_placement_geometry=(
                    screened_placement_geometry9),
                placement_geometry_match=bool(
                    measured_placement_geometry9
                    == screened_placement_geometry9))
            geom9.update(
                pair_lane_id=int(spec9["pair_lane_id"]),
                pair_class_assignment_fixed=True,
                world_mutation_commands=[
                    f"/setblock {slot9[0]} {slot9[1]} {slot9[2]} "
                    f"minecraft:{ore_class9}"],
            )
            geom9.update(self._mine_access_audit(
                grid9, slot9, reachable9))
            geom9.update(self._mine_primary_pocket_audit(
                (free["x"], free["y"], free["z"]), slot9,
                reachable_states9, geom9.get("access_pos"), profile=profile9))
            discovery9 = first_survey_checkpoint_perceptibility(
                profile9["survey"], slot9, occ9,
                max_step=MINE_SURVEY_ROUTE_STEPS,
                max_dist=MINE_CHAIN_RADIUS)
            if discovery9 is not None:
                geom9.update(discovery9)
            proof9 = (None if discovery9 is None else
                      dict(discovery9["survey_perceptibility"]))
            geom9["start_perceptibility"] = proof9
            if (got9 != ore_class9 or not geom9["ok"] or not geom9["accessible"]
                    or (spec_i9 == 0 and not geom9["primary_pocket_ok"])
                    or proof9 is None
                    or not geom9.get("placement_geometry_match", True)
                    or slot9 in route_supports9
                    or int(geom9["survey_checkpoint_step"]) != int(spec9["survey_step"])
                    or list(geom9["survey_view_pos"])
                       != list(spec9["survey_view_pos"])):
                raise RuntimeError(
                    f"screened mine chain slot changed: {slot9} got={got9} geom={geom9}")
            if mine_slot_reachable_stance(occ9, spec9) is None:
                raise RuntimeError(
                    "screened mine chain slot lost its reachable mine "
                    f"stance after placement: {slot9}")
            self.SITES.add((slot9[0], slot9[2]))
            ores.append(dict(kind=ore_class9, x=slot9[0], y=slot9[1], z=slot9[2],
                             source=source9, original=original9,
                             harvest_mode=target_harvest9["harvest_mode"],
                             expected_drop_items=list(
                                 target_harvest9["drop_items"]),
                             **provenance9, **geom9))
            print(f"[mine-stage] {ore_class9} {slot9} source={source9} "
                  f"contacts={geom9['contacts']} exposed={geom9['exposed']} "
                  f"side_exposed={geom9['side_exposed']} support={geom9['support']} "
                  f"access={geom9['access_pos']} d={geom9['access_dist']} "
                  f"start_d={geom9['start_target_horizontal']} "
                  f"path={geom9['start_access_path_steps']} "
                  f"survey_step={geom9['survey_checkpoint_step']} "
                  f"survey_heading_error={geom9['survey_heading_error']} "
                  f"pixels={proof9['visible_area_px']}/{proof9['visible_short_side_px']}",
                  flush=True)
        chain9 = [o for o in ores if o["kind"] == ore_class9]
        if len(chain9) < requested_chain_count9:
            raise RuntimeError(
                "screened mine site lost its verified ore chain: "
                f"{len(chain9)}/{requested_chain_count9}")
        confuser_record9 = dict(confuser_plan9)
        pair_stance9 = tuple(int(v9) for v9 in
                             confuser_record9["commit_view_pos"])
        pair_yaw9 = float(confuser_record9["commit_yaw"])
        target_pair_lane9 = int(
            mine_scene_layout9["target_lane_id"])
        distractor_pair_lane9 = next(
            int(q9["lane_id"]) for q9 in pair_record9["lanes"]
            if int(q9["lane_id"]) != target_pair_lane9)
        lane_cells9 = pair_record9[
            "shared_decision_checkpoint"]["lane_cells"]
        target_pair_cell9 = tuple(
            int(v9) for v9 in lane_cells9[target_pair_lane9])
        confuser_pair_cell9 = tuple(
            int(v9) for v9 in lane_cells9[distractor_pair_lane9])
        target_pair_proof9 = stance_perceptibility(
            pair_stance9, pair_yaw9, target_pair_cell9, occ9)
        confuser_pair_proof9 = stance_perceptibility(
            pair_stance9, pair_yaw9, confuser_pair_cell9, occ9)
        if target_pair_proof9 is None or confuser_pair_proof9 is None:
            raise RuntimeError(
                "biome pair lost target/confuser co-visibility after placement")
        confuser_record9.update(
            verified=True, target_slot=list(target_pair_cell9),
            target_horizontal=target_pair_proof9["survey_target_horizontal"],
            confuser_horizontal=confuser_pair_proof9["survey_target_horizontal"],
            distance_gap=round(
                float(target_pair_proof9["survey_target_horizontal"])
                - float(confuser_pair_proof9["survey_target_horizontal"]), 3),
            target_perceptibility=dict(
                target_pair_proof9["survey_perceptibility"]),
            perceptibility=dict(
                confuser_pair_proof9["survey_perceptibility"]),
            heading_error=confuser_pair_proof9["survey_heading_error"])
        confuser_specs9 = [
            dict(q9) for q9 in mine_scene_layout9["distractor_specs"]]
        confuser_record9.update(
            counterfactual_pair=False,
            independent_multiclass_layout=True,
            screening_class=confuser_class9,
            slots=[list(q9["slot"])
                   for q9 in confuser_specs9],
            certified_slot=list(confuser_pair_cell9),
            staged_classes=list(
                mine_scene_layout9["distractor_kinds"]),
            layout_sha256=mine_scene_layout9["layout_sha256"])
        for spec9 in confuser_specs9:
            slot9 = tuple(int(v9) for v9 in spec9["slot"])
            actual_confuser_class9 = OC.bare_block(
                spec9.get("kind") or confuser_class9)
            self.SITES.add((slot9[0], slot9[2]))
            source9 = str(spec9["source"])
            original9 = str(spec9.get("original") or "stone")
            if OC.harvest_rule(actual_confuser_class9)["target_family"] != "ore":
                raise ValueError(
                    "independent multiclass6 distractors must be ore classes, got "
                    f"{actual_confuser_class9!r}")
            geom9 = self._ore_geometry(grid9, slot9)
            screened_geometry9 = dict(
                spec9.get("geometry") or {})
            screened_placement_geometry9 = str(
                screened_geometry9.get("placement_geometry")
                or ORE_SIDE_PLACEMENT_GEOMETRY)
            measured_placement_geometry9 = (
                ore_replacement_placement_geometry(geom9))
            geom9.update(
                placement_geometry=measured_placement_geometry9,
                screened_placement_geometry=(
                    screened_placement_geometry9),
                placement_geometry_match=bool(
                    measured_placement_geometry9
                    == screened_placement_geometry9))
            geom9.update(self._mine_access_audit(
                grid9, slot9, reachable9))
            discovery9 = first_survey_checkpoint_perceptibility(
                frozen_survey9, slot9, occ9,
                max_step=MINE_SURVEY_ROUTE_STEPS,
                max_dist=MINE_CHAIN_RADIUS)
            if discovery9 is None:
                raise RuntimeError(
                    f"paired confuser lost checkpoint visibility at {slot9}")
            geom9.update(discovery9)
            if (not geom9["ok"] or not geom9["accessible"]
                    or not geom9.get("placement_geometry_match", True)
                    or mine_slot_reachable_stance(occ9, spec9) is None):
                raise RuntimeError(
                    f"paired confuser lost target-grade stance at {slot9}: "
                    f"{geom9}")
            geom9.update(
                pair_lane_id=int(spec9["pair_lane_id"]),
                pair_class_assignment_fixed=True)
            confuser_harvest9 = OC.harvest_rule(actual_confuser_class9)
            confuser_provenance9 = OC.episode_staging_provenance(
                actual_confuser_class9, source9, original9)
            ores.append(dict(kind=actual_confuser_class9, x=slot9[0], y=slot9[1],
                             z=slot9[2], source=source9,
                             original=original9, confuser=True,
                             harvest_mode=confuser_harvest9["harvest_mode"],
                             expected_drop_items=list(
                                 confuser_harvest9["drop_items"]),
                             confuser_commit_step=confuser_record9.get(
                                 "commit_step"),
                             confuser_commit_heading_error=confuser_record9.get(
                                 "heading_error"),
                             **confuser_provenance9, **geom9))
        if confuser_record9.get("verified"):
            print(f"[mine-stage] confuser "
                  f"{confuser_record9.get('staged_classes', [confuser_class9])} "
                  f"{confuser_record9['slots']} "
                  f"pose={confuser_record9.get('commit_pose_source')}"
                  f"@{confuser_record9.get('commit_step')} "
                  f"target_d={confuser_record9['target_horizontal']} "
                  f"confuser_d={confuser_record9['confuser_horizontal']} "
                  f"gap={confuser_record9['distance_gap']} "
                  f"px={confuser_record9['perceptibility']['visible_area_px']}",
                  flush=True)
        tgt = chain9[0]
        visible_start9 = dict(free.get("mine_visible_start") or {})
        episode_start9 = (
            tuple(float(v9) for v9 in visible_start9["position"])
            if setting == "vis" and visible_start9 else
            (float(free["x"]), float(free["y"]),
             float(free["z"])))
        ctx_out9 = dict(tx=tgt["x"], ty=tgt["y"], tz=tgt["z"], dist=9,
                    fam="mine_block", nm=ore_class9, act="mine", stop=4.1,
                    cls=ore_class9, clsname=ore_class9.replace("_", " "),
                    mine_sites=ores,
                    mine_target_kind=ore_class9,
                    mine_harvest_mode=target_harvest9["harvest_mode"],
                    mine_expected_drop_items=list(
                        target_harvest9["drop_items"]),
                    mine_stage_policy=str(
                        free.get("mine_stage_policy")
                        or target_harvest9["stage_policy"]),
                    mine_confuser=confuser_record9,
                    mine_counterfactual_pair=None,
                    mine_screened_pair_geometry=copy.deepcopy(pair_record9),
                    mine_scene_layout=copy.deepcopy(
                        mine_scene_layout9),
                    mine_worldgen_profile=mine_worldgen_profile9,
                    biome_terrain_topology=copy.deepcopy(
                        free.get("biome_terrain_topology") or {}),
                    start_from=(float(episode_start9[0]),
                                float(episode_start9[2])),
                    exact_start=episode_start9,
                    mine_visible_start=(
                        copy.deepcopy(visible_start9)
                        if setting == "vis" else None),
                    fixed_start_yaw=float(free["mine_start_yaw"]),
                    mine_survey_route=route9,
                    mine_survey_checkpoints=checkpoints9,
                    mine_survey_state={
                        "route": route9,
                        "checkpoint_steps": checkpoints9,
                        "cursor": 1,
                    },
                    max_start_retreat=0, no_dig=True)
        if census_scrub_witness9 is not None:
            ctx_out9["census_scrub"] = census_scrub_witness9
        return ctx_out9
