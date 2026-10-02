#!/usr/bin/env python
"""Simulator episode driver: stages a Mine world camera-off or restores a world snapshot, then runs one episode under human or policy control while recording the POV, actions and per-frame labels."""
import os, sys, json, math, time, argparse, zlib, hashlib, copy
from pathlib import Path
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, str(REPO) + "/src")
sys.path.insert(0, os.path.join(REPO, "src/attacca/evaluation"))

import attacca.evaluation.policy as E
import attacca.evaluation.geometry as G
from attacca.evaluation.mine_scene import Stager
from attacca.evaluation.mine_scene import CleanPOVRec
from attacca.evaluation.mine_scene import CELLS
from attacca.evaluation.mine_scene import MINE_REACH
from attacca.evaluation.mine_scene import V12_INDEPENDENT_MULTICLASS6_PROFILE
from attacca.evaluation.motion import MINE_DROP_COLLECT_MAX_TICKS
from attacca.evaluation.motion import MINE_POST_BREAK_MIN_CONTROLS
from attacca.evaluation.motion import MOTION_PROFILE
from attacca.runtime.mob_tracking import normalize_mob_candidates
from attacca.worlds.episode_schema import COMMITTED_SURFACE_TRACKING_CONTRACT
from attacca.worlds.episode_schema import STRUCTURAL_ENTITY_OCCLUSION_POLICY
from attacca.worlds.episode_schema import STRUCTURAL_ENTITY_QUERY_GUARD
from attacca.worlds.episode_schema import VISIBLE_SURFACE_RENDER_POSE_SOURCE
from attacca.worlds.episode_schema import structural_full_cube_surface_recognizable
from attacca.worlds.episode_schema import structural_surface_recognition_fields
from attacca.worlds.episode_schema import structural_surface_recognizable
from attacca.worlds.episode_schema import structural_recognition_contract
from attacca.worlds.episode_schema import class_visibility_roster_contract
from attacca.worlds.episode_schema import RENDERER_VIEWMODEL_ACTION_MARKER
from attacca.worlds.episode_schema import VIEWMODEL_OCCLUSION_AUDIT_FIELD
from attacca.worlds.episode_schema import VIEWMODEL_PIXEL_SUPERVISION_FIELD
from attacca.worlds.episode_schema import viewmodel_occlusion_contract
from attacca.worlds.episode_schema import viewmodel_supervision_step
from attacca.worlds import ore_classes as OC
from attacca.worlds import structural_census as SC
from attacca.worlds.structural_decision_merge import CLASS_CENSUS_FIELDS
from attacca.worlds.structural_decision_merge import FRAME_IDENTITY_FIELD
from attacca.worlds.structural_decision_merge import same_rgb_structural_aux
from attacca.worlds.structural_decision_merge import structural_rgb_identity
from attacca.worlds.human_target_choice import human_marker_pixel_match_ids
from attacca.worlds.human_target_choice import human_marker_pixel_target_choice
from attacca.worlds.human_target_choice import human_raycast_target_choice
from attacca.worlds.human_deferred_labels import HUMAN_DEFERRED_LABEL_CONTRACT
from attacca.worlds.human_deferred_labels import HUMAN_PIXEL_MARKER_CONTRACT
from attacca.worlds.human_deferred_labels import buffer_or_write_row
from attacca.worlds.human_deferred_labels import human_marker_transition
from attacca.worlds.human_deferred_labels import isolated_frozen_marker_instances
from attacca.worlds.human_deferred_labels import interaction_pressed
from attacca.worlds.human_deferred_labels import persistent_nearest_choice
from attacca.worlds.human_deferred_labels import selected_component_removed
from attacca.worlds.human_deferred_labels import source_action_index
from attacca.worlds.human_play_session import DecisionLedger
from attacca.worlds.human_play_session import HUMAN_AUTO_COMMIT_VISIBILITY_CONTRACT
from attacca.worlds.human_play_session import HUMAN_AUTO_COMMIT_VISIBLE_FRAMES
from attacca.worlds.human_play_session import HUMAN_MOTION_PROFILE
from attacca.worlds.human_play_session import HUMAN_SESSION_CONTRACT
from attacca.worlds.human_play_session import HUMAN_WORLD_SEED_RANGE
from attacca.worlds.human_play_session import advance_uncommitted_visible_streak
from attacca.worlds.human_play_session import materialize_raw_session
from attacca.worlds.human_play_session import nearest_face_distance
from attacca.worlds.human_play_session import validate_human_world_seed
from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_FIELD
from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_VALID_FIELD
from attacca.worlds.human_viewmodel import RENDERER_VIEWMODEL_DEPTH_FIELD
from attacca.worlds.human_viewmodel import RENDERER_VIEWMODEL_INFO_FIELD
from attacca.worlds.human_viewmodel import RENDERER_VIEWMODEL_MASK_CONTRACT
from attacca.worlds.human_viewmodel import install_renderer_viewmodel_observation
from attacca.worlds.human_viewmodel import renderer_mask_from_row
from attacca.worlds.human_viewmodel import renderer_mask_row_fields
from attacca.worlds.entity_id_mask import entity_surfaces
from attacca.worlds.world_snapshot import load_world_snapshot_bundle
from attacca.worlds.world_snapshot import publish_world_snapshot_bundle

DEPTH_NEAR_PLANE9 = 0.05
DEPTH_FAR_PLANE9 = 256.0

OUT = os.path.join(REPO, "outputs/attacca", f"mine_{time.strftime('%Y%m%d-%H%M')}")
HUMAN_EXACT_RECORDER_CONTRACT = "xbench_v2_human_exact_story_masks/v1"
MINE_POST_BREAK_MIN_TICKS = MINE_POST_BREAK_MIN_CONTROLS
MINE_POST_BREAK_MAX_TICKS = 100
STRUCTURAL_MOB_QUERY_BOX = (-44, 45, -24, 21, -44, 45)


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def runtime_wall_seconds(start, end):
    elapsed9 = float(end) - float(start)
    if elapsed9 < 0.0:
        raise ValueError("monotonic clock moved backwards")
    return round(elapsed9, 3)


def structural_mob_aabb_voxel_cover(kind, position):
    if not isinstance(position, (tuple, list)) or len(position) != 3:
        raise ValueError("structural mob position must be xyz")
    px9, py9, pz9 = (float(q9) for q9 in position)
    if not all(math.isfinite(q9) for q9 in (px9, py9, pz9)):
        raise ValueError("structural mob position must be finite")
    bare9 = str(kind).lower().split(":")[-1].replace("entity", "")
    fish9 = ("cod", "salmon", "pufferfish", "tropicalfish")
    small9 = ("chicken", "rabbit", "bee", "bat", "parrot")
    medium9 = (
        "sheep", "cow", "pig", "mooshroom", "wolf", "cat", "fox",
        "ocelot", "goat", "panda", "polar_bear")
    riding9 = ("horse", "donkey", "mule", "llama", "traderllama")
    humanoid9 = (
        "zombie", "skeleton", "stray", "husk", "drowned", "creeper",
        "villager", "witch", "pillager", "vindicator", "piglin")
    if any(token9 in bare9 for token9 in fish9):
        half9, below9, height9 = 0.70, 0.25, 1.00
    elif any(token9 in bare9 for token9 in small9):
        half9, below9, height9 = 0.65, 0.25, 1.35
    elif "squid" in bare9 or "dolphin" in bare9:
        half9, below9, height9 = 1.00, 0.25, 1.35
    elif any(token9 in bare9 for token9 in riding9):
        half9, below9, height9 = 1.00, 0.25, 2.35
    elif any(token9 in bare9 for token9 in humanoid9):
        half9, below9, height9 = 0.70, 0.25, 2.35
    elif "spider" in bare9:
        half9, below9, height9 = 1.00, 0.25, 1.35
    elif any(token9 in bare9 for token9 in medium9):
        half9, below9, height9 = 0.80, 0.25, 1.85
    elif any(token9 in bare9 for token9 in
             ("enderman", "irongolem", "ravager")):
        half9, below9, height9 = 1.25, 0.25, 3.50
    else:
        half9, below9, height9 = 2.25, 0.25, 4.25
    lo9 = (px9 - half9, py9 - below9, pz9 - half9)
    hi9 = (px9 + half9, py9 + height9, pz9 + half9)
    starts9 = tuple(int(math.floor(q9)) for q9 in lo9)
    stops9 = tuple(int(math.ceil(q9)) for q9 in hi9)
    cells9 = frozenset(
        (xx9, yy9, zz9)
        for xx9 in range(starts9[0], stops9[0])
        for yy9 in range(starts9[1], stops9[1])
        for zz9 in range(starts9[2], stops9[2]))
    return cells9, lo9, hi9


def mine_scene_shape_audit(mine_sites, target_class, distractor_classes,
                           *, worldgen_profile="current"):
    target9 = str(target_class)
    distractors9 = tuple(str(value9) for value9 in distractor_classes)

    def cells9(kind9):
        return [
            (int(row9["x"]), int(row9["y"]), int(row9["z"]))
            for row9 in (mine_sites or ()) if str(row9.get("kind")) == kind9]

    target_cells9 = cells9(target9)
    target_sizes9 = OC.cluster_sizes(target_cells9)
    rows9 = {}
    for kind9 in distractors9:
        values9 = cells9(kind9)
        rows9[kind9] = {
            "cell_count": len(values9),
            "component_sizes": list(OC.cluster_sizes(values9)),
        }
    symmetric9 = bool(rows9) and all(
        row9["cell_count"] == len(target_cells9)
        and tuple(row9["component_sizes"]) == target_sizes9
        for row9 in rows9.values())
    independent_multiclass6_ok9 = bool(
        str(worldgen_profile) == V12_INDEPENDENT_MULTICLASS6_PROFILE
        and len(target_cells9) == 3
        and len(rows9) == 3
        and all(row9["cell_count"] == 1
                and row9["component_sizes"] == [1]
                for row9 in rows9.values()))
    return {
        "owner": "screen_free_site_and_stager_stage_only",
        "story_world_mutations": 0,
        "target_class": target9,
        "target_cell_count": len(target_cells9),
        "target_component_sizes": list(target_sizes9),
        "distractors": rows9,
        "shape_symmetric": bool(symmetric9),
        "worldgen_profile": str(worldgen_profile),
        "independent_multiclass6_ok": independent_multiclass6_ok9,
    }


POLICY_EVAL_RESERVED_WORLD_SEEDS = frozenset({33})
POLICY_EVAL_TRAINING_WORLD_BASE_RANGES = (
    (32000, 35000), (120000, 125000), (200000, 300000),
    HUMAN_WORLD_SEED_RANGE,
)
POLICY_EVAL_WORLD_IMAGE_STRIDE = 1_000_003
POLICY_EVAL_MAX_WORLD_IMAGES = 3


def policy_eval_world_seed_disjoint(seed, *, allow_registered_range=None):
    seed9 = int(seed)
    if seed9 in POLICY_EVAL_RESERVED_WORLD_SEEDS:
        return False, f"world seed {seed9} is a reserved eval/registry world"
    for image9 in range(0, POLICY_EVAL_MAX_WORLD_IMAGES + 1):
        base9 = seed9 - image9 * POLICY_EVAL_WORLD_IMAGE_STRIDE
        for lo9, hi9 in POLICY_EVAL_TRAINING_WORLD_BASE_RANGES:
            if (allow_registered_range is not None
                    and (int(lo9), int(hi9))
                       == tuple(int(q9) for q9 in allow_registered_range)):
                continue
            if lo9 <= base9 < hi9:
                return False, (
                    f"world seed {seed9} lies in training range [{lo9},{hi9}) "
                    f"(seed image m={image9}, stride {POLICY_EVAL_WORLD_IMAGE_STRIDE})")
    return True, "disjoint from every recorded training/eval world range"


def normalize_name(name: str) -> str:
    return str(name).lower().replace("minecraft:", "").strip()


def policy_eval_wrong_block_delta(info, init_info, target_cls):
    cur9 = (info or {}).get("mine_block", {}) or {}
    old9 = (init_info or {}).get("mine_block", {}) or {}
    target9 = normalize_name(str(target_cls))
    out9 = {}
    for key9, value9 in cur9.items():
        try:
            delta9 = int(
                float(np.asarray(value9).reshape(-1)[0])
                - float(np.asarray(old9.get(key9, 0)).reshape(-1)[0]))
        except Exception:
            continue
        normalized9 = normalize_name(str(key9))
        if delta9 > 0 and normalized9 != target9:
            out9[normalized9] = out9.get(normalized9, 0) + delta9
    return out9


def policy_eval_mine_block_delta(info, init_info, block_kind):
    cur9 = (info or {}).get("mine_block", {}) or {}
    old9 = (init_info or {}).get("mine_block", {}) or {}
    wanted9 = normalize_name(str(block_kind))
    total9 = 0
    for key9, value9 in cur9.items():
        if normalize_name(str(key9)) != wanted9:
            continue
        try:
            total9 += int(
                float(np.asarray(value9).reshape(-1)[0])
                - float(np.asarray(old9.get(key9, 0)).reshape(-1)[0]))
        except Exception:
            continue
    return max(0, int(total9))


def policy_eval_crosshair_target_cell(instances, pov_shape, *, excluded=()):
    try:
        height9, width9 = int(pov_shape[0]), int(pov_shape[1])
    except (TypeError, ValueError, IndexError):
        return None
    center_x9 = {max(0, width9 // 2 - 1), width9 // 2}
    center_y9 = {max(0, height9 // 2 - 1), height9 // 2}
    excluded9 = {tuple(int(q9) for q9 in cell9) for cell9 in (excluded or ())}
    hits9 = []
    for instance9 in instances or ():
        raw_cell9 = (instance9.get("world_position")
                     or instance9.get("cell"))
        try:
            cell9 = tuple(int(q9) for q9 in raw_cell9)
        except (TypeError, ValueError):
            continue
        if len(cell9) != 3 or cell9 in excluded9:
            continue
        proof9 = instance9.get("proof") or {}
        hit9 = False
        for run9 in proof9.get("certified_pixel_runs_yx") or ():
            try:
                y9, x0_9, x1_9 = (int(q9) for q9 in run9)
            except (TypeError, ValueError):
                continue
            if y9 in center_y9 and any(x0_9 <= x9 <= x1_9 for x9 in center_x9):
                hit9 = True
                break
        if hit9:
            aim9 = proof9.get("aim_pixel") or (width9 // 2, height9 // 2)
            try:
                rank9 = ((float(aim9[0]) - width9 / 2.0) ** 2
                         + (float(aim9[1]) - height9 / 2.0) ** 2)
            except (TypeError, ValueError, IndexError):
                rank9 = 0.0
            hits9.append((rank9, cell9))
    return None if not hits9 else min(hits9)[1]


def policy_eval_staged_cell_box(staged_cells, player_pos, max_rel=40):
    if not staged_cells:
        return None
    px9, py9, pz9 = (int(math.floor(float(q9))) for q9 in player_pos)
    rel9 = [(int(x9) - px9, int(y9) - py9, int(z9) - pz9)
            for (x9, y9, z9) in staged_cells]
    if any(abs(q9) > int(max_rel) for cell9 in rel9 for q9 in cell9):
        return None
    xs9, ys9, zs9 = zip(*rel9)
    return np.array([min(xs9), max(xs9) + 1, min(ys9), max(ys9) + 1,
                     min(zs9), max(zs9) + 1], np.int32)


def policy_eval_retire_broken_cells(occupancy, broken_cells):
    if not isinstance(occupancy, G.OccupancyMap):
        raise TypeError("policy broken-cell retirement requires OccupancyMap")
    removed9 = []
    for raw_cell9 in sorted(broken_cells or ()):
        cell9 = tuple(int(q9) for q9 in raw_cell9)
        if cell9 in occupancy.grid:
            occupancy.grid.pop(cell9)
            removed9.append(cell9)
    return removed9


def mine_exact_class_cells(occupancy, block_kind, *, census_roster):
    grid9 = getattr(occupancy, "grid", None)
    revision9 = getattr(occupancy, "revision", None)
    if grid9 is None or revision9 is None:
        raise TypeError(
            "exact mine-class lookup requires tracked occupancy grid/revision")
    target9 = G._bare(str(block_kind))
    roster9 = tuple(G._bare(str(kind9)) for kind9 in census_roster)
    if target9 in roster9:
        if roster9 == SC.SUPPORTED_CLASSES:
            index9, _runtime9 = SC.index_supported_cells_cached(occupancy)
        else:
            index9, _runtime9 = SC.index_supported_cells_cached(
                occupancy, roster=roster9)
        return list(index9[target9])

    scope9 = (id(grid9), int(revision9))
    cache9 = getattr(occupancy, "_xbench_target_only_class_cache", None)
    if not isinstance(cache9, dict) or cache9.get("scope") != scope9:
        cache9 = {"scope": scope9, "classes": {}}
        occupancy._xbench_target_only_class_cache = cache9
    classes9 = cache9["classes"]
    if target9 not in classes9:
        classes9[target9] = tuple(sorted(
            tuple(int(value9) for value9 in cell9)
            for cell9, raw_kind9 in grid9.items()
            if G._bare(str(raw_kind9)) == target9))
    return list(classes9[target9])


POLICY_EVAL_RESULT_FIELDS = (
    "mode", "model", "ckpt", "policy_backend", "cell", "setting", "gmode",
    "biome", "group_id",
    "target_cls", "confuser_cls", "world_seed", "aseed", "policy_rollout_seed",
    "site_seed", "layout_seed",
    "pose_seed", "quota", "success_quota", "succ", "succ_step", "steps", "budget", "wrong", "wrong_ore",
    "wrong_ore_blocks", "staged_target_broken", "staged_confuser_broken",
    "target_breaks", "distractor_breaks", "distractor_classes",
    "snapshot_world_sha256", "snapshot_archive_sha256",
    "snapshot_payload_sha256",
    "selected_classswap_geometry_sha256",
    "selected_classswap_assignment_sha256",
    "goal_cache_hits", "goal_cache_misses",
    "dynamic_goal_summary",
    "first_staged_break_kind", "wrong_class_first_hit",
    "eval_valid", "invalid_reason",
    "frames_kept", "fps", "grounding_jsonl")

POLICY_GOAL_MODES = (
    "plains_grass_front",
)


def policy_eval_result_row(**fields):
    fields.setdefault("success_quota", fields.get("quota"))
    fields.setdefault("selected_classswap_geometry_sha256", None)
    fields.setdefault("selected_classswap_assignment_sha256", None)
    missing9 = sorted(set(POLICY_EVAL_RESULT_FIELDS) - set(fields))
    extra9 = sorted(set(fields) - set(POLICY_EVAL_RESULT_FIELDS))
    if missing9 or extra9:
        raise ValueError(
            f"policy eval row schema drift: missing={missing9} extra={extra9}")
    row9 = {key9: fields[key9] for key9 in POLICY_EVAL_RESULT_FIELDS}
    json.dumps(row9)
    return row9


def main():
    timing_main_started9 = time.monotonic()
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--settings", default="vis,invis")
    ap.add_argument("--biome", default=None)
    ap.add_argument("--free-site", action="store_true")
    ap.add_argument("--world-seed", type=int, default=None)
    ap.add_argument("--action-seed", type=int, default=0)
    ap.add_argument("--site-seed", type=int, default=None,
                    help="free-site selection seed (fixed per episode)")
    ap.add_argument("--layout-seed", type=int, default=None,
                    help="staged layout seed (fixed per episode)")
    ap.add_argument("--pose-seed", type=int, default=None,
                    help="initial spawn pose seed (fixed per episode)")
    ap.add_argument("--story", action="store_true")
    ap.add_argument(
        "--visible-surface-masks", action="store_true",
        help=("emit dense, occlusion-aware chosen-instance visible-surface masks "
              "registered with the render pose"))
    ap.add_argument(
        "--all-class-census", action=argparse.BooleanOptionalAction,
        default=False,
        help=("with dense visible-surface masks, also run the visibility census "
              "over the full mine roster on every kept frame; non-goal classes "
              "without a proof of absence are labeled UNKNOWN"))
    ap.add_argument(
        "--census-scrub", action="store_true",
        help=("with --all-class-census (required), certify per-frame ABSENCE for "
              "the seven-ore roster by construction: during camera-off staging, "
              "after site acceptance and before staged-cell placement, every "
              "natural roster-ore block inside a recognizability-bounded region "
              "(recognition-gate pixel bound; see structural_census.py) is "
              "replaced with stone and audited.  Scrubbed classes the census "
              "does not find then become known-ABSENT (known=1, visible=0) "
              "instead of UNKNOWN whenever the frame's recognition ball stays "
              "inside the scrubbed region."))
    ap.add_argument("--ore-classes", default=",".join(OC.MINE_ORE_POOL),
                    help=("mine-cell class pool; the per-episode target is drawn from it and the "
                          "distractors from the rest, so no class is confined to one role.  Every "
                          "member must satisfy the central fixed-kit harvest/drop contract; a block "
                          "that breaks without its expected drop is a failure, not a wider pool."))
    ap.add_argument("--ore-class", default=None,
                    help="mine target class of the episode")
    ap.add_argument("--ore-distractor-classes", type=int, default=1,
                    help=("how many non-target classes appear as distractors.  Each gets the same "
                          "cell count and cluster-size multiset as the target, so per-class shape "
                          "statistics stay role-independent."))
    ap.add_argument(
        "--mine-worldgen-profile",
        choices=(V12_INDEPENDENT_MULTICLASS6_PROFILE,),
        default=V12_INDEPENDENT_MULTICLASS6_PROFILE,
        help=("mine staging profile: the natural extreme-hills exposed-stone 3+3 "
              "geometry, one fresh solo goal episode, the goal class on three cells "
              "and three distinct non-goal ore classes on the other three cells"))
    ap.add_argument(
        "--benchmark-selected-classswap", action="store_true",
        help=("evaluation-only restore contract for immutable selected-world "
              "class-swap snapshots. The declared generalization targets are "
              "recognized only as the declared exact target; they are never "
              "added to the generation, distractor, census, or training roster."))
    ap.add_argument(
        "--selected-classswap-geometry-sha256", default=None,
        help=("class-independent source geometry/start digest required by "
              "--benchmark-selected-classswap"))
    ap.add_argument(
        "--selected-classswap-assignment-sha256", default=None,
        help=("exact deterministic six-cell assignment digest required by "
              "--benchmark-selected-classswap"))
    ap.add_argument("--solo-episode", action="store_true",
                    help=("stage one target lane per episode instead of a same-scene pair; "
                          "use a fresh world seed per episode and combine with "
                          "--visible-surface-masks --all-class-census and "
                          "--distractor-policy."))
    ap.add_argument("--distractor-policy", choices=("random_other",),
                    default="random_other",
                    help=("--solo-episode only: 'random_other' stages non-goal ore "
                          "classes next to the goal class"))
    ap.add_argument("--quota", type=int, choices=(1, 3, 4, 5), default=None,
                    help="mine chain length (number of goal-class blocks to break)")
    ap.add_argument("--site-attempts", type=int, default=10,
                    help=("number of random site offsets tried in one booted world before "
                          "the world is discarded"))
    ap.add_argument(
        "--human-play", action="store_true",
        help=("control the recorded episode with native keyboard/mouse input while "
              "retaining this runner's canonical clean POV, exact render-pose story "
              "labels, success checker and mask contracts"))
    ap.add_argument("--human-timeout", type=float, default=300.0,
                    help="wall-clock gameplay budget in seconds for --human-play")
    ap.add_argument("--human-fps", type=int, default=20,
                    help="target native input/render rate for --human-play")
    ap.add_argument("--human-scale", type=int, default=2,
                    help="native play window scale (640x360 multiplied by this)")
    ap.add_argument("--human-mouse-sens", type=float, default=0.15,
                    help="degrees per mouse pixel in --human-play")
    ap.add_argument("--ready-file", default=None,
                    help="file touched once Minecraft has booted and the world is prepared")
    ap.add_argument(
        "--human-status-file", default=None,
        help=("atomic native-human lifecycle status: staged_ready, playing, "
              "paused/selecting, postplay_labels, raw_complete"))
    ap.add_argument(
        "--human-start-permit-file", default=None,
        help=("when set, the staged READY window accepts V only while this "
              "queue-owned permit file exists"))
    ap.add_argument(
        "--policy-driver", action="store_true",
        help=("control the recorded episode with the goal-conditioned policy once "
              "recording starts (same seam as --human-play); requires a multiclass6 "
              "staged-world snapshot on a world seed disjoint from the training seeds, "
              "and writes results_policy.jsonl"))
    ap.add_argument("--policy-ckpt", default=None,
                    help="policy .ckpt path")
    ap.add_argument("--policy-cfg-coef", type=float, default=0.0,
                    help="CFG coefficient (default 0.0)")
    ap.add_argument("--policy-goal-lib", default=None,
                    help="goal library root in the load_goal() layout: "
                         "<lib>/mine_<cls>/<gmode>/{goal.png,mask.png}")
    ap.add_argument("--policy-gmode", default="plains_grass_front",
                    choices=POLICY_GOAL_MODES,
                    help="goal exemplar variant directory inside --policy-goal-lib")
    ap.add_argument("--policy-budget", type=int, default=1000,
                    help="policy episode step budget")
    ap.add_argument("--policy-post", type=int, default=12,
                    help="extra steps recorded after the first success")
    ap.add_argument(
        "--policy-success-quota", type=int, default=None,
        help=("number of exact staged target breaks that ends evaluation; "
              "default is the physical task quota. Use 1 for Success@1."))
    ap.add_argument("--policy-record", type=int, default=1,
                    help="1 = save the frame-guarded episode mp4 (policy_<cell>_<setting>.mp4)")
    ap.add_argument("--policy-model-name", default="policy",
                    help="model label written into every result row")
    ap.add_argument(
        "--policy-rollout-seed", type=int, default=None,
        help=("policy sampling RNG only; leaves the staged-world action seed "
              "unchanged so one immutable snapshot can support repeated rollouts"))
    ap.add_argument(
        "--policy-backend", choices=("rocket2_static_goal",),
        default="rocket2_static_goal",
        help="policy conditioning driver (static goal image and mask)")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--world-snapshot-in", default=None,
        help=("load one SHA-pinned staged-world bundle and bypass fresh site "
              "search/staging; restricted to one multiclass6 HumanPlay mine episode"))
    ap.add_argument(
        "--world-snapshot-out", default=None,
        help=("after final camera-off staging, /save-all flush and atomically "
              "publish a reusable staged-world bundle"))
    ap.add_argument(
        "--world-snapshot-only", action="store_true",
        help="publish --world-snapshot-out then exit before recording/GUI")
    ap.add_argument(
        "--snapshot-allow-uncanonical-ready", action="store_true",
        help=("multiclass6 snapshot policy evaluation without a canonical cloudless "
              "READY hash: the restored READY frame is compared against the "
              "snapshot generation source only (never fatal; recorded as "
              "comparison_kind=snapshot_generation_source)"))
    args = ap.parse_args()
    requested_ore_pool9 = tuple(
        OC.bare_block(value9) for value9 in args.ore_classes.split(",")
        if value9)
    benchmark_stage_pool9 = tuple(
        OC.MINE_POC_ORE_POOL + OC.MINE_HELDOUT_EVAL_POOL)
    selected_classswap_stage_pool9 = tuple(OC.MINE_POC_ORE_POOL)
    selected_classswap_targets9 = (
        "coal_ore", "iron_ore", "gold_ore", "lapis_ore", "diamond_ore",
        "emerald_ore", "redstone_ore", "tube_coral_block",
        "brain_coral_block", "bubble_coral_block", "fire_coral_block",
        "horn_coral_block")
    if args.benchmark_selected_classswap:
        if requested_ore_pool9 != selected_classswap_stage_pool9:
            ap.error(
                "selected-world class-swap evaluation requires the exact ordered "
                "seven-ID-ore runtime pool")
        mine_stage_pool9 = tuple(OC.validate_mine_pool(requested_ore_pool9))
        mine_census_roster9 = tuple(
            OC.MINE_TARGET_POOL + OC.MINE_HELDOUT_EVAL_POOL)
    else:
        mine_stage_pool9 = tuple(OC.validate_mine_pool())
        mine_census_roster9 = tuple(OC.MINE_TARGET_POOL)
    mine_target_pool9 = frozenset(mine_census_roster9)

    def publish_human_status9(phase9, **fields9):
        if not args.human_status_file:
            return
        status_path9 = os.path.abspath(args.human_status_file)
        os.makedirs(os.path.dirname(status_path9), exist_ok=True)
        payload9 = {
            "version": 1,
            "contract": "xbench_v2_human_desktop_lifecycle/v1",
            "phase": str(phase9),
            "pid": int(os.getpid()),
            "updated_unix_s": float(time.time()),
            "world_seed": args.world_seed,
            **fields9,
        }
        temp_path9 = f"{status_path9}.tmp.{os.getpid()}"
        with open(temp_path9, "w", encoding="utf-8") as status_f9:
            json.dump(payload9, status_f9, indent=2, sort_keys=True)
            status_f9.write("\n")
            status_f9.flush()
            os.fsync(status_f9.fileno())
        os.replace(temp_path9, status_path9)

    def is_mine_target_kind9(raw9):
        return G._bare(str(raw9 or "")) in mine_target_pool9

    requested_cells9 = [c for c in args.cells.split(",") if c]
    requested_settings9 = [s for s in args.settings.split(",") if s]
    v12_multiclass6_mode9 = bool(
        args.mine_worldgen_profile
        == V12_INDEPENDENT_MULTICLASS6_PROFILE)
    if args.benchmark_selected_classswap:
        if (not v12_multiclass6_mode9
                or not args.policy_driver
                or not args.world_snapshot_in or args.world_snapshot_out
                or args.human_play
                or args.ore_class not in selected_classswap_targets9
                or args.selected_classswap_geometry_sha256 is None
                or args.selected_classswap_assignment_sha256 is None):
            ap.error(
                "--benchmark-selected-classswap requires an eval-only multiclass6 "
                "snapshot restore and one declared class-swap target")
    v12_snapshot_policy_eval9 = bool(
        v12_multiclass6_mode9 and args.policy_driver
        and args.world_snapshot_in)
    if args.world_snapshot_in and args.world_snapshot_out:
        ap.error("--world-snapshot-in and --world-snapshot-out are mutually exclusive")
    if args.world_snapshot_only and not args.world_snapshot_out:
        ap.error("--world-snapshot-only requires --world-snapshot-out")
    if (v12_snapshot_policy_eval9
            and not args.snapshot_allow_uncanonical_ready):
        ap.error(
            "multiclass6 snapshot policy evaluation requires "
            "--snapshot-allow-uncanonical-ready")
    if args.world_snapshot_out and not args.world_snapshot_only:
        ap.error(
            "--world-snapshot-out is terminal and requires --world-snapshot-only")
    if args.world_snapshot_in or args.world_snapshot_out:
        if (requested_cells9 != ["mine"] or len(requested_settings9) != 1
                or not v12_multiclass6_mode9):
            ap.error(
                "staged-world snapshots currently require one mine setting, "
                "and v12_independent_multiclass6")
        if args.world_snapshot_out and not args.human_play:
            ap.error("staged-world snapshot publication requires --human-play")
        if args.world_snapshot_in and not (
                args.human_play or v12_snapshot_policy_eval9):
            ap.error(
                "staged-world snapshot restore requires --human-play or the "
                "multiclass6 snapshot policy-eval contract")
        if any(value9 is None for value9 in (
                args.world_seed, args.site_seed,
                args.layout_seed, args.pose_seed)):
            ap.error(
                "staged-world snapshots require explicit world/site/layout/pose seeds")
    snapshot_bundle9 = (
        load_world_snapshot_bundle(Path(args.world_snapshot_in).resolve())
        if args.world_snapshot_in else None)
    snapshot_restore9 = (
        None if snapshot_bundle9 is None else snapshot_bundle9.payload)
    snapshot_classswap9 = None
    selected_classswap_rewrite_ctx9 = None
    if snapshot_restore9 is not None:
        if (not isinstance(snapshot_restore9, dict)
                or snapshot_restore9.get("contract")
                != "xbench-v2-demo-staged-state/v1"):
            ap.error("--world-snapshot-in has no supported demo staged-state payload")
    if args.benchmark_selected_classswap:
        from attacca.evaluation.mine_worlds import assignment_sha256 as _classswap_assignment_sha2569
        from attacca.evaluation.mine_worlds import deterministic_assignment as _classswap_assignment9
        from attacca.evaluation.mine_worlds import extract_source_layout as _classswap_source_layout9
        from attacca.evaluation.mine_worlds import geometry_start_sha256 as _classswap_geometry_sha2569
        try:
            source_layout9 = _classswap_source_layout9(snapshot_restore9)
            assignment9 = _classswap_assignment9(
                source_layout9, str(args.ore_class))
            geometry_sha9 = _classswap_geometry_sha2569(snapshot_restore9)
            assignment_sha9 = _classswap_assignment_sha2569(assignment9)
        except (TypeError, ValueError) as exc:
            ap.error(f"invalid selected-world source snapshot: {exc}")
        if (geometry_sha9 != str(args.selected_classswap_geometry_sha256)
                or assignment_sha9
                != str(args.selected_classswap_assignment_sha256)):
            ap.error(
                "selected-world geometry/assignment digest differs from the "
                "declared scene manifest")
        snapshot_classswap9 = {
            "contract": "xbench_mine_shared16_classswap_snapshot/v2",
            "target_kind": str(args.ore_class),
            "geometry_sha256": geometry_sha9,
            "assignment_sha256": assignment_sha9,
            "assignment": assignment9,
        }

        def selected_classswap_rewrite_ctx9(source_ctx9, assignment10):
            runtime_ctx10 = copy.deepcopy(source_ctx9)
            target10 = G._bare(str(assignment10["target_kind"]))
            by_cell10 = {
                tuple(int(q10) for q10 in row10["cell"]): dict(row10)
                for row10 in assignment10["staged_cells"]
            }
            if len(by_cell10) != 6:
                raise RuntimeError(
                    "selected-world assignment is not six unique cells")
            drop_items10 = {
                "coal_ore": ["coal"],
                "iron_ore": ["iron_ore"],
                "gold_ore": ["gold_ore"],
                "lapis_ore": ["lapis_lazuli"],
                "diamond_ore": ["diamond"],
                "emerald_ore": ["emerald"],
                "redstone_ore": ["redstone"],
                "nether_gold_ore": ["gold_nugget"],
                "tube_coral_block": [],
                "brain_coral_block": [],
                "bubble_coral_block": [],
                "fire_coral_block": [],
                "horn_coral_block": [],
                "blue_ice": [],
                "red_sandstone": ["red_sandstone"],
                "ancient_debris": ["ancient_debris"],
                "gilded_blackstone": ["gilded_blackstone"],
            }
            if target10 not in drop_items10:
                raise RuntimeError(
                    f"selected-world target has no drop contract: {target10}")
            for key10 in ("cls", "nm", "mine_target_kind", "target_kind"):
                if key10 in runtime_ctx10 or key10 != "target_kind":
                    runtime_ctx10[key10] = target10
            runtime_ctx10["clsname"] = target10.replace("_", " ")
            runtime_ctx10["quota"] = 1
            runtime_ctx10["mine_expected_drop_items"] = list(
                drop_items10[target10])
            runtime_ctx10["mine_harvest_mode"] = (
                "exact_staged_voxel_removal")

            sites10 = list(runtime_ctx10.get("mine_sites") or ())
            if len(sites10) != 6:
                raise RuntimeError(
                    "selected-world source ctx lost its six mine sites")
            for site10 in sites10:
                cell10 = tuple(int(site10[key10])
                               for key10 in ("x", "y", "z"))
                row10 = by_cell10.get(cell10)
                if row10 is None:
                    raise RuntimeError(
                        f"undeclared selected-world source cell {cell10}")
                kind10 = G._bare(str(row10["after"]))
                site10["kind"] = kind10
                site10["expected_drop_items"] = list(drop_items10[kind10])
                site10["harvest_mode"] = "exact_staged_voxel_removal"
                site10["world_mutation_commands"] = [
                    f"/setblock {cell10[0]} {cell10[1]} {cell10[2]} "
                    f"minecraft:{kind10}"]
            runtime_ctx10["mine_sites"] = sites10
            runtime_ctx10.pop("_ore_cluster_plan", None)
            runtime_ctx10.pop("mine_counterfactual_pair", None)

            def rewrite_layout10(value10):
                if isinstance(value10, list):
                    return [rewrite_layout10(item10) for item10 in value10]
                if not isinstance(value10, dict):
                    return value10
                output10 = {
                    key10: rewrite_layout10(item10)
                    for key10, item10 in value10.items()
                }
                raw_cell10 = output10.get("slot")
                if (raw_cell10 is None
                        and all(key10 in output10
                                for key10 in ("x", "y", "z"))):
                    raw_cell10 = [output10["x"], output10["y"],
                                  output10["z"]]
                try:
                    cell10 = (None if raw_cell10 is None else
                              tuple(int(q10) for q10 in raw_cell10))
                except (TypeError, ValueError):
                    cell10 = None
                row10 = by_cell10.get(cell10)
                if row10 is not None and "kind" in output10:
                    output10["kind"] = G._bare(str(row10["after"]))
                    if "role" in output10:
                        output10["role"] = (
                            "target" if row10["role"] == "target"
                            else "distractor")
                return output10

            layout10 = rewrite_layout10(copy.deepcopy(
                runtime_ctx10.get("mine_scene_layout") or {}))
            if layout10.get("contract") \
                    != "independent_v12_exposed_stone_multiclass6/v1":
                raise RuntimeError(
                    "selected-world source has unexpected scene layout")
            source_layout_sha10 = layout10.pop("layout_sha256", None)
            layout10["source_layout_sha256"] = source_layout_sha10
            layout10["target_kind"] = target10
            layout10["distractor_kinds"] = list(
                assignment10["confuser_classes"])
            layout10["canonical_classes"] = [
                target10, *assignment10["confuser_classes"]]
            layout10["layout_sha256"] = hashlib.sha256(json.dumps(
                layout10, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            runtime_ctx10["mine_scene_layout"] = layout10
            runtime_ctx10["mine_confuser"] = {
                "contract": "xbench_mine_shared16_classswap_snapshot/v2",
                "required": True,
                "satisfied": True,
                "target_kind": target10,
                "target_cells": list(assignment10["target_cells"]),
                "confuser_classes": list(
                    assignment10["confuser_classes"]),
                "confuser_cells": list(assignment10["confuser_cells"]),
                "source_screening_record_superseded": True,
            }
            runtime_ctx10["selected_world_classswap"] = copy.deepcopy(
                snapshot_classswap9)
            return runtime_ctx10

        mine_target_pool9 = frozenset(
            set(mine_target_pool9) | {G._bare(str(args.ore_class))})
    if args.human_play:
        if len(requested_cells9) != 1 or len(requested_settings9) != 1:
            ap.error("--human-play requires exactly one cell and one setting per process")
        if not args.story:
            ap.error("--human-play requires --story (manual data without labels is not a dataset)")
        if requested_cells9 != ["mine"]:
            ap.error("human play supports only --cells mine")
        if not args.visible_surface_masks or not args.all_class_census:
            ap.error("--human-play requires exact masks and the all-class census")
        v12_human_staging9 = bool(
            args.free_site and args.solo_episode
            and args.distractor_policy == "random_other"
            and v12_multiclass6_mode9)
        if not v12_human_staging9:
            ap.error("--human-play requires v12_independent_multiclass6 "
                     "solo/random_other staging")
        if int(args.human_fps) != 20 or float(args.human_timeout) <= 0.0:
            ap.error("human raw capture is fixed at 20 Hz; timeout must be positive")
        if not os.environ.get("DISPLAY"):
            ap.error("--human-play requires a native X display (DISPLAY), not xvfb")
        try:
            validate_human_world_seed(args.world_seed)
        except (TypeError, ValueError) as exc:
            ap.error(str(exc))
        seed_ok9, seed_reason9 = policy_eval_world_seed_disjoint(
            args.world_seed, allow_registered_range=HUMAN_WORLD_SEED_RANGE)
        if not seed_ok9:
            ap.error(f"--human-play world-seed guard: {seed_reason9}")
        if args.quota is not None and not args.world_snapshot_only:
            ap.error("--human-play takes its quota from the staged layout; "
                     "--quota is not accepted")
    if v12_multiclass6_mode9:
        requirements9 = {
            "one --cells mine": requested_cells9 == ["mine"],
            "--free-site": bool(args.free_site),
            "--solo-episode": bool(args.solo_episode),
            "--story": bool(args.story),
            "--visible-surface-masks": bool(args.visible_surface_masks),
            "--all-class-census": bool(args.all_class_census),
            "--world-seed": args.world_seed is not None,
            "--layout-seed": args.layout_seed is not None,
            "--quota 3 (or selected-classswap quota 1)": bool(
                args.quota == 3 or args.human_play
                or (args.benchmark_selected_classswap and args.quota == 1)),
            "--distractor-policy random_other": (
                args.distractor_policy == "random_other"),
            "--ore-distractor-classes 3": (
                int(args.ore_distractor_classes) == 3),
            "--biome extreme_hills": args.biome == "extreme_hills",
        }
        missing9 = sorted(
            label9 for label9, ok9 in requirements9.items() if not ok9)
        if missing9:
            ap.error("independent multiclass6 requires " + ", ".join(missing9))
    if args.census_scrub and not (
            args.visible_surface_masks and args.all_class_census):
        ap.error(
            "--census-scrub requires --visible-surface-masks --all-class-census: "
            "certified absence is defined only for the dense all-class census")
    if args.census_scrub and "mine" not in requested_cells9:
        ap.error("--census-scrub applies only to the mine cell")
    if args.census_scrub and not args.free_site:
        ap.error("--census-scrub requires --free-site (screened probe data "
                 "derives the scrub region)")
    if not (args.human_play or args.policy_driver):
        ap.error("an episode needs --human-play or --policy-driver")
    policy_ckpt9 = None
    policy_goal9 = None
    policy_runner_cell9 = {"runner": None}
    if args.policy_driver:
        if not v12_snapshot_policy_eval9:
            ap.error("--policy-driver requires a multiclass6 staged-world snapshot")
        if args.human_play:
            ap.error("--policy-driver and --human-play are exclusive controllers")
        if int(args.policy_budget) < 1 or int(args.policy_post) < 0:
            ap.error("--policy-budget must be >= 1 and --policy-post >= 0")
        policy_ckpt9 = args.policy_ckpt
        if not policy_ckpt9 or not (os.path.isfile(policy_ckpt9)
                                    or os.path.isdir(policy_ckpt9)):
            ap.error(f"--policy-driver needs a real ckpt file or HF dir "
                     f"(--policy-ckpt); got {policy_ckpt9!r}")
        if not args.policy_goal_lib:
            ap.error("static --policy-driver requires --policy-goal-lib")
        seed_ok9, seed_reason9 = policy_eval_world_seed_disjoint(args.world_seed)
        if not seed_ok9:
            ap.error(f"--policy-driver world-seed guard: {seed_reason9}")
        import cv2 as _cv2p
        policy_goal_kind9 = args.ore_class
        if policy_goal_kind9 is None:
            ap.error("--policy-driver cannot resolve its mine target class")
        goal_dir9 = os.path.join(
            args.policy_goal_lib,
            OC.goal_key("mine", policy_goal_kind9), args.policy_gmode)
        goal_img9 = _cv2p.imread(os.path.join(goal_dir9, "goal.png"))
        goal_mask9 = _cv2p.imread(os.path.join(goal_dir9, "mask.png"),
                                  _cv2p.IMREAD_GRAYSCALE)
        if goal_img9 is None or goal_mask9 is None:
            ap.error(f"--policy-goal-lib lacks goal.png/mask.png under {goal_dir9}")
        policy_goal9 = (_cv2p.cvtColor(goal_img9, _cv2p.COLOR_BGR2RGB),
                        (goal_mask9 > 0).astype(np.uint8))
    dense_surface_cells9 = {"mine", "ignite_wood", "open", "portal"}
    unsupported_dense_cells9 = sorted(
        set(requested_cells9) - dense_surface_cells9)
    if args.visible_surface_masks and unsupported_dense_cells9:
        ap.error(
            "--visible-surface-masks currently requires renderer-matched target "
            "geometry; unsupported cells (no false AABB silhouettes): "
            + ",".join(unsupported_dense_cells9))
    global OUT
    if args.out:
        OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    if args.policy_driver:
        policy_runner_cell9["runner"] = E.Rocket2GoalRunner(
            policy_ckpt9, cfg_coef=float(args.policy_cfg_coef))
        print(f"[policy-driver] loaded {args.policy_model_name} ckpt={policy_ckpt9} "
              f"cfg={float(args.policy_cfg_coef)} backend={args.policy_backend}",
              flush=True)
    runtime_overlay9 = (
        os.path.join(REPO, "configs", "xbench_runtime_eval")
        if args.policy_driver
        else os.path.join(REPO, "configs", "xbench_runtime_dense")
        if args.visible_surface_masks else None)
    if args.visible_surface_masks:
        install_renderer_viewmodel_observation()
    timing_boot_started9 = time.monotonic()
    w = E.boot_world(args.world_seed if args.world_seed is not None else 33,
                     biome=args.biome, action_type="env",
                     runtime_overlay=runtime_overlay9,
                     world_snapshot_dir=(
                         None if snapshot_bundle9 is None else
                         str(snapshot_bundle9.world.directory)),
                     world_snapshot_sha256=(
                         None if snapshot_bundle9 is None else
                         snapshot_bundle9.world.sha256),
                     world_snapshot_archive=(
                         None if snapshot_bundle9 is None else
                         str(snapshot_bundle9.archive)),
                     world_snapshot_archive_sha256=(
                         None if snapshot_bundle9 is None else
                         str(snapshot_bundle9.metadata[
                             "world_archive_sha256"])))
    timing_boot_finished9 = time.monotonic()
    runtime_timing9 = {
        "schema_version": 1,
        "clock": "time.monotonic",
        "units": "seconds",
        "world_seed": args.world_seed,
        "action_seed": int(args.action_seed),
        "requested_biome": args.biome,
        "boot_world_wall_s": runtime_wall_seconds(
            timing_boot_started9, timing_boot_finished9),
        "prep_world_wall_s": None,
        "main_to_ready_wall_s": None,
        "episodes": {},
    }

    TRAJ = {"fh": None, "pending_row": None, "note": "", "t": 0, "phase": 0,
            "event": "", "bc_valid": 1, "last_phase": 0, "last_event": "",
            "last_bc_valid": 1, "ray_hit": None, "target_cell": None,
            "aim_point": None, "viewmodel_remaining_controls": 0,
            "motion_phase": "explore", "post_break": 0,
            "drop_collect": 0, "drop_collect_stage": "none",
            "pickup_expected_items": [], "pickup_inventory_delta": {},
            "pickup_counter_delta": {}, "pickup_confirmed": 0,
            "pickup_latency_controls": -1,
            "pickup_abandoned": 0, "pickup_abandon_reason": "none",
            "pickup_attempt_controls": -1,
            "pickup_attribution_valid": 0,
            "pickup_navigation_source": "none",
            "pickup_visible_preempt_proof": None,
            "mine_attack_distance_threshold": -1.0,
            "stuck_recovery_demo": 0, "stuck_recovery_stage": "none",
            "deferred_human_rows": [], "human_preaction_evidence": None}
    _orig_sim_step = w.sim.step

    STORY = {"on": args.story, "mobs": [], "walk": None, "pending_event": "",
             "pending_struct_fields": None, "rows": [], "rec": None}
    STRUCT = {"last_voxel_box": None, "last_mob_box": None,
              "defer_human_labels": False,
              "defer_human_start_t": None,
              "offline_human_label_row": None,
              "human_episode_summary": None}

    def set_phase(phase):
        TRAJ["phase"] = int(phase)
        if STORY["on"]:
            STORY["walk"] = None

    def current_label_player_position9():
        offline9 = STRUCT.get("offline_human_label_row")
        if isinstance(offline9, dict):
            pose9 = offline9.get("render_pose")
            if not isinstance(pose9, dict):
                raise RuntimeError("offline human RGB lacks render_pose")
            try:
                pos9 = tuple(float(pose9[key9]) for key9 in ("x", "y", "z"))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("offline human render position is malformed") from exc
            if not all(math.isfinite(q9) for q9 in pos9):
                raise RuntimeError("offline human render position is non-finite")
            return pos9
        if args.policy_driver:
            x9, y9, z9, _yaw9, _pitch9, _eye_height9, _source9 = (
                render_label_pose9())
            return (float(x9), float(y9), float(z9))
        return tuple(float(q9) for q9 in w.get_pos())

    def current_label_gui_open9():
        offline9 = STRUCT.get("offline_human_label_row")
        if isinstance(offline9, dict):
            value9 = offline9.get("is_gui_open")
            if not isinstance(value9, bool):
                raise RuntimeError("offline human RGB lacks boolean GUI state")
            return value9
        return bool(w.info.get("is_gui_open", False))

    def current_mob_visibility_scan_scope9():
        offline9 = STRUCT.get("offline_human_label_row")
        offline_mobs9 = (offline9.get("structural_mob_observation")
                         if isinstance(offline9, dict) else None)
        raw_box9 = (offline_mobs9.get("query_box_relative")
                    if isinstance(offline_mobs9, dict) else
                    STRUCT.get("last_mob_box"))
        relative9 = None
        if isinstance(raw_box9, (tuple, list)) and len(raw_box9) == 6:
            try:
                relative9 = [int(q9) for q9 in raw_box9]
            except (TypeError, ValueError, OverflowError):
                relative9 = None
        player9 = [float(q9) for q9 in current_label_player_position9()]
        absolute9 = None
        if relative9 is not None:
            absolute9 = [
                player9[0] + relative9[0], player9[0] + relative9[1],
                player9[1] + relative9[2], player9[1] + relative9[3],
                player9[2] + relative9[4], player9[2] + relative9[5],
            ]
        packet_present9 = (
            bool(offline_mobs9.get("packet_present"))
            if isinstance(offline_mobs9, dict) else
            w.info.get("mobs") is not None)
        scope_available9 = bool(relative9 is not None and packet_present9)
        return {
            "version": 1,
            "coordinate_system": "world_xyz_continuous_half_open",
            "player_position": player9,
            "mob_query_relative_xyz_half_open": relative9,
            "mob_query_absolute_xyz_half_open": absolute9,
            "candidate_domain": (
                "all_entities_reported_by_current_mobs_query_box"
                if scope_available9 else "none_unverified"),
            "mobs_packet_present": bool(packet_present9),
            "scope_available": bool(scope_available9),
            "all_reported_target_class_instances_enumerated": bool(
                scope_available9),
            "renderer_entity_ids_available": False,
            "nominal_header_box_is_not_assumed_per_frame": True,
        }

    def current_structural_mob_occluders9():
        offline9 = STRUCT.get("offline_human_label_row")
        offline_mobs9 = (offline9.get("structural_mob_observation")
                         if isinstance(offline9, dict) else None)
        raw9 = (offline_mobs9.get("candidates_absolute")
                if isinstance(offline_mobs9, dict) else w.info.get("mobs"))
        scope9 = current_mob_visibility_scan_scope9()
        cells9 = set()
        aabbs9 = []
        parse_valid9 = (bool(offline_mobs9.get("packet_parse_valid"))
                        if isinstance(offline_mobs9, dict) else
                        raw9 is not None)
        try:
            reported_count9 = None if raw9 is None else int(len(raw9))
        except (TypeError, ValueError, OverflowError):
            reported_count9 = None
            parse_valid9 = False
        candidates9 = ()
        if parse_valid9:
            try:
                candidates9 = normalize_mob_candidates(
                    raw9 or (), current_label_player_position9(),
                    coordinates=("absolute" if isinstance(offline_mobs9, dict)
                                 else "relative"))
                if len(candidates9) > 512:
                    raise ValueError("mobs packet exceeds conservative audit limit")
                for candidate9 in candidates9:
                    cover9, lo9, hi9 = structural_mob_aabb_voxel_cover(
                        candidate9.kind, candidate9.position)
                    cells9.update(cover9)
                    aabbs9.append({
                        "observation_index": int(candidate9.observation_index),
                        "kind": str(candidate9.kind),
                        "position": [float(q9) for q9 in candidate9.position],
                        "lo": [float(q9) for q9 in lo9],
                        "hi": [float(q9) for q9 in hi9],
                        "outer_voxel_count": int(len(cover9)),
                    })
            except (TypeError, ValueError, OverflowError):
                parse_valid9 = False
                candidates9 = ()
                cells9.clear()
                aabbs9.clear()
        meta9 = {
            "policy": STRUCTURAL_ENTITY_OCCLUSION_POLICY,
            "mobs_packet_present": raw9 is not None,
            "mobs_packet_parse_valid": bool(parse_valid9),
            "mob_query_scope_available": bool(scope9["scope_available"]),
            "mob_query_relative_xyz_half_open": (
                scope9.get("mob_query_relative_xyz_half_open")),
            "mob_query_absolute_xyz_half_open": (
                scope9.get("mob_query_absolute_xyz_half_open")),
            "reported_mob_count": reported_count9,
            "normalized_mob_count": int(len(candidates9)),
            "conservative_aabb_count": int(len(aabbs9)),
            "occluder_voxel_count": int(len(cells9)),
            "conservative_aabbs": aabbs9,
            "query_guard": {
                "horizontal_each_side": float(
                    STRUCTURAL_ENTITY_QUERY_GUARD[0]),
                "lower_y_for_outside_body_top": float(
                    STRUCTURAL_ENTITY_QUERY_GUARD[1]),
                "upper_y_for_outside_body_bottom": float(
                    STRUCTURAL_ENTITY_QUERY_GUARD[2]),
            },
            "same_cell_policy": (
                "entity-overlapped semantic cells are ineligible target pixels"),
            "false_negative_bias": (
                "outer voxel cover may over-occlude; never under-occlude a "
                "reported bounded body"),
        }
        STRUCT["last_structural_entity_occlusion"] = meta9
        return frozenset(cells9), meta9

    def structural_entity_query_covers_instances9(instances9):
        _entity_cells9, meta9 = current_structural_mob_occluders9()
        scope9 = current_mob_visibility_scan_scope9()
        raw_box9 = scope9.get("mob_query_absolute_xyz_half_open")
        if (not meta9["mobs_packet_parse_valid"]
                or not scope9["scope_available"]
                or not isinstance(raw_box9, list) or len(raw_box9) != 6):
            return False, meta9
        hx9, lower_y9, upper_y9, hz9 = STRUCTURAL_ENTITY_QUERY_GUARD
        guarded9 = [
            float(raw_box9[0]) + hx9, float(raw_box9[1]) - hx9,
            float(raw_box9[2]) + lower_y9,
            float(raw_box9[3]) - upper_y9,
            float(raw_box9[4]) + hz9, float(raw_box9[5]) - hz9,
        ]

        def point_inside9(point9):
            return bool(
                isinstance(point9, (tuple, list)) and len(point9) == 3
                and guarded9[0] <= float(point9[0]) < guarded9[1]
                and guarded9[2] <= float(point9[1]) < guarded9[3]
                and guarded9[4] <= float(point9[2]) < guarded9[5])

        def cell_inside9(cell9):
            return bool(
                isinstance(cell9, (tuple, list)) and len(cell9) == 3
                and guarded9[0] <= int(cell9[0])
                and int(cell9[0]) + 1 <= guarded9[1]
                and guarded9[2] <= int(cell9[1])
                and int(cell9[1]) + 1 <= guarded9[3]
                and guarded9[4] <= int(cell9[2])
                and int(cell9[2]) + 1 <= guarded9[5])

        covered9 = point_inside9(
            label_eye_pos9())
        certified_cells9 = []
        certified_points9 = []
        for instance9 in instances9:
            proof9 = instance9.get("proof") if isinstance(instance9, dict) else None
            proof9 = proof9 if isinstance(proof9, dict) else {}
            face_records9 = proof9.get("certified_visible_faces")
            if isinstance(face_records9, list) and face_records9:
                certified_cells9.extend(
                    face9.get("cell") for face9 in face_records9
                    if isinstance(face9, dict))
            member_cells9 = proof9.get("visible_member_cells")
            if isinstance(member_cells9, list):
                certified_cells9.extend(member_cells9)
            if not face_records9 and not member_cells9:
                certified_cells9.append(
                    proof9.get("perceptible_target_cell",
                               instance9.get("world_position")))
            certified_points9.append(instance9.get("visible_point"))
        covered9 = bool(
            covered9
            and all(cell_inside9(cell9) for cell9 in certified_cells9)
            and all(point_inside9(point9) for point9 in certified_points9))
        meta9 = dict(meta9)
        meta9.update({
            "guarded_mob_query_absolute_xyz_half_open": guarded9,
            "certified_target_cell_count": int(len(certified_cells9)),
            "certified_target_point_count": int(len(certified_points9)),
            "certified_target_ray_domain_covered": bool(covered9),
        })
        STRUCT["last_structural_entity_occlusion"] = meta9
        return covered9, meta9

    def stamp_current_structural_fields9(ctx9, *, class_exist, chosen_visible,
                                         visible_point=None, face_normal=None,
                                         visible_instances=None,
                                         committed_surface_instance=None,
                                         confuser_instances=None,
                                         class_census=None,
                                         preserve_same_rgb_aux=False,
                                         source="current_geometry",
                                         visibility_debug=None):
        pending9 = TRAJ.get("pending_row")
        if pending9 is None:
            raise RuntimeError("cannot label structural visibility without pending row")
        rows9 = STORY.get("rows") or []
        current9 = (rows9[-1] if rows9
                    and int(rows9[-1].get("traj_t", -1)) == int(pending9["t"])
                    else None)
        frame_identity9 = structural_rgb_identity(pending9, current9)
        preserved_aux9 = (
            same_rgb_structural_aux(pending9, current9)
            if preserve_same_rgb_aux else None)
        if preserved_aux9 is not None:
            if confuser_instances is None:
                confuser_instances = preserved_aux9["confuser_instances"]
            if class_census is None:
                class_census = preserved_aux9["class_census"]
        chosen9 = bool(chosen_visible and visible_point is not None)
        instances9 = ([] if visible_instances is None else
                      [dict(instance9) for instance9 in visible_instances])
        if (preserve_same_rgb_aux and preserved_aux9 is not None
                and is_mine_target_kind9(ctx9.get("cls"))
                and current9 is not None
                and pending9.get("visible_instances")
                   == current9.get("visible_instances")
                and isinstance(current9.get("visible_instances"), list)):
            instances9 = copy.deepcopy(current9["visible_instances"])
            committed_choice9 = ctx9.get("_committed_instance_id")
            chosen_matches9 = 0
            for instance9 in instances9:
                selected9 = bool(
                    chosen9 and committed_choice9 is not None
                    and str(instance9.get("instance_id"))
                       == str(committed_choice9))
                instance9["chosen"] = selected9
                if selected9:
                    chosen_matches9 += 1
                    proof_point9 = instance9.get("visible_point")
                    proof_normal9 = instance9.get("face_normal")
                    if (isinstance(proof_point9, (tuple, list))
                            and len(proof_point9) == 3):
                        visible_point = list(proof_point9)
                    if (isinstance(proof_normal9, (tuple, list))
                            and len(proof_normal9) == 3):
                        face_normal = list(proof_normal9)
            if chosen9 and chosen_matches9 != 1:
                raise RuntimeError(
                    "same-RGB retarget choice is absent from preserved goal masks")
        if visible_instances is not None:
            class_exist = bool(instances9)
        instance_ids9 = []
        chosen_ids9 = []
        for instance9 in instances9:
            if int(instance9.get("visible", 0)) != 1:
                raise RuntimeError(
                    "visible_instances may contain only current-frame visible records")
            instance_id9 = instance9.get("instance_id")
            if instance_id9 is None or not str(instance_id9):
                raise RuntimeError("visible instance is missing a stable instance_id")
            instance_ids9.append(str(instance_id9))
            if instance9.get("chosen") is True:
                chosen_ids9.append(str(instance_id9))
        if len(instance_ids9) != len(set(instance_ids9)):
            raise RuntimeError("visible instance IDs must be unique within one frame")
        if chosen9:
            if len(chosen_ids9) != 1:
                raise RuntimeError(
                    "chosen_visible=1 requires exactly one chosen visible instance")
            chosen_instance_id9 = chosen_ids9[0]
        else:
            if chosen_ids9:
                raise RuntimeError(
                    "chosen_visible=0 cannot publish a chosen visible instance")
            chosen_instance_id9 = None
        confusers9 = ([] if confuser_instances is None else
                      [dict(instance9) for instance9 in confuser_instances])
        confuser_ids9 = []
        for instance9 in confusers9:
            if int(instance9.get("visible", 0)) != 1:
                raise RuntimeError(
                    "confuser_instances may contain only current-frame visible records")
            instance_id9 = instance9.get("instance_id")
            if instance_id9 is None or not str(instance_id9):
                raise RuntimeError("confuser instance is missing a stable instance_id")
            if instance9.get("chosen") is True:
                raise RuntimeError("a confuser instance can never be the chosen target")
            confuser_ids9.append(str(instance_id9))
        if len(confuser_ids9) != len(set(confuser_ids9)):
            raise RuntimeError("confuser instance IDs must be unique within one frame")
        if set(confuser_ids9) & set(instance_ids9):
            raise RuntimeError(
                "an instance cannot be both goal-class and confuser in one frame")
        census_fields9 = {}
        census_instances9 = []
        if class_census is not None:
            census9 = dict(class_census)
            census_goal9 = G._bare(str(ctx9.get("cls") or ""))
            goal_is_census_class9 = census_goal9 in mine_census_roster9
            SC.validate_census(
                census9,
                goal_kind=(census_goal9 if goal_is_census_class9 else None),
                goal_class_exist=(bool(class_exist)
                                  if goal_is_census_class9 else None),
                roster=mine_census_roster9)
            census_instances9 = [dict(instance9) for instance9 in
                                 census9["class_visible_instances"]]
            census_fields9 = {
                "class_visibility_known_bits": int(
                    census9["class_visibility_known_bits"]),
                "class_visibility_visible_bits": int(
                    census9["class_visibility_visible_bits"]),
                "class_visible_instances": census_instances9,
                "class_union_masks": [
                    dict(record9) for record9 in
                    census9.get("class_union_masks", ())],
                "class_visibility_scan_witness": dict(
                    census9["class_visibility_scan_witness"]),
                "class_census_runtime": dict(census9["class_census_runtime"]),
            }
        label_player9 = current_label_player_position9()
        cur_y9 = float(label_player9[1])
        prev_y9 = TRAJ.get("_last_labeled_y")
        TRAJ["_last_labeled_y"] = cur_y9
        jump9 = 0
        act9 = pending9.get("action")
        if isinstance(act9, dict):
            try:
                jump9 = int(np.asarray(act9.get("jump", 0)).reshape(-1)[0])
            except (TypeError, ValueError):
                jump9 = 0
        airborne9 = bool(jump9) or (
            prev_y9 is not None and abs(cur_y9 - prev_y9) > 0.05)
        pending_phase9 = int(pending9.get("phase", TRAJ.get("phase", 0)))
        committed_active9 = bool(
            pending_phase9 in (1, 2)
            or source == "retarget_current_frame_hit")
        if is_mine_target_kind9(ctx9.get("cls")):
            committed_active9 = bool(
                committed_active9
                and ctx9.get("_committed_instance_id") is not None)
        committed_id9 = None
        committed_cell9 = None
        if committed_active9:
            committed_id9 = ctx9.get("_committed_instance_id")
            semantic_raw9 = (
                ctx9.get("_semantic_tx", ctx9.get("tx")),
                ctx9.get("_semantic_ty", ctx9.get("ty")),
                ctx9.get("_semantic_tz", ctx9.get("tz")))
            if all(q9 is not None for q9 in semantic_raw9):
                committed_cell9 = tuple(int(q9) for q9 in semantic_raw9)
            if (committed_id9 is None and ctx9.get("cls")
                    and committed_cell9 is not None):
                committed_id9 = "block:%d:%d:%d" % committed_cell9
            if committed_id9 is None and chosen_instance_id9 is not None:
                committed_id9 = chosen_instance_id9
        committed_active9 = bool(
            committed_active9 and committed_id9 is not None)
        committed_surface9 = (
            None if committed_surface_instance is None else
            copy.deepcopy(dict(committed_surface_instance)))
        if committed_surface9 is not None:
            if not committed_active9:
                raise RuntimeError(
                    "committed_surface_instance requires active commit memory")
            surface_id9 = committed_surface9.get("instance_id")
            if surface_id9 is None or str(surface_id9) != str(committed_id9):
                raise RuntimeError(
                    "committed surface identity must equal committed_instance_id")
            if int(committed_surface9.get("visible", 0)) != 1:
                raise RuntimeError(
                    "committed_surface_instance must describe current RGB pixels")
            if committed_surface9.get("chosen") is True:
                raise RuntimeError(
                    "below-recognition committed surface must not enter chosen list")
            surface_proof9 = committed_surface9.get("proof")
            surface_runs9 = (None if not isinstance(surface_proof9, dict) else
                             surface_proof9.get("certified_pixel_runs_yx"))
            if not surface_runs9:
                raise RuntimeError(
                    "committed surface record requires non-empty certified pixels")
            if surface_proof9.get("oracle_recognizable") is True:
                raise RuntimeError(
                    "recognizable committed surface belongs in visible_instances")
            if chosen9:
                raise RuntimeError(
                    "recognizable chosen record and below-threshold surface overlap")
        chosen_surface_visible9 = bool(chosen9 or committed_surface9 is not None)
        chosen_surface_source9 = (
            "recognizable_visible_instances" if chosen9 else
            "committed_exact_surface_below_recognition"
            if committed_surface9 is not None else "none")
        chosen_distance9 = -1.0
        if (committed_active9 and committed_cell9 is not None
                and not int((TRAJ.get("pending_row") or {}).get(
                    "post_break", TRAJ.get("post_break", 0)))):
            ex9 = float(label_player9[0])
            ey9 = float(label_player9[1]) + 1.62
            ez9 = float(label_player9[2])
            bx9, by9, bz9 = (float(committed_cell9[0]),
                             float(committed_cell9[1]),
                             float(committed_cell9[2]))
            ddx9 = max(bx9 - ex9, 0.0, ex9 - (bx9 + 1.0))
            ddy9 = max(by9 - ey9, 0.0, ey9 - (by9 + 1.0))
            ddz9 = max(bz9 - ez9, 0.0, ez9 - (bz9 + 1.0))
            chosen_distance9 = round(
                float(math.sqrt(ddx9 * ddx9 + ddy9 * ddy9 + ddz9 * ddz9)), 4)
        fields9 = {
            "visibility_airborne_hold": int(airborne9),
            "chosen_distance": chosen_distance9,
            "class_exist": int(bool(class_exist)),
            "class_recognizable": int(bool(class_exist)),
            "visible_instances": instances9,
            "confuser_instances": confusers9,
            "confuser_exist": int(bool(confusers9)),
            "confuser_classes_visible": sorted(
                {str(instance9.get("kind")) for instance9 in confusers9}),
            "chosen_visible": int(chosen9),
            "chosen_instance_id": chosen_instance_id9,
            "chosen_surface_visible": int(chosen_surface_visible9),
            "chosen_surface_instance_id": (
                str(committed_id9 if committed_active9 else
                    chosen_instance_id9)
                if chosen_surface_visible9 else None),
            "chosen_surface_source": chosen_surface_source9,
            "committed_surface_instance": committed_surface9,
            "target_committed": int(committed_active9),
            "committed_instance_id": (
                str(committed_id9) if committed_active9 else None),
            "committed_target_cell": (
                [int(q9) for q9 in committed_cell9]
                if committed_active9 and committed_cell9 is not None else None),
            "visible_point": ([float(q9) for q9 in visible_point]
                              if chosen9 else None),
            "face_normal": ([int(q9) for q9 in face_normal]
                            if chosen9 and face_normal is not None else None),
            "structural_visibility_source": str(source),
        }
        if visibility_debug is not None:
            fields9["visibility_debug"] = visibility_debug
        if frame_identity9 is not None:
            fields9[FRAME_IDENTITY_FIELD] = frame_identity9
        fields9["structural_aux_merge_status"] = (
            "same_rgb_preserved" if preserved_aux9 is not None else
            "fail_closed_no_same_rgb_evidence" if preserve_same_rgb_aux else
            "not_requested")
        fields9.update(census_fields9)
        occ_scope9 = ctx9.get("_struct_occ")
        boxes9 = ([] if occ_scope9 is None else
                  [[int(q9) for q9 in box9]
                   for box9 in getattr(occ_scope9, "boxes", ())])
        nominal_radius9 = 40 if ctx9.get("cls") == "water" else 24
        nominal_vertical9 = (
            18 if is_mine_target_kind9(ctx9.get("cls")) else 6)
        px9, py9, pz9 = (int(math.floor(float(q9))) for q9 in label_player9)
        nominal_corners9 = [
            (px9 + sx9 * nominal_radius9,
             py9 + sy9 * nominal_vertical9,
             pz9 + sz9 * nominal_radius9)
            for sx9 in (-1, 1) for sy9 in (-1, 1) for sz9 in (-1, 1)]
        current_nominal_complete9 = bool(
            occ_scope9 is not None
            and all(occ_scope9.known(q9) for q9 in nominal_corners9))
        fields9["visibility_scan_scope"] = {
            "version": 1,
            "coordinate_system": "absolute_xyz_half_open",
            "known_voxel_boxes": boxes9,
            "candidate_domain": "union_of_known_voxel_boxes",
            "player_position": [float(q9) for q9 in label_player9],
            "nominal_header_radius_is_not_assumed_per_frame": True,
            "nominal_current_centered_coverage_complete": (
                current_nominal_complete9),
            "nominal_horizontal_radius": int(nominal_radius9),
            "nominal_relative_y_radius": int(nominal_vertical9),
        }
        entity_scope_complete9, entity_scope9 = (
            structural_entity_query_covers_instances9(
                census_instances9 if census_instances9 else instances9))
        goal_visibility_known9 = True
        if class_census is not None:
            goal_visibility_known9, _goal_visible9 = SC.class_state(
                G._bare(str(ctx9.get("cls") or "")),
                census_fields9["class_visibility_known_bits"],
                census_fields9["class_visibility_visible_bits"],
                roster=mine_census_roster9)
        fields9["visibility_supervision_valid"] = int(
            entity_scope_complete9 and bool(goal_visibility_known9))
        if class_census is not None:
            fields9["goal_class_visibility_known"] = int(
                goal_visibility_known9)
        fields9["structural_entity_occlusion_scope"] = entity_scope9
        if class_census is None:
            for row9 in (pending9, current9):
                if row9 is None:
                    continue
                for field9 in (*CLASS_CENSUS_FIELDS,
                               "goal_class_visibility_known"):
                    row9.pop(field9, None)
        if visibility_debug is None:
            for row9 in (pending9, current9):
                if row9 is not None:
                    row9.pop("visibility_debug", None)
        if frame_identity9 is None:
            for row9 in (pending9, current9):
                if row9 is not None:
                    row9.pop(FRAME_IDENTITY_FIELD, None)
        pending9.update(fields9)
        if current9 is not None:
            current9.update(fields9)
        else:
            old9 = STORY.get("pending_struct_fields") or {}
            old9.update(fields9)
            STORY["pending_struct_fields"] = old9
        ctx9["_class_exist"] = bool(class_exist)
        ctx9["_class_recognizable"] = bool(class_exist)
        ctx9["_chosen_visible"] = chosen9
        ctx9["_chosen_surface_visible"] = chosen_surface_visible9
        ctx9["_visible_point"] = fields9["visible_point"]
        ctx9["_face_normal"] = fields9["face_normal"]

    def block_visible_instance9(occ9, cell9, point9, normal9, proof9=None,
                                *, instance_id9=None):
        cell9 = tuple(int(q9) for q9 in cell9)
        proof9 = proof9 if isinstance(proof9, dict) else {}
        bbox9 = (proof9.get("certified_visible_bbox_px")
                 or proof9.get("visible_bbox_px") or ())
        return {
            "instance_id": (str(instance_id9) if instance_id9 is not None else
                            f"block:{cell9[0]}:{cell9[1]}:{cell9[2]}"),
            "visible": 1,
            "chosen": False,
            "identity_scope": "world_block",
            "type": G._bare(str(occ9.type_at(cell9) or "")),
            "world_position": [int(q9) for q9 in cell9],
            "visible_point": [float(q9) for q9 in point9],
            "face_normal": ([int(q9) for q9 in normal9]
                            if normal9 is not None else None),
            "screen_evidence": {
                "visible_area_px": int(proof9.get("visible_area_px", 0)),
                "visible_short_side_px": int(
                    proof9.get("visible_short_side_px", 0)),
                "visible_bbox_px": ([int(q9) for q9 in bbox9]
                                    if isinstance(bbox9, (tuple, list)) else []),
                "visibility_measure": str(
                    proof9.get("visibility_measure", "")),
            },
            "proof": dict(proof9),
        }

    def _logged_step(a):
        sensor_only9 = bool(a.pop("_xbench_sensor", False))
        human_control9 = bool(a.pop("_xbench_human_control", False))
        try:
            voxel_box9 = np.asarray(a.get("voxels"), dtype=np.int32).reshape(-1)
            STRUCT["last_voxel_box"] = (
                voxel_box9.tolist() if voxel_box9.size == 6 and (voxel_box9 != 0).any()
                else None)
        except Exception:
            STRUCT["last_voxel_box"] = None
        if STORY["on"]:
            try:
                q9 = np.asarray(a.get("mobs")).reshape(-1)
                realq = q9.size == 6 and (q9 != 0).any()
            except Exception:
                realq = False
            if not realq:
                a["mobs"] = np.array(STRUCTURAL_MOB_QUERY_BOX, np.int32)
        try:
            effective_mob_box9 = np.asarray(
                a.get("mobs"), dtype=np.int32).reshape(-1)
            STRUCT["last_mob_box"] = (
                effective_mob_box9.tolist()
                if effective_mob_box9.size == 6
                and (effective_mob_box9 != 0).any() else None)
        except Exception:
            STRUCT["last_mob_box"] = None
        if TRAJ["fh"] is not None:
            if not sensor_only9 and not human_control9:
                raise RuntimeError(
                    "recorded control step lacks the human-control marker")
            sneak9 = (int(np.asarray(a.get("sneak", 0)).reshape(-1)[0])
                      if "sneak" in a else 0)
            if sneak9:
                raise RuntimeError("zero-sneak dataset contract violated")
        viewmodel_valid9 = None
        viewmodel_audit9 = None
        if TRAJ["fh"] is not None and args.visible_surface_masks:
            viewmodel_action9 = dict(a)
            viewmodel_action9[RENDERER_VIEWMODEL_ACTION_MARKER] = 1
            (viewmodel_valid9, viewmodel_audit9,
             TRAJ["viewmodel_remaining_controls"]) = viewmodel_supervision_step(
                 viewmodel_action9, "sensor" if sensor_only9 else "control",
                 int(TRAJ.get("viewmodel_remaining_controls", 0)))
        pre_render_position9 = (
            tuple(float(q9) for q9 in w.get_pos())
            if args.visible_surface_masks else None)
        out = _orig_sim_step(a)
        renderer_viewmodel_mask9 = None
        renderer_entity_ids9 = None
        if args.visible_surface_masks:
            renderer_viewmodel_mask9 = out[4].get(RENDERER_VIEWMODEL_INFO_FIELD)
            if (not isinstance(renderer_viewmodel_mask9, np.ndarray)
                    or renderer_viewmodel_mask9.dtype != np.uint8
                    or renderer_viewmodel_mask9.shape != (G.H_PX, G.W_PX)
                    or np.any((renderer_viewmodel_mask9 != 0)
                              & (renderer_viewmodel_mask9 != 1))):
                raise RuntimeError(
                    "exact-label RGB lacks its renderer viewmodel mask")
            renderer_entity_ids9 = out[4].get(RENDERER_ENTITY_ID_FIELD)
            renderer_entity_ids_valid9 = out[4].get(
                RENDERER_ENTITY_ID_VALID_FIELD)
            if (renderer_entity_ids_valid9 != 1
                    or not isinstance(renderer_entity_ids9, np.ndarray)
                    or renderer_entity_ids9.dtype != np.uint32
                    or renderer_entity_ids9.shape != (G.H_PX, G.W_PX)):
                raise RuntimeError(
                    "exact-label RGB lacks its renderer entity-ID image")
            if STRUCT.get("defer_human_labels"):
                depth_field9 = out[4].get(RENDERER_VIEWMODEL_DEPTH_FIELD)
                if (isinstance(depth_field9, np.ndarray)
                        and depth_field9.shape == (G.H_PX, G.W_PX)):
                    STRUCT.setdefault("human_frame_depths9", {})[
                        int(TRAJ["t"])] = np.ascontiguousarray(
                            depth_field9, dtype=np.float32)
            if STRUCT.get("defer_human_labels"):
                STRUCT.setdefault("human_frame_entity_surfaces9", {})[
                    int(TRAJ["t"])] = [
                        surface9.record()
                        for surface9 in entity_surfaces(renderer_entity_ids9)]
        render_pose9 = None
        if args.visible_surface_masks:
            w.info = out[4]
            render_pose9 = {
                "x": pre_render_position9[0],
                "y": pre_render_position9[1],
                "z": pre_render_position9[2],
                "yaw": float(w.get_yaw()),
                "pitch": float(w.get_pitch()),
                "eye_height": 1.62,
                "source": VISIBLE_SURFACE_RENDER_POSE_SOURCE,
            }
            renderer_frame_seq9 = int(
                STRUCT.get("current_renderer_frame_seq9", -1)) + 1
            STRUCT["current_renderer_frame_seq9"] = renderer_frame_seq9
            STRUCT["current_renderer_observation9"] = {
                "frame_seq": renderer_frame_seq9,
                "render_pose": dict(render_pose9),
                "renderer_viewmodel_mask": np.ascontiguousarray(
                    renderer_viewmodel_mask9, dtype=np.uint8).copy(),
                "renderer_entity_ids": np.ascontiguousarray(
                    renderer_entity_ids9, dtype=np.uint32).copy(),
            }
        if STORY["on"]:
            w.info = out[4]
            axS, ayS, azS = w.get_pos()
            cur = []
            for m in (out[4].get("mobs") or []):
                try:
                    sm = str(m).lower()
                    kd = ("sheep" if "sheep" in sm else "zombie" if "zombie" in sm
                          else "cow" if "cow" in sm
                          else "spider" if "spider" in sm else "mob")
                    cur.append([kd, round(axS + float(m["x"]), 2), round(ayS + float(m["y"]), 2),
                                round(azS + float(m["z"]), 2)])
                except Exception:
                    pass
            STORY["mobs"] = cur
            if args.human_play:
                packet9 = out[4].get("mobs")
                candidates_abs9 = []
                parse_valid9 = packet9 is not None
                if parse_valid9:
                    try:
                        for candidate9 in normalize_mob_candidates(
                                packet9 or (), (axS, ayS, azS)):
                            candidates_abs9.append({
                                "name": str(candidate9.kind),
                                "x": float(candidate9.position[0]),
                                "y": float(candidate9.position[1]),
                                "z": float(candidate9.position[2]),
                                "observation_index": int(
                                    candidate9.observation_index),
                            })
                    except (TypeError, ValueError, OverflowError):
                        parse_valid9 = False
                        candidates_abs9 = []
                STRUCT["current_human_mob_observation"] = {
                    "packet_present": bool(packet9 is not None),
                    "packet_parse_valid": bool(parse_valid9),
                    "query_box_relative": (
                        list(STRUCT["last_mob_box"])
                        if isinstance(STRUCT.get("last_mob_box"), list) else None),
                    "candidates_absolute": candidates_abs9,
                }
                voxel_packet9 = out[4].get("voxels")
                voxel_box_observed9 = STRUCT.get("last_voxel_box")
                voxel_rows9 = []
                voxel_packet_present9 = bool(
                    voxel_packet9 is not None
                    and isinstance(voxel_box_observed9, list)
                    and len(voxel_box_observed9) == 6)
                voxel_parse_valid9 = voxel_packet_present9
                if voxel_parse_valid9:
                    try:
                        for raw_voxel9 in voxel_packet9 or ():
                            voxel_rows9.append({
                                "x": int(raw_voxel9["x"]),
                                "y": int(raw_voxel9["y"]),
                                "z": int(raw_voxel9["z"]),
                                "type": str(raw_voxel9["type"]),
                            })
                    except (KeyError, TypeError, ValueError, OverflowError):
                        voxel_parse_valid9 = False
                        voxel_rows9 = []
                STRUCT["current_human_voxel_observation"] = {
                    "packet_present": bool(voxel_packet_present9),
                    "packet_parse_valid": bool(voxel_parse_valid9),
                    "query_box_relative": (
                        list(voxel_box_observed9)
                        if isinstance(voxel_box_observed9, list) else None),
                    "player_position": [float(axS), float(ayS), float(azS)],
                    "blocks_relative": voxel_rows9,
                }
        fh = TRAJ["fh"]
        if fh is not None:
            w.info = out[4]
            act = {}
            for k, v in a.items():
                if k in ("voxels", "mobs"):
                    continue
                if k == "camera":
                    vv = np.asarray(v, dtype=float).reshape(-1)
                    act["camera"] = [float(vv[0]) if vv.size >= 1 else 0.0,
                                     float(vv[1]) if vv.size >= 2 else 0.0]
                elif k == "chat":
                    act["chat"] = 1 if v else 0
                else:
                    try:
                        act[k] = int(np.asarray(v).reshape(-1)[0])
                    except Exception:
                        pass
            if args.visible_surface_masks:
                act[RENDERER_VIEWMODEL_ACTION_MARKER] = 1
            ax9, ay9, az9 = w.get_pos()
            phase9 = int(TRAJ["phase"])
            event9 = "" if STORY["on"] else str(TRAJ["event"])
            bc_valid9 = 0 if sensor_only9 else int(TRAJ["bc_valid"])
            row9 = {"t": TRAJ["t"], "action": act,
                    "x": round(ax9, 2), "y": round(ay9, 2), "z": round(az9, 2),
                    "yaw": round(w.get_yaw(), 2), "pitch": round(w.get_pitch(), 2),
                    "note": TRAJ["note"], "phase": phase9,
                    "decision_event": event9, "bc_valid": bc_valid9,
                    "step_kind": "sensor" if sensor_only9 else "control",
                    "motion_phase": str(TRAJ.get("motion_phase", "explore")),
                    "post_break": int(TRAJ.get("post_break", 0)),
                    "drop_collect": int(TRAJ.get("drop_collect", 0)),
                    "drop_collect_stage": str(
                        TRAJ.get("drop_collect_stage", "none")),
                    "pickup_expected_items": list(
                        TRAJ.get("pickup_expected_items") or ()),
                    "pickup_inventory_delta": dict(
                        TRAJ.get("pickup_inventory_delta") or {}),
                    "pickup_counter_delta": dict(
                        TRAJ.get("pickup_counter_delta") or {}),
                    "pickup_confirmed": int(
                        TRAJ.get("pickup_confirmed", 0)),
                    "pickup_latency_controls": int(
                        TRAJ.get("pickup_latency_controls", -1)),
                    "pickup_abandoned": int(
                        TRAJ.get("pickup_abandoned", 0)),
                    "pickup_abandon_reason": str(
                        TRAJ.get("pickup_abandon_reason", "none")),
                    "pickup_attempt_controls": int(
                        TRAJ.get("pickup_attempt_controls", -1)),
                    "pickup_attribution_valid": int(
                        TRAJ.get("pickup_attribution_valid", 0)),
                    "pickup_navigation_source": str(
                        TRAJ.get("pickup_navigation_source", "none")),
                    "pickup_visible_preempt_proof": copy.deepcopy(
                        TRAJ.get("pickup_visible_preempt_proof")),
                    "mine_attack_distance_threshold": float(
                        TRAJ.get("mine_attack_distance_threshold", -1.0)),
                    "stuck_recovery_demo": int(
                        TRAJ.get("stuck_recovery_demo", 0)),
                    "stuck_recovery_stage": str(
                        TRAJ.get("stuck_recovery_stage", "none"))}
            if human_control9:
                row9["human_control"] = 1
                evidence9 = TRAJ.get("human_preaction_evidence")
                if not isinstance(evidence9, dict):
                    raise RuntimeError(
                        "human control lacks atomically captured pre-action evidence")
                row9["human_preaction"] = dict(evidence9)
                TRAJ["human_preaction_evidence"] = None
            if args.visible_surface_masks:
                row9["render_pose"] = dict(render_pose9)
                row9[VIEWMODEL_PIXEL_SUPERVISION_FIELD] = int(
                    viewmodel_valid9)
                row9[VIEWMODEL_OCCLUSION_AUDIT_FIELD] = dict(
                    viewmodel_audit9)
            if args.visible_surface_masks:
                row9.update(renderer_mask_row_fields(
                    renderer_viewmodel_mask9))
            if TRAJ["pending_row"] is not None:
                buffer_or_write_row(
                    TRAJ["pending_row"],
                    deferred=bool(STRUCT.get("defer_human_labels")),
                    start_t=STRUCT.get("defer_human_start_t"),
                    buffer=TRAJ["deferred_human_rows"],
                    write=lambda buffered9: fh.write(
                        json.dumps(buffered9) + "\n"))
            TRAJ["pending_row"] = row9
            TRAJ["last_phase"], TRAJ["last_event"] = phase9, event9
            TRAJ["last_bc_valid"] = bc_valid9
            TRAJ["event"] = ""
            TRAJ["t"] += 1
        return out

    w.sim.step = _logged_step
    S = Stager(w)
    S.mine_class_pool = tuple(mine_stage_pool9)
    S.mine_scrub_classes = tuple(SC.CENSUS_SCRUB_CLASSES)
    S.mine_worldgen_profile = str(args.mine_worldgen_profile)
    if args.human_play:
        S.mine_chain_count = 3 if v12_multiclass6_mode9 else 6
    elif args.benchmark_selected_classswap:
        S.mine_chain_count = 1
    elif args.solo_episode and args.quota is not None:
        S.mine_chain_count = max(3, int(args.quota))
    if args.solo_episode:
        S.plan_mode = "solo"
    timing_prep_started9 = time.monotonic()
    if snapshot_restore9 is None:
        S.prep_world()
    else:
        restored_origin9 = snapshot_restore9.get("world_origin")
        if (not isinstance(restored_origin9, (tuple, list))
                or len(restored_origin9) != 3):
            raise RuntimeError("snapshot staged state lacks world_origin")
        S.world_origin = tuple(float(v9) for v9 in restored_origin9)
    timing_ready9 = time.monotonic()
    runtime_timing9["prep_world_wall_s"] = runtime_wall_seconds(
        timing_prep_started9, timing_ready9)
    runtime_timing9["main_to_ready_wall_s"] = runtime_wall_seconds(
        timing_main_started9, timing_ready9)
    ready_timing9 = {
        key9: runtime_timing9[key9]
        for key9 in (
            "schema_version", "clock", "units", "boot_world_wall_s",
            "prep_world_wall_s", "main_to_ready_wall_s")
    }
    print(
        f"[timing] ready boot_world={runtime_timing9['boot_world_wall_s']:.3f}s "
        f"prep_world={runtime_timing9['prep_world_wall_s']:.3f}s "
        f"main_to_ready={runtime_timing9['main_to_ready_wall_s']:.3f}s",
        flush=True)
    if args.ready_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.ready_file)), exist_ok=True)
        with open(args.ready_file, "w") as ready_f9:
            ready_f9.write(json.dumps({
                "pid": os.getpid(),
                "world_seed": args.world_seed,
                "biome": args.biome,
                "world_origin": {
                    "x": float(S.world_origin[0]),
                    "y": float(S.world_origin[1]),
                    "z": float(S.world_origin[2]),
                    "coordinate_frame": "minecraft_absolute",
                    "capture": "post_boot_pre_free_site",
                },
                "runtime_timing": ready_timing9,
            }) + "\n")
    stage, start_pose, fade = S.stage, S.start_pose, S.fade

    def Rec(tag, thresh=25.0):
        if args.human_play:
            from attacca.evaluation.mine_human import HumanRawPOVRec
            return HumanRawPOVRec(w, OUT, tag, thresh)
        return CleanPOVRec(
            w, OUT, tag, thresh,
            allow_pose_jumps=bool(args.policy_driver))

    def render_label_pose9():
        if args.policy_driver:
            observation9 = STRUCT.get("current_renderer_observation9")
            pose9 = (observation9.get("render_pose")
                     if isinstance(observation9, dict) else None)
        else:
            pending9 = TRAJ.get("pending_row")
            pose9 = None if pending9 is None else pending9.get("render_pose")
        required9 = ("x", "y", "z", "yaw", "pitch", "eye_height", "source")
        if not isinstance(pose9, dict) or any(key9 not in pose9 for key9 in required9):
            raise RuntimeError(
                "dense visible-surface label has no atomically matched render pose")
        values9 = tuple(float(pose9[key9]) for key9 in required9[:-1])
        if not all(math.isfinite(value9) for value9 in values9):
            raise RuntimeError("dense visible-surface render pose is non-finite")
        return values9 + (str(pose9["source"]),)

    def label_eye_pos9():
        x9, y9, z9, _yaw9, _pitch9, eye_height9, _source9 = render_label_pose9()
        return (x9, y9 + eye_height9, z9)

    def label_camera9():
        x9, y9, z9, yaw9, pitch9, eye_height9, source9 = render_label_pose9()
        return ((x9, y9 + eye_height9, z9), yaw9, pitch9, source9)

    def structural_pixel_exclusion9(*, relax_human_swing=False):
        excluded9 = G.native_ui_pixel_exclusion_mask()
        if args.policy_driver:
            observation9 = STRUCT.get("current_renderer_observation9")
            renderer_mask9 = (observation9.get("renderer_viewmodel_mask")
                              if isinstance(observation9, dict) else None)
            if (not isinstance(renderer_mask9, np.ndarray)
                    or renderer_mask9.dtype != np.uint8
                    or renderer_mask9.shape != excluded9.shape
                    or np.any((renderer_mask9 != 0) & (renderer_mask9 != 1))):
                raise RuntimeError(
                    "policy exact raster lacks its current renderer viewmodel mask")
            x09, y09, x19, y19 = G.NATIVE_UI_STATIC_RECTS_XYXY[
                "held_item_conservative"]
            excluded9[y09:y19, x09:x19] = 0
            excluded9 |= renderer_mask9
            return excluded9
        pending9 = TRAJ.get("pending_row")
        action9 = (pending9.get("action")
                   if isinstance(pending9, dict) else None)
        renderer_masked9 = bool(
            isinstance(action9, dict)
            and int(action9.get(RENDERER_VIEWMODEL_ACTION_MARKER, 0)) == 1)
        if not renderer_masked9 and not relax_human_swing:
            return excluded9
        if not isinstance(pending9, dict):
            raise RuntimeError(
                "exact raster has no matching pending trajectory row")
        if (not renderer_masked9
                or int(pending9.get(
                    VIEWMODEL_PIXEL_SUPERVISION_FIELD, 0)) != 1):
            raise RuntimeError(
                "exact raster lacks renderer-masked supervision stamp")
        x09, y09, x19, y19 = G.NATIVE_UI_STATIC_RECTS_XYXY[
            "held_item_conservative"]
        excluded9[y09:y19, x09:x19] = 0
        renderer_mask9 = renderer_mask_from_row(pending9)
        if renderer_mask9.shape != excluded9.shape:
            raise RuntimeError("renderer viewmodel mask shape drift")
        excluded9 |= renderer_mask9
        pending9["renderer_viewmodel_exclusion_mode"] = (
            "renderer_exact")
        pending9["renderer_viewmodel_mask_contract"] = (
            RENDERER_VIEWMODEL_MASK_CONTRACT)
        return excluded9

    def dense_surface_proof9(mask9, point9, bbox9, meta9, *,
                             target_cells9, target_cell9=None,
                             selection_shape9="minecraft_full_cube",
                             require_full_cube_recognition9=False):
        if not isinstance(require_full_cube_recognition9, bool):
            raise ValueError("require_full_cube_recognition9 must be boolean")
        mask9 = np.asarray(mask9, dtype=np.uint8)
        if mask9.shape != (G.H_PX, G.W_PX) or not mask9.any():
            return None
        ys9, xs9 = np.nonzero(mask9)
        bbox_calc9 = [int(xs9.min()), int(ys9.min()),
                      int(xs9.max()), int(ys9.max())]
        if bbox9 is None or list(bbox9) != bbox_calc9:
            raise RuntimeError("dense visible-surface bbox disagrees with its pixels")
        semantic9 = str(meta9.get("mask_semantic", ""))
        if semantic9 != G.CHOSEN_VISIBLE_SURFACE_MASK_SEMANTIC:
            raise RuntimeError(f"unexpected dense mask semantic: {semantic9!r}")
        representative_pixel9 = meta9.get("representative_pixel_xy")
        if (not isinstance(representative_pixel9, (tuple, list))
                or len(representative_pixel9) != 2):
            raise RuntimeError("nonempty dense surface has no representative pixel")
        px9, py9 = (int(q9) for q9 in representative_pixel9)
        if not (0 <= px9 < G.W_PX and 0 <= py9 < G.H_PX
                and int(mask9[py9, px9]) == 1):
            raise RuntimeError("dense representative pixel is outside its support")
        normal9 = meta9.get("representative_surface_normal")
        if not isinstance(normal9, (tuple, list)) or len(normal9) != 3:
            raise RuntimeError("dense representative surface has no normal")
        target_cells9 = [list(int(q9) for q9 in cell9)
                         for cell9 in target_cells9]
        if not target_cells9:
            raise RuntimeError("dense surface proof has no target cells")
        chosen_cell9 = (target_cells9[0] if target_cell9 is None else
                        [int(q9) for q9 in target_cell9])
        area9 = int(mask9.sum())
        short9 = min(bbox_calc9[2] - bbox_calc9[0] + 1,
                     bbox_calc9[3] - bbox_calc9[1] + 1)
        _eye9, _yaw9, _pitch9, render_source9 = label_camera9()
        return {
            **structural_surface_recognition_fields(
                area9, short9, visible_bbox_px=bbox_calc9,
                require_full_cube_fill=require_full_cube_recognition9),
            "visibility_measure": "dense_native_visible_surface_first_hit",
            "visible_area_px": area9,
            "visible_short_side_px": int(short9),
            "visible_bbox_px": bbox_calc9,
            "perceptible_target_cell": chosen_cell9,
            "visible_member_cells": target_cells9,
            "visible_point": [float(q9) for q9 in point9],
            "face_normal": [int(q9) for q9 in normal9],
            "aim_pixel": [px9, py9],
            "sample_pixel": [px9, py9],
            "certified_mask_shape": [G.H_PX, G.W_PX],
            "certified_pixel_runs_yx": G.certified_pixel_runs_yx(mask9),
            "certified_pixel_count": area9,
            "certified_mask_semantics": semantic9,
            "selection_shape": str(selection_shape9),
            "render_pose_source": render_source9,
            "occlusion_contract": str(meta9.get("occlusion_contract", "")),
            "projection_contract": str(meta9.get("projection_contract", "")),
            "tested_ray_count": int(meta9.get("tested_ray_count", 0)),
            "dense_segmentation_supervision": True,
            "unknown_terrain_is_solid": bool(meta9.get(
                "unknown_is_solid", True)),
        }

    def dense_full_cube_surface9(
            occ9, cells9, *, visual_extra9=None, alpha_cutout9=None,
            thin_visual9=None, pixel_exclusion9=None,
            require_full_cube_recognition9=False,
            raster_meta9=None):
        eye9, yaw9, pitch9, _source9 = label_camera9()
        cells9 = tuple(tuple(int(q9) for q9 in cell9) for cell9 in cells9)
        alpha_cutout9 = (G.visual_alpha_cutout_cells(occ9, eye9)
                         if alpha_cutout9 is None else alpha_cutout9)
        thin_visual9 = (G.thin_visual_occluders(
                            occ9, eye9,
                            layer_height=G.SNOW_ONE_LAYER_RENDER_HEIGHT)
                        if thin_visual9 is None else thin_visual9)
        pixel_exclusion9 = (structural_pixel_exclusion9(
                                relax_human_swing=bool(args.human_play))
                            if pixel_exclusion9 is None else pixel_exclusion9)
        result9 = G.visible_surface_mask_blocks(
            eye9, yaw9, pitch9, cells9, occ9,
            extra_visual_solid=visual_extra9,
            alpha_cutout_visual=alpha_cutout9,
            thin_visual_solid=thin_visual9,
            pixel_exclusion_mask=pixel_exclusion9)
        mask9, point9, bbox9, meta9 = result9
        if raster_meta9 is not None:
            raster_meta9.update({
                "candidate_pixel_count": int(
                    meta9.get("candidate_pixel_count", 0)),
                "tested_ray_count": int(meta9.get("tested_ray_count", 0)),
                "first_hit_pixel_count": int(meta9.get("pixel_count", 0)),
                "projected_aabb_count": int(
                    meta9.get("projected_aabb_count", 0)),
            })
        if not mask9.any():
            return None
        proof9 = dense_surface_proof9(
            mask9, point9, bbox9, meta9,
            target_cells9=cells9, target_cell9=cells9[0],
            require_full_cube_recognition9=(
                require_full_cube_recognition9))
        return (point9, tuple(proof9["face_normal"]), proof9)

    def scan_occ(occ, span=24, vertical_span=6):
        span = int(span)
        vertical_span = max(1, int(vertical_span))
        edge = 41 if span > 24 else 2 * span + 1
        for xlo in range(-span, span + 1, edge):
            xhi = min(span + 1, xlo + edge)
            for zlo in range(-span, span + 1, edge):
                zhi = min(span + 1, zlo + edge)
                for ylo in range(-vertical_span, vertical_span + 1, 13):
                    yhi = min(vertical_span + 1, ylo + 13)
                    box = [xlo, xhi, ylo, yhi, zlo, zhi]
                    a = w.sim.noop_action()
                    a["_xbench_sensor"] = 1
                    a["voxels"] = np.array(box, np.int32)
                    w.obs, _, _, _, w.info = w.sim.step(a)
                    occ.ingest(w.info.get("voxels") or [], w.get_pos(), box)

    def nearest_visible_cluster(occ, match, max_dist=40.0, admissible=None,
                                admissibility_cache_key=None,
                                audit=None, visible_instances=None,
                                trace_cell=None, rejection_trace=None,
                                tracked_cell=None,
                                tracked_surface_sink=None):
        if audit is not None and not isinstance(audit, dict):
            raise TypeError("nearest-visible audit sink must be a dictionary")
        if rejection_trace is not None and not isinstance(rejection_trace, dict):
            raise TypeError("rejection-trace sink must be a dictionary")
        if (tracked_surface_sink is not None
                and not isinstance(tracked_surface_sink, dict)):
            raise TypeError("tracked-surface sink must be a dictionary")
        tracked_cell9 = (None if tracked_cell is None else
                         tuple(int(q9) for q9 in tracked_cell))
        if tracked_surface_sink is not None:
            tracked_surface_sink.clear()
        tracked_surface_instance9 = None
        trace_cell9 = (None if trace_cell is None else
                       tuple(int(q9) for q9 in trace_cell))
        tracing9 = rejection_trace is not None and trace_cell9 is not None
        if rejection_trace is not None:
            rejection_trace.clear()
            rejection_trace.update({
                "trace_cell": (None if trace_cell9 is None
                               else list(trace_cell9)),
                "stage": "not_evaluated",
                "detector_cache_hit": False,
            })
        if visible_instances is not None and not isinstance(visible_instances, list):
            raise TypeError("visible-instance sink must be a list")
        if visible_instances is not None:
            visible_instances.clear()
        if admissibility_cache_key is not None:
            if admissible is None:
                raise ValueError(
                    "admissibility_cache_key requires an admissible callback")
            try:
                hash(admissibility_cache_key)
            except TypeError as exc:
                raise TypeError("admissibility_cache_key must be hashable") from exc
        cacheable9 = admissible is None or admissibility_cache_key is not None
        detector_started9 = time.perf_counter()
        pending_t9 = (None if TRAJ.get("pending_row") is None else
                      int(TRAJ["pending_row"]["t"]))
        policy_observation_seq9 = None
        if args.policy_driver:
            observation9 = STRUCT.get("current_renderer_observation9")
            if not isinstance(observation9, dict):
                raise RuntimeError(
                    "policy detector has no current renderer observation")
            try:
                policy_observation_seq9 = int(observation9["frame_seq"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    "policy renderer observation has no valid frame sequence") from exc
        label_eye9, label_yaw9, label_pitch9, label_pose_source9 = label_camera9()

        def block_surface_distance9(cell9):
            bx9, by9, bz9 = (float(q9) for q9 in cell9)
            dx9 = max(bx9 - label_eye9[0], 0.0,
                      label_eye9[0] - (bx9 + 1.0))
            dy9 = max(by9 - label_eye9[1], 0.0,
                      label_eye9[1] - (by9 + 1.0))
            dz9 = max(bz9 - label_eye9[2], 0.0,
                      label_eye9[2] - (bz9 + 1.0))
            return math.sqrt(dx9 * dx9 + dy9 * dy9 + dz9 * dz9)

        pose_key9 = tuple(round(float(q9), 5) for q9 in (
            *label_eye9, label_yaw9, label_pitch9)) + (label_pose_source9,)
        cache_frame_scope9 = (
            pending_t9, policy_observation_seq9,
            id(occ), int(occ.revision), pose_key9)
        cache_key9 = (
            pending_t9, policy_observation_seq9,
            id(occ), int(occ.revision),
            str(match), round(float(max_dist), 4), pose_key9,
            admissibility_cache_key,
            tracked_cell9)
        cache9 = (STRUCT.get("visibility_cache") or {}
                  if STRUCT.get("visibility_cache_frame_scope")
                     == cache_frame_scope9 else {})
        cached9 = cache9.get(cache_key9) if cacheable9 else None
        if (cached9 is not None and visible_instances is not None
                and not bool(cached9.get("instances_complete"))):
            cached9 = None
        if cached9 is not None:
            if tracked_surface_sink is not None:
                cached_surface9 = cached9.get("tracked_surface_instance")
                if cached_surface9 is not None:
                    tracked_surface_sink["instance"] = copy.deepcopy(
                        cached_surface9)
            if tracing9:
                cached_trace9 = cached9.get("rejection_trace")
                if (isinstance(cached_trace9, dict)
                        and cached_trace9.get("trace_cell")
                        == list(trace_cell9)):
                    rejection_trace.update(copy.deepcopy(cached_trace9))
                else:
                    rejection_trace["stage"] = (
                        "unavailable_detector_cache_hit_without_trace")
                rejection_trace["detector_cache_hit"] = True
            if visible_instances is not None:
                visible_instances.extend(
                    dict(instance9) for instance9 in cached9["instances"])
            if audit is not None:
                audit.clear()
                audit.update(dict(cached9["audit"]))
                audit["detector_cache_hit"] = True
                audit["detector_runtime_ms"] = round(
                    1000.0 * (time.perf_counter() - detector_started9), 3)
            choice9 = cached9["choice"]
            if choice9 is None:
                return None
            return (*choice9[:5], dict(choice9[5]))
        if audit is not None:
            audit.clear()
            audit.update(match=str(match), success=False,
                         reason="not_evaluated", components=[])
        ax, ay, az = current_label_player_position9()
        eye = label_eye9
        yaw0 = label_yaw9
        pitch0 = label_pitch9
        entity_occluders9, _entity_occlusion_meta9 = (
            current_structural_mob_occluders9())
        bare_match9 = G._bare(str(match))
        if bare_match9 in mine_target_pool9:
            cells = mine_exact_class_cells(
                occ, bare_match9, census_roster=mine_census_roster9)
        else:
            cells = [c for c, t in occ.grid.items() if match in str(t)]
        if tracing9:
            rejection_trace["matching_cell_count"] = int(len(cells))
            in_candidates9 = trace_cell9 in set(cells)
            rejection_trace["trace_cell_in_candidates"] = bool(in_candidates9)
            if not in_candidates9:
                trace_kind9 = occ.grid.get(trace_cell9)
                rejection_trace.update(
                    stage="candidate_enumeration",
                    reason="cell_not_in_class_candidates",
                    cell_known=bool(occ.known(trace_cell9)),
                    cell_kind_in_occupancy=(
                        None if trace_kind9 is None else str(trace_kind9)))
        if not cells:
            if audit is not None:
                audit.update(reason="no_matching_cells", matching_cell_count=0)
            return None
        if audit is not None:
            audit["matching_cell_count"] = int(len(cells))
        mine_full_cube_match9 = bare_match9 in mine_target_pool9
        full_cube_scene9 = None
        full_cube_pixel_exclusion9 = None
        full_cube_scene_requests9 = 0
        full_cube_scene_grid_passes9 = 0
        full_cube_scene_cache_hits9 = 0
        projected_prefilter_rejections9 = 0
        depth_owner_prefilter_rejections9 = 0
        projection_mask_requests9 = 0
        buried_prefilter_rejections9 = 0
        dense_raster_calls9 = 0
        if ((mine_full_cube_match9 or "_log" in match)
                and args.visible_surface_masks):
            full_cube_pixel_exclusion9 = structural_pixel_exclusion9(
                relax_human_swing=bool(args.human_play))

        def ensure_full_cube_scene9():
            nonlocal full_cube_scene9, full_cube_scene_requests9
            nonlocal full_cube_scene_grid_passes9, full_cube_scene_cache_hits9
            if full_cube_scene9 is not None:
                return full_cube_scene9
            scene9 = dict(G.shared_visible_surface_scene(
                occ, eye, cutout_as_full_cube=False))
            if entity_occluders9:
                scene9["visual_extra_solid"] = frozenset(
                    set(scene9["visual_extra_solid"])
                    | set(entity_occluders9))
                scene9["candidate_face_blockers"] = frozenset(
                    set(scene9["candidate_face_blockers"])
                    | set(entity_occluders9))
            full_cube_scene9 = scene9
            full_cube_scene_requests9 += 1
            full_cube_scene_grid_passes9 += int(
                scene9["occupancy_grid_passes"])
            full_cube_scene_cache_hits9 += int(
                scene9["static_scene_cache_hit"])
            return full_cube_scene9
        best = None
        best_rank9 = None
        comps9 = sorted(G.group_instances(cells), key=lambda comp9: min(comp9))
        if audit is not None:
            audit["component_count"] = int(len(comps9))
        if tracing9:
            rejection_trace["component_count"] = int(len(comps9))
        for comp in comps9:

            probe = sorted(
                comp,
                key=lambda c: (block_surface_distance9(c), c))
            for c in probe:
                traced_here9 = tracing9 and c == trace_cell9
                tracked_here9 = tracked_cell9 is not None and c == tracked_cell9
                if c in entity_occluders9:
                    if traced_here9:
                        rejection_trace.update(
                            stage="entity_occlusion",
                            reason="target_cell_reported_entity_occupied")
                    continue
                cx, cy, cz = c
                dxz = math.hypot(cx + 0.5 - ax, cz + 0.5 - az)
                surface_distance9 = block_surface_distance9(c)
                candidate_rank9 = (surface_distance9, cx, cy, cz)
                if (dxz > max_dist
                        or (visible_instances is None and best_rank9 is not None
                            and candidate_rank9 >= best_rank9)):
                    if traced_here9:
                        rejection_trace.update(
                            stage=("distance" if dxz > max_dist
                                   else "dominated_choice_only_scan"),
                            reason=("horizontal_distance_exceeds_limit"
                                    if dxz > max_dist
                                    else "nearer_selectable_already_found"),
                            horizontal_distance=round(float(dxz), 3),
                            max_distance=float(max_dist))
                    continue
                face9 = None
                proof9 = None
                if mine_full_cube_match9 or "_log" in match:
                    if args.visible_surface_masks:
                        owner_pixels9 = G.depth_owner_pixel_indices(
                            eye, yaw0, pitch0, c,
                            pixel_exclusion_mask=full_cube_pixel_exclusion9)
                        if owner_pixels9 is not None and not len(owner_pixels9):
                            depth_owner_prefilter_rejections9 += 1
                            if traced_here9:
                                rejection_trace.update(
                                    stage="depth_owner_prefilter",
                                    reason=("no_native_first_hit_pixel_owned_"
                                            "by_candidate"))
                            continue
                        projection_mask_requests9 += 1
                        projected9 = G.projected_full_cube_candidate_stats(
                            eye, yaw0, pitch0, c,
                            pixel_exclusion_mask=(
                                full_cube_pixel_exclusion9))
                        projected_bbox9 = projected9["projected_bbox_px"]
                        projected_recognizable9 = bool(
                            projected_bbox9 is not None
                            and (structural_full_cube_surface_recognizable(
                                projected9["projected_area_px"],
                                projected9["projected_short_side_px"],
                                projected_bbox9)
                                 if mine_full_cube_match9 else
                                 structural_surface_recognizable(
                                     projected9["projected_area_px"],
                                     projected9["projected_short_side_px"])))
                        projected_has_pixels9 = bool(
                            projected_bbox9 is not None
                            and int(projected9["projected_area_px"]) > 0)
                        if (not projected_recognizable9
                                and not (tracked_here9
                                         and projected_has_pixels9)):
                            projected_prefilter_rejections9 += 1
                            if traced_here9:
                                rejection_trace.update(
                                    stage="projected_footprint_prefilter",
                                    reason=("projected_footprint_below_"
                                            "recognition_floor"),
                                    projected_area_px=int(
                                        projected9["projected_area_px"]),
                                    projected_short_side_px=int(
                                        projected9["projected_short_side_px"]),
                                    projected_bbox_px=(
                                        projected9["projected_bbox_px"]))
                            continue
                        shared_scene9 = ensure_full_cube_scene9()
                        if (not tracked_here9
                                and not SC.has_known_potentially_visible_face(
                                c, known=occ.known, solid=occ.solid_at,
                                visual_extra_solid=shared_scene9[
                                    "candidate_face_blockers"])):
                            buried_prefilter_rejections9 += 1
                            if traced_here9:
                                rejection_trace.update(
                                    stage="buried_prefilter",
                                    reason=("no_known_potentially_visible_"
                                            "face"))
                            continue
                        dense_raster_calls9 += 1
                        trace_raster_meta9 = {} if traced_here9 else None
                        face9 = dense_full_cube_surface9(
                            occ, (c,),
                            visual_extra9=shared_scene9[
                                "visual_extra_solid"],
                            alpha_cutout9=shared_scene9[
                                "alpha_cutout_visual"],
                            thin_visual9=shared_scene9[
                                "thin_visual_solid"],
                            pixel_exclusion9=full_cube_pixel_exclusion9,
                            raster_meta9=trace_raster_meta9,
                            require_full_cube_recognition9=(
                                mine_full_cube_match9))
                    if face9 is None:
                        if traced_here9:
                            if args.visible_surface_masks:
                                rejection_trace.update(
                                    stage="dense_first_hit_mask_empty",
                                    reason=("no_candidate_pixel_first_hits_"
                                            "target"),
                                    **{key9: int(value9) for key9, value9
                                       in (trace_raster_meta9 or {}).items()})
                                center9 = (cx + 0.5, cy + 0.5, cz + 0.5)
                                dv9 = (center9[0] - eye[0],
                                       center9[1] - eye[1],
                                       center9[2] - eye[2])
                                dist9 = max(1e-6, math.sqrt(
                                    dv9[0] ** 2 + dv9[1] ** 2 + dv9[2] ** 2))
                                blocker9, _hit_face9 = (
                                    G.raycast_visual_alpha_cutout(
                                        eye, dv9, dist9 - 1e-6, occ,
                                        ignore=frozenset((c,)),
                                        extra_solid=shared_scene9[
                                            "visual_extra_solid"],
                                        alpha_cutout_cells=shared_scene9[
                                            "alpha_cutout_visual"],
                                        thin_solid=shared_scene9[
                                            "thin_visual_solid"],
                                        unknown_is_solid=True))
                                blocker_kind9 = (
                                    None if blocker9 is None
                                    else occ.grid.get(
                                        tuple(int(q9) for q9 in blocker9)))
                                rejection_trace.update(
                                    center_ray_first_hit_cell=(
                                        None if blocker9 is None else
                                        [int(q9) for q9 in blocker9]),
                                    center_ray_first_hit_kind=(
                                        None if blocker9 is None else
                                        str(blocker_kind9)
                                        if blocker_kind9 is not None else
                                        "unknown_or_visual_extra"))
                        continue
                    sample9, normal9, proof9 = face9
                    if (args.visible_surface_masks
                            and proof9.get("oracle_recognizable") is not True):
                        if (tracked_here9
                                and proof9.get("certified_pixel_runs_yx")):
                            tracked_surface_instance9 = block_visible_instance9(
                                occ, (cx, cy, cz), sample9, normal9, proof9)
                            tracked_surface_instance9[
                                "surface_for_committed_target"] = True
                            tracked_surface_instance9[
                                "recognition_status"] = (
                                    "visible_below_fresh_recognition_gate")
                        if traced_here9:
                            rejection_trace.update(
                                stage="dense_surface_not_recognizable",
                                reason=str(proof9.get(
                                    "perceptibility_contract") or ""),
                                visible_area_px=int(proof9.get(
                                    "visible_area_px", 0)),
                                visible_short_side_px=int(proof9.get(
                                    "visible_short_side_px", 0)),
                                visible_bbox_px=proof9.get("visible_bbox_px"),
                                recognition_threshold_64px_10side=bool(
                                    proof9.get(
                                        "recognition_threshold_64px_10side")),
                                recognition_full_cube_bbox_fill_gate_result=(
                                    str(proof9.get(
                                        "recognition_full_cube_bbox_fill_"
                                        "gate_result", ""))))
                        continue
                    face9 = (sample9, normal9)
                normal9 = None if face9 is None else face9[1]
                navigation_admissible9 = (
                    None if admissible is None else
                    bool(admissible(c, face9)))
                proof9 = dict(proof9 or {})
                if navigation_admissible9 is not None:
                    proof9["navigation_admissible"] = bool(
                        navigation_admissible9)
                if traced_here9:
                    rejection_trace.update(
                        stage="accepted",
                        reason="visible_instance_certified",
                        visible_area_px=(
                            None if proof9 is None
                            or proof9.get("visible_area_px") is None
                            else int(proof9.get("visible_area_px"))))
                if visible_instances is not None:
                    visible_instances.append(block_visible_instance9(
                        occ, (cx, cy, cz), sample9, normal9, proof9))
                choice_out9 = (surface_distance9, (cx, cy, cz), comp, sample9,
                               normal9, proof9)
                selectable9 = navigation_admissible9 is not False
                if selectable9 and (best_rank9 is None
                                    or candidate_rank9 < best_rank9):
                    best = choice_out9
                    best_rank9 = candidate_rank9
                if visible_instances is None and selectable9:
                    break
        if audit is not None:
            audit.update(
                success=best is not None,
                reason="certified" if best is not None else "no_component_certified")
            if best is not None:
                audit["selected_target_cell"] = list(best[1])
                audit["selected_target_face_normal"] = list(best[4])
            audit["detector_cache_hit"] = False
            audit["projected_prefilter_rejection_count"] = int(
                projected_prefilter_rejections9)
            audit["depth_owner_prefilter_rejection_count"] = int(
                depth_owner_prefilter_rejections9)
            audit["projection_mask_request_count"] = int(
                projection_mask_requests9)
            audit["buried_prefilter_rejection_count"] = int(
                buried_prefilter_rejections9)
            audit["dense_raster_call_count"] = int(dense_raster_calls9)
            audit["static_scene_request_count"] = int(
                full_cube_scene_requests9)
            audit["static_scene_grid_passes"] = int(
                full_cube_scene_grid_passes9)
            audit["static_scene_cache_hit"] = int(
                full_cube_scene_cache_hits9)
            audit["detector_runtime_ms"] = round(
                1000.0 * (time.perf_counter() - detector_started9), 3)
        if tracing9 and rejection_trace.get("stage") == "not_evaluated":
            rejection_trace["stage"] = "not_reached_choice_only_scan"
        if cacheable9:
            cached_choice9 = (None if best is None else
                              (*best[:5], dict(best[5])))
            cache_out9 = dict(cache9)
            cache_out9[cache_key9] = {
                "choice": cached_choice9,
                "instances": [dict(instance9) for instance9 in
                              (visible_instances or ())],
                "instances_complete": bool(visible_instances is not None),
                "audit": (dict(audit) if audit is not None else {}),
                "rejection_trace": (copy.deepcopy(rejection_trace)
                                    if tracing9 else None),
                "tracked_surface_instance": (
                    None if tracked_surface_instance9 is None else
                    copy.deepcopy(tracked_surface_instance9)),
            }
            STRUCT["visibility_cache"] = cache_out9
            STRUCT["visibility_cache_frame_scope"] = cache_frame_scope9
        if (tracked_surface_sink is not None
                and tracked_surface_instance9 is not None):
            tracked_surface_sink["instance"] = copy.deepcopy(
                tracked_surface_instance9)
        return best

    def visible_mine_admissibility_cache_key9(occ9):
        px9, py9, pz9 = (float(q9) for q9 in w.get_pos())
        return (
            "mine_current_occupancy_reachable_stance/v1",
            id(occ9.grid), int(occ9.revision),
            int(math.floor(px9)), int(math.floor(py9)),
            int(math.floor(pz9)), round(py9, 3))

    def screened_mine_preferred_feet9(ctx9, cell9):
        try:
            target9 = tuple(int(q9) for q9 in cell9)
        except (TypeError, ValueError, OverflowError):
            return None
        if len(target9) != 3:
            return None
        goal_kind9 = G._bare(str(ctx9.get("cls") or ""))
        matches9 = []
        for meta9 in (ctx9.get("mine_sites") or ()):
            try:
                meta_cell9 = tuple(int(meta9[key9])
                                   for key9 in ("x", "y", "z"))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if (meta_cell9 != target9
                    or G._bare(str(meta9.get("kind") or "")) != goal_kind9):
                continue
            proof9 = meta9.get("start_perceptibility")
            access9 = meta9.get("access_pos")
            try:
                proof_short9 = int(proof9.get("visible_short_side_px", 0))
            except (AttributeError, TypeError, ValueError, OverflowError):
                proof_short9 = 0
            if (not isinstance(proof9, dict)
                    or proof9.get("perceptible") is not True
                    or proof_short9 < 12
                    or not isinstance(access9, (tuple, list))
                    or len(access9) != 3):
                continue
            try:
                feet9 = tuple(int(q9) for q9 in access9)
                if not all(float(access9[i9]) == float(feet9[i9])
                           for i9 in range(3)):
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            matches9.append(feet9)
        if len(matches9) != 1:
            return None
        return matches9[0]

    def reachable_visible_mine_stance9(ctx9, occ9, cell9):
        if not is_mine_target_kind9(ctx9.get("cls")):
            return None
        try:
            target9 = tuple(int(q9) for q9 in cell9)
        except (TypeError, ValueError, OverflowError):
            return None
        if len(target9) != 3:
            return None
        px9, py9, pz9 = (float(q9) for q9 in w.get_pos())
        scope9 = visible_mine_admissibility_cache_key9(occ9)
        cache9 = ctx9.get("_visible_mine_stance_cache")
        if not isinstance(cache9, dict) or cache9.get("scope") != scope9:
            cache9 = {"scope": scope9, "stances": {}}
            ctx9["_visible_mine_stance_cache"] = cache9
        stances9 = cache9["stances"]
        if target9 not in stances9:
            preferred9 = (screened_mine_preferred_feet9(ctx9, target9)
                          if ctx9.get("mine_sites") else None)
            preferred_kwargs9 = ({} if preferred9 is None else
                                  {"preferred_feet": preferred9})
            stance9 = G.reachable_mine_stance(
                occ9, (px9, py9, pz9), target9, reach=MINE_REACH,
                **preferred_kwargs9)
            stances9[target9] = False if stance9 is None else stance9
        cached9 = stances9[target9]
        return None if cached9 is False else cached9

    def unknown_mine_class_census9(reason9):
        return {
            "class_visibility_known_bits": 0,
            "class_visibility_visible_bits": 0,
            "class_visible_instances": [],
            "class_union_masks": [],
            "class_visibility_scan_witness": {
                "version": 1,
                "complete": False,
                "method": "unavailable",
                "reason": str(reason9),
                "absence_policy": "unknown_no_negative_supervision",
            },
            "class_census_runtime": {
                "census_ms": 0.0,
                "indexed_cell_count": 0,
                "candidate_cell_count": 0,
                "rasterized_candidate_count": 0,
                "sparse_instance_policy": (
                    "nearest_record_plus_exhaustive_class_union"),
                "occupancy_grid_index_passes": 0,
                "index_cache_hit": 0,
                "occupancy_revision": -1,
                "shared_visual_scene_build_count": 0,
                "shared_visual_scene_request_count": 0,
                "shared_visual_scene_grid_passes": 0,
                "static_scene_cache_hit": 0,
                "potentially_exposed_candidate_count": 0,
                "projected_recognition_candidate_count": 0,
                "entity_occlusion_query_guard_available": 0,
                "known_class_count": 0,
                "visible_class_count": 0,
                "negative_supervision_available": 0,
            },
        }

    def mine_supported_class_census9(ctx9, occ9, goal_instances9):
        if not args.visible_surface_masks or not args.all_class_census:
            return None
        goal9 = G._bare(str(ctx9.get("cls") or ""))
        if goal9 not in mine_target_pool9:
            return None
        started9 = time.perf_counter()
        index9, index_runtime9 = SC.index_supported_cells_cached(
            occ9, roster=mine_census_roster9)
        witness9 = SC.exact_scan_coverage_witness(
            w.get_pos(), getattr(occ9, "boxes", ()))
        label_eye9, _label_yaw9, _label_pitch9, _label_source9 = label_camera9()
        entity_cells9, entity_meta9 = current_structural_mob_occluders9()
        raw_entity_box9 = entity_meta9.get(
            "mob_query_absolute_xyz_half_open")
        entity_guarded9 = None
        if (entity_meta9.get("mobs_packet_parse_valid") is True
                and isinstance(raw_entity_box9, list)
                and len(raw_entity_box9) == 6):
            hx9, lower_y9, upper_y9, hz9 = STRUCTURAL_ENTITY_QUERY_GUARD
            entity_guarded9 = (
                float(raw_entity_box9[0]) + hx9,
                float(raw_entity_box9[1]) - hx9,
                float(raw_entity_box9[2]) + lower_y9,
                float(raw_entity_box9[3]) - upper_y9,
                float(raw_entity_box9[4]) + hz9,
                float(raw_entity_box9[5]) - hz9,
            )
            if not (
                    entity_guarded9[0] <= label_eye9[0] < entity_guarded9[1]
                    and entity_guarded9[2] <= label_eye9[1] < entity_guarded9[3]
                    and entity_guarded9[4] <= label_eye9[2] < entity_guarded9[5]):
                entity_guarded9 = None
        domain_box9 = tuple(int(q9) for q9 in
                            witness9["recognition_domain_box"])
        voxel_domain_complete9 = witness9.get("complete") is True
        entity_domain_complete9 = bool(
            entity_guarded9 is not None
            and entity_guarded9[0] <= domain_box9[0]
            and domain_box9[1] <= entity_guarded9[1]
            and entity_guarded9[2] <= domain_box9[2]
            and domain_box9[3] <= entity_guarded9[3]
            and entity_guarded9[4] <= domain_box9[4]
            and domain_box9[5] <= entity_guarded9[5])
        witness9.update({
            "voxel_domain_complete": bool(voxel_domain_complete9),
            "entity_query_domain_complete": bool(entity_domain_complete9),
            "candidate_domain_complete": bool(
                voxel_domain_complete9 and entity_domain_complete9),
            "complete": bool(
                voxel_domain_complete9 and entity_domain_complete9),
            "negative_supervision_available": bool(
                voxel_domain_complete9 and entity_domain_complete9),
        })
        if not witness9["complete"]:
            witness9["incomplete_reason"] = (
                "voxel_or_entity_query_does_not_cover_recognition_domain")
        scrub_certified9 = frozenset()
        scrub_record9 = (ctx9.get("census_scrub")
                         if args.census_scrub else None)
        if scrub_record9 is not None:
            staged_by_kind9 = {}
            for meta9 in (ctx9.get("mine_sites") or ()):
                staged_by_kind9.setdefault(
                    G._bare(str(meta9.get("kind") or "")), []).append(
                        (int(meta9["x"]), int(meta9["y"]), int(meta9["z"])))
            scrub_certified9, scrub_audit9 = SC.scrub_certified_absent_kinds(
                scrub_record9, label_eye9,
                staged_cells_by_kind=staged_by_kind9,
                cell_known=occ9.known)
            witness9["census_scrub_certification"] = scrub_audit9
        scene9 = {}
        scene_builds9 = 0
        scene_grid_passes9 = 0
        scene_cache_hits9 = 0
        exposed_candidates9 = 0
        depth_owner_rejected_candidates9 = 0
        projection_mask_requests9 = 0
        projected_recognition_candidates9 = 0
        class_union_accumulators9 = {}
        pixel_exclusion9 = structural_pixel_exclusion9(
            relax_human_swing=bool(args.human_play))

        def ensure_scene9():
            nonlocal scene_builds9, scene_grid_passes9, scene_cache_hits9
            if scene9:
                return scene9
            terrain9 = G.shared_visible_surface_scene(
                occ9, label_eye9, cutout_as_full_cube=False)
            scene9.update(terrain9)
            if entity_cells9:
                scene9["visual_extra_solid"] = frozenset(
                    set(terrain9["visual_extra_solid"]) | set(entity_cells9))
            if entity_cells9:
                scene9["candidate_face_blockers"] = frozenset(
                    set(terrain9["candidate_face_blockers"])
                    | set(entity_cells9))
            scene9["pixel_exclusion_mask"] = pixel_exclusion9
            scene_builds9 += 1
            scene_grid_passes9 += int(terrain9["occupancy_grid_passes"])
            scene_cache_hits9 += int(terrain9["static_scene_cache_hit"])
            return scene9

        def candidate_filter9(_kind9, cell9):
            nonlocal exposed_candidates9, projected_recognition_candidates9
            nonlocal depth_owner_rejected_candidates9
            nonlocal projection_mask_requests9
            entity_domain9 = bool(
                entity_guarded9 is not None
                and entity_guarded9[0] <= cell9[0]
                and cell9[0] + 1 <= entity_guarded9[1]
                and entity_guarded9[2] <= cell9[1]
                and cell9[1] + 1 <= entity_guarded9[3]
                and entity_guarded9[4] <= cell9[2]
                and cell9[2] + 1 <= entity_guarded9[5])
            recognition_domain9 = bool(
                domain_box9[0] <= cell9[0]
                and cell9[0] + 1 <= domain_box9[1]
                and domain_box9[2] <= cell9[1]
                and cell9[1] + 1 <= domain_box9[3]
                and domain_box9[4] <= cell9[2]
                and cell9[2] + 1 <= domain_box9[5])
            if (not recognition_domain9 or not entity_domain9
                    or cell9 in entity_cells9):
                return False
            owner_pixels9 = G.depth_owner_pixel_indices(
                label_eye9, _label_yaw9, _label_pitch9, cell9,
                pixel_exclusion_mask=pixel_exclusion9)
            if owner_pixels9 is not None and not len(owner_pixels9):
                depth_owner_rejected_candidates9 += 1
                return False
            projection_mask_requests9 += 1
            projected9 = G.projected_full_cube_candidate_stats(
                label_eye9, _label_yaw9, _label_pitch9, cell9,
                pixel_exclusion_mask=pixel_exclusion9)
            projected_bbox9 = projected9["projected_bbox_px"]
            if (projected_bbox9 is None
                    or not structural_full_cube_surface_recognizable(
                        projected9["projected_area_px"],
                        projected9["projected_short_side_px"],
                        projected_bbox9)):
                return False
            projected_recognition_candidates9 += 1
            shared9 = ensure_scene9()
            potentially_exposed9 = SC.has_known_potentially_visible_face(
                cell9, known=occ9.known, solid=occ9.solid_at,
                visual_extra_solid=shared9["candidate_face_blockers"])
            if not potentially_exposed9:
                return False
            exposed_candidates9 += 1
            return True

        def recognize9(_kind9, cell9):
            shared9 = ensure_scene9()
            face9 = dense_full_cube_surface9(
                occ9, (cell9,),
                visual_extra9=shared9["visual_extra_solid"],
                alpha_cutout9=shared9["alpha_cutout_visual"],
                thin_visual9=shared9["thin_visual_solid"],
                pixel_exclusion9=shared9["pixel_exclusion_mask"],
                require_full_cube_recognition9=True)
            if face9 is None:
                return None
            point9, normal9, proof9 = face9
            return block_visible_instance9(
                occ9, cell9, point9, normal9, proof9)

        def accumulate_class_union9(kind9, raw9):
            record9 = dict(raw9)
            proof9 = record9.get("proof")
            runs9 = None if not isinstance(proof9, dict) else proof9.get(
                "certified_pixel_runs_yx")
            if not isinstance(runs9, list) or not runs9:
                raise RuntimeError(
                    f"recognizable census member {kind9!r} lacks certified runs")
            state9 = class_union_accumulators9.get(kind9)
            if state9 is None:
                state9 = {
                    "mask": np.zeros((G.H_PX, G.W_PX), dtype=np.uint8),
                    "representative": record9,
                    "member_count": 0,
                    "tested_ray_count": 0,
                }
                class_union_accumulators9[kind9] = state9
            for run9 in runs9:
                if not isinstance(run9, (tuple, list)) or len(run9) != 3:
                    raise RuntimeError("malformed census certified pixel run")
                yy9, x09, x19 = (int(q9) for q9 in run9)
                if not (0 <= yy9 < G.H_PX and 0 <= x09 <= x19 < G.W_PX):
                    raise RuntimeError("census certified pixel run out of bounds")
                state9["mask"][yy9, x09:x19 + 1] = 1
            state9["member_count"] += 1
            state9["tested_ray_count"] += int(
                proof9.get("tested_ray_count", 0))

        goal_is_census_class9 = goal9 in mine_census_roster9
        census9 = SC.build_supported_class_census(
            index9,
            recognize_instance=recognize9,
            candidate_filter=candidate_filter9,
            rank_key=lambda cell9: (
                math.sqrt(
                    (cell9[0] + 0.5 - label_eye9[0]) ** 2
                    + (cell9[1] + 0.5 - label_eye9[1]) ** 2
                    + (cell9[2] + 0.5 - label_eye9[2]) ** 2),),
            preconfirmed_instances=(
                {goal9: goal_instances9} if goal_is_census_class9 else {}),
            exhaustive_kinds=mine_census_roster9,
            recognizable_instance_sink=accumulate_class_union9,
            coverage_witness=witness9,
            certified_absent_kinds=scrub_certified9,
            roster=mine_census_roster9)
        union_records9 = []
        full_cube_only_fields9 = (
            "visible_bbox_area_px", "visible_bbox_fill_numerator_px",
            "visible_bbox_fill_denominator_px",
            "recognition_full_cube_min_bbox_fill_numerator",
            "recognition_full_cube_min_bbox_fill_denominator",
            "recognition_full_cube_min_visible_area_px",
            "recognition_full_cube_min_visible_short_side_px",
            "recognition_full_cube_area_pass",
            "recognition_full_cube_short_side_pass",
            "recognition_full_cube_bbox_fill_pass",
            "recognition_threshold_full_cube_area_180px",
            "recognition_threshold_full_cube_short_side_21px",
            "recognition_threshold_full_cube_bbox_fill_one_third",
        )
        for kind9 in mine_census_roster9:
            state9 = class_union_accumulators9.get(kind9)
            if state9 is None:
                continue
            mask9 = state9["mask"]
            ys9, xs9 = np.nonzero(mask9)
            if not len(xs9):
                raise RuntimeError("recognizable class union is empty")
            bbox9 = [int(xs9.min()), int(ys9.min()),
                     int(xs9.max()), int(ys9.max())]
            area9 = int(mask9.sum())
            short9 = min(
                bbox9[2] - bbox9[0] + 1,
                bbox9[3] - bbox9[1] + 1)
            representative9 = state9["representative"]
            proof9 = dict(representative9["proof"])
            for field9 in full_cube_only_fields9:
                proof9.pop(field9, None)
            proof9.update(structural_surface_recognition_fields(
                area9, short9, visible_bbox_px=bbox9,
                require_full_cube_fill=False))
            proof9.update({
                "visible_area_px": area9,
                "visible_short_side_px": int(short9),
                "visible_bbox_px": bbox9,
                "certified_mask_shape": [G.H_PX, G.W_PX],
                "certified_pixel_runs_yx": G.certified_pixel_runs_yx(mask9),
                "certified_pixel_count": area9,
                "tested_ray_count": int(state9["tested_ray_count"]),
                "class_union_member_count": int(state9["member_count"]),
                "class_union_complete_enumeration": True,
                "mask_scope": (
                    "all_current_oracle_recognizable_instances_of_class_"
                    "within_census_candidate_domain"),
                "selection_shape": "disconnected_class_union_no_bridge",
                "representative_instance_id": str(
                    representative9["instance_id"]),
            })
            union_records9.append({
                "instance_id": f"class_union:{kind9}",
                "representative_instance_id": str(
                    representative9["instance_id"]),
                "kind": kind9,
                "visible": 1,
                "chosen": False,
                "identity_scope": "current_frame_class_union",
                "type": kind9,
                "world_position": list(
                    representative9.get("world_position") or ()),
                "visible_point": list(representative9["visible_point"]),
                "face_normal": representative9.get("face_normal"),
                "member_instance_count": int(state9["member_count"]),
                "proof": proof9,
            })
        census9["class_union_masks"] = union_records9
        runtime9 = census9["class_census_runtime"]
        runtime9["census_ms"] = round(
            1000.0 * (time.perf_counter() - started9), 3)
        runtime9.update(index_runtime9)
        runtime9["shared_visual_scene_build_count"] = int(
            scene_grid_passes9)
        runtime9["shared_visual_scene_request_count"] = int(scene_builds9)
        runtime9["shared_visual_scene_grid_passes"] = int(
            scene_grid_passes9)
        runtime9["static_scene_cache_hit"] = int(scene_cache_hits9)
        runtime9["entity_occlusion_query_guard_available"] = int(
            entity_guarded9 is not None)
        runtime9["potentially_exposed_candidate_count"] = int(
            exposed_candidates9)
        runtime9["depth_owner_prefilter_rejected_count"] = int(
            depth_owner_rejected_candidates9)
        runtime9["projection_mask_request_count"] = int(
            projection_mask_requests9)
        runtime9["projected_recognition_candidate_count"] = int(
            projected_recognition_candidates9)
        depth_frame9 = G.active_depth_frame()
        if depth_frame9 is not None:
            runtime9.update({
                key9: (round(float(value9), 3)
                       if isinstance(value9, float) else int(value9))
                for key9, value9 in depth_frame9["owner_stats"].items()
            })
        runtime9["known_class_count"] = int(bin(int(
            census9["class_visibility_known_bits"])).count("1"))
        runtime9["visible_class_count"] = int(bin(int(
            census9["class_visibility_visible_bits"])).count("1"))
        runtime9["negative_supervision_available"] = int(
            witness9.get("complete") is True)
        SC.validate_census(
            census9,
            goal_kind=(goal9 if goal_is_census_class9 else None),
            goal_class_exist=(bool(goal_instances9)
                              if goal_is_census_class9 else None),
            roster=mine_census_roster9)
        return census9

    def staged_confusers_from_census9(ctx9, census9):
        if census9 is None:
            return []
        goal9 = G._bare(str(ctx9.get("cls") or ""))
        staged9 = {
            G._bare(str(meta9.get("kind") or ""))
            for meta9 in (ctx9.get("mine_sites") or ())
            if G._bare(str(meta9.get("kind") or "")) != goal9}
        out9 = []
        for raw9 in census9.get("class_visible_instances", ()):
            if str(raw9.get("kind") or "") not in staged9:
                continue
            instance9 = dict(raw9)
            instance9["role"] = "distractor"
            instance9["chosen"] = False
            out9.append(instance9)
        return out9

    def refresh_current_structural_telemetry9(ctx9):
        if current_label_gui_open9():
            stamp_current_structural_fields9(
                ctx9, class_exist=False, chosen_visible=False,
                visible_instances=[],
                class_census=(unknown_mine_class_census9("gui_occluded")
                              if args.visible_surface_masks
                              and args.all_class_census
                              and is_mine_target_kind9(ctx9.get("cls")) else None),
                source="current_rgb_occluded_by_gui")
            return
        occ9 = ctx9.get("_struct_occ")
        if occ9 is None:
            stamp_current_structural_fields9(
                ctx9, class_exist=False, chosen_visible=False,
                visible_instances=[],
                class_census=(unknown_mine_class_census9("geometry_unavailable")
                              if args.visible_surface_masks
                              and args.all_class_census
                              and is_mine_target_kind9(ctx9.get("cls")) else None),
                source="current_geometry_unavailable")
            return
        box9 = STRUCT.get("last_voxel_box")
        if (STRUCT.get("offline_human_label_row") is None
                and box9 is not None and w.info.get("voxels") is not None):
            occ9.ingest(w.info.get("voxels") or [], w.get_pos(), box9)

        eye9, yaw9, pitch9, _render_source9 = label_camera9()
        G.clear_depth_frame()
        if args.visible_surface_masks:
            label_depth9 = STRUCT.get("current_label_depth9")
            if label_depth9 is None and not STRUCT.get("_human_deferred_replay9"):
                label_depth9 = w.info.get(RENDERER_VIEWMODEL_DEPTH_FIELD)
            depth_arr9 = (None if label_depth9 is None
                          else np.asarray(label_depth9, dtype=np.float64))
            if depth_arr9 is not None and np.isfinite(depth_arr9).any():
                try:
                    cellmap9, cellvalid9 = G.backproject_depth_to_cells(
                        depth_arr9, eye9, yaw9, pitch9,
                        near=DEPTH_NEAR_PLANE9, far=DEPTH_FAR_PLANE9)
                    G.set_depth_frame(
                        eye9, yaw9, pitch9, cellmap9, cellvalid9,
                        near=DEPTH_NEAR_PLANE9, far=DEPTH_FAR_PLANE9,
                        source=str(_render_source9),
                        build_owner_index=True,
                        audit_owner_index=False)
                except Exception:
                    G.clear_depth_frame()
        phase9 = int((TRAJ.get("pending_row") or {}).get("phase", TRAJ.get("phase", 0)))
        mine_identity_required9 = bool(
            is_mine_target_kind9(ctx9.get("cls")))
        phase_has_live_commit9 = bool(
            phase9 in (1, 2)
            and (not mine_identity_required9
                 or ctx9.get("_committed_instance_id") is not None))
        class_exist9 = False
        chosen9 = None
        chosen_instance_id9 = None
        visible_instances9 = []
        confuser_instances9 = []
        committed_trace_cell9 = None
        committed_rejection_trace9 = None
        committed_surface_sink9 = {}
        committed_surface_instance9 = None

        if ctx9.get("cls"):
            cls9 = ctx9["cls"]
            detect_dist9 = 36.0
            navigation_admissible9 = None
            if phase9 == 0 and is_mine_target_kind9(cls9):
                def navigation_admissible9(cell9, _face9):
                    return reachable_visible_mine_stance9(
                        ctx9, occ9, cell9) is not None
            committed_trace_cell9 = None
            if phase_has_live_commit9:
                raw_committed9 = (
                    ctx9.get("_semantic_tx", ctx9.get("tx")),
                    ctx9.get("_semantic_ty", ctx9.get("ty")),
                    ctx9.get("_semantic_tz", ctx9.get("tz")))
                if all(q9 is not None for q9 in raw_committed9):
                    committed_trace_cell9 = tuple(
                        int(q9) for q9 in raw_committed9)
            committed_rejection_trace9 = (
                {} if committed_trace_cell9 is not None else None)
            got9 = nearest_visible_cluster(
                occ9, cls9, max_dist=detect_dist9,
                admissible=navigation_admissible9,
                admissibility_cache_key=(
                    visible_mine_admissibility_cache_key9(occ9)
                    if navigation_admissible9 is not None else None),
                visible_instances=visible_instances9,
                trace_cell=committed_trace_cell9,
                rejection_trace=committed_rejection_trace9,
                tracked_cell=committed_trace_cell9,
                tracked_surface_sink=committed_surface_sink9)
            committed_surface_instance9 = committed_surface_sink9.get(
                "instance")
            class_exist9 = bool(visible_instances9)
            if phase_has_live_commit9:
                target9 = tuple(int(q9) for q9 in (
                    ctx9.get("_semantic_tx", ctx9.get("tx")),
                    ctx9.get("_semantic_ty", ctx9.get("ty")),
                    ctx9.get("_semantic_tz", ctx9.get("tz"))))
                chosen_instance_id9 = (
                    f"block:{target9[0]}:{target9[1]}:{target9[2]}")
                exact_instance9 = next(
                    (instance9 for instance9 in visible_instances9
                     if str(instance9.get("instance_id"))
                        == chosen_instance_id9), None)
                if exact_instance9 is not None:
                    point9 = exact_instance9.get("visible_point")
                    normal9 = exact_instance9.get("face_normal")
                    if (isinstance(point9, list) and len(point9) == 3
                            and isinstance(normal9, list)
                            and len(normal9) == 3):
                        chosen9 = (point9, normal9)

        if chosen9 is not None:
            matches9 = 0
            for instance9 in visible_instances9:
                selected9 = str(instance9.get("instance_id")) == str(chosen_instance_id9)
                instance9["chosen"] = bool(selected9)
                matches9 += int(selected9)
            if matches9 != 1:
                chosen9 = None
                chosen_instance_id9 = None

        class_census9 = None
        if (args.visible_surface_masks and args.all_class_census
                and is_mine_target_kind9(ctx9.get("cls"))):
            class_census9 = mine_supported_class_census9(
                ctx9, occ9, visible_instances9)
            confuser_instances9 = staged_confusers_from_census9(
                ctx9, class_census9)

        visibility_debug9 = None
        if committed_trace_cell9 is not None:
            committed_id9 = "block:%d:%d:%d" % committed_trace_cell9
            committed_visible9 = any(
                str(instance9.get("instance_id")) == committed_id9
                for instance9 in visible_instances9)
            trace_out9 = dict(committed_rejection_trace9 or {})
            pending_t9 = int(
                (TRAJ.get("pending_row") or {}).get("t", -1))
            revision_trace9 = STRUCT.get("_occ_revision_label_trace")
            if (not isinstance(revision_trace9, dict)
                    or revision_trace9.get("occ_id") != id(occ9)
                    or revision_trace9.get("revision") != int(occ9.revision)):
                revision_trace9 = {
                    "occ_id": id(occ9),
                    "revision": int(occ9.revision),
                    "first_labeled_traj_t": pending_t9,
                }
                STRUCT["_occ_revision_label_trace"] = revision_trace9
            committed_kind9 = occ9.grid.get(committed_trace_cell9)
            visibility_debug9 = {
                "version": 1,
                "committed_cell": list(committed_trace_cell9),
                "committed_instance_id": committed_id9,
                "committed_visible": int(committed_visible9),
                "committed_surface_visible": int(
                    committed_visible9
                    or committed_surface_instance9 is not None),
                "visible_instance_count": int(len(visible_instances9)),
                "candidate_cell_count": trace_out9.get("matching_cell_count"),
                "component_count": trace_out9.get("component_count"),
                "detector_cache_hit": int(
                    bool(trace_out9.get("detector_cache_hit"))),
                "committed_rejection": (None if committed_visible9 else {
                    key9: value9 for key9, value9 in trace_out9.items()
                    if key9 not in ("matching_cell_count", "component_count",
                                    "detector_cache_hit")}),
                "occupancy_snapshot": {
                    "occ_revision": int(occ9.revision),
                    "known_cell_count": int(len(occ9.grid)),
                    "committed_cell_known": bool(
                        occ9.known(committed_trace_cell9)),
                    "committed_cell_kind": (
                        None if committed_kind9 is None
                        else str(committed_kind9)),
                    "committed_cell_in_scan_boxes": bool(any(
                        box9[0] <= committed_trace_cell9[0] < box9[1]
                        and box9[2] <= committed_trace_cell9[1] < box9[3]
                        and box9[4] <= committed_trace_cell9[2] < box9[5]
                        for box9 in getattr(occ9, "boxes", ()))),
                    "revision_first_labeled_traj_t": int(
                        revision_trace9["first_labeled_traj_t"]),
                    "revision_age_frames_lower_bound": int(
                        max(0, pending_t9 - int(
                            revision_trace9["first_labeled_traj_t"]))),
                },
            }

        G.clear_depth_frame()
        stamp_current_structural_fields9(
            ctx9, class_exist=class_exist9, chosen_visible=chosen9 is not None,
            visible_point=None if chosen9 is None else chosen9[0],
            face_normal=None if chosen9 is None else chosen9[1],
            visible_instances=visible_instances9,
            committed_surface_instance=committed_surface_instance9,
            confuser_instances=confuser_instances9,
            class_census=class_census9,
            source="current_camera+known_geometry",
            visibility_debug=visibility_debug9)

    def policy_eval_current_target_instances9(ctx9):
        occ9 = ctx9.get("_struct_occ")
        cls9 = G._bare(str(ctx9.get("cls") or ""))
        if not isinstance(occ9, G.OccupancyMap) or cls9 not in mine_target_pool9:
            return []
        G.clear_depth_frame()
        depth9 = w.info.get(RENDERER_VIEWMODEL_DEPTH_FIELD)
        depth_arr9 = (None if depth9 is None else
                      np.asarray(depth9, dtype=np.float64))
        if depth_arr9 is not None and np.isfinite(depth_arr9).any():
            try:
                eye9, yaw9, pitch9, source9 = label_camera9()
                cellmap9, valid9 = G.backproject_depth_to_cells(
                    depth_arr9, eye9, yaw9, pitch9,
                    near=DEPTH_NEAR_PLANE9, far=DEPTH_FAR_PLANE9)
                G.set_depth_frame(
                    eye9, yaw9, pitch9, cellmap9, valid9,
                    near=DEPTH_NEAR_PLANE9, far=DEPTH_FAR_PLANE9,
                    source=str(source9), build_owner_index=True,
                    audit_owner_index=False)
            except Exception:
                G.clear_depth_frame()
        instances9 = []
        detector_audit9 = {}
        rejection_trace9 = {}
        target_cells9 = sorted(
            (int(site9["x"]), int(site9["y"]), int(site9["z"]))
            for site9 in (ctx9.get("mine_sites") or ())
            if G._bare(str(site9.get("kind") or "")) == cls9)
        trace_cell9 = target_cells9[0] if target_cells9 else None
        try:
            nearest_visible_cluster(
                occ9, cls9, max_dist=36.0,
                audit=detector_audit9,
                visible_instances=instances9,
                trace_cell=trace_cell9,
                rejection_trace=rejection_trace9)
            if not instances9:
                compact9 = {
                    key9: value9 for key9, value9 in detector_audit9.items()
                    if key9 not in ("components",)
                }
                compact9["trace"] = rejection_trace9
                compact9["target_cells"] = [list(cell9) for cell9 in target_cells9]
                ctx9["_policy_gt_empty_audit"] = compact9
                if not ctx9.get("_policy_gt_empty_audit_printed"):
                    print("[policy-gt-empty] " + json.dumps(compact9), flush=True)
                    ctx9["_policy_gt_empty_audit_printed"] = True
            else:
                ctx9.pop("_policy_gt_empty_audit", None)
            return [copy.deepcopy(value9) for value9 in instances9]
        finally:
            G.clear_depth_frame()

    def clone_human_occupancy9(source9):
        if not isinstance(source9, G.OccupancyMap):
            raise RuntimeError("human exact replay has no occupancy snapshot")
        cloned9 = G.OccupancyMap(
            water_occludes=bool(source9.water_occludes),
            leaves_occlude=bool(source9.leaves_occlude))
        cloned9.grid = dict(source9.grid)
        cloned9.boxes = [tuple(int(q9) for q9 in box9)
                         for box9 in source9.boxes]
        if (len(cloned9.grid) != len(source9.grid)
                or cloned9.boxes != [tuple(q9 for q9 in box9)
                                     for box9 in source9.boxes]):
            raise RuntimeError("human occupancy snapshot copy is incomplete")
        return cloned9

    def finalize_human_deferred_labels9(
            ctx9, occupancy9, *, end_reason9, completed9):
        if not STRUCT.get("defer_human_labels"):
            raise RuntimeError("human deferred finalizer called outside its segment")
        start_t9 = STRUCT.get("defer_human_start_t")
        if start_t9 is None:
            raise RuntimeError("human deferred finalizer has no source frame")
        buffered9 = list(TRAJ.get("deferred_human_rows") or [])
        pending9 = TRAJ.get("pending_row")
        if (pending9 is not None and int(pending9.get("t", -1)) >= int(start_t9)):
            if buffered9 and int(buffered9[-1]["t"]) >= int(pending9["t"]):
                raise RuntimeError("human deferred pending-row order is invalid")
            buffered9.append(pending9)
        if not buffered9:
            raise RuntimeError("human deferred segment contains no trajectory rows")
        action_by_source9 = source_action_index(buffered9)
        traj_by_t9 = {int(row9["t"]): row9 for row9 in buffered9}
        if len(traj_by_t9) != len(buffered9):
            raise RuntimeError("human deferred trajectory contains duplicate t")
        story_rows9 = STORY.get("rows") or []
        joined_story9 = [
            row9 for row9 in story_rows9
            if int(row9.get("traj_t", -1)) in traj_by_t9]
        if not joined_story9:
            raise RuntimeError("human deferred segment has no kept RGB joins")
        if any(int(joined_story9[i9]["f"]) >= int(joined_story9[i9 + 1]["f"])
               for i9 in range(len(joined_story9) - 1)):
            raise RuntimeError("human deferred story frames are not strictly ordered")

        bound_id9 = None
        bound_cells9 = frozenset()
        automatic_bound_id9 = None
        automatic_bound_anchor_xz9 = None
        previous_canonical_id9 = None
        previous_canonical_phase9 = 0
        previous_behavior_phase9 = 0
        original_story_rows9 = STORY["rows"]
        original_pending9 = TRAJ.get("pending_row")
        original_phase9 = int(TRAJ.get("phase", 0))
        original_occ9 = ctx9.get("_struct_occ")
        replay_ctx_fields9 = (
            "_committed_instance_id", "_chosen_instance_id",
            "_chosen_visible", "_semantic_tx", "_semantic_ty",
            "_semantic_tz", "tx", "ty", "tz", "_visual_target_cell")
        original_replay_ctx9 = {
            key9: (key9 in ctx9, copy.deepcopy(ctx9.get(key9)))
            for key9 in replay_ctx_fields9
        }
        original_last_y9 = TRAJ.pop("_last_labeled_y", None)
        labels_started9 = time.perf_counter()
        labelled9 = 0
        unknown9 = 0
        marker_counts9 = {
            "human_commit": 0, "human_switch": 0,
            "human_reaffirm": 0, "human_marker_rejected": 0,
            "human_implicit_commit": 0,
            "human_clear_after_success_or_removal": 0,
        }
        interaction_selected_mismatch9 = 0
        successful_active_chosen_interactions9 = 0
        off_target_interactions9 = 0
        human_bc_excluded_frames9 = 0
        first_truth_visible_index9 = None
        first_valid_marker_index9 = None
        success_source_ts9 = {
            int(audit9["source_rgb_traj_t"])
            for row9 in joined_story9
            for audit9 in [row9.get("human_success_after_action")]
            if isinstance(audit9, dict)
            and audit9.get("source_rgb_traj_t") is not None
        }

        try:
            ctx9["_struct_occ"] = occupancy9
            STRUCT["_human_deferred_replay9"] = True
            for story9 in joined_story9:
                traj9 = traj_by_t9[int(story9["traj_t"])]
                traj9["human_demo"] = 1
                story9["human_demo"] = 1
                pose_traj9 = traj9.get("render_pose")
                if (not isinstance(pose_traj9, dict)
                        or pose_traj9 != story9.get("render_pose")):
                    raise RuntimeError(
                        "human offline traj/story render poses do not match")
                STORY["rows"] = [story9]
                TRAJ["pending_row"] = traj9
                STRUCT["offline_human_label_row"] = story9
                STRUCT["visibility_cache"] = {}
                traj9["phase"] = 0
                story9["phase"] = 0
                traj9["decision_event"] = ""
                story9["decision_event"] = ""

                voxel_observation9 = story9.get(
                    "structural_voxel_observation")
                if not isinstance(voxel_observation9, dict):
                    raise RuntimeError(
                        "human offline RGB lacks voxel observation contract")
                if voxel_observation9.get("packet_present"):
                    if voxel_observation9.get("packet_parse_valid") is not True:
                        raise RuntimeError(
                            "human offline voxel packet is malformed")
                    voxel_box9 = voxel_observation9.get("query_box_relative")
                    voxel_player9 = voxel_observation9.get("player_position")
                    voxel_blocks9 = voxel_observation9.get("blocks_relative")
                    if (not isinstance(voxel_box9, list) or len(voxel_box9) != 6
                            or not isinstance(voxel_player9, list)
                            or len(voxel_player9) != 3
                            or not isinstance(voxel_blocks9, list)):
                        raise RuntimeError(
                            "human offline voxel packet lacks exact query geometry")
                    occupancy9.ingest(
                        voxel_blocks9, voxel_player9, voxel_box9)

                remaining_goal_cells9 = {
                    cell9 for cell9, kind9 in occupancy9.grid.items()
                    if G._bare(str(kind9 or ""))
                    == G._bare(str(ctx9.get("cls") or ""))
                }
                if (cell == "mine" and bound_id9 is not None
                        and selected_component_removed(
                            bound_cells9, remaining_goal_cells9)):
                    clear_audit9 = {
                        "version": 1,
                        "event": "human_clear_after_success_or_removal",
                        "reason": "selected_target_cells_removed",
                        "previous_chosen_instance_id": bound_id9,
                        "source_rgb_traj_t": int(story9["traj_t"]),
                    }
                    bound_id9 = None
                    bound_cells9 = frozenset()
                    automatic_bound_id9 = None
                    automatic_bound_anchor_xz9 = None
                    previous_behavior_phase9 = 0
                    traj9["human_chosen_clear"] = dict(clear_audit9)
                    story9["human_chosen_clear"] = dict(clear_audit9)
                    marker_counts9[
                        "human_clear_after_success_or_removal"] += 1

                success9 = story9.get("human_success_after_action")
                if isinstance(success9, dict):
                    cleared_id9 = bound_id9
                    bound_id9 = None
                    bound_cells9 = frozenset()
                    automatic_bound_id9 = None
                    automatic_bound_anchor_xz9 = None
                    previous_behavior_phase9 = 0
                    if "human_chosen_clear" not in story9:
                        clear_audit9 = {
                            "version": 1,
                            "event": "human_clear_after_success_or_removal",
                            "reason": "task_success_post_action",
                            "previous_chosen_instance_id": cleared_id9,
                            "source_rgb_traj_t": int(story9["traj_t"]),
                        }
                        traj9["human_chosen_clear"] = dict(clear_audit9)
                        story9["human_chosen_clear"] = dict(clear_audit9)
                        marker_counts9[
                            "human_clear_after_success_or_removal"] += 1
                    success_cell9 = success9.get("ray_cell")
                    if (cell == "mine" and isinstance(success_cell9, list)
                            and len(success_cell9) == 3):
                        removed_cell9 = tuple(int(q9) for q9 in success_cell9)
                        old_kind9 = occupancy9.grid.get(removed_cell9)
                        if old_kind9 is not None:
                            raise RuntimeError(
                                "human mine success packet still contains goal cell")
                    elif cell == "mine":
                        raise RuntimeError(
                            "human mine success lacks a first-hit goal cell")

                STRUCT["current_label_depth9"] = (
                    STRUCT.get("human_frame_depths9", {}).get(int(traj9["t"])))
                refresh_current_structural_telemetry9(ctx9)
                instances9 = [dict(q9) for q9 in
                              story9.get("visible_instances", [])]
                frame_index9 = int(story9["f"])
                if instances9 and first_truth_visible_index9 is None:
                    first_truth_visible_index9 = frame_index9
                aux9 = same_rgb_structural_aux(traj9, story9)
                if aux9 is None:
                    raise RuntimeError(
                        "human offline structural labels lack same-RGB identity")
                action_evidence9 = action_by_source9.get(int(story9["traj_t"]))

                def instance_cells9(instance9):
                    component9 = instance9.get("component_cells")
                    if not isinstance(component9, list):
                        proof9 = instance9.get("proof")
                        component9 = (proof9.get("visible_member_cells")
                                      if isinstance(proof9, dict) else None)
                    if not isinstance(component9, list):
                        component9 = [instance9.get("world_position")]
                    return frozenset(
                        tuple(int(q9) for q9 in raw9)
                        for raw9 in component9
                        if isinstance(raw9, (tuple, list)) and len(raw9) == 3)

                def ray_visible_matches9(raw_ray9):
                    if not isinstance(raw_ray9, list) or len(raw_ray9) != 3:
                        return []
                    ray9 = tuple(int(q9) for q9 in raw_ray9)
                    return [instance9 for instance9 in instances9
                            if ray9 in instance_cells9(instance9)]

                pose9 = story9["render_pose"]
                eye9 = (
                    float(pose9["x"]),
                    float(pose9["y"]) + float(pose9.get("eye_height", 1.62)),
                    float(pose9["z"]),
                )

                def automatic_anchor9(instance9):
                    cell9 = instance9.get("world_position")
                    if isinstance(cell9, list) and len(cell9) == 3:
                        return (float(cell9[0]) + 0.5,
                                float(cell9[2]) + 0.5)
                    point9 = instance9.get("visible_point")
                    if isinstance(point9, list) and len(point9) == 3:
                        return (float(point9[0]), float(point9[2]))
                    raise RuntimeError(
                        "visible human target lacks an automatic-choice anchor")

                automatic_decision9 = persistent_nearest_choice(
                    automatic_bound_id9, automatic_bound_anchor_xz9,
                    [(str(instance9["instance_id"]),
                      automatic_anchor9(instance9))
                     for instance9 in instances9],
                    player_xz=(float(pose9["x"]), float(pose9["z"])),
                    closer_margin=1.5)
                automatic_bound_id9 = automatic_decision9["chosen_id"]
                automatic_bound_anchor_xz9 = automatic_decision9["anchor_xz"]
                automatic_record9 = next(
                    (instance9 for instance9 in instances9
                     if str(instance9.get("instance_id"))
                     == str(automatic_bound_id9)), None)

                marker_payload9 = (
                    action_evidence9.get("chosen_marker")
                    if isinstance(action_evidence9, dict) else None)
                marker_choice9 = None
                marker_event9 = None
                if marker_payload9 is not None:
                    if not isinstance(marker_payload9, dict):
                        raise RuntimeError("human chosen marker payload is malformed")
                    if (marker_payload9.get("contract")
                            != HUMAN_PIXEL_MARKER_CONTRACT
                            or int(marker_payload9.get("source_traj_t", -1))
                               != int(story9["traj_t"])
                            or marker_payload9.get("source_render_pose")
                               != story9.get("render_pose")
                            or marker_payload9.get("source_rgb_shape")
                               != [360, 640, 3]
                            or marker_payload9.get(
                                "simulator_steps_during_pause") != 0
                            or marker_payload9.get(
                                "env_action_from_marker_click") is not False):
                        raise RuntimeError(
                            "human marker does not identify the frozen source RGB")
                    rec9 = STORY.get("rec")
                    if (rec9 is None
                            or not (0 <= frame_index9 < len(rec9.frames))):
                        raise RuntimeError(
                            "human marker source RGB is absent from recorder")
                    raw_rgb9 = np.ascontiguousarray(
                        np.asarray(rec9.frames[frame_index9])[:, :, ::-1])
                    if (hashlib.sha256(raw_rgb9.tobytes()).hexdigest()
                            != marker_payload9.get("source_rgb_sha256")):
                        raise RuntimeError(
                            "human marker frozen RGB hash does not match kept RGB")
                    marker_pixel9 = marker_payload9.get("pixel_xy")
                    marker_matches9 = ([] if marker_pixel9 is None else
                                       human_marker_pixel_match_ids(
                                           instances9, marker_pixel9))
                    if len(marker_matches9) > 1:
                        raise RuntimeError(
                            "human marker pixel overlaps multiple target instances")
                    marker_audit9 = {
                        "version": 2,
                        "key": "C",
                        "edge": "key_down",
                        "source": HUMAN_PIXEL_MARKER_CONTRACT,
                        "source_rgb_traj_t": int(story9["traj_t"]),
                        "action_row_t": int(action_evidence9["action_row_t"]),
                        "render_pose": dict(story9["render_pose"]),
                        "pixel_xy": marker_pixel9,
                        "window_xy": marker_payload9.get("window_xy"),
                        "source_rgb_sha256": marker_payload9[
                            "source_rgb_sha256"],
                        "annotation_pause_s": float(
                            marker_payload9.get("annotation_pause_s", 0.0)),
                        "no_simulator_step": True,
                        "no_environment_action": True,
                        "previous_chosen_instance_id": bound_id9,
                    }
                    if len(marker_matches9) == 1:
                        marker_choice9 = human_marker_pixel_target_choice(
                            traj9, story9, marker_pixel9,
                            predecision_aux=aux9)
                        resolved_id9 = marker_payload9.get(
                            "resolved_instance_id")
                        if (resolved_id9 is not None
                                and str(resolved_id9) != str(
                                    marker_choice9["chosen_instance_id"])):
                            raise RuntimeError(
                                "live frozen-screen choice disagrees with "
                                "postplay exact instance identity")
                        marker_event9 = human_marker_transition(
                            bound_id9, marker_choice9["chosen_instance_id"])
                        bound_id9 = str(marker_choice9["chosen_instance_id"])
                        bound_cells9 = frozenset(
                            tuple(int(q9) for q9 in raw9)
                            for raw9 in marker_choice9["component_cells"])
                        marker_audit9.update({
                            "accepted": True,
                            "event": marker_event9,
                            "chosen_instance_id": bound_id9,
                            "frame_identity": marker_choice9["frame_identity"],
                            "visible_point": marker_choice9["visible_point"],
                            "face_normal": marker_choice9["face_normal"],
                            "human_recognized_here": True,
                            "frames_since_first_truth_visible": (
                                None if first_truth_visible_index9 is None else
                                frame_index9 - first_truth_visible_index9),
                        })
                        if first_valid_marker_index9 is None:
                            first_valid_marker_index9 = frame_index9
                    else:
                        marker_event9 = "human_marker_rejected"
                        marker_audit9.update({
                            "accepted": False,
                            "event": marker_event9,
                            "chosen_instance_id": bound_id9,
                            "reason": (
                                "click_outside_raw_rgb"
                                if marker_pixel9 is None else
                                "pixel_not_current_model_recognizable_exact_"
                                "target_surface"),
                        })
                    traj9["human_chosen_marker"] = dict(marker_audit9)
                    story9["human_chosen_marker"] = dict(marker_audit9)
                    marker_counts9[marker_event9] += 1

                traj9["human_recognized_here"] = int(
                    marker_event9 in (
                        "human_commit", "human_switch", "human_reaffirm"))
                story9["human_recognized_here"] = traj9[
                    "human_recognized_here"]

                clicked9 = None
                interaction_attempted9 = bool(
                    action_evidence9 is not None
                    and interaction_pressed(action_evidence9["action"]))
                if interaction_attempted9:
                    click_ray9 = action_evidence9.get("ray_cell")
                    click_matches9 = ray_visible_matches9(click_ray9)
                    if len(click_matches9) == 1:
                        clicked9 = human_raycast_target_choice(
                            traj9, story9, click_ray9,
                            predecision_aux=aux9)
                    elif len(click_matches9) > 1:
                        raise RuntimeError(
                            "human click ray matches multiple visible instances")

                click_on_chosen9 = False
                if clicked9 is not None:
                    clicked_id9 = str(clicked9["chosen_instance_id"])
                    if bound_id9 is None:
                        uncommitted9 = {
                            "version": 1,
                            "event": "human_uncommitted_target_interaction",
                            "chosen_instance_id": None,
                            "interacted_instance_id": clicked_id9,
                            "source_rgb_traj_t": int(story9["traj_t"]),
                            "ray_cell": list(clicked9["ray_cell"]),
                        }
                        traj9["human_uncommitted_target_interaction"] = dict(
                            uncommitted9)
                        story9["human_uncommitted_target_interaction"] = dict(
                            uncommitted9)
                    else:
                        click_on_chosen9 = clicked_id9 == bound_id9
                    if not click_on_chosen9:
                        if bound_id9 is not None:
                            interaction_selected_mismatch9 += 1
                            mismatch9 = {
                                "version": 1,
                                "event": "human_unmarked_target_switch_rejected",
                                "source_rgb_traj_t": int(story9["traj_t"]),
                                "active_human_selected_instance_id": bound_id9,
                                "interacted_instance_id": clicked_id9,
                            }
                            traj9["human_selected_interaction_mismatch"] = dict(
                                mismatch9)
                            story9[
                                "human_selected_interaction_mismatch"] = dict(
                                    mismatch9)

                canonical_id9 = (
                    str(bound_id9) if bound_id9 is not None else None)

                selected_record9 = next(
                    (q9 for q9 in instances9
                     if str(q9.get("instance_id")) == str(canonical_id9)), None)
                selected_instances9 = []
                for instance9 in instances9:
                    instance9 = dict(instance9)
                    instance9["chosen"] = bool(
                        selected_record9 is not None
                        and str(instance9.get("instance_id")) == str(canonical_id9))
                    selected_instances9.append(instance9)

                if click_on_chosen9:
                    phase9 = 2
                else:
                    phase9 = 1 if canonical_id9 is not None else 0
                success_boundary9 = isinstance(success9, dict)
                if success_boundary9 and previous_canonical_phase9 == 2:
                    d6_event9 = "seam" if phase9 == 0 else "seam_commit"
                elif canonical_id9 is None:
                    d6_event9 = ""
                elif previous_canonical_id9 is None:
                    d6_event9 = "commit"
                elif str(previous_canonical_id9) != str(canonical_id9):
                    d6_event9 = "switch"
                else:
                    d6_event9 = ""
                traj9["phase"] = phase9
                story9["phase"] = phase9
                traj9["decision_event"] = d6_event9
                story9["decision_event"] = d6_event9
                previous_canonical_id9 = canonical_id9
                previous_canonical_phase9 = phase9

                human_behavior_phase9 = (
                    2 if click_on_chosen9 and bound_id9 is not None else
                    1 if bound_id9 is not None else 0)
                traj9["human_behavior_phase"] = human_behavior_phase9
                story9["human_behavior_phase"] = human_behavior_phase9
                traj9["human_selected_instance_id"] = bound_id9
                story9["human_selected_instance_id"] = bound_id9
                traj9["human_selected_visible"] = int(
                    bound_id9 is not None and selected_record9 is not None)
                story9["human_selected_visible"] = traj9[
                    "human_selected_visible"]
                previous_behavior_phase9 = human_behavior_phase9

                grounding_unknown_precommit9 = bool(
                    first_valid_marker_index9 is None
                    and bound_id9 is None)
                census_for_stamp9 = aux9["class_census"]
                if grounding_unknown_precommit9:
                    precommit_geometry9 = json.loads(json.dumps(instances9))
                    for output9 in (traj9, story9):
                        output9["human_grounding_unknown_precommit"] = 1
                        output9["human_precommit_geometry_visible"] = int(
                            bool(instances9))
                        output9["human_precommit_geometry_instances"] = (
                            json.loads(json.dumps(precommit_geometry9)))
                        output9[
                            "human_precommit_geometry_suggested_instance_id"] = (
                                automatic_bound_id9)
                        if aux9["class_census"] is not None:
                            output9["human_precommit_class_census_audit"] = (
                                json.loads(json.dumps(aux9["class_census"])))
                    if aux9["class_census"] is None:
                        traj9["human_precommit_class_census_audit"] = {
                            "status": "not_emitted_all_class_census_disabled"}
                        story9["human_precommit_class_census_audit"] = dict(
                            traj9["human_precommit_class_census_audit"])
                else:
                    traj9["human_grounding_unknown_precommit"] = 0
                    story9["human_grounding_unknown_precommit"] = 0

                selected_point9 = (None if selected_record9 is None else
                                   selected_record9.get("visible_point"))
                selected_normal9 = (None if selected_record9 is None else
                                    selected_record9.get("face_normal"))
                ctx9["_committed_instance_id"] = canonical_id9
                ctx9["_chosen_instance_id"] = (
                    canonical_id9 if selected_record9 is not None else None)
                ctx9["_chosen_visible"] = bool(selected_record9 is not None)
                if selected_record9 is not None:
                    raw_cell9 = selected_record9.get("world_position")
                    if isinstance(raw_cell9, list) and len(raw_cell9) == 3:
                        selected_cell9 = tuple(int(q9) for q9 in raw_cell9)
                        (ctx9["_semantic_tx"], ctx9["_semantic_ty"],
                         ctx9["_semantic_tz"]) = selected_cell9
                        ctx9["tx"], ctx9["ty"], ctx9["tz"] = selected_cell9
                stamp_current_structural_fields9(
                    ctx9, class_exist=bool(instances9),
                    chosen_visible=selected_record9 is not None,
                    visible_point=selected_point9,
                    face_normal=selected_normal9,
                    visible_instances=selected_instances9,
                    committed_surface_instance=story9.get(
                        "committed_surface_instance"),
                    confuser_instances=aux9["confuser_instances"],
                    class_census=census_for_stamp9,
                    source=("human_precommit_exact_class_uncommitted"
                            if grounding_unknown_precommit9 else
                            "human_postplay_current_camera+known_geometry"))

                if clicked9 is not None and action_evidence9 is not None:
                    ray_cell9 = [int(q9) for q9 in clicked9["ray_cell"]]
                    ray_point9 = action_evidence9.get("ray_point")
                    if not isinstance(ray_point9, list) or len(ray_point9) != 3:
                        raise RuntimeError("human target click lacks exact ray point")
                    traj9["ray_hit"] = list(ray_cell9)
                    traj9["target_cell"] = list(ray_cell9)
                    traj9["aim_point"] = [float(q9) for q9 in ray_point9]
                    story9["ray_hit"] = list(ray_cell9)
                    story9["target_cell"] = list(ray_cell9)
                    story9["aim_point"] = [float(q9) for q9 in ray_point9]
                    audit9 = {
                        "version": 1,
                        "source": clicked9["source"],
                        "chosen_instance_id": clicked9["chosen_instance_id"],
                        "active_chosen_instance_id": bound_id9,
                        "click_on_active_chosen": bool(click_on_chosen9),
                        "human_marker_missing": False,
                        "ray_cell": list(ray_cell9),
                        "event": ("human_interact_uncommitted"
                                  if bound_id9 is None else
                                  "human_interact_chosen"
                                  if click_on_chosen9 else
                                  "human_interact_off_chosen"),
                        "event_resolution":
                            "postplay_exact_current_rgb_interaction/v2",
                    }
                    traj9["human_target_choice"] = dict(audit9)
                    story9["human_target_choice"] = dict(audit9)
                    if (int(story9["traj_t"]) in success_source_ts9
                            and click_on_chosen9):
                        successful_active_chosen_interactions9 += 1

                exclusion_reasons9 = []
                if "human_selected_interaction_mismatch" in story9:
                    exclusion_reasons9.append(
                        "human_selected_interaction_mismatch")
                if "human_uncommitted_target_interaction" in story9:
                    exclusion_reasons9.append(
                        "human_uncommitted_target_interaction")
                if interaction_attempted9 and clicked9 is None:
                    off_target_interactions9 += 1
                    exclusion_reasons9.append(
                        "human_interaction_not_on_target_surface")
                if exclusion_reasons9:
                    context_audit9 = {
                        "version": 1,
                        "bc_valid_preserved": True,
                        "reasons": list(dict.fromkeys(exclusion_reasons9)),
                        "truth_labels_preserved": True,
                    }
                    traj9["human_interaction_context_audit"] = dict(
                        context_audit9)
                    story9["human_interaction_context_audit"] = dict(
                        context_audit9)
                story9["deferred_visibility"] = {
                    "version": 1, "contract": HUMAN_DEFERRED_LABEL_CONTRACT,
                    "status": "finalized_exact_postplay",
                }
                labelled9 += 1

            post_break_gap9 = False
            for story9 in joined_story9:
                if isinstance(story9.get("human_success_after_action"), dict):
                    post_break_gap9 = True
                marker9 = story9.get("human_chosen_marker")
                if (isinstance(marker9, dict)
                        and marker9.get("accepted") is True):
                    post_break_gap9 = False
                traj9 = traj_by_t9[int(story9["traj_t"])]
                story9["human_post_break_gap"] = int(post_break_gap9)
                traj9["human_post_break_gap"] = int(post_break_gap9)
        finally:
            STRUCT["offline_human_label_row"] = None
            STRUCT.pop("_human_deferred_replay9", None)
            STRUCT.pop("current_label_depth9", None)
            STRUCT.pop("human_frame_depths9", None)
            STRUCT.pop("human_frame_entity_surfaces9", None)
            STORY["rows"] = original_story_rows9
            TRAJ["pending_row"] = original_pending9
            TRAJ["phase"] = original_phase9
            ctx9["_struct_occ"] = original_occ9
            for key9, (present9, value9) in original_replay_ctx9.items():
                if present9:
                    ctx9[key9] = value9
                else:
                    ctx9.pop(key9, None)
            if original_last_y9 is None:
                TRAJ.pop("_last_labeled_y", None)
            else:
                TRAJ["_last_labeled_y"] = original_last_y9

        valid_marker_count9 = sum(marker_counts9[event9] for event9 in (
            "human_commit", "human_switch", "human_reaffirm"))
        marker_evidence_complete9 = bool(
            str(end_reason9) in ("success", "timeout_stat_fired")
            and int(completed9) >= int(ctx9.get("quota", 1))
            and valid_marker_count9 >= int(ctx9.get("quota", 1))
            and marker_counts9["human_marker_rejected"] == 0
            and marker_counts9["human_implicit_commit"] == 0
            and successful_active_chosen_interactions9
                >= int(ctx9.get("quota", 1)))
        marker_complete9 = bool(marker_evidence_complete9)
        marker_summary9 = {
            "version": 1,
            "marker_contract": HUMAN_PIXEL_MARKER_CONTRACT,
            "human_end_reason": str(end_reason9),
            "human_completed": int(completed9),
            "human_marker_event_counts": dict(marker_counts9),
            "human_valid_marker_count": int(valid_marker_count9),
            "human_marker_missing_count": int(
                marker_counts9["human_implicit_commit"]),
            "human_selected_interaction_mismatch_count": int(
                interaction_selected_mismatch9),
            "human_off_target_interaction_count": int(
                off_target_interactions9),
            "human_successful_active_chosen_interaction_count": int(
                successful_active_chosen_interactions9),
            "human_bc_excluded_frame_count": int(
                human_bc_excluded_frames9),
            "human_first_truth_visible_frame": first_truth_visible_index9,
            "human_first_valid_marker_frame": first_valid_marker_index9,
            "human_recognition_latency_frames": (
                None if first_truth_visible_index9 is None
                or first_valid_marker_index9 is None else
                int(first_valid_marker_index9 - first_truth_visible_index9)),
            "human_marker_evidence_complete": bool(
                marker_evidence_complete9),
            "human_marker_contract_complete": bool(marker_complete9),
            "scripted_selfcheck": False,
            "human_terminal_phase": int(joined_story9[-1]["phase"]),
        }
        joined_story9[-1]["human_episode_summary"] = dict(marker_summary9)
        traj_by_t9[int(joined_story9[-1]["traj_t"])][
            "human_episode_summary"] = dict(marker_summary9)

        fh9 = TRAJ.get("fh")
        if fh9 is None:
            raise RuntimeError("human deferred finalizer lost trajectory file")
        for row9 in TRAJ["deferred_human_rows"]:
            fh9.write(json.dumps(row9) + "\n")
        TRAJ["deferred_human_rows"] = []
        STRUCT["defer_human_labels"] = False
        STRUCT["defer_human_start_t"] = None
        elapsed_ms9 = round(1000.0 * (time.perf_counter() - labels_started9), 3)
        print(
            f"[human-label] postplay exact replay labelled={labelled9} "
            f"unknown={unknown9} kept={len(joined_story9)} "
            f"markers={marker_counts9} complete={int(marker_complete9)} "
            f"elapsed_ms={elapsed_ms9}", flush=True)
        return {
            "contract": HUMAN_DEFERRED_LABEL_CONTRACT,
            "labelled_frames": int(labelled9),
            "unknown_frames": int(unknown9),
            "kept_frames": int(len(joined_story9)),
            "wall_ms": float(elapsed_ms9),
            **marker_summary9,
        }

    results = {}
    for cell in [c for c in args.cells.split(",") if c]:
        for setting in [s for s in args.settings.split(",") if s]:
            pose_seed9 = int(args.action_seed if args.pose_seed is None else args.pose_seed)
            layout_seed9 = int(args.action_seed if args.layout_seed is None else args.layout_seed)
            if cell == "mine" and args.benchmark_selected_classswap:
                assignment10 = snapshot_classswap9["assignment"]
                selected_target10 = G._bare(str(
                    assignment10.get("target_kind") or ""))
                selected_distractors10 = tuple(
                    G._bare(str(kind10)) for kind10 in
                    assignment10.get("confuser_classes") or ())
                staged_rows10 = list(
                    assignment10.get("staged_cells") or ())
                if (selected_target10
                        != G._bare(str(args.ore_class or ""))
                        or len(staged_rows10) != 6
                        or sum(row10.get("role") == "target"
                               and G._bare(str(row10.get("after") or ""))
                               == selected_target10
                               for row10 in staged_rows10) != 1
                        or len(selected_distractors10) != 5
                        or len(set(selected_distractors10)) != 5
                        or selected_target10 in selected_distractors10
                        or any(kind10 not in selected_classswap_stage_pool9
                               for kind10 in selected_distractors10)):
                    raise RuntimeError(
                        "selected-world class-swap assignment is not exact "
                        "target1 + distinct ID-confuser5")
                ore_roles9 = {
                    "target": selected_target10,
                    "distractors": selected_distractors10,
                    "seed": None,
                    "contract": (
                        "xbench_mine_shared16_classswap_snapshot/v2"),
                }
            else:
                ore_roles9 = (OC.assign_ore_roles(
                    cell, args.world_seed or 0, layout_seed9,
                    pool=tuple(c for c in args.ore_classes.split(",") if c),
                    n_distractor_classes=int(args.ore_distractor_classes),
                    forced_target=args.ore_class) if cell == "mine" else None)
            if ore_roles9 is not None:
                print(f"[oreclass] {cell}/{setting} target={ore_roles9['target']} "
                      f"distractors={list(ore_roles9['distractors'])} "
                      f"seed={ore_roles9.get('seed', ore_roles9.get('physical_scene_role_seed'))}",
                      flush=True)
            episode_key9 = f"{cell}/{setting}"
            episode_timing9 = {
                "free_site_enabled": bool(args.free_site),
                "free_site_search_wall_s": 0.0,
                "site_offset_budget": (
                    int(args.site_attempts) if args.free_site else 0),
                "site_offsets_attempted": 0,
                "site_accepted_offset_index_1based": None,
                "free_site_exhausted": False,
                "screened_site_cache_mode": "disabled",
                "screened_site_cache_path": None,
                "screened_site_cache_validation_wall_s": 0.0,
                "screened_site_cache_live_audit": None,
                "post_site_pre_record_wall_s": None,
                "episode_recording_wall_s": None,
                "video_save_wall_s": None,
                "episode_total_through_save_wall_s": None,
                "recording_started": False,
                "frames": None,
                "result": None,
            }
            runtime_timing9["episodes"][episode_key9] = episode_timing9
            free9 = False
            timing_site_started9 = time.monotonic()
            if snapshot_restore9 is not None:
                identity9 = dict(snapshot_restore9.get("identity") or {})
                site_seed9 = int(args.site_seed)
                requested_snapshot_target9 = (
                    str(snapshot_classswap9["assignment"][
                        "original_source_target_kind"])
                    if args.benchmark_selected_classswap
                    else str(ore_roles9["target"]))
                expected_identity9 = {
                    "world_seed": int(args.world_seed),
                    "biome": str(args.biome),
                    "cell": str(cell),
                    "setting": str(setting),
                    "site_seed": int(args.site_seed),
                    "layout_seed": int(layout_seed9),
                    "pose_seed": int(pose_seed9),
                    "action_seed": int(args.action_seed),
                    "target_kind": requested_snapshot_target9,
                    "mine_worldgen_profile": str(args.mine_worldgen_profile),
                    "census_scrub": bool(args.census_scrub),
                }
                if identity9 != expected_identity9:
                    raise RuntimeError(
                        "snapshot identity differs from requested episode: "
                        f"snapshot={identity9} requested={expected_identity9}")
                free9 = copy.deepcopy(snapshot_restore9.get("free_site"))
                if not isinstance(free9, dict):
                    raise RuntimeError("snapshot staged state lacks free_site")
                timing_site_finished9 = time.monotonic()
                episode_timing9.update({
                    "screened_site_cache_mode": "staged_world_snapshot",
                    "free_site_search_wall_s": 0.0,
                    "site_offsets_attempted": 0,
                    "site_accepted_offset_index_1based": int(
                        free9.get("attempt", 0) or 0),
                    "free_site_exhausted": False,
                })
                print(
                    f"[snapshot] restored accepted site attempt="
                    f"{free9.get('attempt')} sha="
                    f"{snapshot_bundle9.world.sha256}", flush=True)
            elif args.free_site:
                site_seed9 = (int(args.site_seed) if args.site_seed is not None else zlib.crc32(
                    f"site:{cell}:{setting}:w{args.world_seed}:a{args.action_seed}".encode()))
                free9 = S.screen_free_site(
                    cell, np.random.default_rng(site_seed9), setting=setting,
                    attempts=int(args.site_attempts),
                    mine_target_kind=(None if ore_roles9 is None else
                                      ore_roles9["target"]),
                    mine_confuser_kind=(
                        None if ore_roles9 is None
                        or not ore_roles9["distractors"]
                        else ore_roles9["distractors"][0]))
                timing_site_finished9 = time.monotonic()
                accepted_site_attempt9 = (
                    int(free9["attempt"])
                    if isinstance(free9, dict) and free9.get("attempt") is not None
                    else None)
                episode_timing9.update({
                    "free_site_search_wall_s": runtime_wall_seconds(
                        timing_site_started9, timing_site_finished9),
                    "site_offsets_attempted": (
                        accepted_site_attempt9
                        if accepted_site_attempt9 is not None else
                        int(args.site_attempts)),
                    "site_accepted_offset_index_1based": accepted_site_attempt9,
                    "free_site_exhausted": free9 is None,
                })
                print(
                    f"[timing] {episode_key9} site_mode="
                    f"{episode_timing9['screened_site_cache_mode']} "
                    f"free_site_search="
                    f"{episode_timing9['free_site_search_wall_s']:.3f}s "
                    f"offsets={episode_timing9['site_offsets_attempted']}/"
                    f"{episode_timing9['site_offset_budget']} accepted="
                    f"{accepted_site_attempt9 if accepted_site_attempt9 is not None else 'none'}",
                    flush=True)
                if free9 is None:
                    print(f"[freesite] NO viable site for {cell} (seed {args.world_seed}) "
                          f"after {int(args.site_attempts)} offsets -- DISCARD this world",
                          flush=True)
                    results[f"{cell}/{setting}"] = -1
                    continue
                free9["layout_seed"] = layout_seed9
            if snapshot_restore9 is not None:
                source_ctx9 = copy.deepcopy(snapshot_restore9.get("ctx"))
                if not isinstance(source_ctx9, dict):
                    raise RuntimeError("snapshot staged state lacks runtime ctx")
                kit_item9 = f"{OC.MINE_KIT_ITEM} 1"
                if args.benchmark_selected_classswap:
                    classswap_target9 = G._bare(str(
                        ((snapshot_classswap9 or {}).get("assignment") or {})
                        .get("target_kind") or ""))
                    if classswap_target9 in {
                            "ancient_debris", "obsidian", "crying_obsidian"}:
                        kit_item9 = "minecraft:diamond_pickaxe 1"
                    print(f"[kit] selected-classswap target={classswap_target9} "
                          f"kit={kit_item9}", flush=True)
                S.kit(kit_item9)
                if not S.go(free9["x"], free9["y"], free9["z"], 0, 8, n=8):
                    raise RuntimeError(
                        "snapshot could not reach its accepted site for live audit")
                source_grid9 = S.scan_mine_terrain(span=24)
                source_bounds9 = tuple(
                    int(q9) for q9 in S._last_mine_scan_bounds)
                source_expected_cells9 = {
                    (int(meta9["x"]), int(meta9["y"]), int(meta9["z"])):
                    G._bare(str(meta9["kind"]))
                    for meta9 in (source_ctx9.get("mine_sites") or ())
                }
                if len(source_expected_cells9) != 6:
                    raise RuntimeError(
                        "snapshot scene does not contain six unique staged cells")
                source_wrong_cells9 = [
                    [*cell9, kind9, G._bare(str(source_grid9.get(cell9) or ""))]
                    for cell9, kind9 in source_expected_cells9.items()
                    if G._bare(str(source_grid9.get(cell9) or "")) != kind9
                ]
                source_found_roster9 = {
                    cell9: G._bare(str(raw9))
                    for cell9, raw9 in source_grid9.items()
                    if G._bare(str(raw9)) in benchmark_stage_pool9
                }
                source_unexpected_cells9 = [
                    [*cell9, kind9]
                    for cell9, kind9 in source_found_roster9.items()
                    if source_expected_cells9.get(cell9) != kind9
                ]
                if source_wrong_cells9 or source_unexpected_cells9:
                    raise RuntimeError(
                        "snapshot live source-cell audit failed: "
                        f"wrong={source_wrong_cells9} "
                        f"unexpected={source_unexpected_cells9[:16]}")

                if args.benchmark_selected_classswap:
                    from attacca.evaluation.mine_worlds import canonical_grid_sha256 as _classswap_grid_sha2569
                    from attacca.evaluation.mine_worlds import grid_diff as _classswap_grid_diff9
                    from attacca.evaluation.mine_worlds import pixel_exclusion as _classswap_pixel_exclusion9
                    from attacca.evaluation.mine_worlds import assert_exact_diff as _classswap_assert_exact_diff9
                    from attacca.evaluation.mine_worlds import expected_source_to_variant_diff as _classswap_expected_diff9

                    assignment10 = snapshot_classswap9["assignment"]
                    hydration_cell10 = tuple(
                        int(q10) for q10 in
                        assignment10["coral_hydration_cell"])
                    hydration_block10 = str(
                        assignment10["coral_hydration_block"])
                    hydration_before10 = G._bare(str(
                        source_grid9.get(hydration_cell10) or ""))
                    invalid_hydration_sources10 = {
                        "", "air", "cave_air", "void_air", "water", "lava",
                        "stone_brick_stairs",
                    } | set(benchmark_stage_pool9) | set(
                        selected_classswap_targets9)
                    if hydration_before10 in invalid_hydration_sources10:
                        raise RuntimeError(
                            "selected-world hidden coral hydration backing "
                            "is not a pinned solid terrain cell: "
                            f"{hydration_cell10}="
                            f"{hydration_before10!r}")
                    hydration_command10 = (
                        f"/setblock {hydration_cell10[0]} "
                        f"{hydration_cell10[1]} {hydration_cell10[2]} "
                        f"minecraft:{hydration_block10}")
                    w.cmd(hydration_command10)
                    for _ in range(4):
                        w.step_noop()
                    hydrated_grid10 = S.scan_mine_terrain(span=24)
                    hydrated_bounds10 = tuple(
                        int(q10) for q10 in S._last_mine_scan_bounds)
                    if hydrated_bounds10 != source_bounds9:
                        raise RuntimeError(
                            "coral hydration changed live scan bounds")
                    hydration_actual10 = _classswap_grid_diff9(
                        source_grid9, hydrated_grid10)
                    hydration_expected10 = [{
                        "cell": list(hydration_cell10),
                        "before": hydration_before10,
                        "after": "stone_brick_stairs",
                    }]
                    _classswap_assert_exact_diff9(
                        hydration_actual10, hydration_expected10)
                    source_grid9 = hydrated_grid10
                    commands10 = []
                    for row10 in sorted(
                            assignment10["staged_cells"],
                            key=lambda item10: tuple(item10["cell"])):
                        x10, y10, z10 = (
                            int(q10) for q10 in row10["cell"])
                        kind10 = G._bare(str(row10["after"]))
                        command10 = (
                            f"/setblock {x10} {y10} {z10} "
                            f"minecraft:{kind10}")
                        w.cmd(command10)
                        commands10.append(command10)
                    for _ in range(4):
                        w.step_noop()
                    variant_grid9 = S.scan_mine_terrain(span=24)
                    variant_bounds9 = tuple(
                        int(q9) for q9 in S._last_mine_scan_bounds)
                    if variant_bounds9 != source_bounds9:
                        raise RuntimeError(
                            "selected-world remap changed live scan bounds")
                    actual_diff10 = _classswap_grid_diff9(
                        source_grid9, variant_grid9)
                    expected_diff10 = _classswap_expected_diff9(assignment10)
                    _classswap_assert_exact_diff9(
                        actual_diff10, expected_diff10)
                    ctx = selected_classswap_rewrite_ctx9(
                        source_ctx9, assignment10)
                    expected_cells9 = {
                        (int(meta9["x"]), int(meta9["y"]), int(meta9["z"])):
                        G._bare(str(meta9["kind"]))
                        for meta9 in (ctx.get("mine_sites") or ())
                    }
                    eval_classes10 = set(benchmark_stage_pool9) | {
                        G._bare(str(assignment10["target_kind"]))}
                    variant_found_roster10 = {
                        cell10: G._bare(str(raw10))
                        for cell10, raw10 in variant_grid9.items()
                        if G._bare(str(raw10)) in eval_classes10
                    }
                    variant_wrong10 = [
                        [*cell10, kind10,
                         G._bare(str(variant_grid9.get(cell10) or ""))]
                        for cell10, kind10 in expected_cells9.items()
                        if G._bare(str(variant_grid9.get(cell10) or ""))
                        != kind10
                    ]
                    variant_unexpected10 = [
                        [*cell10, kind10]
                        for cell10, kind10 in
                        variant_found_roster10.items()
                        if expected_cells9.get(cell10) != kind10
                    ]
                    if variant_wrong10 or variant_unexpected10:
                        raise RuntimeError(
                            "selected-world live variant audit failed: "
                            f"wrong={variant_wrong10} "
                            f"unexpected={variant_unexpected10[:16]}")
                    runtime_proof10 = {
                        "version": 1,
                        "contract": (
                            "xbench_mine_shared16_runtime_remap_proof/v2"),
                        "source_snapshot": str(snapshot_bundle9.directory),
                        "source_world_sha256": snapshot_bundle9.world.sha256,
                        "source_world_archive_sha256": (
                            snapshot_bundle9.metadata.get(
                                "world_archive_sha256")),
                        "source_payload_sha256": (
                            snapshot_bundle9.metadata.get("payload_sha256")),
                        "geometry_sha256": snapshot_classswap9[
                            "geometry_sha256"],
                        "assignment_sha256": snapshot_classswap9[
                            "assignment_sha256"],
                        "target_kind": assignment10["target_kind"],
                        "confuser_classes": list(
                            assignment10["confuser_classes"]),
                        "scan_bounds": list(source_bounds9),
                        "source_grid_sha256": _classswap_grid_sha2569(
                            source_grid9),
                        "variant_grid_sha256": _classswap_grid_sha2569(
                            variant_grid9),
                        "mutation_commands": commands10,
                        "coral_hydration_cell": list(hydration_cell10),
                        "coral_hydration_source_block": hydration_before10,
                        "coral_hydration_block": hydration_block10,
                        "coral_hydration_command": hydration_command10,
                        "coral_hydration_actual_diff": hydration_actual10,
                        "coral_hydration_applies_to_all_targets": True,
                        "expected_source_to_variant_diff": expected_diff10,
                        "actual_source_to_variant_diff": actual_diff10,
                        "exact_diff_match": True,
                        "variant_cells": [
                            {
                                "cell": [int(q10) for q10 in cell10],
                                "kind": kind10,
                            }
                            for cell10, kind10 in
                            sorted(expected_cells9.items())
                        ],
                    }
                    runtime_proof_path10 = os.path.join(
                        OUT, "selected_classswap_runtime_proof.json")
                    runtime_proof_temp10 = (
                        f"{runtime_proof_path10}.tmp.{os.getpid()}")
                    with open(runtime_proof_temp10, "w",
                              encoding="utf-8") as proof_f10:
                        json.dump(runtime_proof10, proof_f10, indent=2,
                                  sort_keys=True)
                        proof_f10.write("\n")
                        proof_f10.flush()
                        os.fsync(proof_f10.fileno())
                    os.replace(runtime_proof_temp10,
                               runtime_proof_path10)
                    episode_timing9["selected_world_classswap"] = {
                        **runtime_proof10,
                        "proof_path": runtime_proof_path10,
                    }
                    restored_grid9 = variant_grid9
                    found_roster9 = variant_found_roster10
                    print(
                        "[snapshot] selected-world remap exact_diff=1 "
                        f"target={assignment10['target_kind']} "
                        f"changed={len(actual_diff10)}", flush=True)
                else:
                    ctx = source_ctx9
                    expected_cells9 = source_expected_cells9
                    restored_grid9 = source_grid9
                    found_roster9 = source_found_roster9
                print(
                    f"[snapshot] live scene audit six_cells=6 "
                    f"central_roster={len(found_roster9)}", flush=True)
            else:
                ctx = stage(cell, setting, free_site=free9,
                            ore_class=None if ore_roles9 is None else ore_roles9["target"],
                            confuser_class=(None if ore_roles9 is None
                                            or not ore_roles9["distractors"]
                                            else ore_roles9["distractors"][0]),
                            confuser_classes=(
                                None if ore_roles9 is None else
                                ore_roles9["distractors"]),
                            census_scrub=bool(args.census_scrub and cell == "mine"))
            human_quota_seed9 = None
            if args.human_play:
                ctx["quota"] = 3
            else:
                ctx["quota"] = int(args.quota)
            print(f"[quota] {cell}/{setting} -> {ctx['quota']}", flush=True)
            water_free9 = cell == "scoop_water" and setting == "invis"
            allow_night9 = cell == "sleep"
            start_bright_min9 = 20 if allow_night9 else 25
            pose_dist9 = ctx["dist"]
            if snapshot_restore9 is not None:
                restored_pose9 = copy.deepcopy(
                    snapshot_restore9.get("accepted_start_pose"))
                if (not isinstance(restored_pose9, dict)
                        or any(key9 not in restored_pose9 for key9 in
                               ("x", "y", "z", "yaw", "pitch"))):
                    raise RuntimeError(
                        "snapshot staged state lacks accepted_start_pose")
                restored_pose9["verified"] = True
                S.last_pose_spec = restored_pose9
                pose_ok9 = S.go(
                    float(restored_pose9["x"]),
                    float(restored_pose9["y"]),
                    float(restored_pose9["z"]),
                    float(restored_pose9["yaw"]),
                    float(restored_pose9["pitch"]), n=12)
            elif ctx.get("exact_start"):
                fx2, _, fz2 = ctx["exact_start"]
                dx2, dz2 = fx2 - (ctx["tx"] + 0.5), fz2 - (ctx["tz"] + 0.5)
                b0 = math.degrees(math.atan2(dx2, -dz2))
                pose_dist9 = max(1.0, math.hypot(dx2, dz2))
                pose_ok9 = start_pose(
                    ctx["tx"], ctx["tz"], pose_dist9, setting, jitter=b0,
                    max_retreat=0, require_water_free=water_free9,
                    allow_night=allow_night9, fixed_start=ctx["exact_start"],
                    require_forward_clear=True,
                    fixed_yaw=(ctx.get("fixed_start_yaw")
                               if setting == "invis" else None),
                    allow_enclosed=bool(ctx.get("natural_ore_exposure")))
            else:
                raise RuntimeError("mine context lacks its screened exact_start")
            if not (cell == "scoop_water" and setting == "invis"):
                fade()
            if not pose_ok9 or float(np.asarray(w.info["pov"]).mean()) < start_bright_min9:
                best_mean9 = float(np.asarray(w.info["pov"]).mean())
                print(f"[demo] {cell}/{setting}: no verified natural start "
                      f"(best mean={best_mean9:.1f}) -- DISCARD", flush=True)
                results[f"{cell}/{setting}"] = -1
                continue
            accepted_start_pose9 = dict(S.last_pose_spec or {})
            required_pose_fields9 = ("x", "y", "z", "yaw", "pitch")
            if (accepted_start_pose9.get("verified") is not True
                    or any(key9 not in accepted_start_pose9
                           for key9 in required_pose_fields9)):
                print(f"[pose-restore] {cell}/{setting}: invalid accepted pose contract "
                      f"{accepted_start_pose9!r} -- DISCARD", flush=True)
                results[f"{cell}/{setting}"] = -1
                continue
            STORY_INST = []
            if args.story or (cell == "mine" and ctx["quota"] > 1):
                if cell == "mine":
                    target_cls9 = str(ctx["cls"])
                    have_tgt9 = sum(m9["kind"] == target_cls9
                                    for m9 in ctx.get("mine_sites", []))
                    want_tgt9 = (int(ctx["quota"])
                                 if args.benchmark_selected_classswap
                                 else max(ctx["quota"], 3))
                    missing_tgt9 = max(0, want_tgt9 - have_tgt9)
                    if missing_tgt9:
                        print(
                            f"[oreclass] screened {target_cls9} chain lost "
                            f"{missing_tgt9} targets; story is read-only -- DISCARD",
                            flush=True)
                        results[f"{cell}/{setting}"] = -1
                        continue
                    if args.story and ore_roles9 is not None:
                        if args.benchmark_selected_classswap:
                            confuser_counts10 = {}
                            for site10 in ctx["mine_sites"]:
                                kind10 = str(site10["kind"])
                                if kind10 != target_cls9:
                                    confuser_counts10[kind10] = (
                                        confuser_counts10.get(kind10, 0) + 1)
                            ctx["_ore_cluster_plan"] = {
                                "owner": (
                                    "benchmark_selected_classswap_exact_six_cell_assignment"),
                                "target_class": target_cls9,
                                "target_cell_count": have_tgt9,
                                "target_component_sizes": [1],
                                "distractors": {
                                    kind10: {
                                        "cell_count": count10,
                                        "component_sizes": [1] * count10,
                                    }
                                    for kind10, count10 in sorted(
                                        confuser_counts10.items())
                                },
                                "worldgen_profile": args.mine_worldgen_profile,
                                "selected_classswap_ok": (
                                    have_tgt9 == 1
                                    and len(ctx["mine_sites"]) == 6
                                    and sum(confuser_counts10.values()) == 5),
                            }
                            if not ctx["_ore_cluster_plan"][
                                    "selected_classswap_ok"]:
                                raise RuntimeError(
                                    "selected-world target1/confuser5 shape drift")
                        else:
                            ctx["_ore_cluster_plan"] = mine_scene_shape_audit(
                                ctx["mine_sites"], target_cls9,
                                ore_roles9["distractors"],
                                worldgen_profile=args.mine_worldgen_profile)
                        if (v12_multiclass6_mode9
                                and not args.benchmark_selected_classswap
                                and not ctx["_ore_cluster_plan"].get(
                                    "independent_multiclass6_ok")):
                            print(
                                f"[mine-shape] invalid independent multiclass6 "
                                f"{ctx['_ore_cluster_plan']} -- DISCARD",
                                flush=True)
                            results[f"{cell}/{setting}"] = -1
                            continue
                    for meta9 in ctx["mine_sites"]:
                        STORY_INST.append(dict(
                            kind=meta9["kind"], x=meta9["x"], y=meta9["y"],
                            z=meta9["z"],
                            role=("target" if meta9["kind"] == target_cls9
                                  else "distractor"),
                            source=meta9.get("source"),
                            contacts=meta9.get("contacts"),
                            exposed=meta9.get("exposed"),
                            side_exposed=meta9.get("side_exposed"),
                            support=meta9.get("support"),
                            placement_geometry=meta9.get(
                                "placement_geometry"),
                            screened_placement_geometry=meta9.get(
                                "screened_placement_geometry"),
                            placement_geometry_match=meta9.get(
                                "placement_geometry_match"),
                            accessible=meta9.get("accessible"),
                            access_dist=meta9.get("access_dist"),
                            access_pos=meta9.get("access_pos"),
                            survey_checkpoint_step=meta9.get(
                                "survey_checkpoint_step"),
                            survey_heading_error=meta9.get(
                                "survey_heading_error")))
                    staged_coal9 = sum(m9["kind"] == target_cls9 for m9 in ctx["mine_sites"])
                    if staged_coal9 < ctx["quota"]:
                        print(f"[quota] mine needs {ctx['quota']} exposed ores, found {staged_coal9} "
                              "-- DISCARD", flush=True)
                        results[f"{cell}/{setting}"] = -1
                        continue
                for _ in range(4):
                    w.step_noop()
            hp = float(w.info.get("health", 20) or 20)
            if hp < 19.5:
                w.cmd("/effect give @a minecraft:regeneration 4 255 true")
                for recovery_tick9 in range(240):
                    w.step_noop()
                    hp = float(w.info.get("health", 20) or 20)
                    if hp >= 19.5 and recovery_tick9 > 20:
                        break
            w.cmd("/effect clear @a")
            w.cmd("/effect give @a minecraft:resistance 999999 4 true")
            for _ in range(4):
                w.step_noop()
            restore_ok9 = S.go(
                float(accepted_start_pose9["x"]),
                float(accepted_start_pose9["y"]),
                float(accepted_start_pose9["z"]),
                float(accepted_start_pose9["yaw"]),
                float(accepted_start_pose9["pitch"]), n=8)
            restored_values9 = (*w.get_pos(), w.get_yaw(), w.get_pitch())
            expected_values9 = tuple(
                float(accepted_start_pose9[key9])
                for key9 in required_pose_fields9)
            restored_match9 = (
                restore_ok9
                and all(round(float(restored_values9[i9]), 2)
                        == round(expected_values9[i9], 2)
                        for i9 in range(3))
                and round(wrap(float(restored_values9[3]) - expected_values9[3]), 2)
                    == 0.0
                and round(float(restored_values9[4]) - expected_values9[4], 2)
                    == 0.0)
            if not restored_match9:
                print(f"[pose-restore] {cell}/{setting}: expected={expected_values9} "
                      f"observed={restored_values9} acknowledged={restore_ok9} "
                      "-- DISCARD", flush=True)
                results[f"{cell}/{setting}"] = -1
                continue
            staged_rgb9 = np.ascontiguousarray(
                np.asarray(w.info["pov"], dtype=np.uint8))
            staged_rgb_sha9 = hashlib.sha256(staged_rgb9.tobytes()).hexdigest()
            if args.benchmark_selected_classswap:
                target_cells10 = tuple(
                    tuple(int(q10) for q10 in row10["cell"])
                    for row10 in snapshot_classswap9["assignment"][
                        "staged_cells"]
                    if row10.get("role") == "target")
                if len(target_cells10) != 1:
                    raise RuntimeError(
                        "selected-world start proof lost singular target identity")
                proof_occ10 = S.mine_occupancy(
                    restored_grid9, source_bounds9)
                proof_eye10 = (
                    float(accepted_start_pose9["x"]),
                    float(accepted_start_pose9["y"]) + G.EYE,
                    float(accepted_start_pose9["z"]),
                )
                proof_scene10 = G.shared_visible_surface_scene(
                    proof_occ10, proof_eye10, cutout_as_full_cube=False)
                target_mask10, _point10, _bbox10, mask_meta10 = (
                    G.visible_surface_mask_blocks(
                        proof_eye10,
                        float(accepted_start_pose9["yaw"]),
                        float(accepted_start_pose9["pitch"]),
                        target_cells10,
                        proof_occ10,
                        extra_visual_solid=proof_scene10[
                            "visual_extra_solid"],
                        alpha_cutout_visual=proof_scene10[
                            "alpha_cutout_visual"],
                        thin_visual_solid=proof_scene10[
                            "thin_visual_solid"],
                        unknown_is_solid=True,
                        pixel_exclusion_mask=_classswap_pixel_exclusion9(
                            w.info)))
                target_mask10 = np.ascontiguousarray(
                    np.asarray(target_mask10, dtype=np.uint8))
                target_pixels10 = int(np.count_nonzero(target_mask10))
                target_mask_path10 = os.path.join(
                    OUT, "snapshot_restored_target_mask.png")
                if not cv2.imwrite(target_mask_path10, target_mask10 * 255):
                    raise RuntimeError(
                        "failed to write selected-world start target mask")
                start_proof10 = {
                    "version": 1,
                    "contract": (
                        "xbench_mine_shared16_start_hidden_proof/v2"),
                    "target_kind": snapshot_classswap9["target_kind"],
                    "target_cells": [list(cell10)
                                     for cell10 in target_cells10],
                    "start_pose": [
                        float(accepted_start_pose9[key10])
                        for key10 in ("x", "y", "z", "yaw", "pitch")],
                    "rgb_sha256": staged_rgb_sha9,
                    "mask_array_sha256": hashlib.sha256(
                        target_mask10.tobytes()).hexdigest(),
                    "exact_visible_surface_pixels": target_pixels10,
                    "binary_start_hidden": target_pixels10 == 0,
                    "surface_visibility_contract": (
                        "exact_actual_visible_surface_zero_pixels/v1"),
                    "tested_ray_count": int(mask_meta10.get(
                        "tested_ray_count", 0)),
                    "candidate_pixel_count": int(mask_meta10.get(
                        "candidate_pixel_count", 0)),
                    "mask_path": target_mask_path10,
                }
                start_proof_path10 = os.path.join(
                    OUT, "selected_classswap_start_hidden_proof.json")
                start_proof_temp10 = (
                    f"{start_proof_path10}.tmp.{os.getpid()}")
                with open(start_proof_temp10, "w",
                          encoding="utf-8") as proof_f10:
                    json.dump(start_proof10, proof_f10, indent=2,
                              sort_keys=True)
                    proof_f10.write("\n")
                    proof_f10.flush()
                    os.fsync(proof_f10.fileno())
                os.replace(start_proof_temp10, start_proof_path10)
                episode_timing9["selected_world_classswap_start_hidden"] = {
                    **start_proof10,
                    "proof_path": start_proof_path10,
                }
                if target_pixels10 != 0:
                    raise RuntimeError(
                        "selected-world invisible start exposes target pixels: "
                        f"target={snapshot_classswap9['target_kind']} "
                        f"pixels={target_pixels10}")
                print(
                    "[snapshot] selected-world start target surface=0px",
                    flush=True)
            if snapshot_restore9 is not None:
                import cv2 as _snapshot_cv2_9
                restored_rgb_path9 = os.path.join(
                    OUT, "snapshot_restored_ready.png")
                if not _snapshot_cv2_9.imwrite(
                        restored_rgb_path9, staged_rgb9[:, :, ::-1]):
                    raise RuntimeError("failed to write snapshot restored RGB")
                expected_rgb_sha9 = str(
                    snapshot_restore9.get("staged_rgb_sha256") or "")
                comparison_rgb_sha9 = expected_rgb_sha9
                rgb_exact9 = bool(
                    comparison_rgb_sha9
                    and comparison_rgb_sha9 == staged_rgb_sha9)
                episode_timing9["snapshot_restore"] = {
                    "bundle": str(snapshot_bundle9.directory),
                    "world_sha256": snapshot_bundle9.world.sha256,
                    "source_rgb_sha256": expected_rgb_sha9,
                    "restored_rgb_sha256": staged_rgb_sha9,
                    "comparison_rgb_sha256": comparison_rgb_sha9,
                    "comparison_kind": "snapshot_generation_source",
                    "source_rgb_byte_exact": bool(
                        expected_rgb_sha9 == staged_rgb_sha9),
                    "rgb_byte_exact": rgb_exact9,
                    "live_six_cell_audit": True,
                }
                print(
                    f"[snapshot] restored_rgb={restored_rgb_path9} "
                    f"byte_exact={int(rgb_exact9)}", flush=True)
                if args.policy_driver and not rgb_exact9:
                    episode_timing9["snapshot_restore"][
                        "rgb_mismatch_nonfatal"] = True
                    print(
                        "[snapshot] ready RGB not canonical; continuing",
                        flush=True)
            if args.world_snapshot_out:
                if (args.site_seed is None or args.layout_seed is None
                        or args.pose_seed is None or args.world_seed is None):
                    raise RuntimeError(
                        "snapshot publication requires explicit world/site/layout/pose seeds")
                import cv2 as _snapshot_cv2_9
                source_rgb_path9 = os.path.join(
                    OUT, "snapshot_source_ready.png")
                if not _snapshot_cv2_9.imwrite(
                        source_rgb_path9, staged_rgb9[:, :, ::-1]):
                    raise RuntimeError("failed to write snapshot source RGB")
                snapshot_payload9 = {
                    "contract": "xbench-v2-demo-staged-state/v1",
                    "identity": {
                        "world_seed": int(args.world_seed),
                        "biome": str(args.biome),
                        "cell": str(cell),
                        "setting": str(setting),
                        "site_seed": int(args.site_seed),
                        "layout_seed": int(layout_seed9),
                        "pose_seed": int(pose_seed9),
                        "action_seed": int(args.action_seed),
                        "target_kind": str(ore_roles9["target"]),
                        "mine_worldgen_profile": str(args.mine_worldgen_profile),
                        "census_scrub": bool(args.census_scrub),
                    },
                    "world_origin": tuple(float(v9) for v9 in S.world_origin),
                    "free_site": copy.deepcopy(free9),
                    "ctx": copy.deepcopy(ctx),
                    "accepted_start_pose": copy.deepcopy(
                        accepted_start_pose9),
                    "staged_rgb_sha256": staged_rgb_sha9,
                    "staged_rgb_shape": tuple(int(v9) for v9 in staged_rgb9.shape),
                }
                published9 = publish_world_snapshot_bundle(
                    w, Path(args.world_snapshot_out).resolve(),
                    payload=snapshot_payload9)
                episode_timing9["snapshot_publish"] = {
                    "bundle": str(published9.directory),
                    "world_sha256": published9.world.sha256,
                    **dict(published9.metadata.get("producer_timing") or {}),
                }
                print(
                    f"[snapshot] published={published9.directory} "
                    f"sha={published9.world.sha256} bytes="
                    f"{published9.metadata['world_size_bytes']}", flush=True)
                if args.world_snapshot_only:
                    episode_timing9["post_site_pre_record_wall_s"] = (
                        runtime_wall_seconds(
                            timing_site_finished9, time.monotonic()))
                    episode_timing9["result"] = "snapshot_published"
                    results[f"{cell}/{setting}"] = 1
                    continue
            m_start = float(np.asarray(w.info["pov"]).mean())
            timing_recording_started9 = time.monotonic()
            episode_timing9["post_site_pre_record_wall_s"] = runtime_wall_seconds(
                timing_site_finished9, timing_recording_started9)
            episode_timing9["recording_started"] = True
            rec = Rec(f"{'policy' if args.policy_driver else 'demo'}_{cell}_{setting}",
                      thresh=min(25.0, max(8.0, 0.4 * m_start)))
            STORY["rec"] = rec if args.story else None
            STORY["pending_event"] = ""
            STORY["pending_struct_fields"] = None
            STORY["rows"] = []
            TRAJ["fh"] = open(os.path.join(OUT, f"traj_{cell}_{setting}.jsonl"), "w")
            TRAJ.update(t=0, note="episode start", phase=0, event="", bc_valid=1,
                        last_phase=0, last_event="", last_bc_valid=1, pending_row=None,
                        ray_hit=None, target_cell=None, aim_point=None,
                        viewmodel_remaining_controls=0,
                        motion_phase="explore", post_break=0,
                        mine_attack_distance_threshold=-1.0,
                        stuck_recovery_demo=0,
                        stuck_recovery_stage="none",
                        deferred_human_rows=[], human_preaction_evidence=None)
            STRUCT["visibility_cache"] = {}
            STRUCT["defer_human_labels"] = False
            STRUCT["defer_human_start_t"] = None
            STRUCT["offline_human_label_row"] = None
            STRUCT["human_episode_summary"] = None
            _grab0 = rec.grab

            def _grab_noted(note="", _g=_grab0):
                TRAJ["note"] = note
                return _g(note)

            rec.grab = _grab_noted
            if args.story:
                sfh = open(os.path.join(OUT, f"story_{cell}_{setting}.jsonl"), "w")
                site_meta9 = ({k9: free9[k9] for k9 in ("x", "y", "z", "attempt", "flatness")
                               if k9 in free9} if isinstance(free9, dict) else None)
                story_timing9 = {
                    "schema_version": runtime_timing9["schema_version"],
                    "clock": runtime_timing9["clock"],
                    "units": runtime_timing9["units"],
                    "boot_world_wall_s": runtime_timing9["boot_world_wall_s"],
                    "prep_world_wall_s": runtime_timing9["prep_world_wall_s"],
                    "main_to_ready_wall_s": runtime_timing9["main_to_ready_wall_s"],
                    "free_site_search_wall_s": episode_timing9[
                        "free_site_search_wall_s"],
                    "site_offset_budget": episode_timing9["site_offset_budget"],
                    "site_offsets_attempted": episode_timing9[
                        "site_offsets_attempted"],
                    "site_accepted_offset_index_1based": episode_timing9[
                        "site_accepted_offset_index_1based"],
                    "screened_site_cache_mode": episode_timing9[
                        "screened_site_cache_mode"],
                    "screened_site_cache_validation_wall_s": episode_timing9[
                        "screened_site_cache_validation_wall_s"],
                    "screened_site_cache_live_audit": episode_timing9[
                        "screened_site_cache_live_audit"],
                    "post_site_pre_record_wall_s": episode_timing9[
                        "post_site_pre_record_wall_s"],
                    "final_summary_file": "runtime_timing.json",
                }
                story_header_payload9 = {"manifest": STORY_INST, "cell": cell,
                                      "setting": setting, "quota": ctx["quota"],
                                      "episode_source": (
                                          "human_play" if args.human_play else "oracle"),
                                      "oracle_motion_profile": (
                                          HUMAN_MOTION_PROFILE
                                          if args.human_play else MOTION_PROFILE),
                                      "human_session_contract": (
                                          HUMAN_SESSION_CONTRACT
                                          if args.human_play else None),
                                      "human_quota_seed": (
                                          human_quota_seed9
                                          if args.human_play else None),
                                      "ore_host_variant": ctx.get(
                                          "ore_host_variant"),
                                      "anchored_ore_host": None,
                                      "oracle_motion_schema": {
                                          "version": 4,
                                          "phase_field": "motion_phase",
                                          "post_break_field": "post_break",
                                          "drop_collect_field": "drop_collect",
                                          "pickup_confirmed_field":
                                              "pickup_confirmed",
                                          "pickup_success_evidence":
                                              "positive_expected_item_inventory_delta",
                                          "pickup_required_after_every_break": False,
                                          "pickup_terminal_outcome_required":
                                              "confirmed_xor_explicitly_abandoned",
                                          "pickup_abandonment_fields": [
                                              "pickup_abandoned",
                                              "pickup_abandon_reason",
                                              "pickup_attempt_controls",
                                              "pickup_attribution_valid"],
                                          "pickup_collect_timeout_controls":
                                              MINE_DROP_COLLECT_MAX_TICKS,
                                          "attack_distance_field":
                                              "mine_attack_distance_threshold",
                                          "stuck_recovery_rate": 0.0,
                                          "post_break_ticks": [
                                              MINE_POST_BREAK_MIN_TICKS,
                                              MINE_POST_BREAK_MAX_TICKS],
                                      },
                                      "pov_recording_schema": {
                                          "version": 1,
                                          "filename": f"{rec.tag}.mp4",
                                          "raw_unannotated": True,
                                          "review_annotations_external": True,
                                      },
                                      "human_control_schema": ({
                                          "version": 1,
                                          "enabled": True,
                                          "source": "native_keyboard_mouse",
                                          "recorder_contract":
                                              HUMAN_EXACT_RECORDER_CONTRACT,
                                          "visibility_timing_contract":
                                              HUMAN_DEFERRED_LABEL_CONTRACT,
                                          "live_loop_dense_visibility": False,
                                          "live_loop_target_only_exact_depth_"
                                          "visibility": True,
                                          "postplay_exact_visibility": True,
                                          "renderer_viewmodel_mask": {
                                              "contract":
                                                  RENDERER_VIEWMODEL_MASK_CONTRACT,
                                              "transport":
                                                  "same_tick_hand_only_alpha_"
                                                  "split_from_clean_rgb",
                                              "action_marker":
                                                  RENDERER_VIEWMODEL_ACTION_MARKER,
                                              "pixel_supervision_during_attack": True,
                                          },
                                          "start_gate": {
                                              "key": "V",
                                              "before_first_recorded_rgb": True,
                                              "simulator_steps_while_waiting": 0,
                                              "frames_recorded_while_waiting": 0,
                                              "timeout_starts_after_gate": True,
                                          },
                                          "chosen_marker": {
                                              "key": "C",
                                              "resume_key": "V",
                                              "transport": (
                                                  "freeze_current_rgb_then_ui_pixel_"
                                                  "sidecar_not_env_action"),
                                              "capture_toggle_key": "TAB",
                                              "selection_source":
                                                  "clicked_pixel_membership_in_exact_"
                                                  "visible_surface_mask_that_passes_"
                                                  "the_current_model_resolution_"
                                                  "recognizability_gate",
                                              "coordinate_contract":
                                                  HUMAN_PIXEL_MARKER_CONTRACT,
                                              "semantic": (
                                                  "human_recognized_and_intends_"
                                                  "this_visual_instance_not_"
                                                  "physical_reachability"),
                                              "annotation_simulator_steps": 0,
                                              "annotation_frames_recorded": 0,
                                              "break_auto_freeze": True,
                                              "visible_streak_auto_freeze": {
                                                  "contract":
                                                      HUMAN_AUTO_COMMIT_VISIBILITY_CONTRACT,
                                                  "consecutive_frames":
                                                      HUMAN_AUTO_COMMIT_VISIBLE_FRAMES,
                                                  "automatic_choice": False,
                                                  "c_press_cancels_pending_prompt": True,
                                                  "visible_candidate_requires_click_"
                                                  "before_v": True,
                                              },
                                              "quota_freeze_then_natural_tail": False,
                                              "quota_freeze_v_to_save": True,
                                              "automatic_nearest_commit_before_marker": False,
                                              "canonical_chosen_before_marker": None,
                                              "precommit_geometry_is_audit_only": True,
                                              "canonical_phase_boundary":
                                                  "accepted_C_pixel",
                                              "rejected_click_semantics": {
                                                  "canonical_state_change": False,
                                                  "environment_action": False,
                                                  "below_recognizability_gate":
                                                      "reject_and_reclick_on_same_"
                                                      "frozen_rgb",
                                                  "reason_field":
                                                      "human_chosen_marker.reason",
                                                  "human_recognition_override": False,
                                              },
                                              "precommit_canonical_state": {
                                                  "phase": "EXPLORE",
                                                  "chosen_instance_id": None,
                                                  "bc_valid": True,
                                                  "class_exist":
                                                      "exact_class_visibility",
                                                  "class_census": "exact_12_class",
                                                  "intended_instance_identity":
                                                      "UNKNOWN_until_C",
                                              },
                                              "implicit_click_fallback_audited": False,
                                              "uncommitted_target_interaction":
                                                  "physical_truth_only_bc_invalid_"
                                                  "chosen_stays_null",
                                          },
                                          "exact_target_roster": (
                                              "structural_blocks_and_fixed_components;"
                                              "mob_targets_fail_closed"),
                                          "scripted_selfcheck": False,
                                          "training_data_valid": True,
                                          "policy_replaced_only": True,
                                      } if args.human_play else None),
                                      "phase_schema": {"explore": 0, "approach": 1,
                                                       "interact": 2},
                                      "decision_events": [
                                          "commit", "switch", "seam", "seam_commit",
                                          "commit_abandon_unreachable"],
                                      "human_marker_events": [
                                          "human_commit", "human_switch",
                                          "human_reaffirm",
                                          "human_marker_rejected",
                                          "human_uncommitted_target_interaction",
                                          "human_clear_after_success_or_removal"],
                                      "human_decision_event_schema": ({
                                          "version": 1,
                                          "row_field": "human_decision_events",
                                          "session_field": "decision_events",
                                          "same_rgb_multiple_events": True,
                                          "events": [
                                              "freeze_enter",
                                              "selection_rejected",
                                              "selection_accepted",
                                              "resume",
                                              "break_detected",
                                              "abort",
                                              "timeout",
                                          ],
                                          "freeze_reasons": [
                                              "c_press", "visible_streak",
                                              "break_auto", "quota",
                                              "manual_pause"],
                                      } if args.human_play else None),
                                      "visible_instance_schema": {
                                          "version": (
                                              3 if args.visible_surface_masks else 1),
                                          "scope": (
                                              ("all_current_oracle_recognizable_goal_instances_"
                                               "with_exact_actual_visible_surface_masks_within_"
                                               "the_exact_per_frame_visibility_scan_scope"
                                               if args.visible_surface_masks else
                                               "all_current_perceptible_goal_instances_within_"
                                               "the_exact_per_frame_visibility_scan_scope")),
                                          "scan_volume": {
                                              "role": "nominal_explore_scan_request",
                                              "horizontal_radius": (
                                                  40 if ctx.get("cls") == "water" else 24),
                                              "relative_y_min": (
                                                  -18 if is_mine_target_kind9(
                                                      ctx.get("cls")) else -6),
                                              "relative_y_max_exclusive": (
                                                  19 if is_mine_target_kind9(
                                                      ctx.get("cls")) else 7),
                                              "vertical_packet_height_max": 13,
                                          },
                                          "per_frame_scope_field": "visibility_scan_scope",
                                          "committed_surface_tracking_contract": (
                                              COMMITTED_SURFACE_TRACKING_CONTRACT
                                              if args.visible_surface_masks else None),
                                          "chosen_contract": (
                                              "chosen_visible=1 iff exactly one visible instance "
                                              "has chosen=true; chosen_instance_id matches it"),
                                          "committed_surface_contract": ({
                                              "version": 1,
                                              "fresh_recognition_fields": [
                                                  "class_recognizable",
                                                  "class_exist",
                                                  "visible_instances",
                                              ],
                                              "identity_memory_fields": [
                                                  "target_committed",
                                                  "committed_instance_id",
                                                  "committed_target_cell",
                                              ],
                                              "current_surface_fields": [
                                                  "chosen_surface_visible",
                                                  "chosen_surface_instance_id",
                                                  "chosen_surface_source",
                                                  "committed_surface_instance",
                                              ],
                                              "rule": (
                                                  "fresh commits require a recognizable visible_"
                                                  "instances record; after commit, the exact same "
                                                  "instance may publish an occlusion-aware nonempty "
                                                  "surface mask below that threshold without "
                                                  "becoming selectable or changing class_exist"),
                                              "fully_occluded": (
                                                  "target_committed remains 1 while chosen_surface_"
                                                  "visible=0 and committed_surface_instance=null"),
                                          } if args.visible_surface_masks else None),
                                          "mask_contract": (
                                              "chosen_instance_actual_visible_surface/v1; dense "
                                              "native-pixel union; terrain/entity occlusion aware; "
                                              "render-pose registered; no convex bridging"
                                              if args.visible_surface_masks else
                                              "certified tested-geometry first-hit pixel lower "
                                              "bound over generator proxy geometry; not full "
                                              "renderer segmentation"),
                                          "render_pose_contract": (
                                              {"version": 1,
                                               "per_frame_field": "render_pose",
                                               "source":
                                                   VISIBLE_SURFACE_RENDER_POSE_SOURCE,
                                               "endpoint_pose_retained_separately": True}
                                              if args.visible_surface_masks else None),
                                          "recognition_contract": (
                                              structural_recognition_contract()
                                              if args.visible_surface_masks else None),
                                          "viewmodel_occlusion_contract": (
                                              viewmodel_occlusion_contract()
                                              if args.visible_surface_masks else None),
                                          "renderer_viewmodel_mask_contract": ({
                                              "contract":
                                                  RENDERER_VIEWMODEL_MASK_CONTRACT,
                                              "action_marker":
                                                  RENDERER_VIEWMODEL_ACTION_MARKER,
                                              "pixel_supervision_during_attack":
                                                  True,
                                              "raw_pov_channels": 3,
                                          } if args.visible_surface_masks else None),
                                          "mob_identity_contract": (
                                              "chosen identity is persistent when tracker proof "
                                              "permits; unchosen mob instance IDs are frame-local"),
                                          "mob_observation_box": list(
                                              STRUCTURAL_MOB_QUERY_BOX),
                                          "bow_hostile_population_contract": (
                                              "doMobSpawning=false; reset kills all nonplayers; "
                                              "stage summons exactly one tagged zombie, so the "
                                              "unique selector is the complete zombie goal set"),
                                          "fixed_vegetation_model": (
                                              "plain mob classes use vendored vanilla-1.16.5 "
                                              "crossed-quad base-texture alpha with deterministic "
                                              "renderer block offsets for modeled grass/fern "
                                              "plants; mob shape remains an AABB proxy; exact-"
                                              "attribute selectors and unsupported cutouts remain "
                                              "fail-closed full cubes"),
                                      },
                                          "class_visibility_schema": ({
                                              **class_visibility_roster_contract(
                                              mine_census_roster9),
                                          "state_encoding": {
                                              "visible": "known=1,visible=1",
                                              "not_visible": "known=1,visible=0",
                                              "unknown": "known=0,visible=0",
                                          },
                                          "known_bits_field": (
                                              "class_visibility_known_bits"),
                                          "visible_bits_field": (
                                              "class_visibility_visible_bits"),
                                          "sparse_instances_field": (
                                              "class_visible_instances"),
                                          "class_union_masks_field": (
                                              "class_union_masks"),
                                          "class_union_scope": (
                                              "all_current_oracle_recognizable_"
                                              "instances_per_visible_class"),
                                          "absence_witness_field": (
                                              "class_visibility_scan_witness"),
                                          "recognition_domain": (
                                              "conservative_render_distance_12_"
                                              "192_block_radius_world_height"),
                                          "partial_scan_policy": ((
                                              "positive_is_known; scrub_certified_"
                                              "absent_is_known_not_visible; other_"
                                              "nonvisible_is_unknown")
                                              if args.census_scrub else (
                                              "positive_is_known; nonvisible_is_unknown")),
                                      } if args.visible_surface_masks
                                      and args.all_class_census
                                      and is_mine_target_kind9(
                                          ctx.get("cls")) else None),
                                      "all_class_census_enabled": bool(
                                          args.visible_surface_masks
                                          and args.all_class_census
                                          and is_mine_target_kind9(
                                              ctx.get("cls"))),
                                      "world_seed": args.world_seed, "action_seed": args.action_seed,
                                      "runtime_timing": story_timing9,
                                      "world_origin": {
                                          "x": float(S.world_origin[0]),
                                          "y": float(S.world_origin[1]),
                                          "z": float(S.world_origin[2]),
                                          "coordinate_frame": "minecraft_absolute",
                                          "capture": "post_boot_pre_free_site",
                                      },
                                      "site_seed": (site_seed9 if args.free_site else None),
                                      "layout_seed": layout_seed9,
                                      "pose_seed": pose_seed9,
                                      "counterfactual_scene_contract": None,
                                      "reciprocal_goal_scene_contract": None,
                                      "counterfactual_goal_kind": None,
                                      "counterfactual_pair_group_id": None,
                                      "counterfactual_goal_only_selector": False,
                                      "counterfactual_role_assignment": None,
                                      "interaction_ids": None,
                                      "requested_layout_seed": args.layout_seed,
                                      "requested_pose_seed": args.pose_seed,
                                      "diagnostic_only": False,
                                      "natural_start_pose": None,
                                      "biome": args.biome,
                                      "start_pose": accepted_start_pose9,
                                      "site": site_meta9, "terrain": None,
                                      "mine_survey": ({
                                          "source": "terrain_only_pre_target",
                                          "route": ctx.get("mine_survey_route", ()),
                                          "checkpoint_steps": ctx.get(
                                              "mine_survey_checkpoints", ()),
                                      } if cell == "mine" else None),
                                      "mine_visible_start": ctx.get(
                                          "mine_visible_start"),
                                      "target_kind": ctx.get(
                                          "target_kind", ctx.get("cls")),
                                      "ore_class_draw": ore_roles9,
                                      "ore_cluster_plan": ctx.get("_ore_cluster_plan"),
                                      "mine_counterfactual_pair": ctx.get(
                                          "mine_counterfactual_pair"),
                                      "mine_scene_layout": ctx.get(
                                          "mine_scene_layout"),
                                      "mine_worldgen_profile": str(
                                          args.mine_worldgen_profile),
                                      "mine_controller_profile": "v2_continuous",
                                      "biome_terrain_topology": ctx.get(
                                          "biome_terrain_topology"),
                                      "mine_confuser": ctx.get("mine_confuser"),
                                      "target": {
                                          "x": ctx.get("_semantic_tx", ctx.get("tx")),
                                          "y": ctx.get("_semantic_ty", ctx.get("ty")),
                                          "z": ctx.get("_semantic_tz", ctx.get("tz"))}}
                if args.census_scrub and ctx.get("census_scrub") is not None:
                    story_header_payload9["census_scrub_witness"] = (
                        ctx["census_scrub"])
                sfh.write(json.dumps(story_header_payload9) + "\n")
                _g2 = rec.grab
                def _grab_story(note="", _g=_g2, _ctx=ctx):
                    kept0 = len(rec.frames)
                    pending_before_grab9 = TRAJ.get("pending_row")
                    r = _g(note)
                    if len(rec.frames) == kept0:
                        if (isinstance(pending_before_grab9, dict)
                                and pending_before_grab9.get("step_kind") == "control"
                                and int(pending_before_grab9.get(
                                    "bc_valid", 0)) == 1):
                            raise RuntimeError(
                                "bc-valid control RGB was not preserved by recorder: "
                                f"t={pending_before_grab9.get('t')}")
                        return r
                    ax8, ay8, az8 = w.get_pos()
                    event8 = str(STORY.get("pending_event", ""))
                    STORY["pending_event"] = ""
                    pending8 = TRAJ.get("pending_row")
                    if pending8 is None or int(pending8["t"]) != max(0, TRAJ["t"] - 1):
                        raise RuntimeError("story/traj pending-row alignment failure")
                    pending8["note"] = note
                    semantic_tx8 = _ctx.get("_semantic_tx", _ctx.get("tx"))
                    semantic_ty8 = _ctx.get("_semantic_ty", _ctx.get("ty"))
                    semantic_tz8 = _ctx.get("_semantic_tz", _ctx.get("tz"))
                    story_row8 = {"f": len(rec.frames) - 1, "note": note,
                        "traj_t": int(pending8["t"]),
                        "phase": int(pending8["phase"]), "decision_event": event8,
                        "bc_valid": int(pending8["bc_valid"]),
                        "motion_phase": str(pending8.get(
                            "motion_phase", "explore")),
                        "post_break": int(pending8.get("post_break", 0)),
                        "drop_collect": int(pending8.get(
                            "drop_collect", 0)),
                        "drop_collect_stage": str(pending8.get(
                            "drop_collect_stage", "none")),
                        "pickup_expected_items": list(pending8.get(
                            "pickup_expected_items") or ()),
                        "pickup_inventory_delta": copy.deepcopy(pending8.get(
                            "pickup_inventory_delta") or {}),
                        "pickup_counter_delta": copy.deepcopy(pending8.get(
                            "pickup_counter_delta") or {}),
                        "pickup_confirmed": int(pending8.get(
                            "pickup_confirmed", 0)),
                        "pickup_latency_controls": int(pending8.get(
                            "pickup_latency_controls", -1)),
                        "pickup_abandoned": int(pending8.get(
                            "pickup_abandoned", 0)),
                        "pickup_abandon_reason": str(pending8.get(
                            "pickup_abandon_reason", "none")),
                        "pickup_attempt_controls": int(pending8.get(
                            "pickup_attempt_controls", -1)),
                        "pickup_attribution_valid": int(pending8.get(
                            "pickup_attribution_valid", 0)),
                        "pickup_navigation_source": str(pending8.get(
                            "pickup_navigation_source", "none")),
                        "pickup_visible_preempt_proof": copy.deepcopy(
                            pending8.get("pickup_visible_preempt_proof")),
                        "mine_attack_distance_threshold": float(pending8.get(
                            "mine_attack_distance_threshold", -1.0)),
                        "stuck_recovery_demo": int(pending8.get(
                            "stuck_recovery_demo", 0)),
                        "stuck_recovery_stage": str(pending8.get(
                            "stuck_recovery_stage", "none")),
                        "x": round(ax8, 2), "y": round(ay8, 2), "z": round(az8, 2),
                        "yaw": round(w.get_yaw(), 2), "pitch": round(w.get_pitch(), 2),
                        "tx": semantic_tx8, "ty": semantic_ty8, "tz": semantic_tz8,
                        "class_exist": 0, "visible_instances": [],
                        "visibility_supervision_valid": 1,
                        "chosen_visible": 0,
                        "chosen_instance_id": None,
                        "visible_point": None, "face_normal": None,
                        "ray_hit": pending8.get("ray_hit"),
                        "target_cell": pending8.get("target_cell"),
                        "aim_point": pending8.get("aim_point"),
                        "mine_attack_gate": copy.deepcopy(
                            pending8.get("mine_attack_gate")),
                        "post_action_ray_hit": pending8.get(
                            "post_action_ray_hit"),
                        "is_gui_open": bool(w.info.get("is_gui_open", False)),
                        "walk": copy.deepcopy(STORY.get("walk")),
                        "mobs": STORY["mobs"]}
                    if args.human_play:
                        mob_observation8 = STRUCT.get(
                            "current_human_mob_observation")
                        if not isinstance(mob_observation8, dict):
                            mob_observation8 = {
                                "packet_present": False,
                                "packet_parse_valid": False,
                                "query_box_relative": None,
                                "candidates_absolute": [],
                            }
                        story_row8["structural_mob_observation"] = json.loads(
                            json.dumps(mob_observation8))
                        voxel_observation8 = STRUCT.get(
                            "current_human_voxel_observation")
                        if not isinstance(voxel_observation8, dict):
                            voxel_observation8 = {
                                "packet_present": False,
                                "packet_parse_valid": False,
                                "query_box_relative": None,
                                "player_position": None,
                                "blocks_relative": [],
                            }
                        story_row8["structural_voxel_observation"] = json.loads(
                            json.dumps(voxel_observation8))
                    if args.visible_surface_masks:
                        render_pose8 = pending8.get("render_pose")
                        if not isinstance(render_pose8, dict):
                            raise RuntimeError(
                                "kept dense-mask RGB lacks its matched render pose")
                        story_row8["render_pose"] = dict(render_pose8)
                        viewmodel_valid8 = pending8.get(
                            VIEWMODEL_PIXEL_SUPERVISION_FIELD)
                        viewmodel_audit8 = pending8.get(
                            VIEWMODEL_OCCLUSION_AUDIT_FIELD)
                        if (isinstance(viewmodel_valid8, bool)
                                or viewmodel_valid8 not in (0, 1)
                                or not isinstance(viewmodel_audit8, dict)):
                            raise RuntimeError(
                                "kept dense-mask RGB lacks its dynamic-viewmodel "
                                "supervision contract")
                        story_row8[VIEWMODEL_PIXEL_SUPERVISION_FIELD] = int(
                            viewmodel_valid8)
                        story_row8[VIEWMODEL_OCCLUSION_AUDIT_FIELD] = dict(
                            viewmodel_audit8)
                    struct_fields8 = STORY.get("pending_struct_fields")
                    STORY["pending_struct_fields"] = None
                    if struct_fields8 is not None:
                        story_row8.update(struct_fields8)
                    STORY["rows"].append(story_row8)
                    if not STRUCT.get("defer_human_labels"):
                        refresh_current_structural_telemetry9(_ctx)
                    else:
                        story_row8["deferred_visibility"] = {
                            "version": 1,
                            "contract": HUMAN_DEFERRED_LABEL_CONTRACT,
                            "status": "pending_postplay_exact_replay",
                        }
                    return r
                if not args.policy_driver:
                    rec.grab = _grab_story
            init_info = dict(w.info)
            set_phase(0)
            TRAJ["bc_valid"] = 0
            for _ in range(8):
                warm9 = w.sim.noop_action()
                warm9["_xbench_sensor"] = 1
                w.obs, _, _, _, w.info = w.sim.step(warm9)
            if args.policy_driver:
                import torch as policy_torch9
                policy_rollout_seed9 = (
                    int(args.action_seed) if args.policy_rollout_seed is None
                    else int(args.policy_rollout_seed))
                policy_seed_key9 = f"policy_{cell}_{setting}_w{args.world_seed}"
                if policy_rollout_seed9 != 0:
                    policy_seed_key9 += f"_r{policy_rollout_seed9}"
                policy_torch9.manual_seed(zlib.crc32(policy_seed_key9.encode()))
                prunner9 = policy_runner_cell9["runner"]
                if prunner9 is not None:
                    prunner9.reset()
                    prunner9.set_obj_id(2)
                policy_detached_traj_fh9, TRAJ["fh"] = TRAJ["fh"], None
                if policy_detached_traj_fh9 is not None:
                    policy_detached_traj_fh9.close()
                staged_cells9 = {
                    (int(m9["x"]), int(m9["y"]), int(m9["z"])): str(m9["kind"])
                    for m9 in ctx.get("mine_sites", [])}
                if (v12_snapshot_policy_eval9
                        and not isinstance(ctx.get("_struct_occ"), G.OccupancyMap)):
                    rb_px9, rb_py9, rb_pz9 = (int(math.floor(float(q9))) for q9 in w.get_pos())
                    rb_dxz9 = max([max(abs(cx9 - rb_px9), abs(cz9 - rb_pz9))
                                   for (cx9, cy9, cz9) in staged_cells9] or [0])
                    rb_dy9 = max([abs(cy9 - rb_py9) for (cx9, cy9, cz9) in staged_cells9] or [0])
                    rb_span9 = int(min(72, max(24, rb_dxz9 + 8)))
                    rb_y9 = int(min(36, max(12, rb_dy9 + 6)))
                    if rb_span9 != 24 or rb_y9 != 12:
                        print(f"[snapshot] policy GT occupancy rebuild box widened to "
                              f"span={rb_span9} y=+-{rb_y9} (staged cells {rb_dxz9}/{rb_dy9} "
                              f"blocks from the start)", flush=True)
                    restored_grid9 = S.scan_mine_terrain(
                        span=rb_span9, y_down=rb_y9, y_up=rb_y9)
                    restored_bounds9 = tuple(
                        int(q9) for q9 in S._last_mine_scan_bounds)
                    restored_mismatch9 = {
                        cell9: {
                            "expected": str(kind9),
                            "observed": str(restored_grid9.get(cell9)),
                        }
                        for cell9, kind9 in staged_cells9.items()
                        if G._bare(str(restored_grid9.get(cell9, "")))
                           != G._bare(str(kind9))
                    }
                    if restored_mismatch9:
                        raise RuntimeError(
                            "snapshot policy occupancy rebuild staged-cell mismatch: "
                            f"{restored_mismatch9}")
                    ctx["_struct_occ"] = S.mine_occupancy(
                        restored_grid9, restored_bounds9)
                    ctx["_struct_box"] = list(restored_bounds9)
                    print(
                        "[snapshot] policy GT occupancy rebuilt "
                        f"known={len(restored_grid9)} bounds={list(restored_bounds9)}",
                        flush=True)
                rec.grab("policy episode start | pre-action")
                broken_step9 = {}
                policy_left_domain_ticks9 = 0
                policy_target_cls9 = str(ctx["cls"])
                policy_success_quota9 = (
                    int(ctx["quota"]) if args.policy_success_quota is None
                    else int(args.policy_success_quota))
                if not (1 <= policy_success_quota9 <= int(ctx["quota"])):
                    raise RuntimeError(
                        "policy success quota must be within the staged target "
                        f"cardinality: {policy_success_quota9}/{ctx['quota']}")
                policy_distractor_classes9 = [
                    str(value9) for value9 in ore_roles9["distractors"]]
                policy_confuser_cls9 = ",".join(policy_distractor_classes9)
                policy_traj_fh9 = open(
                    os.path.join(OUT, f"policy_traj_{cell}_{setting}.jsonl"), "w")
                policy_succ9, policy_succ_step9 = 0, -1
                policy_eval_valid9, policy_invalid_reason9 = 1, None
                policy_target_break_rows9 = []
                policy_distractor_break_rows9 = {
                    kind9: [] for kind9 in policy_distractor_classes9}
                policy_stat_break_events9 = []
                policy_stat_counts9 = {
                    kind9: 0 for kind9 in (
                        [policy_target_cls9] + policy_distractor_classes9)}
                policy_target_attack_latch9 = None

                def observe_policy_break_stats9(step9):
                    nonlocal policy_succ9, policy_succ_step9
                    nonlocal policy_eval_valid9, policy_invalid_reason9
                    newly_retired9 = set()
                    for kind9 in policy_stat_counts9:
                        current9 = policy_eval_mine_block_delta(
                            w.info, init_info, kind9)
                        previous9 = int(policy_stat_counts9[kind9])
                        capacity9 = sum(
                            1 for value9 in staged_cells9.values()
                            if str(value9) == str(kind9))
                        if current9 < previous9 or current9 > capacity9:
                            policy_eval_valid9 = 0
                            policy_invalid_reason9 = (
                                f"mine_block_stat_outside_staged_capacity:"
                                f"{kind9}:{previous9}->{current9}/{capacity9}")
                            return
                        increment9 = current9 - previous9
                        if not increment9:
                            continue
                        policy_stat_counts9[kind9] = current9
                        for _ in range(increment9):
                            if kind9 == policy_target_cls9:
                                latch9 = policy_target_attack_latch9
                                cell9 = None
                                if (latch9 is not None
                                        and int(step9) - int(latch9["step"]) <= 32
                                        and tuple(latch9["cell"]) not in broken_step9):
                                    cell9 = tuple(int(q9) for q9 in latch9["cell"])
                                if cell9 is None:
                                    policy_eval_valid9 = 0
                                    policy_invalid_reason9 = (
                                        "target_mine_stat_without_exact_crosshair_identity")
                                    return
                                broken_step9.setdefault(cell9, int(step9))
                                newly_retired9.add(cell9)
                                row9 = {
                                    "cell": [int(q9) for q9 in cell9],
                                    "step": int(step9),
                                    "source": "mine_block_stat+exact_crosshair_gt",
                                }
                                policy_target_break_rows9.append(row9)
                            else:
                                candidates9 = [
                                    cell9 for cell9, value9 in staged_cells9.items()
                                    if str(value9) == str(kind9)
                                    and cell9 not in broken_step9]
                                if len(candidates9) != 1:
                                    policy_eval_valid9 = 0
                                    policy_invalid_reason9 = (
                                        f"distractor_stat_identity_ambiguous:{kind9}")
                                    return
                                cell9 = tuple(int(q9) for q9 in candidates9[0])
                                broken_step9.setdefault(cell9, int(step9))
                                newly_retired9.add(cell9)
                                row9 = {
                                    "cell": [int(q9) for q9 in cell9],
                                    "step": int(step9),
                                    "source": "mine_block_stat+unique_staged_class",
                                }
                                policy_distractor_break_rows9[kind9].append(row9)
                            policy_stat_break_events9.append({
                                "kind": str(kind9), **row9})
                    policy_occ9 = ctx.get("_struct_occ")
                    if newly_retired9 and isinstance(policy_occ9, G.OccupancyMap):
                        policy_eval_retire_broken_cells(
                            policy_occ9, newly_retired9)
                    if (not policy_succ9 and policy_eval_valid9
                            and policy_stat_counts9[policy_target_cls9]
                            >= policy_success_quota9):
                        policy_succ9, policy_succ_step9 = 1, int(step9)

                policy_stp9 = -1
                policy_t0_9 = time.time()
                policy_gimg9, policy_gmask9 = policy_goal9
                for policy_stp9 in range(int(args.policy_budget)):
                    full_pov_rgb9 = np.ascontiguousarray(
                        np.asarray(w.info["pov"], dtype=np.uint8))
                    paction9, _ = prunner9.act(
                        w.obs, policy_gimg9, policy_gmask9)
                    penv9 = w.sim.agent_action_to_env_action(paction9)
                    pa9 = w.sim.noop_action()
                    pa9.update(penv9)
                    pa9["sneak"] = 0
                    policy_gt_instances9 = policy_eval_current_target_instances9(
                        ctx)
                    if v12_snapshot_policy_eval9:
                        if policy_eval_staged_cell_box(
                                staged_cells9, w.get_pos()) is None:
                            policy_left_domain_ticks9 += 1
                            if policy_left_domain_ticks9 == 1:
                                print(f"[policy-eval] step {policy_stp9}: agent beyond the "
                                      "staged-cell audit range; episode continues (no cut)",
                                      flush=True)
                        try:
                            attack9 = bool(int(np.asarray(
                                penv9.get("attack", 0)).reshape(-1)[0]))
                        except Exception:
                            attack9 = False
                        if attack9:
                            crosshair_cell9 = policy_eval_crosshair_target_cell(
                                policy_gt_instances9, full_pov_rgb9.shape,
                                excluded=broken_step9)
                            if crosshair_cell9 is not None:
                                policy_target_attack_latch9 = {
                                    "cell": tuple(crosshair_cell9),
                                    "step": int(policy_stp9),
                                }
                    w.obs, _, policy_term9, policy_trunc9, w.info = w.sim.step(pa9)
                    if v12_snapshot_policy_eval9:
                        observe_policy_break_stats9(policy_stp9)
                    rec.grab("policy")
                    policy_px9, _, policy_pz9 = w.get_pos()
                    policy_traj_fh9.write(json.dumps({
                        "t": policy_stp9, "x": round(float(policy_px9), 2),
                        "z": round(float(policy_pz9), 2),
                        "succ": int(policy_succ9)}) + "\n")
                    if w.info.get("is_gui_open", False):
                        pgui9 = w.sim.noop_action()
                        pgui9["inventory"] = 1
                        w.obs, _, _, _, w.info = w.sim.step(pgui9)
                    if policy_succ9 and policy_stp9 >= policy_succ_step9 + int(args.policy_post):
                        break
                    if v12_snapshot_policy_eval9 and not policy_eval_valid9:
                        break
                    if policy_term9 or policy_trunc9:
                        break
                policy_wall9 = max(time.time() - policy_t0_9, 1e-6)
                if v12_snapshot_policy_eval9:
                    exact_progress9 = {
                        "quota": int(ctx["quota"]),
                        "broken": int(policy_stat_counts9[policy_target_cls9]),
                        "complete": int(policy_succ9),
                        "breaks": list(policy_target_break_rows9),
                    }
                    if not policy_eval_valid9:
                        policy_succ9, policy_succ_step9 = 0, -1
                policy_traj_fh9.close()
                if w.info.get("is_gui_open", False):
                    pgui9 = w.sim.noop_action()
                    pgui9["inventory"] = 1
                    w.obs, _, _, _, w.info = w.sim.step(pgui9)
                policy_wrong9 = policy_eval_wrong_block_delta(
                    w.info, init_info, policy_target_cls9)
                policy_ore_pool9 = frozenset(mine_stage_pool9)
                policy_wrong_ore_blocks9 = {
                    key9: value9 for key9, value9 in policy_wrong9.items()
                    if key9 in policy_ore_pool9}
                distractor_breaks9 = {
                    kind9: list(rows9)
                    for kind9, rows9 in policy_distractor_break_rows9.items()}
                cache_stats9 = dict(
                    getattr(prunner9, "goal_cache_stats", {}) or {}) \
                    if prunner9 is not None else {}
                first_broken9 = (None if not policy_stat_break_events9 else min(
                    policy_stat_break_events9,
                    key=lambda value9: (int(value9["step"]), value9["cell"])))
                first_broken_kind9 = (
                    None if first_broken9 is None else first_broken9["kind"])
                policy_row9 = policy_eval_result_row(
                    mode="policy_v12_multiclass6_eval",
                    model=str(args.policy_model_name),
                    ckpt=str(policy_ckpt9),
                    policy_backend=str(args.policy_backend),
                    cell=cell, setting=setting,
                    gmode=str(args.policy_gmode), biome=str(args.biome),
                    group_id=str(None),
                    target_cls=policy_target_cls9,
                    confuser_cls=policy_confuser_cls9,
                    world_seed=int(args.world_seed), aseed=int(args.action_seed),
                    policy_rollout_seed=int(policy_rollout_seed9),
                    site_seed=int(site_seed9), layout_seed=int(layout_seed9),
                    pose_seed=int(pose_seed9), quota=int(ctx["quota"]),
                    success_quota=int(policy_success_quota9),
                    succ=int(policy_succ9), succ_step=int(policy_succ_step9),
                    steps=int(policy_stp9 + 1), budget=int(args.policy_budget),
                    wrong=policy_wrong9,
                    wrong_ore=int(sum(policy_wrong_ore_blocks9.values())),
                    wrong_ore_blocks=policy_wrong_ore_blocks9,
                    staged_target_broken=int(
                        policy_stat_counts9[policy_target_cls9]),
                    staged_confuser_broken=int(sum(
                        policy_stat_counts9[kind9]
                        for kind9 in policy_distractor_classes9)),
                    target_breaks=exact_progress9["breaks"],
                    distractor_breaks=distractor_breaks9,
                    distractor_classes=policy_distractor_classes9,
                    snapshot_world_sha256=(
                        snapshot_bundle9.world.sha256
                        if snapshot_bundle9 is not None else None),
                    snapshot_archive_sha256=(
                        snapshot_bundle9.metadata.get("world_archive_sha256")
                        if snapshot_bundle9 is not None else None),
                    snapshot_payload_sha256=(
                        snapshot_bundle9.metadata.get("payload_sha256")
                        if snapshot_bundle9 is not None else None),
                    selected_classswap_geometry_sha256=(
                        snapshot_classswap9["geometry_sha256"]
                        if snapshot_classswap9 is not None else None),
                    selected_classswap_assignment_sha256=(
                        snapshot_classswap9["assignment_sha256"]
                        if snapshot_classswap9 is not None else None),
                    goal_cache_hits=int(cache_stats9.get("hits", 0)),
                    goal_cache_misses=int(cache_stats9.get("misses", 0)),
                    dynamic_goal_summary=None,
                    first_staged_break_kind=first_broken_kind9,
                    wrong_class_first_hit=int(
                        first_broken_kind9 is not None
                        and first_broken_kind9 != policy_target_cls9),
                    eval_valid=int(policy_eval_valid9),
                    invalid_reason=policy_invalid_reason9,
                    frames_kept=int(len(rec.frames)),
                    fps=round((policy_stp9 + 1) / policy_wall9, 2),
                    grounding_jsonl=None)
                with open(os.path.join(OUT, "results_policy.jsonl"), "a") as prf9:
                    prf9.write(json.dumps(policy_row9) + "\n")
                print("[policy-ep] " + json.dumps(policy_row9), flush=True)
                policy_label9 = "SUCCESS" if policy_succ9 else "FAIL"
                for _ in range(8):
                    w.step_noop()
                    rec.grab(f"RESULT: {policy_label9}")
                ok = int(policy_succ9)
                timing_recording_finished9 = time.monotonic()
                episode_timing9.update({
                    "episode_recording_wall_s": runtime_wall_seconds(
                        timing_recording_started9, timing_recording_finished9),
                    "frames": int(len(rec.frames)),
                    "result": int(ok),
                })
                results[f"{cell}/{setting}"] = ok
                if args.story:
                    sfh.flush()
                    sfh.close()
                    STORY["rows"] = []
                print(f"[demo] {cell:14s} {setting:5s} -> POLICY {policy_label9} "
                      f"({len(rec.frames)} frames)", flush=True)
                timing_save_started9 = time.monotonic()
                if int(args.policy_record):
                    rec.save()
                timing_save_finished9 = time.monotonic()
                episode_timing9.update({
                    "video_save_wall_s": runtime_wall_seconds(
                        timing_save_started9, timing_save_finished9),
                    "episode_total_through_save_wall_s": runtime_wall_seconds(
                        timing_recording_started9, timing_save_finished9),
                })
                STORY["rec"] = None
                continue
            scan_span9 = 24
            scan_vertical_span9 = 18
            if args.human_play:
                initial_occ9 = G.OccupancyMap()
                scan_occ(initial_occ9, span=scan_span9,
                         vertical_span=scan_vertical_span9)
                ctx["_struct_occ"] = initial_occ9
                ctx["_struct_box"] = None

            human_ready_ui9 = None
            human_ready_gate9 = None
            human_ready_mission9 = None
            if args.human_play:
                human_f0_source9 = TRAJ.get("pending_row")
                if (not isinstance(human_f0_source9, dict)
                        or human_f0_source9.get("step_kind") != "sensor"
                        or int(human_f0_source9.get("bc_valid", -1)) != 0):
                    raise RuntimeError(
                        "native-human canonical f0 is not the pending sensor source")
                human_f0_source9["step_kind"] = "control"
                human_f0_source9["bc_valid"] = 1
                human_f0_source9["human_canonical_source"] = 1
                human_ready_goal_name9 = str(
                    ctx.get("target_kind") or ctx.get("clsname")
                    or ctx.get("cls") or ctx.get("nm") or cell)
                human_ready_mission9 = (
                    f"{cell}: {human_ready_goal_name9}")
                from attacca.evaluation.mine_human import HumanUI
                window_label9 = str(os.environ.get(
                    "XBENCH_HUMAN_WINDOW_LABEL", "")).strip()
                window_x9 = os.environ.get("XBENCH_HUMAN_WINDOW_X")
                window_y9 = os.environ.get("XBENCH_HUMAN_WINDOW_Y")
                human_ready_ui9 = HumanUI(
                    scale=int(args.human_scale),
                    mouse_sens=float(args.human_mouse_sens),
                    caption=(
                        f"HumanPlay — {window_label9}" if window_label9
                        else f"HumanPlay — w{args.world_seed} "
                             f"{human_ready_goal_name9}"),
                    window_x=(None if window_x9 is None else int(window_x9)),
                    window_y=(None if window_y9 is None else int(window_y9)),
                    prevent_initial_focus=True)
                layout_payload9 = ctx.get("mine_scene_layout")
                layout_digest9 = (None if layout_payload9 is None else
                                  hashlib.sha256(json.dumps(
                                      layout_payload9, sort_keys=True,
                                      separators=(",", ":"),
                                      default=str).encode()).hexdigest())
                staged_rgb9 = np.ascontiguousarray(
                    np.asarray(w.info["pov"], dtype=np.uint8))
                staged_rgb_path9 = os.path.join(OUT, "staged_ready.png")
                import cv2 as _human_ready_cv2_9
                if not _human_ready_cv2_9.imwrite(
                        staged_rgb_path9, staged_rgb9[:, :, ::-1]):
                    raise RuntimeError(
                        "failed to save staged-ready diagnostic RGB")
                publish_human_status9(
                    "staged_ready", cell=cell, setting=setting,
                    target_kind=human_ready_goal_name9,
                    quota=int(ctx["quota"]),
                    mine_worldgen_profile=str(args.mine_worldgen_profile),
                    layout_digest=layout_digest9,
                    staged_ready_rgb=staged_rgb_path9,
                    staged_ready_rgb_sha256=hashlib.sha256(
                        staged_rgb9.tobytes()).hexdigest(),
                    simulator_steps_while_waiting=0,
                    frames_recorded_while_waiting=0)
                def human_start_allowed9():
                    return bool(
                        not args.human_start_permit_file
                        or os.path.isfile(args.human_start_permit_file))
                human_ready_gate9 = human_ready_ui9.wait_for_start(
                    np.asarray(w.info["pov"]), human_ready_mission9,
                    start_allowed=human_start_allowed9)
                if human_ready_gate9.get("started") is not True:
                    human_ready_ui9.close()
                    prestart9 = {
                        "version": 1,
                        "contract": HUMAN_SESSION_CONTRACT,
                        "episode_source": "human_play",
                        "oracle_motion_profile": HUMAN_MOTION_PROFILE,
                        "cell": cell,
                        "setting": setting,
                        "world_seed": int(args.world_seed),
                        "action_seed": int(args.action_seed),
                        "quota": int(ctx["quota"]),
                        "quota_seed": human_quota_seed9,
                        "completed": 0,
                        "end_reason": "prestart_abort",
                        "aborted": True,
                        "timed_out": False,
                        "census_enabled": True,
                        "census_roster": list(mine_census_roster9),
                        "frame_count": 0,
                        "traj_row_count": 0,
                        "freeze_recorded_frame_count": 0,
                        "start_gate": dict(human_ready_gate9),
                        "decision_events": [],
                        "block_changes": [],
                    }
                    for name9 in (
                            f"human_session_{cell}_{setting}.json",
                            "meta.json"):
                        with open(os.path.join(OUT, name9), "w",
                                  encoding="utf-8") as prestart_f9:
                            json.dump(prestart9, prestart_f9,
                                      indent=2, sort_keys=True)
                            prestart_f9.write("\n")
                    if args.story:
                        sfh.flush()
                        sfh.close()
                        STORY["rows"] = []
                    STORY["rec"] = None
                    print(
                        f"[human] {cell}/{setting} -> prestart_abort "
                        "frames=0", flush=True)
                    publish_human_status9(
                        "prestart_abort", cell=cell, setting=setting,
                        target_kind=human_ready_goal_name9,
                        reason=human_ready_gate9.get("reason"))
                    w.close()
                    return 0
                human_f0_source9["human_start_gate"] = dict(
                    human_ready_gate9)
                publish_human_status9(
                    "playing", cell=cell, setting=setting,
                    target_kind=human_ready_goal_name9,
                    quota=int(ctx["quota"]), completed=0)
            rec.grab("episode start | pre-action")
            TRAJ["bc_valid"] = 1

            human_completed9 = 0
            human_end_reason9 = None
            if args.human_play:
                mission9 = str(human_ready_mission9)
                ui9 = human_ready_ui9
                pov9 = np.asarray(w.info["pov"])
                human_live_occ9 = ctx.get("_struct_occ")
                if not isinstance(human_live_occ9, G.OccupancyMap):
                    raise RuntimeError(
                        "human exact recorder has no pre-play occupancy")
                human_replay_occ9 = clone_human_occupancy9(human_live_occ9)
                human_ray_occ9 = clone_human_occupancy9(human_live_occ9)
                human_marker_occ9 = clone_human_occupancy9(human_live_occ9)
                source_pending9 = TRAJ.get("pending_row")
                if source_pending9 is None:
                    raise RuntimeError("human play has no canonical source RGB")
                STRUCT["defer_human_start_t"] = int(source_pending9["t"])
                STRUCT["defer_human_labels"] = True
                live_steps9 = 0
                decision_ledger9 = DecisionLedger()
                active_selection9 = None
                pending_auto_freeze_reason9 = None
                visible_commit_streak9 = 0

                def visible_component_cells9(record9):
                    raw_cells9 = record9.get("component_cells")
                    proof9 = record9.get("proof")
                    if raw_cells9 is None and isinstance(proof9, dict):
                        raw_cells9 = proof9.get("visible_member_cells")
                    if raw_cells9 is None:
                        raw_cells9 = [record9.get("world_position")]
                    return frozenset(
                        tuple(int(q9) for q9 in raw9)
                        for raw9 in raw_cells9
                        if isinstance(raw9, (tuple, list))
                        and len(raw9) == 3)

                def current_human_pair9():
                    pending9 = TRAJ.get("pending_row")
                    story9 = STORY["rows"][-1] if STORY.get("rows") else None
                    if (not isinstance(pending9, dict)
                            or not isinstance(story9, dict)
                            or int(story9.get("traj_t", -1))
                               != int(pending9.get("t", -2))):
                        raise RuntimeError(
                            "human decision lacks an exact current RGB join")
                    return pending9, story9

                def record_human_decision9(kind9, *, freeze_reason9=None,
                                           **fields9):
                    pending9, story9 = current_human_pair9()
                    event9 = decision_ledger9.append(
                        kind9, source_traj_t=int(pending9["t"]),
                        source_frame=int(story9["f"]),
                        freeze_reason=freeze_reason9, **fields9)
                    for output9 in (pending9, story9):
                        output9.setdefault("human_decision_events", []).append(
                            dict(event9))
                    return event9

                def exact_frozen_context9():
                    pending9, story9 = current_human_pair9()
                    instances9 = isolated_frozen_marker_instances(
                        ctx=ctx, struct=STRUCT, traj=TRAJ,
                        replay_occupancy=human_marker_occ9,
                        source_story=story9,
                        refresh=lambda: refresh_current_structural_telemetry9(
                            ctx))
                    aux9 = same_rgb_structural_aux(pending9, story9)
                    if aux9 is None:
                        raise RuntimeError(
                            "frozen human RGB lacks same-frame exact geometry")
                    return pending9, story9, instances9, aux9

                def live_exact_goal_recognizable9():
                    pending9, story9 = current_human_pair9()
                    pose9 = pending9.get("render_pose")
                    if not isinstance(pose9, dict):
                        raise RuntimeError(
                            "live target visibility lacks current render pose")
                    target_kind9 = G._bare(str(ctx.get("cls") or ""))
                    target_cells9 = sorted(
                        tuple(int(q9) for q9 in cell9)
                        for cell9, kind9 in human_marker_occ9.grid.items()
                        if G._bare(str(kind9 or "")) == target_kind9)
                    if not target_cells9:
                        audit9 = {
                            "version": 1,
                            "contract": HUMAN_AUTO_COMMIT_VISIBILITY_CONTRACT,
                            "recognizable_visible": False,
                            "recognizable_instances": [],
                            "remaining_target_cell_count": 0,
                        }
                        for output9 in (pending9, story9):
                            output9["human_live_target_visibility"] = dict(audit9)
                        return False
                    eye9 = (
                        float(pose9["x"]),
                        float(pose9["y"])
                        + float(pose9.get("eye_height", 1.62)),
                        float(pose9["z"]),
                    )
                    yaw9, pitch9 = float(pose9["yaw"]), float(pose9["pitch"])
                    depth9 = w.info.get(RENDERER_VIEWMODEL_DEPTH_FIELD)
                    if (not isinstance(depth9, np.ndarray)
                            or depth9.shape != (G.H_PX, G.W_PX)):
                        raise RuntimeError(
                            "live target visibility lacks exact renderer depth")
                    exclusion9 = structural_pixel_exclusion9(
                        relax_human_swing=True)
                    owner_pixels9 = G.backproject_depth_owner_pixels_for_blocks(
                        depth9, eye9, yaw9, pitch9,
                        target_cells9,
                        near=DEPTH_NEAR_PLANE9, far=DEPTH_FAR_PLANE9,
                        pixel_exclusion_mask=exclusion9)
                    recognizable9 = []
                    for cell9 in target_cells9:
                        if math.hypot(
                                float(cell9[0]) + 0.5 - float(pose9["x"]),
                                float(cell9[2]) + 0.5 - float(pose9["z"])) > 36.0:
                            continue
                        pixels9 = owner_pixels9.get(cell9)
                        area9 = 0 if pixels9 is None else int(len(pixels9))
                        if area9 <= 0:
                            continue
                        owned_rows9 = pixels9 // int(G.W_PX)
                        owned_cols9 = pixels9 % int(G.W_PX)
                        bbox9 = [
                            int(owned_cols9.min()), int(owned_rows9.min()),
                            int(owned_cols9.max()), int(owned_rows9.max()),
                        ]
                        short9 = min(
                            bbox9[2] - bbox9[0] + 1,
                            bbox9[3] - bbox9[1] + 1)
                        if structural_full_cube_surface_recognizable(
                                area9, int(short9), bbox9):
                            recognizable9.append({
                                "instance_id": (
                                    f"block:{cell9[0]}:{cell9[1]}:{cell9[2]}"),
                                "cell": [int(q9) for q9 in cell9],
                                "visible_area_px": int(area9),
                                "visible_short_side_px": int(short9),
                                "visible_bbox_px": bbox9,
                            })
                    audit9 = {
                        "version": 1,
                        "contract": HUMAN_AUTO_COMMIT_VISIBILITY_CONTRACT,
                        "recognizable_visible": bool(recognizable9),
                        "recognizable_instances": recognizable9,
                        "remaining_target_cell_count": int(len(target_cells9)),
                    }
                    for output9 in (pending9, story9):
                        output9["human_live_target_visibility"] = copy.deepcopy(
                            audit9)
                    return bool(recognizable9)

                def advance_live_visibility_latch9():
                    nonlocal visible_commit_streak9
                    recognizable9 = live_exact_goal_recognizable9()
                    visible_commit_streak9, trigger9 = (
                        advance_uncommitted_visible_streak(
                            visible_commit_streak9,
                            recognizable_visible=bool(recognizable9),
                            has_active_selection=(active_selection9 is not None)))
                    pending9, story9 = current_human_pair9()
                    for output9 in (pending9, story9):
                        output9["human_live_target_visibility"][
                            "consecutive_visible_frames"] = int(
                                visible_commit_streak9)
                        output9["human_live_target_visibility"][
                            "auto_freeze_threshold_frames"] = int(
                                HUMAN_AUTO_COMMIT_VISIBLE_FRAMES)
                    return bool(trigger9)

                def resolve_frozen_pixel9(marker9, frozen9):
                    pending9, story9, instances9, aux9 = frozen9
                    choice9 = human_marker_pixel_target_choice(
                        pending9, story9, marker9.get("pixel_xy"),
                        predecision_aux=aux9)
                    chosen_id9 = str(choice9["chosen_instance_id"])
                    choice_cells9 = frozenset(
                        tuple(int(q9) for q9 in raw9)
                        for raw9 in choice9["component_cells"])
                    if not choice_cells9:
                        raise RuntimeError(
                            "clicked world instance has no exact component")
                    selected9 = next(
                        (record9 for record9 in instances9
                         if str(record9.get("instance_id")) == chosen_id9),
                        None)
                    if selected9 is None:
                        raise RuntimeError(
                            "frozen selection disappeared from exact instances")
                    pose9 = pending9["render_pose"]
                    eye9 = (float(pose9["x"]),
                            float(pose9["y"])
                            + float(pose9.get("eye_height", 1.62)),
                            float(pose9["z"]))
                    distance9 = nearest_face_distance(
                        eye9, choice9["component_cells"])
                    kind9 = G._bare(str(
                        selected9.get("kind") or selected9.get("type") or ""))
                    world_cell9 = selected9.get("world_position")
                    if (not kind9 or not isinstance(world_cell9, list)
                            or len(world_cell9) != 3):
                        raise RuntimeError(
                            "frozen selection lacks class/cell identity")
                    payload9 = dict(marker9)
                    payload9.update({
                        "resolved_instance_id": chosen_id9,
                        "resolved_kind": kind9,
                        "resolved_target_cell": [int(q9) for q9 in world_cell9],
                        "resolved_component_cells": [
                            [int(q9) for q9 in cell9]
                            for cell9 in choice9["component_cells"]],
                        "resolved_distance": float(distance9),
                    })
                    return {
                        "marker": payload9,
                        "overlay": {
                            "kind": kind9,
                            "distance": float(distance9),
                            "proof": choice9["proof"],
                        },
                    }

                def enter_human_freeze9(reason9):
                    nonlocal deadline9, annotation_pause_s9
                    nonlocal annotation_pause_count9
                    pending9, _story9 = current_human_pair9()
                    source_t9 = int(pending9["t"])
                    traj_clock9 = int(TRAJ["t"])
                    record_human_decision9(
                        "freeze_enter", freeze_reason9=reason9)
                    frozen_context9 = exact_frozen_context9()
                    marker9 = None
                    pause9 = 0.0
                    attempts9 = []
                    selection_enabled9 = reason9 not in (
                        "quota", "manual_pause")
                    publish_human_status9(
                        "paused" if reason9 == "manual_pause" else
                        "selecting",
                        cell=cell, setting=setting,
                        target_kind=human_ready_goal_name9,
                        freeze_reason=str(reason9),
                        quota=int(ctx["quota"]),
                        completed=int(human_completed9))
                    resolver9 = (None if not selection_enabled9 else
                                 lambda marker10: resolve_frozen_pixel9(
                                     marker10, frozen_context9))
                    candidate_overlays9 = [
                        {"proof": record9.get("proof")}
                        for record9 in frozen_context9[2]
                        if visible_component_cells9(record9)
                        and isinstance(record9.get("proof"), dict)
                        and selection_enabled9]
                    allow_empty9 = bool(
                        not selection_enabled9 or not candidate_overlays9)
                    marker9, pause9, attempts9 = ui9.freeze_select_target(
                        pov9, mission9, reason=reason9,
                        resolve_pixel=resolver9, allow_empty=allow_empty9,
                        candidate_overlays=candidate_overlays9,
                        selection_enabled=selection_enabled9)
                    pause9 = max(0.0, float(pause9))
                    annotation_pause_s9 += pause9
                    annotation_pause_count9 += 1
                    deadline9 += pause9
                    for attempt9 in attempts9:
                        if attempt9.get("accepted") is not True:
                            record_human_decision9(
                                "selection_rejected", freeze_reason9=reason9,
                                pixel_xy=attempt9.get("pixel_xy"),
                                window_xy=attempt9.get("window_xy"),
                                reason=attempt9.get("reason"))
                    if marker9 is not None:
                        record_human_decision9(
                            "selection_accepted", freeze_reason9=reason9,
                            pixel_xy=marker9.get("pixel_xy"),
                            window_xy=marker9.get("window_xy"),
                            resolved_instance_id=marker9.get(
                                "resolved_instance_id"),
                            resolved_target_cell=marker9.get(
                                "resolved_target_cell"),
                            resolved_kind=marker9.get("resolved_kind"),
                            resolved_distance=marker9.get(
                                "resolved_distance"))
                    if not (ui9.abort or ui9.quit):
                        record_human_decision9(
                            "resume", freeze_reason9=reason9,
                            selected_instance_id=(
                                None if marker9 is None else
                                marker9.get("resolved_instance_id")))
                        publish_human_status9(
                            "playing", cell=cell, setting=setting,
                            target_kind=human_ready_goal_name9,
                            quota=int(ctx["quota"]),
                            completed=int(human_completed9))
                    if (int(TRAJ["t"]) != traj_clock9
                            or TRAJ.get("pending_row") is not pending9
                            or int(pending9["t"]) != source_t9):
                        raise RuntimeError(
                            "human freeze advanced simulator/trajectory")
                    return marker9, pause9

                try:
                    ui9.abort = False
                    ui9.grab_mouse(True)
                    ui9.mouse_delta = [0.0, 0.0]
                    ui9.pressed.clear()
                    ui9.consume_chosen_marker()
                    ui9.consume_pause_marker()

                    live_started9 = time.monotonic()
                    deadline9 = time.monotonic() + float(args.human_timeout)
                    annotation_pause_s9 = 0.0
                    annotation_pause_count9 = 0
                    frame_period9 = 1.0 / float(args.human_fps)
                    human_frame9 = 0
                    noop9 = w.sim.noop_action()
                    if advance_live_visibility_latch9():
                        pending_auto_freeze_reason9 = "visible_streak"
                    while True:
                        wall_frame_started9 = time.monotonic()
                        ui9.pump()
                        if ui9.quit or ui9.abort:
                            human_end_reason9 = (
                                "quit" if ui9.quit else "aborted")
                            record_human_decision9(
                                "abort", reason=human_end_reason9)
                            break

                        if time.monotonic() >= deadline9:
                            human_end_reason9 = "timeout"
                            record_human_decision9("timeout")
                            break

                        pending_before_marker9 = TRAJ.get("pending_row")
                        if pending_before_marker9 is None:
                            raise RuntimeError(
                                "human marker has no canonical source RGB")
                        source_render_pose9 = pending_before_marker9.get(
                            "render_pose")
                        if not isinstance(source_render_pose9, dict):
                            raise RuntimeError(
                                "human source RGB lacks its render pose")
                        marker_selection9 = None
                        c_pressed9 = bool(ui9.consume_chosen_marker())
                        freeze_reason9 = pending_auto_freeze_reason9
                        if freeze_reason9 == "visible_streak" and c_pressed9:
                            freeze_reason9 = "c_press"
                            pending_auto_freeze_reason9 = None
                        if freeze_reason9 is None and c_pressed9:
                            freeze_reason9 = "c_press"
                        if freeze_reason9 is None and ui9.consume_pause_marker():
                            freeze_reason9 = "manual_pause"
                        if freeze_reason9 is not None:
                            marker_selection9, _marker_pause9 = (
                                enter_human_freeze9(freeze_reason9))
                            pending_auto_freeze_reason9 = None
                            if ui9.quit or ui9.abort:
                                human_end_reason9 = (
                                    "quit" if ui9.quit else "aborted")
                                record_human_decision9(
                                    "abort", reason=human_end_reason9)
                                break
                            if marker_selection9 is not None:
                                active_selection9 = {
                                    "instance_id": str(marker_selection9[
                                        "resolved_instance_id"]),
                                    "kind": str(marker_selection9[
                                        "resolved_kind"]),
                                    "target_cell": tuple(marker_selection9[
                                        "resolved_target_cell"]),
                                    "component_cells": frozenset(
                                        tuple(int(q9) for q9 in raw9)
                                        for raw9 in marker_selection9[
                                            "resolved_component_cells"]),
                                }
                                visible_commit_streak9 = 0
                            if freeze_reason9 == "quota":
                                active_selection9 = None
                                human_end_reason9 = "success"
                                break

                        action9 = ui9.poll_action(noop9)
                        if "sneak" in action9:
                            action9["sneak"] = 0
                        action9["_xbench_human_control"] = 1
                        pressed9 = any(int(np.asarray(
                            action9.get(button9, 0)).reshape(-1)[0])
                            for button9 in ("attack", "use")
                            if button9 in action9)
                        if pressed9:
                            action9["voxels"] = np.array(
                                [-6, 7, -4, 5, -6, 7], np.int32)

                        pending_source9 = TRAJ.get("pending_row")
                        if pending_source9 is None:
                            raise RuntimeError("human control has no source RGB")
                        eye9, yaw9, pitch9, source9 = label_camera9()
                        _right9, _up9, forward9 = G._basis(yaw9, pitch9)
                        ray_cell9, ray_entry9 = G.raycast(
                            eye9, forward9, MINE_REACH, human_ray_occ9,
                            unknown_is_solid=True)
                        ray_point9 = (None if ray_cell9 is None else [
                            float(eye9[index9])
                            + float(forward9[index9]) * float(ray_entry9)
                            for index9 in range(3)])
                        TRAJ["human_preaction_evidence"] = {
                            "version": 1,
                            "contract": HUMAN_DEFERRED_LABEL_CONTRACT,
                            "source_traj_t": int(pending_source9["t"]),
                            "render_pose_source": str(source9),
                            "ray_cell": (None if ray_cell9 is None else
                                         [int(q9) for q9 in ray_cell9]),
                            "ray_entry_distance": (
                                None if ray_cell9 is None else float(ray_entry9)),
                            "ray_point": ray_point9,
                            "chosen_marker": (
                                None if marker_selection9 is None else {
                                    **dict(marker_selection9),
                                    "key": "C",
                                    "edge": "key_down",
                                    "source_traj_t": int(
                                        pending_source9["t"]),
                                    "source_render_pose": dict(
                                        pending_source9.get("render_pose") or {}),
                                    "source_rgb_shape": list(
                                        np.asarray(pov9).shape),
                                    "source_rgb_sha256": hashlib.sha256(
                                        np.asarray(pov9, dtype=np.uint8)
                                        .tobytes()).hexdigest(),
                                    "annotation_pause_s": round(
                                        float(_marker_pause9), 6),
                                    "simulator_steps_during_pause": 0,
                                    "env_action_from_marker_click": False,
                                }),
                            "decision_events": list(
                                pending_source9.get(
                                    "human_decision_events") or []),
                        }

                        w.obs, _, terminated9, truncated9, w.info = (
                            w.sim.step(action9))
                        live_voxel_box9 = np.asarray(
                            action9.get("voxels", []), dtype=np.int32).reshape(-1)
                        removed9 = []
                        if (live_voxel_box9.size == 6
                                and bool((live_voxel_box9 != 0).any())):
                            raw_voxels9 = w.info.get("voxels")
                            if raw_voxels9 is None:
                                raise RuntimeError(
                                    "human mutation probe returned no voxel packet")
                            removed9 = human_ray_occ9.ingest(
                                raw_voxels9, w.get_pos(), live_voxel_box9)
                            human_marker_occ9.ingest(
                                raw_voxels9, w.get_pos(), live_voxel_box9)
                        TRAJ["target_cell"] = None
                        TRAJ["ray_hit"] = None
                        TRAJ["aim_point"] = None
                        rec.grab(
                            "human interact" if pressed9
                            else "human demonstration")
                        live_steps9 += 1
                        pov9 = np.asarray(w.info["pov"])
                        permitted_cells9 = (
                            frozenset() if active_selection9 is None else
                            active_selection9["component_cells"])
                        wrong_removed9 = [
                            {"cell": [int(q9) for q9 in cell10],
                             "kind": G._bare(str(kind10 or ""))}
                            for cell10, kind10 in removed9
                            if cell10 not in permitted_cells9]
                        if wrong_removed9:
                            wrong_break9 = {
                                "version": 1,
                                "contract": HUMAN_SESSION_CONTRACT,
                                "active_chosen_instance_id": (
                                    None if active_selection9 is None else
                                    active_selection9["instance_id"]),
                                "removed_blocks": wrong_removed9,
                            }
                            wrong_traj9, wrong_story9 = current_human_pair9()
                            wrong_traj9["human_wrong_block_break"] = dict(
                                wrong_break9)
                            wrong_story9["human_wrong_block_break"] = dict(
                                wrong_break9)
                        ui9.draw(
                            pov9, mission9,
                            f"frame={human_frame9} | "
                            f"{max(0.0, deadline9 - time.monotonic()):.1f}s left | "
                            "SHIFT disabled")
                        remaining_goal_cells9 = {
                            cell10 for cell10, kind10 in
                            human_ray_occ9.grid.items()
                            if G._bare(str(kind10 or ""))
                               == G._bare(str(ctx.get("cls") or ""))}
                        chosen_broken9 = bool(
                            active_selection9 is not None
                            and selected_component_removed(
                                active_selection9["component_cells"],
                                remaining_goal_cells9))
                        if (not chosen_broken9
                                and pending_auto_freeze_reason9 is None
                                and advance_live_visibility_latch9()):
                            pending_auto_freeze_reason9 = "visible_streak"
                        if chosen_broken9:
                            success_row9 = (STORY["rows"][-1]
                                            if STORY.get("rows") else None)
                            success_traj9 = TRAJ.get("pending_row")
                            success_evidence9 = (
                                success_traj9.get("human_preaction")
                                if isinstance(success_traj9, dict) else None)
                            if (success_row9 is None
                                    or not isinstance(success_evidence9, dict)):
                                raise RuntimeError(
                                    "human success lacks post-action RGB evidence")
                            source_ray9 = success_evidence9.get("ray_cell")
                            removed_selected9 = [
                                cell10 for cell10, _kind10 in removed9
                                if cell10 in active_selection9[
                                    "component_cells"]]
                            success_cell9 = (
                                list(source_ray9)
                                if isinstance(source_ray9, list)
                                and tuple(source_ray9) in active_selection9[
                                    "component_cells"] else
                                list(removed_selected9[0])
                                if removed_selected9 else
                                list(active_selection9["target_cell"]))
                            success_audit9 = {
                                "version": 1,
                                "source_action_row_t": int(success_traj9["t"]),
                                "source_rgb_traj_t": int(
                                    success_evidence9["source_traj_t"]),
                                "ray_cell": success_cell9,
                                "removed_blocks": [
                                    {"cell": [int(q9) for q9 in cell10],
                                     "kind": G._bare(str(kind10 or ""))}
                                    for cell10, kind10 in removed9
                                    if cell10 in active_selection9[
                                        "component_cells"]],
                                "chosen_instance_id": active_selection9[
                                    "instance_id"],
                                "chosen_kind": active_selection9["kind"],
                                "contract": HUMAN_DEFERRED_LABEL_CONTRACT,
                            }
                            success_row9["human_success_after_action"] = dict(
                                success_audit9)
                            success_traj9["human_success_after_action"] = dict(
                                success_audit9)
                            record_human_decision9(
                                "break_detected",
                                chosen_instance_id=active_selection9[
                                    "instance_id"],
                                broken_cell=success_cell9,
                                completed_after=int(human_completed9 + 1))
                            human_completed9 += 1
                            active_selection9 = None
                            pending_auto_freeze_reason9 = (
                                "quota" if human_completed9 >= int(ctx["quota"])
                                else "break_auto")
                            if pending_auto_freeze_reason9 == "quota":
                                ui9.play_completion_sound()
                        if terminated9 or truncated9:
                            human_end_reason9 = "sim_terminated"
                            break
                        elapsed9 = time.monotonic() - wall_frame_started9
                        if elapsed9 < frame_period9:
                            time.sleep(frame_period9 - elapsed9)
                        human_frame9 += 1
                    if human_end_reason9 is None:
                        human_end_reason9 = "timeout"
                finally:
                    ui9.grab_mouse(False)
                    ui9.close()
                live_wall_total9 = max(
                    0.0, time.monotonic() - live_started9)
                live_elapsed9 = max(
                    1e-9, live_wall_total9 - annotation_pause_s9)
                publish_human_status9(
                    "postplay_labels", cell=cell, setting=setting,
                    target_kind=human_ready_goal_name9,
                    quota=int(ctx["quota"]), completed=int(human_completed9),
                    end_reason=str(human_end_reason9),
                    live_control_steps=int(live_steps9))
                human_label_runtime9 = finalize_human_deferred_labels9(
                    ctx, human_replay_occ9,
                    end_reason9=human_end_reason9,
                    completed9=human_completed9)
                TRAJ["phase"] = int(
                    human_label_runtime9["human_terminal_phase"])
                TRAJ["last_phase"] = int(TRAJ["phase"])
                TRAJ["event"] = ""
                human_label_runtime9.update({
                    "live_control_steps": int(live_steps9),
                    "live_wall_total_s": round(float(live_wall_total9), 6),
                    "annotation_pause_count": int(annotation_pause_count9),
                    "annotation_pause_wall_s": round(
                        float(annotation_pause_s9), 6),
                    "live_control_wall_s_excluding_annotation": round(
                        float(live_elapsed9), 6),
                    "live_control_hz": round(
                        float(live_steps9) / live_elapsed9, 3),
                })
                episode_timing9["human_deferred_labels"] = human_label_runtime9
                episode_timing9.update({
                    "human_end_reason": str(human_end_reason9),
                    "human_marker_event_counts": dict(
                        human_label_runtime9["human_marker_event_counts"]),
                    "human_marker_missing_count": int(
                        human_label_runtime9["human_marker_missing_count"]),
                    "human_marker_contract_complete": bool(
                        human_label_runtime9[
                            "human_marker_contract_complete"]),
                })
                STRUCT["human_episode_summary"] = {
                    key9: value9 for key9, value9 in
                    human_label_runtime9.items()
                    if key9.startswith("human_")
                    or key9 in ("version", "marker_contract")
                }
                human_block_changes9 = []
                for row9 in STORY.get("rows", []):
                    for field9, change_kind9 in (
                            ("human_success_after_action", "chosen_break"),
                            ("human_wrong_block_break", "off_target_break")):
                        audit9 = row9.get(field9)
                        if not isinstance(audit9, dict):
                            continue
                        human_block_changes9.append({
                            "source_frame": int(row9["f"]),
                            "source_traj_t": int(row9["traj_t"]),
                            "change_kind": change_kind9,
                            **dict(audit9),
                        })
                human_session_doc9 = {
                    "version": 1,
                    "contract": HUMAN_SESSION_CONTRACT,
                    "episode_source": "human_play",
                    "oracle_motion_profile": HUMAN_MOTION_PROFILE,
                    "cell": cell,
                    "setting": setting,
                    "world_seed": int(args.world_seed),
                    "action_seed": int(args.action_seed),
                    "quota": int(ctx["quota"]),
                    "quota_seed": human_quota_seed9,
                    "completed": int(human_completed9),
                    "end_reason": str(human_end_reason9),
                    "aborted": bool(
                        human_end_reason9 in ("aborted", "quit")),
                    "timed_out": bool(human_end_reason9 == "timeout"),
                    "census_enabled": True,
                    "census_roster": list(mine_census_roster9),
                    "freeze_recorded_frame_count": 0,
                    "renderer_viewmodel_mask": {
                        "contract": RENDERER_VIEWMODEL_MASK_CONTRACT,
                        "action_marker":
                            RENDERER_VIEWMODEL_ACTION_MARKER,
                        "raw_pov_channels": 3,
                        "pixel_supervision_during_attack": True,
                    },
                    "start_gate": (
                        None if human_ready_gate9 is None else
                        dict(human_ready_gate9)),
                    "decision_events": decision_ledger9.document(),
                    "block_changes": human_block_changes9,
                    "runtime": dict(human_label_runtime9),
                }
                with open(os.path.join(
                        OUT, f"human_session_{cell}_{setting}.json"),
                        "w", encoding="utf-8") as human_meta_f9:
                    json.dump(human_session_doc9, human_meta_f9,
                              indent=2, sort_keys=True)
                    human_meta_f9.write("\n")
                print(
                    f"[human] {cell}/{setting} -> {human_end_reason9} "
                    f"completed={human_completed9}/{ctx['quota']} "
                    f"live_hz={human_label_runtime9['live_control_hz']}",
                    flush=True)

            completed9 = int(human_completed9)
            ok = int(completed9 >= ctx["quota"])
            result_label9 = "SUCCESS" if ok == 1 else "FAIL"
            timing_recording_finished9 = time.monotonic()
            episode_timing9.update({
                "episode_recording_wall_s": runtime_wall_seconds(
                    timing_recording_started9, timing_recording_finished9),
                "frames": int(len(rec.frames)),
                "result": int(ok),
            })
            results[f"{cell}/{setting}"] = ok
            if TRAJ["fh"] is not None:
                if TRAJ["pending_row"] is not None:
                    TRAJ["fh"].write(json.dumps(TRAJ["pending_row"]) + "\n")
                    TRAJ["pending_row"] = None
                TRAJ["fh"].close()
                TRAJ["fh"] = None
            if args.story:
                if (args.human_play and STORY.get("rows")
                        and isinstance(
                            STRUCT.get("human_episode_summary"), dict)):
                    STORY["rows"][-1]["human_episode_summary"] = dict(
                        STRUCT["human_episode_summary"])
                for story_row9 in STORY.get("rows", []):
                    sfh.write(json.dumps(story_row9) + "\n")
                sfh.flush()
                sfh.close()
                STORY["rows"] = []
            print(f"[demo] {cell:14s} {setting:5s} -> {result_label9} "
                  f"({len(rec.frames)} frames)", flush=True)
            timing_save_started9 = time.monotonic()
            rec.save()
            if args.human_play:
                raw_session9 = materialize_raw_session(
                    OUT, cell=cell, setting=setting)
                episode_timing9["human_raw_session"] = raw_session9
                publish_human_status9(
                    "raw_complete", cell=cell, setting=setting,
                    target_kind=str(
                        ctx.get("target_kind") or ctx.get("clsname")
                        or ctx.get("cls") or cell),
                    quota=int(ctx.get("quota", 0)),
                    completed=int(human_completed9),
                    end_reason=str(human_end_reason9),
                    frame_count=int(raw_session9.get("frame_count", 0)))
            timing_save_finished9 = time.monotonic()
            episode_timing9.update({
                "video_save_wall_s": runtime_wall_seconds(
                    timing_save_started9, timing_save_finished9),
                "episode_total_through_save_wall_s": runtime_wall_seconds(
                    timing_recording_started9, timing_save_finished9),
            })
            print(
                f"[timing] {episode_key9} pre_record="
                f"{episode_timing9['post_site_pre_record_wall_s']:.3f}s "
                f"record={episode_timing9['episode_recording_wall_s']:.3f}s "
                f"save={episode_timing9['video_save_wall_s']:.3f}s "
                f"through_save={episode_timing9['episode_total_through_save_wall_s']:.3f}s",
                flush=True)
            STORY["rec"] = None

    with open(os.path.join(OUT, "demo.json"), "w") as f:
        json.dump(results, f, indent=1)
    for result_key9, result_value9 in results.items():
        if result_key9 in runtime_timing9["episodes"]:
            runtime_timing9["episodes"][result_key9]["result"] = int(result_value9)
    timing_summary_written9 = time.monotonic()
    runtime_timing9["main_total_wall_s"] = runtime_wall_seconds(
        timing_main_started9, timing_summary_written9)
    with open(os.path.join(OUT, "runtime_timing.json"), "w") as timing_f9:
        json.dump(runtime_timing9, timing_f9, indent=1)
    print(f"[timing] total={runtime_timing9['main_total_wall_s']:.3f}s "
          "summary=runtime_timing.json", flush=True)
    print(f"[done] -> {OUT}", flush=True)
    w.close()


if __name__ == "__main__":
    sys.exit(main())
