#!/usr/bin/env python3
"""Stage engine shared by the long-horizon chains: one policy rollout per stage with frame guards and success checks."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO) + "/src")
if str(REPO / "src/attacca/evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "src/attacca/evaluation"))

os.environ.setdefault("MINESTUDIO_DIR", str(REPO / ".minestudio"))
os.environ.setdefault("MINESTUDIO_GPU_RENDER", "1")
os.environ.setdefault("RENDER_DEVICES", "0")

from attacca.evaluation.policy import Rocket2GoalRunner
from attacca.evaluation import place as PLACE
from attacca.evaluation.place_scene import _owner_pixels_from_depth
from attacca.evaluation.place_scene import drain_toasts_camera_off
from attacca.evaluation.place_scene import held_item
from attacca.evaluation.place_scene import pose
from attacca.evaluation.place_scene import queried_types
from attacca.evaluation.place_scene import strict_depth_arrays
from attacca.evaluation.mine_scene import Stager
from attacca.worlds.entity_id_mask import ENTITY_ID_CODE_CLASSES
from attacca.worlds.entity_id_mask import entity_surfaces
from attacca.worlds.entity_id_mask import occlude_entity_ids_by_viewmodel
from attacca.worlds.eval_tail import post_success_frames_recorded
from attacca.worlds.eval_tail import post_success_tail_complete
from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_FIELD
from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_VALID_FIELD
from attacca.worlds.human_viewmodel import RENDERER_VIEWMODEL_INFO_FIELD
from attacca.worlds.interaction_eval_scenes import MINE_INTERACTION_ID
from attacca.worlds.interaction_eval_scenes import portal_activation_success
from attacca.worlds.interaction_eval_scenes import portal_resource_conversion_counts
from attacca.worlds.interaction_eval_scenes import portal_resource_conversion_success


CONTRACT = None
FRAME_CONTRACT = None
USE_INTERACTION_ID = 3
FPS = 20.0
MASK_SHAPE = (360, 640)
WORLD_SEEDS = {"portal": 1_820_000}
RESOURCE_SCOOP_TASKS = frozenset(("water_scoop",))
RESOURCE_POUR_TASKS = frozenset(("lava_pour",))
RESOURCE_MINE_TASKS = frozenset(("obsidian_mine",))
RESOURCE_TASKS = (
    RESOURCE_SCOOP_TASKS | RESOURCE_POUR_TASKS | RESOURCE_MINE_TASKS)
def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(PLACE.json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    value = np.ascontiguousarray(rgb, dtype=np.uint8)
    if value.shape != (360, 640, 3) or float(value.mean()) < 25.0:
        raise RuntimeError(f"invalid RGB for {path}: {value.shape}/{value.mean()}")
    if not cv2.imwrite(str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR)):
        raise OSError(path)


def make_writer(path: Path, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer {path}")
    return writer


def action_flag(action: Mapping[str, Any], key: str) -> bool:
    try:
        return bool(np.asarray(action.get(key, 0)).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return False


def inventory_count(world, kind: str) -> int:
    inventory = world.info.get("inventory") or ()
    slots = inventory.values() if isinstance(inventory, Mapping) else inventory
    total = 0
    for row in slots:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("type", "")).removeprefix("minecraft:") == kind:
            total += int(row.get("quantity", 1))
    return total


def hotbar_slot_for_item(world, kind: str) -> int:
    inventory = world.info.get("inventory") or ()
    items = (inventory.items() if isinstance(inventory, Mapping)
             else enumerate(inventory))
    matches = []
    for raw_slot, row in items:
        if not isinstance(row, Mapping):
            continue
        if (str(row.get("type", "")).removeprefix("minecraft:") == kind
                and int(row.get("quantity", 0)) > 0):
            slot = int(raw_slot)
            if 0 <= slot < 9:
                matches.append(slot + 1)
    if not matches:
        raise RuntimeError(f"no hotbar slot contains {kind!r}")
    return min(matches)


def query_box(world, cells: Sequence[Sequence[int]]) -> np.ndarray:
    bx, by, bz = (math.floor(value) for value in world.get_pos())
    xs, ys, zs = zip(*(tuple(int(q) for q in cell) for cell in cells))
    pad = 2
    return np.asarray([
        min(xs) - bx - pad, max(xs) + 1 - bx + pad,
        min(ys) - by - pad, max(ys) + 1 - by + pad,
        min(zs) - bz - pad, max(zs) + 1 - bz + pad,
    ], dtype=np.int32)


def current_query(world) -> dict[tuple[int, int, int], str]:
    origin = tuple(math.floor(value) for value in world.get_pos())
    return queried_types(world.info, origin)


def block_union(world, cells: Sequence[Sequence[int]], *,
                render_position: Sequence[float]) -> np.ndarray:
    output = np.zeros(MASK_SHAPE, np.uint8)
    if not cells:
        return output
    hand, depth = strict_depth_arrays(world)
    markers = [SimpleNamespace(cell=tuple(int(q) for q in cell))
               for cell in cells]
    _eye, owners, _stats = _owner_pixels_from_depth(
        hand, depth, pose(world, render_position=render_position), markers,
        lookup_scope="chain_stage_exact_actionable_cells")
    flat = output.reshape(-1)
    for marker in markers:
        flat[np.asarray(owners.get(marker.cell, ()), dtype=np.int32)] = 1
    return output


def entity_union(world, kinds: Sequence[str]) -> np.ndarray:
    if int(world.info.get(RENDERER_ENTITY_ID_VALID_FIELD, 0)) != 1:
        raise RuntimeError("renderer entity-ID transport is invalid")
    hand = np.asarray(world.info[RENDERER_VIEWMODEL_INFO_FIELD], dtype=np.uint8)
    packed = occlude_entity_ids_by_viewmodel(
        world.info[RENDERER_ENTITY_ID_FIELD], hand)
    selected = set(kinds)
    output = np.zeros(MASK_SHAPE, np.uint8)
    for surface in entity_surfaces(packed):
        if surface.kind in selected:
            output |= np.asarray(surface.mask, dtype=np.uint8)
    return output


def load_copied_goal(source: Path, out: Path, *, semantic: str
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    goal_path, mask_path = source / "goal.png", source / "mask.png"
    goal_bgr = cv2.imread(str(goal_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if goal_bgr is None or mask is None or not int((mask > 0).sum()):
        raise RuntimeError(f"goal is unavailable: {source}")
    target_goal, target_mask = out / "goal.png", out / "mask.png"
    shutil.copy2(goal_path, target_goal)
    shutil.copy2(mask_path, target_mask)
    return (
        cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB),
        np.ascontiguousarray(mask > 0, dtype=np.uint8),
        {"source": str(source.resolve()), "goal": str(target_goal.resolve()),
         "mask": str(target_mask.resolve()), "goal_sha256": sha256_file(target_goal),
         "mask_sha256": sha256_file(target_mask), "semantic": semantic},
    )


class Adapter:
    def __init__(self, world, stager: Stager, scene, task: str):
        self.world, self.stager, self.scene, self.task = world, stager, scene, task
        self.success_step: int | None = None
        self.policy_control_complete_step: int | None = None
        self.completed: set[Any] = set()
        self.latest_query: dict[tuple[int, int, int], str] = {}
        self.last_resource_use_step: int | None = None
        self.terminal_audit: dict[str, Any] = {}
        self.success_evidence: dict[str, Any] | None = None
        if task in RESOURCE_SCOOP_TASKS:
            self.latest_query = {cell: "water" for cell in scene.water_cells}
        elif task in RESOURCE_POUR_TASKS:
            self.latest_query = {cell: "lava" for cell in scene.lava_cells}
        elif task in RESOURCE_MINE_TASKS:
            self.latest_query = {
                cell: "obsidian" for cell in scene.lava_cells}

    @property
    def interaction_id(self) -> int:
        return (MINE_INTERACTION_ID
                if self.task in RESOURCE_MINE_TASKS
                else USE_INTERACTION_ID)

    @property
    def quota(self) -> int:
        if self.task in RESOURCE_POUR_TASKS:
            return len(self.scene.lava_cells)
        if self.task in RESOURCE_SCOOP_TASKS:
            return 1
        raise ValueError(f"unsupported adapter task {self.task!r}")

    @property
    def tracking_quota(self) -> int:
        return self.quota

    @property
    def progress(self) -> int:
        if self.task in RESOURCE_SCOOP_TASKS:
            return int(inventory_count(self.world, "water_bucket") > 0)
        if self.task in RESOURCE_POUR_TASKS:
            return portal_resource_conversion_counts(
                self.latest_query, self.scene)["obsidian"]
        if self.task == "portal_build":
            return len(self.completed)
        return int(self.success_step is not None)

    def pre_context(self) -> dict[str, Any]:
        if self.task in RESOURCE_TASKS:
            return {
                "bucket_before": inventory_count(self.world, "bucket"),
                "water_bucket_before": inventory_count(
                    self.world, "water_bucket"),
                "obsidian_inventory_before": inventory_count(
                    self.world, "obsidian"),
                "held_before": held_item(self.world),
            }
        return {}

    def prepare_action(self, action: dict[str, Any]) -> None:
        if self.task in RESOURCE_SCOOP_TASKS:
            action["voxels"] = query_box(self.world, self.scene.water_cells)
        elif self.task in (RESOURCE_POUR_TASKS | RESOURCE_MINE_TASKS):
            action["voxels"] = query_box(self.world, self.scene.lava_cells)
        elif self.task in ("portal_build", "portal_ignite"):
            cells = (
                self.scene.complete_frame_cells
                + self.scene.portal_interior_cells)
            action["voxels"] = query_box(self.world, cells)

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        note = ""
        success_now = False
        if self.task in RESOURCE_TASKS:
            self.latest_query = current_query(self.world)
            if use:
                self.last_resource_use_step = int(step)
            if self.task in RESOURCE_SCOOP_TASKS:
                water_buckets = inventory_count(self.world, "water_bucket")
                success_now = bool(
                    self.last_resource_use_step is not None
                    and int(step) - self.last_resource_use_step <= 6
                    and water_buckets > 0
                    and self.success_step is None)
                if use or success_now:
                    note = (
                        f"SCOOP water_bucket={water_buckets} "
                        f"held={held_item(self.world)}")
            elif self.task in RESOURCE_POUR_TASKS:
                counts = portal_resource_conversion_counts(
                    self.latest_query, self.scene)
                control_complete_now = bool(
                    use
                    and int(context.get("water_bucket_before", 0)) > 0
                    and inventory_count(self.world, "water_bucket") == 0
                    and inventory_count(self.world, "bucket") >= 1
                    and counts["other"] == 0
                    and self.policy_control_complete_step is None)
                if control_complete_now:
                    self.policy_control_complete_step = int(step)
                success_now = bool(
                    self.last_resource_use_step is not None
                    and inventory_count(self.world, "water_bucket") == 0
                    and inventory_count(self.world, "bucket") >= 1
                    and portal_resource_conversion_success(
                        self.latest_query, self.scene)
                    and self.success_step is None)
                if use or self.last_resource_use_step is not None:
                    note = (
                        f"POUR obsidian={counts['obsidian']}/"
                        f"{len(self.scene.lava_cells)} "
                        f"cobble={counts['cobblestone']}")
                    if control_complete_now:
                        note += " | BUCKET EMPTIED; WATER SETTLING"
        elif self.task in ("portal_build", "portal_ignite"):
            self.latest_query = current_query(self.world)
            for cell in self.scene.build_frame_cells:
                if str(self.latest_query.get(cell, "")).removeprefix(
                        "minecraft:") == "obsidian":
                    self.completed.add(cell)
            if self.task == "portal_build":
                frame_complete = all(
                    str(self.latest_query.get(cell, "air")).removeprefix(
                        "minecraft:") == "obsidian"
                    for cell in self.scene.complete_frame_cells)
                success_now = frame_complete and self.success_step is None
            elif portal_activation_success(self.latest_query, self.scene):
                success_now = self.success_step is None
            if use:
                note = f"RMB held={held_item(self.world)}"
        if success_now:
            self.success_step = int(step)
            self.success_evidence = {
                "step": int(step), "task": self.task,
                "event_time_query": {
                    ",".join(map(str, cell)): value
                    for cell, value in self.latest_query.items()},
            }
        return {
            "success_now": bool(success_now),
            "control_complete_now": bool(
                self.task in RESOURCE_POUR_TASKS
                and self.policy_control_complete_step == int(step)),
            "note": note,
        }

    def finish_audit(self) -> bool:
        if self.success_step is None:
            self.terminal_audit = {"performed": False}
            return False
        self.terminal_audit = {
            "performed": True, "passed": True,
            "verdict_source": "latched_success_event_before_non_scoring_tail",
            "evidence": self.success_evidence,
        }
        return True


def verify_video(path: Path, *, expected_frames: int,
                 expected_size: tuple[int, int]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    observed_size = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            observed_size = (int(frame.shape[1]), int(frame.shape[0]))
    finally:
        capture.release()
    if decoded != int(expected_frames) or observed_size != expected_size:
        raise RuntimeError(
            f"video receipt mismatch {path}: frames={decoded}/"
            f"{expected_frames}, size={observed_size}/{expected_size}")
    return {
        "path": str(path.resolve()), "decoded_frames": decoded,
        "size": list(observed_size), "sha256": sha256_file(path),
    }


def assess_guarded_frame(
        rgb: np.ndarray, *, current_y: float, previous_y: float | None,
        threshold: float = 25.0) -> tuple[str | None, float]:
    frame = np.asarray(rgb)
    if frame.shape != (360, 640, 3) or frame.dtype != np.uint8:
        raise ValueError(
            f"raw POV must be uint8 360x640 RGB, got {frame.dtype}{frame.shape}")
    if not bool(np.isfinite(frame).all()) or not math.isfinite(float(current_y)):
        raise ValueError("raw POV/frame pose contains non-finite values")
    dy = 0.0 if previous_y is None else float(current_y) - float(previous_y)
    reason = None
    if float(frame.mean()) < float(threshold):
        reason = "near_black_mean_below_25"
    elif abs(dy) > 1.5:
        reason = "falling_abs_delta_y_above_1_5"
    return reason, float(current_y)


def rollout(*, world, runner: Rocket2GoalRunner, adapter: Adapter,
            out: Path, goal_rgb: np.ndarray, goal_mask: np.ndarray,
            goal_meta: Mapping[str, Any], goal_label: str,
            budget: int, post: int,
            post_action_mode: str = "policy",
            observation_wait_override: int | None = None,
            guard_frames: bool = False,
            dynamic_goal_provider: Any | None = None) -> dict[str, Any]:
    if CONTRACT is None or FRAME_CONTRACT is None:
        raise RuntimeError("the chain pipeline must set CONTRACT and FRAME_CONTRACT")
    if post_action_mode not in ("policy", "observe_noop"):
        raise ValueError(f"unsupported post_action_mode {post_action_mode!r}")
    raw_path = out / "raw_policy.mp4"
    raw_writer = make_writer(raw_path, (640, 360))
    runner.prepare_task(adapter.task)
    runner.set_obj_id(adapter.interaction_id)
    rows = []
    guard_previous_y = None
    guarded_kept_frames = 0
    guarded_drop_counts: dict[str, int] = {}
    started = time.monotonic()
    terminated = truncated = False
    default_observation_wait = (
        200 if adapter.task in RESOURCE_POUR_TASKS else 0)
    observation_wait = (
        default_observation_wait if observation_wait_override is None
        else int(observation_wait_override))
    if observation_wait < 0:
        raise ValueError("observation_wait_override must be non-negative")
    try:
        for step in range(int(budget) + int(post) + observation_wait):
            if (step >= int(budget) and adapter.success_step is None
                    and adapter.policy_control_complete_step is None
                    and not bool(getattr(
                        adapter, "allow_step_after_budget",
                        lambda _step: False)(step))):
                break
            budget_grace_observation = bool(
                step >= int(budget)
                and getattr(
                    adapter, "allow_step_after_budget",
                    lambda _step: False)(step))
            post_success_observation = adapter.success_step is not None
            post_policy_observation = bool(
                post_success_observation
                or adapter.policy_control_complete_step is not None
                or budget_grace_observation)
            policy_inference_skipped = bool(
                post_success_observation
                and post_action_mode == "observe_noop")
            pre_step_system_event = (
                None if policy_inference_skipped else
                getattr(adapter, "before_policy_step", lambda _step: None)(step))
            pre_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
            pre_action_position = tuple(
                float(value) for value in world.get_pos())
            progress_before = int(adapter.progress)
            held_before = held_item(world)
            context = adapter.pre_context()
            goal_condition = None
            if (not policy_inference_skipped
                    and dynamic_goal_provider is not None):
                goal_rgb, goal_mask, goal_condition = (
                    dynamic_goal_provider.resolve(
                        step, pre_rgb,
                        np.ascontiguousarray(
                            world.obs["image"], dtype=np.uint8)))
            if policy_inference_skipped:
                action = {}
                exist = None
                prediction = {}
                env_action = world.sim.noop_action()
            else:
                action, exist = runner.act(world.obs, goal_rgb, goal_mask)
                prediction = runner.capture_prediction()
                env_action = world.sim.agent_action_to_env_action(action)
            policy_action_suppressed = bool(
                post_policy_observation
                and post_action_mode == "observe_noop")
            if policy_action_suppressed:
                env_action = world.sim.noop_action()
            use = action_flag(env_action, "use")
            attack = action_flag(env_action, "attack")
            adapter.prepare_action(env_action)
            world.obs, reward, terminated, truncated, world.info = (
                world.sim.step(env_action))
            post_action_position = tuple(
                float(value) for value in world.get_pos())
            event = adapter.after_step(
                step=step, use=use, attack=attack, context=context)
            note = str(event["note"])
            if policy_action_suppressed:
                observation_label = (
                    "SUCCESS TAIL" if post_success_observation
                    else "ENVIRONMENT SETTLING")
                note = f"{note} {observation_label}".strip()
            phase = prediction.get("phase_argmax")
            if phase:
                note = f"{note} pred_phase={phase}".strip()
            drop_reason = None
            if guard_frames:
                drop_reason, guard_previous_y = assess_guarded_frame(
                    pre_rgb, current_y=pre_action_position[1],
                    previous_y=guard_previous_y)
            video_frame_index = None
            if drop_reason is None:
                video_frame_index = int(guarded_kept_frames)
                guarded_kept_frames += 1
            else:
                guarded_drop_counts[drop_reason] = (
                    int(guarded_drop_counts.get(drop_reason, 0)) + 1)
            if video_frame_index is not None:
                raw_writer.write(cv2.cvtColor(pre_rgb, cv2.COLOR_RGB2BGR))
            rows.append({
                "contract": FRAME_CONTRACT,
                "task": adapter.task,
                "step": int(step),
                "interaction_id": int(adapter.interaction_id),
                "post_success_observation": bool(post_success_observation),
                "post_policy_observation": bool(post_policy_observation),
                "post_success_action_mode": str(post_action_mode),
                "policy_action_suppressed": bool(policy_action_suppressed),
                "policy_inference_skipped": bool(policy_inference_skipped),
                "action_context": PLACE.json_safe(context),
                **({
                    "pre_step_system_event": PLACE.json_safe(
                        pre_step_system_event),
                    "budget_grace_observation": bool(
                        budget_grace_observation),
                } if hasattr(adapter, "before_policy_step") else {}),
                **({"goal_condition": PLACE.json_safe(goal_condition)}
                   if dynamic_goal_provider is not None else {}),
                "use": int(use), "attack": int(attack),
                "held_item_before": held_before,
                "held_item_after": held_item(world),
                "action": PLACE.json_safe(action),
                "executed_env_action": PLACE.json_safe(env_action),
                "returned_exist_probability": (
                    None if exist is None else float(exist)),
                "prediction": prediction,
                "pre_action_position": [
                    float(value) for value in pre_action_position],
                "post_action_position": [
                    float(value) for value in post_action_position],
                "telemetry": None,
                "video_frame_kept": video_frame_index is not None,
                "video_frame_index": video_frame_index,
                "video_drop_reason": drop_reason,
                "exact_actionable_pixels": None,
                "exact_mask_semantic": "not_materialized_deferred_or_metrics_eval",
                "progress_before": progress_before,
                "progress_after": int(adapter.progress),
                "quota": int(adapter.quota),
                "tracking_quota": int(adapter.tracking_quota),
                "success_event": bool(event["success_now"]),
                "note": note, "reward": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            if terminated or truncated:
                break
            if post_success_tail_complete(adapter.success_step, step, post):
                break
    finally:
        raw_writer.release()
    terminal_success = adapter.finish_audit()
    raw_receipt = verify_video(
        raw_path, expected_frames=guarded_kept_frames,
        expected_size=(640, 360))
    phase_head_available = any(
        row["prediction"].get("phase_argmax") is not None for row in rows)
    if not phase_head_available:
        raise RuntimeError(
            f"{adapter.task} produced no phase-head predictions")
    with (out / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(
                PLACE.json_safe(row), sort_keys=True) + "\n")
    result = {
        "contract": CONTRACT,
        "state": "completed",
        "task": adapter.task,
        "world_seed": WORLD_SEEDS[
            "portal" if adapter.task.startswith("portal_") else adapter.task],
        "interaction_id": int(adapter.interaction_id),
        "policy_rollout_seed": int(runner.policy_rollout_seed),
        "policy_runtime": runner.rollout_metadata(),
        "policy_movement_camera_use_rewritten": bool(
            post_action_mode == "observe_noop"
            and adapter.success_step is not None),
        "policy_action_unmodified_before_success": bool(
            adapter.policy_control_complete_step is None),
        "policy_action_unmodified_before_control_complete": True,
        "goal": dict(goal_meta),
        "budget": int(budget),
        "post_success_frames_requested": int(post),
        "post_success_tail_is_non_scoring": True,
        "post_success_action_mode": str(post_action_mode),
        "post_success_policy_inference": (
            "stopped" if post_action_mode == "observe_noop" else "continued"),
        "continue_after_success_to_budget": False,
        "artifact_mode": "deferred",
        "dynamic_goal_summary": (
            dynamic_goal_provider.summary()
            if dynamic_goal_provider is not None else None),
        "goal_conditioning": (
            str(dynamic_goal_provider.goal_conditioning)
            if dynamic_goal_provider is not None
            else "goal_exemplar_rgb_mask"),
        "live_exact_mask_materialized": False,
        "prediction_overlay_rendered_inline": False,
        "prediction_overlay_deferred": True,
        "telemetry_probe_enabled": False,
        "frame_guard": {
            "enabled": bool(guard_frames),
            "near_black_mean_threshold": 25.0,
            "falling_abs_delta_y_threshold": 1.5,
            "audit_rows": len(rows),
            "kept_video_frames": int(guarded_kept_frames),
            "dropped_video_frames": int(len(rows) - guarded_kept_frames),
            "drop_reason_counts": dict(sorted(guarded_drop_counts.items())),
            "previous_y_anchors_every_observation": True,
        },
        "success_step": adapter.success_step,
        "post_success_frames_recorded": post_success_frames_recorded(
            adapter.success_step, len(rows)),
        "policy_control_complete_step": adapter.policy_control_complete_step,
        "policy_stopped_after_control_complete": bool(
            adapter.policy_control_complete_step is not None),
        "steps": len(rows),
        "success": int(terminal_success),
        "terminal_audit": adapter.terminal_audit,
        "final_progress": int(adapter.progress),
        "quota": int(adapter.quota),
        "tracking_quota": int(adapter.tracking_quota),
        "use_frames": sum(int(row["use"]) for row in rows),
        "attack_frames": sum(int(row["attack"]) for row in rows),
        "phase_argmax_counts": {
            name: sum(
                int(row["prediction"].get("phase_argmax") == name)
                for row in rows)
            for name in ("EXPLORE", "APPROACH", "INTERACT")},
        "phase_head_available": bool(phase_head_available),
        "raw_video": str(raw_path.resolve()),
        "review_video": None,
        "raw_video_receipt": raw_receipt,
        "review_video_receipt": None,
        "frames_jsonl": str((out / "frames.jsonl").resolve()),
        "rollout_wall_seconds": float(time.monotonic() - started),
    }
    if observation_wait_override is not None:
        result["observation_wait_override"] = int(observation_wait)
    atomic_json(out / "result.json", result)
    return result


def initial_artifacts(world, out: Path) -> None:
    initial_rgb = np.ascontiguousarray(world.info["pov"], dtype=np.uint8)
    save_rgb(out / "frame0_rgb.png", initial_rgb)
    atomic_json(out / "frame0.json", {
        "frame0_exact_pixels": None,
        "start_target_hidden": None,
        "live_exact_mask_materialized": False,
    })
