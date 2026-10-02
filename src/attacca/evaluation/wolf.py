"""CCFW wolf stage: staging, scoring and equip support, and the hand-off from the furnace cooking stage."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence
import uuid

import cv2
import numpy as np

from attacca.evaluation import dpx as DPX
from attacca.evaluation import chain_engine as ENGINE
from attacca.evaluation.cooking_adapters import CAUSAL_FIFO_STEPS
from attacca.evaluation.cooking_adapters import HUNT_REACH
from attacca.evaluation.cooking_adapters import live_target_visibility
from attacca.evaluation.cooking import cook_beef_in_furnace
from attacca.evaluation.crafting import DiamondPickaxeCraftingMacro
from attacca.evaluation.crafting import MacroArtifactRecorder
from attacca.evaluation.crafting import MacroAuditError
from attacca.evaluation.crafting import find_item_slot
from attacca.evaluation.crafting import inventory_totals
from attacca.evaluation.crafting import player_slots
from attacca.evaluation.crafting import state_snapshot
from attacca.worlds.coal_cow_furnace_wolf_scene import WOLF_INITIAL_HEALTH
from attacca.worlds.coal_cow_furnace_wolf_scene import WOLF_INTERACTION_ID
from attacca.worlds.coal_cow_furnace_wolf_scene import WOLF_STAGE_KEY
from attacca.worlds.coal_cow_furnace_wolf_scene import WOLF_TAG
from attacca.worlds.human_viewmodel import RENDERER_ENTITY_ID_FIELD
from attacca.worlds.human_viewmodel import RENDERER_UNIQUE_WOLF_HEALTH_FIELD


MOBS_QUERY_BOX = np.asarray((-64, 65, -8, 17, -64, 65), dtype=np.int32)
EQUIP_CONTRACT = "xbench_real_gui_cooked_beef_hotbar_equip/v1"
WOLF_FEED_CONTRACT = "xbench_ccfw_wolf_feed_score/v3"


def uuid_int_array_to_string(values: Sequence[int]) -> str:
    if len(values) != 4:
        raise ValueError("UUID int-array must contain four integers")
    payload = b"".join(
        (int(value) & 0xffffffff).to_bytes(4, byteorder="big")
        for value in values)
    return str(uuid.UUID(bytes=payload))


def _bound_log_camera_off(world, stager, *, attempts: int = 16) -> Path:
    for _ in range(int(attempts)):
        if stager.bind_entity_log_camera_off():
            path = stager.LOGSTATE.get("log")
            if path and os.path.isfile(path):
                return Path(path)
        world.step_noop()
    raise RuntimeError("could not bind this simulator instance to its Minecraft log")


def _command_reply_camera_off(world, stager, command: str, *,
                              attempts: int = 10) -> str:
    log_path = _bound_log_camera_off(world, stager)
    offset = log_path.stat().st_size
    world.cmd(command)
    for _ in range(int(attempts)):
        world.step_noop()
        try:
            with log_path.open(errors="ignore") as stream:
                stream.seek(offset)
                tail = stream.read()
        except OSError as exc:
            raise RuntimeError(f"Minecraft log became unavailable: {log_path}") from exc
        if ("entity data:" in tail or "has the following entity data:" in tail
                or "Base value of attribute" in tail
                or "No entity was found" in tail or "Found no elements" in tail):
            return tail
    raise RuntimeError(f"no fresh Minecraft command reply for {command!r}")


def _parse_int_array(reply: str, *, label: str) -> tuple[int, int, int, int]:
    match = re.search(
        r"\[I;\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
        reply, re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"could not parse {label} int-array from fresh reply")
    values = tuple(int(value) for value in match.groups())
    if any(value < -(1 << 31) or value >= (1 << 31) for value in values):
        raise RuntimeError(f"{label} contains a value outside signed int32")
    return values


def _parse_scalar(reply: str, *, label: str) -> float:
    matches = re.findall(
        r"entity data:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?:[bdfsL])?",
        reply, re.IGNORECASE)
    if not matches:
        raise RuntimeError(f"could not parse {label} scalar from fresh reply")
    return float(matches[-1])


def _parse_pos(reply: str) -> tuple[float, float, float]:
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    match = re.search(
        rf"entity data:\s*\[({number})d,\s*({number})d,\s*({number})d\]",
        reply, re.IGNORECASE)
    if match is None:
        raise RuntimeError("could not parse wolf Pos from fresh reply")
    return tuple(float(value) for value in match.groups())


def _parse_attribute_base(reply: str) -> float:
    match = re.search(
        r"Base value of attribute .*? is\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        reply, re.IGNORECASE)
    if match is None:
        raise RuntimeError("could not parse wolf max-health attribute")
    return float(match.group(1))


def runtime_player_uuid_camera_off(world, stager) -> tuple[int, int, int, int]:
    reply = _command_reply_camera_off(
        world, stager, "/data get entity @p UUID")
    return _parse_int_array(reply, label="player UUID")


def _wolf_rows(info: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = []
    for row in info.get("mobs") or ():
        if (isinstance(row, Mapping)
                and "wolf" in str(row.get("name", "")).lower()):
            rows.append(row)
    return rows


def audit_staged_wolf_camera_off(world, stager, *, expected_position,
                                 expected_owner_uuid, expected_invulnerable: bool,
                                 expected_max_health: float) -> dict[str, Any]:
    selector_body = f"type=minecraft:wolf,tag={WOLF_TAG},limit=1"
    selector = f"@e[{selector_body}]"
    owner = _parse_int_array(_command_reply_camera_off(
        world, stager, f"/data get entity {selector} Owner"), label="wolf Owner")
    wolf_uuid_i = _parse_int_array(_command_reply_camera_off(
        world, stager, f"/data get entity {selector} UUID"), label="wolf UUID")
    position = _parse_pos(_command_reply_camera_off(
        world, stager, f"/data get entity {selector} Pos"))
    scalars = {}
    for path in ("Sitting", "PersistenceRequired", "Invulnerable", "Health"):
        scalars[path] = _parse_scalar(_command_reply_camera_off(
            world, stager, f"/data get entity {selector} {path}"), label=path)
    max_health = _parse_attribute_base(_command_reply_camera_off(
        world, stager,
        f"/attribute {selector} minecraft:generic.max_health base get"))
    if stager.entity_exists(f"type=minecraft:wolf,tag=!{WOLF_TAG},limit=1"):
        raise RuntimeError("an unowned/untagged wolf is present after CCFW staging")

    action = world.sim.noop_action()
    action["mobs"] = MOBS_QUERY_BOX.copy()
    world.obs, _reward, terminated, truncated, world.info = world.sim.step(action)
    if terminated or truncated:
        raise RuntimeError("simulator ended during staged wolf mobs readback")
    rows = _wolf_rows(world.info)
    renderer_health = float(world.info.get(
        RENDERER_UNIQUE_WOLF_HEALTH_FIELD, float("nan")))
    wolf_uuid = uuid_int_array_to_string(wolf_uuid_i)
    mobs_id = "" if len(rows) != 1 else str(rows[0].get("id", ""))
    expected = tuple(float(value) for value in expected_position)
    errors = []
    if tuple(owner) != tuple(int(value) for value in expected_owner_uuid):
        errors.append(f"Owner mismatch {owner}/{tuple(expected_owner_uuid)}")
    if math.dist(position, expected) > .2:
        errors.append(f"position mismatch {position}/{expected}")
    if scalars["Sitting"] != 1.0:
        errors.append(f"Sitting={scalars['Sitting']}")
    if scalars["PersistenceRequired"] != 1.0:
        errors.append(f"PersistenceRequired={scalars['PersistenceRequired']}")
    if bool(int(scalars["Invulnerable"])) != bool(expected_invulnerable):
        errors.append(f"Invulnerable={scalars['Invulnerable']}")
    if abs(scalars["Health"] - WOLF_INITIAL_HEALTH) > 1e-4:
        errors.append(f"Health={scalars['Health']}")
    if abs(max_health - float(expected_max_health)) > 1e-4:
        errors.append(f"max_health={max_health}/{expected_max_health}")
    if len(rows) != 1:
        errors.append(f"mobs wolf rows={len(rows)}")
    if not math.isfinite(renderer_health):
        errors.append("renderer unique-wolf Health is unavailable")
    elif abs(renderer_health - WOLF_INITIAL_HEALTH) > 1e-4:
        errors.append(f"renderer Health={renderer_health}")
    if errors:
        raise RuntimeError("staged wolf audit failed: " + "; ".join(errors))
    return {
        "contract": "xbench_ccfw_staged_wolf_live_audit/v1",
        "tag": WOLF_TAG,
        "position": list(position),
        "expected_position": list(expected),
        "owner_uuid_int_array": list(owner),
        "wolf_uuid_int_array": list(wolf_uuid_i),
        "wolf_uuid": wolf_uuid,
        "mobs_id": mobs_id,
        "mobs_id_available": bool(mobs_id),
        "mobs_life_available": bool(
            len(rows) == 1 and rows[0].get("life") is not None),
        "mobs_identity_binding": (
            "unique tagged NBT wolf bound to sole current mobs wolf row; "
            "this MineStudio 1.16 mobs payload has no stable id"
            if not mobs_id else
            "unique tagged NBT wolf bound to matching sole current mobs row"),
        "tamed_owner_match": True,
        "Sitting": True,
        "PersistenceRequired": True,
        "Invulnerable": bool(int(scalars["Invulnerable"])),
        "Health": float(scalars["Health"]),
        "renderer_same_tick_Health": renderer_health,
        "max_health": float(max_health),
        "unique_tagged_wolf": True,
        "untagged_wolf_exists": False,
        "mobs_readback": dict(rows[0]),
        "camera_off": True,
    }


class CookedBeefEquipMacro(DiamondPickaxeCraftingMacro):

    def run(self) -> dict[str, Any]:
        self.macro = "system_real_gui_cooked_beef_hotbar_equip"
        initial = state_snapshot(self.info)
        if int(inventory_totals(self.info).get("cooked_beef", 0)) != 1:
            raise MacroAuditError("equip precondition requires exactly one cooked_beef")
        source = find_item_slot(self.info, "cooked_beef", exact_quantity=1)
        hotbar_empty = [slot for slot in range(9)
                        if player_slots(self.info)[slot]["type"] == "none"]
        if not hotbar_empty:
            raise MacroAuditError("no empty hotbar slot for cooked_beef")
        destination = int(hotbar_empty[0])
        self._open_player_inventory()
        assert self.geometry is not None
        self.click(self.geometry.player_slot(source), button="left")
        self.click(self.geometry.player_slot(destination), button="left")
        slots = player_slots(self.info)
        if (slots[source]["type"] != "none"
                or slots[destination]["type"] != "cooked_beef"
                or int(slots[destination]["quantity"]) != 1
                or int(inventory_totals(self.info).get("cooked_beef", 0)) != 1):
            raise MacroAuditError("cooked_beef did not move exactly once to hotbar")
        cursor_probe = self._cursor_empty_probe(slot_id=source)
        self._close_gui()
        self._select_hotbar(destination + 1, "cooked_beef")
        final = state_snapshot(self.info)
        return {
            "contract": EQUIP_CONTRACT,
            "success": True,
            "input_route": "MineStudio inventory/camera/attack/hotbar",
            "commands_issued": 0,
            "teleports": 0,
            "direct_inventory_mutation": False,
            "world_camera_rotation": False,
            "source_slot": source,
            "hotbar_destination_slot": destination,
            "hotbar_key": destination + 1,
            "cursor_empty_probe": cursor_probe,
            "initial": initial,
            "final": final,
        }


def equip_cooked_beef_hotbar(world: Any, out: Path) -> dict[str, Any]:
    out = Path(out).expanduser().resolve()
    with MacroArtifactRecorder(out) as recorder:
        result = CookedBeefEquipMacro(
            world, recorder=recorder, settle_frames=1).run()
    return {**result, **recorder.artifact_manifest()}


def prepare_cooked_beef_for_wolf(world: Any, *, cook_out: Path,
                                 equip_out: Path) -> tuple[dict, dict]:
    cooking = cook_beef_in_furnace(world, cook_out)
    equip = equip_cooked_beef_hotbar(world, equip_out)
    return cooking, equip


def load_wolf_goal(source: Path, out: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    source = source.expanduser().resolve()
    meta_path = source / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"wolf goal metadata is missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("contract") != "xbench_hunt_cross_world_renderer_entity_id_goal/v2"
            or meta.get("target") != "wolf"):
        raise ValueError(f"invalid wolf goal contract: {source}")
    rgb_path, mask_path = source / "goal.png", source / "mask.png"
    if (DPX._sha256(rgb_path) != meta.get("goal_rgb_sha256")
            or DPX._sha256(mask_path) != meta.get("goal_mask_sha256")):
        raise ValueError(f"wolf goal SHA mismatch: {source}")
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or raw_mask is None or raw_mask.shape != bgr.shape[:2]:
        raise ValueError(f"could not decode wolf goal: {source}")
    mask = np.ascontiguousarray(raw_mask > 0, dtype=np.uint8)
    if not int(mask.sum()) or set(np.unique(raw_mask).tolist()) - {0, 255}:
        raise ValueError(f"wolf goal mask is empty/non-binary: {source}")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("goal.png", "mask.png", "meta.json"):
        shutil.copy2(source / name, out / name)
    copied = {**meta, "source": str(source),
              "goal": str((out / "goal.png").resolve()),
              "mask": str((out / "mask.png").resolve()),
              "goal_sha256": meta["goal_rgb_sha256"],
              "mask_sha256": meta["goal_mask_sha256"],
              "goal_role": "whole_exemplar_variant"}
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask, copied


class WolfFeedStageAdapter(ENGINE.Adapter):

    def __init__(self, world, stager, scene, task: str, *, wolf_position,
                 wolf_uuid: str, initial_health: float = WOLF_INITIAL_HEALTH,
                 interaction_id: int = WOLF_INTERACTION_ID):
        super().__init__(world, stager, scene, task)
        if task != WOLF_STAGE_KEY or int(interaction_id) != WOLF_INTERACTION_ID:
            raise ValueError("wolf_feed requires the USE interaction ID")
        self._wolf_position = tuple(float(value) for value in wolf_position)
        self.wolf_uuid = str(wolf_uuid).lower()
        self.initial_health = float(initial_health)
        self.initial_cooked = ENGINE.inventory_count(world, "cooked_beef")
        self.initial_held = ENGINE.held_item(world)
        if self.initial_held != "cooked_beef":
            raise ValueError("wolf_feed requires cooked_beef to be equipped")
        self.last_health = self.initial_health
        self.pending_uses: list[dict[str, Any]] = []
        self.wrong_target_use_events: list[dict[str, Any]] = []
        self.step_events: dict[int, dict[str, Any]] = {}
        self.max_visible_pixels = 0
        self.contact_use_count = 0
        self.health_change_seen = False
        self.consumption_seen = False

    @property
    def interaction_id(self) -> int:
        return WOLF_INTERACTION_ID

    @property
    def quota(self) -> int:
        return 1

    @property
    def progress(self) -> int:
        return int(self.success_step is not None)

    def _surface(self) -> dict[str, Any]:
        packed = int(np.asarray(
            self.world.info[RENDERER_ENTITY_ID_FIELD])[180, 320])
        kind = ENGINE.ENTITY_ID_CODE_CLASSES.get(packed >> 16)
        entity_id = packed & 0xffff
        player = tuple(float(value) for value in self.world.get_pos())
        distance = math.dist(
            (player[0], player[1] + 1.0, player[2]),
            (self._wolf_position[0], self._wolf_position[1] + .7,
             self._wolf_position[2]))
        return {
            "crosshair_surface_kind": kind,
            "crosshair_entity_id": None if not entity_id else int(entity_id),
            "target_in_reach": bool(distance <= HUNT_REACH),
            "target_distance": float(distance),
        }

    def pre_context(self) -> dict[str, Any]:
        return {
            **self._surface(),
            "cooked_beef_before": ENGINE.inventory_count(
                self.world, "cooked_beef"),
            "held_before": ENGINE.held_item(self.world),
            "wolf_health_before": float(self.last_health),
        }

    def prepare_action(self, action: dict[str, Any]) -> None:
        action["mobs"] = MOBS_QUERY_BOX.copy()

    def _health_readback(self) -> tuple[float | None, dict[str, Any] | None]:
        rows = _wolf_rows(self.world.info)
        if self.wolf_uuid:
            matching = [row for row in rows
                        if str(row.get("id", "")).lower() == self.wolf_uuid]
        else:
            matching = rows if len(rows) == 1 else []
        if len(matching) != 1:
            return None, None
        row = matching[0]
        try:
            life = float(self.world.info[RENDERER_UNIQUE_WOLF_HEALTH_FIELD])
        except (KeyError, TypeError, ValueError):
            return None, dict(row)
        if not math.isfinite(life):
            return None, dict(row)
        return life, dict(row)

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        del attack
        eligible_feed_use = bool(
            use and context.get("held_before") == "cooked_beef"
            and context.get("target_in_reach") is True)
        exact_target_at_use = bool(
            eligible_feed_use
            and context.get("crosshair_surface_kind") == "wolf")
        if eligible_feed_use:
            self.pending_uses.append({
                "step": int(step),
                "crosshair_entity_id": context.get("crosshair_entity_id"),
                "distance": context.get("target_distance"),
                "cooked_beef_before": int(context["cooked_beef_before"]),
                "health_before": float(context["wolf_health_before"]),
                "exact_target_at_use": exact_target_at_use,
                "exact_target_confirmation_step": None,
            })
        elif use:
            self.wrong_target_use_events.append({
                "step": int(step),
                "held": context.get("held_before"),
                "crosshair_surface_kind": context.get("crosshair_surface_kind"),
                "target_in_reach": context.get("target_in_reach"),
                "target_distance": context.get("target_distance"),
            })
        if (context.get("crosshair_surface_kind") == "wolf"
                and context.get("target_in_reach") is True):
            for row in self.pending_uses:
                if row["exact_target_confirmation_step"] is None:
                    row["exact_target_confirmation_step"] = int(step)
        self.pending_uses = [
            row for row in self.pending_uses
            if 0 <= int(step) - int(row["step"]) <= CAUSAL_FIFO_STEPS]
        cooked_after = ENGINE.inventory_count(self.world, "cooked_beef")
        health, mob_row = self._health_readback()
        if health is not None:
            self.last_health = float(health)
        self.consumption_seen = bool(
            self.consumption_seen
            or (self.initial_cooked == 1 and cooked_after == 0))
        self.health_change_seen = bool(
            self.health_change_seen
            or self.last_health > self.initial_health + 1e-4)
        causal = next((row for row in reversed(self.pending_uses)
                       if (row["exact_target_at_use"]
                           or row["exact_target_confirmation_step"] is not None)), None)
        success_now = bool(
            self.success_step is None and causal is not None
            and self.initial_cooked == 1 and cooked_after == 0
            and self.last_health > self.initial_health + 1e-4)
        if success_now:
            self.success_step = int(step)
            self.contact_use_count += 1
            self.success_evidence = {
                "action_step": int(causal["step"]),
                "observed_step": int(step),
                "confirmation_delay_steps": int(step) - int(causal["step"]),
                "wolf_uuid": self.wolf_uuid,
                "target_distance": causal["distance"],
                "cooked_beef_before": int(causal["cooked_beef_before"]),
                "cooked_beef_after": int(cooked_after),
                "cooked_beef_delta": int(cooked_after - self.initial_cooked),
                "health_before": float(self.initial_health),
                "health_after": float(self.last_health),
                "health_delta": float(self.last_health - self.initial_health),
                "exact_target_at_use": causal["exact_target_at_use"],
                "exact_target_confirmation_step": (
                    causal["exact_target_confirmation_step"]),
                "renderer_confirmation_delay_allowed": True,
                "causal_fifo_steps": CAUSAL_FIFO_STEPS,
                "same_causal_fifo_mobs_readback": True,
                "policy_use_required": True,
                "exact_wolf_surface_required": True,
            }
        visibility = live_target_visibility(
            self.world, entity_kinds=(
                () if self.success_step is not None else ("wolf",)))
        pixels = int(visibility["target_visible_pixels_live"])
        self.max_visible_pixels = max(self.max_visible_pixels, pixels)
        self.step_events[int(step)] = {
            "target_contact_use": exact_target_at_use,
            "causal_target_contact_use_confirmed": bool(success_now),
            "feed_use_pending_confirmation": bool(
                eligible_feed_use and not success_now),
            "exact_target_at_use": exact_target_at_use,
            "wrong_target_use": bool(use and not eligible_feed_use),
            "cooked_beef_after": int(cooked_after),
            "wolf_health_after": None if health is None else float(health),
            "wolf_mobs_identity_match": mob_row is not None,
            **visibility,
        }
        return {
            "success_now": success_now,
            "control_complete_now": False,
            "note": (
                f"FEED wolf beef={cooked_after} health={self.last_health:.1f}"
                if use or success_now else ""),
        }

    def finish_audit(self) -> bool:
        cooked_after = ENGINE.inventory_count(self.world, "cooked_beef")
        passed = bool(
            self.success_step is not None and self.initial_cooked == 1
            and cooked_after == 0
            and self.last_health > self.initial_health + 1e-4)
        if passed:
            failure_mode = None
        elif self.initial_cooked != 1:
            failure_mode = "cooked_beef_unavailable"
        elif self.max_visible_pixels <= 0:
            failure_mode = "wolf_never_visible_timeout"
        elif self.wrong_target_use_events and not self.contact_use_count:
            failure_mode = "wrong_target_use"
        elif self.consumption_seen and not self.health_change_seen:
            failure_mode = "consumption_without_health_change"
        elif self.health_change_seen and not self.consumption_seen:
            failure_mode = "health_change_without_consumption"
        else:
            failure_mode = "wolf_feed_timeout"
        self.terminal_audit = {
            "contract": WOLF_FEED_CONTRACT,
            "performed": self.success_step is not None,
            "passed": passed,
            "failure_mode": failure_mode,
            "wolf_uuid": self.wolf_uuid,
            "initial_cooked_beef": int(self.initial_cooked),
            "final_cooked_beef": int(cooked_after),
            "cooked_beef_delta": int(cooked_after - self.initial_cooked),
            "initial_held_item": self.initial_held,
            "initial_health": float(self.initial_health),
            "final_health": float(self.last_health),
            "health_delta": float(self.last_health - self.initial_health),
            "max_visible_pixels": int(self.max_visible_pixels),
            "contact_use_count": int(self.contact_use_count),
            "wrong_target_use_events": list(self.wrong_target_use_events),
            "final_gui_open": bool(
                self.world.info.get("is_gui_open", False)),
            "causal_fifo_mobs_health_readback": True,
            "causal_fifo_steps": CAUSAL_FIFO_STEPS,
            "scored_stage_commands": 0,
            "scored_stage_teleports": 0,
            "success_evidence": self.success_evidence,
        }
        return passed
