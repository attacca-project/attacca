#!/usr/bin/env python3
"""Crafting macros for the DPX chain that operate Minecraft's crafting GUI through ordinary actions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import cv2
import numpy as np


CONTRACT = "xbench_diamond_pickaxe_crafting_macro/v1"
FRAME_CONTRACT = "xbench_diamond_pickaxe_system_macro_frame/v1"
AUDIT_LIMITATION = (
    "MineStudio live telemetry exposes PlayerInventory slots 0..35 but not "
    "the open crafting-grid/output slots or the carried cursor stack; those "
    "are visually recorded and cursor-empty is actively probed in a known "
    "empty player slot."
)
NATIVE_WIDTH = 640
NATIVE_HEIGHT = 360
GUI_WIDTH = 176
GUI_HEIGHT = 166
GUI_SCALE = 1
CURSOR_SIZE = 16
MOUSE_PIXELS_PER_CAMERA_DEGREE = 20.0 / 3.0
EQUIPPED_PROOF_FRAMES = 10


class MacroAuditError(RuntimeError):
    pass


def namespaceless(value: object) -> str:
    return str(value or "").removeprefix("minecraft:")


def _number(value: object, default: int = 0) -> int:
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return int(default)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def player_slots(info: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    raw = info.get("inventory") or {}
    items = raw.items() if isinstance(raw, Mapping) else enumerate(raw)
    present: dict[int, Mapping[str, Any]] = {}
    for raw_slot, row in items:
        if not isinstance(row, Mapping):
            continue
        try:
            slot_id = int(raw_slot)
        except (TypeError, ValueError):
            slot_id = _number(row.get("slot_id"), -1)
        if 0 <= slot_id < 36:
            present[slot_id] = row
    output: dict[int, dict[str, Any]] = {}
    for slot_id in range(36):
        row = present.get(slot_id, {})
        kind = namespaceless(row.get("type", "none"))
        quantity = _number(row.get("quantity"), 0)
        if kind in ("", "air", "none") or quantity <= 0:
            kind, quantity = "none", 0
        output[slot_id] = {"type": kind, "quantity": int(quantity)}
    return output


def inventory_totals(info: Mapping[str, Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in player_slots(info).values():
        kind, quantity = str(row["type"]), int(row["quantity"])
        if kind != "none" and quantity > 0:
            totals[kind] = totals.get(kind, 0) + quantity
    return dict(sorted(totals.items()))


def normalized_counter(info: Mapping[str, Any], family: str) -> dict[str, int]:
    raw = info.get(family) or {}
    if not isinstance(raw, Mapping):
        return {}
    return {
        namespaceless(key): _number(value)
        for key, value in raw.items()
        if _number(value) != 0
    }


def counter_delta(
        before: Mapping[str, int], after: Mapping[str, int], kind: str) -> int:
    key = namespaceless(kind)
    return int(after.get(key, 0)) - int(before.get(key, 0))


def held_item(info: Mapping[str, Any]) -> str:
    equipped = info.get("equipped_items") or {}
    if not isinstance(equipped, Mapping):
        return ""
    mainhand = equipped.get("mainhand") or {}
    if isinstance(mainhand, str):
        try:
            mainhand = json.loads(mainhand)
        except (TypeError, ValueError, json.JSONDecodeError):
            return namespaceless(mainhand)
    if not isinstance(mainhand, Mapping):
        return ""
    return namespaceless(mainhand.get("type", ""))


def state_snapshot(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gui_open": bool(info.get("is_gui_open", False)),
        "held_item": held_item(info),
        "player_slots": {
            str(slot): dict(row) for slot, row in player_slots(info).items()},
        "inventory_totals": inventory_totals(info),
        "craft_item": normalized_counter(info, "craft_item"),
    }


@dataclass(frozen=True)
class SlotTarget:
    name: str
    gui_kind: str
    x: int
    y: int
    player_slot_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gui_kind": self.gui_kind,
            "xy": [self.x, self.y],
            "player_slot_id": self.player_slot_id,
        }


@dataclass(frozen=True)
class GuiGeometry:
    kind: str
    screen_width: int
    screen_height: int
    gui_scale: int
    origin_x: int
    origin_y: int
    width: int = GUI_WIDTH
    height: int = GUI_HEIGHT

    @classmethod
    def native(cls, kind: str, *, pov_shape: Sequence[int],
               gui_scale: int = GUI_SCALE) -> "GuiGeometry":
        shape = tuple(int(v) for v in pov_shape)
        if shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3):
            raise MacroAuditError(
                f"GUI macro requires 640x360 RGB, got {shape}")
        if int(gui_scale) != GUI_SCALE:
            raise MacroAuditError(
                f"GUI macro requires gui scale 1, got {gui_scale}")
        if kind not in ("player", "workbench"):
            raise ValueError(f"unknown GUI kind {kind!r}")
        x0 = (NATIVE_WIDTH - GUI_WIDTH) // 2
        y0 = (NATIVE_HEIGHT - GUI_HEIGHT) // 2
        if (x0, y0) != (232, 97):
            raise MacroAuditError(
                f"centered vanilla GUI origin changed: {(x0, y0)}")
        return cls(kind, NATIVE_WIDTH, NATIVE_HEIGHT, GUI_SCALE, x0, y0)

    @property
    def cursor_open_xy(self) -> tuple[int, int]:
        return self.screen_width // 2, self.screen_height // 2

    def _target(self, name: str, x: int, y: int,
                player_slot_id: int | None = None) -> SlotTarget:
        return SlotTarget(
            name=name, gui_kind=self.kind,
            x=self.origin_x + int(x), y=self.origin_y + int(y),
            player_slot_id=player_slot_id)

    def player_slot(self, slot_id: int) -> SlotTarget:
        slot_id = int(slot_id)
        if not 0 <= slot_id < 36:
            raise ValueError(f"player slot out of range: {slot_id}")
        if slot_id < 9:
            col, rel_y = slot_id, 150
        else:
            offset = slot_id - 9
            col, row = offset % 9, offset // 9
            rel_y = 92 + 18 * row
        return self._target(
            f"player_slot_{slot_id}", 16 + 18 * col, rel_y,
            player_slot_id=slot_id)

    def input_slot(self, index: int) -> SlotTarget:
        index = int(index)
        if self.kind == "player":
            if not 0 <= index < 4:
                raise ValueError(f"player crafting slot out of range: {index}")
            col, row = index % 2, index // 2
            return self._target(
                f"player_craft_{index}", 106 + 18 * col, 26 + 18 * row)
        if not 0 <= index < 9:
            raise ValueError(f"workbench crafting slot out of range: {index}")
        col, row = index % 3, index // 3
        return self._target(
            f"workbench_craft_{index}", 38 + 18 * col, 25 + 18 * row)

    def output_slot(self) -> SlotTarget:
        rel = (162, 36) if self.kind == "player" else (132, 43)
        return self._target(f"{self.kind}_output", *rel)

    def manifest(self) -> dict[str, Any]:
        return {
            "contract": "minecraft_java_1_16_5_centered_gui_geometry/v1",
            "kind": self.kind,
            "screen_size": [self.screen_width, self.screen_height],
            "gui_scale": self.gui_scale,
            "gui_size": [self.width, self.height],
            "origin_xy": [self.origin_x, self.origin_y],
            "cursor_size": CURSOR_SIZE,
            "cursor_open_xy": list(self.cursor_open_xy),
            "mouse_pixels_per_camera_degree": MOUSE_PIXELS_PER_CAMERA_DEGREE,
        }


def find_item_slot(info: Mapping[str, Any], kind: str,
                   *, exact_quantity: int | None = None) -> int:
    wanted = namespaceless(kind)
    matches = []
    for slot_id, row in player_slots(info).items():
        if row["type"] != wanted:
            continue
        if exact_quantity is not None and int(row["quantity"]) != exact_quantity:
            continue
        matches.append(slot_id)
    if len(matches) != 1:
        raise MacroAuditError(
            f"expected one {wanted} player slot, found {matches}")
    return matches[0]


def find_empty_slot(info: Mapping[str, Any], *, main_inventory_only: bool = True,
                    exclude: Sequence[int] = ()) -> int:
    excluded = {int(value) for value in exclude}
    candidates = range(9, 36) if main_inventory_only else range(36)
    for slot_id in candidates:
        if slot_id not in excluded and player_slots(info)[slot_id]["type"] == "none":
            return slot_id
    raise MacroAuditError("no known-empty player inventory slot is available")


class MacroArtifactRecorder:

    def __init__(self, out_dir: str | Path, *, fps: float = 20.0):
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.out_dir / "system_macro_raw.mp4"
        self.jsonl_path = self.out_dir / "system_macro_frames.jsonl"
        for path in (self.raw_path, self.jsonl_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite macro artifact {path}")
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        self.rows: list[dict[str, Any]] = []
        self._jsonl = self.jsonl_path.open("x", encoding="utf-8")
        self._raw_writer: cv2.VideoWriter | None = None
        self._closed = False

    def _writer(self) -> cv2.VideoWriter:
        if self._raw_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            size = (NATIVE_WIDTH, NATIVE_HEIGHT)
            self._raw_writer = cv2.VideoWriter(
                str(self.raw_path), fourcc, self.fps, size)
            if not self._raw_writer.isOpened():
                self.close()
                raise RuntimeError("could not open macro video writer")
        return self._raw_writer

    def capture(self, rgb: np.ndarray, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("macro recorder is closed")
        frame = np.asarray(rgb, dtype=np.uint8)
        if frame.shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3):
            raise MacroAuditError(f"invalid macro RGB shape {frame.shape}")
        safe_row = json_safe(dict(row))
        self.rows.append(safe_row)
        self._jsonl.write(json.dumps(safe_row, sort_keys=True) + "\n")
        self._jsonl.flush()
        self._writer().write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def save_evidence(self, name: str, rgb: np.ndarray) -> str:
        path = self.out_dir / str(name)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence {path}")
        frame = np.asarray(rgb, dtype=np.uint8)
        if frame.shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3):
            raise MacroAuditError(f"invalid evidence RGB shape {frame.shape}")
        if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write evidence {path}")
        return str(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _video_receipt(cls, path: Path, *, expected_frames: int,
                       expected_size: tuple[int, int]) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(path))
        decoded = 0
        size = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                size = [int(frame.shape[1]), int(frame.shape[0])]
        finally:
            capture.release()
        if decoded != int(expected_frames) or size != list(expected_size):
            raise RuntimeError(
                f"macro video receipt mismatch {path}: "
                f"frames={decoded}/{expected_frames}, "
                f"size={size}/{list(expected_size)}")
        return {"path": str(path), "sha256": cls._sha256(path),
                "decoded_frames": decoded, "size": size}

    def artifact_manifest(self) -> dict[str, Any]:
        if not self._closed:
            raise RuntimeError("close macro recorder before requesting receipts")
        raw_receipt = self._video_receipt(
            self.raw_path, expected_frames=len(self.rows),
            expected_size=(NATIVE_WIDTH, NATIVE_HEIGHT))
        jsonl_receipt = {
            "path": str(self.jsonl_path),
            "sha256": self._sha256(self.jsonl_path),
            "rows": len(self.rows),
            "size_bytes": self.jsonl_path.stat().st_size,
        }
        return {
            "raw_video": str(self.raw_path),
            "review_video": None,
            "system_macro_frames": str(self.jsonl_path),
            "raw_video_receipt": raw_receipt,
            "review_video_receipt": None,
            "system_macro_frames_receipt": jsonl_receipt,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._raw_writer is not None:
            self._raw_writer.release()
        self._jsonl.close()

    def __enter__(self) -> "MacroArtifactRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class DiamondPickaxeCraftingMacro:

    def __init__(self, world: Any, *, recorder: MacroArtifactRecorder,
                 settle_frames: int = 2, gui_scale: int = GUI_SCALE):
        self.world = world
        self.recorder = recorder
        self.settle_frames = int(settle_frames)
        if not 1 <= self.settle_frames <= 3:
            raise ValueError("settle_frames must be in [1, 3]")
        self.equipped_proof_frames = EQUIPPED_PROOF_FRAMES
        self.gui_scale = int(gui_scale)
        self.cursor_xy: tuple[int, int] | None = None
        self.geometry: GuiGeometry | None = None
        self.frame_index = len(self.recorder.rows)
        self.macro = ""
        self._operation_index = 0

    @property
    def info(self) -> Mapping[str, Any]:
        info = getattr(self.world, "info", None)
        if not isinstance(info, Mapping):
            raise MacroAuditError("world.info is unavailable")
        return info

    def _pov(self) -> np.ndarray:
        pov = np.asarray(self.info.get("pov"), dtype=np.uint8)
        if pov.shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3):
            raise MacroAuditError(f"GUI macro requires 640x360 POV, got {pov.shape}")
        return pov.copy()

    def _noop(self) -> MutableMapping[str, Any]:
        action = self.world.sim.noop_action()
        if not isinstance(action, MutableMapping):
            action = dict(action)
        else:
            action = deepcopy(action)
        return action

    def _recorded_step(
            self, action: MutableMapping[str, Any], *, event: str,
            target: SlotTarget | None = None, mouse_button: str | None = None,
            operation_id: int | None = None,
            extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        pre = state_snapshot(self.info)
        result = self.world.sim.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise MacroAuditError("sim.step did not return the MineStudio 5-tuple")
        obs, _reward, terminated, truncated, info = result
        self.world.obs, self.world.info = obs, info
        post = state_snapshot(self.info)
        inventory_delta = {
            kind: int(post["inventory_totals"].get(kind, 0))
                  - int(pre["inventory_totals"].get(kind, 0))
            for kind in sorted(
                set(pre["inventory_totals"]) | set(post["inventory_totals"]))
            if int(post["inventory_totals"].get(kind, 0))
               != int(pre["inventory_totals"].get(kind, 0))
        }
        craft_item_delta = {
            kind: int(post["craft_item"].get(kind, 0))
                  - int(pre["craft_item"].get(kind, 0))
            for kind in sorted(set(pre["craft_item"]) | set(post["craft_item"]))
            if int(post["craft_item"].get(kind, 0))
               != int(pre["craft_item"].get(kind, 0))
        }
        row: dict[str, Any] = {
            "contract": FRAME_CONTRACT,
            "frame_index": self.frame_index,
            "macro": self.macro,
            "source": "system_real_minecraft_gui_input",
            "event": event,
            "operation_id": operation_id,
            "action": {
                key: json_safe(value) for key, value in action.items()
                if key == "camera" or _number(value) != 0
            },
            "cursor_xy": (None if self.cursor_xy is None
                          else list(self.cursor_xy)),
            "target": None if target is None else target.as_dict(),
            "mouse_button": mouse_button,
            "pre": pre,
            "post": post,
            "inventory_delta": inventory_delta,
            "craft_item_delta": craft_item_delta,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "telemetry_limitation": AUDIT_LIMITATION,
        }
        if extra:
            row.update(json_safe(dict(extra)))
        self.recorder.capture(self._pov(), row)
        self.frame_index += 1
        if terminated or truncated:
            raise MacroAuditError(
                f"sim terminated during system macro event {event!r}")
        return row

    def _settle(self, *, after: str, operation_id: int | None = None,
                count: int | None = None) -> None:
        total = self.settle_frames if count is None else int(count)
        for index in range(total):
            self._recorded_step(
                self._noop(), event="settle", operation_id=operation_id,
                extra={"after": after, "settle_index": index + 1,
                       "settle_count": total})

    def _button(self, name: str, *, event: str) -> None:
        self._operation_index += 1
        action = self._noop()
        action[name] = 1
        self._recorded_step(
            action, event=event, operation_id=self._operation_index,
            extra={"button_action": name})
        self._settle(after=event, operation_id=self._operation_index)

    def _open_player_inventory(self) -> None:
        if bool(self.info.get("is_gui_open", False)):
            raise MacroAuditError("Macro A requires a closed GUI precondition")
        self._button("inventory", event="open_player_inventory")
        if not bool(self.info.get("is_gui_open", False)):
            raise MacroAuditError("inventory hotkey did not open the player GUI")
        self._bind_geometry("player")

    def _bind_geometry(self, kind: str) -> None:
        if not bool(self.info.get("is_gui_open", False)):
            raise MacroAuditError(f"cannot bind {kind} geometry while GUI is closed")
        self.geometry = GuiGeometry.native(
            kind, pov_shape=self._pov().shape, gui_scale=self.gui_scale)
        self.cursor_xy = self.geometry.cursor_open_xy
        self._recorded_step(
            self._noop(), event="runtime_geometry_asserted",
            extra={"geometry": self.geometry.manifest(),
                   "cursor_position_source": "minecraft_gui_open_center"})
        self._cursor_motion_probe()

    def _cursor_motion_probe(self) -> None:
        if self.cursor_xy is None:
            raise MacroAuditError("cursor origin is unknown")
        start = self.cursor_xy
        probe = (start[0] + 12, start[1])
        attempts = []
        passed = False
        for attempt in range(1, 4):
            before = self._pov()
            self._move_cursor_to_xy(probe, event="cursor_motion_probe_out")
            outward = self._pov()
            self._move_cursor_to_xy(start, event="cursor_motion_probe_return")
            returned = self._pov()
            x0, x1 = start[0] - 4, probe[0] + CURSOR_SIZE + 4
            y0, y1 = start[1] - 4, start[1] + CURSOR_SIZE + 4
            changed_out = int(np.count_nonzero(
                np.max(cv2.absdiff(before[y0:y1, x0:x1],
                                  outward[y0:y1, x0:x1]), axis=2) >= 24))
            changed_return = int(np.count_nonzero(
                np.max(cv2.absdiff(outward[y0:y1, x0:x1],
                                  returned[y0:y1, x0:x1]), axis=2) >= 24))
            attempts.append({
                "attempt": attempt,
                "changed_pixels_out": changed_out,
                "changed_pixels_return": changed_return,
            })
            if changed_out >= 4 and changed_return >= 4:
                passed = True
                break
            recovery_dx = 0
            if changed_out < 4 <= changed_return:
                recovery_dx = probe[0] - start[0]
            elif changed_return < 4 <= changed_out:
                recovery_dx = start[0] - probe[0]
            if recovery_dx:
                action = self._noop()
                action["camera"] = np.array(
                    [0.0, recovery_dx / MOUSE_PIXELS_PER_CAMERA_DEGREE],
                    dtype=np.float32)
                self._operation_index += 1
                self._recorded_step(
                    action, event="cursor_motion_probe_recovery",
                    operation_id=self._operation_index,
                    extra={"physical_recovery_delta_pixels": [recovery_dx, 0],
                           "logical_cursor_xy_unchanged": list(start)})
                self._settle(
                    after="cursor_motion_probe_recovery",
                    operation_id=self._operation_index)
                attempts[-1]["physical_recovery_delta_pixels"] = recovery_dx
        if not passed:
            raise MacroAuditError(
                "camera action did not visibly move and return the GUI cursor: "
                f"attempts={attempts}")
        self._recorded_step(
            self._noop(), event="cursor_motion_probe_passed",
            extra={"start_xy": list(start), "probe_xy": list(probe),
                   "changed_pixels_out": changed_out,
                   "changed_pixels_return": changed_return,
                   "probe_attempts": attempts})

    def _move_cursor_to_xy(self, xy: Sequence[int], *, event: str) -> None:
        if self.cursor_xy is None:
            raise MacroAuditError("cursor position is unknown")
        target = (int(xy[0]), int(xy[1]))
        if not (0 <= target[0] < NATIVE_WIDTH
                and 0 <= target[1] < NATIVE_HEIGHT):
            raise MacroAuditError(f"cursor target outside screen: {target}")
        start = self.cursor_xy
        dx, dy = target[0] - start[0], target[1] - start[1]
        action = self._noop()
        action["camera"] = np.array(
            [dy / MOUSE_PIXELS_PER_CAMERA_DEGREE,
             dx / MOUSE_PIXELS_PER_CAMERA_DEGREE], dtype=np.float32)
        self.cursor_xy = target
        self._operation_index += 1
        self._recorded_step(
            action, event=event, operation_id=self._operation_index,
            extra={"cursor_start_xy": list(start),
                   "cursor_target_xy": list(target),
                   "cursor_delta_pixels": [dx, dy],
                   "mouse_pixels_per_camera_degree":
                       MOUSE_PIXELS_PER_CAMERA_DEGREE})
        self._settle(after=event, operation_id=self._operation_index)

    def move_cursor(self, target: SlotTarget) -> None:
        if self.geometry is None or target.gui_kind != self.geometry.kind:
            raise MacroAuditError(
                f"slot {target.name} is not in the active GUI geometry")
        self._move_cursor_to_xy(
            (target.x, target.y), event=f"move_to:{target.name}")

    def click(self, target: SlotTarget, *, button: str) -> dict[str, Any]:
        if button not in ("left", "right"):
            raise ValueError(f"unknown mouse button {button!r}")
        self.move_cursor(target)
        self._operation_index += 1
        action = self._noop()
        action["attack" if button == "left" else "use"] = 1
        pre_target = (
            None if target.player_slot_id is None else
            player_slots(self.info)[target.player_slot_id])
        row = self._recorded_step(
            action, event=f"click:{target.name}", target=target,
            mouse_button=button, operation_id=self._operation_index,
            extra={
                "pre_target_player_slot": pre_target,
                "gui_slot_telemetry_available":
                    target.player_slot_id is not None,
            })
        self._settle(
            after=f"click:{target.name}",
            operation_id=self._operation_index)
        row["post_target_player_slot_after_settle"] = (
            None if target.player_slot_id is None else
            player_slots(self.info)[target.player_slot_id])
        return row

    def _require_totals(self, expected: Mapping[str, int], *, phase: str) -> None:
        totals = inventory_totals(self.info)
        mismatches = {
            namespaceless(kind): {
                "expected": int(quantity),
                "actual": int(totals.get(namespaceless(kind), 0)),
            }
            for kind, quantity in expected.items()
            if int(totals.get(namespaceless(kind), 0)) != int(quantity)
        }
        if mismatches:
            raise MacroAuditError(f"{phase} inventory mismatch: {mismatches}")

    def _require_craft_delta(self, baseline: Mapping[str, int], kind: str,
                             *, phase: str) -> int:
        delta = counter_delta(
            baseline, normalized_counter(self.info, "craft_item"), kind)
        if delta <= 0:
            raise MacroAuditError(
                f"{phase} has no positive craft_item:{namespaceless(kind)} delta")
        return delta

    def _cursor_empty_probe(self, *, slot_id: int) -> dict[str, Any]:
        assert self.geometry is not None
        slot_id = int(slot_id)
        before_slots = player_slots(self.info)
        before_totals = inventory_totals(self.info)
        if before_slots[slot_id]["type"] != "none":
            raise MacroAuditError(
                f"cursor-empty probe slot {slot_id} is not empty")
        before_path = self.recorder.save_evidence(
            f"{self.macro}_cursor_empty_probe_before.png", self._pov())
        self.click(self.geometry.player_slot(slot_id), button="left")
        after_slots = player_slots(self.info)
        after_totals = inventory_totals(self.info)
        after_path = self.recorder.save_evidence(
            f"{self.macro}_cursor_empty_probe_after.png", self._pov())
        passed = bool(
            after_slots[slot_id]["type"] == "none"
            and after_slots[slot_id]["quantity"] == 0
            and after_totals == before_totals)
        audit = {
            "contract": "known_empty_player_slot_cursor_probe/v1",
            "slot_id": slot_id,
            "slot_before": before_slots[slot_id],
            "slot_after": after_slots[slot_id],
            "inventory_totals_before": before_totals,
            "inventory_totals_after": after_totals,
            "visual_evidence_before": before_path,
            "visual_evidence_after": after_path,
            "telemetry_limitation": AUDIT_LIMITATION,
            "passed": passed,
        }
        self._recorded_step(
            self._noop(), event="cursor_empty_probe_result", extra=audit)
        if not passed:
            raise MacroAuditError(f"cursor-empty active probe failed: {audit}")
        return audit

    def _close_gui(self) -> None:
        if not bool(self.info.get("is_gui_open", False)):
            raise MacroAuditError("cannot close a GUI that is not open")
        self._button("inventory", event="close_gui")
        if bool(self.info.get("is_gui_open", False)):
            raise MacroAuditError("inventory hotkey did not close the GUI")

    def _select_hotbar(self, one_based: int, expected_item: str) -> None:
        one_based = int(one_based)
        if not 1 <= one_based <= 9:
            raise ValueError("hotbar key must be in [1, 9]")
        self._button(f"hotbar.{one_based}", event=f"select_hotbar_{one_based}")
        actual = held_item(self.info)
        if actual != namespaceless(expected_item):
            raise MacroAuditError(
                f"held item is {actual!r}, expected {namespaceless(expected_item)!r}")

    def run_macro_a(self) -> dict[str, Any]:
        self.macro = "macro_a_sticks"
        initial = state_snapshot(self.info)
        pre_open_log_count = int(
            inventory_totals(self.info).get("oak_log", 0))
        if pre_open_log_count < 1:
            raise MacroAuditError(
                "Macro A precondition requires at least one oak_log")
        self._require_totals(
            {"oak_log": pre_open_log_count, "oak_planks": 0, "stick": 0,
             "iron_pickaxe": 1}, phase="Macro A precondition")
        craft_before = normalized_counter(self.info, "craft_item")

        self._open_player_inventory()
        assert self.geometry is not None
        post_open = state_snapshot(self.info)
        initial_log_count = int(
            inventory_totals(self.info).get("oak_log", 0))
        if initial_log_count < pre_open_log_count:
            raise MacroAuditError(
                "oak_log count decreased while opening player inventory")
        self._require_totals(
            {"oak_log": initial_log_count, "oak_planks": 0, "stick": 0,
             "iron_pickaxe": 1}, phase="Macro A post-open precondition")
        log_slot = find_item_slot(self.info, "oak_log")
        log_stack_quantity = int(player_slots(self.info)[log_slot]["quantity"])
        planks_slot = find_empty_slot(
            self.info, main_inventory_only=True)
        sticks_slot = find_empty_slot(
            self.info, main_inventory_only=True, exclude=(planks_slot,))
        probe_slot = find_empty_slot(
            self.info, main_inventory_only=True,
            exclude=(planks_slot, sticks_slot))

        if log_stack_quantity == 1:
            self.click(self.geometry.player_slot(log_slot), button="left")
            self.click(self.geometry.input_slot(0), button="left")
        elif log_stack_quantity == 2:
            self.click(self.geometry.player_slot(log_slot), button="right")
            self.click(self.geometry.input_slot(0), button="left")
        else:
            self.click(self.geometry.player_slot(log_slot), button="right")
            self.click(self.geometry.input_slot(0), button="right")
            self.click(self.geometry.player_slot(log_slot), button="left")
        remaining_logs = initial_log_count - 1
        self._require_totals(
            {"oak_log": remaining_logs}, phase="one log placed in 2x2 grid")
        self.click(self.geometry.output_slot(), button="left")
        self.click(self.geometry.player_slot(planks_slot), button="left")
        self._require_totals(
            {"oak_log": remaining_logs, "oak_planks": 4},
            phase="planks output deposited")
        planks_craft_delta = self._require_craft_delta(
            craft_before, "oak_planks", phase="planks recipe")

        self.click(self.geometry.player_slot(planks_slot), button="left")
        self.click(self.geometry.input_slot(0), button="right")
        self.click(self.geometry.input_slot(2), button="right")
        self.click(self.geometry.player_slot(planks_slot), button="left")
        self._require_totals(
            {"oak_planks": 2}, phase="two planks placed vertically")
        self.click(self.geometry.output_slot(), button="left")
        self.click(self.geometry.player_slot(sticks_slot), button="left")
        self._require_totals(
            {"oak_log": remaining_logs, "oak_planks": 2, "stick": 4,
             "iron_pickaxe": 1}, phase="sticks output deposited")
        sticks_craft_delta = self._require_craft_delta(
            craft_before, "stick", phase="sticks recipe")

        cursor_probe = self._cursor_empty_probe(slot_id=probe_slot)
        self._move_cursor_to_xy(
            self.geometry.cursor_open_xy,
            event="return_cursor_to_center_before_close")
        self._close_gui()
        self._select_hotbar(1, "iron_pickaxe")
        self._require_totals(
            {"oak_log": remaining_logs, "oak_planks": 2, "stick": 4,
             "iron_pickaxe": 1}, phase="Macro A final")
        final = state_snapshot(self.info)
        result = {
            "contract": CONTRACT,
            "macro": self.macro,
            "success": True,
            "input_route": "MineStudio env inventory/camera/attack/use",
            "inventory_injection_count": 0,
            "direct_inventory_mutation": False,
            "geometry": self.geometry.manifest(),
            "player_slots": {
                "log_source": log_slot,
                "planks_destination": planks_slot,
                "sticks_destination": sticks_slot,
                "cursor_probe": probe_slot,
            },
            "craft_item_delta": {
                "oak_planks": planks_craft_delta,
                "stick": sticks_craft_delta,
            },
            "item_conservation": {
                "oak_log_pre_open": pre_open_log_count,
                "oak_log_arrived_on_gui_open": (
                    initial_log_count - pre_open_log_count),
                "oak_log_initial": initial_log_count,
                "oak_log_consumed": 1,
                "oak_log_remaining": remaining_logs,
                "oak_planks_produced": 4,
                "oak_planks_consumed": 2,
                "oak_planks_remaining": 2,
                "sticks_produced": 4,
                "sticks_remaining": 4,
            },
            "cursor_empty_probe": cursor_probe,
            "initial": initial,
            "post_open": post_open,
            "final": final,
            "gui_closed": not bool(self.info.get("is_gui_open", False)),
            "held_item": held_item(self.info),
            "telemetry_limitation": AUDIT_LIMITATION,
        }
        self._recorded_step(
            self._noop(), event="macro_a_postcondition_passed", extra=result)
        return json_safe(result)

    def run_macro_b(self) -> dict[str, Any]:
        self.macro = "macro_b_diamond_pickaxe"
        if not bool(self.info.get("is_gui_open", False)):
            raise MacroAuditError(
                "Macro B requires the policy-opened workbench GUI")
        initial = state_snapshot(self.info)
        self._require_totals(
            {"diamond": 3, "stick": 4, "oak_planks": 2,
             "diamond_pickaxe": 0}, phase="Macro B precondition")
        craft_before = normalized_counter(self.info, "craft_item")
        diamond_slot = find_item_slot(
            self.info, "diamond", exact_quantity=3)
        stick_slot = find_item_slot(self.info, "stick", exact_quantity=4)
        self._bind_geometry("workbench")
        assert self.geometry is not None

        self.click(self.geometry.player_slot(diamond_slot), button="left")
        for grid_slot in (0, 1, 2):
            self.click(self.geometry.input_slot(grid_slot), button="right")
        self._require_totals(
            {"diamond": 0}, phase="three diamonds placed in recipe")

        self.click(self.geometry.player_slot(stick_slot), button="left")
        for grid_slot in (4, 7):
            self.click(self.geometry.input_slot(grid_slot), button="right")
        self.click(self.geometry.player_slot(stick_slot), button="left")
        self._require_totals(
            {"stick": 2}, phase="two sticks placed in recipe")

        destination = next(
            (slot for slot in range(1, 9)
             if player_slots(self.info)[slot]["type"] == "none"), None)
        if destination is None:
            raise MacroAuditError(
                "no empty non-pickaxe hotbar slot for diamond pickaxe")
        self.click(self.geometry.output_slot(), button="left")
        self.click(self.geometry.player_slot(destination), button="left")
        self._require_totals(
            {"diamond": 0, "stick": 2, "oak_planks": 2,
             "diamond_pickaxe": 1}, phase="pickaxe output deposited")
        pickaxe_craft_delta = self._require_craft_delta(
            craft_before, "diamond_pickaxe", phase="diamond-pickaxe recipe")

        probe_slot = find_empty_slot(
            self.info, main_inventory_only=True, exclude=(stick_slot,))
        cursor_probe = self._cursor_empty_probe(slot_id=probe_slot)
        self._close_gui()
        self._select_hotbar(destination + 1, "diamond_pickaxe")

        proof_start = self.frame_index
        for index in range(self.equipped_proof_frames):
            self._recorded_step(
                self._noop(), event="equipped_noop_proof_tail",
                extra={"proof_tail_index": index + 1,
                       "proof_tail_count": self.equipped_proof_frames,
                       "camera_forced": False})
        proof_frames = self.frame_index - proof_start
        if proof_frames != self.equipped_proof_frames:
            raise MacroAuditError(
                f"pickaxe proof tail must contain exactly "
                f"{self.equipped_proof_frames} noops, got {proof_frames}")

        self._require_totals(
            {"diamond": 0, "stick": 2, "oak_planks": 2,
             "diamond_pickaxe": 1}, phase="Macro B final")
        if held_item(self.info) != "diamond_pickaxe":
            raise MacroAuditError("diamond pickaxe did not remain equipped")
        final = state_snapshot(self.info)
        result = {
            "contract": CONTRACT,
            "macro": self.macro,
            "success": True,
            "input_route": (
                "MineStudio env camera/attack/use/inventory/"
                f"hotbar.{destination + 1}"),
            "inventory_injection_count": 0,
            "direct_inventory_mutation": False,
            "geometry": self.geometry.manifest(),
            "player_slots": {
                "diamond_source": diamond_slot,
                "stick_source": stick_slot,
                "pickaxe_destination_hotbar_index": destination,
                "cursor_probe": probe_slot,
            },
            "craft_item_delta": {
                "diamond_pickaxe": pickaxe_craft_delta},
            "item_conservation": {
                "diamonds_consumed": 3,
                "sticks_consumed": 2,
                "sticks_remaining": 2,
                "oak_planks_remaining": 2,
                "diamond_pickaxes_produced": 1,
            },
            "cursor_empty_probe": cursor_probe,
            "initial": initial,
            "final": final,
            "gui_closed": not bool(self.info.get("is_gui_open", False)),
            "held_item": held_item(self.info),
            "equipped_noop_proof_tail_frames": proof_frames,
            "camera_forced_in_proof_tail": False,
            "telemetry_limitation": AUDIT_LIMITATION,
        }
        return json_safe(result)

