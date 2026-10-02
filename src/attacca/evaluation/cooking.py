"""Furnace cooking macro for the CCFW chain using real input actions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

from attacca.evaluation.crafting import DiamondPickaxeCraftingMacro
from attacca.evaluation.crafting import GuiGeometry
from attacca.evaluation.crafting import MacroArtifactRecorder
from attacca.evaluation.crafting import MacroAuditError
from attacca.evaluation.crafting import SlotTarget
from attacca.evaluation.crafting import find_empty_slot
from attacca.evaluation.crafting import find_item_slot
from attacca.evaluation.crafting import inventory_totals
from attacca.evaluation.crafting import state_snapshot


MACRO_FRAME_CONTRACT = "xbench_real_world_input_macro_frame/v1"
COOK_CONTRACT = "xbench_real_gui_furnace_beef_cook_macro/v1"


class RealWorldInputMacro:
    def __init__(self, world: Any, recorder: MacroArtifactRecorder,
                 *, name: str):
        self.world = world
        self.recorder = recorder
        self.name = str(name)
        self.frame_index = len(recorder.rows)

    def _noop(self) -> MutableMapping[str, Any]:
        action = self.world.sim.noop_action()
        return deepcopy(action) if isinstance(action, MutableMapping) else dict(action)

    def step(self, action: MutableMapping[str, Any], *, event: str,
             extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        pre_position = [float(v) for v in self.world.get_pos()]
        pre = state_snapshot(self.world.info)
        obs, _reward, terminated, truncated, info = self.world.sim.step(action)
        self.world.obs, self.world.info = obs, info
        row = {
            "contract": MACRO_FRAME_CONTRACT,
            "frame_index": self.frame_index,
            "macro": self.name,
            "source": "system_real_minecraft_input",
            "event": event,
            "action": {key: np.asarray(value).tolist()
                       for key, value in action.items()
                       if key == "camera" or bool(np.asarray(value).reshape(-1)[0])},
            "pre_action_position": pre_position,
            "post_action_position": [float(v) for v in self.world.get_pos()],
            "pre": pre,
            "post": state_snapshot(self.world.info),
            "commands_issued": 0,
            "teleports": 0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        if extra:
            row.update(dict(extra))
        rgb = np.ascontiguousarray(self.world.info["pov"], dtype=np.uint8)
        self.recorder.capture(rgb, row)
        self.frame_index += 1
        if terminated or truncated:
            raise MacroAuditError(f"sim ended during {event}")
        return row


class FurnaceCookingMacro(DiamondPickaxeCraftingMacro):

    def _bind_furnace_geometry(self) -> None:
        if not bool(self.info.get("is_gui_open", False)):
            raise MacroAuditError("furnace macro requires an open GUI")
        self.geometry = GuiGeometry(
            "furnace", 640, 360, self.gui_scale, 232, 97)
        self.cursor_xy = self.geometry.cursor_open_xy
        self._recorded_step(
            self._noop(), event="runtime_furnace_geometry_asserted",
            extra={"geometry": self.geometry.manifest(),
                   "slot_coordinates_source":
                       "Minecraft 1.16.5 AbstractFurnaceContainer centres"})
        self._cursor_motion_probe()

    def _furnace_target(self, name: str, rel_x: int, rel_y: int) -> SlotTarget:
        assert self.geometry is not None
        return SlotTarget(name, "furnace",
                          self.geometry.origin_x + rel_x,
                          self.geometry.origin_y + rel_y)

    def run(self, *, cook_wait_steps: int = 220) -> dict[str, Any]:
        self.macro = "system_real_gui_furnace_beef_cook"
        initial = state_snapshot(self.info)
        totals0 = inventory_totals(self.info)
        raw0 = int(totals0.get("beef", 0))
        coal0 = int(totals0.get("coal", 0))
        cooked0 = int(totals0.get("cooked_beef", 0))
        if raw0 < 1 or coal0 < 1 or cooked0 != 0:
            raise MacroAuditError(
                f"furnace precondition mismatch: beef={raw0}, coal={coal0}, "
                f"cooked_beef={cooked0}")
        raw_slot = find_item_slot(self.info, "beef")
        coal_slot = find_item_slot(self.info, "coal")
        destination = find_empty_slot(
            self.info, main_inventory_only=True, exclude=(raw_slot, coal_slot))
        self._bind_furnace_geometry()
        assert self.geometry is not None
        input_slot = self._furnace_target("furnace_raw_beef_input", 64, 25)
        fuel_slot = self._furnace_target("furnace_coal_fuel", 64, 61)
        output_slot = self._furnace_target("furnace_cooked_beef_output", 124, 43)

        self.click(self.geometry.player_slot(raw_slot), button="left")
        self.click(input_slot, button="right")
        self.click(self.geometry.player_slot(raw_slot), button="left")
        if int(inventory_totals(self.info).get("beef", 0)) != raw0 - 1:
            raise MacroAuditError("exactly one raw beef did not enter furnace input")
        self.click(self.geometry.player_slot(coal_slot), button="left")
        self.click(fuel_slot, button="right")
        self.click(self.geometry.player_slot(coal_slot), button="left")
        if int(inventory_totals(self.info).get("coal", 0)) != coal0 - 1:
            raise MacroAuditError("exactly one coal did not enter furnace fuel")

        for index in range(int(cook_wait_steps)):
            self._recorded_step(
                self._noop(), event="vanilla_furnace_cook_wait",
                extra={"wait_index": index + 1,
                       "wait_count": int(cook_wait_steps)})
        self.click(output_slot, button="left")
        self.click(self.geometry.player_slot(destination), button="left")
        cooked1 = int(inventory_totals(self.info).get("cooked_beef", 0))
        if cooked1 != cooked0 + 1:
            raise MacroAuditError(
                f"cooked beef output missing after vanilla wait: {cooked1}")
        cursor_probe_slot = find_empty_slot(
            self.info, main_inventory_only=True, exclude=(destination,))
        cursor_probe = self._cursor_empty_probe(slot_id=cursor_probe_slot)
        final_open = state_snapshot(self.info)
        self._close_gui()
        final = state_snapshot(self.info)
        return {
            "contract": COOK_CONTRACT,
            "success": True,
            "input_route": "MineStudio camera/attack/use/inventory/noop",
            "commands_issued": 0,
            "teleports": 0,
            "item_injection_count": 0,
            "direct_inventory_mutation": False,
            "vanilla_cook_wait_steps": int(cook_wait_steps),
            "inventory_conservation": {
                "raw_beef_initial": raw0, "raw_beef_loaded": 1,
                "coal_initial": coal0, "coal_loaded": 1,
                "cooked_beef_initial": cooked0,
                "cooked_beef_collected": cooked1 - cooked0,
            },
            "player_slots": {
                "raw_beef_source": raw_slot, "coal_source": coal_slot,
                "cooked_beef_destination": destination,
            },
            "geometry": self.geometry.manifest(),
            "cursor_empty_probe": cursor_probe,
            "initial": initial,
            "final_gui_open": final_open,
            "final": final,
            "gui_closed": not bool(self.info.get("is_gui_open", False)),
        }


def cook_beef_in_furnace(world: Any, out: Path, *,
                         cook_wait_steps: int = 220) -> dict[str, Any]:
    out = Path(out).expanduser().resolve()
    with MacroArtifactRecorder(out) as recorder:
        macro = FurnaceCookingMacro(world, recorder=recorder, settle_frames=1)
        result = macro.run(cook_wait_steps=int(cook_wait_steps))
    return {**result, **recorder.artifact_manifest()}
