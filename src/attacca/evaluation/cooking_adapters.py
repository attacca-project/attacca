from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from attacca.evaluation import dpx as DPX
from attacca.evaluation import chain_engine as ENGINE
from attacca.evaluation.chain_adapters import GuiOpenStageAdapter
from attacca.worlds.coal_cow_furnace_chain_scene import COW_TAG


CAUSAL_FIFO_STEPS = DPX.CAUSAL_FIFO_STEPS
HUNT_REACH = 4.5


class CowHuntStageAdapter(ENGINE.Adapter):

    def __init__(self, world, stager, scene, task: str, *,
                 interaction_id: int = 0):
        super().__init__(world, stager, scene, task)
        if int(interaction_id) != 0:
            raise ValueError("cow hunt requires interaction_id=0")
        from attacca.worlds.entity_id_mask import ENTITY_ID_SEMANTIC_CODES
        from attacca.worlds.human_viewmodel import RENDERER_SCENE_ALIVE_CLASSES_FIELD
        self._alive_field = RENDERER_SCENE_ALIVE_CLASSES_FIELD
        self._cow_code = int(ENTITY_ID_SEMANTIC_CODES["cow"])
        self._cow_bit = 1 << (self._cow_code - 1)
        self.initial_alive_mask = int(world.info.get(self._alive_field, -1))
        if self.initial_alive_mask != self._cow_bit:
            raise RuntimeError(
                f"cow stage requires exactly one live cow class bit: "
                f"{self.initial_alive_mask:#x}/{self._cow_bit:#x}")
        self.initial_event_info = {"kill_entity": copy.deepcopy(
            (world.info or {}).get("kill_entity", {}) or {})}
        self.initial_beef = ENGINE.inventory_count(world, "beef")
        self.step_events: dict[int, dict[str, Any]] = {}
        self.death_event: dict[str, Any] | None = None

    @property
    def interaction_id(self) -> int:
        return 0

    @property
    def quota(self) -> int:
        return 1

    @property
    def progress(self) -> int:
        return int(self.success_step is not None)

    @property
    def kill_delta(self) -> int:
        return DPX._stat_delta(
            self.world.info, self.initial_event_info, "kill_entity", "cow")

    def pre_context(self) -> dict[str, Any]:
        return {"alive_mask_before": int(
                    self.world.info.get(self._alive_field, -1)),
                "kill_delta_before": self.kill_delta,
                "beef_before": ENGINE.inventory_count(self.world, "beef")}

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        del use
        after_mask = int(self.world.info.get(self._alive_field, -1))
        before_mask = int(context["alive_mask_before"])
        died_now = bool(before_mask & self._cow_bit and not after_mask & self._cow_bit)
        if died_now:
            self.death_event = {
                "step": int(step), "before_mask": before_mask,
                "after_mask": after_mask, "cow_bit": self._cow_bit,
            }
        success_now = bool(
            self.success_step is None and self.kill_delta > 0)
        if success_now:
            self.success_step = int(step)
            self.success_evidence = {
                "step": int(step), "unique_tag": COW_TAG,
                "kill_entity_cow_delta": self.kill_delta,
                "renderer_alive_transition": self.death_event,
                "score_contract": (
                    "unique_staged_cow_kill_entity_delta/v1"),
                "natural_beef_at_score": max(
                    0, ENGINE.inventory_count(self.world, "beef") - self.initial_beef),
                "success_requires_beef_pickup": False,
            }
        beef = ENGINE.inventory_count(self.world, "beef")
        self.step_events[int(step)] = {
            "cow_died_this_step": died_now,
            "alive_mask_after": after_mask,
            "kill_entity_cow_delta": self.kill_delta,
            "beef_after": beef,
            "natural_beef_delta": max(0, beef - self.initial_beef),
        }
        return {"success_now": success_now, "control_complete_now": False,
                "note": (f"HUNT cow kill={self.kill_delta} alive={int(bool(after_mask & self._cow_bit))}")
                        if attack or success_now else ""}

    def finish_audit(self) -> bool:
        after_mask = int(self.world.info.get(self._alive_field, -1))
        passed = bool(
            self.success_step is not None and self.kill_delta > 0)
        beef = ENGINE.inventory_count(self.world, "beef")
        self.terminal_audit = {
            "performed": self.success_step is not None,
            "passed": passed,
            "unique_tag": COW_TAG,
            "initial_alive_mask": self.initial_alive_mask,
            "final_alive_mask": after_mask,
            "kill_entity_cow_delta": self.kill_delta,
            "score_contract": "unique_staged_cow_kill_entity_delta/v1",
            "policy_attack_fifo_required": False,
            "exact_cow_surface_required": False,
            "renderer_alive_transition_required": False,
            "reach_required": False,
            "natural_beef_delta_at_stage_end": max(0, beef - self.initial_beef),
            "beef_pickup_scored_separately": True,
            "success_evidence": self.success_evidence,
        }
        return passed


def live_target_visibility(world, *, cells: Sequence[Sequence[int]] = (),
                           entity_kinds: Sequence[str] = ()) -> dict[str, Any]:
    if cells:
        position = tuple(float(value) for value in world.get_pos())
        pixels = int(ENGINE.block_union(
            world, tuple(tuple(map(int, cell)) for cell in cells),
            render_position=position).sum())
    elif entity_kinds:
        pixels = int(ENGINE.entity_union(world, tuple(entity_kinds)).sum())
    else:
        pixels = 0
    return {"target_visible_pixels_live": pixels}


class AuditedGuiOpenStageAdapter(GuiOpenStageAdapter):

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        event = super().after_step(
            step=step, use=use, attack=attack, context=context)
        self.step_events[int(step)].update(live_target_visibility(
            self.world, cells=(self.target_cell,)))
        return event
