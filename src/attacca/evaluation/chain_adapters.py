from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from attacca.evaluation import dpx as DPX
from attacca.evaluation import chain_engine as ENGINE


GUI_INTERACTION_STATS = {
    "furnace": "interact_with_furnace",
}
CAUSAL_FIFO_STEPS = DPX.CAUSAL_FIFO_STEPS
AIR_KINDS = frozenset(("air", "cave_air", "void_air"))


class MineStageAdapter(ENGINE.Adapter):
    def __init__(self, world, stager, scene, task: str,
                 target_cells: Sequence[Sequence[int]], *, target_kind: str,
                 pickup_kind: str, quota: int, interaction_id: int = 2):
        super().__init__(world, stager, scene, task)
        self.target_cells = tuple(tuple(map(int, cell)) for cell in target_cells)
        self.target_kind = DPX._kind(target_kind)
        self.pickup_kind = DPX._kind(pickup_kind)
        self._quota = int(quota)
        self._interaction_id = int(interaction_id)
        if (not self.target_kind or self.target_kind in AIR_KINDS
                or self._quota <= 0 or self._quota > len(self.target_cells)
                or len(set(self.target_cells)) != len(self.target_cells)):
            raise ValueError("mine stage needs unique target cells and a feasible quota")
        if self._interaction_id != 2:
            raise ValueError("mine stages require interaction_id=2")
        self.policy_attribution_mode = (
            "exact_crosshair_attack_cell_transition_fifo_and_mine_stat")
        self.latest_query = dict.fromkeys(self.target_cells, self.target_kind)
        self.initial_event_info = {"mine_block": copy.deepcopy(
            (world.info or {}).get("mine_block", {}) or {})}
        self.initial_pickup_inventory = ENGINE.inventory_count(world, self.pickup_kind)
        self.pending_attacks: list[dict[str, Any]] = []
        self.removed_cells: set[tuple[int, int, int]] = set()
        self.attributed_cells: set[tuple[int, int, int]] = set()
        self.removal_events: list[dict[str, Any]] = []
        self.step_events: dict[int, dict[str, Any]] = {}
        self.item_injection_count = 0

    @property
    def interaction_id(self) -> int:
        return self._interaction_id

    @property
    def quota(self) -> int:
        return self._quota

    @property
    def tracking_quota(self) -> int:
        return len(self.target_cells)

    @property
    def progress(self) -> int:
        return min(self.quota, len(self.attributed_cells), self.mine_event_delta)

    @property
    def mine_event_delta(self) -> int:
        return DPX._stat_delta(
            self.world.info, self.initial_event_info, "mine_block", self.target_kind)

    def pre_context(self) -> dict[str, Any]:
        hit = DPX._crosshair_target_cell(
            self.world,
            tuple(cell for cell in self.target_cells if cell not in self.removed_cells))
        return {
            "crosshair_target_cell": None if hit is None else list(hit),
            "policy_attribution_mode": self.policy_attribution_mode,
            "mine_event_delta_before": self.mine_event_delta,
            "pickup_count_before": ENGINE.inventory_count(self.world, self.pickup_kind),
            "inventory_before": DPX._inventory_counts(self.world),
            "held_before": ENGINE.held_item(self.world),
        }

    def prepare_action(self, action: dict[str, Any]) -> None:
        action["voxels"] = ENGINE.query_box(self.world, self.target_cells)

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        del use
        crosshair = context.get("crosshair_target_cell")
        if attack and crosshair is not None:
            self.pending_attacks.append({
                "step": int(step),
                "cell": None if crosshair is None else tuple(map(int, crosshair)),
            })
        self.pending_attacks = [row for row in self.pending_attacks
                                if 0 <= int(step) - row["step"] <= CAUSAL_FIFO_STEPS]
        previous = self.latest_query
        current = ENGINE.current_query(self.world)
        newly_removed = []
        for cell in self.target_cells:
            if (cell in self.removed_cells or not current
                    or DPX._kind(previous.get(cell, "")) != self.target_kind
                    or DPX._kind(current.get(cell, "air")) not in AIR_KINDS):
                continue
            self.removed_cells.add(cell)
            newly_removed.append(list(cell))
            causal = next((row for row in reversed(self.pending_attacks)
                           if row["cell"] == cell), None)
            if causal is not None:
                self.attributed_cells.add(cell)
            self.removal_events.append({
                "removal_step": int(step), "cell": list(cell),
                "policy_attack_attributed": causal is not None,
                "attack_step": None if causal is None else causal["step"],
                "mine_block_stat_delta": self.mine_event_delta,
            })
        if current:
            self.latest_query = {cell: current.get(cell, "air") for cell in self.target_cells}
        success_now = self.success_step is None and self.progress >= self.quota
        if success_now:
            self.success_step = int(step)
            self.success_evidence = {
                "step": int(step), "target_kind": self.target_kind,
                "quota": self.quota, "mine_block_stat_delta": self.mine_event_delta,
                "policy_attack_attributed_cells": [list(cell) for cell in sorted(self.attributed_cells)],
                "policy_attribution_mode": self.policy_attribution_mode,
                "pickup_required_for_score": False,
            }
        event = {
            "inventory_after": DPX._inventory_counts(self.world),
            "pickup_count_after": ENGINE.inventory_count(self.world, self.pickup_kind),
            "mine_block_stat_delta_after": self.mine_event_delta,
            "newly_removed_cells": newly_removed,
            "wrong_block_deltas": DPX._wrong_mine_deltas(
                self.world.info, self.initial_event_info, self.target_kind),
        }
        self.step_events[int(step)] = event
        return {"success_now": bool(success_now), "control_complete_now": False,
                "note": (f"MINE {self.target_kind} {self.progress}/{self.quota} "
                         f"stat={self.mine_event_delta}") if attack or newly_removed else ""}

    def finish_audit(self) -> bool:
        passed = self.success_step is not None and self.progress >= self.quota
        inventory = ENGINE.inventory_count(self.world, self.pickup_kind)
        wrong = DPX._wrong_mine_deltas(
            self.world.info, self.initial_event_info, self.target_kind)
        self.terminal_audit = {
            "performed": self.success_step is not None, "passed": bool(passed),
            "target_kind": self.target_kind, "quota": self.quota,
            "mine_block_stat_delta": self.mine_event_delta,
            "policy_attack_attributed_cells": [list(cell) for cell in sorted(self.attributed_cells)],
            "removed_cells": [list(cell) for cell in sorted(self.removed_cells)],
            "removal_events": self.removal_events,
            "policy_attribution_mode": self.policy_attribution_mode,
            "pre_action_dense_target_mask_materialized": True,
            "success_requires_exact_initial_cell_removal": True,
            "success_requires_policy_attack_causal_fifo": True,
            "success_requires_mine_block_stat_delta": True,
            "pickup_item": self.pickup_kind, "pickup_inventory_count": inventory,
            "natural_pickup_delta": max(0, inventory - self.initial_pickup_inventory),
            "pickup_required_for_score": False, "item_injection_count": 0,
            "wrong_block_deltas": wrong, "wrong_target_blocks_removed": sum(wrong.values()),
            "clean_success": bool(passed and not wrong),
            "success_evidence": self.success_evidence,
        }
        return bool(passed)


class GuiOpenStageAdapter(ENGINE.Adapter):
    def __init__(self, world, stager, scene, task: str,
                 target_cell: Sequence[int], *, target_kind: str,
                 interaction_id: int = 3):
        super().__init__(world, stager, scene, task)
        self.target_kind = DPX._kind(target_kind)
        if self.target_kind not in GUI_INTERACTION_STATS:
            raise ValueError(
                f"{self.target_kind} has no verified GUI-open stat contract")
        if int(interaction_id) != 3:
            raise ValueError("GUI-open stages require interaction_id=3")
        self.target_cell = tuple(map(int, target_cell))
        self.interaction_stat = GUI_INTERACTION_STATS[self.target_kind]
        self.policy_attribution_mode = "exact_target_use_fresh_stat_gui_open_fifo"
        self.latest_query = {self.target_cell: self.target_kind}
        self.initial_event_info = {"custom": copy.deepcopy(
            (world.info or {}).get("custom", {}) or {})}
        self.pending_uses: list[dict[str, Any]] = []
        self.step_events: dict[int, dict[str, Any]] = {}

    @property
    def interaction_id(self) -> int:
        return 3

    @property
    def quota(self) -> int:
        return 1

    @property
    def progress(self) -> int:
        return int(self.success_step is not None)

    @property
    def interact_stat_delta(self) -> int:
        return DPX._stat_delta(
            self.world.info, self.initial_event_info, "custom", self.interaction_stat)

    def pre_context(self) -> dict[str, Any]:
        hit = DPX._crosshair_target_cell(self.world, (self.target_cell,))
        return {
            "crosshair_exact_target": hit == self.target_cell,
            "crosshair_target_cell": None if hit is None else list(hit),
            "target_before": DPX._kind(self.latest_query.get(self.target_cell, "")),
            "gui_open_before": bool(self.world.info.get("is_gui_open", False)),
            "interact_stat_delta_before": self.interact_stat_delta,
            "inventory_before": DPX._inventory_counts(self.world),
            "held_before": ENGINE.held_item(self.world),
            "policy_attribution_mode": self.policy_attribution_mode,
        }

    def prepare_action(self, action: dict[str, Any]) -> None:
        action["voxels"] = ENGINE.query_box(self.world, (self.target_cell,))

    def after_step(self, *, step: int, use: bool, attack: bool,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        del attack
        if (use and context.get("gui_open_before") is False
                and context.get("target_before") == self.target_kind):
            self.pending_uses.append({
                "step": int(step), "cell": list(self.target_cell),
                "stat_before": int(context["interact_stat_delta_before"]),
                "gui_open_step": None,
                "exact_target_at_use": bool(
                    context.get("crosshair_exact_target") is True),
                "exact_target_at_confirmation": False,
            })
        self.pending_uses = [row for row in self.pending_uses
                             if 0 <= int(step) - row["step"] <= CAUSAL_FIFO_STEPS]
        self.latest_query = {**self.latest_query, **ENGINE.current_query(self.world)}
        gui = bool(self.world.info.get("is_gui_open", False))
        stat = self.interact_stat_delta
        if not gui:
            self.pending_uses = [row for row in self.pending_uses
                                 if row["gui_open_step"] is None]
        elif context.get("gui_open_before") is False:
            for row in self.pending_uses:
                if row["gui_open_step"] is None:
                    row["gui_open_step"] = int(step)
                    row["exact_target_at_confirmation"] = bool(
                        context.get("crosshair_exact_target") is True)
        causal = next((row for row in reversed(self.pending_uses)
                       if row["gui_open_step"] is not None
                       and stat > row["stat_before"]
                       and (row["exact_target_at_use"]
                            or row["exact_target_at_confirmation"])), None)
        success_now = bool(
            self.success_step is None and gui and causal is not None
            and DPX._kind(self.latest_query.get(self.target_cell, "")) == self.target_kind)
        if success_now:
            self.success_step = int(step)
            self.success_evidence = {
                "step": int(step), "policy_use_step": causal["step"],
                "gui_open_step": causal["gui_open_step"],
                "confirmation_delay_steps": int(step) - causal["step"],
                "target_kind": self.target_kind, "target_cell": list(self.target_cell),
                "interaction_stat": self.interaction_stat,
                "interaction_stat_delta": stat,
                "interaction_stat_fresh_delta": stat - causal["stat_before"],
                "is_gui_open_after": True,
                "exact_target_at_use": causal["exact_target_at_use"],
                "exact_target_at_confirmation": (
                    causal["exact_target_at_confirmation"]),
                "renderer_confirmation_delay_allowed": True,
                "policy_attribution_mode": self.policy_attribution_mode,
                "scene_requires_unique_target_gui_instance": True,
            }
        self.step_events[int(step)] = {
            "gui_open_after": gui, "interaction_stat": self.interaction_stat,
            "interaction_stat_delta_after": stat, "policy_use_causal": causal is not None,
            "inventory_after": DPX._inventory_counts(self.world),
        }
        return {"success_now": success_now, "control_complete_now": False,
                "note": (f"OPEN {self.target_kind} stat={stat} gui={int(gui)}")
                if use or success_now else ""}

    def finish_audit(self) -> bool:
        passed = bool(self.success_step is not None
                      and self.world.info.get("is_gui_open", False)
                      and self.success_evidence["interaction_stat_fresh_delta"] > 0)
        self.terminal_audit = {
            "performed": self.success_step is not None, "passed": passed,
            "target_kind": self.target_kind, "target_cell": list(self.target_cell),
            "interaction_stat": self.interaction_stat,
            "interaction_stat_delta": self.interact_stat_delta,
            "is_gui_open_after": bool(self.world.info.get("is_gui_open", False)),
            "policy_attribution_mode": self.policy_attribution_mode,
            "pre_action_dense_target_mask_materialized": True,
            "success_evidence": self.success_evidence,
        }
        return passed
