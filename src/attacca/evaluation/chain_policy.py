#!/usr/bin/env python3
"""Policy backend selection shared by the long-horizon chains."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src/attacca/evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "src/attacca/evaluation"))

from attacca.evaluation.policy import Rocket2GoalRunner


ROCKET_BACKENDS = ("in_tree_rocket2",)
POLICY_BACKENDS = ROCKET_BACKENDS

CHAIN_TASKS = frozenset({
    "portal_build", "portal_ignite", "water_scoop", "lava_pour",
    "obsidian_mine", "oak_log_mine", "diamond_ore_mine3",
    "crafting_table_open", "coal_ore_mine", "cow_hunt", "furnace_find",
    "wolf_feed",
})


def sha256_file(path: Path) -> str:
    if path.is_dir():
        path = path / "model.safetensors"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_policy_rng(torch_module, rollout_seed: int) -> None:
    rollout_seed = int(rollout_seed)
    if rollout_seed < 0:
        raise ValueError("policy_rollout_seed must be non-negative")
    random.seed(rollout_seed)
    np.random.seed(rollout_seed)
    torch_module.manual_seed(rollout_seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(rollout_seed)


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy-backend", choices=POLICY_BACKENDS,
        default="in_tree_rocket2")
    parser.add_argument("--policy-rollout-seed", type=int, default=0)


class ChainPolicyRunner:

    def __init__(
            self, checkpoint: Path, *, backend: str, cfg_coef: float,
            policy_rollout_seed: int, device: str = "cuda"):
        import torch

        if backend not in POLICY_BACKENDS:
            raise ValueError(backend)
        if int(policy_rollout_seed) < 0:
            raise ValueError("policy_rollout_seed must be non-negative")
        self.backend = str(backend)
        self.policy_rollout_seed = int(policy_rollout_seed)
        self.torch = torch
        self.current_task: str | None = None
        self.stage_conditioning: dict[str, dict[str, Any]] = {}
        checkpoint = checkpoint.expanduser().resolve()
        self.model_contract = {
            "policy_backend": self.backend,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "cfg_coef": float(cfg_coef),
        }
        self.inner = Rocket2GoalRunner(
            str(checkpoint), cfg_coef=float(cfg_coef), device=device)
        seed_policy_rng(self.torch, self.policy_rollout_seed)

    def prepare_task(self, task: str) -> dict[str, Any]:
        task = str(task)
        if task not in CHAIN_TASKS:
            raise ValueError(f"unregistered chain stage {task!r}")
        self.current_task = task
        evidence = {
            "backend": self.backend,
            "conditioning": "goal_exemplar_rgb_mask",
            "task": task,
        }
        self.stage_conditioning[task] = dict(evidence)
        return dict(evidence)

    def capture_prediction(self) -> dict[str, Any]:
        from attacca.evaluation.place import _capture_prediction
        return _capture_prediction(self.inner)

    def rollout_metadata(self) -> dict[str, Any]:
        task = self.current_task
        return {
            "policy_backend": self.backend,
            "policy_rollout_seed": self.policy_rollout_seed,
            "stage_conditioning": (
                None if task is None else self.stage_conditioning.get(task)),
        }

    def set_obj_id(self, obj_id: int) -> None:
        self.inner.set_obj_id(int(obj_id))

    def act(self, obs: Mapping[str, Any], *goal) -> tuple[dict, float]:
        return self.inner.act(obs, *goal)


def build_runner(
        args: argparse.Namespace, *,
        device: str = "cuda") -> ChainPolicyRunner:
    return ChainPolicyRunner(
        Path(args.checkpoint), backend=str(args.policy_backend),
        cfg_coef=float(args.cfg_coef),
        policy_rollout_seed=int(args.policy_rollout_seed),
        device=device)
