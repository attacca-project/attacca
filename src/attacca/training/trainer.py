#!/usr/bin/env python
# Adapted from ROCKET-2 (https://github.com/CraftJarvis/ROCKET-2,
# commit 5ea1b3b2e4a4b43af8f00a75f232bb490ec8fb6a, train.py).
# Upstream metadata: Date: 2025-01-14 09:36:12; LastEditors: caishaofei-mus1
# 1744260356@qq.com; LastEditTime: 2025-03-16 15:09:09;
# FilePath: /ROCKET-2/train.py.
# Reused: Hydra/Lightning training structure. Modified: Attacca model warm
# start, mixed data modules, target-map and phase losses, and checkpoint filtering.
"""Attacca training on the released human-play packs."""
from datetime import timedelta
from pathlib import Path

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from minestudio.offline import MineLightning
from minestudio.offline.utils import convert_to_normal
from minestudio.offline.lightning_callbacks import SmartCheckpointCallback, SpeedMonitorCallback
from minestudio.data.minecraft import RawDataModule

from attacca.models.model import CrossViewRocket
from attacca.training.losses import MaskedBehaviorCloneCallback
from attacca.training.losses import SyntheticExploreAuxCallback
from attacca.training.phase import PhaseHeadCallback
from attacca.data.dataset import build_mixed_datamodule


class StripDinoCheckpoint(L.Callback):
    _P = ("view_backbone.",)

    def _is_dino(self, k):
        return any(p in k for p in self._P)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        sd = checkpoint.get("state_dict")
        if sd is not None:
            checkpoint["state_dict"] = {k: v for k, v in sd.items() if not self._is_dino(k)}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k, v in pl_module.state_dict().items():
                if self._is_dino(k) and k not in sd:
                    sd[k] = v


def _enforce_spatial_token_contract(rocket_policy):
    failures = []
    expected_tokens = 11
    if getattr(rocket_policy, "num_step_tokens", None) != expected_tokens:
        failures.append(
            f"num_step_tokens={getattr(rocket_policy, 'num_step_tokens', None)!r}, "
            f"expected {expected_tokens}")
    if not bool(getattr(rocket_policy, "use_union_token", False)):
        failures.append("use_union_token is not active")
    if not hasattr(rocket_policy, "union_mask_head"):
        failures.append("predicted union head is missing")
    if not hasattr(rocket_policy, "union_token_out"):
        failures.append("11-token post-training residual projection is missing")
    dino_trainable = [name for name, param in rocket_policy.view_backbone.named_parameters()
                      if param.requires_grad]
    if dino_trainable:
        failures.append(f"frozen DINO contract violated: {dino_trainable[:4]}")
    if failures:
        raise SystemExit(
            "[train] GROUNDING-TOKEN CONTRACT FAILURE: "
            + "; ".join(failures))
    print("[train] grounding-token contract OK: [9 view, interaction, prev_action]; "
          "pooled UNION token added to the prev_action token; DINO frozen", flush=True)


def _enforce_goal_fusion_contract(rocket_policy, model_cfg):
    mode = str(model_cfg.get("goal_fusion_mode", "official_mask_vit"))
    expected_cache = bool(model_cfg.get("cache_static_goal_features", False))
    if bool(getattr(rocket_policy, "cache_static_goal_features", False)) != expected_cache:
        raise SystemExit(
            "[train] GOAL-CACHE CONTRACT FAILURE: "
            "cache_static_goal_features was not applied to the constructed model")
    if getattr(rocket_policy, "goal_fusion_mode", None) != mode:
        raise SystemExit(
            "[train] GOAL-FUSION CONTRACT FAILURE: goal_fusion_mode was "
            "not applied to the constructed model")
    if mode != "official_mask_vit":
        raise SystemExit(f"[train] unknown goal_fusion_mode={mode!r}")
    if not hasattr(rocket_policy, "mask_backbone"):
        raise SystemExit(
            "[train] GOAL-FUSION CONTRACT FAILURE: official_mask_vit "
            "requires the released one-channel mask backbone")
    dino_trainable = [
        name for name, param in rocket_policy.view_backbone.named_parameters()
        if param.requires_grad]
    if dino_trainable:
        raise SystemExit(
            "[train] GOAL-FUSION CONTRACT FAILURE: shared DINO is trainable: "
            f"{dino_trainable[:4]}")
    print(f"[train] goal-fusion contract OK: {mode}; shared DINO frozen",
          flush=True)


def build_data_module(args):
    data_module_cls = build_mixed_datamodule(RawDataModule)
    raw_synthetic_lmdb = args.get("synthetic_lmdb", None)
    synthetic_lmdb_arg = (
        None if raw_synthetic_lmdb is None
        else str(raw_synthetic_lmdb) if isinstance(raw_synthetic_lmdb, str)
        else [str(pack_dir) for pack_dir in raw_synthetic_lmdb])
    synth_kw = {"synthetic_lmdb": synthetic_lmdb_arg,
                "synthetic_seed": int(args.get("synthetic_seed", 0)),
                "synthetic_only": bool(args.get("synthetic_only", False)),
                "synthetic_pad_short_min_length": (
                    None if args.get("synthetic_pad_short_min_length", None) is None
                    else int(args.synthetic_pad_short_min_length)),
                "synthetic_require_visible_surface_masks": bool(
                    args.get("synthetic_require_visible_surface_masks", False)),
                "synthetic_donor_infeasible_policy": str(
                    args.get("synthetic_donor_infeasible_policy", "error")),
                "synthetic_goal_donor_min_resized_pixels": int(
                    args.get("synthetic_goal_donor_min_resized_pixels", 0)),
                "synthetic_goal_donor_min_resized_short_side": int(
                    args.get("synthetic_goal_donor_min_resized_short_side", 0)),
                "synthetic_goal_donor_max_attempts": int(
                    args.get("synthetic_goal_donor_max_attempts", 24)),
                "synthetic_episode_validation_fraction": float(
                    args.get("synthetic_episode_validation_fraction", 0.0)),
                "synthetic_episode_split_seed": int(
                    args.get("synthetic_episode_split_seed", 0)),
                "synthetic_camera_quantization": str(
                    args.get("synthetic_camera_quantization", "packed"))}
    return data_module_cls(
        data_params=dict(win_len=args.win_len),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
        episode_continuous_batch=args.episode_continuous_batch,
        **synth_kw,
    )


def train(args, output_dir):
    output_dir = Path(output_dir).resolve()
    if args.get("seed", None) is not None:
        L.seed_everything(int(args.seed), workers=True)
        print(f"[train] seed_everything({int(args.seed)}, workers=True)", flush=True)

    mcfg = dict(args.model)
    if (not bool(mcfg.get("use_union_token", False))
            or str(mcfg.get("phase_cond_mode", "off")) != "film"):
        raise SystemExit("[train] Attacca requires model.use_union_token=true and "
                         "model.phase_cond_mode=film")
    if not bool(args.get("synthetic_lmdb", None)) or not bool(args.get("synthetic_only", False)):
        raise SystemExit("[train] training requires synthetic_lmdb and synthetic_only=true")
    if not args.init_from:
        raise SystemExit("[train] init_from must name the released checkpoint")
    if not args.run_tag:
        raise SystemExit("[train] REFUSING: a non-empty run_tag is required "
                         "(checkpoint directories are weights_<run_tag>/checkpoints_<run_tag>).")
    print(f"[train] 11-token grounding-residual model + warm-start from {args.init_from} "
          "(strict=False)", flush=True)
    base = CrossViewRocket.from_pretrained(args.init_from, revision="main")
    rocket_policy = CrossViewRocket(**mcfg)
    base_sd = base.state_dict()
    missing, unexpected = rocket_policy.load_state_dict(base_sd, strict=False)
    del base_sd
    miss_non_dino = [k for k in missing if ".view_backbone." not in k]
    print(f"[train] warm-start: {len(missing)} missing ({len(miss_non_dino)} non-DINO = new modules), "
          f"{len(unexpected)} unexpected. sample new: {miss_non_dino[:8]}", flush=True)
    assert not unexpected, f"unexpected keys (arch mismatch with the released model): {unexpected[:8]}"
    allowed_new = ("union_", "phase_")
    bad = [k for k in miss_non_dino if not k.startswith(allowed_new)]
    assert not bad, f"unexpected new parameter keys: {bad[:8]}"
    del base

    if getattr(rocket_policy, "phase_cond_mode", None) != "film":
        raise SystemExit(
            "[train] model flag mismatch: phase_cond_mode was not applied to the "
            "constructed model")
    _enforce_goal_fusion_contract(rocket_policy, mcfg)
    _enforce_spatial_token_contract(rocket_policy)

    if args.get("seed", None) is not None:
        L.seed_everything(int(args.seed), workers=True)
        print(f"[train] runtime RNG reset after model init: {int(args.seed)}",
              flush=True)

    dw_uniform = bool(args.get("bc_decision_weight_uniform", False))
    bc_cb = MaskedBehaviorCloneCallback(weight=args.objective_weight,
                                        decision_weight_uniform=dw_uniform)
    print(f"[train] BC={type(bc_cb).__name__} bc_decision_weight_uniform={dw_uniform}", flush=True)

    obj_cbs = [bc_cb]
    synth_union_mask_weight = float(args.get("synthetic_union_mask_weight", 0.0))
    if synth_union_mask_weight <= 0:
        raise SystemExit("[train] synthetic_union_mask_weight must be > 0")
    obj_cbs.append(SyntheticExploreAuxCallback(
        union_mask_weight=synth_union_mask_weight))
    print(f"[train] aux (dense): union target map weight {synth_union_mask_weight}",
          flush=True)

    phase_w = float(args.get("phase_aux_weight", 0.0) or 0.0)
    if phase_w <= 0:
        raise SystemExit("[train] phase_aux_weight must be > 0 (model.phase_cond_mode=film)")
    obj_cbs.append(PhaseHeadCallback(weight=phase_w))
    print(f"[train] phase head CE weight={phase_w} (visibility labels); policy head FiLM-conditioned on "
          f"predicted phase, stopgrad={mcfg.get('phase_cond_stopgrad', True)}", flush=True)

    mine_lightning = MineLightning(
        mine_policy=rocket_policy,
        log_freq=int(args.get("log_freq", 1)),
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        callbacks=obj_cbs,
        hyperparameters=convert_to_normal(args),
    )
    _orig_on_load = mine_lightning.on_load_checkpoint

    def _backfill_dino_on_load(checkpoint):
        _orig_on_load(checkpoint)
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k, v in mine_lightning.state_dict().items():
                if "view_backbone." in k and k not in sd:
                    sd[k] = v

    mine_lightning.on_load_checkpoint = _backfill_dino_on_load

    mine_data = build_data_module(args)

    tag = f"_{args.run_tag}" if args.run_tag else ""
    weight_dir = output_dir / f"weights{tag}"
    checkpoint_dir = output_dir / f"checkpoints{tag}"
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        SpeedMonitorCallback(),
        SmartCheckpointCallback(dirpath=str(weight_dir), filename="weight-{epoch}-{step}",
                                save_top_k=-1, every_n_train_steps=args.save_freq, save_weights_only=True),
        SmartCheckpointCallback(dirpath=str(checkpoint_dir), filename="ckpt-{epoch}-{step}",
                                save_top_k=1, every_n_train_steps=args.save_freq + 1, save_weights_only=False),
    ]
    if args.get("policy_only_ckpt", False):
        callbacks.append(StripDinoCheckpoint())
        print("[train] policy_only_ckpt=True -> ckpts drop frozen DINO (~382MB vs ~1GB); resume backfills from timm", flush=True)

    trainer_kw = dict(
        logger=CSVLogger(save_dir=str(output_dir),
                         name=f"csv_{args.run_tag}" if args.run_tag else "csv"),
        devices=args.devices,
        precision="bf16",
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(hours=3)),
        use_distributed_sampler=not args.episode_continuous_batch,
        callbacks=callbacks,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=str(output_dir),
    )
    if args.max_steps and args.max_steps > 0:
        trainer_kw["max_steps"] = args.max_steps

    L.Trainer(**trainer_kw).fit(model=mine_lightning, datamodule=mine_data, ckpt_path=args.ckpt_path)
