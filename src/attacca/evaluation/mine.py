#!/usr/bin/env python3
"""Mine benchmark: restores each world snapshot in a fresh process, applies the scene's target class assignment and runs one policy rollout per scene and policy seed (subcommand evaluate)."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO) + "/src")
sys.path.insert(0, str(REPO / "src/attacca/evaluation"))

from attacca.worlds.world_snapshot import load_world_snapshot_bundle
import attacca.evaluation.mine_episode as DEMO
import attacca.evaluation.mine_worlds as SELECTED_CLASSSWAP


CONTRACT = "xbench_v12_humanplay_policy_eval/v1"
SCENE_CONTRACT = "independent_v12_exposed_stone_multiclass6/v1"
PROFILE = "v12_independent_multiclass6"
ORES = (
    "coal_ore", "iron_ore", "gold_ore", "lapis_ore",
    "diamond_ore", "emerald_ore", "redstone_ore",
)
SUPPORTED_ORES = ORES + SELECTED_CLASSSWAP.GENERALIZATION_TARGETS
SELECTED_CLASSSWAP_TARGETS = SELECTED_CLASSSWAP.EVAL_TARGETS
SELECTED_CLASSSWAP_SCENE_CONTRACT = SELECTED_CLASSSWAP.SCENE_MANIFEST_CONTRACT
SELECTED_CLASSSWAP_RUNTIME_REMAP_CONTRACT = (
    "xbench_mine_shared16_runtime_remap/v2")
SELECTED_CLASSSWAP_EVAL_RUNTIME_CONTRACT = (
    "xbench_selected_world_classswap_eval_runtime/v1")
EVAL_GOAL_MODES = ("plains_grass_front",)
SITE_ATTEMPTS = 18
CFG_COEF = 0.0
BUDGET = 1000
if not set(EVAL_GOAL_MODES).issubset(set(DEMO.POLICY_GOAL_MODES)):
    raise RuntimeError("demo policy goal-mode vocabulary is stale")


def _is_selected_classswap(scene: dict) -> bool:
    return str(scene.get("contract")) == SELECTED_CLASSSWAP_SCENE_CONTRACT


def _scene_roster(scene: dict) -> tuple[str, ...]:
    roster = tuple(str(value) for value in (scene.get("class_roster") or ORES))
    target = str(scene.get("target_kind"))
    if (not roster or len(set(roster)) != len(roster)
            or any(value not in SUPPORTED_ORES for value in roster)
            or target not in SELECTED_CLASSSWAP_TARGETS):
        raise ValueError(f"invalid scene class roster: {roster}")
    return roster


def _scene_snapshot(scene: dict) -> dict:
    snapshot = scene.get("source_snapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("bundle"):
        raise ValueError("scene lacks source_snapshot bundle")
    return snapshot


def _scene_manifest_ready(scene: dict) -> bool:
    return (_is_selected_classswap(scene)
            and scene.get("state") == "manifest_ready_runtime_remap_pending")


def _benchmark_mode_args(scene: dict) -> list[str]:
    values = []
    for field, flag in (
            ("geometry_sha256", "--selected-classswap-geometry-sha256"),
            ("assignment_sha256", "--selected-classswap-assignment-sha256")):
        digest = str(scene.get(field, ""))
        if (len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(
                f"selected-world {field} is not a lowercase SHA256")
        values.extend([flag, digest])
    return ["--benchmark-selected-classswap", *values]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _pin_selected_eval_runtime(root: Path, manifest_path: Path) -> Path:
    runtime_path = root / "selected_classswap_eval_runtime.json"
    runtime = {
        "contract": SELECTED_CLASSSWAP_EVAL_RUNTIME_CONTRACT,
        "scene_manifest_sha256": _sha256(manifest_path),
        "scenes": {},
        "uncanonicalized": True,
    }
    if runtime_path.is_file():
        if json.loads(runtime_path.read_text()) != runtime:
            raise ValueError(
                "selected-world eval-runtime manifest identity differs")
    else:
        _atomic_json(runtime_path, runtime)
    return runtime_path


def reusable_terminal_receipt(receipt: dict) -> bool:
    if receipt.get("status") != "complete":
        return False
    result = receipt.get("result") or {}
    if int(result.get("eval_valid", 0) or 0) == 1:
        return True
    return (result.get("invalid_reason")
            == "staged_cells_outside_scrubbed_eval_domain")


def validate_scene_bundle(bundle_path: Path, expected: dict) -> None:
    _scene_roster(expected)
    bundle = load_world_snapshot_bundle(bundle_path.resolve())
    payload = bundle.payload
    identity = dict(payload.get("identity") or {})
    for key in ("world_seed", "site_seed", "layout_seed", "pose_seed",
                "action_seed"):
        if int(identity.get(key, -1)) != int(expected[key]):
            raise ValueError(f"snapshot {key} differs from scene manifest")
    identity_target = str((expected.get("runtime_remap") or {}).get(
        "original_source_target_kind"))
    if (identity.get("setting") != expected["setting"]
            or identity.get("target_kind") != identity_target
            or identity.get("mine_worldgen_profile") != PROFILE
            or identity.get("biome") != "extreme_hills"
            or identity.get("census_scrub") is not True):
        raise ValueError("snapshot semantic identity differs from scene manifest")
    ctx = dict(payload.get("ctx") or {})
    layout = dict(ctx.get("mine_scene_layout") or {})
    sites = list(ctx.get("mine_sites") or ())
    cells = [
        (int(site["x"]), int(site["y"]), int(site["z"])) for site in sites]
    source_declaration = expected.get("source_snapshot")
    runtime_remap = expected.get("runtime_remap")
    if (not isinstance(source_declaration, dict)
            or not source_declaration.get("immutable_source")
            or source_declaration.get("derived_snapshot_published")
            is not False
            or not isinstance(runtime_remap, dict)):
        raise ValueError(
            "selected-world scene lacks immutable source/remap declarations")
    if Path(source_declaration.get("bundle", "")).resolve() \
            != bundle_path.resolve():
        raise ValueError(
            "selected-world source bundle differs from scene manifest")
    if payload.get("selected_world_classswap") is not None:
        raise ValueError(
            "selected-world class-swap requires an unmodified source snapshot")
    source_metadata = dict(bundle.metadata)
    source_metadata["world_sha256"] = bundle.world.sha256
    try:
        source_layout = SELECTED_CLASSSWAP.extract_source_layout(payload)
        assignment = SELECTED_CLASSSWAP.deterministic_assignment(
            source_layout, expected["target_kind"])
        geometry_sha256 = SELECTED_CLASSSWAP.geometry_start_sha256(payload)
        assignment_sha256 = SELECTED_CLASSSWAP.assignment_sha256(
            assignment)
        source_pins = SELECTED_CLASSSWAP.source_snapshot_hash_pins(
            source_metadata, payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid selected-world source snapshot: {exc}") from exc
    declared_source_pins = {
        "source_world_sha256": source_declaration.get("world_sha256"),
        "source_world_archive_sha256": source_declaration.get(
            "world_archive_sha256"),
        "source_payload_sha256": source_declaration.get("payload_sha256"),
        "source_staged_rgb_sha256": source_declaration.get(
            "staged_rgb_sha256"),
    }
    derived_plan = [{
        "cell": list(row["cell"]),
        "role": str(row["role"]),
        "source_kind": str(row["source_kind"]),
        "runtime_kind": str(row["after"]),
    } for row in assignment["staged_cells"]]
    runtime_source_pins = {
        key: runtime_remap.get(key) for key in (
            "source_world_sha256", "source_world_archive_sha256",
            "source_payload_sha256")
    }
    if (declared_source_pins != source_pins
            or expected.get("geometry_sha256") != geometry_sha256
            or expected.get("assignment_sha256") != assignment_sha256
            or runtime_remap.get("contract")
            != SELECTED_CLASSSWAP_RUNTIME_REMAP_CONTRACT
            or runtime_remap.get("camera_off_staging_required") is not True
            or runtime_remap.get("publish_derived_snapshot") is not False
            or runtime_remap.get("source_target_kind")
            != assignment["source_target_kind"]
            or runtime_remap.get("original_source_target_kind")
            != assignment["original_source_target_kind"]
            or runtime_remap.get("target_kind") != expected["target_kind"]
            or runtime_remap.get("geometry_sha256") != geometry_sha256
            or runtime_remap.get("assignment_sha256")
            != assignment_sha256
            or runtime_source_pins != {
                key: source_pins[key] for key in runtime_source_pins}
            or runtime_remap.get("exact_six_cell_plan") != derived_plan
            or runtime_remap.get("expected_source_to_runtime_diff")
            != SELECTED_CLASSSWAP.expected_source_to_variant_diff(
                assignment)
            or runtime_remap.get("coral_hydration_cell")
            != assignment.get("coral_hydration_cell")
            or runtime_remap.get("coral_hydration_block")
            != assignment.get("coral_hydration_block")
            or runtime_remap.get(
                "coral_hydration_applies_to_all_targets") is not True):
        raise ValueError(
            "selected-world class-swap manifest/remap differs from "
            "the source snapshot")
    source_target = assignment["original_source_target_kind"]
    source_target_sites = [
        row for row in sites
        if SELECTED_CLASSSWAP.norm_kind(row.get("kind")) == source_target]
    source_distractors = [
        SELECTED_CLASSSWAP.norm_kind(row.get("kind")) for row in sites
        if SELECTED_CLASSSWAP.norm_kind(row.get("kind")) != source_target]
    if (ctx.get("cls") != source_target or int(ctx.get("quota", -1)) != 3
            or layout.get("contract") != SCENE_CONTRACT
            or len(sites) != 6 or len(set(cells)) != 6
            or len(source_target_sites) != 3
            or len(source_distractors) != 3
            or len(set(source_distractors)) != 3
            or any(value not in tuple(ORES) + ("nether_gold_ore",)
                   for value in source_distractors)):
        raise ValueError(
            "source snapshot is not exact original target3 + distinct "
            "confuser3")


def checkpoint_contract(path: Path, name: str,
                        policy_backend: str = "rocket2_static_goal") -> dict:
    if path.is_dir():
        from attacca.models.inference_checkpoint import FORMAT_VERSION
        from attacca.models.inference_checkpoint import INFERENCE_CONTRACT
        from attacca.models.inference_checkpoint import MODEL_TYPE
        from attacca.models.inference_checkpoint import is_attacca_checkpoint
        from attacca.models.inference_checkpoint import read_attacca_config
        cfg_path = path / "config.json"
        weight_path = path / "model.safetensors"
        if not cfg_path.is_file() or not weight_path.is_file():
            raise ValueError(f"{name}: Attacca checkpoint directory is incomplete: {path}")
        if not is_attacca_checkpoint(path):
            raise ValueError(f"{name}: not an Attacca checkpoint directory: {path}")
        cfg = read_attacca_config(path)
        mode = str(cfg.get("goal_fusion_mode", "official_mask_vit"))
        resolved = {
            "goal_fusion_mode": mode,
            "num_step_tokens": 11,
            "use_union_token": bool(cfg.get("use_union_token", False)),
            "cache_static_goal_features": bool(
                cfg.get("cache_static_goal_features", False)),
            "model_config": cfg,
            "checkpoint_format": "attacca_safetensors",
            "weight_sha256": _sha256(weight_path),
            "config_sha256": _sha256(cfg_path),
            "policy_backend": policy_backend,
            "model_type": MODEL_TYPE,
            "format_version": FORMAT_VERSION,
            "inference_contract": INFERENCE_CONTRACT,
        }
        return resolved
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("hyper_parameters", {}).get("model")
    if not isinstance(cfg, dict):
        raise ValueError(f"{name}: checkpoint lacks hyper_parameters.model")
    mode = str(cfg.get("goal_fusion_mode", "official_mask_vit"))
    resolved = {
        "goal_fusion_mode": mode,
        "num_step_tokens": 11,
        "use_union_token": bool(cfg.get("use_union_token", False)),
        "cache_static_goal_features": bool(
            cfg.get("cache_static_goal_features", False)),
        "model_config": cfg,
    }
    del ckpt
    return resolved


def _checkpoint_digest(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(value for value in path.rglob("*") if value.is_file()):
            digest.update(str(child.relative_to(path)).encode())
            digest.update(b"\0")
            digest.update(_sha256(child).encode())
            digest.update(b"\0")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def load_models(path: Path, *, require_count: int | None = None) -> list[dict]:
    raw = json.loads(path.read_text())
    rows = raw.get("models") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("models JSON must contain a nonempty models list")
    names = set()
    out = []
    for row in rows:
        name = str(row["name"])
        policy_backend = str(row.get(
            "policy_backend", "rocket2_static_goal"))
        if policy_backend != "rocket2_static_goal":
            raise ValueError(f"{name}: unknown policy backend {policy_backend}")
        ckpt = Path(row["ckpt"]).expanduser().resolve()
        if name in names or not (ckpt.is_file() or ckpt.is_dir()):
            raise ValueError(f"duplicate model name or missing checkpoint: {name} {ckpt}")
        names.add(name)
        out.append({
            **dict(row), "name": name, "ckpt": str(ckpt),
            "policy_backend": policy_backend,
            "ckpt_sha256": _checkpoint_digest(ckpt),
            "resolved_contract": checkpoint_contract(
                ckpt, name, policy_backend),
        })
    if require_count is not None and len(out) != int(require_count):
        raise ValueError(
            f"model manifest has {len(out)} rows, expected {int(require_count)}")
    return out


def goal_digests(goal_lib: Path, target: str, gmode: str) -> dict:
    import cv2
    import numpy as np
    root = goal_lib / f"mine_{target}" / gmode
    goal, mask = root / "goal.png", root / "mask.png"
    if not goal.is_file() or not mask.is_file():
        raise FileNotFoundError(f"goal donor missing under {root}")
    raw_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise ValueError(f"goal donor mask cannot be decoded: {mask}")
    resized = cv2.resize(
        (raw_mask > 0).astype(np.uint8), (224, 224),
        interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(resized)
    area = int(resized.sum())
    short = (0 if area == 0 else int(min(
        int(xs.max()) - int(xs.min()) + 1,
        int(ys.max()) - int(ys.min()) + 1)))
    if area < 1000 or short < 20:
        raise ValueError(
            f"goal donor {target} fails final resized 1000/20 gate: "
            f"area={area} short={short}")
    return {
        "goal_sha256": _sha256(goal),
        "goal_mask_sha256": _sha256(mask),
        "goal_resized_area": area,
        "goal_resized_short_side": short,
    }


def rollout_command(scene: dict, model: dict, *, goal_lib: Path,
                    out: Path, args, policy_rollout_seed: int = 0) -> list[str]:
    roster = _scene_roster(scene)
    snapshot = _scene_snapshot(scene)
    return [
        args.python, "-u", str(REPO / "src/attacca/evaluation" / "mine_episode.py"),
        "--out", str(out), "--cells", "mine",
        "--settings", scene["setting"], "--biome", "extreme_hills",
        "--world-seed", str(scene["world_seed"]),
        "--site-seed", str(scene["site_seed"]),
        "--layout-seed", str(scene["layout_seed"]),
        "--pose-seed", str(scene["pose_seed"]),
        "--action-seed", str(scene["action_seed"]),
        "--ore-class", scene["target_kind"],
        "--ore-classes", ",".join(roster), "--quota", "1",
        "--site-attempts", str(SITE_ATTEMPTS),
        "--free-site", "--story",
        "--visible-surface-masks", "--all-class-census", "--solo-episode",
        *_benchmark_mode_args(scene),
        "--mine-worldgen-profile", PROFILE,
        "--distractor-policy", "random_other", "--ore-distractor-classes", "3",
        "--census-scrub", "--world-snapshot-in", snapshot["bundle"],
        "--snapshot-allow-uncanonical-ready",
        "--policy-driver", "--policy-ckpt", model["ckpt"],
        "--policy-model-name", model["name"],
        "--policy-backend", str(model.get(
            "policy_backend", "rocket2_static_goal")),
        "--policy-rollout-seed", str(int(policy_rollout_seed)),
        "--policy-goal-lib", str(goal_lib), "--policy-gmode", args.gmode,
        "--policy-cfg-coef", str(model.get("cfg_coef", CFG_COEF)),
        "--policy-budget", str(BUDGET), "--policy-post", str(args.post),
        "--policy-success-quota", str(int(getattr(args, "success_quota", 1))),
        "--policy-record", "1",
    ]


def _execute_rollout_jobs(jobs, *, workers, run_one, on_complete):
    jobs9 = list(jobs)
    by_model9 = {}
    for job9 in jobs9:
        model_name9 = str(job9["model"]["name"])
        by_model9.setdefault(model_name9, []).append(job9)
    batches9 = list(by_model9.values())
    completed9 = []
    for batch9 in batches9:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=int(workers)) as executor9:
            futures9 = [executor9.submit(run_one, job9) for job9 in batch9]
            for future9 in concurrent.futures.as_completed(futures9):
                receipt9 = future9.result()
                completed9.append(receipt9)
                on_complete(receipt9)
    return completed9


def evaluate(args) -> None:
    if not (1 <= int(args.success_quota) <= 3):
        raise SystemExit("--success-quota must be in [1,3]")
    if int(args.post) != 20:
        raise SystemExit("the benchmark requires --post 20")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = (Path(args.scene_manifest).expanduser().resolve()
                     if args.scene_manifest else root / "scenes.json")
    manifest = json.loads(manifest_path.read_text())
    resolved_scene_manifest = root / "scenes.json"
    if resolved_scene_manifest.is_file():
        if json.loads(resolved_scene_manifest.read_text()) != manifest:
            raise SystemExit(
                "existing resolved scene manifest differs; use a new eval root")
    else:
        _atomic_json(resolved_scene_manifest, manifest)
    all_scenes = manifest["scenes"]
    if len(all_scenes) != int(manifest["expected_scenes"]) or any(
            not _scene_manifest_ready(scene) for scene in all_scenes):
        raise SystemExit("scene manifest is not complete")
    try:
        selected_runtime_path9 = _pin_selected_eval_runtime(
            root, resolved_scene_manifest)
        for scene in all_scenes:
            validate_scene_bundle(
                Path(_scene_snapshot(scene)["bundle"]), scene)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    requested = set(args.scene_id or ())
    available = {str(scene["scene_id"]) for scene in all_scenes}
    unknown = sorted(requested - available)
    if unknown:
        raise SystemExit(f"unknown requested scene ids: {unknown}")
    scenes = [scene for scene in all_scenes
              if not requested or str(scene["scene_id"]) in requested]
    models = load_models(
        Path(args.models_json).resolve(), require_count=args.require_model_count)
    resolved_manifest_path = root / "models_resolved.json"
    resolved_manifest = {
        "contract": CONTRACT,
        "models": models,
    }
    if resolved_manifest_path.is_file():
        if json.loads(resolved_manifest_path.read_text()) != resolved_manifest:
            raise SystemExit(
                "existing resolved model manifest differs; use a new eval root")
    else:
        _atomic_json(resolved_manifest_path, resolved_manifest)
    goal_lib = Path(args.goal_lib).resolve()
    workers = int(args.workers)
    if workers < 1:
        raise SystemExit("--workers must be positive")
    env = os.environ.copy()
    env.setdefault("MINESTUDIO_DIR", str(REPO / ".minestudio"))
    env.setdefault("MINESTUDIO_GPU_RENDER", "1")
    env.setdefault("RENDER_DEVICES", "0")
    env.setdefault("MINESTUDIO_MAX_RESETTING_ENV_COUNT", str(workers))
    receipts = root / "receipts"
    jobs = []
    rollout_seeds = tuple(dict.fromkeys(int(value) for value in (
        args.policy_rollout_seed if args.policy_rollout_seed is not None else (0,))))
    if not rollout_seeds:
        raise SystemExit("at least one --policy-rollout-seed is required")
    eval_config_path9 = root / "eval_config.json"
    eval_config9 = {
        "contract": CONTRACT,
        "scene_manifest_sha256": _sha256(resolved_scene_manifest),
        "models_manifest_sha256": _sha256(resolved_manifest_path),
        "goal_library": str(goal_lib),
        "goal_mode": str(args.gmode),
        "cfg_coef": float(CFG_COEF),
        "budget": int(BUDGET),
        "post": int(args.post),
        "success_quota": int(args.success_quota),
        "scene_ids": [str(scene["scene_id"]) for scene in scenes],
        "policy_rollout_seeds": list(rollout_seeds),
        "selected_eval_runtime_sha256": _sha256(selected_runtime_path9),
    }
    if eval_config_path9.is_file():
        existing_eval_config9 = json.loads(eval_config_path9.read_text())
        if existing_eval_config9 != eval_config9:
            raise SystemExit("existing eval configuration differs; use a new eval root")
    else:
        _atomic_json(eval_config_path9, eval_config9)
    for model in models:
        for scene in scenes:
            for rollout_seed in rollout_seeds:
                backend9 = str(model.get(
                    "policy_backend", "rocket2_static_goal"))
                goal = goal_digests(
                    goal_lib, scene["target_kind"], args.gmode)
                effective_cfg9 = float(model.get("cfg_coef", CFG_COEF))
                snapshot9 = _scene_snapshot(scene)
                key_payload = {
                    "evaluator_contract": CONTRACT,
                    "model_sha256": model["ckpt_sha256"],
                    "model_contract": model["resolved_contract"],
                    "scene_id": scene["scene_id"],
                    "snapshot_world_sha256": snapshot9["world_sha256"],
                    "snapshot_archive_sha256": snapshot9.get(
                        "archive_sha256",
                        snapshot9.get("world_archive_sha256")),
                    "snapshot_payload_sha256": snapshot9["payload_sha256"],
                    "scene_manifest_sha256": _sha256(resolved_scene_manifest),
                    "selected_geometry_sha256": scene["geometry_sha256"],
                    "selected_assignment_sha256": scene["assignment_sha256"],
                    "selected_eval_ready_rgb_sha256": None,
                    "policy_backend": backend9,
                    "goal_mode": args.gmode,
                    **goal,
                    "cfg_coef": effective_cfg9, "budget": BUDGET,
                    "post": args.post, "success_quota": int(args.success_quota),
                    "policy_rollout_seed": int(rollout_seed),
                }
                key = hashlib.sha256(json.dumps(
                    key_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                rollout_id = f"{scene['scene_id']}__pseed{int(rollout_seed):08d}"
                receipt_path = receipts / model["name"] / f"{rollout_id}.json"
                out = root / "rollouts" / model["name"] / rollout_id
                if receipt_path.is_file():
                    old = json.loads(receipt_path.read_text())
                    if old.get("resume_key") != key:
                        raise SystemExit(f"resume identity changed for {receipt_path}")
                    if reusable_terminal_receipt(old):
                        continue
                    archive_root = root / "invalid_receipts" / model["name"] / rollout_id
                    archive_root.mkdir(parents=True, exist_ok=True)
                    attempt = 0
                    while (archive_root / f"receipt_{attempt:02d}.json").exists():
                        attempt += 1
                    os.replace(receipt_path, archive_root / f"receipt_{attempt:02d}.json")
                    if out.exists():
                        os.replace(out, archive_root / f"rollout_{attempt:02d}")
                out.mkdir(parents=True, exist_ok=True)
                command = rollout_command(
                    scene, model, goal_lib=goal_lib, out=out, args=args,
                    policy_rollout_seed=rollout_seed)
                log_path = out / "rollout.log"
                jobs.append({
                    "model": model, "scene": scene, "key": key,
                    "rollout_id": rollout_id,
                    "policy_rollout_seed": int(rollout_seed),
                    "key_payload": key_payload, "receipt_path": receipt_path,
                    "out": out, "command": command, "log_path": log_path,
                })

    def run_one9(job9):
        started = time.time()
        with job9["log_path"].open("w") as log:
            proc = subprocess.run(job9["command"], cwd=REPO, env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
        rows = []
        result_path = job9["out"] / "results_policy.jsonl"
        if result_path.is_file():
            rows = [json.loads(line) for line in result_path.read_text().splitlines()
                    if line.strip()]
        status = "complete" if proc.returncode == 0 and len(rows) == 1 else "failed"
        receipt = {
            "contract": CONTRACT, "resume_key": job9["key"],
            "identity": job9["key_payload"],
            "model": job9["model"],
            "scene_id": job9["scene"]["scene_id"],
            "rollout_id": job9["rollout_id"],
            "policy_rollout_seed": job9["policy_rollout_seed"],
            "returncode": proc.returncode, "status": status,
            "result": rows[0] if len(rows) == 1 else None,
            "out": str(job9["out"]), "log": str(job9["log_path"]),
            "started_unix_s": started, "finished_unix_s": time.time(),
        }
        _atomic_json(job9["receipt_path"], receipt)
        return receipt

    if jobs:
        print(json.dumps({
            "event": "evaluate_start", "pending": len(jobs),
            "workers": workers, "goal_mode": args.gmode,
        }, sort_keys=True), flush=True)
        completed9 = 0

        def report_complete9(receipt9):
            nonlocal completed9
            completed9 += 1
            print(json.dumps({
                "event": "evaluate_progress", "completed": completed9,
                "total": len(jobs), "status": receipt9["status"],
                "model": receipt9["model"]["name"],
                "scene_id": receipt9["scene_id"],
                "policy_rollout_seed": receipt9["policy_rollout_seed"],
            }, sort_keys=True), flush=True)

        _execute_rollout_jobs(
            jobs, workers=workers,
            run_one=run_one9, on_complete=report_complete9)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", required=True)
    common.add_argument("--python", default=sys.executable)

    run = sub.add_parser("evaluate", parents=[common])
    run.add_argument("--models-json", required=True)
    run.add_argument("--goal-lib", required=True)
    run.add_argument(
        "--gmode", default="plains_grass_front", choices=EVAL_GOAL_MODES,
        help="goal-library mode (one per eval root)")
    run.add_argument("--post", type=int, default=20)
    run.add_argument(
        "--success-quota", type=int, default=1,
        help="exact staged target breaks required for episode success (default: 1)")
    run.add_argument(
        "--workers", type=int, default=1,
        help="bounded concurrent fresh-process rollouts (default: 1)")
    run.add_argument("--require-model-count", type=int, default=10)
    run.add_argument(
        "--policy-rollout-seed", action="append", type=int,
        help=("independent policy sampling seed; repeat for multiple rollouts "
              "of the same immutable physical scene (default: 0)"))
    run.add_argument(
        "--scene-manifest",
        help=("read a complete scenes.json (the evaluation scene manifest) "
              "and pin a copy into this eval root"))
    run.add_argument(
        "--scene-id", action="append",
        help=("evaluate only this scene id from the evaluation manifest; "
              "repeat for several scenes"))
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
