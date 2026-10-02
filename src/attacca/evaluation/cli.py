import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

from attacca.paths import ROOT


SHORT_TASKS = ("mine", "hunt", "place")
CHAIN_TASKS = ("dpx", "ccfw", "wlo")
TASKS = SHORT_TASKS + CHAIN_TASKS
INFRA_FAILURE = re.compile(
    r"CUDA out of memory|OutOfMemory|Failed to launch|EOFError.*minerl|"
    r"sim.*init|Malmo.*(timeout|crash)|java.*IllegalState|could not bind",
    re.IGNORECASE,
)


def _path(value):
    return Path(value).expanduser().resolve()


def _command(module, *values):
    return [sys.executable, "-u", "-m", f"attacca.evaluation.{module}",
            *map(str, values)]


def _option(command, flag, *values):
    command.extend([flag, *map(str, values)])


def _selected_tasks(task):
    return {"short": SHORT_TASKS, "chains": CHAIN_TASKS,
            "all": TASKS}.get(task, (task,))


def _config(task):
    path = ROOT / "configs" / "eval" / f"{task}.yaml"
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"evaluation configuration must be a mapping: {path}")
    return config


def _environment(task, config, render):
    env = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
        "MINESTUDIO_MAX_RESETTING_ENV_COUNT": os.environ.get(
            "MINESTUDIO_MAX_RESETTING_ENV_COUNT", "8"),
    }
    if task in CHAIN_TASKS:
        env.update(OMP_NUM_THREADS="2", MINESTUDIO_MAX_RESETTING_ENV_COUNT="8")
    mode = render or config["render"]
    env["MINESTUDIO_GPU_RENDER"] = (
        "1" if mode == "gpu" else "0") if render else os.environ.get(
            "MINESTUDIO_GPU_RENDER", "1" if mode == "gpu" else "0")
    return env


def build_plan(args):
    tasks = _selected_tasks(args.task)
    output = _path(args.output_dir)
    checkpoint = _path(args.checkpoint) if args.checkpoint else None
    assets = _path(args.asset_root) if args.asset_root else None
    goals = _path(args.goal_root) if args.goal_root else ROOT / "assets" / "goals"
    plan = {
        "task": args.task, "tasks": list(tasks), "model_label": args.label,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "asset_root": str(assets) if assets else None,
        "goal_root": str(goals), "output_dir": str(output),
        "workers": args.workers, "expected_episodes": {}, "jobs": [],
        "summarize": bool(args.summarize or args.summarize_only),
    }
    if args.summarize_only:
        return plan
    for task in tasks:
        cfg = _config(task)
        env = _environment(task, cfg, args.render)
        if task == "mine":
            seeds = args.seeds if args.seeds is not None else cfg["policy_seeds"]
            scenes = [f"shared16_w{world}_{target}" for world in cfg["worlds"]
                      for target in cfg["targets"]]
            command = _command(
                "mine", "evaluate", "--root", output / "mine", "--models-json",
                output / "mine_models.json", "--scene-manifest",
                output / "mine_scenes.json", "--goal-lib", goals / "mine",
                "--gmode", cfg["goal_mode"], "--workers", args.workers,
                "--post", cfg["post"], "--success-quota", cfg["success_quota"],
                "--require-model-count", 1,
            )
            for seed in seeds:
                _option(command, "--policy-rollout-seed", seed)
            _option(command, "--python", sys.executable)
            for scene in scenes:
                _option(command, "--scene-id", scene)
            plan["jobs"].append({
                "task": task, "kind": "mine", "command": command,
                "environment": env, "output_root": str(output / "mine"),
                "expected_episodes": len(scenes) * len(seeds),
                "scene_ids": scenes, "policy_seeds": list(seeds),
                "source_manifest": str(assets / "mine" / "scenes.json"),
                "prepared_manifest": str(output / "mine_scenes.json"),
                "models_json": str(output / "mine_models.json"),
            })
        elif task == "hunt":
            seeds = args.seeds if args.seeds is not None else cfg["policy_seeds"]
            for split, targets in cfg["targets"].items():
                for seed in seeds:
                    root = output / f"hunt_{split}_seed{seed}"
                    command = _command(
                        "hunt", "--out-root", root, "--checkpoint", checkpoint,
                        "--model-label", args.label, "--goal-source-root",
                        goals / f"hunt_{split}", "--snapshot-bank-root",
                        assets / "hunt" / f"{split}_bank", "--layouts", cfg["layouts"],
                        "--targets", *targets, "--benchmark-seed", cfg["benchmark_seed"],
                        "--world-seed-base", cfg["world_seed_base"],
                        "--scene-seed-base", cfg["scene_seed_base"],
                        "--policy-seed-base", seed, "--budget", cfg["budget"],
                        "--post", cfg["post"], "--cfg-coef", "0.0",
                        "--workers", args.workers,
                    )
                    plan["jobs"].append({
                        "task": task, "kind": "hunt", "command": command,
                        "environment": env, "output_root": str(root),
                        "resume_if_output_exists": True,
                        "expected_episodes": len(targets) * cfg["layouts"],
                    })
        elif task == "place":
            for namespace in cfg["policy_seed_namespaces"]:
                root = output / f"place_{namespace}"
                command = _command(
                    "place", "--out-root", root, "--checkpoint", checkpoint,
                    "--model-label", args.label, "--goal-source-root", goals / "place",
                    "--snapshot-bank-root", assets / "place" / "bank",
                    "--targets", *cfg["targets"], "--layout-indices", *cfg["layout_indices"],
                    "--start-indices", *cfg["start_indices"],
                )
                if namespace != "place_policy":
                    _option(command, "--policy-seed-namespace", namespace)
                command.extend(map(str, ["--cfg-coef", "0.0", "--workers", args.workers,
                                        "--budget", cfg["budget"], "--post", cfg["post"]]))
                plan["jobs"].append({
                    "task": task, "kind": "place", "command": command,
                    "environment": env, "output_root": str(root),
                    "resume_if_output_exists": True, "max_passes": cfg["max_passes"],
                    "expected_episodes": len(cfg["targets"]) * len(cfg["layout_indices"])
                    * len(cfg["start_indices"]),
                })
        else:
            seeds = args.seeds if args.seeds is not None else list(range(
                cfg["policy_seed_start"], cfg["policy_seed_stop"]))
            for seed in seeds:
                root = output / "chains" / args.label / task / f"seed{seed:02d}"
                command = _command(
                    task, "--out", root / "attempt1", "--checkpoint", checkpoint,
                    "--model-label", args.label, "--policy-backend", "in_tree_rocket2",
                    "--policy-rollout-seed", seed, "--cfg-coef", "0.0",
                )
                if task == "dpx":
                    command.extend(map(str, [
                        "--goal-root", goals / "dpx", "--scene-version", cfg["scene_version"],
                        "--oak-budget", cfg["oak_budget"], "--diamond-budget", cfg["diamond_budget"],
                        "--table-budget", cfg["table_budget"], "--clean-world",
                        "--start-pose", cfg["start_pose"],
                    ]))
                elif task == "ccfw":
                    for flag, goal in (("coal", "coal_ore_mine"), ("cow", "cow"),
                                       ("furnace", "furnace_open"), ("wolf", "wolf")):
                        _option(command, f"--{flag}-goal", goals / "ccfw" / goal)
                    command.extend(map(str, [
                        "--stage-budgets", *cfg["stage_budgets"], "--wolf-budget", cfg["wolf_budget"],
                        "--start-yaw-offset", cfg["start_yaw_offset"], "--completion-mode",
                        cfg["completion_mode"],
                    ]))
                else:
                    command.extend(map(str, [
                        "--stage-budget", cfg["stage_budget"], "--scoop-budget", cfg["scoop_budget"],
                        "--post", cfg["post"], "--portal-marker-kind", cfg["portal_marker_kind"],
                    ]))
                    for flag, goal in (("water-goal-source", "water"), ("lava-goal-source", "lava"),
                                       ("obsidian-goal-source", "obsidian"),
                                       ("portal-obsidian-goal-source", "portal_frame"),
                                       ("portal-goal-root", "portal_markers")):
                        _option(command, f"--{flag}", goals / "wlo" / goal)
                plan["jobs"].append({
                    "task": task, "kind": "chain", "command": command,
                    "environment": env, "output_root": str(root), "policy_seed": seed,
                    "expected_episodes": 1, "max_attempts": cfg["max_attempts"],
                    "retry_only_infrastructure_errors": True,
                    "skip_if_result_exists": True,
                })
        plan["expected_episodes"][task] = sum(
            job["expected_episodes"] for job in plan["jobs"] if job["task"] == task)
    return plan


def _subprocess_env(overrides=None):
    env = os.environ.copy()
    env.update(overrides or {})
    paths = [str(ROOT / "src"), str(ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if env.get("CONDA_PREFIX"):
        env.setdefault("JAVA_HOME", env["CONDA_PREFIX"])
        env["PATH"] = str(Path(env["CONDA_PREFIX"]) / "bin") + os.pathsep + env.get("PATH", "")
    return env


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _prepare_mine(job, plan):
    source = Path(job["source_manifest"])
    manifest = json.loads(source.read_text())
    for scene in manifest["scenes"]:
        for key in ("source_snapshot", "snapshot"):
            snapshot = scene.get(key)
            if isinstance(snapshot, dict) and snapshot.get("bundle"):
                bundle = Path(snapshot["bundle"])
                if not bundle.is_absolute():
                    snapshot["bundle"] = str((source.parent / bundle).resolve())
    _write_json(job["prepared_manifest"], manifest)
    _write_json(job["models_json"], [{
        "name": plan["model_label"], "ckpt": plan["checkpoint"],
        "policy_backend": "rocket2_static_goal", "cfg_coef": 0.0,
    }])


def _mine_complete(job, label):
    root = Path(job["output_root"]) / "receipts" / label
    completed = 0
    for scene in job["scene_ids"]:
        for seed in job["policy_seeds"]:
            path = root / f"{scene}__pseed{seed:08d}.json"
            if path.is_file():
                receipt = json.loads(path.read_text())
                completed += receipt.get("status") == "complete" and isinstance(
                    receipt.get("result"), dict)
    return completed == job["expected_episodes"]


def _run_short(job, plan):
    if job["kind"] == "mine":
        _prepare_mine(job, plan)
    root = Path(job["output_root"])
    env = _subprocess_env(job["environment"])
    for attempt in range(1, job.get("max_passes", 1) + 1):
        command = list(job["command"])
        if job.get("resume_if_output_exists") and root.is_dir():
            command.append("--resume")
        result = subprocess.run(command, cwd=ROOT, env=env)
        if job["kind"] == "place":
            done = sum(path.is_file() for path in (root / "episodes").glob("*/result.json"))
            print(f"place {root.name}: {done}/{job['expected_episodes']} episodes with a result "
                  f"(pass {attempt}/{job['max_passes']})", flush=True)
            if result.returncode == 0 and done >= job["expected_episodes"]:
                return 0
            if result.returncode not in (0, 2):
                return result.returncode
        else:
            if result.returncode:
                return result.returncode
            if job["kind"] == "mine" and not _mine_complete(job, plan["model_label"]):
                print("Mine evaluation has missing or failed rollout receipts.", file=sys.stderr)
                return 2
            return 0
    return 2


def _has_chain_result(root):
    return (root / "result.json").is_file() or any(
        path.is_file() for path in root.glob("*/result.json"))


def _archive_attempt(path):
    if not path.exists():
        return
    candidate = path.with_name(f"{path.name}.stale_{time.strftime('%H%M%S')}")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.stale_{time.strftime('%H%M%S')}_{index}")
        index += 1
    path.rename(candidate)


def _run_chain(job, plan):
    root = Path(job["output_root"])
    if _has_chain_result(root):
        print(f"SKIP {job['task']} seed={job['policy_seed']}: result exists", flush=True)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    logs = Path(plan["output_dir"]) / "chains" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tag = f"{plan['model_label']}_{job['task']}_seed{job['policy_seed']:02d}"
    env = _subprocess_env(job["environment"])
    for attempt in range(1, job["max_attempts"] + 1):
        out = root / f"attempt{attempt}"
        _archive_attempt(out)
        command = list(job["command"])
        command[command.index("--out") + 1] = str(out)
        log_path = logs / f"{tag}_attempt{attempt}.log"
        print(f"CHAIN {job['task']} seed={job['policy_seed']} attempt={attempt} out={out}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (logs / f"{tag}.exit").write_text(f"{result.returncode}\n")
        if result.returncode == 0:
            if not (out / "result.json").is_file():
                print(f"Missing chain result: {out / 'result.json'}", file=sys.stderr)
                return 2
            return 0
        with log_path.open(errors="replace") as log:
            infrastructure_failure = any(INFRA_FAILURE.search(line) for line in log)
        if not infrastructure_failure:
            return result.returncode
    return 1


def summarize(plan):
    output = Path(plan["output_dir"])
    code = 0
    for task in CHAIN_TASKS:
        root = output / "chains" / plan["model_label"] / task
        if root.is_dir():
            command = _command("summarize_chains", root, task, "--json",
                               output / f"{plan['model_label']}_{task}_stage_cumulative.json")
            code = subprocess.run(command, cwd=ROOT, env=_subprocess_env()).returncode or code
    command = _command("summarize_short", "--root", output)
    return subprocess.run(command, cwd=ROOT, env=_subprocess_env()).returncode or code


def execute(plan):
    output = Path(plan["output_dir"])
    if plan["jobs"]:
        if not Path(plan["checkpoint"]).exists():
            raise ValueError(f"checkpoint does not exist: {plan['checkpoint']}")
        if any(job["task"] in SHORT_TASKS for job in plan["jobs"]):
            if not Path(plan["asset_root"]).is_dir():
                raise ValueError(f"asset root does not exist: {plan['asset_root']}")
        if any(job["kind"] == "chain" for job in plan["jobs"]) and not os.environ.get("DISPLAY"):
            raise ValueError("chain evaluation needs DISPLAY; run under xvfb-run -a for headless rendering")
        output.mkdir(parents=True, exist_ok=True)
    code = 0
    for task in plan["tasks"]:
        jobs = [job for job in plan["jobs"] if job["task"] == task]
        if task in CHAIN_TASKS and jobs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=plan["workers"]) as executor:
                futures = [executor.submit(_run_chain, job, plan) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    code = future.result() or code
        else:
            for job in jobs:
                result = _run_short(job, plan)
                code = result or code
                if result:
                    break
    if plan["summarize"]:
        code = summarize(plan) or code
    return 1 if code else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train-free Attacca benchmark evaluation.")
    parser.add_argument("--task", choices=TASKS + ("short", "chains", "all"), default="all")
    parser.add_argument("--checkpoint", help="Attacca inference directory or trained .ckpt")
    parser.add_argument("--asset-root", help="extracted attacca_eval_assets_release_v1 directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="policy seeds for Mine, Hunt or chains; not applicable to Place")
    parser.add_argument("--label", default="attacca")
    parser.add_argument("--goal-root", help="defaults to assets/goals in this repository")
    parser.add_argument("--render", choices=("software", "gpu"),
                        help="otherwise use MINESTUDIO_GPU_RENDER or task-specific defaults")
    parser.add_argument("--dry-run", action="store_true", help="print the plan JSON without creating files")
    summary = parser.add_mutually_exclusive_group()
    summary.add_argument("--summarize", action="store_true", default=True,
                         help="summarize results after evaluation (default)")
    summary.add_argument("--no-summarize", dest="summarize", action="store_false",
                         help="skip post-evaluation result summaries")
    parser.add_argument("--summarize-only", action="store_true", help="summarize existing results without evaluation")
    args = parser.parse_args(argv)
    tasks = _selected_tasks(args.task)
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be in [1,8]")
    if not args.label or args.label in (".", "..") or any(value in args.label for value in ("/", "\\", "\x00")):
        parser.error("--label must be a nonempty directory name")
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
            parser.error("--seeds must be unique integers in [0, 2**32)")
        if "place" in tasks:
            parser.error("Place uses two fixed seed namespaces; --seeds cannot be used with place, short or all")
    if not args.summarize_only:
        if not args.checkpoint:
            parser.error("--checkpoint is required for evaluation")
        if any(task in SHORT_TASKS for task in tasks) and not args.asset_root:
            parser.error("--asset-root is required for Mine, Hunt and Place")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        return execute(plan)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
