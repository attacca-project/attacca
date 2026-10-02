#!/usr/bin/env python3
"""Summarize Mine, Hunt and Place benchmark results into clean success and precision per target split (ID, OOD, all)."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

TASKS = ("mine", "hunt", "place")
ROSTERS = {
    "mine": {
        "id": ("coal_ore", "iron_ore", "gold_ore", "lapis_ore", "diamond_ore",
               "emerald_ore", "redstone_ore"),
        "ood": ("brain_coral_block", "fire_coral_block", "horn_coral_block"),
    },
    "hunt": {
        "id": ("cow", "pig", "chicken", "white_sheep", "gray_sheep",
               "light_gray_sheep", "brown_sheep"),
        "ood": ("polar_bear", "donkey", "panda"),
    },
    "place": {
        "id": ("oak_log", "birch_log", "pumpkin", "melon", "mossy_cobblestone",
               "hay_block", "bookshelf"),
        "ood": ("red_sandstone", "dead_brain_coral_block", "green_terracotta"),
    },
}
SPLITS = ("id", "ood", "all")
EPISODE_FIELDS = ("task", "model", "target", "split", "episode", "clean_success",
                  "target_interactions", "wrong_interactions")


def _split(task, target):
    for split in ("id", "ood"):
        if target in ROSTERS[task][split]:
            return split
    return "other"


def _episode(task, model, target, episode, clean_success,
             target_interactions, wrong_interactions):
    return {
        "task": task, "model": str(model), "target": str(target),
        "split": _split(task, str(target)), "episode": episode,
        "clean_success": int(clean_success),
        "target_interactions": int(target_interactions),
        "wrong_interactions": int(wrong_interactions),
    }


def mine_episodes(root):
    rows, incomplete = [], []
    for path in sorted((root / "mine" / "receipts").glob("*/*.json")):
        receipt = json.loads(path.read_text())
        result = receipt.get("result")
        if receipt.get("status") != "complete" or not isinstance(result, dict):
            incomplete.append(str(path))
            continue
        scene_id = str(receipt.get("scene_id") or "")
        target = result.get("target_cls") or re.sub(r"^[^_]+_w\d+_", "", scene_id)
        wrong = result.get("wrong_ore_blocks") or {}
        wrong_count = (sum(int(value) for value in wrong.values())
                       if isinstance(wrong, dict) else int(wrong))
        breaks = result.get("target_breaks")
        target_count = len(breaks) if isinstance(breaks, list) else int(breaks or 0)
        rows.append(_episode(
            "mine", path.parent.name, target,
            f"mine/{path.parent.name}/{receipt.get('rollout_id') or path.stem}",
            bool(result.get("succ")) and wrong_count == 0, target_count, wrong_count))
    return rows, incomplete


def hunt_episodes(root):
    rows, incomplete = [], []
    for run in sorted(path for path in root.glob("hunt_*") if path.is_dir()):
        for task_root in sorted(run.glob("rollouts/*/*")):
            if not task_root.is_dir():
                continue
            path = task_root / "task_result.json"
            if not path.is_file():
                incomplete.append(str(task_root))
                continue
            result = json.loads(path.read_text())
            killed = int(bool(result.get("target_kill_success")))
            rows.append(_episode(
                "hunt", result.get("model_label"), result.get("target") or task_root.name,
                f"{run.name}/{task_root.parent.name}/{task_root.name}",
                bool(result.get("strict_success")), killed,
                int(result.get("wrong_kill") or 0)))
    return rows, incomplete


def place_episodes(root):
    rows, incomplete = [], []
    for run in sorted(path for path in root.glob("place_*") if path.is_dir()):
        episodes = run / "episodes"
        if not episodes.is_dir():
            continue
        for episode in sorted(path for path in episodes.iterdir() if path.is_dir()):
            path = episode / "result.json"
            if not path.is_file():
                incomplete.append(str(episode))
                continue
            result = json.loads(path.read_text())
            target = ((result.get("task") or {}).get("target")
                      or re.sub(r"_w\d+_s\d+$", "", episode.name))
            wrong_count = int(result.get("off_target_placement_total") or 0)
            rows.append(_episode(
                "place", result.get("model_label"), target,
                f"{run.name}/{episode.name}",
                bool(result.get("strict_success")) and wrong_count == 0,
                int(result.get("completed_target_count") or 0), wrong_count))
    return rows, incomplete


def _rate(rows, key):
    return sum(row[key] for row in rows) / len(rows) if rows else None


def cell(rows, classes):
    interacted = [row for row in rows
                  if row["target_interactions"] + row["wrong_interactions"] > 0]
    precision = (sum(row["target_interactions"]
                     / (row["target_interactions"] + row["wrong_interactions"])
                     for row in interacted) / len(interacted)
                 if interacted else None)
    present = {row["target"] for row in rows}
    return {
        "n": len(rows),
        "clean_success": _rate(rows, "clean_success"),
        "precision": precision,
        "precision_n": len(interacted),
        "missing_classes": [name for name in classes if name not in present],
    }


def summarize(rows):
    models = {}
    for model in sorted({row["model"] for row in rows}):
        tasks = {}
        for task in TASKS:
            task_rows = [row for row in rows
                         if row["model"] == model and row["task"] == task]
            if not task_rows:
                continue
            roster = ROSTERS[task]
            classes = {"id": roster["id"], "ood": roster["ood"],
                       "all": roster["id"] + roster["ood"]}
            tasks[task] = {
                split: cell([row for row in task_rows if row["target"] in classes[split]],
                            classes[split])
                for split in SPLITS}
            tasks[task]["outside_roster"] = sum(
                row["split"] == "other" for row in task_rows)
        models[model] = tasks
    return models


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


def print_table(models, incomplete):
    header = ("model", "task", "split", "n", "clean_success", "precision")
    lines = [header]
    for model, tasks in models.items():
        for task, cells in tasks.items():
            for split in SPLITS:
                c = cells[split]
                lines.append((model, task, split.upper() if split != "all" else "all",
                              str(c["n"]), _fmt(c["clean_success"]), _fmt(c["precision"])))
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    for line in lines:
        print("  ".join(value.ljust(width) for value, width in zip(line, widths)).rstrip())
    for model, tasks in models.items():
        for task, cells in tasks.items():
            missing = cells["all"]["missing_classes"]
            if missing:
                print(f"{model} {task}: no episodes for {', '.join(missing)}")
            if cells["outside_roster"]:
                print(f"{model} {task}: {cells['outside_roster']} episodes outside the class roster")
    for task, paths in incomplete.items():
        if paths:
            print(f"{task}: {len(paths)} episodes without a result")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Short-horizon metrics per task (Mine, Hunt, Place) and target split "
            "(ID, OOD, all). clean success: Mine target block broken with no block of "
            "another roster class mined; Hunt target killed with no other animal killed, "
            "held item kept and agent inside the pen; Place all target placements "
            "completed with no off-target placement. precision: per-episode "
            "target / (target + wrong) interactions, averaged over episodes with at "
            "least one interaction."))
    parser.add_argument(
        "--root", nargs="+", required=True, type=Path,
        help=("evaluation root(s) written by eval.py; "
              "the summary and episode table are written to the first root"))
    args = parser.parse_args(argv)

    rows, incomplete = [], {task: [] for task in TASKS}
    for root in args.root:
        root = root.expanduser().resolve()
        for task, reader in (("mine", mine_episodes), ("hunt", hunt_episodes),
                             ("place", place_episodes)):
            task_rows, task_incomplete = reader(root)
            rows.extend(task_rows)
            incomplete[task].extend(task_incomplete)
    if not rows:
        print("no Mine, Hunt or Place episode results under "
              + ", ".join(str(root) for root in args.root))
        return 0

    models = summarize(rows)
    first = args.root[0].expanduser().resolve()
    json_path = first / "short_horizon_summary.json"
    tsv_path = first / "short_horizon_episodes.tsv"
    first.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "roots": [str(root.expanduser().resolve()) for root in args.root],
        "rosters": {task: {split: list(names) for split, names in roster.items()}
                    for task, roster in ROSTERS.items()},
        "incomplete": incomplete,
        "models": models,
    }, indent=2) + "\n")
    with tsv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print_table(models, incomplete)
    print(f"summary: {json_path}")
    print(f"episodes: {tsv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
