#!/usr/bin/env python3
"""Stage-cumulative success of a long-horizon chain from per-seed result.json files (usage: summarize_chains.py RUN_ROOT {dpx|ccfw|wlo|stage1,stage2,...} [--json OUT])."""
import argparse, json, re, pathlib

CHAIN_STAGES = {
    "dpx": "oak_log_mine,diamond_ore_mine3,crafting_table_open",
    "ccfw": "coal_ore_mine,cow_hunt,furnace_find,wolf_feed",
    "wlo": "scoop,pour,mine,portal,ignite",
}


def load_seed(sd):
    cands = []
    if (sd / "result.json").exists():
        cands.append(("", sd / "result.json"))
    for a in sorted(sd.glob("attempt*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        if (a / "result.json").exists():
            cands.append((a.name, a / "result.json"))
    if not cands:
        return None, None, 0
    tag, p = cands[-1]
    return tag, json.load(open(p)), len([c for c in cands if c[0]])


def main():
    parser = argparse.ArgumentParser(
        description="Stage-cumulative success of a long-horizon chain from per-seed result.json files.")
    parser.add_argument("run_root", help="directory with one seedNN/ directory per rollout seed")
    parser.add_argument("chain", help="dpx, ccfw, wlo, or a comma-separated stage list")
    parser.add_argument("--json", help="also write the per-seed table to this JSON file")
    args = parser.parse_args()
    root = pathlib.Path(args.run_root); stages = CHAIN_STAGES.get(args.chain, args.chain).split(",")
    out = args.json
    seeds = sorted([p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"seed\d+", p.name)],
                   key=lambda p: int(p.name[4:]))
    cum = [0] * len(stages); rows = []; missing = []
    shas = set(); labels = set()
    for sd in seeds:
        tag, r, natt = load_seed(sd)
        if r is None:
            missing.append(sd.name); continue
        shas.add(r.get("checkpoint_sha256") or (r.get("model_contract") or {}).get("checkpoint_sha256"))
        labels.add(r.get("model_label"))
        res = r.get("results", {})
        ok = True; per = []; reached = 0
        ss = r.get("stage_success") or {}
        for i, s in enumerate(stages):
            v = res.get(s) or {}
            succ = int(v.get("success") or 0) == 1
            if s in ss and int(ss[s] or 0) != int(succ):
                print(f"WARN {sd.name} {s}: results.success={int(succ)} stage_success={ss[s]}")
            per.append(dict(stage=s, success=int(succ), steps=v.get("steps"),
                            success_step=v.get("success_step"), state=v.get("state")))
            ok = ok and succ
            if ok:
                cum[i] += 1; reached = i + 1
        rows.append(dict(seed=int(sd.name[4:]), attempt=tag, n_attempts=natt,
                         stages_passed=reached, chain=int(reached == len(stages)),
                         per_stage=per, state=r.get("state")))
    n = len(rows)
    print(f"root={root}\nn_seeds_with_result={n} missing={missing}")
    print("model_label=", labels, " ckpt_sha256=", {s[:8] if s else s for s in shas})
    print("stage-cumulative counts:", dict(zip(stages, cum)))
    print("stage-cumulative rates :", [round(c / n, 2) if n else None for c in cum])
    multi = [r["seed"] for r in rows if r["n_attempts"] > 1]
    print("seeds restarted after a simulator launch error:", multi)
    if out:
        json.dump(dict(root=str(root), stages=stages, n=n, missing=missing,
                       cum=cum, rows=rows, ckpt_sha256=sorted(s for s in shas if s),
                       labels=sorted(l for l in labels if l)), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
