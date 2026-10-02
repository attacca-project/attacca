# Evaluation

| `--task` | Runs | Needs `--asset-root` |
| --- | --- | --- |
| `mine`, `hunt`, `place` | One short-horizon task | Yes |
| `short` | Mine, Hunt and Place | Yes |
| `dpx`, `ccfw`, `wlo` | Diamond Pickaxe, Wolf Feeding or Nether Portal | No |
| `chains` | All three chains | No |
| `all` (default) | All six tasks | Yes |

```bash
MINESTUDIO_GPU_RENDER=0 xvfb-run -a python eval.py --task all --checkpoint checkpoints/attacca \
  --asset-root attacca_eval_assets_release_v1 --output-dir results/attacca --workers 1
```

Each short-horizon task runs 200 episodes (140 seen and 60 held-out classes), and each chain runs 50 episodes.
`--workers` (1 to 8) sets how many episodes run in parallel, and `--dry-run` prints the commands without running them.

Results are written to the output directory: `short_horizon_summary.json` for Mine, Hunt and Place, and `attacca_<task>_stage_cumulative.json` for each chain.
Rerunning with the same `--output-dir` continues unfinished episodes.

To check the downloaded files before evaluating:

```bash
python tools/check_release.py --mode eval --checkpoint checkpoints/attacca \
  --asset-root attacca_eval_assets_release_v1 --verify-hashes
```
