# Data

```text
willsuh/attacca                       model
  config.json
  model.safetensors
willsuh/attacca-dataset               dataset (--repo-type dataset)
  attacca_dataset_release_v1/         training data (--data-root)
    data/<pack>/data.mdb
    metadata/packs.json
    metadata/episodes.jsonl
    examples/read_lmdb.py
  attacca_eval_assets_release_v1.tar  evaluation worlds (--asset-root after extraction)
```

## Training data

Four LMDB packs with 1,160 human demonstrations and 284,961 frames.

| Pack | Episodes | Task |
| --- | ---: | --- |
| `mine_ore_822` | 822 | Mine |
| `hunt_open_100` | 100 | Hunt, open field |
| `hunt_fenced_98` | 98 | Hunt, fenced arena |
| `place_cave_140` | 140 | Place |

`examples/read_lmdb.py` shows how to read a pack. Frame records are pickled, so only load the data from this release.

## Evaluation assets

`attacca_eval_assets_release_v1.tar` holds the world snapshots for Mine, Hunt and Place. Extract it with `tar -xf` and keep its empty directories, which are part of the world hashes.
The Diamond Pickaxe, Wolf Feeding and Nether Portal chains build their worlds directly and do not need these assets.
