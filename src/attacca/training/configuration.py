"""Configuration and read-only dataset checks for the training entry point."""

import json
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from attacca.paths import ROOT


DEFAULT_CONFIG = ROOT / "configs" / "attacca.yaml"
DATA_PACKS = (
    "mine_ore_822",
    "hunt_open_100",
    "hunt_fenced_98",
    "place_cave_140",
)
OUTPUT_ENV = "ATTACCA_TRAIN_OUTPUT"


def _positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def validate_data_root(data_root):
    import lmdb

    root = Path(data_root).expanduser().resolve()
    packs = [root / "data" / name for name in DATA_PACKS]
    for pack in packs:
        database = pack / "data.mdb"
        if not database.is_file() or database.stat().st_size == 0:
            raise ValueError(f"missing or empty LMDB: {database}")
        try:
            with lmdb.open(
                str(pack), readonly=True, lock=False, create=False,
                readahead=False, meminit=False,
            ) as env:
                with env.begin(write=False) as txn:
                    schema_data = txn.get(b"__schema__")
                    if not schema_data:
                        raise ValueError("missing __schema__")
                    schema = json.loads(schema_data.decode("utf-8"))
                    if not isinstance(schema, dict) or schema.get("version") != 13 or schema.get("window") != 128:
                        raise ValueError("expected an Attacca schema-13 pack with window=128")
                    if not txn.get(b"__episodes__"):
                        raise ValueError("missing __episodes__")
        except (lmdb.Error, ValueError, UnicodeError) as exc:
            raise ValueError(f"cannot read dataset pack {pack}: {exc}") from exc
    return root, packs


def load_training_config(
    config_path, data_root, output_dir, *, overrides=(), devices=None,
    batch_size=None, num_workers=None, seed=None,
):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"configuration file does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise ValueError("the training configuration must be a mapping")
    if "defaults" in config or "hydra" in config:
        raise ValueError("--config must contain a complete configuration, not Hydra defaults")
    for override in overrides:
        if "=" not in override or not override.split("=", 1)[0]:
            raise ValueError(f"--set requires key=value, got {override!r}")
        key = override.split("=", 1)[0]
        if key.startswith(("+", "~")):
            raise ValueError("--set uses plain dotted keys, without Hydra prefixes")
        if key.split(".", 1)[0] not in config:
            raise ValueError(f"unknown configuration key: {key}")
    for override in overrides:
        key, raw_value = override.split("=", 1)
        changes = OmegaConf.from_dotlist([override])
        if isinstance(OmegaConf.select(config, key), str) and isinstance(OmegaConf.select(changes, key), bool):
            OmegaConf.update(changes, key, raw_value.strip())
        config = OmegaConf.merge(config, changes)
    for key, value in (
        ("devices", devices), ("batch_size", batch_size),
        ("num_workers", num_workers), ("seed", seed),
    ):
        if value is not None:
            config[key] = value
    OmegaConf.resolve(config)
    required = set(OmegaConf.load(DEFAULT_CONFIG))
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"incomplete training configuration; missing: {', '.join(missing)}")
    if config.ckpt_path is not None:
        raise ValueError("checkpoint resume is not supported by this training entry point")
    for key in ("batch_size", "prefetch_factor", "save_freq", "max_steps", "accumulate_grad_batches"):
        _positive_integer(config[key], key)
    for key in ("num_workers", "seed", "synthetic_seed", "synthetic_episode_split_seed"):
        _positive_integer(config[key], key, allow_zero=True)
    if isinstance(config.devices, int):
        _positive_integer(config.devices, "devices")
    else:
        selected_devices = OmegaConf.to_container(config.devices)
        if not isinstance(selected_devices, list) or not selected_devices:
            raise ValueError("devices must be a positive count or a list of device IDs")
        for device in selected_devices:
            _positive_integer(device, "device ID", allow_zero=True)
        if len(set(selected_devices)) != len(selected_devices):
            raise ValueError("devices must not contain duplicate device IDs")
    if config.win_len != 128 or config.model.timesteps != 128:
        raise ValueError("the released dataset and recipe require win_len=model.timesteps=128")
    if not config.synthetic_only:
        raise ValueError("the released recipe requires synthetic_only=true")
    if not isinstance(config.init_from, str) or not config.init_from:
        raise ValueError("init_from must identify a pretrained ROCKET-2 model")
    init_path = Path(config.init_from).expanduser()
    if init_path.exists():
        config.init_from = str(init_path.resolve())
    root, packs = validate_data_root(data_root)
    output = Path(output_dir).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("--output-dir must be outside the dataset directory")
    if output == ROOT or output in ROOT.parents:
        raise ValueError("--output-dir must not be the repository or one of its parents")
    config.synthetic_lmdb = [str(pack) for pack in packs]
    config.run_tag = config.run_tag or output.name
    if not isinstance(config.run_tag, str) or any(c in config.run_tag for c in ("/", "\\")):
        raise ValueError("run_tag must be a name, not a path")
    return config, root, output


def is_distributed_child(output_dir):
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise ValueError("LOCAL_RANK must be an integer") from exc
    inherited_output = os.environ.get(OUTPUT_ENV)
    return (
        local_rank > 0 and inherited_output is not None
        and Path(inherited_output).resolve() == Path(output_dir).resolve()
    )


def validate_output_directory(output_dir):
    output = Path(output_dir)
    child = is_distributed_child(output)
    if child:
        if not output.is_dir() or not (output / "config.yaml").is_file():
            raise ValueError("distributed child could not find its parent training output")
    elif output.exists():
        raise ValueError(f"output directory already exists; choose a new --output-dir: {output}")
    return child


def prepare_output_directory(output_dir, config):
    output = Path(output_dir)
    child = validate_output_directory(output)
    if not child:
        output.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(config, output / "config.yaml", resolve=True)
    os.environ[OUTPUT_ENV] = str(output)
    os.environ["MINESTUDIO_SAVE_DIR"] = str(output)
