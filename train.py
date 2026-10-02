#!/usr/bin/env python
"""Train Attacca on the released dataset."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="downloaded attacca_dataset_release_v1 directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new directory for configuration, logs and checkpoints")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "attacca.yaml",
                        help="complete training YAML (default: configs/attacca.yaml)")
    parser.add_argument("--devices", type=int, help="number of training devices (default: 3)")
    parser.add_argument("--batch-size", type=int, help="batch size per device (default: 4)")
    parser.add_argument("--num-workers", type=int, help="data workers per device (default: 6)")
    parser.add_argument("--seed", type=int, help="training seed (default: 2026)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a configuration key; may be repeated")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate data/configuration and print the recipe without training or writes")
    return parser


def main(argv=None):
    parser = build_parser()
    options = parser.parse_args(argv)
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException
    from attacca.training.configuration import (
        load_training_config, prepare_output_directory, validate_output_directory,
    )

    try:
        config, data_root, output = load_training_config(
            options.config, options.data_root, options.output_dir,
            overrides=options.overrides, devices=options.devices,
            batch_size=options.batch_size, num_workers=options.num_workers, seed=options.seed,
        )
        validate_output_directory(output)
    except (OSError, TypeError, ValueError, OmegaConfBaseException) as exc:
        parser.error(str(exc))
    if options.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        print(f"Data root: {data_root}")
        print(f"Output directory: {output}")
        print("Validation passed. No training was started and no output directory was created.")
        return 0
    config_path = options.config.expanduser().resolve()
    relaunch = [
        str(Path(__file__).resolve()), "--data-root", str(data_root),
        "--output-dir", str(output), "--config", str(config_path),
    ]
    for name in ("devices", "batch_size", "num_workers", "seed"):
        value = getattr(options, name)
        if value is not None:
            relaunch += ["--" + name.replace("_", "-"), str(value)]
    for override in options.overrides:
        relaunch += ["--set", override]
    sys.argv = relaunch
    try:
        prepare_output_directory(output, config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    from attacca.training.trainer import train

    train(config, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
