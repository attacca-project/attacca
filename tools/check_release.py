import argparse
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys


sys.dont_write_bytecode = True
CODE_ROOT = Path(__file__).resolve().parents[1]
PACKS = ('mine_ore_822', 'hunt_open_100', 'hunt_fenced_98', 'place_cave_140')
GOAL_COUNTS = {'mine': 10, 'hunt_id': 7, 'hunt_ood': 3, 'place': 10,
               'dpx': 3, 'ccfw': 4, 'wlo': 6}
HASH_FIELDS = ('payload_sha256', 'world_sha256', 'world_archive_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty_file(path):
    require(path.is_file() and path.stat().st_size > 0,
            f'Missing or empty file: {path}')
    return path


def read_json(path):
    nonempty_file(path)
    value = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict), f'Expected a JSON object: {path}')
    return value


def check_checkpoint(value):
    require(bool(value), 'Pass --checkpoint (or set CKPT).')
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        require(path.suffix != '.safetensors',
                'Pass the model directory containing config.json and model.safetensors, not the weights file.')
        nonempty_file(path)
        print('PASS checkpoint: nonempty checkpoint file present.', flush=True)
        return
    read_json(path / 'config.json')
    weights_path = nonempty_file(path / 'model.safetensors')
    sys.path.insert(0, str(CODE_ROOT / 'src'))
    from attacca.models.inference_checkpoint import is_attacca_checkpoint
    from attacca.models.inference_checkpoint import read_attacca_config

    require(is_attacca_checkpoint(path),
            'Not an Attacca model directory: config.json must declare model_type "attacca".')
    read_attacca_config(path)
    from safetensors import safe_open

    with safe_open(weights_path, framework='np') as weights:
        names = weights.keys()
        require(bool(names), 'The safetensors file contains no model tensors.')
        require(not any(name.startswith('mine_policy.') for name in names),
                'Attacca inference tensor keys must not have a mine_policy. prefix.')
        for name in names:
            weights.get_slice(name).get_shape()
    print(f'PASS checkpoint: Attacca configuration and '
          f'safetensors structure ({len(names)} tensors).', flush=True)


def root_path(value, option, variable):
    require(bool(value), f'Pass {option} (or set {variable}).')
    path = Path(value).expanduser().resolve()
    require(path.is_dir(), f'Missing directory: {path}')
    return path


def relative_path(base, reference, root):
    require(isinstance(reference, str) and bool(reference),
            f'Missing path reference in {base}')
    posix = PurePosixPath(reference)
    require(not posix.is_absolute() and not PureWindowsPath(reference).drive
            and '\\' not in reference and '..' not in posix.parts,
            f'Nonportable path reference: {reference!r}')
    path = (base / reference).resolve()
    require(path.is_relative_to(root), f'Path escapes release root: {reference!r}')
    require(path.exists(), f'Missing referenced path: {path}')
    return path


def check_train(value):
    import lmdb

    root = root_path(value, '--data-root', 'DATA_ROOT')
    manifest = read_json(root / 'metadata' / 'packs.json')
    packs = manifest.get('packs')
    require(isinstance(packs, dict), 'metadata/packs.json lacks a packs object')
    for name in PACKS:
        record = packs.get(name)
        require(isinstance(record, dict), f'Missing pack metadata: {name}')
        path = relative_path(root, record.get('path'), root)
        expected = (root / 'data' / name / 'data.mdb').resolve()
        require(path == expected, f'Pack {name} must use data/{name}/data.mdb')
        nonempty_file(path)
        env = lmdb.open(str(path.parent), readonly=True, lock=False, create=False)
        try:
            with env.begin(write=False) as txn:
                raw_schema = txn.get(b'__schema__')
                require(bool(raw_schema), f'{name}: missing __schema__')
                schema = json.loads(raw_schema)
                require(isinstance(schema, dict), f'{name}: invalid __schema__ JSON')
                require(schema.get('version') == 13, f'{name}: expected schema version 13')
                require(schema.get('version') == record.get('schema_version'),
                        f'{name}: LMDB and pack metadata schema versions differ')
                require(bool(txn.get(b'__episodes__')), f'{name}: missing __episodes__')
        finally:
            env.close()
    print('PASS train: 4 LMDB packs, schema 13, episode headers present.', flush=True)


def check_goals():
    from PIL import Image

    count = 0
    for family, expected in GOAL_COUNTS.items():
        directory = CODE_ROOT / 'assets' / 'goals' / family
        goals = {path.parent: path for path in directory.rglob('goal.png')}
        masks = {path.parent: path for path in directory.rglob('mask.png')}
        require(len(goals) == expected, f'{family}: expected {expected} goal images, found {len(goals)}')
        require(goals.keys() == masks.keys(), f'{family}: unpaired goal.png or mask.png')
        for parent, goal in goals.items():
            sizes = []
            for path in (goal, masks[parent]):
                nonempty_file(path)
                with Image.open(path) as image:
                    require(image.format == 'PNG', f'Expected PNG: {path}')
                    image.load()
                    require(min(image.size) > 0, f'Empty image: {path}')
                    sizes.append(image.size)
            require(sizes[0] == sizes[1], f'Goal and mask sizes differ: {parent}')
            count += 1
    print(f'PASS goals: {count} valid RGB/mask pairs in 7 families.', flush=True)


def check_hash_records(record, metadata, label, prefix=''):
    for name in HASH_FIELDS:
        recorded = record.get(prefix + name)
        expected = metadata.get(name)
        require(isinstance(recorded, str) and re.fullmatch(r'[0-9a-fA-F]{64}', recorded),
                f'{label}: missing or invalid {prefix + name}')
        require(isinstance(expected, str) and recorded.lower() == expected.lower(),
                f'{label}: {prefix + name} differs from snapshot.json')


def check_assets(value, verify_hashes):
    root = root_path(value, '--asset-root', 'ASSET_ROOT')
    bundles = {}

    def register(manifest_path, record, label):
        path = relative_path(manifest_path.parent, record.get('bundle'), root)
        require(path.is_dir(), f'Bundle is not a directory: {path}')
        if path not in bundles:
            metadata = read_json(relative_path(path, 'snapshot.json', root))
            require(metadata.get('contract') == 'xbench-staged-world-snapshot/v3',
                    f'Unsupported snapshot contract: {path}')
            require(metadata.get('world_dir') == 'world', f'Invalid world_dir: {path}')
            require(metadata.get('world_archive') == 'world.zip', f'Invalid world_archive: {path}')
            nonempty_file(relative_path(path, 'world/level.dat', root))
            nonempty_file(relative_path(path, 'world.zip', root))
            bundles[path] = metadata
        metadata = bundles[path]
        check_hash_records(record, metadata, label)
        return path, metadata

    manifest_path = root / 'mine' / 'scenes.json'
    manifest = read_json(manifest_path)
    scenes = manifest.get('scenes')
    require(isinstance(scenes, list) and len(scenes) == 100, 'Mine requires 100 scenes')
    require(manifest.get('expected_scenes') == 100 and manifest.get('physical_scene_count') == 100,
            'Mine manifest scene counts must equal 100')
    scene_ids = set()
    mine_bundles = set()
    for index, scene in enumerate(scenes):
        require(isinstance(scene, dict), f'Invalid Mine scene {index}')
        scene_id = scene.get('scene_id')
        require(isinstance(scene_id, str) and scene_id and scene_id not in scene_ids,
                f'Missing or duplicated Mine scene_id at index {index}')
        scene_ids.add(scene_id)
        source = scene.get('source_snapshot')
        require(isinstance(source, dict), f'Mine {scene_id}: missing source_snapshot')
        bundle, metadata = register(manifest_path, source, f'Mine {scene_id}')
        mine_bundles.add(bundle)
        remap = scene.get('runtime_remap')
        require(isinstance(remap, dict), f'Mine {scene_id}: missing runtime_remap')
        check_hash_records(remap, metadata, f'Mine {scene_id}', prefix='source_')
    require(len(mine_bundles) == 10, f'Mine requires 10 unique bundles, found {len(mine_bundles)}')

    for relative, expected, label, identity_field in (
        ('hunt/id_bank/bank_manifest.json', 70, 'Hunt ID', 'layout_id'),
        ('hunt/ood_bank/bank_manifest.json', 30, 'Hunt OOD', 'layout_id'),
        ('place/bank/bank_manifest.json', 100, 'Place', 'world_index'),
    ):
        manifest_path = root / relative
        manifest = read_json(manifest_path)
        receipts = manifest.get('receipts')
        require(isinstance(receipts, list) and len(receipts) == expected,
                f'{label} requires {expected} receipts')
        require(manifest.get('expected_scenes') == expected
                and manifest.get('accepted_scenes') == expected,
                f'{label}: manifest scene counts must equal {expected}')
        family_bundles = set()
        identities = set()
        for index, record in enumerate(receipts):
            require(isinstance(record, dict), f'{label}: invalid receipt {index}')
            identity = (record.get(identity_field), record.get('target'))
            require(None not in identity and identity not in identities,
                    f'{label}: missing or duplicated scene identity at index {index}')
            identities.add(identity)
            bundle, metadata = register(manifest_path, record, f'{label} receipt {index}')
            family_bundles.add(bundle)
        require(len(family_bundles) == expected,
                f'{label}: expected {expected} unique bundles, found {len(family_bundles)}')
    require(len(bundles) == 210, f'Expected 210 unique bundles, found {len(bundles)}')
    print('PASS assets: Mine 100 scenes/10 bundles; Hunt ID 70; Hunt OOD 30; Place 100.', flush=True)
    if verify_hashes:
        sys.path.insert(0, str(CODE_ROOT / 'src'))
        from attacca.worlds.world_snapshot import load_world_snapshot_bundle

        for index, path in enumerate(sorted(bundles), 1):
            load_world_snapshot_bundle(path)
            if index % 25 == 0 or index == len(bundles):
                print(f'PASS hashes: {index}/{len(bundles)} bundles verified.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Check local Attacca training and evaluation release files.')
    parser.add_argument('--mode', choices=('train', 'eval', 'all'), default='all')
    parser.add_argument('--data-root', default=os.environ.get('DATA_ROOT'),
                        help='attacca_dataset_release_v1 directory (default: $DATA_ROOT)')
    parser.add_argument('--asset-root', default=os.environ.get('ASSET_ROOT'),
                        help='extracted attacca_eval_assets_release_v1 directory (default: $ASSET_ROOT)')
    parser.add_argument('--checkpoint', default=os.environ.get('CKPT'),
                        help='Attacca model directory with config.json and model.safetensors (default: $CKPT)')
    parser.add_argument('--verify-hashes', action='store_true',
                        help='Verify each world tree, archive and snapshot payload; slower than the default file checks.')
    args = parser.parse_args()
    failures = []

    def run(label, operation):
        try:
            operation()
        except Exception as error:
            failures.append(label)
            print(f'FAIL {label}: {error}', file=sys.stderr, flush=True)

    if args.mode in ('train', 'all'):
        run('train', lambda: check_train(args.data_root))
    if args.mode in ('eval', 'all'):
        run('checkpoint', lambda: check_checkpoint(args.checkpoint))
        run('goals', check_goals)
        run('assets', lambda: check_assets(args.asset_root, args.verify_hashes))
    if failures:
        print(f'Release checks failed: {", ".join(failures)}.', file=sys.stderr, flush=True)
        return 1
    print('Release file checks passed. Model loading and simulator execution were not tested.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
