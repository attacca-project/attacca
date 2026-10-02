import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tempfile
from collections.abc import Mapping


sys.dont_write_bytecode = True
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / 'src'))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from attacca.models.inference_checkpoint import CONFIG_NAME
from attacca.models.inference_checkpoint import WEIGHTS_NAME
from attacca.models.inference_checkpoint import make_attacca_config
from attacca.models.inference_checkpoint import read_attacca_config


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_public_config(value, path='config'):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f'{path}: configuration keys must be strings')
            if any(token in key.lower() for token in ('seed', 'run_tag', 'dataset_dir', 'synthetic_lmdb')):
                raise ValueError(f'{path}.{key}: training metadata is not part of the inference format')
            check_public_config(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_public_config(item, f'{path}[{index}]')
    elif isinstance(value, str):
        if value.startswith(('/', '\\', '~/', 'file://')) or PureWindowsPath(value).drive:
            raise ValueError(f'{path}: local paths are not part of the inference format')
        if any(token in value.lower() for token in ('/home/', '/scratch/', '/work/', '/runs/', '/dataset1/')):
            raise ValueError(f'{path}: private paths are not part of the inference format')


def inference_state(checkpoint):
    source = checkpoint.get('state_dict')
    if not isinstance(source, Mapping) or not source:
        raise ValueError('Checkpoint lacks a nonempty state_dict')
    state = {}
    for name, tensor in source.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError('Every state_dict entry must have a string name and a tensor value')
        clean_name = name.removeprefix('mine_policy.')
        if not clean_name or clean_name in state:
            raise ValueError(f'Ambiguous state_dict name after prefix removal: {name!r}')
        if tensor.layout != torch.strided:
            raise ValueError(f'Unsupported non-dense tensor: {name}')
        state[clean_name] = tensor.detach().cpu().contiguous().clone()
    return state


def verify_export(directory, expected_config, expected_state):
    observed_config = make_attacca_config(read_attacca_config(directory))
    if observed_config != expected_config:
        raise RuntimeError('Exported configuration differs from the source model configuration')
    check_public_config(observed_config)
    with safe_open(directory / WEIGHTS_NAME, framework='pt', device='cpu') as tensors:
        if tensors.metadata():
            raise RuntimeError('The inference weights must not contain optional metadata')
        if set(tensors.keys()) != set(expected_state):
            raise RuntimeError('Exported tensor names differ from the source checkpoint')
        for name, expected in expected_state.items():
            observed = tensors.get_tensor(name)
            if observed.dtype != expected.dtype or observed.shape != expected.shape:
                raise RuntimeError(f'Exported tensor dtype or shape differs: {name}')
            expected_bytes = expected.reshape(-1).view(torch.uint8)
            observed_bytes = observed.reshape(-1).view(torch.uint8)
            if not torch.equal(expected_bytes, observed_bytes):
                raise RuntimeError(f'Exported tensor bytes differ: {name}')


def export_checkpoint(checkpoint_path, output_dir):
    source_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f'Checkpoint is not a file: {source_path}')
    destination = Path(output_dir).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f'Output already exists: {destination}')
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Expected a Lightning checkpoint mapping')
    parameters = checkpoint.get('hyper_parameters')
    if not isinstance(parameters, Mapping):
        raise ValueError('Checkpoint lacks hyper_parameters')
    model_config = parameters.get('model')
    if not isinstance(model_config, dict):
        raise ValueError('Checkpoint lacks hyper_parameters.model')
    config = make_attacca_config(model_config)
    check_public_config(config)
    state = inference_state(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.', dir=destination.parent))
    try:
        save_file(state, temporary / WEIGHTS_NAME)
        with (temporary / CONFIG_NAME).open('x', encoding='utf-8') as stream:
            json.dump(config, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        with (temporary / WEIGHTS_NAME).open('rb') as stream:
            os.fsync(stream.fileno())
        verify_export(temporary, config, state)
        if os.path.lexists(destination):
            raise FileExistsError(f'Output appeared while exporting: {destination}')
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    result = {
        'output_dir': str(destination),
        'tensor_count': len(state),
        'tensor_dtypes': sorted({str(tensor.dtype) for tensor in state.values()}),
        'tensor_bytes': sum(tensor.numel() * tensor.element_size() for tensor in state.values()),
        'all_tensor_bytes_verified': True,
        'files': {
            name: {'bytes': (destination / name).stat().st_size,
                   'sha256': sha256(destination / name)}
            for name in (CONFIG_NAME, WEIGHTS_NAME)
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Export an Attacca Lightning checkpoint to model.safetensors and config.json. '
                    'Only load trusted checkpoints: the input uses Python pickle. '
                    'Weights are preserved without quantization; existing output directories are refused.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    try:
        result = export_checkpoint(args.checkpoint, args.output_dir)
    except Exception as error:
        print(f'Export failed: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
