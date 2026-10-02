import copy
import json
from collections.abc import Mapping
from pathlib import Path


MODEL_TYPE = "attacca"
FORMAT_VERSION = 1
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
INFERENCE_CONTRACT = {
    "image_size": [224, 224],
    "image_layout": "BTHWC",
    "image_color_space": "RGB",
    "image_dtype": "uint8",
    "image_scale": 255.0,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "goal_mask_layout": "BTHW",
    "goal_mask_values": [0, 1],
    "goal_image_resize": "opencv_INTER_LINEAR",
    "goal_mask_resize": "opencv_INTER_NEAREST",
    "action_space": "minestudio_vpt",
    "button_classes": 8641,
    "camera_classes": 121,
    "camera_quantization": "mu_law",
    "camera_maxval": 10,
    "camera_binsize": 2,
    "camera_mu": 10.0,
    "previous_action": "environment_action",
    "interaction_ids": {"hunt": 0, "mine": 2, "use": 3},
    "tokens_per_step": 11,
    "recurrent_state": "transformer_xl",
    "weights": "complete_state_dict",
}

_RESERVED = {"model_type", "format_version", "inference_contract"}
_REQUIRED = {
    "view_backbone": "timm/vit_base_patch16_224.dino",
    "mask_backbone": "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "hiddim": 1024,
    "num_heads": 8,
    "num_layers": 4,
    "timesteps": 128,
    "mem_len": 128,
    "use_prev_action": True,
    "num_view_tokens": 9,
    "use_union_token": True,
    "phase_cond_mode": "film",
}
_FIXED = {
    "goal_fusion_mode": "official_mask_vit",
    "action_space": None,
}
_BOOLEANS = {"cache_static_goal_features", "phase_cond_stopgrad", "pretrained_backbones"}


def validate_model_config(model_config):
    if not isinstance(model_config, Mapping):
        raise ValueError("Attacca model configuration must be an object")
    config = copy.deepcopy(dict(model_config))
    allowed = set(_REQUIRED) | set(_FIXED) | _BOOLEANS
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unsupported Attacca model fields: {sorted(unknown)}")
    for key, expected in _REQUIRED.items():
        if key not in config or type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f"Attacca requires {key}={expected!r}")
    for key, expected in _FIXED.items():
        if key in config and (type(config[key]) is not type(expected) or config[key] != expected):
            raise ValueError(f"Unsupported Attacca setting: {key}={config[key]!r}")
    for key in _BOOLEANS & config.keys():
        if type(config[key]) is not bool:
            raise ValueError(f"Attacca {key} must be a boolean")
    return config


def model_constructor_config(model_config):
    return validate_model_config(model_config)


def make_attacca_config(model_config):
    config = validate_model_config(model_config)
    config.update({
        "model_type": MODEL_TYPE,
        "format_version": FORMAT_VERSION,
        "inference_contract": copy.deepcopy(INFERENCE_CONTRACT),
    })
    return config


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def _read_document(directory):
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"Expected an Attacca checkpoint directory: {path}")
    document = json.loads(
        (path / CONFIG_NAME).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_invalid_json_constant,
    )
    if not isinstance(document, dict):
        raise ValueError("Checkpoint config.json must contain an object")
    return document


def _constructor_config(document):
    if document.get("model_type") != MODEL_TYPE:
        raise ValueError("Checkpoint model_type must be attacca")
    if type(document.get("format_version")) is not int or document["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported Attacca format_version: {document.get('format_version')!r}")
    contract = document.get("inference_contract")
    if json.dumps(contract, sort_keys=True, allow_nan=False) != json.dumps(INFERENCE_CONTRACT, sort_keys=True):
        raise ValueError("Unsupported Attacca inference_contract")
    return validate_model_config({key: value for key, value in document.items() if key not in _RESERVED})


def read_attacca_config(directory):
    return _constructor_config(_read_document(directory))


def is_attacca_checkpoint(path):
    directory = Path(path)
    if not directory.is_dir() or not (directory / CONFIG_NAME).is_file():
        return False
    document = _read_document(directory)
    if document.get("model_type") != MODEL_TYPE:
        if "format_version" in document or "inference_contract" in document:
            raise ValueError("Attacca inference metadata requires model_type=attacca")
        return False
    _constructor_config(document)
    return True
