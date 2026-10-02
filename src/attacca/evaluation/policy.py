#!/usr/bin/env python
"""Policy runner used by all evaluations (goal encoding, recurrent state and action sampling for Attacca checkpoints) and the simulator world boot helper."""
import os, sys, hashlib
import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, str(REPO) + "/src")

OBJ_ID_MINE = 2


BIOME_SPAWN = {"plains": "plains", "desert": "desert", "snowy": "icy", "icy": "icy",
               "forest": "forest", "taiga": "taiga", "extreme_hills": "extreme_hills",
               "savanna": "savanna", "jungle": "jungle", "mesa": "mesa"}


class Rocket2GoalRunner:
    def __init__(self, ckpt, cfg_coef=0.0, device="cuda"):
        import torch
        from attacca.models.model import load_cross_view_rocket
        from attacca.models.cfg_wrapper import CFGWrapper
        from attacca.models.inference_checkpoint import FORMAT_VERSION
        from attacca.models.inference_checkpoint import INFERENCE_CONTRACT
        from attacca.models.inference_checkpoint import MODEL_TYPE
        from attacca.models.inference_checkpoint import is_attacca_checkpoint
        from attacca.models.inference_checkpoint import read_attacca_config
        ckpt = os.fspath(ckpt)
        if ckpt.endswith('.safetensors'):
            raise ValueError('Set checkpoint to the directory containing config.json and model.safetensors.')
        self.torch = torch
        self.model_contract = None
        if is_attacca_checkpoint(ckpt):
            agent = load_cross_view_rocket(ckpt)
            self.model_contract = {
                "checkpoint_format": "attacca_safetensors",
                "model_type": MODEL_TYPE,
                "format_version": FORMAT_VERSION,
                "inference_contract": INFERENCE_CONTRACT,
                "model_config": read_attacca_config(ckpt),
            }
        elif ckpt.endswith(".ckpt"):
            agent = load_cross_view_rocket(ckpt)
        else:
            raise ValueError(
                "Expected an Attacca checkpoint directory (config.json and "
                f"model.safetensors) or a training .ckpt file: {ckpt}")
        self.model = agent
        self.agent = CFGWrapper(agent.to(device).eval(), k=cfg_coef)
        self.device = device
        self.obj_id = OBJ_ID_MINE
        self.state = None
        self._goal_feature_cache = {}
        self.goal_cache_stats = {"hits": 0, "misses": 0}

    def reset(self):
        self.state = None

    def set_obj_id(self, obj_id):
        self.obj_id = int(obj_id)

    def _goal_cache_key(self, goal_image, goal_mask):
        image = np.ascontiguousarray(goal_image)
        mask = np.ascontiguousarray(goal_mask)
        return (
            image.shape, image.dtype.str, hashlib.sha256(image).hexdigest(),
            mask.shape, mask.dtype.str, hashlib.sha256(mask).hexdigest(),
        )

    def _encoded_goal(self, goal_image, goal_mask):
        key = self._goal_cache_key(goal_image, goal_mask)
        cached = self._goal_feature_cache.get(key)
        if cached is not None:
            self.goal_cache_stats["hits"] += 1
            return cached
        self.goal_cache_stats["misses"] += 1
        torch = self.torch
        image = torch.from_numpy(np.ascontiguousarray(goal_image)).to(self.device)
        mask = torch.from_numpy(np.ascontiguousarray(goal_mask)).to(self.device)
        with torch.inference_mode():
            cached = self.model.encode_goal_tokens(
                image[None, None], mask[None, None])[0, 0].detach()
        if len(self._goal_feature_cache) >= 32:
            self._goal_feature_cache.pop(next(iter(self._goal_feature_cache)))
        self._goal_feature_cache[key] = cached
        return cached

    def act(self, obs, goal_image, goal_mask):
        torch = self.torch
        gimg = cv2.resize(np.asarray(goal_image), (224, 224), interpolation=cv2.INTER_LINEAR)
        gmask = cv2.resize((np.asarray(goal_mask) > 0).astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
        encoded_goal = self._encoded_goal(gimg, gmask)
        env_prev_action = obs["env_prev_action"]
        cross_view = {
            "cross_view_image": gimg,
            "cross_view_obj_id": torch.tensor(self.obj_id),
            "cross_view_obj_mask": torch.tensor(gmask, dtype=torch.uint8),
        }
        cross_view["encoded_goal_tokens"] = encoded_goal
        inp = {
            "image": obs["image"],
            "env_prev_action": env_prev_action,
            "cross_view": cross_view,
        }
        with torch.inference_mode():
            action, self.state = self.agent.get_action(inp, self.state, input_shape="*")
        exist = (float(self.agent.cache_latents["exist"].sigmoid().item())
                 if (hasattr(self.agent, "cache_latents") and self.agent.cache_latents.get("exist") is not None) else -1.0)
        return action, exist


def boot_world(seed, biome="plains", action_type="agent", runtime_overlay=None,
               world_snapshot_dir=None, world_snapshot_sha256=None,
               world_snapshot_archive=None,
               world_snapshot_archive_sha256=None):
    if world_snapshot_sha256 is not None and world_snapshot_dir is None:
        raise ValueError(
            "world_snapshot_sha256 requires world_snapshot_dir")
    if world_snapshot_dir is not None and world_snapshot_archive is None:
        raise ValueError(
            "world_snapshot_dir requires world_snapshot_archive")
    if runtime_overlay is None:
        raise ValueError("boot_world requires runtime_overlay")
    from attacca.runtime.ms_world import ControlledWorldMS
    cfg = {
        "experiment": {"seed": seed},
        "world": {
            "image_size": [224, 224],
            "render_size": [640, 360],
            "action_type": action_type,
            "biome": BIOME_SPAWN.get(biome, biome),
            "runtime_overlay": os.path.abspath(str(runtime_overlay)),
        },
    }
    if world_snapshot_dir is not None:
        cfg["world"]["world_snapshot_dir"] = os.path.abspath(
            str(world_snapshot_dir))
        cfg["world"]["world_snapshot_sha256"] = world_snapshot_sha256
        cfg["world"]["world_snapshot_archive"] = os.path.abspath(
            str(world_snapshot_archive))
        cfg["world"]["world_snapshot_archive_sha256"] = (
            world_snapshot_archive_sha256)
    w = ControlledWorldMS(cfg)
    w.reset()
    w.cmd("/gamerule doDaylightCycle false")
    w.cmd("/time set 6000")
    for _ in range(20):
        w.step_noop()
    return w
