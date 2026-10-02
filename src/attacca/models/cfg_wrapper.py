# Adapted from ROCKET-2 (https://github.com/CraftJarvis/ROCKET-2,
# commit 5ea1b3b2e4a4b43af8f00a75f232bb490ec8fb6a, cfg_wrapper.py).
# Upstream metadata: Date: 2025-02-23 22:18:55; LastEditors: caishaofei-mus1
# 1744260356@qq.com; LastEditTime: 2025-03-19 13:36:18;
# FilePath: /ROCKET-2/cfg_wrapper.py.
# Reused: CFGWrapper. Modified: guidance is restricted to scale zero and the
# conditional input/state path is used directly for Attacca inference.
"""Inference wrapper that samples actions from the goal-conditioned policy (guidance scale 0)."""
import torch
from typing import Dict, List, Optional, Tuple, Any
from minestudio.models.base_policy import recursive_tensor_op, dict_map


class CFGWrapper:

    def __init__(self, model, k: float = 0.0):
        if float(k) != 0.0:
            raise ValueError(f"only guidance scale 0 is supported, got {k!r}")
        self.model = model

    @torch.inference_mode()
    def get_action(self,
                   input: Dict[str, Any],
                   state_in: Optional[List[torch.Tensor]],
                   deterministic: bool = False,
                   input_shape: str = "BT*",
                   **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        cond_input = input.copy()
        cond_input['cross_view'] = input['cross_view'].copy()
        cond_latents, cond_state_out = self.get_action_once(cond_input, state_in, deterministic, input_shape, **kwargs)
        self.cache_latents = dict_map(lambda tensor: tensor[0][0], cond_latents)
        action = self.model.pi_head.sample(
            cond_latents['pi_logits'], deterministic)
        state_out = recursive_tensor_op(lambda x: x[0], cond_state_out)
        return dict_map(lambda tensor: tensor[0][0], action), state_out

    @torch.inference_mode()
    def get_action_once(self,
                   input: Dict[str, Any],
                   state_in: Optional[List[torch.Tensor]],
                   deterministic: bool = False,
                   input_shape: str = "BT*",
                   **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        assert input_shape == '*'
        input = dict_map(self.model._batchify, input)
        if state_in is not None:
            state_in = recursive_tensor_op(lambda x: x.unsqueeze(0), state_in)
        latents, state_out = self.model.forward(input, state_in, **kwargs)
        return latents, state_out
