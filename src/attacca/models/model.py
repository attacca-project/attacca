# Adapted from ROCKET-2 (https://github.com/CraftJarvis/ROCKET-2,
# commit 5ea1b3b2e4a4b43af8f00a75f232bb490ec8fb6a, model.py).
# Upstream metadata: Date: 2025-03-19 20:24:46; LastEditors: caishaofei-mus1
# 1744260356@qq.com; LastEditTime: 2025-03-21 12:06:55;
# FilePath: /ROCKET2-OSS/model.py.
# Reused: CrossViewRocket and ActionEmbeddingLayer. Modified: dense target-map
# head, residual target-map feedback, and phase FiLM.
"""Policy model (ROCKET-2 cross-view policy with a dense target-map head, target-map feedback and behavioral-phase conditioning) and checkpoint loading."""
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from einops import rearrange, repeat
from typing import List, Dict, Optional
from rich import print

import timm
from huggingface_hub import PyTorchModelHubMixin
from minestudio.models.base_policy import MinePolicy
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from attacca.models.inference_checkpoint import WEIGHTS_NAME
from attacca.models.inference_checkpoint import model_constructor_config
from attacca.models.inference_checkpoint import read_attacca_config

BINARY_KEYS = [
    "forward", "back", "left", "right", "inventory", "sprint", "sneak", "jump", "attack", "use",
    "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4", "hotbar_5", "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9"
]

class ActionEmbeddingLayer(nn.Module):

    def __init__(self, hiddim: int):
        super().__init__()
        self.camera_layer = nn.Linear(2, hiddim)
        self.binary_layers = nn.ModuleDict({
            f"act_{key}": nn.Embedding(2, hiddim) for key in BINARY_KEYS
        })

    def forward(self, action: Dict) -> torch.Tensor:
        x = self.camera_layer(action['camera'].float())
        for key in BINARY_KEYS:
            x += self.binary_layers[f"act_{key}"](action[key.replace("_", ".")])
        return x


class CrossViewRocket(MinePolicy, PyTorchModelHubMixin):

    def __init__(self,
        view_backbone: str = 'timm/vit_base_patch16_224.dino',
        mask_backbone: str = 'timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k',
        hiddim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        timesteps: int = 128,
        mem_len: int = 128,
        use_prev_action: bool = False,
        num_view_tokens: int = 1,
        action_space = None,
        goal_fusion_mode: str = 'official_mask_vit',
        use_union_token: bool = False,
        cache_static_goal_features: bool = False,
        phase_cond_mode: str = 'off',
        phase_cond_stopgrad: bool = True,
        pretrained_backbones: bool = True,
        **kwargs,
    ):
        super().__init__(hiddim=hiddim, action_space=action_space)
        if not pretrained_backbones:
            view_backbone = view_backbone.removeprefix('timm/')
            mask_backbone = mask_backbone.removeprefix('timm/')
        self.view_backbone = timm.create_model(
            view_backbone, pretrained=pretrained_backbones, features_only=True)
        data_config = timm.data.resolve_model_data_config(self.view_backbone)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
        self.goal_fusion_mode = str(goal_fusion_mode)
        if self.goal_fusion_mode != 'official_mask_vit':
            raise ValueError(
                f"goal_fusion_mode must be official_mask_vit, got {goal_fusion_mode!r}")
        self.cache_static_goal_features = bool(cache_static_goal_features)
        self.updim_obs = nn.Conv2d(self.view_backbone.feature_info[-1]['num_chs'], hiddim, kernel_size=1, bias=False)
        self.mask_backbone = timm.create_model(
            mask_backbone, pretrained=pretrained_backbones, features_only=True,
            in_chans=1)
        vision_dim = (
            self.view_backbone.feature_info[-1]['num_chs']
            + self.mask_backbone.feature_info[-1]['num_chs'])
        self.updim_cross = nn.Conv2d(
            vision_dim, hiddim, kernel_size=1, bias=False)
        self.num_view_tokens = num_view_tokens
        self.view_cls_tokens = nn.Parameter(torch.randn(1, self.num_view_tokens, hiddim) * 1e-3)
        self.view_resampler = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim,
                nhead=num_heads,
                dim_feedforward=hiddim*2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers,
        )
        self.interaction = nn.Embedding(10, hiddim)
        self.num_step_tokens = self.num_view_tokens + 1

        self.index_bias = -2
        self.use_prev_action = use_prev_action
        if self.use_prev_action:
            self.action_embedding_layer = ActionEmbeddingLayer(hiddim)
            self.num_step_tokens += 1
            self.index_bias -= 1

        self.use_union_token = bool(use_union_token)
        if self.use_union_token:
            if not use_prev_action or int(num_view_tokens) != 9:
                raise ValueError(
                    "the grounding token contract requires num_view_tokens=9 and "
                    "use_prev_action=True")

        self.phase_cond_mode = str(phase_cond_mode)
        self.phase_cond_stopgrad = bool(phase_cond_stopgrad)
        if self.phase_cond_mode not in ('off', 'film'):
            raise ValueError(f"phase_cond_mode must be off|film, got {phase_cond_mode!r}")
        if self.phase_cond_mode == 'film':
            self.phase_head = nn.Linear(hiddim, 3)
            self.phase_film_beta = nn.Linear(3, hiddim, bias=False)
            self.phase_film_gamma = nn.Linear(3, hiddim, bias=False)
            nn.init.zeros_(self.phase_film_beta.weight)
            nn.init.zeros_(self.phase_film_gamma.weight)

        self.dropout_embedding = nn.Parameter(torch.randn(1, 1, hiddim) * 1e-3)

        print(f"number of tokens per timestep is {self.num_step_tokens}")
        self.recurrent = ResidualRecurrentBlocks(
            hidsize=hiddim,
            timesteps=timesteps*self.num_step_tokens,
            recurrence_type="transformer",
            is_residual=True,
            use_pointwise_layer=True,
            pointwise_ratio=4,
            pointwise_use_activation=False,
            attention_mask_style="clipped_causal",
            attention_heads=num_heads,
            attention_memory_size=(mem_len+timesteps)*self.num_step_tokens,
            n_block=num_layers,
            inject_condition=False,
        )
        self.lastlayer = FanInInitReLULayer(hiddim, hiddim, layer_type="linear", batch_norm=False, layer_norm=True)
        self.final_ln = nn.LayerNorm(hiddim)

        self.aux_vis_head = nn.Linear(hiddim, 1+2+4)

        if self.use_union_token:
            if int(hiddim) != 1024:
                raise ValueError(
                    "the grounding token constructor requires hiddim=1024")

            def _mask_head_mlp():
                head_hidden = int(kwargs.get('spatial_mask_head_hidden', 256))
                return nn.Sequential(
                    nn.Linear(hiddim, head_hidden), nn.ReLU(),
                    nn.Linear(head_hidden, 1))

            self.union_null_embedding = nn.Parameter(
                torch.randn(1, hiddim) * 1e-3)
            self.union_feat_proj = nn.Linear(hiddim, 640)
            self.union_map_mlp = nn.Sequential(
                nn.Linear(196, 256), nn.ReLU(), nn.Linear(256, 256))
            self.union_geom_mlp = nn.Sequential(
                nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128))
            self.union_token_ln = nn.LayerNorm(hiddim)

            self.union_token_out = nn.Linear(hiddim, hiddim, bias=False)
            nn.init.zeros_(self.union_token_out.weight)
            self.union_mask_head = _mask_head_mlp()

        for param in self.view_backbone.parameters():
            param.requires_grad = False

    def encode_goal_tokens(self, goal_image: torch.Tensor,
                           goal_mask: torch.Tensor) -> torch.Tensor:
        if goal_image.ndim != 5 or goal_mask.ndim != 4:
            raise ValueError(
                "goal encoding expects image [B,T,H,W,C] and mask [B,T,H,W]")
        if goal_image.shape[:2] != goal_mask.shape[:2]:
            raise ValueError("goal image/mask batch-time axes differ")
        b_goal, t_goal = goal_image.shape[:2]
        cross_view_rgb = rearrange(
            goal_image, 'b t h w c -> (b t) c h w')
        cross_view_rgb = self.transforms(cross_view_rgb)
        x_cross_image = self.view_backbone(cross_view_rgb)[-1]

        cross_view_mask = rearrange(
            goal_mask, 'b t h w -> (b t) 1 h w') * 1.0
        x_cross_mask = self.mask_backbone(cross_view_mask)[-1]
        x_cross = self.updim_cross(torch.cat([
            x_cross_image, x_cross_mask], dim=1))
        x_cross = rearrange(
            x_cross, '(b t) c h w -> b t (h w) c',
            b=b_goal, t=t_goal)
        return x_cross

    def _goal_tokens_for_view(self, cross_view: Dict, b: int, t: int,
                              dtype: torch.dtype, device: torch.device):
        cached = cross_view.get('encoded_goal_tokens')
        if cached is not None:
            if cached.ndim != 4 or cached.shape[0] != b:
                raise ValueError(
                    "encoded_goal_tokens must have shape [B,1|T,196,D]")
            if cached.shape[1] not in (1, t):
                raise ValueError(
                    f"encoded_goal_tokens time axis {cached.shape[1]} is not 1 or {t}")
            expected_hidden = int(self.updim_cross.out_channels)
            if cached.shape[2] != 196 or cached.shape[3] != expected_hidden:
                raise ValueError(
                    "encoded_goal_tokens must contain exactly 196 H-dimensional patches")
            goal_tokens = cached.to(device=device, dtype=dtype)
        else:
            goal_image = cross_view['cross_view_image']
            goal_mask = cross_view['cross_view_obj_mask']
            if self.cache_static_goal_features:
                goal_image = goal_image[:, :1]
                goal_mask = goal_mask[:, :1]
            goal_tokens = self.encode_goal_tokens(goal_image, goal_mask)
        if goal_tokens.shape[1] == 1 and t != 1:
            goal_tokens = goal_tokens.expand(-1, t, -1, -1)
        return rearrange(goal_tokens, 'b t n c -> (b t) n c')

    def encode_view_tokens(self, agent_view: torch.Tensor, cross_view: Dict):
        b, t = agent_view.shape[:2]

        obs_rgb = rearrange(agent_view, 'b t h w c -> (b t) c h w')
        obs_rgb = self.transforms(obs_rgb)
        x_obs = self.updim_obs(self.view_backbone(obs_rgb)[-1])
        x_obs = rearrange(x_obs, 'b c h w -> b (h w) c')

        x_cross = self._goal_tokens_for_view(
            cross_view, b, t, x_obs.dtype, x_obs.device)

        x_cls = self.view_cls_tokens.expand(x_obs.shape[0], -1, -1)
        x_all = self.view_resampler(torch.cat([x_cls, x_obs, x_cross], dim=1))
        x_view = rearrange(
            x_all[:, :self.num_view_tokens, :], "(b t) n c -> b t n c", b=b)
        obs_mixed = (
            x_all[:, self.num_view_tokens:self.num_view_tokens + x_obs.shape[1], :]
            if self.use_union_token else None)

        return x_view, obs_mixed

    @staticmethod
    def _spatial_geometry(p: torch.Tensor):
        if p.ndim != 2 or p.shape[-1] != 196:
            raise ValueError(f"spatial probability map must be [BT,196], got {tuple(p.shape)}")
        centers = torch.linspace(
            -1.0 + 1.0 / 14.0, 1.0 - 1.0 / 14.0, 14,
            device=p.device, dtype=p.dtype)
        gy = centers.repeat_interleave(14)
        gx = centers.repeat(14)
        w = p / p.sum(-1, keepdim=True).clamp_min(1e-6)
        cx = (w * gx).sum(-1)
        cy = (w * gy).sum(-1)
        width = (w * (gx[None] - cx[:, None]).square()).sum(-1).clamp_min(1e-6).sqrt()
        height = (w * (gy[None] - cy[:, None]).square()).sum(-1).clamp_min(1e-6).sqrt()
        geom = torch.stack(
            [cx, cy, width, height, p.mean(-1), p.max(-1).values], dim=-1)
        return w, geom

    def _grounding_token(self, obs_mixed: torch.Tensor, b: int, t: int):
        bt, n, h = obs_mixed.shape
        if (bt, n, h) != (b * t, 196, 1024):
            raise ValueError(
                "grounding expects post-resampler CURRENT tokens "
                f"[B*T,196,1024], got {tuple(obs_mixed.shape)}")
        aux = {}
        union_logits = self.union_mask_head(obs_mixed).squeeze(-1)
        p_union_float = torch.sigmoid(union_logits.float())
        p_union_live = p_union_float.to(obs_mixed.dtype)
        aux['union_mask_logits'] = rearrange(
            union_logits, '(b t) n -> b t n', b=b)

        wu_live, union_geom6_live = self._spatial_geometry(p_union_live)
        union_feat_live = (obs_mixed * wu_live.unsqueeze(-1)).sum(1)

        p_union = p_union_live.detach()
        union_feat = union_feat_live.detach()

        eps = torch.finfo(p_union.dtype).eps
        entropy = -(p_union.clamp(eps, 1 - eps) * p_union.clamp(eps, 1 - eps).log()
                    + (1 - p_union).clamp(eps, 1 - eps)
                    * (1 - p_union).clamp(eps, 1 - eps).log()).mean(-1)
        union_geometry = torch.stack(
            [p_union.mean(-1), p_union.max(-1).values, entropy], dim=-1)
        union_token = self.union_token_ln(torch.cat([
            self.union_feat_proj(union_feat),
            self.union_map_mlp(p_union),
            self.union_geom_mlp(union_geometry),
        ], dim=-1))

        return (
            rearrange(union_token, '(b t) c -> b t 1 c', b=b),
            aux,
        )

    def _recurrent_initial_state(self, batch_size: int) -> List[torch.Tensor]:
        return list(self.recurrent.initial_state(batch_size))

    def temporal_reason(self, x: torch.Tensor, memory: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        b, t = x.shape[:2]
        if not hasattr(self, 'first') or self.first.shape[:2] != (b, t):
            self.first = torch.tensor([[False]], device=x.device).repeat(b, t)
        if memory is None:
            memory = [state.to(x.device) for state in self._recurrent_initial_state(b)]
        z, memory = self.recurrent(x, self.first, memory)
        z = F.relu(z, inplace=False)
        z = self.lastlayer(z)
        z = self.final_ln(z)
        return z, memory

    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:

        b, t = input['image'].shape[:2]

        x_view, obs_mixed = self.encode_view_tokens(
            input['image'], input['cross_view'])
        x = x_view

        grounding_aux = {}
        union_token = None
        if self.use_union_token:
            union_token, grounding_aux = self._grounding_token(
                obs_mixed, b, t)
        grounding_delta = None
        if self.use_union_token:
            grounding_delta = self.union_token_out(union_token.squeeze(-2))

        x_cond = self.interaction(input['cross_view']['cross_view_obj_id'] + 1)
        x_cond = rearrange(x_cond, "b t c -> b t 1 c")
        x = torch.cat([x, x_cond], dim=-2)

        if self.use_prev_action:
            x_prev_a = self.action_embedding_layer(input['env_prev_action'])
            if 'prev_action_dropout' in input:
                dropout_mask = input['prev_action_dropout'][..., None]
                dropout_embedding = repeat(self.dropout_embedding, "1 1 c -> b t c", b=b, t=t)
                x_prev_a = x_prev_a * dropout_mask + dropout_embedding * (1 - dropout_mask)
            if self.use_union_token:
                x_prev_a = x_prev_a + grounding_delta

            x_prev_a = rearrange(x_prev_a, "b t c -> b t 1 c")
            x = torch.cat([x, x_prev_a], dim=-2)

        x = rearrange(x, "b t n c -> b (t n) c", b=b)
        z, memory = self.temporal_reason(x, memory)
        z = rearrange(z, "b (t n) c -> b t n c", t=t)

        aux_vis_logits = self.aux_vis_head(z[:, :, self.index_bias, :])
        exist = aux_vis_logits[:, :, 0:1]
        point = aux_vis_logits[:, :, 1:3]
        bbox = aux_vis_logits[:, :, 3:7]

        z_readout = z[:, :, -1, :]
        phase_logits = None
        if self.phase_cond_mode == 'film':
            phase_logits = self.phase_head(z_readout)
            p = torch.softmax(phase_logits.float(), dim=-1).to(z_readout.dtype)
            if self.phase_cond_stopgrad:
                p = p.detach()
            z_readout = (z_readout + self.phase_film_beta(p)
                         + z_readout * self.phase_film_gamma(p))
        pi_logits = self.pi_head(z_readout)
        vpred =  self.value_head(z_readout)
        latents = {
            "pi_logits": pi_logits,
            "vpred": vpred,
            "exist": exist,
            "point": point,
            "bbox": bbox,
        }
        if phase_logits is not None:
            latents["phase_logits"] = phase_logits
        if self.use_union_token:
            latents.update(grounding_aux)
        return latents, memory


    def initial_state(self, batch_size: int = None) -> List[torch.Tensor]:
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self._recurrent_initial_state(1)]
        return [t.to(self.device) for t in self._recurrent_initial_state(batch_size)]

def load_cross_view_rocket(ckpt_path: str):
    path = Path(ckpt_path)
    if path.is_dir():
        from safetensors.torch import load_file
        model_cfg = model_constructor_config(read_attacca_config(path))
        model_cfg['pretrained_backbones'] = False
        model = CrossViewRocket(**model_cfg)
        state_dict = load_file(str(path / WEIGHTS_NAME), device='cpu')
        model.load_state_dict(state_dict, strict=True)
        return model
    if path.suffix == '.safetensors':
        raise ValueError(
            "Pass the checkpoint directory containing config.json and "
            "model.safetensors, not the safetensors file")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or not isinstance(
            ckpt.get('hyper_parameters', {}).get('model'), dict):
        raise ValueError(
            "Lightning checkpoint lacks hyper_parameters.model; architecture "
            "cannot be reconstructed safely")
    model_cfg = model_constructor_config(ckpt['hyper_parameters']['model'])
    model = CrossViewRocket(**model_cfg)
    state_dict = {
        (k.removeprefix('mine_policy.')): v
        for k, v in ckpt['state_dict'].items()}
    fresh_state = model.state_dict()
    missing_names = sorted(set(fresh_state) - set(state_dict))
    unexpected_names = sorted(set(state_dict) - set(fresh_state))
    dino_prefixes = ('view_backbone.',)
    backfilled_dino = [
        key for key in missing_names if key.startswith(dino_prefixes)]
    bad_missing = [key for key in missing_names if key not in backfilled_dino]
    if bad_missing or unexpected_names:
        raise RuntimeError(
            "checkpoint/model architecture mismatch: "
            f"missing={bad_missing[:12]} unexpected={unexpected_names[:12]}")
    for key in backfilled_dino:
        state_dict[key] = fresh_state[key]
    model.load_state_dict(state_dict, strict=True)
    return model
