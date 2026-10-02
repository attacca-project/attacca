"""Training objectives: masked behavior cloning and the dense target-map loss."""
import torch
import torch.nn.functional as F
from typing import Dict, Any
from minestudio.offline.mine_callbacks.callback import ObjectiveCallback
from minestudio.offline.mine_callbacks import BehaviorCloneCallback


def _get_seg(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    assert 'segmentation' in batch, "key `segmentation` is required for the target-map loss."
    return batch['segmentation']


class SyntheticExploreAuxCallback(ObjectiveCallback):

    def __init__(self, union_mask_weight: float = 0.5):
        super().__init__()
        self.union_mask_weight = float(union_mask_weight)
        if self.union_mask_weight <= 0:
            raise ValueError("union_mask_weight must be > 0")

    @staticmethod
    def _frame_field(batch, name, like, dtype=torch.float32):
        if name not in batch:
            raise KeyError(
                f"auxiliary loss requires frame field `{name}`; "
                "use the released dataset with dense mask labels"
            )
        value = torch.as_tensor(batch[name], device=like.device, dtype=dtype)
        if value.ndim == like.ndim + 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.shape != like.shape:
            raise ValueError(
                f"`{name}` must have shape {tuple(like.shape)}, got {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _mask14_target(seg, name, logits):
        if name not in seg:
            raise KeyError(f"spatial grounding loss requires segmentation.{name}")
        target = torch.as_tensor(
            seg[name], device=logits.device, dtype=torch.float32)
        if target.ndim != 4 or target.shape[:2] != logits.shape[:2]:
            raise ValueError(
                f"segmentation.{name} must be [B,T,H,W] aligned with logits; "
                f"got {tuple(target.shape)} vs {tuple(logits.shape)}")
        if bool((~torch.isfinite(target)).any().item()):
            raise ValueError(f"segmentation.{name} contains non-finite values")
        target = target.clamp(0, 1)
        b, t = target.shape[:2]
        target = F.adaptive_avg_pool2d(
            target.reshape(b * t, 1, *target.shape[-2:]), (14, 14))
        target = target.reshape(b, t, 196)
        if logits.shape != target.shape:
            raise ValueError(
                f"{name} logits must be [B,T,196], got {tuple(logits.shape)}")
        return target

    @staticmethod
    def _bce_dice_per_frame(logits, target):
        bce = F.binary_cross_entropy_with_logits(
            logits.float(), target.float(), reduction='none').mean(dim=-1)
        prob = torch.sigmoid(logits.float())
        intersection = (prob * target).sum(dim=-1)
        dice = 1.0 - ((2.0 * intersection + 1.0)
                      / (prob.sum(dim=-1) + target.sum(dim=-1) + 1.0))
        return bce + dice

    @staticmethod
    def _balanced_binary_frames(per_frame, valid, positive):
        pos = valid * positive
        neg = valid * (1.0 - positive)
        pos_loss = (per_frame * pos).sum() / pos.sum().clamp_min(1.0)
        neg_loss = (per_frame * neg).sum() / neg.sum().clamp_min(1.0)
        has_pos = (pos.sum() > 0).to(per_frame.dtype)
        has_neg = (neg.sum() > 0).to(per_frame.dtype)
        return ((has_pos * pos_loss + has_neg * neg_loss)
                / (has_pos + has_neg).clamp_min(1.0))

    def __call__(self, batch, batch_idx, step_name, latents, mine_policy):
        seg = _get_seg(batch)
        union_logits = latents.get('union_mask_logits')
        if union_logits is None:
            raise RuntimeError(
                "L_union requested but model emitted latents['union_mask_logits']=None")
        frame_like = union_logits[..., 0]

        valid = torch.as_tensor(
            batch.get('mask', torch.ones_like(frame_like)),
            device=frame_like.device, dtype=torch.float32,
        )
        synthetic = torch.as_tensor(
            batch.get('synthetic', torch.zeros_like(frame_like)),
            device=frame_like.device, dtype=torch.float32,
        )
        if synthetic.dim() < valid.dim():
            synthetic = synthetic.unsqueeze(-1).expand_as(valid)
        if valid.shape != frame_like.shape or synthetic.shape != frame_like.shape:
            raise ValueError(
                "mask/synthetic shape mismatch: "
                f"prediction={tuple(frame_like.shape)} mask={tuple(valid.shape)} "
                f"synthetic={tuple(synthetic.shape)}"
            )
        synthetic_frame = (valid * synthetic) > 0
        synthetic_base = valid * synthetic

        union_target = self._mask14_target(seg, 'union_mask', union_logits)
        union_valid = self._frame_field(
            batch, 'union_mask_valid', frame_like)
        if bool((synthetic_frame & (~torch.isfinite(union_valid)
                                    | ~((union_valid == 0) | (union_valid == 1)))).any().item()):
            raise ValueError(
                "`union_mask_valid` must be finite and binary on synthetic frames")
        union_base = synthetic_base * union_valid
        union_per_frame = self._bce_dice_per_frame(
            union_logits, union_target)
        union_positive = (union_target.sum(dim=-1) > 0).to(frame_like.dtype)
        union_mask_loss = self._balanced_binary_frames(
            union_per_frame, union_base, union_positive)

        # The ROCKET-2 exist/point/box head is kept but not supervised. It
        # stays in the autograd graph with zero weight so that its all-zero
        # gradients enter global-norm clipping exactly as in the released run.
        unsupervised_heads = 0.0 * (latents['exist'].sum() + latents['point'].sum()
                                    + latents['bbox'].sum())
        total = unsupervised_heads + self.union_mask_weight * union_mask_loss
        return {
            'loss': total,
            'synthetic_union_mask_loss': union_mask_loss.detach(),
            'synthetic_union_mask_frames': union_base.sum().detach(),
            'synthetic_union_mask_weight': self.union_mask_weight,
        }


class MaskedBehaviorCloneCallback(BehaviorCloneCallback):

    def __init__(self, weight: float = 0.01, decision_weight_uniform: bool = False):
        super().__init__(weight=weight)
        self.decision_weight_uniform = bool(decision_weight_uniform)

    def __call__(self, batch, batch_idx, step_name, latents, mine_policy):
        assert 'agent_action' in batch, "key `agent_action` is required for behavior cloning."
        agent_action = batch['agent_action']
        pi_logits = latents['pi_logits']
        log_prob = mine_policy.pi_head.logprob(agent_action, pi_logits, return_dict=True)
        entropy = mine_policy.pi_head.entropy(pi_logits, return_dict=True)
        camera_mask = (agent_action['camera'] != 60).float().squeeze(-1)
        global_mask = batch.get('mask', torch.ones_like(camera_mask)).float()
        bc_valid = batch.get('bc_valid', torch.ones_like(global_mask)).float()
        temporal_binding = batch.get(
            'temporal_binding_bc_valid', torch.ones_like(global_mask)).float()
        decision_weight = batch['decision_weight'].float()
        if (bc_valid.shape != global_mask.shape
                or temporal_binding.shape != global_mask.shape
                or decision_weight.shape != global_mask.shape):
            raise ValueError(
                "BC frame fields must match mask shape: "
                f"mask={tuple(global_mask.shape)} bc_valid={tuple(bc_valid.shape)} "
                f"temporal_binding={tuple(temporal_binding.shape)} "
                f"decision_weight={tuple(decision_weight.shape)}"
            )
        if (not bool(torch.isfinite(bc_valid).all().item())
                or not bool(((bc_valid == 0) | (bc_valid == 1)).all().item())):
            raise ValueError("`bc_valid` must be finite and binary")
        if (not bool(torch.isfinite(temporal_binding).all().item())
                or not bool(((temporal_binding == 0)
                             | (temporal_binding == 1)).all().item())):
            raise ValueError(
                "`temporal_binding_bc_valid` must be finite and binary")
        if not bool(torch.isfinite(decision_weight).all().item()) or bool((decision_weight < 0).any().item()):
            raise ValueError("`decision_weight` must be finite and non-negative")
        if self.decision_weight_uniform:
            decision_weight = torch.ones_like(decision_weight)
        global_mask = (
            global_mask * bc_valid * temporal_binding * decision_weight)

        logp_camera = (log_prob['camera'] * global_mask * camera_mask).sum(-1)
        logp_buttons = (log_prob['buttons'] * global_mask).sum(-1)
        entropy_camera = (entropy['camera'] * global_mask * camera_mask).sum(-1)
        entropy_buttons = (entropy['buttons'] * global_mask).sum(-1)
        camera_loss, button_loss = -logp_camera, -logp_buttons
        bc_loss = camera_loss + button_loss
        ent = entropy_camera + entropy_buttons
        return {
            'loss': bc_loss.mean() * self.weight,
            'camera_loss': camera_loss.mean(),
            'button_loss': button_loss.mean(),
            'entropy': ent.mean(),
            'bc_weight': self.weight,
            'bc_valid_frac': bc_valid.mean(),
            'bc_temporal_binding_frac': temporal_binding.mean(),
            'bc_decision_weight_mean': decision_weight.mean(),
            'bc_effective_frame_weight': global_mask.mean(),
        }
