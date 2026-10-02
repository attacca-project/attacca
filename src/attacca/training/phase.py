"""Behavioral-phase head supervision (explore, approach, interact) for phase-conditioned policies."""
from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn.functional as F
from minestudio.offline.mine_callbacks.callback import ObjectiveCallback


class PhaseHeadCallback(ObjectiveCallback):

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = float(weight)

    def __call__(self, batch: Dict[str, Any], batch_idx, step_name, latents, mine_policy):
        logits = latents.get("phase_logits")
        if logits is None:
            raise RuntimeError("phase_aux requires model.phase_cond_mode=film (phase_logits missing)")
        while logits.dim() > 3:
            logits = logits.squeeze(-2)
        b, t, k = logits.shape
        dev = logits.device
        phase = batch["phase"].long().to(dev)
        valid = batch.get("mask", torch.ones(b, t, device=dev)).float() * ((phase >= 0) & (phase <= 2)).float()
        exist = batch["class_exist"].long().to(dev)
        committed = batch.get("target_committed", torch.zeros(b, t, device=dev)).long().to(dev)
        vis_valid = batch.get("visibility_supervision_valid", torch.ones(b, t, device=dev)).float()
        phase = torch.where(phase == 2, phase, ((exist > 0) | (committed > 0)).long())
        valid = valid * vis_valid
        ce = F.cross_entropy(logits.float().reshape(-1, k), phase.clamp(0, 2).reshape(-1), reduction="none").view(b, t)
        n = valid.sum().clamp_min(1.0)
        loss = (ce * valid).sum() / n
        w = self.weight
        out = {"loss": w * loss, "phase_ce": loss.detach()}
        with torch.no_grad():
            pred = logits.argmax(-1)
            acc = ((pred == phase).float() * valid).sum() / n
            out.update({"phase_acc": acc, "phase_frames": valid.sum().detach(),
                        "w_phase": torch.tensor(float(w), device=dev)})
            for c, name in enumerate(("explore", "approach", "interact")):
                m = valid * (phase == c).float()
                out[f"phase_acc_{name}"] = (((pred == c).float() * m).sum() / m.sum().clamp_min(1.0))
                out[f"phase_frac_{name}"] = m.sum() / n
        return out
