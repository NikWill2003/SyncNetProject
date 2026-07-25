from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from ...core.registry import LossFn
from .contracts import CoalitionsBatch, CoalitionsOutput

def build_coalitions_loss_fn() -> LossFn:

    def coalitions_token_ce(
            out: CoalitionsOutput, batch: CoalitionsBatch
            ) -> tuple[Tensor, dict[str, float]]:
        
        # logits (B, T, N, K); targets (B, T, N); loss_mask (B, T, N)
        B, T, N, K = out['logits'].shape
        flat_logits = out['logits'].reshape(-1, K).float()
        flat_targets = batch['targets'].reshape(-1)
        flat_mask = batch['loss_mask'].reshape(-1)
        tok = F.cross_entropy(
            flat_logits,
            flat_targets.clamp(min=0, max=K - 1),
            reduction='none',
        )
        denom = flat_mask.sum().clamp(min=1.0)
        loss = (tok * flat_mask).sum() / denom
        with torch.no_grad():
            pred = flat_logits.argmax(-1)
            correct = ((pred == flat_targets).float() * flat_mask).sum()
            acc = (correct / denom).item()

        return (
            loss, 
            {'coalitions_token_ce': loss.item(), 
             'token_acc': acc}
             )

    return coalitions_token_ce #type: ignore

