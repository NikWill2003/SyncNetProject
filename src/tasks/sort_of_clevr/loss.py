from __future__ import annotations

from torch import Tensor
import torch.nn as nn

from ...core.registry import LossFn
from .contracts import SortOfClevrBatch, SortOfClevrOutput


def build_cross_entropy() -> LossFn:
    ce_loss = nn.CrossEntropyLoss()

    def cross_entropy(
            out: SortOfClevrOutput, batch: SortOfClevrBatch
            ) -> tuple[Tensor, dict[str, float]]:
        
        logits = out['logits']
        answers = batch['answers']

        loss = ce_loss(logits.float(), answers)
        return loss, {'cross_entropy': loss.item()}

    return cross_entropy #type: ignore
