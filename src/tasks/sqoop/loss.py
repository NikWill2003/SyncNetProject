from __future__ import annotations

from torch import Tensor
import torch.nn as nn

from ...core.registry import LossFn
from .contracts import SqoopBatch, SqoopOutput


def build_cross_entropy() -> LossFn:
    ce_loss = nn.CrossEntropyLoss()

    def cross_entropy(
            out: SqoopOutput, batch: SqoopBatch
            ) -> tuple[Tensor, dict[str, float]]:
        loss = ce_loss(out['logits'].float(), batch['answers'])
        return loss, {'cross_entropy': loss.item()}

    return cross_entropy  # type: ignore
