from __future__ import annotations

from typing import NotRequired, TypedDict

from torch import Tensor


class SqoopBatch(TypedDict):
    images: Tensor     # (B, 3, H, W) float in [0, 1]
    questions: Tensor  # (B, 3) long, joint SHAPES+RELATIONS vocab
    answers: Tensor    # (B,) long in {0, 1}


class SqoopOutput(TypedDict):
    logits: Tensor  # (B, 2)
    traces: NotRequired[dict]
    metrics: NotRequired[dict]
