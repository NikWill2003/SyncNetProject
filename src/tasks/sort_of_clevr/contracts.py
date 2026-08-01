from __future__ import annotations

from typing import NotRequired, TypedDict

from torch import Tensor


class SortOfClevrBatch(TypedDict):
    images: Tensor 
    questions: Tensor 
    answers: Tensor      


class SortOfClevrOutput(TypedDict):
    logits: Tensor 
    traces: NotRequired[dict]
    metrics: NotRequired[dict]  # scalar diagnostics, sectioned under model/ by the trainer
