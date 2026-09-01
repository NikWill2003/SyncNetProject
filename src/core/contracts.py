"""The one batch/output contract.

Sort-of-CLEVR and SQOOP differ in the *values* they put in these fields,
never in the fields themselves:

    questions   SOC (B, 18) a one-hot bundle; SQOOP (B, 3) token indices.
                Which of those it is, is the question encoder's business
                (see models/common/qst_enc.py), not the contract's.
    answers     SOC 10-way; SQOOP binary. That is `answer_dim`, an int.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from torch import Tensor



class VQABatch(TypedDict):
    images: Tensor                      # (B, 3, H, W) float in [0, 1]
    questions: Tensor                   # (B, Q) long
    answers: Tensor                     # (B,) long


class VQAOutput(TypedDict):
    logits: Tensor                      # (B, answer_dim)
    traces: NotRequired[dict]
    metrics: NotRequired[dict]          # scalars, sectioned under model/ by the trainer
