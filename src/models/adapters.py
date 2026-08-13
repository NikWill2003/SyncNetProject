"""Adapter bases: batch dict in, task output dict out.

Two calling conventions exist in this codebase and they are not the same:

  * the Trainer calls `model(batch)` -- one positional dict
  * shared models take `model(images, questions, **overrides)`

Adapters are the seam. A task adapter holds an `inner` shared model,
unpacks the batch, and returns the task's output TypedDict. Callbacks
that need a non-standard forward (traces, `t_override`) call the adapter
with a batch dict too, so there is exactly one convention outside this
file.

`T` is exposed as a property rather than an attribute so that
`hasattr(model, 'T')` stays False for models with no dynamics horizon --
that is the duck-check t_variance uses to skip itself.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


class VQAAdapter(nn.Module):
    """Common surface for task adapters over a shared VQA model."""

    is_syncnet = False
    has_rotors = False

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    @property
    def T(self) -> int:
        # AttributeError here is load-bearing: it makes hasattr false.
        return self.inner.T  # type: ignore[return-value]

    @property
    def cfg(self) -> Any:
        return self.inner.cfg  # type: ignore[return-value]


class ImageQuestionAdapter(VQAAdapter):
    """For models reading both images and questions."""

    def forward(self, batch: dict, **overrides) -> dict:
        return self._wrap(
            self.inner(batch['images'], batch['questions'], **overrides)
        )

    @staticmethod
    def _wrap(out) -> dict:
        # transformer returns raw logits; syncnet returns a full dict
        return out if isinstance(out, dict) else {'logits': out}


class QuestionOnlyAdapter(VQAAdapter):
    """For the guessing floor: no image pathway at all."""

    def forward(self, batch: dict, **overrides) -> dict:
        return {'logits': self.inner(batch['questions'])}
