"""Sort-of-CLEVR callbacks.

Accuracy overall, per question family, and per subtype within a family.
The 18-dim question vector encodes the family at Q_TYPE_IDX and the
subtype at SUB_Q_TYPE_IDX, so the breakdown is only meaningful here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...core.callbacks import BaseCallBack, CallbackSpec, accuracy
from ...core.config import CallbackConfig
from ...core.contracts import VQABatch, VQAOutput
from .spec import (
    Q_TYPES_OFFSET, Q_TYPE_IDX, SUBTYPE_NAMES, SUB_Q_TYPE_IDX,
)

if TYPE_CHECKING:
    from torch import Tensor

    from ...core.config import Config


@dataclass
class AccuracyCBCfg(CallbackConfig):
    name: str = 'accuracy'
    # report the nine subtypes as well as the three families
    subtypes: bool = True


class AccuracyCB(BaseCallBack):

    def __init__(self, subtypes: bool = True) -> None:
        super().__init__()
        self.subtypes = subtypes

    def _groups(self, questions: Tensor) -> dict[str, Tensor]:
        groups = {
            family: questions[:, Q_TYPE_IDX + offset] == 1
            for family, offset in Q_TYPES_OFFSET.items()
        }
        if self.subtypes:
            for (family, subtype), sname in SUBTYPE_NAMES.items():
                fam = questions[:, Q_TYPE_IDX + Q_TYPES_OFFSET[family]] == 1
                sub = questions[:, SUB_Q_TYPE_IDX + subtype] == 1
                groups[sname] = fam & sub
        return groups

    def metrics(self, out: VQAOutput, batch: VQABatch) -> dict[str, float]:
        logits, answers = out['logits'], batch['answers']
        metrics = {'accuracy': accuracy(logits, answers)}
        metrics.update({
            f'{name}_accuracy': accuracy(logits[mask], answers[mask])
            for name, mask in self._groups(batch['questions']).items()
        })
        return metrics

    def on_train_step_end(
            self, trainer, out: VQAOutput, batch: VQABatch,
            ) -> Optional[dict[str, float]]:
        return self.metrics(out, batch)

    def on_eval_step_end(
            self, trainer, out: VQAOutput, batch: VQABatch,
            ) -> Optional[dict[str, float]]:
        return self.metrics(out, batch)

    def on_test_step_end(
            self, trainer, out: VQAOutput, batch: VQABatch,
            ) -> Optional[dict[str, float]]:
        return self.metrics(out, batch)

    @classmethod
    def from_config(
            cls, cfg: Config, cb_cfg: AccuracyCBCfg,
            ) -> AccuracyCB:
        return cls(subtypes=bool(cb_cfg.subtypes))


CALLBACKS: dict[str, CallbackSpec] = {
    'accuracy': CallbackSpec(AccuracyCBCfg, AccuracyCB),
}


# The synchrony diagnostics read phase traces, so they are gated on the
# model declaring 'sync'. They are registered here because `interventions`
# slices this dataset's question taxonomy; they move to a model-side home
# when the sync models are rewritten.
try:
    from ...analysis import ANALYSIS_CALLBACKS
    CALLBACKS.update(ANALYSIS_CALLBACKS)
except Exception:
    pass
