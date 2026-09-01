"""SQOOP callbacks.

Accuracy overall and per spatial relation. A question is the token triple
[x, rel, y], so the relation is a single column and the breakdown is a
lookup -- but it indexes this dataset's joint vocabulary, so it is only
meaningful here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...core.callbacks import BaseCallBack, CallbackSpec, accuracy
from ...core.config import CallbackConfig
from ...core.contracts import VQABatch, VQAOutput
from .spec import N_SHAPES, RELATIONS

if TYPE_CHECKING:
    from torch import Tensor

    from ...core.config import Config


@dataclass
class AccuracyCBCfg(CallbackConfig):
    name: str = 'accuracy'
    # report the four relations as well as the overall number
    relations: bool = True


class AccuracyCB(BaseCallBack):

    def __init__(self, relations: bool = True) -> None:
        super().__init__()
        self.relations = relations

    def _groups(self, questions: Tensor) -> dict[str, Tensor]:
        if not self.relations:
            return {}
        # relations are indexed after the shapes in the joint vocabulary
        rel_idx = questions[:, 1] - N_SHAPES
        return {rel: rel_idx == i for i, rel in enumerate(RELATIONS)}

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
        return cls(relations=bool(cb_cfg.relations))


CALLBACKS: dict[str, CallbackSpec] = {
    'accuracy': CallbackSpec(AccuracyCBCfg, AccuracyCB),
}


# t_variance is dataset-agnostic but model-gated; see soc/callbacks.py.
try:
    from ...analysis import ANALYSIS_CALLBACKS
    CALLBACKS['t_variance'] = ANALYSIS_CALLBACKS['t_variance']
except Exception:
    pass
