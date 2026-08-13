from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from torch import Tensor

from ....core.callbacks import BaseCallBack, accuracy
from ....core.config import CallbackConfig
from ....core.registry import CallbackSpec
from ..contracts import SqoopBatch, SqoopOutput
from ..data.constants import N_SHAPES, RELATIONS

if TYPE_CHECKING:
    from ....core import Config


@dataclass
class SqoopAccuracyCallbackCfg(CallbackConfig):
    name: str = 'accuracy'


def _sqoop_accuracy(
        logits: Tensor, answers: Tensor, questions: Tensor
        ) -> dict[str, float]:
    out = {'accuracy': accuracy(logits, answers)}
    rel_idx = questions[:, 1] - N_SHAPES  # (B,) in [0, 4)
    for i, rel in enumerate(RELATIONS):
        mask = rel_idx == i
        out[f'{rel}_accuracy'] = accuracy(logits[mask], answers[mask])
    return out


class sqoop_accuracy_callback(BaseCallBack):

    def on_train_step_end(
            self, trainer, out: SqoopOutput, batch: SqoopBatch
            ) -> Optional[dict[str, float]]:
        return _sqoop_accuracy(
            out['logits'], batch['answers'], batch['questions']
        )

    def on_eval_step_end(
            self, trainer, out: SqoopOutput, batch: SqoopBatch
            ) -> Optional[dict[str, float]]:
        return _sqoop_accuracy(
            out['logits'], batch['answers'], batch['questions']
        )

    def on_test_step_end(
            self, trainer, out: SqoopOutput, batch: SqoopBatch
            ) -> Optional[dict[str, float]]:
        return _sqoop_accuracy(
            out['logits'], batch['answers'], batch['questions']
        )

    @classmethod
    def from_config(
            cls, cfg: 'Config', cb_cfg: SqoopAccuracyCallbackCfg
            ) -> 'sqoop_accuracy_callback':
        return cls()


sqoop_metric_callbacks: dict[str, CallbackSpec] = {
    'accuracy': CallbackSpec(
        config=SqoopAccuracyCallbackCfg,
        callback_class=sqoop_accuracy_callback,
    ),
}
