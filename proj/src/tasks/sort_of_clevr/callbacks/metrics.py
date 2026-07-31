from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from torch import Tensor

from ....core.callbacks import BaseCallBack, accuracy
from ....core.config import CallbackConfig
from ....core.registry import CallbackSpec
from ..contracts import SortOfClevrBatch, SortOfClevrOutput
from ..data import constants as C

if TYPE_CHECKING:
    from ....core import Config


@dataclass
class AccuracyCallbackCfg(CallbackConfig):
    name: str = 'accuracy'


@dataclass
class QtypeAccuracyCallbackCfg(CallbackConfig):
    name: str = 'qtype_accuracy'


# TODO:
# - what metrics should i add to this?
# - accuracy across question type?
# - accuracy across question subtype?
# - 


def sort_of_clevr_accuracy(
        logits: Tensor, answers: Tensor
        ) -> dict[str, float]:
        return {'accuracy': accuracy(logits, answers)}


def sort_of_clevr_qtype_accuracy(
        logits: Tensor, answers: Tensor, questions: Tensor,
        ) -> dict[str, float]:

        # questions: (B, Q_dim)
        non_relational_qtype_idx = C.Q_TYPE_IDX + C.Q_TYPES_OFFSET['non_relational']
        binary_qtype_idx = C.Q_TYPE_IDX + C.Q_TYPES_OFFSET['binary']
        ternary_qtype_idx = C.Q_TYPE_IDX + C.Q_TYPES_OFFSET['ternary']

        non_relational_idx = (    
            questions[:, non_relational_qtype_idx] == 1
        ) 

        binary_idx = (
            questions[:, binary_qtype_idx] == 1
        )
        
        ternary_idx = (
            questions[:, ternary_qtype_idx] == 1
        )

        return {
            'non_relational_accuracy': accuracy(logits[non_relational_idx], answers[non_relational_idx]),
            'binary_accuracy': accuracy(logits[binary_idx], answers[binary_idx]),
            'ternary_accuracy': accuracy(logits[ternary_idx], answers[ternary_idx])
        }


class sort_of_clevr_accuracy_callback(BaseCallBack):
    
    def on_train_step_end(
            self, trainer, out: SortOfClevrOutput, batch: SortOfClevrBatch
            ) -> Optional[dict[str, float]]:

        return sort_of_clevr_accuracy(out['logits'], batch['answers'])

    def on_eval_step_end(
            self, trainer, out: SortOfClevrOutput, batch: SortOfClevrBatch
            ) -> Optional[dict[str, float]]:

        return sort_of_clevr_accuracy(out['logits'], batch['answers'])

    def on_test_step_end(
            self, trainer, out: SortOfClevrOutput, batch: SortOfClevrBatch
            ) -> Optional[dict[str, float]]:

        return sort_of_clevr_accuracy(out['logits'], batch['answers'])
    
    @classmethod
    def from_config(
        cls, cfg: Config, cb_cfg: AccuracyCallbackCfg
        ) -> sort_of_clevr_accuracy_callback:
        
        return cls()
    
class sort_of_clevr_qtype_accuracy_callback(BaseCallBack):    
    
    def on_train_step_end(
            self, trainer, out: SortOfClevrOutput, batch: SortOfClevrBatch
            ) -> Optional[dict[str, float]]:

        return sort_of_clevr_qtype_accuracy(
            out['logits'], batch['answers'], batch['questions']
        )

    def on_eval_step_end(
            self, trainer, out: SortOfClevrOutput, batch: SortOfClevrBatch
            ) -> Optional[dict[str, float]]:

        return sort_of_clevr_qtype_accuracy(
            out['logits'], batch['answers'], batch['questions']
        )

    def on_test_step_end(
            self, trainer, out: SortOfClevrOutput, batch: SortOfClevrBatch
            ) -> Optional[dict[str, float]]:

        return sort_of_clevr_qtype_accuracy(
            out['logits'], batch['answers'], batch['questions']
        )
    
    @classmethod
    def from_config(
        cls, cfg: Config, cb_cfg: QtypeAccuracyCallbackCfg
        ) -> sort_of_clevr_qtype_accuracy_callback:

        return cls()
    
        
sort_of_clevr_metric_callbacks: dict[str, CallbackSpec] = {
    'accuracy': CallbackSpec(
        config=AccuracyCallbackCfg,
        callback_class=sort_of_clevr_accuracy_callback,
    ),
    'qtype_accuracy': CallbackSpec(
        config=QtypeAccuracyCallbackCfg,
        callback_class=sort_of_clevr_qtype_accuracy_callback,
    ),
}