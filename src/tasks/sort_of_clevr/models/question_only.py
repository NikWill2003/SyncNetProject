from __future__ import annotations

from dataclasses import dataclass

from ....core.config import ModelConfig
from ....models import (
    IdentityQuestionEncoder, QuestionOnlyAdapter, VQAQuestionOnly,
)
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput, SortOfClevrBatch
from ..data import constants as C


@dataclass
class SortOfClevrQuestionOnlyConfig(ModelConfig):
    name: str = 'sort_of_clevr_question_only'
    hidden_dim: int = 128
    n_layers: int = 2


class SortOfClevrQuestionOnly(QuestionOnlyAdapter):

    def forward(
            self, batch: SortOfClevrBatch, **overrides
            ) -> SortOfClevrOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls,
            cfg: SortOfClevrQuestionOnlyConfig,
            data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrQuestionOnly:
        inner = VQAQuestionOnly(
            q_encoder=IdentityQuestionEncoder(C.QUESTION_SIZE),
            answer_dim=C.ANSWER_SIZE,
            hidden_dims=[int(cfg.hidden_dim)] * int(cfg.n_layers),
        )
        return cls(inner)
