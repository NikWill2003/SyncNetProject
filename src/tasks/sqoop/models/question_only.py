from __future__ import annotations

from dataclasses import dataclass

from ....core.config import ModelConfig
from ....models import (
    QuestionOnlyAdapter, TokenEmbedQuestionEncoder, VQAQuestionOnly,
)
from ..config import SqoopDataConfig
from ..contracts import SqoopOutput, SqoopBatch
from ..data import constants as C


@dataclass
class SqoopQuestionOnlyConfig(ModelConfig):
    name: str = 'sqoop_question_only'
    emb_dim: int = 32
    hidden_dim: int = 128
    n_layers: int = 2


class SqoopQuestionOnly(QuestionOnlyAdapter):
    """Leakage gate, not a baseline.

    SQOOP's generator balances labels exactly 50/50 within every
    (x, rel, y) cell, so the Bayes-optimal question-only accuracy is
    0.500 by construction. This model converging anywhere meaningfully
    above that is a statement about the *dataset*, and every downstream
    SQOOP number is void until it reads 0.500.
    """

    def forward(self, batch: SqoopBatch, **overrides) -> SqoopOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SqoopQuestionOnlyConfig, data_cfg: SqoopDataConfig,
            ) -> SqoopQuestionOnly:
        inner = VQAQuestionOnly(
            q_encoder=TokenEmbedQuestionEncoder(
                C.VOCAB_SIZE, C.QUESTION_LEN, int(cfg.emb_dim),
            ),
            answer_dim=C.ANSWER_SIZE,
            hidden_dims=[int(cfg.hidden_dim)] * int(cfg.n_layers),
        )
        return cls(inner)
