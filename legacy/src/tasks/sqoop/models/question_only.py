from __future__ import annotations

from dataclasses import dataclass, field

from ....models.question_only import VQAQuestionOnly
from ....core.config import ModelConfig
from ..config import SqoopDataConfig
from ..contracts import SqoopOutput, SqoopBatch
from ..data import constants as C

@dataclass
class SQOOPQuestionOnlyConfig(ModelConfig):
    name: str = 'sqoop_question_only'
    hidden_dims: list[int] = field(default_factory=list)

class SQOOPQuestionOnly(VQAQuestionOnly):

    def __init__(
            self, 
            answer_dim: int,
            question_dim: int,
            hidden_dims: list[int]
        ) -> None:
        
        super().__init__(
            answer_dim=answer_dim,
            question_dim=question_dim,
            hidden_dims=hidden_dims
        )

    def forward(
            self, batch: SqoopBatch, **overrides
        ) -> SqoopOutput:

        logits = super().forward(batch['questions'])
        return {'logits': logits}

    @classmethod
    def from_config(
        cls, 
        cfg: SQOOPQuestionOnlyConfig,
        data_cfg: SqoopDataConfig
        ) -> SQOOPQuestionOnly:

        return cls(
            C.ANSWER_SIZE, 
            C.QUESTION_LEN, 
            cfg.hidden_dims
            )
