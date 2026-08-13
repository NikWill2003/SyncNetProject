from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ....models.question_only import VQAQuestionOnly
from ....core.config import ModelConfig
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput, SortOfClevrBatch
from ..data import constants as C

@dataclass
class SortOfClevrQuestionOnlyConfig(ModelConfig):
    name: str = 'sort_of_clevr_question_only'
    hidden_dims: list[int] = field(default_factory=list)

class SortOfClevrQuestionOnly(VQAQuestionOnly):

    has_rotors = False
    is_syncnet = False

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
            self, batch: SortOfClevrBatch, **overrides
        ) -> SortOfClevrOutput:

        logits = super().forward(batch['questions'])
        return {'logits': logits}

    @classmethod
    def from_config(
        cls, 
        cfg: SortOfClevrQuestionOnlyConfig,
        data_cfg: SortOfClevrDataConfig
        ) -> SortOfClevrQuestionOnly:

        return cls(
            C.ANSWER_SIZE, 
            C.QUESTION_SIZE, 
            cfg.hidden_dims
            )
