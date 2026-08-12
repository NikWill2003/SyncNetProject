from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ....core.config import ModelConfig
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput
from ..data import constants as C

@dataclass
class SortOfClevrQuestionOnlyConfig(ModelConfig):
    name: str = 'sort_of_clevr_question_only'
    hidden_dim: int = 128
    n_layers: int = 2


class SortOfClevrQuestionOnly(nn.Module):

    has_rotors = False
    is_syncnet = False

    def __init__(
            self, answer_dim: int, q_dim: int,
            hidden_dim: int = 128, n_layers: int = 2
            ) -> None:
        
        super().__init__()
        layers: list[nn.Module] = []
        d = q_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers.append(nn.Linear(d, answer_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, images: torch.Tensor, questions: torch.Tensor, **kwargs
        ) -> SortOfClevrOutput:
        
        return {'logits': self.net(questions.float())}

    @classmethod
    def from_config(
        cls, 
        cfg: SortOfClevrQuestionOnlyConfig,
        data_cfg: SortOfClevrDataConfig
        ) -> SortOfClevrQuestionOnly:

        return cls(
            C.ANSWER_SIZE, C.QUESTION_SIZE, cfg.hidden_dim, cfg.n_layers
            )
