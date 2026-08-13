from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

class VQAQuestionOnly(nn.Module):

    def __init__(
            self, 
            answer_dim: int, 
            question_dim: int,
            hidden_dims: list[int]
            ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        d = question_dim
        for dim in hidden_dims:
            layers += [nn.Linear(d, dim), nn.GELU()]
            d = dim

        layers.append(nn.Linear(d, answer_dim))
        self.head = nn.Sequential(*layers)

    def forward(
        self, questions: torch.Tensor
        ) -> Tensor:
        
        return self.head(questions.float())
