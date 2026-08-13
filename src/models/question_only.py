from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .question_encoders import QuestionEncoder


class VQAQuestionOnly(nn.Module):
    """Guessing floor: question in, answer out, no image pathway.

    Its job is to measure the label information carried by the question
    alone. On SQOOP that number is a dataset test rather than a model
    result -- the generator balances labels exactly 50/50 per (x, rel, y),
    so anything meaningfully above 0.500 means the generator leaks.
    """

    def __init__(
            self,
            q_encoder: QuestionEncoder,
            answer_dim: int,
            hidden_dims: list[int],
            ) -> None:
        super().__init__()
        self.q_encoder = q_encoder

        layers: list[nn.Module] = []
        d = q_encoder.out_dim
        for dim in hidden_dims:
            layers += [nn.Linear(d, dim), nn.GELU()]
            d = dim
        layers.append(nn.Linear(d, answer_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, questions: Tensor) -> Tensor:
        return self.head(self.q_encoder.flat(questions))
