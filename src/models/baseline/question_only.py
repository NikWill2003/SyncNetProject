from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

import torch.nn as nn

from ...core.config import ModelConfig
from ...core.contracts import VQABatch, VQAOutput
from ..common.qst_enc import build_question_encoder


@dataclass
class VQAQuestionOnlyConfig(ModelConfig):
    name: str = 'question_only'
    q_encoder: dict[str, Any] = field(
        default_factory=lambda: {'name': 'identity'}
    )

    hidden_dim: int = 128
    n_layers: int = 2


class VQAQuestionOnly(nn.Module):

    allowed_q_encoders = {'identity', 'mlp', 'lstm'}

    def __init__(self, cfg: VQAQuestionOnlyConfig, dataset: str, answer_dim: int) -> None:
        super().__init__()

        self.question_encoder = build_question_encoder(cfg.q_encoder, dataset, self.allowed_q_encoders)
        
        dim = self.question_encoder.output_shape[0]

        layers: list[nn.Module] = []
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(dim, cfg.hidden_dim), nn.ReLU()]
            dim = cfg.hidden_dim

        layers.append(nn.Linear(dim, answer_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, batch: VQABatch, **overrides) -> VQAOutput:

        question = self.question_encoder(batch['questions'])

        return {'logits': self.head(question)}

    @classmethod
    def from_config(cls, model_cfg: VQAQuestionOnlyConfig, dataset: str, answer_dim: int,
        ) -> VQAQuestionOnly:

        return cls(model_cfg, dataset, answer_dim)