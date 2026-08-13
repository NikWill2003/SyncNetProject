from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..core.config import ModelConfig
from .encoders import build_encoder
from .question_encoders import QuestionEncoder


@dataclass
class VQAConvLSTMConfig(ModelConfig):
    """No-routing floor. Axes kept minimal on purpose: this arm exists to
    be beaten, and tuning it into a strong model defeats its function."""

    lstm_hidden: int = 128
    hidden_dim: int = 256

    encoder: dict[str, Any] = field(default_factory=lambda: {
        'name': 'cnn', 'ch': 128, 'hidden': 64,
    })


class VQAConvLSTM(nn.Module):
    """Bahdanau et al.'s no-routing floor: the question's LSTM state is
    broadcast over the *spatial* feature map, fused by convs, and
    flattened into the head.

    Flattened, not pooled. Global pooling before the head discards
    absolute position, and on SQOOP's hard negatives -- where both the
    positive and the negative scene contain x, y, and the same relation
    holding for *some* pair -- position is the entire label signal, so a
    pooled variant is pinned at chance by construction rather than by
    lacking capacity.
    """

    has_rotors = False
    is_syncnet = False

    def __init__(
            self,
            q_encoder: QuestionEncoder,
            encoder: nn.Module,
            answer_dim: int,
            lstm_hidden: int = 128,
            hidden_dim: int = 256,
            ) -> None:
        super().__init__()
        self.q_encoder = q_encoder
        self.encoder = encoder
        self.lstm = nn.LSTM(q_encoder.emb_dim, lstm_hidden, batch_first=True)

        ch = encoder.ch
        self.pos_emb = nn.Parameter(
            0.02 * torch.randn(1, ch, encoder.spatial, encoder.spatial)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch + lstm_hidden, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 16, 3, padding=1), nn.ReLU(),
        )
        n_tok = encoder.spatial * encoder.spatial
        self.head = nn.Sequential(
            nn.Linear(16 * n_tok, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, answer_dim),
        )

    def forward(self, images: Tensor, questions: Tensor, **kwargs) -> Tensor:
        feats = self.encoder(images) + self.pos_emb          # (B, ch, H, W)
        _, (h_n, _) = self.lstm(self.q_encoder(questions))   # (1, B, Hq)
        _, _, H, W = feats.shape
        q_map = h_n[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        fused = self.fuse(torch.cat([feats, q_map], dim=1))
        return self.head(fused.flatten(1))

    @staticmethod
    def build_encoder(spec: dict, img_size: int) -> nn.Module:
        return build_encoder(spec, img_size)
