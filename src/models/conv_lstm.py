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

    # Both default to the SPATIALLY STRONG variant, which is what has been
    # run so far. Neither is in Bahdanau et al.'s Conv+LSTM, whose whole
    # function is to be spatially weak -- their bar sits off the top of a
    # 14% error axis at every #rhs, ours reached 0.41% at rhs=18. Set both
    # off to recover something like the published floor.
    use_pos_emb: bool = True   # learned absolute position on conv features
    readout: str = 'flatten'   # flatten | pool

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
            use_pos_emb: bool = True,
            readout: str = 'flatten',
            ) -> None:
        super().__init__()
        if readout not in ('flatten', 'pool'):
            raise ValueError(f'readout must be flatten|pool, got {readout!r}')
        self.readout = readout
        self.q_encoder = q_encoder
        self.encoder = encoder
        self.lstm = nn.LSTM(q_encoder.emb_dim, lstm_hidden, batch_first=True)

        ch = encoder.ch
        # A CNN is translation-equivariant, so without this the model has
        # no access to absolute position -- and left_of/above are pure
        # position predicates.
        self.pos_emb = nn.Parameter(
            0.02 * torch.randn(1, ch, encoder.spatial, encoder.spatial)
        ) if use_pos_emb else None
        self.fuse = nn.Sequential(
            nn.Conv2d(ch + lstm_hidden, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 16, 3, padding=1), nn.ReLU(),
        )
        n_tok = encoder.spatial * encoder.spatial
        in_dim = 16 * n_tok if readout == 'flatten' else 16
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, answer_dim),
        )

    def forward(self, images: Tensor, questions: Tensor, **kwargs) -> Tensor:
        feats = self.encoder(images)                         # (B, ch, H, W)
        if self.pos_emb is not None:
            feats = feats + self.pos_emb
        _, (h_n, _) = self.lstm(self.q_encoder(questions))   # (1, B, Hq)
        _, _, H, W = feats.shape
        q_map = h_n[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        fused = self.fuse(torch.cat([feats, q_map], dim=1))
        if self.readout == 'pool':
            return self.head(fused.mean(dim=(2, 3)))
        return self.head(fused.flatten(1))

    @staticmethod
    def build_encoder(spec: dict, img_size: int) -> nn.Module:
        return build_encoder(spec, img_size)
