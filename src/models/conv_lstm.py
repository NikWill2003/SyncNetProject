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

    # How the question is pooled to a vector, and where it meets the image.
    # These two axes make CNN+MLP and Conv+LSTM presets of one model:
    #   CNN+MLP    q_pool='mlp',  fusion='readout'   (Santoro et al. 2017)
    #   Conv+LSTM  q_pool='lstm', fusion='spatial'   (Bahdanau et al. 2019)
    # The off-diagonal settings are runnable too, which is the point: the
    # question pathway becomes a measured variable rather than a confound
    # between the two tasks' baselines.
    q_pool: str = 'lstm'       # mlp | lstm
    fusion: str = 'spatial'    # spatial | readout

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
            q_pool: str = 'lstm',
            fusion: str = 'spatial',
            ) -> None:
        super().__init__()
        if readout not in ('flatten', 'pool'):
            raise ValueError(f'readout must be flatten|pool, got {readout!r}')
        if q_pool not in ('mlp', 'lstm'):
            raise ValueError(f'q_pool must be mlp|lstm, got {q_pool!r}')
        if fusion not in ('spatial', 'readout'):
            raise ValueError(f'fusion must be spatial|readout, got {fusion!r}')
        self.readout = readout
        self.q_pool = q_pool
        self.fusion = fusion
        self.q_encoder = q_encoder
        self.encoder = encoder

        # Both branches emit a (B, lstm_hidden) vector, so the two settings
        # are interchangeable everywhere downstream.
        if q_pool == 'lstm':
            self.lstm = nn.LSTM(q_encoder.emb_dim, lstm_hidden,
                                batch_first=True)
        else:
            self.q_mlp = nn.Sequential(
                nn.Linear(q_encoder.out_dim, lstm_hidden), nn.ReLU(),
                nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(),
            )

        ch = encoder.ch
        # A CNN is translation-equivariant, so without this the model has
        # no access to absolute position -- and left_of/above are pure
        # position predicates.
        self.pos_emb = nn.Parameter(
            0.02 * torch.randn(1, ch, encoder.spatial, encoder.spatial)
        ) if use_pos_emb else None
        # Under `spatial` the question is broadcast over the map and enters
        # the convolutions; under `readout` the convolutions see the image
        # alone and the question is concatenated at the head.
        fuse_in = ch + lstm_hidden if fusion == 'spatial' else ch
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 16, 3, padding=1), nn.ReLU(),
        )
        n_tok = encoder.spatial * encoder.spatial
        in_dim = 16 * n_tok if readout == 'flatten' else 16
        if fusion == 'readout':
            in_dim += lstm_hidden
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, answer_dim),
        )

    def _question_vector(self, questions: Tensor) -> Tensor:
        if self.q_pool == 'lstm':
            _, (h_n, _) = self.lstm(self.q_encoder(questions))
            return h_n[0]                                    # (B, Hq)
        return self.q_mlp(self.q_encoder.flat(questions))     # (B, Hq)

    def forward(self, images: Tensor, questions: Tensor, **kwargs) -> Tensor:
        feats = self.encoder(images)                         # (B, ch, H, W)
        if self.pos_emb is not None:
            feats = feats + self.pos_emb
        q = self._question_vector(questions)
        if self.fusion == 'spatial':
            _, _, H, W = feats.shape
            q_map = q.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            fused = self.fuse(torch.cat([feats, q_map], dim=1))
        else:
            fused = self.fuse(feats)
        v = fused.mean(dim=(2, 3)) if self.readout == 'pool' \
            else fused.flatten(1)
        if self.fusion == 'readout':
            v = torch.cat([v, q], dim=1)
        return self.head(v)

    @staticmethod
    def build_encoder(spec: dict, img_size: int) -> nn.Module:
        return build_encoder(spec, img_size)
