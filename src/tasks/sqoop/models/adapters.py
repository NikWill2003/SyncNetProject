"""SQOOP models.

SQOOP questions are 3 token indices [x, rel, y] over a 40-token vocab,
answers binary -- vs sort_of_clevr's one-hot question vectors and 10-way
answers. Rather than duplicating the syncnet architectures, `SqoopAdapter`
wraps a sort_of_clevr model class: it embeds the 3 tokens, flattens to a
q_dim = 3 * emb_dim float vector, and constructs the inner model with
that q_dim and answer_dim = 2. Inner models treat `questions` as an
opaque float conditioning vector (FiLM + h/state init), so nothing inside
them changes. `t_override` / `return_trace` / `scramble_state` pass
through, and the adapter exposes `.T`, so the t_variance callback works
unchanged.

`SqoopConvLSTM` is the no-routing floor from Bahdanau et al.: CNN pool +
LSTM over embedded question tokens -> MLP. It has no `T`, which also
exercises t_variance's skip path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, make_dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import MISSING

from ....core.config import ModelConfig
from ....core.encoders import (
    PatchifyEncoder, CNNEncoder, EncoderConfig,
    PatchifyEncoderConfig, CNNEncoderConfig,
)
# SQOOP wraps the Sort-of-CLEVR models: the same architecture, with the
# [x, rel, y] token triple embedded and flattened into the question
# vector the inner model expects.
from ...sort_of_clevr.models.syncnet import (
    SortOfClevrSyncNet, SortOfClevrSyncNetConfig, _to_grouped,
)
from ..config import SqoopDataConfig
from ..contracts import SqoopOutput
from ..data.constants import VOCAB_SIZE, QUESTION_LEN, ANSWER_SIZE


def _build_encoder(encoder_cfg, img_size: int):
    if encoder_cfg.name == 'patchify':
        return PatchifyEncoder.from_config(encoder_cfg, img_size=img_size)
    if encoder_cfg.name == 'cnn':
        return CNNEncoder.from_config(encoder_cfg, img_size=img_size)
    raise ValueError(f'unsupported encoder: {encoder_cfg.name}')


# ---------------------------------------------------------------- adapter

class SqoopAdapter(nn.Module):

    is_syncnet = False

    def __init__(self, inner: nn.Module, emb_dim: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, emb_dim)
        self.inner = inner

    @property
    def T(self) -> int:
        return self.inner.T

    @property
    def has_rotors(self) -> bool:
        return getattr(self.inner, 'has_rotors', False)

    def forward(
            self, images: torch.Tensor, questions: torch.Tensor, **kwargs
            ) -> SqoopOutput:
        q = self.embed(questions).flatten(1)  # (B, 3*emb)
        return self.inner(images, q, **kwargs)


def _adapted_config(base_config_cls, model_name: str):
    """Subclass a sort_of_clevr model config, adding emb_dim and
    overriding the registry name. (make_dataclass because class bodies
    cannot close over enclosing-function names.)"""
    cls_name = 'Sqoop' + base_config_cls.__name__.replace(
        'SortOfClevr', ''
    )
    return make_dataclass(
        cls_name,
        [
            ('name', str, field(default=model_name)),
            ('emb_dim', int, field(default=32)),
        ],
        bases=(base_config_cls,),
    )


SqoopSyncNetConfig = _adapted_config(
    SortOfClevrSyncNetConfig, 'sqoop_syncnet'
)


def _make_adapter_class(inner_cls, arch_fields: list[str]):
    """Adapter class with a from_config building the given inner model."""

    class _Adapter(SqoopAdapter):

        @classmethod
        def from_config(cls, cfg, data_cfg: SqoopDataConfig) -> '_Adapter':
            encoder = _build_encoder(cfg.encoder_cfg, int(data_cfg.img_size))
            kwargs = {f: getattr(cfg, f) for f in arch_fields}
            inner = inner_cls(
                encoder=encoder,
                answer_dim=ANSWER_SIZE,
                q_dim=QUESTION_LEN * int(cfg.emb_dim),
                **kwargs,
            )
            return cls(inner, emb_dim=int(cfg.emb_dim))

    _Adapter.__name__ = 'Sqoop' + inner_cls.__name__.replace(
        'SortOfClevr', ''
    )
    _Adapter.__qualname__ = _Adapter.__name__
    return _Adapter


class SqoopSyncNet(SqoopAdapter):
    """The unified Sort-of-CLEVR SyncNet on SQOOP.

    The inner model builds its own encoder from data_cfg.img_size, so
    (unlike the older adapters) no encoder is passed in; SqoopDataConfig
    supplies img_size just as the SOC one does.
    """

    @classmethod
    def from_config(cls, cfg, data_cfg: SqoopDataConfig) -> 'SqoopSyncNet':
        inner = SortOfClevrSyncNet(
            _to_grouped(cfg),
            data_cfg,                                    # type: ignore[arg-type]
            q_dim=QUESTION_LEN * int(cfg.emb_dim),
            answer_dim=ANSWER_SIZE,
        )
        return cls(inner, emb_dim=int(cfg.emb_dim))


# ---------------------------------------------------------------- floor

@dataclass
class SqoopConvLSTMConfig(ModelConfig):
    name: str = 'sqoop_conv_lstm'

    emb_dim: int = 32
    lstm_hidden: int = 128
    hidden_dim: int = 256

    encoder_cfg: EncoderConfig = MISSING


class SqoopConvLSTM(nn.Module):
    """No-routing floor after Bahdanau et al.: question LSTM state
    broadcast over the *spatial* feature map, fused by convs, flattened
    into the head. (The first version global-pooled the map, which on
    SQOOP's hard negatives provably carries zero label signal -- fixed
    2026-08; position enters via a learned spatial embedding.)"""

    has_rotors = False
    is_syncnet = False

    def __init__(
            self,
            encoder,
            emb_dim: int = 32,
            lstm_hidden: int = 128,
            hidden_dim: int = 256,
            ):
        super().__init__()
        self.encoder = encoder
        self.embed = nn.Embedding(VOCAB_SIZE, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden, batch_first=True)
        ch = encoder.ch
        self.pos_emb = nn.Parameter(
            0.02 * torch.randn(1, ch, encoder.spatial, encoder.spatial)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch + lstm_hidden, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, 16, 3, padding=1),
            nn.ReLU(),
        )
        n_tok = encoder.spatial * encoder.spatial
        self.head = nn.Sequential(
            nn.Linear(16 * n_tok, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ANSWER_SIZE),
        )

    def forward(
            self, images: torch.Tensor, questions: torch.Tensor, **kwargs
            ) -> SqoopOutput:
        feats = self.encoder(images) + self.pos_emb          # (B, ch, H, W)
        _, (h_n, _) = self.lstm(self.embed(questions))       # (1, B, Hq)
        B, _, H, W = feats.shape
        q_map = h_n[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        fused = self.fuse(torch.cat([feats, q_map], dim=1))
        logits = self.head(fused.flatten(1))
        return {'logits': logits}

    @classmethod
    def from_config(
            cls, cfg: SqoopConvLSTMConfig, data_cfg: SqoopDataConfig
            ) -> 'SqoopConvLSTM':
        encoder = _build_encoder(cfg.encoder_cfg, int(data_cfg.img_size))
        return cls(
            encoder,
            emb_dim=int(cfg.emb_dim),
            lstm_hidden=int(cfg.lstm_hidden),
            hidden_dim=int(cfg.hidden_dim),
        )


# ---------------------------------------------------------------- floor 2

@dataclass
class SqoopQuestionOnlyConfig(ModelConfig):
    name: str = 'sqoop_question_only'
    emb_dim: int = 32
    hidden_dim: int = 128
    n_layers: int = 2


class SqoopQuestionOnly(nn.Module):
    """Question-only guessing floor. SQOOP's generator balances labels
    exactly 50/50 *per (x, rel, y) question*, so this model converging to
    0.50 on every split is a leakage test of the dataset itself: anything
    above 0.5 +- noise means question->label information exists without
    perception, i.e. a generator bug."""

    has_rotors = False
    is_syncnet = False

    def __init__(self, emb_dim: int = 32, hidden_dim: int = 128,
                 n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, emb_dim)
        layers: list[nn.Module] = []
        d = QUESTION_LEN * emb_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, ANSWER_SIZE))
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor, questions: torch.Tensor,
                **kwargs) -> SqoopOutput:
        del images  # deliberately unused
        return {'logits': self.net(self.embed(questions).flatten(1))}

    @classmethod
    def from_config(cls, cfg: 'SqoopQuestionOnlyConfig',
                    data_cfg: SqoopDataConfig) -> 'SqoopQuestionOnly':
        return cls(int(cfg.emb_dim), int(cfg.hidden_dim), int(cfg.n_layers))
