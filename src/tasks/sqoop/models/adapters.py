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
from ...sort_of_clevr.models import (
    SortOfClevrSyncNetV1, SortOfClevrSyncNetV1Config,
    SortOfClevrSyncNetV3, SortOfClevrSyncNetV3Config,
    SortOfClevrRecurrentSyncNet, SortOfClevrRecurrentSyncNetConfig,
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


SqoopRecurrentSyncNetConfig = _adapted_config(
    SortOfClevrRecurrentSyncNetConfig, 'sqoop_recurrent_syncnet'
)
SqoopSyncNetV3Config = _adapted_config(
    SortOfClevrSyncNetV3Config, 'sqoop_syncnet_v3'
)
SqoopSyncNetV1Config = _adapted_config(
    SortOfClevrSyncNetV1Config, 'sqoop_syncnet_v1'
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


_RECURRENT_FIELDS = [
    'n_modules', 'module_dim', 'msg_dim', 'use_film', 'beta_init',
    'learn_beta', 'content_dim', 'query_hidden', 'T', 'dt', 'omega_init',
    'k_hidden', 'deterministic_phase', 'hidden_dim',
]
_V3_FIELDS = [
    'rotor_dim', 'use_film', 'ksize', 'n_modules', 'content_dim',
    'query_hidden', 'hidden_dim', 'T', 'gamma', 'dt', 'beta_init',
    'learn_beta', 'use_top_down', 'top_down_alpha_init',
]
_V1_FIELDS = [
    'rotor_dim', 'use_film', 'ksize', 'use_omega', 'init_omg',
    'global_omg', 'n_modules', 'content_dim', 'query_hidden',
    'hidden_dim', 'T', 'gamma', 'dt', 'beta_init', 'learn_beta',
]

SqoopRecurrentSyncNet = _make_adapter_class(
    SortOfClevrRecurrentSyncNet, _RECURRENT_FIELDS
)
SqoopSyncNetV3 = _make_adapter_class(SortOfClevrSyncNetV3, _V3_FIELDS)
SqoopSyncNetV1 = _make_adapter_class(SortOfClevrSyncNetV1, _V1_FIELDS)


# ---------------------------------------------------------------- floor

@dataclass
class SqoopConvLSTMConfig(ModelConfig):
    name: str = 'sqoop_conv_lstm'
    forward_args: dict[str, Any] = field(default_factory=dict)

    emb_dim: int = 32
    lstm_hidden: int = 128
    hidden_dim: int = 256

    encoder_cfg: EncoderConfig = MISSING


class SqoopConvLSTM(nn.Module):
    """No-routing floor: CNN global pool + question LSTM -> MLP."""

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
        self.head = nn.Sequential(
            nn.Linear(encoder.ch + lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ANSWER_SIZE),
        )

    def forward(
            self, images: torch.Tensor, questions: torch.Tensor, **kwargs
            ) -> SqoopOutput:
        feats = self.encoder(images).mean(dim=(-2, -1))      # (B, ch)
        _, (h_n, _) = self.lstm(self.embed(questions))       # (1, B, H)
        logits = self.head(torch.cat([feats, h_n[0]], dim=-1))
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
