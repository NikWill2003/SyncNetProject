from __future__ import annotations

from dataclasses import dataclass

from ....models import ImageQuestionAdapter, TokenEmbedQuestionEncoder
from ....models.relnet import VQARelNet, VQARelNetConfig
from ....models.encoders import build_encoder
from ..config import SqoopDataConfig
from ..contracts import SqoopOutput, SqoopBatch
from ..data import constants as C


@dataclass
class SqoopRelNetConfig(VQARelNetConfig):
    name: str = 'sqoop_relnet'
    emb_dim: int = 32


class SqoopRelNet(ImageQuestionAdapter):
    """Relation Network (Santoro et al. 2017). Conditioning is
    intrinsic -- the question enters each pair inside g_theta -- so
    this model does not take the q_conditioning axis."""


    def forward(self, batch: SqoopBatch, **overrides) -> SqoopOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SqoopRelNetConfig, data_cfg: SqoopDataConfig,
            ) -> SqoopRelNet:
        inner = VQARelNet(
            q_encoder=TokenEmbedQuestionEncoder(
                C.VOCAB_SIZE, C.QUESTION_LEN, int(cfg.emb_dim),
            ),
            encoder=build_encoder(dict(cfg.encoder), int(data_cfg.img_size)),
            answer_dim=C.ANSWER_SIZE,
            g_hidden=int(cfg.g_hidden),
            g_layers=int(cfg.g_layers),
            f_hidden=int(cfg.f_hidden),
            f_dropout=float(cfg.f_dropout),
            pair_spatial=int(cfg.pair_spatial),
        )
        return cls(inner)
