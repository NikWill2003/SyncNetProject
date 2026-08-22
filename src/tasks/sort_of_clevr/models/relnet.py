from __future__ import annotations

from dataclasses import dataclass

from ....models import ImageQuestionAdapter, IdentityQuestionEncoder
from ....models.relnet import VQARelNet, VQARelNetConfig
from ....models.encoders import build_encoder
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput, SortOfClevrBatch
from ..data import constants as C


@dataclass
class SortOfClevrRelNetConfig(VQARelNetConfig):
    name: str = 'sort_of_clevr_relnet'


class SortOfClevrRelNet(ImageQuestionAdapter):
    """Relation Network (Santoro et al. 2017). Conditioning is
    intrinsic -- the question enters each pair inside g_theta -- so
    this model does not take the q_conditioning axis."""


    def forward(self, batch: SortOfClevrBatch, **overrides) -> SortOfClevrOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SortOfClevrRelNetConfig, data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrRelNet:
        inner = VQARelNet(
            q_encoder=IdentityQuestionEncoder(C.QUESTION_SIZE),
            encoder=build_encoder(dict(cfg.encoder), int(data_cfg.img_size)),
            answer_dim=C.ANSWER_SIZE,
            g_hidden=int(cfg.g_hidden),
            g_layers=int(cfg.g_layers),
            f_hidden=int(cfg.f_hidden),
            f_dropout=float(cfg.f_dropout),
            pair_spatial=int(cfg.pair_spatial),
        )
        return cls(inner)
