from __future__ import annotations

from dataclasses import dataclass

from ....models import ImageQuestionAdapter, IdentityQuestionEncoder
from ....models.film import VQAFiLM, VQAFiLMConfig
from ....models.encoders import build_encoder
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput, SortOfClevrBatch
from ..data import constants as C


@dataclass
class SortOfClevrFiLMConfig(VQAFiLMConfig):
    name: str = 'sort_of_clevr_film'


class SortOfClevrFiLM(ImageQuestionAdapter):
    """FiLM (Perez et al. 2018). Distinct from the
    q_conditioning='film' axis: that is one modulation at the
    encoder output, this is a separate modulation per block."""


    def forward(self, batch: SortOfClevrBatch, **overrides) -> SortOfClevrOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SortOfClevrFiLMConfig, data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrFiLM:
        inner = VQAFiLM(
            q_encoder=IdentityQuestionEncoder(C.QUESTION_SIZE),
            encoder=build_encoder(dict(cfg.encoder), int(data_cfg.img_size)),
            answer_dim=C.ANSWER_SIZE,
            n_blocks=int(cfg.n_blocks),
            block_ch=int(cfg.block_ch),
            classifier_ch=int(cfg.classifier_ch),
            classifier_hidden=int(cfg.classifier_hidden),
            use_coords=bool(cfg.use_coords),
        )
        return cls(inner)
