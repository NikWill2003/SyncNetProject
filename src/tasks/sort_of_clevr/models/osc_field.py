from __future__ import annotations

from dataclasses import dataclass

from ....models import IdentityQuestionEncoder, ImageQuestionAdapter
from ....models.osc_field import OscField, OscFieldConfig
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput, SortOfClevrBatch
from ..data import constants as C


@dataclass
class SortOfClevrOscFieldConfig(OscFieldConfig):
    name: str = 'sort_of_clevr_osc_field'


class SortOfClevrOscField(ImageQuestionAdapter):

    is_syncnet = True

    def forward(self, batch: SortOfClevrBatch, **overrides) -> SortOfClevrOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SortOfClevrOscFieldConfig, data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrOscField:
        inner = OscField(
            cfg,
            q_encoder=IdentityQuestionEncoder(C.QUESTION_SIZE),
            img_size=int(data_cfg.img_size),
            answer_dim=C.ANSWER_SIZE,
        )
        return cls(inner)
