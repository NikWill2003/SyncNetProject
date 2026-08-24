from __future__ import annotations

from dataclasses import dataclass

from ....models import IdentityQuestionEncoder, ImageQuestionAdapter
from ....models.phasebind import PhaseBind, PhaseBindConfig
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput, SortOfClevrBatch
from ..data import constants as C


@dataclass
class SortOfClevrPhaseBindConfig(PhaseBindConfig):
    name: str = 'sort_of_clevr_phasebind'


class SortOfClevrPhaseBind(ImageQuestionAdapter):

    is_syncnet = True

    def forward(self, batch: SortOfClevrBatch, **overrides) -> SortOfClevrOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SortOfClevrPhaseBindConfig, data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrPhaseBind:
        inner = PhaseBind(
            cfg,
            q_encoder=IdentityQuestionEncoder(C.QUESTION_SIZE),
            img_size=int(data_cfg.img_size),
            answer_dim=C.ANSWER_SIZE,
            object_colours=list(C.COLOURS.values()),
        )
        return cls(inner)
