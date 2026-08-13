from __future__ import annotations

from dataclasses import dataclass

from ....models import (
    IdentityQuestionEncoder, ImageQuestionAdapter, VQASyncNet,
    VQASyncNetConfig,
)
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput, SortOfClevrBatch
from ..data import constants as C


@dataclass
class SortOfClevrSyncNetConfig(VQASyncNetConfig):
    """All axes live on the shared config; the task only names it."""
    name: str = 'sort_of_clevr_syncnet'


class SortOfClevrSyncNet(ImageQuestionAdapter):

    is_syncnet = True

    def forward(
            self, batch: SortOfClevrBatch, **overrides
            ) -> SortOfClevrOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls,
            cfg: SortOfClevrSyncNetConfig,
            data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrSyncNet:
        inner = VQASyncNet(
            cfg,
            q_encoder=IdentityQuestionEncoder(C.QUESTION_SIZE),
            img_size=int(data_cfg.img_size),
            answer_dim=C.ANSWER_SIZE,
        )
        return cls(inner)
