from __future__ import annotations

from dataclasses import dataclass

from ....models import ImageQuestionAdapter
from ....models.busnet import BusNet, BusNetConfig
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput, SortOfClevrBatch
from ..data import constants as C


@dataclass
class SortOfClevrBusNetConfig(BusNetConfig):
    name: str = 'sort_of_clevr_busnet'


class SortOfClevrBusNet(ImageQuestionAdapter):

    is_syncnet = True

    def forward(self, batch: SortOfClevrBatch, **overrides) -> SortOfClevrOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SortOfClevrBusNetConfig, data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrBusNet:
        inner = BusNet(
            cfg,
            img_size=int(data_cfg.img_size),
            answer_dim=C.ANSWER_SIZE,
            object_colours=list(C.COLOURS.values()),
        )
        return cls(inner)
