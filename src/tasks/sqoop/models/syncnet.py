from __future__ import annotations

from dataclasses import dataclass

from ....models import (
    ImageQuestionAdapter, TokenEmbedQuestionEncoder, VQASyncNet,
    VQASyncNetConfig,
)
from ..config import SqoopDataConfig
from ..contracts import SqoopOutput, SqoopBatch
from ..data import constants as C


@dataclass
class SqoopSyncNetConfig(VQASyncNetConfig):
    """Shared syncnet axes plus the token-embedding width.

    Note `q_emb_dim` (shared, the broadcast_cat projection) and `emb_dim`
    (here, the [x, rel, y] token embedding) are different things and both
    apply on SQOOP.
    """
    name: str = 'sqoop_syncnet'
    emb_dim: int = 32


class SqoopSyncNet(ImageQuestionAdapter):

    is_syncnet = True

    def forward(self, batch: SqoopBatch, **overrides) -> SqoopOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SqoopSyncNetConfig, data_cfg: SqoopDataConfig,
            ) -> SqoopSyncNet:
        inner = VQASyncNet(
            cfg,
            q_encoder=TokenEmbedQuestionEncoder(
                C.VOCAB_SIZE, C.QUESTION_LEN, int(cfg.emb_dim),
            ),
            img_size=int(data_cfg.img_size),
            answer_dim=C.ANSWER_SIZE,
        )
        return cls(inner)
