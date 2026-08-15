from __future__ import annotations

from dataclasses import dataclass

from ....models import ImageQuestionAdapter, TokenEmbedQuestionEncoder
from ....models.conv_lstm import VQAConvLSTM, VQAConvLSTMConfig
from ....models.encoders import build_encoder
from ..config import SqoopDataConfig
from ..contracts import SqoopOutput, SqoopBatch
from ..data import constants as C


@dataclass
class SqoopConvLSTMConfig(VQAConvLSTMConfig):
    name: str = 'sqoop_conv_lstm'
    emb_dim: int = 32


class SqoopConvLSTM(ImageQuestionAdapter):

    def forward(self, batch: SqoopBatch, **overrides) -> SqoopOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SqoopConvLSTMConfig, data_cfg: SqoopDataConfig,
            ) -> SqoopConvLSTM:
        inner = VQAConvLSTM(
            q_encoder=TokenEmbedQuestionEncoder(
                C.VOCAB_SIZE, C.QUESTION_LEN, int(cfg.emb_dim),
            ),
            encoder=build_encoder(dict(cfg.encoder), int(data_cfg.img_size)),
            answer_dim=C.ANSWER_SIZE,
            lstm_hidden=int(cfg.lstm_hidden),
            hidden_dim=int(cfg.hidden_dim),
            use_pos_emb=bool(cfg.use_pos_emb),
            readout=str(cfg.readout),
        )
        return cls(inner)
