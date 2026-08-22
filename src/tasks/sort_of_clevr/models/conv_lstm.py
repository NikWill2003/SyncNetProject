from __future__ import annotations

from dataclasses import dataclass

from ....models import ImageQuestionAdapter, IdentityQuestionEncoder
from ....models.conv_lstm import VQAConvLSTM, VQAConvLSTMConfig
from ....models.encoders import build_encoder
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput, SortOfClevrBatch
from ..data import constants as C


@dataclass
class SortOfClevrCNNMLPConfig(VQAConvLSTMConfig):
    """CNN+MLP, the control from Santoro et al. 2017: it answers the
    non-relational questions and fails the relational ones.

    Same class as the SQOOP conv baseline, different preset. The question
    is pooled by an MLP rather than an LSTM (a Sort-of-CLEVR question is
    one fixed-width vector, not a sequence) and meets the image at the
    readout rather than being broadcast over the feature map.
    """
    name: str = 'sort_of_clevr_cnn_mlp'
    q_pool: str = 'mlp'
    fusion: str = 'readout'


class SortOfClevrCNNMLP(ImageQuestionAdapter):

    def forward(
            self, batch: SortOfClevrBatch, **overrides
            ) -> SortOfClevrOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls,
            cfg: SortOfClevrCNNMLPConfig,
            data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrCNNMLP:
        inner = VQAConvLSTM(
            q_encoder=IdentityQuestionEncoder(C.QUESTION_SIZE),
            encoder=build_encoder(dict(cfg.encoder), int(data_cfg.img_size)),
            answer_dim=C.ANSWER_SIZE,
            lstm_hidden=int(cfg.lstm_hidden),
            hidden_dim=int(cfg.hidden_dim),
            use_pos_emb=bool(cfg.use_pos_emb),
            readout=str(cfg.readout),
            q_pool=str(cfg.q_pool),
            fusion=str(cfg.fusion),
        )
        return cls(inner)
