from __future__ import annotations

from dataclasses import dataclass

from ....core.config import ModelConfig
from ....models import (
    ImageQuestionAdapter, TokenEmbedQuestionEncoder, VQATransformer,
)
from ..config import SqoopDataConfig
from ..contracts import SqoopOutput, SqoopBatch
from ..data import constants as C


@dataclass
class SqoopTransformerConfig(ModelConfig):
    name: str = 'sqoop_transformer'

    # question encoder
    emb_dim: int = 32

    # patch encoder. patch_emb_dim < hidden_dim is required by
    # broadcast_cat (the question occupies the remaining width), and
    # broadcast_cat is the default, so the default must satisfy it.
    patch_emb_dim: int = 96
    patch_size: int = 8

    # transformer
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 2
    dropout: float = 0.0

    pos_enc: str = 'learnt_2d'      # learnt_1d | learnt_2d

    # film | broadcast_cat | token | token_seq
    # token_seq keeps [x, rel, y] as three sequence positions
    q_conditioning: str = 'broadcast_cat'

    share_layer_weights: bool = False

    # cls | mean | flatten -- see VQATransformer. cls is a learned pooling.
    readout: str = 'cls'

    # patchify = a single strided Conv2d, i.e. a linear projection of raw
    # pixels per patch. cnn = a multi-layer conv stack. Kept as a plain
    # string rather than a dict so it is sweepable: a dict literal in a
    # hydra sweeper line is split on its own commas.
    encoder_name: str = 'patchify'    # patchify | cnn
    encoder_hidden: int = 64          # cnn only


class SqoopTransformer(ImageQuestionAdapter):

    def forward(self, batch: SqoopBatch, **overrides) -> SqoopOutput:
        return super().forward(batch, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(
            cls, cfg: SqoopTransformerConfig, data_cfg: SqoopDataConfig,
            ) -> SqoopTransformer:
        inner = VQATransformer(
            q_encoder=TokenEmbedQuestionEncoder(
                C.VOCAB_SIZE, C.QUESTION_LEN, int(cfg.emb_dim),
            ),
            img_size=int(data_cfg.img_size),
            answer_size=C.ANSWER_SIZE,
            patch_size=int(cfg.patch_size),
            patch_emb_dim=int(cfg.patch_emb_dim),
            hidden_dim=int(cfg.hidden_dim),
            n_heads=int(cfg.n_heads),
            n_layers=int(cfg.n_layers),
            ffn_mult=int(cfg.ffn_mult),
            dropout=float(cfg.dropout),
            pos_enc=str(cfg.pos_enc),
            q_conditioning=str(cfg.q_conditioning),
            share_layer_weights=bool(cfg.share_layer_weights),
            readout=str(cfg.readout),
            encoder=(
                None if str(cfg.encoder_name) == 'patchify'
                else {'name': str(cfg.encoder_name),
                      'ch': int(cfg.patch_emb_dim),
                      'hidden': int(cfg.encoder_hidden)}
            ),
        )
        return cls(inner)
