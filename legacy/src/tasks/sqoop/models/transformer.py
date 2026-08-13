from __future__ import annotations

from dataclasses import dataclass

from ....core.config import ModelConfig
from ....models.vqa_transformer import VQATransformer
from ..config import SqoopDataConfig
from ..contracts import SqoopOutput, SqoopBatch

@dataclass
class SQOOPTransformerConfig(ModelConfig):
    name: str = 'sqoop_transformer_config'
    # encoder
    patch_emb_dim: int = 128
    patch_size: int = 5

    # transformer
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 2
    dropout: float = 0.0

    # positional encoding
    pos_enc: str = 'learnt_2d' # learnt_1d | learnt_2d

    # where the question enters
    q_conditioning: str = 'token' # token_proj | token_emb | broadcast_cat | film

    share_layer_weights: bool = False # false -> normal transformer, true -> looped

class SQOOPTransformer(VQATransformer):
    def __init__(
            self, 
            img_size: int, 
            question_size: int, 
            answer_size: int,
            vocab_size: int,
            cfg: SQOOPTransformerConfig
        ) -> None:

        # model_dict_config if dataclass, but will likely be a omegaconf dict so that needs to be converted
        model_dict_config: dict = cfg.__dict__
        super().__init__(
            img_size=img_size,
            question_size=question_size,
            answer_size=answer_size,
            vocab_size=vocab_size,
            **model_dict_config
        )

    def forward(
        self, batch: SqoopBatch, **overrides
    ) -> SqoopOutput:
        logits = super().forward(
            batch['images'], batch['questions']
        )
        return {'logits': logits}

    @classmethod
    def from_config(
        cls,
        cfg: SQOOPTransformerConfig,
        data_cfg: SqoopDataConfig,
    ) -> SQOOPTransformer:
        from src.tasks.sqoop.data import constants as C
        
        return cls(
            img_size=data_cfg.img_size,
            question_size=C.QUESTION_LEN,
            answer_size=C.ANSWER_SIZE,
            vocab_size=C.VOCAB_SIZE,
            cfg=cfg,
            )
