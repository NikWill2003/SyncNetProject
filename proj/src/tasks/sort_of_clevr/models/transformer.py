from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
import torch
from torch import Tensor
from omegaconf import MISSING

from ....core.config import ModelConfig
from ....core.encoders import (
    PatchifyEncoder, CNNEncoder, EncoderConfig
    )
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput
from ..data import constants as C


@dataclass
class SortOfClevrTransformerConfig(ModelConfig):
    name: str = 'sort_of_clevr_transformer'
    forward_args: dict[str, Any] = field(default_factory=dict)

    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 2
    dropout: float = 0.0

    encoder_cfg: EncoderConfig = MISSING

Encoder = PatchifyEncoder | CNNEncoder

class SortOfClevrTransformer(nn.Module):
    """Tokens + question + CLS -> self-attention -> CLS readout."""

    def __init__(
            self, 
            encoder: Encoder,
            answer_dim: int, 
            q_dim: int, 
            hidden_dim: int, 
            n_heads: int,
            n_layers: int, 
            ffn_mult: int, 
            dropout: float
            ):
        
        super().__init__()
        
        self.encoder = encoder
        self.token_dim = encoder.ch
        self.n_tokens = encoder.n_tokens
        
        self.tok_proj = (
            nn.Linear(encoder.ch, hidden_dim) if self.token_dim != hidden_dim else nn.Identity()
            )
        
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, self.n_tokens, hidden_dim) * 0.02)
        self.q_pos_enc = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * ffn_mult, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
            )
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.GELU(),
            nn.Linear(hidden_dim, answer_dim),
        )

    def forward(
            self, images: Tensor, questions: Tensor, **kwargs
            ) -> SortOfClevrOutput:

        B, D = images.size(0), self.token_dim
        
        tokens = self.encoder(images).permute(0, 2, 3, 1).view((B, -1, D))

        # add positional encodings:
        tok = self.tok_proj(tokens) + self.pos_enc
        q = self.q_proj(questions.float()).unsqueeze(1) + self.q_pos_enc
        
        cls = self.cls.expand(B, -1, -1)
        
        # input is cls token, input tokens and question token:
        x = torch.cat([cls, tok, q], dim=1)
        x = self.blocks(x)
        
        # return feed the cls token through the head to get the answer
        return {
            'logits': self.head(x[:, 0])
            }
    
    @classmethod
    def from_config(
        cls, 
        cfg: SortOfClevrTransformerConfig, 
        data_cfg: SortOfClevrDataConfig,
        ) -> SortOfClevrTransformer:

        if cfg.encoder_cfg.name == 'patchify':
            encoder = PatchifyEncoder.from_config(
                cfg.encoder_cfg, img_size=data_cfg.img_size # type: ignore
                )
            
        elif cfg.encoder_cfg.name == 'cnn':
            encoder = CNNEncoder.from_config(
                cfg.encoder_cfg, img_size=data_cfg.img_size # type: ignore
            )
        else:
            raise ValueError(
                f'sort-of-clevr transformer does not'
                f'support the encoder: {cfg.encoder_cfg.name}'
                )

        return SortOfClevrTransformer(
            encoder=encoder,
            answer_dim=C.ANSWER_SIZE,
            q_dim=C.QUESTION_SIZE,
            hidden_dim=cfg.hidden_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            ffn_mult=cfg.ffn_mult,
            dropout=cfg.dropout,
        )