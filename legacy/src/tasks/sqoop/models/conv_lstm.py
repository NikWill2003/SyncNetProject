from __future__ import annotations

from dataclasses import dataclass, field, make_dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import MISSING

@dataclass
class SqoopConvLSTMConfig(ModelConfig):
    name: str = 'sqoop_conv_lstm'

    emb_dim: int = 32
    lstm_hidden: int = 128
    hidden_dim: int = 256

    encoder_cfg: EncoderConfig = MISSING


class SqoopConvLSTM(nn.Module):
    """No-routing floor after Bahdanau et al.: question LSTM state
    broadcast over the *spatial* feature map, fused by convs, flattened
    into the head. (The first version global-pooled the map, which on
    SQOOP's hard negatives provably carries zero label signal -- fixed
    2026-08; position enters via a learned spatial embedding.)"""

    has_rotors = False
    is_syncnet = False

    def __init__(
            self,
            encoder,
            emb_dim: int = 32,
            lstm_hidden: int = 128,
            hidden_dim: int = 256,
            ):
        super().__init__()
        self.encoder = encoder
        self.embed = nn.Embedding(VOCAB_SIZE, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden, batch_first=True)
        ch = encoder.ch
        self.pos_emb = nn.Parameter(
            0.02 * torch.randn(1, ch, encoder.spatial, encoder.spatial)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch + lstm_hidden, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, 16, 3, padding=1),
            nn.ReLU(),
        )
        n_tok = encoder.spatial * encoder.spatial
        self.head = nn.Sequential(
            nn.Linear(16 * n_tok, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ANSWER_SIZE),
        )

    def forward(
            self, images: torch.Tensor, questions: torch.Tensor, **kwargs
            ) -> SqoopOutput:
        feats = self.encoder(images) + self.pos_emb          # (B, ch, H, W)
        _, (h_n, _) = self.lstm(self.embed(questions))       # (1, B, Hq)
        B, _, H, W = feats.shape
        q_map = h_n[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        fused = self.fuse(torch.cat([feats, q_map], dim=1))
        logits = self.head(fused.flatten(1))
        return {'logits': logits}

    @classmethod
    def from_config(
            cls, cfg: SqoopConvLSTMConfig, data_cfg: SqoopDataConfig
            ) -> 'SqoopConvLSTM':
        encoder = _build_encoder(cfg.encoder_cfg, int(data_cfg.img_size))
        return cls(
            encoder,
            emb_dim=int(cfg.emb_dim),
            lstm_hidden=int(cfg.lstm_hidden),
            hidden_dim=int(cfg.hidden_dim),
        )
