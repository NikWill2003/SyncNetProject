"""Shared Workspace transformer (Goyal et al., ICLR 2022, arXiv:2103.01197).

The workspace mechanism is adapted from the authors' released code at
github.com/anirudh9119/shared_workspace (`sort_of_clevr/transformer_utilities/relational_memory_volatile.py`)
"""

from __future__ import annotations
 
import math
from dataclasses import dataclass
from typing import cast
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
 
from .transformer import VQATransformer, VQATransformerConfig
 
 
@dataclass
class VQAWorkspaceTransformerConfig(VQATransformerConfig):
    name: str = 'workspace_transformer'
 
    share_layer_weights: bool = True
 
    mem_slots: int = 8
    topk_frac: float = 0.2
    persist_memory: bool = True
 
    key_size: int = 32
    attention_mlp_layers: int = 4
    gate_style: str = 'unit'
 
 
class GroupLinear(nn.Module):
    """One independent affine map per workspace slot."""
 
    def __init__(self, d_in: int, d_out: int, n_blocks: int) -> None:
        super().__init__()
 
        a = 1.0 / math.sqrt(d_out)
        self.weight = nn.Parameter(torch.empty(n_blocks, d_in, d_out).uniform_(-a, a))
        self.bias = nn.Parameter(torch.empty(n_blocks, d_out).uniform_(-a, a))
 
    def forward(self, x: Tensor) -> Tensor:
        return torch.bmm(x.transpose(0, 1), self.weight).transpose(0, 1) + self.bias
 
 
class MeanPoolGate(nn.Module):
    """Input half of the write gate.
 
    Averages over the token axis, so the gate sees the aggregate input and
    cannot tell which token supplied what. That is what the original does.
    """
 
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
 
        self.w = nn.Parameter(torch.randn(d_in))
        self.linear = nn.Linear(d_in, d_out)
 
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(torch.relu(self.w * x).mean(dim=1))
 
 
class SharedWorkspace(nn.Module):
    """Write with competition, then broadcast.
 
    The projections are shared between the two directions: `attend` always
    takes queries from `sink` and keys and values from `source`, and the two
    steps simply swap which is which.
    """
 
    def __init__(self, cfg: VQAWorkspaceTransformerConfig) -> None:
        super().__init__()
 
        if cfg.gate_style not in ('unit', 'memory', None):
            raise ValueError(f'gate_style must be unit or memory or None, got {cfg.gate_style!r}')
 
        self.n_heads = cfg.n_heads
        self.mem_slots = cfg.mem_slots
        self.mem_size = cfg.hidden_dim
        self.head_size = cfg.hidden_dim // cfg.n_heads
        self.key_size = cfg.key_size
        self.topk_frac = cfg.topk_frac
        self.gate_style = cfg.gate_style
 
        self.query_proj = nn.Linear(self.mem_size, self.key_size * self.n_heads)
        self.key_proj = nn.Linear(self.mem_size, self.key_size * self.n_heads)
        self.value_proj = nn.Linear(self.mem_size, self.head_size * self.n_heads)
 
        self.attention_mlp = nn.ModuleList(
            nn.Linear(self.mem_size, self.mem_size) for _ in range(cfg.attention_mlp_layers)
        )
        self.norm1 = nn.LayerNorm(self.mem_size)
        self.norm2 = nn.LayerNorm(self.mem_size)
 
        if self.gate_style in ('unit', 'memory'):
            n_gates = 2 * (self.mem_size if self.gate_style == 'unit' else 1)
            self.input_gate = MeanPoolGate(self.mem_size, n_gates)
            self.memory_gate = GroupLinear(self.mem_size, n_gates, self.mem_slots)
 
        self.forget_bias = nn.Parameter(torch.tensor(1.0))
        self.input_bias = nn.Parameter(torch.tensor(0.0))
 
    def attend(self, source: Tensor, sink: Tensor, compete: bool) -> Tensor:
        def split(x: Tensor) -> Tensor:
            return x.reshape(x.size(0), x.size(1), self.n_heads, -1).permute(0, 2, 1, 3)
 
        query = split(self.query_proj(sink))
        key = split(self.key_proj(source))
        value = split(self.value_proj(source))
 
        scores = torch.softmax(torch.matmul(query, key.transpose(2, 3)), dim=-1)
 
        if compete:
            n_source = scores.size(-1)
            topk = max(1, min(round(self.topk_frac * n_source), n_source))
            indices = torch.topk(scores, k=topk, dim=-1).indices
            scores = scores * torch.zeros_like(scores).scatter_(-1, indices, 1.0)
 
        attended = torch.matmul(scores, value).permute(0, 2, 1, 3).contiguous()
 
        return attended.reshape(attended.size(0), attended.size(1), -1)
 
    def forward(self, tokens: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        new_memory = self.norm1(memory + self.attend(tokens, memory, compete=True))
 
        projected = new_memory
        for layer in self.attention_mlp:
            projected = F.relu(layer(projected))
 
        new_memory = self.norm2(new_memory + projected)
 
        if self.gate_style in ('unit', 'memory'):
            gates = self.memory_gate(torch.tanh(memory)) + self.input_gate(tokens).unsqueeze(1)
            input_gate, forget_gate = torch.split(gates, gates.size(-1) // 2, dim=-1)
 
            new_memory = (
                torch.sigmoid(input_gate + self.input_bias) * torch.tanh(new_memory)
                + torch.sigmoid(forget_gate + self.forget_bias) * memory
            )
 
        return self.attend(new_memory, tokens, compete=False), new_memory
 
 
class WorkspaceEncoderLayer(nn.Module):
    """Pre-norm encoder block with the workspace in place of self-attention.
 
    Structurally identical to the nn.TransformerEncoderLayer the base model
    builds: norm_first, gelu, same feedforward width and dropout placement.
    """
 
    def __init__(self, cfg: VQAWorkspaceTransformerConfig) -> None:
        super().__init__()
 
        self.workspace = SharedWorkspace(cfg)
 
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.linear1 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * cfg.ffn_mult)
        self.linear2 = nn.Linear(cfg.hidden_dim * cfg.ffn_mult, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
 
    def forward(self, tokens: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        attended, memory = self.workspace(self.norm1(tokens), memory)
        tokens = tokens + self.dropout1(attended)
 
        projected = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(tokens)))))
        tokens = tokens + self.dropout2(projected)
 
        return tokens, memory
 
 
class VQAWorkspaceTransformer(VQATransformer):
 
    layers: nn.ModuleList
 
    def build_layers(self, cfg: VQATransformerConfig) -> None:
        cfg = cast(VQAWorkspaceTransformerConfig, cfg)
 
        self.persist_memory = cfg.persist_memory
        self.mem_slots = cfg.mem_slots
        self.mem_size = cfg.hidden_dim
 
        if cfg.share_layer_weights:
            shared = WorkspaceEncoderLayer(cfg)
            self.layers = nn.ModuleList([shared] * cfg.n_layers)
        else:
            self.layers = nn.ModuleList(
                WorkspaceEncoderLayer(cfg) for _ in range(cfg.n_layers)
            )
 
    def initial_memory(self, tokens: Tensor) -> Tensor:
        batch_size = tokens.size(0)
 
        state = torch.eye(self.mem_slots, device=tokens.device, dtype=tokens.dtype)
        state = state.unsqueeze(0).expand(batch_size, -1, -1)
 
        pad = self.mem_size - self.mem_slots
 
        if pad > 0:
            zeros = torch.zeros(
                batch_size, self.mem_slots, pad, device=tokens.device, dtype=tokens.dtype
            )
            state = torch.cat([state, zeros], dim=-1)
 
        return state[:, :, :self.mem_size].contiguous()
 
    def run_layers(self, tokens: Tensor) -> Tensor:
        memory = self.initial_memory(tokens)
 
        for layer in self.layers:
            tokens, new_memory = layer(tokens, memory)
            memory = new_memory if self.persist_memory else self.initial_memory(tokens)
 
        return tokens
 
