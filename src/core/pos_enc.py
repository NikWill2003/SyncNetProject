import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoder1D(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int) -> None:
        super().__init__()

        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, hidden_dim) * 0.02
        )

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (B, T, H)

        return tokens + self.pos_encoding


class PositionalEncoder2D(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        row_len: int,
        col_len: int,
    ) -> None:
        super().__init__()

        self.row_len = row_len
        self.col_len = col_len

        # (1, R, 1, H)
        self.row_pos_encoding = nn.Parameter(
            torch.randn(1, row_len, 1, hidden_dim) * 0.02
        )

        # (1, 1, C, H)
        self.col_pos_encoding = nn.Parameter(
            torch.randn(1, 1, col_len, hidden_dim) * 0.02
        )

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (B, T, H)
        # assumes flattened row-major spatial tokens

        # broadcast add makes cartesian product
        pos = self.row_pos_encoding + self.col_pos_encoding 
        pos = pos.flatten(1, 2) # (1, R*C, H)

        return tokens + pos