"""Question encoders: the task-varying half of a VQA model.

Sort-of-CLEVR and SQOOP differ in exactly one place upstream of the
architecture -- how a question is represented:

    sort_of_clevr   a fixed-width one-hot/float vector, QUESTION_SIZE wide
    sqoop           three token indices [x, rel, y] over a 40-token vocab

Everything downstream (patch encoder, transformer, syncnet dynamics,
readout) is identical. So the models in this package take a
`QuestionEncoder` rather than a question width, and a task adapter's only
architectural job is to supply the right one.

The contract is a *sequence*: forward returns (B, L, D). Most conditioning
modes want the flattened (B, L*D) vector and call `.flat()`; the
`token_seq` transformer mode wants the L tokens as sequence positions.
Sort-of-CLEVR simply has L = 1.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class QuestionEncoder(nn.Module):
    """Base: maps a task's raw question tensor to (B, seq_len, emb_dim)."""

    seq_len: int
    emb_dim: int

    @property
    def out_dim(self) -> int:
        """Width of the flattened question vector."""
        return self.seq_len * self.emb_dim

    def forward(self, questions: Tensor) -> Tensor:
        raise NotImplementedError

    def flat(self, questions: Tensor) -> Tensor:
        """(B, seq_len * emb_dim) -- the conditioning vector."""
        return self.forward(questions).flatten(1)


class IdentityQuestionEncoder(QuestionEncoder):
    """Sort-of-CLEVR: the question vector is already a float feature.

    Wrapped as a length-1 sequence so that the sequence-valued
    conditioning modes work on both tasks without branching.
    """

    def __init__(self, question_size: int) -> None:
        super().__init__()
        self.seq_len = 1
        self.emb_dim = question_size

    def forward(self, questions: Tensor) -> Tensor:
        return questions.float().unsqueeze(1)


class TokenEmbedQuestionEncoder(QuestionEncoder):
    """SQOOP: embed the [x, rel, y] token indices.

    Note the alternative -- feeding the three indices straight into a
    Linear as floats -- treats a categorical vocabulary as an ordinal
    scale, so 'B' would sit between 'A' and 'C' by construction. That is
    not a small inefficiency on SQOOP: the systematic split turns on
    which *pairs* were seen, so any spurious metric over shape identity
    hands the model unearned generalisation across pairs.
    """

    def __init__(
            self, vocab_size: int, question_len: int, emb_dim: int
            ) -> None:
        super().__init__()
        self.seq_len = question_len
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)

    def forward(self, questions: Tensor) -> Tensor:
        return self.embed(questions.long())
