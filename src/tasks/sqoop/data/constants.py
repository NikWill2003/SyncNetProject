"""Vocabulary and encoding constants for SQOOP.

Question encoding: every question is exactly [x, rel, y], stored as three
indices into a single joint vocabulary of SHAPES + RELATIONS. Shapes occupy
indices [0, N_SHAPES), relations [N_SHAPES, N_SHAPES + N_RELATIONS).
Answers are binary: 0 = false, 1 = true.
"""

from __future__ import annotations

import string

# Order matters: indices are persisted in the .npz files. Do not reorder.
# Matches the original generator (ascii uppercase, then digits 1-9 then 0).
SHAPES: list[str] = list(string.ascii_uppercase) + [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
]
RELATIONS: list[str] = ['left_of', 'right_of', 'above', 'below']

N_SHAPES = len(SHAPES)          # 36
N_RELATIONS = len(RELATIONS)    # 4
VOCAB_SIZE = N_SHAPES + N_RELATIONS  # 40

SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}
REL_TO_IDX = {r: N_SHAPES + i for i, r in enumerate(RELATIONS)}
IDX_TO_TOKEN = SHAPES + RELATIONS

QUESTION_LEN = 3
ANSWER_SIZE = 2  # false / true


def encode_question(x: str, rel: str, y: str) -> list[int]:
    return [SHAPE_TO_IDX[x], REL_TO_IDX[rel], SHAPE_TO_IDX[y]]


def decode_question(idxs) -> str:
    return ' '.join(IDX_TO_TOKEN[int(i)] for i in idxs)
