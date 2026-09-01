from typing import Any

import numpy as np
import torch

from .spec import *
COLOURS_KEY = list(COLOURS.keys())

# TODO: simplify this

def to_cpu_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


def one_hot_idx(x: torch.Tensor, start: int, end: int) -> int:
    return int(x[start:end].argmax().item())


def answer_idx(answer: torch.Tensor | np.ndarray | int) -> int:
    a = to_cpu_tensor(answer)

    if a.ndim == 0:
        return int(a.item())

    if a.numel() == 1:
        return int(a.item())

    if a.ndim == 1 and a.shape[0] == ANSWER_SIZE:
        looks_one_hot = (
            a.dtype.is_floating_point
            or int((a != 0).sum().item()) == 1
        )
        if looks_one_hot:
            return int(a.argmax().item())
        raise ValueError(
            'Ambiguous answer of shape (ANSWER_SIZE,): pass a scalar '
            'label, or a float one-hot vector.'
        )

    raise ValueError(f'Could not decode answer with shape {tuple(a.shape)}.')


def translate_question(question: torch.Tensor | np.ndarray) -> str:
    q = to_cpu_tensor(question)

    if q.shape != (QUESTION_SIZE,):
        raise ValueError(
            f'Expected question shape {(QUESTION_SIZE,)}, got {tuple(q.shape)}.'
        )

    colour_1 = COLOURS_KEY[one_hot_idx(q, 0, 6)]
    q_type = one_hot_idx(q, Q_TYPE_IDX, Q_TYPE_IDX + 3)
    subtype = one_hot_idx(q, SUB_Q_TYPE_IDX, SUB_Q_TYPE_IDX + 3)

    colour_2 = (
        COLOURS_KEY[one_hot_idx(q, 6, 12)] if q_type == 2 else None
    )

    if q_type == 0:
        if subtype == 0:
            return f'What shape is the {colour_1} object?'
        if subtype == 1:
            return f'Is the {colour_1} object on the left side of the image?'
        if subtype == 2:
            return f'Is the {colour_1} object in the top half of the image?'

    if q_type == 1:
        if subtype == 0:
            return f'What shape is the object closest to the {colour_1} object?'
        if subtype == 1:
            return f'What shape is the object furthest from the {colour_1} object?'
        if subtype == 2:
            return (
                f'How many objects have the same shape as the {colour_1} object, '
                f'excluding itself?'
            )

    if q_type == 2:
        if subtype == 0:
            return (
                f'How many objects lie inside the box between the {colour_1} '
                f'object and the {colour_2} object?'
            )
        if subtype == 1:
            return (
                f'Is there any object on the band between the {colour_1} '
                f'object and the {colour_2} object?'
            )
        if subtype == 2:
            return (
                f'How many objects form an obtuse triangle with the {colour_1} '
                f'object and the {colour_2} object?'
            )

    raise ValueError(
        f'Unknown question type/subtype: q_type={q_type}, subtype={subtype}.'
    )


def translate_answer(
    answer: torch.Tensor | np.ndarray | int,
    question: torch.Tensor | np.ndarray | None = None,
) -> str:
    idx = answer_idx(answer)

    if question is None:
        return ANSWERS[idx]

    q = to_cpu_tensor(question)

    if q.shape != (QUESTION_SIZE,):
        raise ValueError(
            f'Expected question shape {(QUESTION_SIZE,)}, got {tuple(q.shape)}.'
        )

    q_type = one_hot_idx(q, Q_TYPE_IDX, Q_TYPE_IDX + 3)
    subtype = one_hot_idx(q, SUB_Q_TYPE_IDX, SUB_Q_TYPE_IDX + 3)

    count_question = (
        (q_type == 1 and subtype == 2)
        or (q_type == 2 and subtype in {0, 2})
    )

    if count_question:
        if idx < COUNT_OFFSET:
            raise ValueError(
                f'count question with non-count answer index {idx}'
            )
        return COUNT_ANSWERS[idx]

    return ANSWERS[idx]