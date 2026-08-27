"""Pairwise multi-question Sort-of-CLEVR.

Each scene is drawn by the standard generator (same colours, sizes,
separation rule) and carries `n_questions` PAIRWISE questions about
DISJOINT pairs of objects:

    left_of      is A left of B          (A.x < B.x)
    above        is A above B            (A.y < B.y)
    same_shape   do A and B share a shape

A pairwise question needs exactly the two named modules to exchange
information, so `n_questions` questions are `n_questions` disjoint
conversations, which is the coalition structure a shared medium must
separate (two groups 90 degrees apart on S^1; a third would leak). The standard binary
questions would not do: "closest to red" needs the hub to hear every
object, so two of them share their leaves.

Question layout follows the task's QUESTION_SIZE=18 vector so the same
colour slots and subtype slots are used: q[A] = 1, q[6 + B] = 1,
q[Q_TYPE_IDX + 1] = 1 (binary-type flag), q[SUB_Q_TYPE_IDX + subtype] = 1.
Answers: 0 = yes, 1 = no.

make_pairwise_dataset -> dict of numpy arrays:
    images     (N, H, W, 3) uint8, BGR as the task stores them
    questions  (N, n_questions, 18) float32
    answers    (N, n_questions) int64
    positions  (N, 6, 2) int16 (x, y) by colour index
    shapes     (N, 6) uint8, 1 = rectangle
"""

from __future__ import annotations

import random

import numpy as np

from . import constants as C
from .generator import generate_sample

PAIR_SUBTYPES = ('left_of', 'above', 'same_shape')


def _pair_question(objects, a: int, b: int, subtype: int) -> tuple[np.ndarray, int]:
    q = np.zeros(C.QUESTION_SIZE, dtype=np.float32)
    n = len(C.COLOURS)
    q[a] = 1
    q[n + b] = 1
    q[C.Q_TYPE_IDX + 1] = 1
    q[C.SUB_Q_TYPE_IDX + subtype] = 1
    pa, pb = objects[a][1], objects[b][1]
    if subtype == 0:
        yes = pa[0] < pb[0]
    elif subtype == 1:
        yes = pa[1] < pb[1]
    else:
        yes = objects[a][2] == objects[b][2]
    return q, 0 if yes else 1


def make_pairwise_dataset(n_scenes: int, n_questions: int = 2, img_size: int = 75,
                          obj_size: int = 5, seed: int = 0) -> dict[str, np.ndarray]:
    n = len(C.COLOURS)
    if 2 * n_questions > n:
        raise ValueError(f'at most {n // 2} disjoint pairs among {n} objects')
    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    images, questions, answers, positions, shapes = [], [], [], [], []
    for _ in range(n_scenes):
        img, _t, _b, _n, objects = generate_sample(img_size, obj_size, 1, -1)
        perm = rng.permutation(n)
        qs, ans = [], []
        for k in range(n_questions):
            a, b = int(perm[2 * k]), int(perm[2 * k + 1])
            q, y = _pair_question(objects, a, b, int(rng.integers(len(PAIR_SUBTYPES))))
            qs.append(q); ans.append(y)
        images.append(img)
        questions.append(np.stack(qs))
        answers.append(np.array(ans, dtype=np.int64))
        positions.append(np.array([o[1] for o in objects], dtype=np.int16))
        shapes.append(np.array([1 if o[2] == 'r' else 0 for o in objects], dtype=np.uint8))
    return {
        'images': np.stack(images), 'questions': np.stack(questions),
        'answers': np.stack(answers), 'positions': np.stack(positions), 'shapes': np.stack(shapes),
    }


def to_tensors(data: dict[str, np.ndarray], device='cpu'):
    """images -> float (N, 3, H, W) in [0, 1] (channel order kept, as the task
    loader does); questions -> float (N, n_q, 18); answers -> long (N, n_q)."""
    import torch
    x = torch.from_numpy(data['images']).permute(0, 3, 1, 2).float().div_(255).to(device)
    q = torch.from_numpy(data['questions']).float().to(device)
    y = torch.from_numpy(data['answers']).long().to(device)
    return x, q, y


def make_pairwise_from_npz(path, n_questions: int = 2, seed: int = 0,
                           max_scenes: int | None = None) -> dict[str, np.ndarray]:
    """Pairwise questions on the scenes of an existing split (train.npz /
    val.npz / test.npz), using the stored `object_positions` and
    `object_shapes`. Same scenes as every other model; only the questions
    differ. The stored questions/answers are ignored."""
    d = np.load(path)
    images, pos, shp = d['images'], d['object_positions'], d['object_shapes']
    if max_scenes is not None:
        images, pos, shp = images[:max_scenes], pos[:max_scenes], shp[:max_scenes]
    n = len(C.COLOURS)
    if 2 * n_questions > n:
        raise ValueError(f'at most {n // 2} disjoint pairs among {n} objects')
    rng = np.random.default_rng(seed)
    questions = np.zeros((len(images), n_questions, C.QUESTION_SIZE), dtype=np.float32)
    answers = np.zeros((len(images), n_questions), dtype=np.int64)
    for i in range(len(images)):
        objects = [(c, pos[i, c], 'r' if shp[i, c] == 1 else 'c') for c in range(n)]
        perm = rng.permutation(n)
        for k in range(n_questions):
            a, b = int(perm[2 * k]), int(perm[2 * k + 1])
            questions[i, k], answers[i, k] = _pair_question(objects, a, b, int(rng.integers(len(PAIR_SUBTYPES))))
    return {'images': images, 'questions': questions, 'answers': answers,
            'positions': pos.astype(np.int16), 'shapes': shp.astype(np.uint8)}
