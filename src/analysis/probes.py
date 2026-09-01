"""Registered-prediction probes, as a library.

memorisation_gate: can this architecture fit ARBITRARY labels on a small
fixed set of its own inputs? The plumbing certificate behind the SQOOP
diagnosis -- a model whose gradient path is alive passes easily; a model
whose perception collapses to a constant cannot tell its training items
apart and stays at chance no matter the labels. Registered predictions:
M1 and the token models pass; the pixel field models fail on
stimulus-starved inputs.

grad_decomposition: the fraction of gradient norm arriving in each named
component after one step on real data -- the cheap first look at which part
of a composition training is actually sculpting.

Pure functions over built models; no wandb, no trainer. verify/
verify_gates.py is the by-hand runner.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def memorisation_gate(model: torch.nn.Module, batch: dict, n_classes: int,
                      steps: int = 300, lr: float = 3e-4,
                      seed: int = 0) -> dict[str, float]:
    """Fit random labels on the given fixed batch. Returns first/last train
    accuracy and the verdict at the customary .9 bar."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, n_classes, (batch['answers'].shape[0],), generator=g)
    b = {**batch, 'answers': y}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    acc_first = acc = 0.0
    model.train()
    for t in range(steps):
        out = model(b)
        loss = F.cross_entropy(out['logits'], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        acc = (out['logits'].argmax(-1) == y).float().mean().item()
        if t == 0:
            acc_first = acc
    return {'acc_first': acc_first, 'acc_last': acc,
            'passes': float(acc > 0.9), 'steps': float(steps)}


def grad_decomposition(model: torch.nn.Module, batch: dict) -> dict[str, float]:
    """One backward on real data; per-top-level-component share of the
    global gradient norm."""
    model.zero_grad(set_to_none=True)
    out = model(batch)
    F.cross_entropy(out['logits'], batch['answers']).backward()
    per: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        head = name.split('.')[0]
        per[head] = per.get(head, 0.0) + float(p.grad.pow(2).sum())
    total = sum(per.values()) or 1.0
    return {k: v / total for k, v in sorted(per.items())}
