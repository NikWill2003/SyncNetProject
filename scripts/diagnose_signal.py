#!/usr/bin/env python3
"""Is the .500 flatline a PLATEAU or NOISE CANCELLATION?

Both look identical in a loss curve and need opposite fixes:

  plateau   : gradients are tiny and aligned-but-useless -- the surface is
              flat, so no optimiser setting rescues it. Fix = change the
              task/curriculum, or give the model a binding prior.
  noise     : gradients are LARGE per batch but point in unrelated
              directions, so the average cancels. Fix = optimisation
              (bigger batch, lower LR, longer warmup, sharper competition).

Method: take K independent batches, compute the full gradient of each, then
report per-batch norms, the norm of their mean, and pairwise cosines.

    SNR = ||mean_i g_i|| / mean_i ||g_i||

  SNR ~ 1      -> batches agree; there IS a consistent descent direction
  SNR ~ 1/sqrt(K) -> pure noise; batch gradients cancel (the balance story)
  all norms ~0 -> genuine plateau

Compare a model that flatlines (canonical) against one that works (gated),
and sqoop against Sort-of-CLEVR, on the SAME instrument.

    python scripts/diagnose_signal.py task=sqoop \\
        experiment=sqoop/sync/thesis/canonical dataset.train_size=5184 \\
        dataset.test_size=5184
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.registry import (build_dataloaders, build_loss_fn,  # noqa: E402
                               build_model, register_configs)

K_BATCHES = 8


def flat_grad(model) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()])


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.train.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(cfg).to(device)
    loaders = build_dataloaders(cfg, device)
    train = loaders[0] if isinstance(loaders, (tuple, list)) else loaders['train']
    loss_fn = build_loss_fn(cfg)

    grads, losses, accs = [], [], []
    it = iter(train)
    for _ in range(K_BATCHES):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train)
            batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(batch)
        res = loss_fn(out if isinstance(out, dict) else {'logits': out}, batch)
        loss = res[0] if isinstance(res, (tuple, list)) else res
        logits = out['logits'] if isinstance(out, dict) else out
        loss.backward()
        grads.append(flat_grad(model).detach().float().cpu())
        losses.append(float(loss))
        accs.append(float((logits.argmax(-1) == batch['answers']).float().mean()))

    G = torch.stack(grads)
    norms = G.norm(dim=1)
    mean_norm = G.mean(0).norm()
    snr = float(mean_norm / norms.mean())
    Gn = G / (norms[:, None] + 1e-12)
    cos = (Gn @ Gn.T)
    off = cos[~torch.eye(len(G), dtype=bool)]

    print('\n================ gradient signal diagnosis ================')
    print(f'model              : {cfg.model.name}   task: {cfg.dataset.name}')
    print(f'batches            : {K_BATCHES} x {cfg.train.train_bs}')
    print(f'loss               : {sum(losses)/len(losses):.4f}   (ln2 = 0.6931)')
    print(f'batch accuracy     : {sum(accs)/len(accs):.4f}')
    print(f'mean ||g_i||       : {norms.mean():.4e}   (per-batch gradient size)')
    print(f'||mean g||         : {mean_norm:.4e}   (what the optimiser follows)')
    print(f'SNR                : {snr:.4f}   (1/sqrt(K) = {1/K_BATCHES**0.5:.4f} means pure noise)')
    print(f'pairwise cosine    : {off.mean():+.4f} +/- {off.std():.4f}')
    print('----------------------------------------------------------')
    if norms.mean() < 1e-6:
        print('VERDICT: PLATEAU -- gradients are ~zero. No optimiser setting helps;')
        print('         the task gives no descent direction from this init.')
    elif snr < 2.0 / K_BATCHES ** 0.5:
        print('VERDICT: NOISE CANCELLATION -- large per-batch gradients that')
        print('         disagree, so the average vanishes. This is the exact-')
        print('         balance story: partial features cancel. Fixes are')
        print('         optimisation-side (bigger batch, lower LR, longer')
        print('         warmup) or curriculum (easier split first).')
    else:
        print('VERDICT: CONSISTENT SIGNAL -- batches agree on a direction, so')
        print('         the flatline is NOT a gradient problem: look at the')
        print('         readout, the loss, or label handling instead.')
    print('==========================================================\n')


if __name__ == '__main__':
    register_configs()
    main()
