"""Metric callbacks for the coalitions task.

These are where the experiment's claims are actually measured. On eval/test
steps they report, per batch:

  per-regime accuracy   independent / joint / post-disconnect / readout, since
                        overall accuracy is dominated by independent steps
  gate_sep              E[g | required] - E[g | not required]   (selectivity)
  gate_auc              ROC-AUC of the (undirected) gate vs the oracle r_t
  offpair_leak          mean gate on required-OFF pairs during active episodes
                        -- the frustration signature: a phase model on a STAR
                        cannot suppress spoke-spoke leakage
  acc_joint__<GRAPH>    joint accuracy split by active graph
  rho_err_corr          Pearson corr across graphs between rho(G, d) and joint
                        ERROR -- the calibration axis. Positive & strong => the
                        phase model fails exactly where the geometry says it
                        must; ~0 for unconstrained gates.

Nothing here is a training target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
from torch import Tensor

from ....core.callbacks import BaseCallBack
from ....core.config import CallbackConfig
from ....core.registry import CallbackSpec
from ..contracts import CoalitionsBatch, CoalitionsOutput
from ..data import constants as C
from ..data.graphs import catalogue, pair_index

if TYPE_CHECKING:
    from ....training import Trainer


@dataclass
class CoalitionsMetricsCallbackCfg(CallbackConfig):
    name: str = 'coalitions_metrics'


def _unwrap(model):
    return getattr(model, '_orig_mod', model)


def _masked_acc(logits: Tensor, targets: Tensor, mask: Tensor) -> float:
    if mask.sum() == 0:
        return float('nan')
    pred = logits.argmax(-1)
    correct = ((pred == targets) & (mask > 0)).float().sum()
    return (correct / mask.sum()).item()


def _auc(scores: Tensor, labels: Tensor) -> float:
    """Rank-based ROC-AUC. scores, labels are 1-D, labels in {0, 1}."""
    pos = labels > 0.5
    n_pos = pos.sum().item()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device,
                                dtype=torch.float)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc.item()


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float('nan')
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    xt = xt - xt.mean()
    yt = yt - yt.mean()
    denom = xt.norm() * yt.norm()
    if denom < 1e-8:
        return float('nan')
    return (xt @ yt / denom).item()


class coalitions_metrics_callback(BaseCallBack):

    def _compute(
            self, trainer: Trainer, batch: CoalitionsBatch,
            ) -> Optional[dict[str, float]]:

        model = _unwrap(trainer.model)
        if not getattr(model, 'is_coalitions', False):
            return None

        streams = batch['streams']
        commands = batch['commands']
        targets = batch['targets']
        loss_mask = batch['loss_mask']
        regime = batch['regime']
        oracle_adj = batch['oracle_adj']
        active_gid = batch['active_gid']

        N = model.N
        cat = catalogue(N)
        empty_gid = next(i for i, g in enumerate(cat) if g.name == 'EMPTY')
        pidx = pair_index(N)
        pairs = list(pidx.keys())
        ii = torch.tensor([i for (i, _) in pairs], device=streams.device)
        jj = torch.tensor([j for (_, j) in pairs], device=streams.device)

        metrics: dict[str, float] = {}

        with torch.inference_mode():
            out = model(streams, commands, oracle_adj=oracle_adj,
                        return_trace=True)
            logits = out['logits']
            gate = out['traces']['gate']            # (B, T, N, N)

        # -- per-regime accuracy --
        for code, name in [
            (C.REGIME_INDEP, 'indep'),
            (C.REGIME_JOINT, 'joint'),
            (C.REGIME_POST, 'post'),
            (C.REGIME_READOUT, 'readout'),
        ]:
            m = loss_mask * (regime == code).float()
            metrics[f'acc_{name}'] = _masked_acc(logits, targets, m)
        metrics['accuracy'] = _masked_acc(logits, targets, loss_mask)

        # -- gate selectivity vs oracle --
        g_pair = 0.5 * (gate[..., ii, jj] + gate[..., jj, ii])   # (B,T,P) undirected
        req = oracle_adj                                          # (B,T,P) in {0,1}
        req_flat = req.reshape(-1)
        g_flat = g_pair.reshape(-1)
        on = req_flat > 0.5
        off = ~on
        if on.any():
            metrics['gate_on'] = g_flat[on].mean().item()
        if off.any():
            metrics['gate_off'] = g_flat[off].mean().item()
        if on.any() and off.any():
            metrics['gate_sep'] = (
                g_flat[on].mean() - g_flat[off].mean()
            ).item()
        metrics['gate_auc'] = _auc(g_flat, req_flat)

        # -- frustration signature: leakage on required-off pairs during
        #    active episodes (any edge required somewhere this step) --
        active_step = (active_gid != empty_gid)                  # (B, T)
        active_pair = active_step.unsqueeze(-1).expand_as(req)
        leak_mask = active_pair & (req < 0.5)
        if leak_mask.any():
            metrics['offpair_leak'] = g_pair[leak_mask].mean().item()

        # -- per-graph joint accuracy + rho-error calibration --
        rho_map = self._rho_for_model(trainer, model, cat)
        rho_vals, err_vals = [], []
        joint_mask_all = loss_mask * (regime == C.REGIME_JOINT).float()
        gid_grid = active_gid.unsqueeze(-1).expand(-1, -1, N)     # (B,T,N)
        for gi, g in enumerate(cat):
            if g.family == 'empty':
                continue
            gm = joint_mask_all * (gid_grid == gi).float()
            if gm.sum() == 0:
                continue
            acc = _masked_acc(logits, targets, gm)
            metrics[f'acc_joint__{g.name}'] = acc
            if rho_map is not None and g.name in rho_map:
                rho_vals.append(rho_map[g.name])
                err_vals.append(1.0 - acc)
        if len(rho_vals) >= 2:
            metrics['rho_err_corr'] = _pearson(rho_vals, err_vals)

        return metrics

    @staticmethod
    def _rho_for_model(trainer, model, cat) -> Optional[dict[str, float]]:
        loader = getattr(trainer, 'eval_dataloader', None)
        rho = getattr(loader, 'rho', None)
        if rho is None:
            return None
        # pick the ladder dimension matching the model's oscillator (phase
        # gate); default to d = 2 otherwise.
        d = getattr(getattr(model, 'gate', None), 'd', 2)
        dims = rho.get('dims', {})
        if d not in dims:
            if not dims:
                return None
            d = min(dims)          # closest available
        names = rho['names']
        vals = dims[d]
        return {str(n): float(v) for n, v in zip(names, vals)}

    def on_eval_step_end(
            self, trainer, out: CoalitionsOutput, batch: CoalitionsBatch,
            ):
        return self._compute(trainer, batch)

    def on_test_step_end(
            self, trainer, out: CoalitionsOutput, batch: CoalitionsBatch,
            ):
        return self._compute(trainer, batch)


coalitions_metric_callbacks: dict[str, CallbackSpec] = {
    'coalitions_metrics': CallbackSpec(
        config=CoalitionsMetricsCallbackCfg,
        callback_class=coalitions_metrics_callback,
    ),
}
