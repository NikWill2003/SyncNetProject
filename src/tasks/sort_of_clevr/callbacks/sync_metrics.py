from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass

from ....core.callbacks import BaseCallBack
from ....core.config import CallbackConfig
from ....core.registry import CallbackSpec
from ..contracts import SortOfClevrBatch, SortOfClevrOutput

if TYPE_CHECKING:
    from ....training import Trainer


def _unwrap(model):
    # torch.compile / accelerate may wrap the module
    return getattr(model, '_orig_mod', model)


def order_parameter(rotors: Tensor, rotor_dim: int) -> float:
    """Kuramoto order parameter generalised to n-dim rotors.

    rotors: (B, ch, H, W). Per rotor r: R_r = || spatial mean of unit
    vectors ||, in [0, 1]. 1 = all locations synchronised, ~0 = incoherent.
    Returns mean over batch and rotors.
    """
    B, ch, H, W = rotors.shape
    n = rotor_dim
    x = rotors.reshape(B, ch // n, n, H * W)
    x = F.normalize(x, dim=2)                    # safety for v3 states
    mean_vec = x.mean(dim=-1)                    # (B, R, n)
    return mean_vec.norm(dim=-1).mean().item()


def routing_entropy(attn: Tensor) -> float:
    """Normalised entropy of routing distributions.

    attn: (B, M, P), rows sum to 1. Returns mean entropy / log(P),
    so 1 = uniform routing, 0 = one-hot routing.
    """
    P = attn.shape[-1]
    ent = -(attn.clamp(min=1e-12) * attn.clamp(min=1e-12).log()).sum(-1)
    return (ent / torch.log(torch.tensor(float(P)))).mean().item()


class sort_of_clevr_sync_metrics_callback(BaseCallBack):
    """Synchrony diagnostics. No-op for models without `is_syncnet`.

    Eval/test only (extra forward passes). Reports:
      order_param        final-step bottom order parameter (rotor models)
      order_param_gain   final minus first step (did dynamics synchronise?)
      routing_entropy    final-step routing entropy, normalised to [0, 1]
      attn_max           mean max routing weight (peakedness)
      scrambled_accuracy accuracy with spatially permuted phases/state
      scramble_drop      accuracy - scrambled_accuracy. The causal test:
                         ~0 means the module layer does not use spatially
                         structured bottom information for this batch.
    """

    def _compute(
            self, trainer: Trainer,
            images: Tensor, questions: Tensor, answers: Tensor,
            ) -> Optional[dict[str, float]]:

        model = _unwrap(trainer.model)
        if not getattr(model, 'is_syncnet', False):
            return None

        metrics: dict[str, float] = {}

        with torch.inference_mode():
            b = {'images': images, 'questions': questions}
            out = model(b, return_trace=True)

            traces = out['traces']
            attn_key = 'attn'
            rotor_key = 'rotors' if 'rotors' in traces else 'state'

            if getattr(model, 'has_rotors', False):
                r_first = order_parameter(
                    traces[rotor_key][0], model.rotor_dim
                )
                r_last = order_parameter(
                    traces[rotor_key][-1], model.rotor_dim
                )
                metrics['order_param'] = r_last
                metrics['order_param_gain'] = r_last - r_first

            metrics['routing_entropy'] = routing_entropy(
                traces[attn_key][-1]
            )
            metrics['attn_max'] = (
                traces[attn_key][-1].max(dim=-1).values.mean().item()
            )

            # causal intervention: destroy image-aligned structure in what
            # the module layer reads, preserve everything else
            scramble_kwarg = (
                {'scramble_phase': True}
                if getattr(model, 'has_rotors', False)
                else {'scramble_state': True}
            )
            out_s = model(b, **scramble_kwarg)

            acc = (
                out['logits'].argmax(-1) == answers
            ).float().mean().item()
            acc_s = (
                out_s['logits'].argmax(-1) == answers
            ).float().mean().item()

            metrics['scrambled_accuracy'] = acc_s
            metrics['scramble_drop'] = acc - acc_s

        return metrics

    def on_eval_step_end(
            self, trainer: Trainer,
            out: SortOfClevrOutput, batch: SortOfClevrBatch,
            ) -> Optional[dict[str, float]]:
        return self._compute(
            trainer, batch['images'], batch['questions'], batch['answers']
        )

    def on_test_step_end(
            self, trainer: Trainer,
            out: SortOfClevrOutput, batch: SortOfClevrBatch,
            ) -> Optional[dict[str, float]]:
        return self._compute(
            trainer, batch['images'], batch['questions'], batch['answers']
        )


@dataclass
class SyncMetricsCallbackCfg(CallbackConfig):
    name: str = 'sync_metrics'


sort_of_clevr_sync_metric_callbacks: dict[str, CallbackSpec] = {
    'sync_metrics': CallbackSpec(
        config=SyncMetricsCallbackCfg,
        callback_class=sort_of_clevr_sync_metrics_callback,
    ),
}
