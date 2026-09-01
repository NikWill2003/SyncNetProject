"""Test-time interventions: is the mechanism load-bearing?

Run once at the end of training on the best weights. For each named
intervention the trained model is evaluated on the test split with one
runtime override, and the accuracy drop relative to an un-intervened pass
over the same batches is logged, overall and per question family.

    gate_open      every channel open            was selectivity used?
    gate_zero      no messages                   were messages used at all?
    gate_frozen    gate held at its t=0 value    did the dynamics matter?
    gate_shuffle   another sample's gate         was the gate input-dependent?
    phase_freeze   no phase dynamics
    phase_shuffle  module phases permuted        did phase identity matter?
    phase_zero     all phases set to 0           BusNet: the open bus at test time
    tok_freeze     token phases not evolved      (PhaseBind)
    tok_shuffle    token phases permuted         (PhaseBind / OscField:
                                                  destroys the binding)
    lambda0        phase term removed from read  (PhaseBind / OscField)

A model declares which override values it implements (GATE_OVERRIDES,
PHASE_OVERRIDES on the inner model); anything else is skipped rather than
silently reported as a zero drop.

Logged to the wandb summary as test_interventions/<name>_accuracy,
<name>_drop, <name>_ternary_accuracy, <name>_ternary_drop, and
baseline_accuracy / baseline_ternary_accuracy for the matched pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

from ..core.callbacks import BaseCallBack
from ..core.config import CallbackConfig
from ..core.callbacks import CallbackSpec
from ..training.logging import section
from ..datasets.soc import spec as C
from .sync_metrics import _unwrap

if TYPE_CHECKING:
    from ..training import Trainer


DEFAULT_INTERVENTIONS: list[dict[str, str]] = [
    {'name': 'gate_open', 'gate_override': 'open'},
    {'name': 'gate_zero', 'gate_override': 'zero'},
    {'name': 'gate_frozen', 'gate_override': 'frozen'},
    {'name': 'gate_shuffle', 'gate_override': 'shuffle'},
    {'name': 'phase_freeze', 'phase_override': 'freeze'},
    {'name': 'phase_shuffle', 'phase_override': 'shuffle'},
    {'name': 'phase_zero', 'phase_override': 'zero'},
    {'name': 'anchor_shuffle', 'phase_override': 'anchor_shuffle'},
    {'name': 'tok_freeze', 'phase_override': 'freeze_tokens'},
    {'name': 'tok_shuffle', 'phase_override': 'shuffle_tokens'},
    {'name': 'lambda0', 'phase_override': 'lambda0'},
]


@dataclass
class InterventionsCallbackCfg(CallbackConfig):
    name: str = 'interventions'
    interventions: list[Any] = field(
        default_factory=lambda: [dict(d) for d in DEFAULT_INTERVENTIONS])
    max_batches: int = 0        # 0 = the whole test split


class sort_of_clevr_interventions_callback(BaseCallBack):

    def __init__(self, interventions: list[dict], max_batches: int) -> None:
        self.interventions = [dict(d) for d in interventions]
        self.max_batches = max_batches

    @classmethod
    def from_config(cls, cfg, cb_cfg: InterventionsCallbackCfg):
        return cls([dict(d) for d in cb_cfg.interventions], int(cb_cfg.max_batches))

    # ------------------------------------------------------------------

    @staticmethod
    def _supported(inner, spec: dict) -> bool:
        g = spec.get('gate_override')
        p = spec.get('phase_override')
        if g is not None and g not in getattr(inner, 'GATE_OVERRIDES', frozenset()):
            return False
        if p is not None and p not in getattr(inner, 'PHASE_OVERRIDES', frozenset()):
            return False
        return g is not None or p is not None

    @torch.inference_mode()
    def _evaluate(self, trainer: Trainer, **overrides) -> dict[str, float]:
        model = _unwrap(trainer.model)
        model.eval()
        keys = ['all', 'non_relational', 'binary', 'ternary'] + list(C.SUBTYPE_NAMES.values())
        correct = {k: 0 for k in keys}
        total = {k: 0 for k in keys}
        for b_idx, batch in enumerate(iter(trainer.test_dataloader)):
            if self.max_batches and b_idx >= self.max_batches:
                break
            out = model(batch,
                        **overrides)
            hit = (out['logits'].argmax(-1) == batch['answers'])
            qs = batch['questions']
            sels = {'all': torch.ones_like(hit)}
            for fam, off in C.Q_TYPES_OFFSET.items():
                sels[fam] = qs[:, C.Q_TYPE_IDX + off] == 1
            for (fam, sub), sname in C.SUBTYPE_NAMES.items():
                sels[sname] = sels[fam] & (qs[:, C.SUB_Q_TYPE_IDX + sub] == 1)
            for key, sel in sels.items():
                correct[key] += int(hit[sel].sum().item())
                total[key] += int(sel.sum().item())
        return {k: correct[k] / max(total[k], 1) for k in keys}

    def on_train_end(self, trainer: Trainer) -> None:
        try:
            self._run(trainer)
        except Exception:
            if trainer.logger is not None:
                trainer.logger.warning('interventions failed', exc_info=True)
            else:
                import traceback; traceback.print_exc()

    def _run(self, trainer: Trainer) -> None:
        model = _unwrap(trainer.model)
        inner = getattr(model, 'inner', model)
        todo = [s for s in self.interventions if self._supported(inner, s)]
        if not todo:
            trainer.log_info('interventions: model supports none, skipping')
            return
        trainer.log_info('interventions: running %s', [s['name'] for s in todo])

        # a matched un-intervened pass (random phase init makes forwards
        # stochastic, so the baseline is re-measured rather than reused)
        torch.manual_seed(0)
        base = self._evaluate(trainer)
        results: dict[str, float] = {
            'baseline_accuracy': base['all'],
            'baseline_ternary_accuracy': base['ternary'],
            'baseline_binary_accuracy': base['binary'],
        }
        table: dict[str, dict[str, float]] = {}
        for spec in todo:
            kw = {k: v for k, v in spec.items() if k != 'name'}
            torch.manual_seed(0)
            r = self._evaluate(trainer, **kw)
            n = spec['name']
            results[f'{n}_accuracy'] = r['all']
            results[f'{n}_drop'] = base['all'] - r['all']
            results[f'{n}_ternary_accuracy'] = r['ternary']
            results[f'{n}_ternary_drop'] = base['ternary'] - r['ternary']
            results[f'{n}_binary_drop'] = base['binary'] - r['binary']
            for sname in C.SUBTYPE_NAMES.values():
                results[f'{n}_{sname}_drop'] = base[sname] - r[sname]
            table[n] = {k: base[k] - r[k] for k in base}
        trainer.summary(section(results, 'interventions'), 'test')
        trainer.intervention_results = results   # readable without wandb (notebooks)
        self._fig(trainer, table)

    def _fig(self, trainer: Trainer, table: dict[str, dict[str, float]]) -> None:
        """Heatmap of accuracy drop: intervention x question subtype."""
        if not table:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from pathlib import Path
            rows = list(table)
            cols = ['all', 'non_relational', 'binary', 'ternary'] + list(C.SUBTYPE_NAMES.values())
            mat = [[table[r].get(c, float('nan')) for c in cols] for r in rows]
            fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 1.5, 0.45 * len(rows) + 1.2))
            vmax = max(0.05, max(abs(v) for row in mat for v in row if v == v))
            im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
            for i, row in enumerate(mat):
                for j, v in enumerate(row):
                    if v == v:
                        ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.03).set_label('accuracy drop (baseline - intervened)', fontsize=7)
            ax.set_title('test-time interventions', fontsize=9)
            fig.tight_layout()
            viz_dir = Path(trainer.out_dir) / 'viz'
            viz_dir.mkdir(exist_ok=True, parents=True)
            p = viz_dir / 'interventions.png'
            fig.savefig(p, dpi=130); plt.close(fig)
            trainer.log_info('interventions: wrote %s', p)
            if trainer.cfg.wandb.enabled and trainer.accelerator.is_main_process:
                import wandb
                run = trainer.accelerator.get_tracker('wandb', unwrap=True)
                if isinstance(run, wandb.Run):
                    run.log({'viz/interventions': wandb.Image(str(p))}, step=trainer.opt_step)
        except Exception:
            if trainer.logger is not None:
                trainer.logger.warning('interventions: figure failed', exc_info=True)


sort_of_clevr_interventions_callbacks: dict[str, CallbackSpec] = {
    'interventions': CallbackSpec(
        config=InterventionsCallbackCfg,
        callback_class=sort_of_clevr_interventions_callback,
    ),
}
