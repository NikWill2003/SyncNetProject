from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ....core.callbacks import BaseCallBack
from ....core.config import CallbackConfig
from ....core.registry import CallbackSpec
from .sync_metrics import _unwrap
from ..data import constants as C

if TYPE_CHECKING:
    from ....training import Trainer


@dataclass
class TVarianceCallbackCfg(CallbackConfig):
    name: str = 't_variance'
    t_values: list[int] = field(
        default_factory=lambda: [0, 1, 2, 4, 6, 8, 12, 16]
    )
    n_repeats: int = 5      # stochastic-init forwards per T
    max_batches: int = 4    # cap of test batches per forward (cost control)


class sort_of_clevr_t_variance_callback(BaseCallBack):
    """Test-time T generalisation + variance ablation, run once at the end
    of training on the best (early-stopping) weights.

    For each T in `t_values` the *trained* model is run `n_repeats` times
    on (a capped slice of) the test set with `t_override=T`. Models with
    stochastic per-forward initialisation (random phases / random state)
    give a distribution of accuracies per T; the figure shows the mean
    with a +-1 std band, plus the train-time T as a vertical marker.

    Reads:
      * whether accuracy at the trained T transfers to other T
        (the T=0-matches-T=4 negative result, now as a curve), and
      * how sensitive the trained solution is to its stochastic init
        (the band width), per T.

    Output: {out_dir}/viz/t_variance.png, logged to wandb as
    viz/t_variance plus per-T scalar summaries (t_variance/acc_mean_T{T},
    t_variance/acc_std_T{T}).

    Works with any model whose forward accepts `t_override` (V1, V3,
    recurrent_syncnet). Skips itself with a log line otherwise.
    """

    def __init__(
            self,
            t_values: list[int],
            n_repeats: int,
            max_batches: int,
            ) -> None:
        self.t_values = list(t_values)
        self.n_repeats = n_repeats
        self.max_batches = max_batches

    @classmethod
    def from_config(
            cls, cfg, cb_cfg: TVarianceCallbackCfg
            ) -> 'sort_of_clevr_t_variance_callback':
        return cls(
            t_values=list(cb_cfg.t_values),
            n_repeats=cb_cfg.n_repeats,
            max_batches=cb_cfg.max_batches,
        )

    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _accuracy_once(self, trainer: Trainer, t: int) -> dict[str, float]:
        """overall / binary / ternary accuracy at test-time T = t."""
        model = _unwrap(trainer.model)
        model.eval()

        keys = ('all', 'binary', 'ternary')
        correct = {k: 0 for k in keys}
        total = {k: 0 for k in keys}
        bin_idx = C.Q_TYPE_IDX + C.Q_TYPES_OFFSET['binary']
        tern_idx = C.Q_TYPE_IDX + C.Q_TYPES_OFFSET['ternary']
        for b_idx, batch in enumerate(iter(trainer.test_dataloader)):
            if b_idx >= self.max_batches:
                break
            out = model(
                {'images': batch['images'], 'questions': batch['questions']},
                t_override=t,
            )
            hit = (out['logits'].argmax(-1) == batch['answers'])
            qs = batch['questions']
            for k, sel in (('all', torch.ones_like(hit)),
                           ('binary', qs[:, bin_idx] == 1),
                           ('ternary', qs[:, tern_idx] == 1)):
                correct[k] += int(hit[sel].sum().item())
                total[k] += int(sel.sum().item())

        return {k: correct[k] / max(total[k], 1) for k in keys}

    def on_train_end(self, trainer: Trainer) -> None:
        model = _unwrap(trainer.model)

        # duck-check the interface instead of poking signatures
        if not hasattr(model, 'T'):
            trainer.log_info(
                't_variance: model has no dynamics horizon T, skipping'
            )
            return

        trainer.log_info(
            't_variance: sweeping T=%s with %d repeats',
            self.t_values, self.n_repeats,
        )

        means, stds = [], []
        fam_means: dict[str, list[float]] = {'binary': [], 'ternary': []}
        for t in self.t_values:
            reps = [self._accuracy_once(trainer, t) for _ in range(self.n_repeats)]
            accs = torch.tensor([r['all'] for r in reps])
            means.append(accs.mean().item())
            stds.append(accs.std(unbiased=False).item())
            for k in fam_means:
                fam_means[k].append(float(torch.tensor([r[k] for r in reps]).mean()))

        means_t = torch.tensor(means)
        stds_t = torch.tensor(stds)

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(self.t_values, means, marker='o', ms=4, lw=1.5,
                color='tab:blue', label='test accuracy (mean)')
        ax.fill_between(
            self.t_values,
            (means_t - stds_t).tolist(),
            (means_t + stds_t).tolist(),
            alpha=0.25, color='tab:blue',
            label=f'±1 std over {self.n_repeats} inits',
        )
        ax.plot(self.t_values, fam_means['binary'], marker='s', ms=3, lw=1,
                color='tab:green', label='binary')
        ax.plot(self.t_values, fam_means['ternary'], marker='^', ms=3, lw=1,
                color='tab:purple', label='ternary')
        train_T = int(getattr(model, 'T'))
        ax.axvline(train_T, color='tab:red', ls='--', lw=1,
                   label=f'train T={train_T}')
        ax.set_xlabel('test-time dynamics steps T')
        ax.set_ylabel('test accuracy')
        ax.set_title(
            f'{model.__class__.__name__} | test-time T ablation '
            f'| step {trainer.opt_step}'
        )
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()

        viz_dir = Path(trainer.out_dir) / 'viz'
        viz_dir.mkdir(exist_ok=True, parents=True)
        path = viz_dir / 't_variance.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)

        trainer.log_info('t_variance: wrote %s', path)

        if trainer.cfg.wandb.enabled and trainer.accelerator.is_main_process:
            try:
                import wandb
                run = trainer.accelerator.get_tracker(
                    'wandb', unwrap=True
                )
                if isinstance(run, wandb.Run):
                    run.log(
                        {'viz/t_variance': wandb.Image(str(path))},
                        step=trainer.opt_step,
                    )
                    for t, m, s in zip(self.t_values, means, stds):
                        run.summary[f't_variance/acc_mean_T{t}'] = m
                        run.summary[f't_variance/acc_std_T{t}'] = s
                    for k, vals in fam_means.items():
                        for t, m in zip(self.t_values, vals):
                            run.summary[f't_variance/{k}_mean_T{t}'] = m
            except Exception:
                # never let a plot kill a long run at the finish line
                if trainer.logger is not None:
                    trainer.logger.warning(
                        't_variance: wandb logging failed', exc_info=True
                    )


sort_of_clevr_t_variance_callbacks: dict[str, CallbackSpec] = {
    't_variance': CallbackSpec(
        config=TVarianceCallbackCfg,
        callback_class=sort_of_clevr_t_variance_callback,
    ),
}
