from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataclasses import dataclass

from ....core.callbacks import BaseCallBack
from ....core.config import CallbackConfig
from ....core.registry import CallbackSpec
from ..contracts import SortOfClevrBatch, SortOfClevrOutput
from .sync_metrics import _unwrap, order_parameter, routing_entropy

if TYPE_CHECKING:
    from ....training import Trainer


def _phase_hue_maps(
        steps: list[Tensor], n_samples: int
        ) -> list[list[Tensor]]:
    """Project per-position feature vectors to a hue angle, consistently
    across dynamics steps.

    steps: list over T of (B, ch, H, W). The 2D projection basis is fitted
    on the final step and reused for all steps, so colour identity is
    comparable over time within one figure. Returns [T][n_samples] of
    (H, W) hue arrays in [0, 1].
    """
    B, ch, H, W = steps[0].shape
    k = min(n_samples, B)

    ref = (
        steps[-1][:k]
        .permute(0, 2, 3, 1)
        .reshape(-1, ch)
        .float()
    )
    ref = ref - ref.mean(0, keepdim=True)
    # basis: top-2 right singular vectors of the final-step features
    _, _, v = torch.pca_lowrank(ref, q=2)

    out: list[list[Tensor]] = []
    for x in steps:
        x_f = x[:k].permute(0, 2, 3, 1).float()          # (k, H, W, ch)
        proj = x_f @ v                                    # (k, H, W, 2)
        hue = (torch.atan2(proj[..., 1], proj[..., 0]) + torch.pi) / (
            2 * torch.pi
        )
        out.append([hue[i].cpu() for i in range(k)])
    return out


class sort_of_clevr_sync_viz_callback(BaseCallBack):
    """Renders one figure per eval interval into {out_dir}/viz/ (and to
    wandb when enabled). No-op for models without `is_syncnet`.

    Figure layout, per sample row: input image | phase/state hue map at
    t = 1, T//2, T | final routing map per module. Below: order-parameter
    and routing-entropy trajectories over the T steps, batch-averaged.

    Runs on the first batch of each eval only (deduped on trainer.opt_step)
    to keep eval cost flat.
    """

    def __init__(self, n_samples: int = 3) -> None:
        self.n_samples = n_samples
        self._last_rendered_step: int = -1

    @classmethod
    def from_config(
            cls, cfg, cb_cfg: 'SyncVizCallbackCfg'
            ) -> 'sort_of_clevr_sync_viz_callback':
        return cls(n_samples=cb_cfg.n_samples)

    def _render(
            self, trainer: Trainer,
            images: Tensor, questions: Tensor,
            tag: str,
            ) -> None:

        model = _unwrap(trainer.model)
        if not getattr(model, 'is_syncnet', False):
            return

        step = trainer.opt_step
        if step == self._last_rendered_step:
            return
        self._last_rendered_step = step

        with torch.inference_mode():
            out = model(images, questions, return_trace=True)

        traces = out['traces']
        state_key = 'rotors' if 'rotors' in traces else 'state'
        states = traces[state_key]
        attns = traces['attn']
        T = len(states)
        k = min(self.n_samples, images.shape[0])
        M = attns[0].shape[1]
        H = W = model.spatial

        t_idx = sorted(set([0, T // 2, T - 1]))
        hues = _phase_hue_maps([states[t] for t in t_idx], k)

        rotor_dim = getattr(model, 'rotor_dim', 1)
        r_traj = [order_parameter(s, rotor_dim) for s in states]
        h_traj = [routing_entropy(a) for a in attns]

        n_cols = 1 + len(t_idx) + M
        fig, axes = plt.subplots(
            k + 1, n_cols,
            figsize=(2.0 * n_cols, 2.0 * (k + 1)),
            squeeze=False,
        )

        for i in range(k):
            img = images[i].detach().float().cpu()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[i][0].imshow(img.permute(1, 2, 0).numpy())
            axes[i][0].set_ylabel(f'sample {i}', fontsize=8)
            if i == 0:
                axes[i][0].set_title('input', fontsize=8)

            for j, t in enumerate(t_idx):
                ax = axes[i][1 + j]
                ax.imshow(
                    hues[j][i].numpy(),
                    cmap='hsv', vmin=0.0, vmax=1.0,
                )
                if i == 0:
                    ax.set_title(f'{state_key} t={t + 1}', fontsize=8)

            attn_final = attns[-1][i].reshape(M, H, W).cpu()
            for m in range(M):
                ax = axes[i][1 + len(t_idx) + m]
                ax.imshow(attn_final[m].numpy(), cmap='viridis')
                if i == 0:
                    ax.set_title(f'route m{m}', fontsize=8)

        for ax_row in axes[:k]:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])

        gs = axes[k][0].get_gridspec()
        for ax in axes[k]:
            ax.remove()
        ax_r = fig.add_subplot(gs[k, : n_cols // 2])
        ax_h = fig.add_subplot(gs[k, n_cols // 2 :])
        ax_r.plot(range(1, T + 1), r_traj, marker='o', ms=3)
        ax_r.set_title('order parameter R(t)', fontsize=8)
        ax_r.set_ylim(0, 1)
        ax_h.plot(range(1, T + 1), h_traj, marker='o', ms=3, color='tab:red')
        ax_h.set_title('routing entropy (norm.)', fontsize=8)
        ax_h.set_ylim(0, 1)
        for ax in (ax_r, ax_h):
            ax.tick_params(labelsize=7)
            ax.set_xlabel('t', fontsize=7)

        fig.suptitle(
            f'{getattr(model, "__class__").__name__} | {tag} '
            f'| opt step {step}',
            fontsize=9,
        )
        fig.tight_layout()

        viz_dir = Path(trainer.out_dir) / 'viz'
        viz_dir.mkdir(exist_ok=True, parents=True)
        path = viz_dir / f'{tag}_step{step:07d}.png'
        fig.savefig(path, dpi=110)
        plt.close(fig)

        if trainer.cfg.wandb.enabled:
            try:
                import wandb
                tracker = trainer.accelerator.get_tracker('wandb')
                tracker.log(                                # type: ignore
                    {f'viz/{tag}': wandb.Image(str(path))},
                    step=step,
                )
            except Exception:
                pass  # never let visualisation kill a 3-day run

    def on_eval_step_end(
            self, trainer: Trainer,
            out: SortOfClevrOutput, batch: SortOfClevrBatch,
            ) -> Optional[dict[str, float]]:
        self._render(trainer, batch['images'], batch['questions'], tag='eval')
        return None

    def on_test_step_end(
            self, trainer: Trainer,
            out: SortOfClevrOutput, batch: SortOfClevrBatch,
            ) -> Optional[dict[str, float]]:
        # force one render at test time regardless of the eval dedupe
        self._last_rendered_step = -1
        self._render(trainer, batch['images'], batch['questions'], tag='test')
        return None


@dataclass
class SyncVizCallbackCfg(CallbackConfig):
    name: str = 'sync_viz'
    n_samples: int = 3   # sample rows per rendered figure


sort_of_clevr_sync_viz_callbacks: dict[str, CallbackSpec] = {
    'sync_viz': CallbackSpec(
        config=SyncVizCallbackCfg,
        callback_class=sort_of_clevr_sync_viz_callback,
    ),
}
