"""Binding analysis: what the modules read, how the gates form, how the
phases move -- as figures, each backed by scalar metrics.

Runs once at the end of training on the best weights, over a few test
batches with `return_trace=True`. Ground-truth object masks are read off
the pixels with the ObjectTokenizer (exact for this generator) and
projected onto the model's token grid, so "which object does module k
read" is measured, not eyeballed.

Metrics (test_binding/*):
  attn_on_objects     attention mass on object tokens vs background
  obj_coverage        fraction of the six objects that are some module's
                      dominant object (1 = every object owned; segregation)
  module_purity       mean_m max_o A[m,o] / sum_o A[m,o]  (1 = one object per module)
  queried_attn        share of object-attention on the queried colour(s)
  coalition_score     mean gate on edges incident to a queried object's
                      module minus mean gate on the other edges. > 0 means
                      the gate forms the question's coalition (the star
                      for binary questions). Also _binary / _ternary.
  gate_qdep           per-sample deviation of the gate from the batch mean
                      (0 = the same gate for every input)
  R_t{t}              order parameter after step t  (sync onset)
  gate_entropy_t{t}   incoming-channel entropy after step t
  n_clusters_eff      participation ratio of the alignment matrix's
                      eigenvalues (1 = global sync, ~M/2 = spread)
  within_obj_R        token phases: coherence within an object
  between_obj_align   token phases: alignment between different objects
                      (binding by synchrony = high within, low between)

Figures ({out_dir}/viz/*.png, wandb viz/*):
  gates_by_family     mean gate matrix per question family, per step
  attention_maps      per-module attention over the image, n examples
  phase_dynamics      R(t), gate entropy(t), per-sample phase trajectories
  module_object       mean module x object attention matrix per family
  token_phases        token phase maps (PhaseBind / OscField)

Never raises: an analysis failure at the finish line is logged, not fatal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ....core.callbacks import BaseCallBack
from ....core.config import CallbackConfig
from ....core.registry import CallbackSpec
from ....training.logging import section
from ....models.object_tokens import ObjectTokenizer
from ..data import constants as C
from .sync_metrics import _unwrap

if TYPE_CHECKING:
    from ....training import Trainer


FAMILIES = ['non_relational', 'binary', 'ternary']


@dataclass
class BindingAnalysisCfg(CallbackConfig):
    name: str = 'binding_analysis'
    max_batches: int = 4
    n_examples: int = 3
    obj_size: int = 5


def _entropy(p: Tensor) -> Tensor:
    n = p.shape[-1]
    return -(p.clamp(min=1e-12) * p.clamp(min=1e-12).log()).sum(-1) / float(np.log(n))


class sort_of_clevr_binding_analysis_callback(BaseCallBack):

    def __init__(self, max_batches: int, n_examples: int, obj_size: int) -> None:
        self.max_batches = max_batches
        self.n_examples = n_examples
        self.obj_size = obj_size

    @classmethod
    def from_config(cls, cfg, cb_cfg: BindingAnalysisCfg):
        return cls(int(cb_cfg.max_batches), int(cb_cfg.n_examples), int(cb_cfg.obj_size))

    # ------------------------------------------------------------------
    # collection

    @torch.inference_mode()
    def _collect(self, trainer: Trainer, model) -> dict:
        """Run the model with traces on a few test batches; stack everything."""
        acc: dict[str, list] = {}
        n = 0
        for b_idx, batch in enumerate(iter(trainer.test_dataloader)):
            if b_idx >= self.max_batches:
                break
            out = model({'images': batch['images'], 'questions': batch['questions']},
                        return_trace=True)
            tr = out['traces'] or {}
            acc.setdefault('images', []).append(batch['images'].float().cpu())
            acc.setdefault('questions', []).append(batch['questions'].cpu())
            acc.setdefault('answers', []).append(batch['answers'].cpu())
            acc.setdefault('pred', []).append(out['logits'].argmax(-1).cpu())
            for k in ('phase', 'gates', 'attn', 'tok_phase'):
                if k in tr and tr[k]:
                    acc.setdefault(k, []).append(torch.stack([t.float().cpu() for t in tr[k]], 0))
            if 'field' in tr and tr['field']:
                acc.setdefault('field', []).append(tr['field'][-1].float().cpu())
            n += 1
        if n == 0:
            return {}
        out = {}
        for k, v in acc.items():
            dim = 1 if k in ('phase', 'gates', 'attn', 'tok_phase') else 0
            out[k] = torch.cat(v, dim)
        return out

    @staticmethod
    def _family(q: Tensor) -> Tensor:
        """(B,) 0 non-relational, 1 binary, 2 ternary."""
        return q[:, C.Q_TYPE_IDX:C.Q_TYPE_IDX + 3].float().argmax(-1)

    @staticmethod
    def _queried(q: Tensor) -> Tensor:
        """(B, 6) 1 for each queried colour."""
        n = len(C.COLOURS)
        return ((q[:, :n] + q[:, n:2 * n]) > 0).float()

    # ------------------------------------------------------------------

    def on_train_end(self, trainer: Trainer) -> None:
        model = _unwrap(trainer.model)
        if not getattr(model, 'is_syncnet', False):
            trainer.log_info('binding_analysis: not a synchrony model, skipping')
            return
        try:
            self._run(trainer, model)
        except Exception:
            if trainer.logger is not None:
                trainer.logger.warning('binding_analysis failed', exc_info=True)
            else:
                import traceback; traceback.print_exc()

    def _run(self, trainer: Trainer, model) -> None:
        model.eval()
        data = self._collect(trainer, model)
        if not data:
            trainer.log_info('binding_analysis: no test data, skipping')
            return
        inner = getattr(model, 'inner', model)
        images, questions = data['images'], data['questions']
        B = images.shape[0]
        fam = self._family(questions)
        queried = self._queried(questions)                        # (B, 6)
        n_obj = len(C.COLOURS)

        # ground-truth object masks at pixel resolution
        tok = ObjectTokenizer(list(C.COLOURS.values()), images.shape[-1], self.obj_size)
        diff = (images.unsqueeze(1) - tok.colours[None, :, :, None, None]).abs()
        pix_mask = (diff.amax(dim=2) < tok.tol).float()          # (B, 6, H, W)

        metrics: dict[str, float] = {}
        viz_dir = Path(trainer.out_dir) / 'viz'
        viz_dir.mkdir(exist_ok=True, parents=True)
        figs: dict[str, Path] = {}

        # object masks on the token grid
        objtok = None
        if getattr(inner, 'objects', False):
            objtok = torch.eye(n_obj).unsqueeze(0).expand(B, n_obj, n_obj)   # token p = object p
        elif getattr(inner, 'spatial', None):
            S = int(inner.spatial)
            objtok = F.adaptive_avg_pool2d(pix_mask, S).flatten(2)         # (B, 6, P) coverage

        # ---------------- module models ----------------
        if 'attn' in data and 'gates' in data:
            attn = data['attn']                                    # (T, B, M, P)
            gates = data['gates']                                  # (T, B, M, M)
            T, _, M, P = attn.shape
            a_last = attn[-1]
            if objtok is not None and objtok.shape[-1] == P:
                A = torch.einsum('bmp,bop->bmo', a_last, objtok)   # (B, M, 6)
                metrics['attn_on_objects'] = A.sum(-1).mean().item()
                dom = A.argmax(-1)                                 # (B, M)
                owned = torch.zeros(B, n_obj).scatter_(1, dom, 1.0)
                metrics['obj_coverage'] = owned.mean().item()
                metrics['module_purity'] = (A.max(-1).values / (A.sum(-1) + 1e-6)).mean().item()
                qa = (A * queried.unsqueeze(1)).sum((-1, -2)) / (A.sum((-1, -2)) + 1e-6)
                metrics['queried_attn'] = qa.mean().item()
                for fi, fname in enumerate(FAMILIES):
                    sel = fam == fi
                    if sel.any():
                        metrics[f'queried_attn_{fname}'] = qa[sel].mean().item()
                # coalition score: module -> object map is the identity under
                # the object partition, else each module's dominant object
                if getattr(inner, 'cfg', None) is not None and getattr(inner.cfg, 'partition', '') == 'object':
                    mod_obj = torch.arange(M).unsqueeze(0).expand(B, M)
                else:
                    mod_obj = dom
                rel_mod = queried.gather(1, mod_obj)               # (B, M) module maps to a queried object
                rel_edge = (rel_mod.unsqueeze(1) + rel_mod.unsqueeze(2)) > 0   # (B, M, M)
                off = ~torch.eye(M, dtype=torch.bool)
                g_last = gates[-1]
                rel = rel_edge & off; irr = (~rel_edge) & off
                cs = torch.zeros(B)
                valid = torch.zeros(B, dtype=torch.bool)
                for b in range(B):
                    if rel[b].any() and irr[b].any():
                        cs[b] = g_last[b][rel[b]].mean() - g_last[b][irr[b]].mean()
                        valid[b] = True
                if valid.any():
                    metrics['coalition_score'] = cs[valid].mean().item()
                    for fi, fname in enumerate(FAMILIES[1:], start=1):
                        sel = valid & (fam == fi)
                        if sel.any():
                            metrics[f'coalition_score_{fname}'] = cs[sel].mean().item()
            # gate input-dependence and per-step structure
            off = ~torch.eye(M, dtype=torch.bool)
            g_mean = gates[-1].mean(0, keepdim=True)
            metrics['gate_qdep'] = ((gates[-1] - g_mean)[:, off].norm(dim=-1)
                                    / (g_mean[:, off].norm(dim=-1) + 1e-6)).mean().item()
            for t in range(T):
                gi = gates[t] / (gates[t].sum(-1, keepdim=True) + 1e-6)
                metrics[f'gate_entropy_t{t + 1}'] = _entropy(gi).mean().item()
            if 'phase' in data:
                ph = data['phase']                                 # (T, B, M, d)
                for t in range(T):
                    metrics[f'R_t{t + 1}'] = ph[t].mean(1).norm(dim=-1).mean().item()
                dots = torch.einsum('bid,bjd->bij', ph[-1], ph[-1])
                ev = torch.linalg.eigvalsh(0.5 * (1 + dots)).clamp(min=0)
                metrics['n_clusters_eff'] = ((ev.sum(-1) ** 2) / ((ev ** 2).sum(-1) + 1e-6)).mean().item()

            figs['gates_by_family'] = self._fig_gates(gates, fam, viz_dir)
            figs['attention_maps'] = self._fig_attention(images, attn[-1], questions, fam, pix_mask, inner, viz_dir)
            if 'phase' in data:
                figs['phase_dynamics'] = self._fig_phase(data['phase'], gates, viz_dir)
            if objtok is not None and objtok.shape[-1] == P:
                figs['module_object'] = self._fig_module_object(A, fam, viz_dir)

        # ---------------- token / field phases ----------------
        ztok = None
        if 'tok_phase' in data:
            ztok = data['tok_phase'][-1]                           # (B, P, d)
        elif 'field' in data:
            z = data['field']                                      # (B, C, S, S)
            K, d = int(inner.K), int(inner.d)
            ztok = z.view(B, K, d, -1).permute(0, 3, 1, 2).reshape(B, -1, K * d)   # concat groups
            ztok = F.normalize(ztok, dim=-1)
        if ztok is not None and objtok is not None and objtok.shape[-1] == ztok.shape[1]:
            w = objtok / (objtok.sum(-1, keepdim=True) + 1e-6)     # (B, 6, P)
            zo = torch.einsum('bop,bpd->bod', w, ztok)             # mean phase per object
            metrics['within_obj_R'] = zo.norm(dim=-1).mean().item()
            zon = F.normalize(zo, dim=-1)
            pair = torch.einsum('bod,bqd->boq', zon, zon)
            offo = ~torch.eye(n_obj, dtype=torch.bool)
            metrics['between_obj_align'] = pair[:, offo].mean().item()
            bg = (1 - objtok.sum(1).clamp(max=1.0))                # background coverage
            zb = F.normalize(torch.einsum('bp,bpd->bd', bg / (bg.sum(-1, keepdim=True) + 1e-6), ztok), dim=-1)
            metrics['obj_bg_align'] = torch.einsum('bod,bd->bo', zon, zb).mean().item()
            figs['token_phases'] = self._fig_token_phases(images, ztok, questions, objtok, inner, viz_dir)

        metrics['n_samples'] = float(B)
        trainer.summary(section(metrics, 'binding'), 'test')
        self._log_figs(trainer, figs)

    # ------------------------------------------------------------------
    # figures

    def _fig_gates(self, gates: Tensor, fam: Tensor, viz_dir: Path) -> Path:
        T, B, M, _ = gates.shape
        fig, axes = plt.subplots(3, T, figsize=(1.6 * T + 1.5, 4.8), squeeze=False)
        for fi, fname in enumerate(FAMILIES):
            sel = fam == fi
            for t in range(T):
                ax = axes[fi, t]
                if sel.any():
                    im = ax.imshow(gates[t][sel].mean(0).numpy(), vmin=0, vmax=1, cmap='viridis')
                ax.set_xticks([]); ax.set_yticks([])
                if fi == 0:
                    ax.set_title(f't={t + 1}', fontsize=8)
                if t == 0:
                    ax.set_ylabel(fname, fontsize=8)
        fig.suptitle('mean gate g[receiver, sender] by question family', fontsize=9)
        fig.tight_layout()
        p = viz_dir / 'gates_by_family.png'
        fig.savefig(p, dpi=120); plt.close(fig)
        return p

    def _fig_attention(self, images, a_last, questions, fam, pix_mask, inner, viz_dir) -> Path:
        B, M, P = a_last.shape
        n = min(self.n_examples, B)
        # pick one example per family where possible
        idx = []
        for fi in range(3):
            cand = (fam == fi).nonzero().flatten()
            if len(cand):
                idx.append(int(cand[0]))
        while len(idx) < n:
            idx.append(len(idx))
        idx = idx[:n]
        H = images.shape[-1]
        fig, axes = plt.subplots(n, M + 1, figsize=(1.7 * (M + 1), 1.8 * n), squeeze=False)
        names = list(C.COLOURS.keys())
        S = getattr(inner, 'spatial', None)
        for r, b in enumerate(idx):
            img = images[b].permute(1, 2, 0)[..., [2, 1, 0]].numpy()   # BGR -> RGB
            q = questions[b]
            cols = [names[i] for i in range(len(names)) if q[i] > 0] + \
                   [names[i] for i in range(len(names)) if q[len(names) + i] > 0]
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f'{FAMILIES[int(fam[b])]}: {",".join(cols)}', fontsize=7)
            for m in range(M):
                ax = axes[r, m + 1]
                ax.imshow(img, alpha=0.35)
                a = a_last[b, m]
                if S and a.numel() == S * S:
                    heat = F.interpolate(a.view(1, 1, S, S), size=(H, H), mode='nearest')[0, 0]
                    ax.imshow(heat.numpy(), cmap='magma', alpha=0.65, vmin=0, vmax=max(float(a.max()), 1e-6))
                elif a.numel() == len(names):
                    # object tokens: paint each object's mask with its attention
                    heat = (pix_mask[b] * a[:, None, None]).sum(0)
                    ax.imshow(heat.numpy(), cmap='magma', alpha=0.65, vmin=0, vmax=max(float(a.max()), 1e-6))
                ax.set_title(f'module {m}', fontsize=7)
            for ax in axes[r]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle('final-step read attention per module', fontsize=9)
        fig.tight_layout()
        p = viz_dir / 'attention_maps.png'
        fig.savefig(p, dpi=120); plt.close(fig)
        return p

    def _fig_phase(self, phase: Tensor, gates: Tensor, viz_dir: Path) -> Path:
        T, B, M, d = phase.shape
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        R = phase.mean(2).norm(dim=-1)                               # (T, B)
        ts = np.arange(1, T + 1)
        axes[0].plot(ts, R.mean(1).numpy(), marker='o', ms=3)
        axes[0].fill_between(ts, (R.mean(1) - R.std(1)).numpy(), (R.mean(1) + R.std(1)).numpy(), alpha=0.25)
        axes[0].set_ylim(0, 1); axes[0].set_xlabel('step'); axes[0].set_title('order parameter R(t)', fontsize=9)
        gi = gates / (gates.sum(-1, keepdim=True) + 1e-6)
        ent = _entropy(gi).mean((1, 2))
        axes[1].plot(ts, ent.numpy(), marker='o', ms=3, color='tab:orange')
        axes[1].set_ylim(0, 1); axes[1].set_xlabel('step'); axes[1].set_title('incoming-gate entropy(t)', fontsize=9)
        n = min(self.n_examples, B)
        ref = phase[:, :, 0]                                          # (T, B, d)
        for b in range(n):
            if d == 2:
                ang = torch.atan2(phase[:, b, :, 1], phase[:, b, :, 0]).numpy()   # (T, M)
                for m in range(M):
                    axes[2].plot(ts, np.unwrap(ang[:, m]), lw=1, alpha=0.8, color=plt.cm.tab10(m % 10),
                                 ls=['-', '--', ':'][b % 3])
                axes[2].set_ylabel('phase (rad, unwrapped)', fontsize=8)
            else:
                al = (phase[:, b] * ref[:, b].unsqueeze(1)).sum(-1).numpy()   # (T, M) alignment to module 0
                for m in range(M):
                    axes[2].plot(ts, al[:, m], lw=1, alpha=0.8, color=plt.cm.tab10(m % 10), ls=['-', '--', ':'][b % 3])
                axes[2].set_ylabel('alignment with module 0', fontsize=8)
        axes[2].set_xlabel('step'); axes[2].set_title(f'module phases, {n} samples (colour = module)', fontsize=9)
        fig.tight_layout()
        p = viz_dir / 'phase_dynamics.png'
        fig.savefig(p, dpi=120); plt.close(fig)
        return p

    def _fig_module_object(self, A: Tensor, fam: Tensor, viz_dir: Path) -> Path:
        names = list(C.COLOURS.keys())
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
        for fi, fname in enumerate(FAMILIES):
            sel = fam == fi
            ax = axes[fi]
            if sel.any():
                ax.imshow(A[sel].mean(0).numpy(), cmap='viridis', vmin=0)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=6, rotation=45)
            ax.set_ylabel('module', fontsize=8); ax.set_title(fname, fontsize=9)
        fig.suptitle('attention mass of module on object (final step)', fontsize=9)
        fig.tight_layout()
        p = viz_dir / 'module_object.png'
        fig.savefig(p, dpi=120); plt.close(fig)
        return p

    def _fig_token_phases(self, images, ztok, questions, objtok, inner, viz_dir) -> Path:
        B, P, d = ztok.shape
        n = min(self.n_examples, B)
        S = getattr(inner, 'spatial', None)
        H = images.shape[-1]
        fig, axes = plt.subplots(n, 2, figsize=(4.2, 2.0 * n), squeeze=False)
        for r in range(n):
            img = images[r].permute(1, 2, 0)[..., [2, 1, 0]].numpy()
            axes[r, 0].imshow(img); axes[r, 0].set_title('image', fontsize=7)
            # reference: mean phase of the first queried object
            q = self._queried(questions[r:r + 1])[0]
            o = int(q.argmax())
            w = objtok[r, o] / (objtok[r, o].sum() + 1e-6)
            ref = F.normalize(torch.einsum('p,pd->d', w, ztok[r]), dim=-1)
            al = ztok[r] @ ref                                        # (P,) alignment with queried object
            if S and P == S * S:
                heat = F.interpolate(al.view(1, 1, S, S), size=(H, H), mode='nearest')[0, 0]
                axes[r, 1].imshow(img, alpha=0.3)
                axes[r, 1].imshow(heat.numpy(), cmap='coolwarm', vmin=-1, vmax=1, alpha=0.7)
            else:
                axes[r, 1].bar(range(P), al.numpy(), color=[np.array(c[::-1]) / 255 for c in C.COLOURS.values()][:P])
                axes[r, 1].set_ylim(-1, 1)
            axes[r, 1].set_title(f'alignment with {list(C.COLOURS)[o]} object', fontsize=7)
            for ax in axes[r]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle('token phases (final step)', fontsize=9)
        fig.tight_layout()
        p = viz_dir / 'token_phases.png'
        fig.savefig(p, dpi=120); plt.close(fig)
        return p

    def _log_figs(self, trainer: Trainer, figs: dict[str, Path]) -> None:
        for k, p in figs.items():
            trainer.log_info('binding_analysis: wrote %s', p)
        if not (trainer.cfg.wandb.enabled and trainer.accelerator.is_main_process):
            return
        try:
            import wandb
            run = trainer.accelerator.get_tracker('wandb', unwrap=True)
            if isinstance(run, wandb.Run):
                run.log({f'viz/{k}': wandb.Image(str(p)) for k, p in figs.items()},
                        step=trainer.opt_step)
        except Exception:
            if trainer.logger is not None:
                trainer.logger.warning('binding_analysis: wandb logging failed', exc_info=True)


sort_of_clevr_binding_analysis_callbacks: dict[str, CallbackSpec] = {
    'binding_analysis': CallbackSpec(
        config=BindingAnalysisCfg,
        callback_class=sort_of_clevr_binding_analysis_callback,
    ),
}
