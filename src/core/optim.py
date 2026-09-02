from torch.optim import (
    Optimizer, AdamW
    )
from torch.optim.lr_scheduler import (
    LRScheduler, ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
    )
import torch.nn as nn

from .config import OptimConfig

def build_optim(model: nn.Module, optim_cfg: OptimConfig) -> Optimizer:
    if optim_cfg.optimiser == 'adamw':
        return AdamW(
            model.parameters(), 
            optim_cfg.lr, 
            weight_decay=optim_cfg.weight_decay,
            fused=True
            )
    else:
        raise ValueError(f'unrecognised optimiser: {optim_cfg.optimiser}')
    
def build_lr_scheduler(
        optim: Optimizer, n_steps: int, optim_cfg: OptimConfig
        ) -> LRScheduler:

    match optim_cfg.lr_scheduler:
        case 'constant':
            return ConstantLR(optim, 1, 1)
        
        case 'cosine_annealing':
            return CosineAnnealingLR(
                optim,
                T_max=n_steps,
                **optim_cfg.lr_scheduler_params
            )

        case 'warmup_stable_decay':
            params = dict(optim_cfg.lr_scheduler_params)
            warmup = int(params.pop('warmup_steps', max(1, n_steps // 50)))
            warmup = max(1, min(warmup, n_steps - 2))
            decay = int(params.pop('decay_steps', max(1, int(n_steps * float(params.pop('decay_frac', 0.2))))))
            decay = max(1, min(decay, n_steps - warmup - 1))
            stable_end = n_steps - decay
            return SequentialLR(
                optim,
                schedulers=[
                    LinearLR(optim, start_factor=1e-8, end_factor=1.0, total_iters=warmup),
                    ConstantLR(optim, factor=1.0, total_iters=stable_end - warmup),
                    CosineAnnealingLR(optim, T_max=decay, **params),
                ],
                milestones=[warmup, stable_end],
            )

        case 'warmup_cosine':

            params = dict(optim_cfg.lr_scheduler_params)
            # if not specified use 2pct of steps
            warmup = int(params.pop('warmup_steps', max(1, n_steps // 50)))
            warmup = max(1, min(warmup, n_steps - 1))
            return SequentialLR(
                optim,
                schedulers=[
                    LinearLR(optim, start_factor=1e-8, end_factor=1.0,
                             total_iters=warmup),
                    CosineAnnealingLR(optim, T_max=n_steps - warmup, **params),
                ],
                milestones=[warmup],
            )

        
        case _:
            raise ValueError(f'unrecognised lr scheduler: {optim_cfg.lr_scheduler}')