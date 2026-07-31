from torch.optim import (
    Optimizer, AdamW
    )
from torch.optim.lr_scheduler import (
    LRScheduler, ConstantLR, CosineAnnealingLR
    )
import torch.nn as nn

from .config import OptimConfig

def build_optim(model: nn.Module, optim_cfg: OptimConfig) -> Optimizer:
    if optim_cfg.optimiser == 'adamw':
        return AdamW(
            model.parameters(), 
            optim_cfg.lr, 
            weight_decay=optim_cfg.weight_decay
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
        
        case _:
            raise ValueError(f'unrecognised lr scheduler: {optim_cfg.lr_scheduler}')