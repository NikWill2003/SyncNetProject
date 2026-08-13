import random
from typing import Iterator, Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_param_count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def move_tensor_dict(batch: dict[str, torch.Tensor], device: str):
    non_blocking = (device == 'cuda')
    
    return {
        key: item.to(device, non_blocking=non_blocking) 
        if isinstance(item, torch.Tensor) 
        else item 
        for key, item in batch.items()
        }


def get_batch_dict_size(batch: dict[str, torch.Tensor]) -> int:
    return next(iter(batch.values())).size(0)


def batch_iter(
        dataloader: DataLoader | Iterator[dict[str, Any]]
        ) -> Iterator[dict[str, torch.Tensor]]:
    
    while True:
        batch_iterator = iter(dataloader)
        for batch in batch_iterator:
            yield batch
