from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from ..base import build_loaders, resolve_paths, to_float_images
from ...core.contracts import VQABatch

if TYPE_CHECKING:
    from ...core.config import Config

# val_unseen is written to disk for diagnostics but is not loaded here:
# early stopping on it would leak the thing the systematic split measures.
SPLITS = {'train': 'train', 'eval': 'val_seen', 'test': 'test_unseen'}


def _load_split(path: str) -> dict:
    """One split as CPU tensors: uint8 images (N, 3, H, W), long questions
    (N, 3), long answers (N,). One question per image."""
    with np.load(path) as data:
        images = torch.from_numpy(data['images'])          # (N, H, W, 3) u8
        return {
            'images': images.permute(0, 3, 1, 2).contiguous(),
            'questions': torch.from_numpy(data['questions']).long(),
            'answers': torch.from_numpy(data['answers']).long(),
            **({'scenes': torch.from_numpy(data['scenes'])} if 'scenes' in data else {}),
        }


class SqoopDataset(Dataset):
    """CPU dataset for loader_mode='dataloader'. Casts per item."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.split = _load_split(path)

    def __len__(self) -> int:
        return self.split['answers'].size(0)

    def __getitem__(self, idx: int) -> VQABatch:
        return {
            'images': to_float_images(self.split['images'][idx]),
            'questions': self.split['questions'][idx],
            'answers': self.split['answers'][idx],
            **({'scenes': self.split['scenes'][idx]} if 'scenes' in self.split else {}),
        }


class SqoopOnDeviceLoader:
    """Whole split cached on-device as uint8; float cast per batch."""

    def __init__(
            self, path: str, batch_size: int, device: str,
            shuffle: bool = False,
            ) -> None:
        non_blocking = device.startswith('cuda')
        self.split = {
            k: v.to(device, non_blocking=non_blocking)
            for k, v in _load_split(path).items()
        }
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.N = self.split['answers'].size(0)

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[VQABatch]:
        order = (
            torch.randperm(self.N, device=self.device) if self.shuffle
            else torch.arange(self.N, device=self.device)
        )
        for i in range(0, self.N, self.batch_size):
            idx = order[i: i + self.batch_size]
            yield {
                'images': to_float_images(self.split['images'][idx]),
                'questions': self.split['questions'][idx],
                'answers': self.split['answers'][idx],
                **({'scenes': self.split['scenes'][idx]} if 'scenes' in self.split else {}),
            }


def build_dataloaders(cfg: 'Config', device: str):
    from .generator import prepare_sqoop

    paths = resolve_paths(cfg, SPLITS, prepare_sqoop)
    return build_loaders(
        cfg, device, paths, SqoopDataset, SqoopOnDeviceLoader)
