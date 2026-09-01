from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from ..base import build_loaders, resolve_paths, to_float_images
from ...core.contracts import VQABatch

if TYPE_CHECKING:
    from ...core.config import Config

SPLITS = {'train': 'train', 'eval': 'val', 'test': 'test'}


def _load_split(path: str) -> dict:
    """One split as CPU tensors.

    Several questions are asked about each scene, so images are stored
    once per scene and `image_idx` says which scene a question refers to.
    """
    with np.load(path) as data:
        images = torch.from_numpy(data['images'])          # (n_img, H, W, 3)
        return {
            'images': images.permute(0, 3, 1, 2).contiguous(),
            'questions': torch.from_numpy(data['questions']).long(),
            'answers': torch.from_numpy(data['answers']).long(),
            'image_idx': torch.from_numpy(data['image_idx']).long(),
            # unified scene tokens (n_scenes, n_colours, 3) = x, y, shape,
            # assembled from the generator's object_positions/object_shapes
            **({'scenes': torch.cat([
                    torch.from_numpy(np.asarray(data['object_positions'])).long(),
                    torch.from_numpy(np.asarray(data['object_shapes'])).long().unsqueeze(-1),
                ], -1)} if 'object_positions' in data else {}),
        }


class SortOfClevrDataset(Dataset):
    """CPU dataset for loader_mode='dataloader'. Casts per item."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.split = _load_split(path)

    def __len__(self) -> int:
        return self.split['answers'].size(0)

    def __getitem__(self, idx: int) -> VQABatch:
        image = self.split['images'][self.split['image_idx'][idx]]
        return {
            'images': to_float_images(image),
            'questions': self.split['questions'][idx],
            'answers': self.split['answers'][idx],
            **({'scenes': self.split['scenes'][self.split['image_idx'][idx]]} if 'scenes' in self.split else {}),
        }


class SortOfClevrOnDeviceLoader:
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
            images = self.split['images'][self.split['image_idx'][idx]]
            yield {
                'images': to_float_images(images),
                'questions': self.split['questions'][idx],
                'answers': self.split['answers'][idx],
                **({'scenes': self.split['scenes'][self.split['image_idx'][idx]]}
                   if 'scenes' in self.split else {}),
            }


def build_dataloaders(cfg: 'Config', device: str):
    from .generator import prepare_sort_of_clevr

    paths = resolve_paths(cfg, SPLITS, prepare_sort_of_clevr)
    return build_loaders(
        cfg, device, paths, SortOfClevrDataset, SortOfClevrOnDeviceLoader)
