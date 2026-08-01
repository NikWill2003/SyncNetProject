"""SQOOP loaders.

Follows the sort_of_clevr loader pattern (Dataset for the 'dataloader'
mode, an on-device cached iterator for 'gpu_cached'), with two deliberate
differences driven by SQOOP's scale:

1. The GPU cache holds images as uint8 and casts to float [0, 1] per
   batch -- ~4x smaller cache than float32; models still see the same
   float-[0,1] convention as sort_of_clevr. The cast is safe because
   advanced indexing + .float() always produces a fresh tensor.
2. No image->question fan-out: SQOOP is flat, one image per question.

Splits on disk: train / val_seen / val_unseen / test_unseen. The trainer
consumes exactly three loaders; which splits fill the eval and test slots
is set by cfg.dataset.eval_split / test_split (defaults: val_seen,
test_unseen). Early stopping therefore runs on seen pairs, as it must.
"""

from __future__ import annotations

from typing import Iterator, TYPE_CHECKING
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..contracts import SqoopBatch

if TYPE_CHECKING:
    from ....core.config import Config


def _load_sqoop(path: str, max_examples: int = 0, subsample_seed: int = 0):
    """One split as CPU tensors: uint8 images (N, 3, H, W), long
    questions (N, 3), long answers (N,)."""
    with np.load(path) as data:
        images = torch.from_numpy(data['images'])          # (N, H, W, 3) u8
        images = images.permute(0, 3, 1, 2).contiguous()   # (N, 3, H, W)
        questions = torch.from_numpy(data['questions']).long()
        answers = torch.from_numpy(data['answers']).long()

    if 0 < max_examples < images.size(0):
        gen = torch.Generator().manual_seed(subsample_seed)
        keep = torch.randperm(images.size(0), generator=gen)[:max_examples]
        images, questions, answers = (
            images[keep], questions[keep], answers[keep]
        )

    return {'images': images, 'questions': questions, 'answers': answers}


def _to_model_images(images_u8: torch.Tensor) -> torch.Tensor:
    return images_u8.float().div_(255.0)


class SqoopDataset(Dataset):
    """CPU dataset for the 'dataloader' mode. Casts per item."""

    def __init__(self, path: str, max_examples: int = 0, seed: int = 0):
        super().__init__()
        d = _load_sqoop(path, max_examples, seed)
        self.images = d['images']
        self.questions = d['questions']
        self.answers = d['answers']

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> SqoopBatch:
        return {
            'images': _to_model_images(self.images[idx]),
            'questions': self.questions[idx],
            'answers': self.answers[idx],
        }


class SqoopOnDeviceLoader:
    """Whole split cached on-device as uint8; float cast per batch."""

    def __init__(
            self,
            path: str,
            batch_size: int,
            device: str,
            shuffle: bool = False,
            max_examples: int = 0,
            seed: int = 0,
            ) -> None:
        d = _load_sqoop(path, max_examples, seed)
        non_blocking = device.startswith('cuda')
        self.images = d['images'].to(device, non_blocking=non_blocking)
        self.questions = d['questions'].to(device, non_blocking=non_blocking)
        self.answers = d['answers'].to(device, non_blocking=non_blocking)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.N = self.images.size(0)

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[SqoopBatch]:
        if self.shuffle:
            order = torch.randperm(self.N, device=self.device)
        else:
            order = torch.arange(self.N, device=self.device)

        for i in range(0, self.N, self.batch_size):
            idx = order[i: i + self.batch_size]
            yield {
                'images': _to_model_images(self.images[idx]),
                'questions': self.questions[idx],
                'answers': self.answers[idx],
            }


def build_dataloaders(cfg: 'Config', device: str):
    d = cfg.dataset
    root = Path(d.root) / d.dir

    paths = {
        'train': root / 'train.npz',
        'eval': root / f'{d.eval_split}.npz',
        'test': root / f'{d.test_split}.npz',
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f'sqoop {k} split missing: {p} '
                f'(run: python prepare_dataset.py task=sqoop '
                f'dataset.rhs_variety={d.rhs_variety})'
            )

    mode = cfg.train.loader_mode
    if mode == 'gpu_cached':
        train = SqoopOnDeviceLoader(
            str(paths['train']), cfg.train.train_bs, device, shuffle=True,
            max_examples=int(d.max_train_examples), seed=int(d.seed),
        )
        ev = SqoopOnDeviceLoader(
            str(paths['eval']), cfg.train.val_bs, device, shuffle=False,
        )
        test = SqoopOnDeviceLoader(
            str(paths['test']), cfg.train.val_bs, device, shuffle=False,
        )
        return train, ev, test

    if mode == 'dataloader':
        train_ds = SqoopDataset(
            str(paths['train']),
            max_examples=int(d.max_train_examples), seed=int(d.seed),
        )
        train = DataLoader(
            train_ds, batch_size=cfg.train.train_bs, shuffle=True,
            num_workers=cfg.train.num_workers, pin_memory=True,
        )
        ev = DataLoader(
            SqoopDataset(str(paths['eval'])), batch_size=cfg.train.val_bs,
            shuffle=False, num_workers=cfg.train.num_workers,
            pin_memory=True,
        )
        test = DataLoader(
            SqoopDataset(str(paths['test'])), batch_size=cfg.train.val_bs,
            shuffle=False, num_workers=cfg.train.num_workers,
            pin_memory=True,
        )
        return train, ev, test

    raise ValueError(f'unknown loader_mode: {mode!r}')
