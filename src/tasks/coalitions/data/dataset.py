"""Dataset and on-device loader for `coalitions`, mirroring sort_of_clevr.

Batch dict keys (all leading dim = batch), which become the model's forward
kwargs and the loss/callback kwargs:
    streams, commands, targets, loss_mask, regime, oracle_adj, active_gid
`length` and the rho_* arrays are metadata; rho_* are exposed on the loader
object (not per-batch) for the calibration callback.
"""

from __future__ import annotations

from typing import Iterator, TYPE_CHECKING, cast
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..contracts import CoalitionsBatch

if TYPE_CHECKING:
    from ....core.config import Config


_BATCH_KEYS = (
    'streams', 'commands', 'targets', 'loss_mask',
    'regime', 'oracle_adj', 'active_gid',
)

_DTYPES = {
    'streams': torch.long,
    'commands': torch.long,
    'targets': torch.long,
    'loss_mask': torch.float32,
    'regime': torch.long,
    'oracle_adj': torch.float32,   # float so it doubles as the oracle gate
    'active_gid': torch.long,
}


def _move(batch: dict[str, torch.Tensor], device: str):
    nb = (device == 'cuda')
    return {k: v.to(device, non_blocking=nb) for k, v in batch.items()}


def _load(path: str):
    with np.load(path, allow_pickle=True) as data:
        tensors = {
            k: torch.from_numpy(np.asarray(data[k])).to(_DTYPES[k])
            for k in _BATCH_KEYS
        }
        rho = {
            'names': list(np.asarray(data['rho_names'])),
            'dims': {},
        }
        for key in data.files:
            if key.startswith('rho_d'):
                d = int(key[len('rho_d'):])
                rho['dims'][d] = np.asarray(data[key], dtype=np.float32)
        n = tensors['streams'].shape[0]
    return tensors, n, rho


class CoalitionsDataset(Dataset):

    def __init__(self, path: str) -> None:
        super().__init__()
        self.tensors, self.n, self.rho = _load(path)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> CoalitionsBatch:
        return cast(CoalitionsBatch, {k: v[idx] for k, v in self.tensors.items()})


class CoalitionsOnDeviceLoader:

    def __init__(
            self, path: str, batch_size: int, device: str,
            shuffle: bool = False,
            ) -> None:
        tensors, n, rho = _load(path)
        self.tensors = _move(tensors, device)
        self.rho = rho
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.N = n

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[CoalitionsBatch]:
        if self.shuffle:
            order = torch.randperm(self.N, device=self.device)
        else:
            order = torch.arange(self.N, device=self.device)
        for i in range(0, self.N, self.batch_size):
            idx = order[i: i + self.batch_size]
            yield cast(CoalitionsBatch, {k: v[idx] for k, v in self.tensors.items()})

def build_dataloaders(
        cfg: Config, device: str
        ) -> tuple[DataLoader | CoalitionsOnDeviceLoader, ...]:
    
    data_dir_path = Path(cfg.dataset.root) / cfg.dataset.dir
    
    train_path = data_dir_path / 'train.npz'
    val_path = data_dir_path / 'val.npz'
    test_path = data_dir_path / 'test.npz'


    if not all(path.exists() for path in [train_path, val_path, test_path]):
        from .generator import prepare_coalitions
        prepare_coalitions(cfg.dataset)  # type: ignore

    if cfg.train.loader_mode == 'gpu_cached':
        return (
            CoalitionsOnDeviceLoader(
                str(train_path),
                batch_size=cfg.train.train_bs,
                device=device,
                shuffle=True
                ),
            CoalitionsOnDeviceLoader(
                str(val_path),
                batch_size=cfg.train.val_bs,
                device=device,
                shuffle=False
                ),
            CoalitionsOnDeviceLoader(
                str(test_path),
                batch_size=cfg.train.val_bs,
                device=device,
                shuffle=False
            )
        )
    elif cfg.train.loader_mode == 'dataloader':
        return (
            DataLoader(
                CoalitionsDataset(str(train_path)),
                batch_size=cfg.train.train_bs,
                shuffle=True,
                num_workers=cfg.train.num_workers,
                persistent_workers=(cfg.train.num_workers > 1),
                pin_memory=(device != 'cpu')
            ),
            DataLoader(
                CoalitionsDataset(str(val_path)),
                batch_size=cfg.train.val_bs,
                shuffle=False,
                num_workers=0,
                pin_memory=(device != 'cpu')
            ),
            DataLoader(
                CoalitionsDataset(str(test_path)),
                batch_size=cfg.train.val_bs,
                shuffle=False,
                num_workers=0,
                pin_memory=(device != 'cpu')
            )
        )
    else:
        raise ValueError(f'unrecognised dataloader mode: {cfg.train.loader_mode}')