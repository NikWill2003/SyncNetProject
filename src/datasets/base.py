"""Pieces both dataset loaders use.

Each dataset keeps its own `loader.py`, because how its npz maps to a
batch genuinely differs -- Sort-of-CLEVR stores images once per scene and
asks several questions about each, SQOOP stores one question per image.
What does not differ is the image cast and the loader-mode dispatch, so
those live here rather than being written twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ..core.config import Config


def to_float_images(images_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [0, 1].

    Splits are cached as uint8 and cast per batch: at SQOOP's default size
    the float copy would be four times the VRAM for no gain.
    """
    return images_u8.float().div(255.0)


def resolve_paths(cfg: 'Config', stems: dict[str, str], prepare) -> dict[str, Path]:
    """Locate the npz for each trainer slot, generating them if missing."""
    root = Path(cfg.dataset.root) / cfg.dataset.dir
    paths = {slot: root / f'{stem}.npz' for slot, stem in stems.items()}

    if not all(path.exists() for path in paths.values()):
        prepare(cfg.dataset)

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f'prepare() did not produce {missing}')

    return paths


def build_loaders(
        cfg: 'Config',
        device: str,
        paths: dict[str, Path],
        dataset_cls: Callable[[str], Any],
        on_device_cls: Callable[..., Any],
        ) -> tuple[Any, ...]:
    """(train, eval, test) in the trainer's slot order."""
    mode = cfg.train.loader_mode
    sizes = {'train': cfg.train.train_bs, 'eval': cfg.train.val_bs,
             'test': cfg.train.val_bs}
    order = ('train', 'eval', 'test')

    if mode == 'gpu_cached':
        return tuple(
            on_device_cls(
                str(paths[slot]), sizes[slot], device,
                shuffle=(slot == 'train'),
            )
            for slot in order
        )

    if mode == 'dataloader':
        return tuple(
            DataLoader(
                dataset_cls(str(paths[slot])),
                batch_size=sizes[slot], shuffle=(slot == 'train'),
                num_workers=cfg.train.num_workers if slot == 'train' else 0,
                pin_memory=(device != 'cpu'),
            )
            for slot in order
        )

    raise ValueError(f'unknown loader_mode: {mode!r}')

