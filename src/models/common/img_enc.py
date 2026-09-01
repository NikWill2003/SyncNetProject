from typing import Any

import torch.nn as nn
from torch import Tensor

from ...datasets.soc import spec as SOCConst
from ...datasets.sqoop import spec as SQOOPConst


class PatchifyEncoder(nn.Module):

    def __init__(self, img_size: int, ch: int = 128, patch_size: int = 5) -> None:
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(f'img_size ({img_size}) must be divisible by patch_size ({patch_size})')

        self.patchify = nn.Conv2d(3, ch, patch_size, stride=patch_size, padding=0)

        self.spatial = (img_size - patch_size) // patch_size + 1
        self.n_tokens = self.spatial * self.spatial
        self.ch = ch
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        return self.patchify(x)


class SOCCNNEncoder(nn.Module):
    """CNN encoder used for Sort-of-CLEVR."""

    def __init__(self, img_size: int, ch: int = 24) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),

            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),

            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),

            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )

        spatial = img_size
        for _ in range(4):
            spatial = (spatial + 1) // 2

        self.spatial = spatial
        self.n_tokens = spatial * spatial
        self.ch = ch

    def forward(self, x: Tensor) -> Tensor:
        # -> (B, self.ch, self.spatial, self.spatial)
        return self.cnn(x)


class SQOOPCNNEncoder(nn.Module):
    """CNN encoder used for SQOOP."""

    def __init__(self, img_size: int, ch: int = 64) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),

            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),

            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),

            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )

        self.spatial = img_size // 4
        self.n_tokens = self.spatial * self.spatial
        self.ch = ch

    def forward(self, x: Tensor) -> Tensor:
        # -> (B, self.ch, self.spatial, self.spatial)
        return self.cnn(x)


ImageEncoder = PatchifyEncoder | SOCCNNEncoder | SQOOPCNNEncoder


def build_image_encoder(enc_config: dict[str, Any], dataset: str) -> ImageEncoder:
    name = enc_config.get('name')

    if dataset == 'sort_of_clevr':
        img_size = SOCConst.IMG_SIZE
    elif dataset == 'sqoop':
        img_size = SQOOPConst.IMG_SIZE
    else:
        raise ValueError(f'unrecognised dataset: {dataset!r}')

    if name == 'patchify':
        return PatchifyEncoder(
            img_size=img_size,
            ch=enc_config.get('ch', 128),
            patch_size=enc_config.get('patch_size', 5),
        )

    if name == 'cnn':
        if dataset == 'sort_of_clevr':
            return SOCCNNEncoder(img_size=img_size, ch=enc_config.get('ch', 24))

        if dataset == 'sqoop':
            return SQOOPCNNEncoder(img_size=img_size, ch=enc_config.get('ch', 64))

    raise ValueError(f'unrecognised encoder {name!r} for dataset {dataset!r}')