from __future__ import annotations

from dataclasses import dataclass

from omegaconf import MISSING
import torch.nn as nn


@dataclass
class EncoderConfig:
    name: str = MISSING


@dataclass
class PatchifyEncoderConfig(EncoderConfig):
    name: str = 'patchify'  
    ch: int = 128
    patch_size: int = 5


@dataclass
class CNNEncoderConfig(EncoderConfig):
    name: str = 'cnn'  
    ch: int = 128
    hidden: int = 64

class PatchifyEncoder(nn.Module):

    def __init__(self, img_size: int, ch: int = 128, patch_size: int = 5):
        super().__init__()

        self.patchify = nn.Conv2d(3, ch, patch_size, stride=patch_size, padding=0)
        self.spatial = (img_size - patch_size) // patch_size + 1
        self.n_tokens = self.spatial * self.spatial
        self.ch = ch
        self.patch_size = patch_size

    def forward(self, x):
        return self.patchify(x)
    
    @classmethod
    def from_config(cls, cfg: PatchifyEncoderConfig, img_size: int) -> PatchifyEncoder:
        return cls(
            img_size=img_size,
            ch=cfg.ch,
            patch_size=cfg.patch_size
        )


class CNNEncoder(nn.Module):

    def __init__(self, img_size: int, ch: int = 128, hidden: int = 64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, 2 * hidden, 3, stride=2, padding=1),
            nn.GroupNorm(8, 2 * hidden), nn.SiLU(),
            nn.Conv2d(2 * hidden, ch, 3, stride=2, padding=1),
        )
        spatial = img_size
        for _ in range(3):
            spatial = (spatial + 1) // 2
        self.spatial = spatial
        self.n_tokens = spatial * spatial
        self.ch = ch

    def forward(self, x):
        # x: (B, C, H, W)
        return self.cnn(x) # -> (B, )
    
    @classmethod
    def from_config(cls, cfg: CNNEncoderConfig, img_size: int) -> CNNEncoder:
        return cls(
            img_size=img_size,
            ch=cfg.ch,
            hidden=cfg.hidden
        )


def build_encoder(spec: dict, img_size: int) -> nn.Module:
    """Build an encoder from the plain-dict convention used by configs.

    `spec` is {'name': 'patchify'|'cnn', ...kwargs}. A dict rather than a
    structured config group so that adding an encoder argument needs no
    matching schema change, and so every model that takes an encoder
    spells it the same way.
    """
    kwargs = dict(spec)
    name = kwargs.pop('name', 'patchify')
    kwargs.pop('obj_size', None)        # objects-encoder key, ignored here
    if name == 'patchify':
        kwargs.pop('hidden', None)
        return PatchifyEncoder(img_size=img_size, **kwargs)
    if name == 'cnn':
        kwargs.pop('patch_size', None)
        return CNNEncoder(img_size=img_size, **kwargs)
    raise ValueError(f'unsupported encoder: {name!r}')
