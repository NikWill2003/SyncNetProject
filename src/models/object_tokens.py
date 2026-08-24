"""Object tokens without perception.

Sort-of-CLEVR draws one flat-coloured object per colour on a white
background with no anti-aliasing, so an exact object list can be read off
the pixels: a per-colour mask gives area, centroid and bounding box, and
area / bbox-area separates a circle (~0.7) from a square (1.0). This is
the "state description" version of the task (as in the original CLEVR
paper's state-description split), computed inside the model so that no
dataset or loader changes are needed.

It exists to answer one question: with perception removed, do the
modules bind to objects and do the gates form the question's coalition?
A token here IS an entity, so module-token binding is object binding and
the module-module gate has a job the grid never gave it (e.g. the
queried object must hear every other object for "closest shape": a
star graph, which is frustrated on S^1 -- see rho.py).

`found` is exported as a metric (should read n_colours, e.g. 6.0); if it
does not, the images are not what this tokeniser assumes and every
object-token result is invalid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ObjectTokenizer(nn.Module):

    def __init__(self, colours_bgr: list[tuple[int, int, int]],
                 img_size: int, obj_size: int, tol: float = 0.08) -> None:
        super().__init__()
        cols = torch.tensor(colours_bgr, dtype=torch.float32) / 255.0
        self.register_buffer('colours', cols)                   # (n, 3)
        self.n_objects = cols.shape[0]
        self.tol = tol
        self.img_size = img_size
        self.side = 2 * obj_size + 1                            # drawn square side
        ar = torch.arange(img_size, dtype=torch.float32)
        self.register_buffer('ar', ar)
        # feature layout: one-hot(n) | cx cy | area | fill | w h
        self.feat_dim = self.n_objects + 6

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """images (B, 3, H, W) in [0, 1], channel order as the colour table.
        -> feats (B, n, feat_dim), found (B, n) bool."""
        B, _, H, W = images.shape
        n = self.n_objects
        diff = (images.unsqueeze(1) - self.colours[None, :, :, None, None]).abs()
        mask = (diff.amax(dim=2) < self.tol).float()            # (B, n, H, W)
        area = mask.sum(dim=(-1, -2))                           # (B, n)
        found = area > 0
        a_safe = area.clamp(min=1.0)
        xs = self.ar[:W]; ys = self.ar[:H]
        cx = (mask.sum(-2) * xs).sum(-1) / a_safe               # (B, n)
        cy = (mask.sum(-1) * ys).sum(-1) / a_safe
        colm = mask.amax(-2)                                    # (B, n, W)
        rowm = mask.amax(-1)                                    # (B, n, H)
        xmax = (colm * xs).amax(-1)
        xmin = (colm * xs + (1 - colm) * W).amin(-1)
        ymax = (rowm * ys).amax(-1)
        ymin = (rowm * ys + (1 - rowm) * H).amin(-1)
        w = (xmax - xmin + 1).clamp(min=1.0)
        h = (ymax - ymin + 1).clamp(min=1.0)
        fill = area / (w * h)
        s = float(self.side)
        onehot = torch.eye(n, device=images.device).unsqueeze(0).expand(B, n, n)
        feats = torch.cat([
            onehot,
            (cx / W).unsqueeze(-1), (cy / H).unsqueeze(-1),
            (area / (s * s)).unsqueeze(-1), fill.unsqueeze(-1),
            (w / s).unsqueeze(-1), (h / s).unsqueeze(-1),
        ], dim=-1)
        feats = feats * found.unsqueeze(-1).float()
        return feats, found
