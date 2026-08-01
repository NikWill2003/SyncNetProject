"""SQOOP dataset generator.

Port of scripts/generate_sqoop.py from Bahdanau et al. 2019
(systematic-generalization-sqoop). Scene semantics -- object placement,
hard-negative construction, the systematic #rhs/lhs split -- follow the
original exactly. Deviations from the original, all deliberate:

 1. Output is one .npz per split (uint8 HWC images, int64 questions of
    shape (N, 3) in the joint SHAPES+RELATIONS vocab, int64 answers),
    not h5py + PNG buffers + program annotations.
 2. Four splits instead of three: train / val_seen / val_unseen /
    test_unseen. val_seen holds *training pairs* rendered in fresh
    scenes, because early stopping must not peek at unseen pairs.
 3. Fully seeded and reproducible (single base seed fans out per split);
    the original mixed `random` module state with per-split RandomStates.
 4. Pillow >= 10: font.getsize() -> font.getbbox().
 5. Font-size semantics fixed: the requested object size is the rendered
    glyph bbox height, not the font point size.
 6. DejaVuSans-Bold instead of arial.ttf (present on linux boxes).
 7. Rotation removed entirely (the original sampled angles but its
    rotation code was commented out -- dead state affecting nothing but
    spacing; we drop it for honesty).
 8. Answers still alternate positive/negative per pair slot ((i % 2)
    == 0 -> true), as in the original, so classes are exactly balanced.

Original quirks kept ON PURPOSE (bit-for-bit semantics parity):
 * `relate('above')` means pos_y > other -- i.e. *lower* on screen with
   image-coordinate y. Consistent across generation, so learnable; do
   not "fix".
 * Negative scenes are hard negatives: they contain x rel (something)
   and (something) rel y while x rel y is false.
 * The 5 px per-axis minimum separation check rejects on |dx| < 5 OR
   |dy| < 5 against every existing object, exactly as the original.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

from .constants import (
    SHAPES, RELATIONS, encode_question,
)

_FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _font_for_size(target_px: int) -> ImageFont.FreeTypeFont:
    """Font whose capital-letter bbox height is ~target_px (deviation 5)."""
    if target_px not in _FONT_CACHE:
        pt = target_px
        for _ in range(6):
            f = ImageFont.truetype(_FONT_PATH, pt)
            _, t, _, b = f.getbbox('A')
            h = b - t
            if h == target_px or pt <= 4:
                break
            pt = max(4, round(pt * target_px / max(h, 1)))
        _FONT_CACHE[target_px] = ImageFont.truetype(_FONT_PATH, pt)
    return _FONT_CACHE[target_px]


class _Object:
    __slots__ = ('size', 'font', 'pos', 'shape')

    def __init__(self, size: int, pos=None, shape=None):
        self.size = size
        self.font = _font_for_size(size)
        self.pos = pos
        self.shape = shape

    def overlap(self, other: '_Object') -> bool:
        min_dist = (self.size + other.size) // 2 + 1
        return (abs(self.pos[0] - other.pos[0]) < min_dist and
                abs(self.pos[1] - other.pos[1]) < min_dist)

    def relate(self, rel: str, other: '_Object') -> bool:
        if rel == 'left_of':
            return self.pos[0] < other.pos[0]
        if rel == 'right_of':
            return self.pos[0] > other.pos[0]
        if rel == 'above':
            return self.pos[1] > other.pos[1]
        if rel == 'below':
            return self.pos[1] < other.pos[1]
        raise ValueError(rel)


def _get_random_spot(
        rng: np.random.RandomState,
        objects: list[_Object],
        img_size: int,
        min_obj: int,
        max_obj: int,
        rel: str | None = None,
        rel_holds: bool = False,
        rel_obj: int = 0,
        ) -> _Object | None:
    size = int(rng.randint(min_obj, max_obj + 1))
    obj = _Object(size)

    min_c = obj.size // 2 + 1
    max_c = img_size - obj.size // 2 - 1

    if rel is not None:
        anchor = objects[rel_obj].pos
        if not rel_holds:
            max_cx = anchor[0] if rel == 'left_of' else max_c
            min_cx = anchor[0] if rel == 'right_of' else min_c
            max_cy = anchor[1] if rel == 'below' else max_c
            min_cy = anchor[1] if rel == 'above' else min_c
        else:
            min_cx = anchor[0] if rel == 'left_of' else min_c
            max_cx = anchor[0] if rel == 'right_of' else max_c
            min_cy = anchor[1] if rel == 'below' else min_c
            max_cy = anchor[1] if rel == 'above' else max_c
        if min_cx >= max_cx or min_cy >= max_cy:
            return None
    else:
        min_cx = min_cy = min_c
        max_cx = max_cy = max_c

    for _ in range(10):
        x = int(rng.randint(min_cx, max_cx))
        y = int(rng.randint(min_cy, max_cy))
        obj.pos = (x, y)
        if (any(abs(x - o.pos[0]) < 5 for o in objects) or
                any(abs(y - o.pos[1]) < 5 for o in objects)):
            continue
        if any(obj.overlap(o) for o in objects):
            continue
        return obj
    return None


def _fill_scene(
        rng: np.random.RandomState,
        objects: list[_Object],
        num_objects: int,
        img_size: int,
        min_obj: int,
        max_obj: int,
        restrict: bool,
        ) -> list[_Object]:
    orig = list(objects)
    restricted = {o.shape for o in orig} if restrict else set()

    out = list(orig)
    failures = 0
    while len(out) < num_objects:
        while True:
            shape = SHAPES[rng.randint(len(SHAPES))]
            if shape not in restricted:
                break
        new = _get_random_spot(rng, out, img_size, min_obj, max_obj)
        if new is None:
            failures += 1
            if failures == 10:
                out = list(orig)
                failures = 0
            continue
        new.shape = shape
        out.append(new)
    return out


def _draw(objects: list[_Object], img_size: int) -> np.ndarray:
    img = Image.new('RGB', (img_size, img_size))
    for obj in objects:
        glyph = Image.new('RGBA', (obj.size + 4, obj.size + 4))
        d = ImageDraw.Draw(glyph)
        _, t, _, _ = obj.font.getbbox(obj.shape)
        d.text((0, -t), obj.shape, font=obj.font, fill='green')
        img.paste(
            glyph,
            (obj.pos[0] - glyph.size[0] // 2, obj.pos[1] - glyph.size[1] // 2),
            glyph,
        )
    return np.asarray(img, dtype=np.uint8)


def _gen_example(
        pair: tuple[str, str],
        rel: str,
        label: bool,
        rng: np.random.RandomState,
        num_objects: int,
        img_size: int,
        min_obj: int,
        max_obj: int,
        ):
    """One (scene, question, answer) attempt. Returns None on rejection."""
    x, y = pair
    if label:
        obj1 = _get_random_spot(rng, [], img_size, min_obj, max_obj)
        if obj1 is None:
            return None
        obj2 = _get_random_spot(rng, [obj1], img_size, min_obj, max_obj)
        if not obj2 or not obj1.relate(rel, obj2):
            return None
        obj1.shape, obj2.shape = x, y
        scene = _fill_scene(
            rng, [obj1, obj2], num_objects, img_size, min_obj, max_obj,
            restrict=False,
        )
    else:
        obj1 = _get_random_spot(rng, [], img_size, min_obj, max_obj)
        if obj1 is None:
            return None
        obj2 = _get_random_spot(
            rng, [obj1], img_size, min_obj, max_obj,
            rel=rel, rel_holds=False,
        )
        if not obj2 or obj1.relate(rel, obj2):
            return None
        obj1.shape, obj2.shape = x, y
        scene = _fill_scene(
            rng, [obj1, obj2], num_objects, img_size, min_obj, max_obj,
            restrict=True,
        )
        # hard negative: require x rel y' and x' rel y to hold
        obj3, obj4 = scene[2], scene[3]
        if not obj1.relate(rel, obj4):
            return None
        if not obj3.relate(rel, obj2):
            return None

    return _draw(scene, img_size), encode_question(x, rel, y), int(label)


def _gen_split(
        pairs: list[tuple[str, str]],
        seed: int,
        num_objects: int,
        img_size: int,
        min_obj: int,
        max_obj: int,
        ) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    rels = [RELATIONS[rng.randint(len(RELATIONS))] for _ in pairs]

    images = np.empty((len(pairs), img_size, img_size, 3), dtype=np.uint8)
    questions = np.empty((len(pairs), 3), dtype=np.int64)
    answers = np.empty((len(pairs),), dtype=np.int64)

    for i, (pair, rel) in enumerate(zip(pairs, rels)):
        label = (i % 2) == 0
        while True:
            out = _gen_example(
                pair, rel, label, rng,
                num_objects, img_size, min_obj, max_obj,
            )
            if out is not None:
                break
        images[i], questions[i], answers[i] = out

    return {'images': images, 'questions': questions, 'answers': answers}


def prepare_sqoop(data_cfg) -> None:
    """Build train / val_seen / val_unseen / test_unseen npz files."""
    cfg = data_cfg
    out_dir = Path(cfg.root) / cfg.dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg.seed)
    rhs = int(cfg.rhs_variety)

    # ------------------------------------------------- pair bookkeeping
    py_rng = random.Random(base_seed)
    all_pairs = {(x, y) for x in SHAPES for y in SHAPES if x != y}
    unseen = set(all_pairs)
    train_pairs_unique: list[tuple[str, str]] = []
    for i, x in enumerate(SHAPES):
        ys = py_rng.sample(SHAPES[:i] + SHAPES[i + 1:], rhs)
        for y in ys:
            unseen.remove((x, y))
            train_pairs_unique.append((x, y))

    train_pairs = train_pairs_unique * int(cfg.num_repeats)
    py_rng.shuffle(train_pairs)
    if int(cfg.max_train_pairs) > 0:
        train_pairs = train_pairs[: int(cfg.max_train_pairs)]

    left = sorted(unseen)
    py_rng.shuffle(left)
    val_slice = len(left) // 2
    val_unseen_pairs = [
        p for p in left[:val_slice] for _ in range(int(cfg.num_repeats_eval))
    ]
    test_unseen_pairs = [
        p for p in left[val_slice:] for _ in range(int(cfg.num_repeats_eval))
    ]
    # deviation 2: fresh scenes over *seen* pairs, for early stopping
    val_seen_pairs = [
        p for p in train_pairs_unique
        for _ in range(int(cfg.num_repeats_eval))
    ]
    py_rng.shuffle(val_seen_pairs)

    print(
        f'sqoop rhs={rhs}: {len(train_pairs_unique)} train pairs '
        f'({len(train_pairs)} train ex), {len(left)} unseen pairs '
        f'({len(val_unseen_pairs)} val_unseen / {len(test_unseen_pairs)} '
        f'test_unseen ex), {len(val_seen_pairs)} val_seen ex'
    )

    splits = {
        'train': (train_pairs, base_seed + 1),
        'val_seen': (val_seen_pairs, base_seed + 2),
        'val_unseen': (val_unseen_pairs, base_seed + 3),
        'test_unseen': (test_unseen_pairs, base_seed + 4),
    }

    for name, (pairs, seed) in splits.items():
        print(f'building {name} ({len(pairs)} examples)...')
        arrays = _gen_split(
            pairs, seed,
            num_objects=int(cfg.num_objects),
            img_size=int(cfg.img_size),
            min_obj=int(cfg.min_obj_size),
            max_obj=int(cfg.max_obj_size),
        )
        np.savez_compressed(out_dir / f'{name}.npz', **arrays)
        print(f'saved {name}.npz to {out_dir}')

    # manifest (handles dataclass and DictConfig inputs -- past fix)
    if is_dataclass(cfg):
        manifest = asdict(cfg)
    else:
        manifest = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(OmegaConf.create(manifest), out_dir / 'manifest.yaml')
