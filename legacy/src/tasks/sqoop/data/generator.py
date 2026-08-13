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
 8. LABELS ARE BALANCED PER (pair, rel) CELL, NOT GLOBALLY. The original
    (and the first version of this port) alternated on the global example
    index, `(i % 2) == 0 -> true`. That makes the *split* exactly balanced
    while leaving each individual question -- which is exactly a
    (pair, rel) cell -- Binomially skewed, ~0.59 majority class at the
    repeat counts we use. A question-only model can read that skew off
    the question tokens alone and beat 0.500, which destroys the control
    the whole SQOOP result rests on. We now assign relations and labels
    deterministically: every (pair, rel) cell gets exactly
    repeats // len(RELATIONS) examples, exactly half of them positive,
    and the resulting schedule is shuffled once. Bayes-optimal
    question-only accuracy is therefore exactly 0.500 on every split, by
    construction rather than in expectation. Costs a divisibility
    constraint on num_repeats (see _check_repeats).
 9. Both unbounded `while True` loops are capped. Exhaustion RAISES
    rather than falling back to an easier scene: a fallback would have to
    change the label or the constraint, and either silently re-introduces
    the class skew that deviation 8 exists to remove. If generation
    cannot satisfy a request, that is a configuration bug and should be
    loud.
10. Optional `restrict_positive` (default False = original semantics).
    See the DISTRACTOR SHAPE ASYMMETRY note in _gen_example.

Original quirks kept ON PURPOSE (bit-for-bit semantics parity):
 * `relate('above')` means pos_y > other -- i.e. *lower* on screen with
   image-coordinate y. Consistent across generation, so learnable; do
   not "fix".
 * Negative scenes are hard negatives: they contain x rel (something)
   and (something) rel y while x rel y is false.
 * The 5 px per-axis minimum separation check rejects on |dx| < 5 OR
   |dy| < 5 against every existing object, exactly as the original.

NOTE ON REPRODUCIBILITY: distractor shapes are now drawn uniformly from
the allowed list instead of by rejection sampling against the restricted
set. Same distribution, different RNG consumption, so byte-identical
reproduction of datasets built before this change is not possible. The
label rebalance already made regeneration mandatory.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

from .constants import (
    SHAPES, RELATIONS, encode_question,
)
from ..config import SqoopDataConfig

_FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

# A (pair, rel) cell must be splittable into equal positive/negative
# halves, and every relation must get the same number of cells, so the
# per-pair repeat count has to be a multiple of 2 * |RELATIONS|.
_REPEAT_STEP = 2 * len(RELATIONS)

# scene[2] and scene[3] carry the hard negative; fewer than 4 objects
# and _gen_example indexes off the end of the scene.
_MIN_NUM_OBJECTS = 4

# Attempts before we declare a (pair, rel, label) request unsatisfiable.
_MAX_EXAMPLE_ATTEMPTS = 2000
_MAX_FILL_RESETS = 20


def _cfg_get(cfg, key: str, default):
    """Attribute lookup tolerant of both dataclass and DictConfig cfgs."""
    try:
        val = getattr(cfg, key)
    except Exception:
        return default
    return default if val is None else val


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
        self.shape: str|None = shape

    def overlap(self, other: '_Object') -> bool:
        assert (self.pos is not None) and (other.pos is not None)
        min_dist = (self.size + other.size) // 2 + 1
        return (abs(self.pos[0] - other.pos[0]) < min_dist and
                abs(self.pos[1] - other.pos[1]) < min_dist)

    def relate(self, rel: str, other: '_Object') -> bool:
        assert (self.pos is not None) and (other.pos is not None)
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
    assert(all(o.pos is not None for o in objects))
    size = int(rng.randint(min_obj, max_obj + 1))
    obj = _Object(size)

    min_c = obj.size // 2 + 1
    max_c = img_size - obj.size // 2 - 1

    if rel is not None:
        anchor = objects[rel_obj].pos
        assert anchor is not None
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
    else:
        min_cx = min_cy = min_c
        max_cx = max_cy = max_c

    # Also guards the unconstrained branch: with a large max_obj_size
    # against a small img_size the plain range collapses too, and
    # rng.randint would raise instead of rejecting.
    if min_cx >= max_cx or min_cy >= max_cy:
        return None

    for _ in range(10):
        x = int(rng.randint(min_cx, max_cx))
        y = int(rng.randint(min_cy, max_cy))
        obj.pos = (x, y)
        if (any(abs(x - o.pos[0]) < 5 for o in objects) or #type: ignore
                any(abs(y - o.pos[1]) < 5 for o in objects)): #type: ignore
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
        ) -> list[_Object] | None:
    """Pad a scene up to num_objects with distractors, or None on failure.

    Returning None matters: the original spun here forever. The reset on
    ten consecutive placement failures never terminates when the scene
    simply cannot hold num_objects at this img_size / obj size -- it is
    not a rejection loop that eventually succeeds, it is a livelock, and
    it sits *inside* the outer retry so an outer cap alone would never
    fire.
    """
    orig = list(objects)
    restricted = {o.shape for o in orig} if restrict else set()
    allowed = [s for s in SHAPES if s not in restricted]
    if not allowed:
        raise ValueError(
            'no distractor shapes available: SHAPES is exhausted by the '
            'restricted set'
        )

    out = list(orig)
    failures = 0
    resets = 0
    while len(out) < num_objects:
        # Draw from the allowed list directly. The original rejection
        # loop over SHAPES was itself unbounded and would hang if the
        # restricted set ever covered the vocabulary.
        shape = allowed[int(rng.randint(len(allowed)))]
        new = _get_random_spot(rng, out, img_size, min_obj, max_obj)
        if new is None:
            failures += 1
            if failures == 10:
                out = list(orig)
                failures = 0
                resets += 1
                if resets > _MAX_FILL_RESETS:
                    return None
            continue
        new.shape = shape
        out.append(new)
    return out


def _draw(objects: list[_Object], img_size: int) -> np.ndarray:
    assert(all(o.pos is not None for o in objects))

    img = Image.new('RGB', (img_size, img_size))
    for obj in objects:
        assert (obj.pos is not None) and (obj.shape is not None)
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
        restrict_positive: bool = False,
        ):
    """One (scene, question, answer) attempt. Returns None on rejection.

    DISTRACTOR SHAPE ASYMMETRY (the `restrict` flag below)
    ------------------------------------------------------
    In the original, negative scenes pass restrict=True -- distractors may
    not reuse shape x or shape y -- while positive scenes pass
    restrict=False. So a negative scene contains exactly one x and exactly
    one y, and a positive scene may contain several of either.

    That is a one-sided, non-relational shortcut: `count(x) > 1 or
    count(y) > 1` implies the answer is True, and never fires on a
    negative. At |SHAPES| = 36 and num_objects = 5 it fires on roughly
    1 - (34/36)^3 = 16% of positives, so a model that learns only to
    count duplicate glyphs sits near 0.58 without ever representing a
    spatial relation. On a benchmark whose entire purpose is to measure
    relational compositional generalisation, that contaminates the
    headline number and, worse, contaminates it more for models that are
    better at detection than at relations -- which is the axis under test.

    Kept as the default for parity with Bahdanau et al., because the
    published baselines carry the same shortcut and dropping it silently
    would make our numbers non-comparable. Set restrict_positive=True for
    a clean run; report both if the gap is large.

    A second, subtler asymmetry has no flag and is not fixed: negatives
    additionally require obj1 rel obj4 and obj3 rel obj2, so distractor
    *positions* in negatives are rejection-biased in a way positives are
    not. Worth measuring in the validator (compare the marginal x/y
    distributions of scene[2:4] across classes) before trusting any small
    effect.
    """
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
            restrict=restrict_positive,
        )
        if scene is None:
            return None
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
        if scene is None:
            return None
        # hard negative: require x rel y' and x' rel y to hold
        obj3, obj4 = scene[2], scene[3]
        if not obj1.relate(rel, obj4):
            return None
        if not obj3.relate(rel, obj2):
            return None

    return _draw(scene, img_size), encode_question(x, rel, y), int(label)


def _check_repeats(repeats: int, name: str) -> None:
    if repeats <= 0 or repeats % _REPEAT_STEP:
        raise ValueError(
            f'{name}={repeats} must be a positive multiple of '
            f'{_REPEAT_STEP} (= 2 labels x {len(RELATIONS)} relations). '
            'Each (pair, rel) cell has to split into equal positive and '
            'negative halves; anything else leaves a per-question class '
            'skew that a question-only model can exploit.'
        )


def _build_schedule(
        pairs_unique: list[tuple[str, str]],
        repeats: int,
        rng: np.random.RandomState,
        ) -> list[tuple[tuple[str, str], str, bool]]:
    """Exactly balanced (pair, rel, label) schedule, globally shuffled.

    Balance is by construction, not by sampling: every (pair, rel) cell
    holds repeats // |RELATIONS| examples, half positive. Labels are then
    decorrelated from position by one global shuffle -- assigning them in
    strict alternation within a cell would hit the same marginal but tie
    the label to the index within the cell, which is a fresh leak for
    anything order-sensitive.
    """
    per_cell = repeats // len(RELATIONS)
    schedule: list[tuple[tuple[str, str], str, bool]] = []
    for pair in pairs_unique:
        for rel in RELATIONS:
            for k in range(per_cell):
                schedule.append((pair, rel, k < per_cell // 2))

    order = rng.permutation(len(schedule))
    schedule = [schedule[i] for i in order]

    # cheap self-check: this is the property the question-only control
    # depends on, so assert it here rather than discovering it downstream
    pos = Counter()
    tot = Counter()
    for pair, rel, label in schedule:
        tot[(pair, rel)] += 1
        pos[(pair, rel)] += int(label)
    for cell, n in tot.items():
        assert pos[cell] * 2 == n, f'unbalanced cell {cell}: {pos[cell]}/{n}'

    return schedule


def _gen_split(
        schedule: list[tuple[tuple[str, str], str, bool]],
        seed: int,
        num_objects: int,
        img_size: int,
        min_obj: int,
        max_obj: int,
        restrict_positive: bool = False,
        ) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(schedule)

    images = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
    questions = np.empty((n, 3), dtype=np.int64)
    answers = np.empty((n,), dtype=np.int64)

    total_attempts = 0
    worst_attempts = 0

    for i, (pair, rel, label) in enumerate(schedule):
        for attempt in range(1, _MAX_EXAMPLE_ATTEMPTS + 1):
            out = _gen_example(
                pair, rel, label, rng,
                num_objects, img_size, min_obj, max_obj,
                restrict_positive=restrict_positive,
            )
            if out is not None:
                break
        else:
            # Deliberately fatal. Every alternative -- flip the label,
            # drop the hard-negative requirement, shrink the scene --
            # trades a crash for a silent distribution shift correlated
            # with constraint difficulty, which is strictly worse than
            # the skew this rewrite removed.
            raise RuntimeError(
                f'could not generate example {i} '
                f'(pair={pair}, rel={rel}, label={label}) in '
                f'{_MAX_EXAMPLE_ATTEMPTS} attempts. Check num_objects='
                f'{num_objects} against img_size={img_size} and object '
                f'sizes {min_obj}-{max_obj}: negatives need room for a '
                'hard negative and are far more constrained than '
                'positives.'
            )
        total_attempts += attempt
        worst_attempts = max(worst_attempts, attempt)
        images[i], questions[i], answers[i] = out

    print(
        f'  attempts/example: mean {total_attempts / max(n, 1):.2f}, '
        f'max {worst_attempts}; positive rate {answers.mean():.4f}'
    )
    return {'images': images, 'questions': questions, 'answers': answers}


def prepare_sqoop(data_cfg: SqoopDataConfig) -> None:
    """Build train / val_seen / val_unseen / test_unseen npz files."""
    cfg = data_cfg
    out_dir = Path(cfg.root) / cfg.dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg.seed)
    rhs = int(cfg.rhs_variety)

    num_objects = int(cfg.num_objects)
    if num_objects < _MIN_NUM_OBJECTS:
        raise ValueError(
            f'num_objects={num_objects} but the hard-negative construction '
            f'reads scene[2] and scene[3], so at least {_MIN_NUM_OBJECTS} '
            'objects are required. Below that the negative branch raises '
            'IndexError on the first negative example.'
        )

    restrict_positive = bool(_cfg_get(cfg, 'restrict_positive', False))

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

    # ------------------------------------------------- repeat budgets
    repeats = int(cfg.num_repeats)
    _check_repeats(repeats, 'num_repeats')

    # max_train_pairs now caps the *per-pair repeat count* rather than
    # truncating a shuffled example list. Truncation after shuffling
    # would slice cells apart and undo the balance guarantee.
    max_train = int(cfg.max_train_pairs)
    if max_train > 0:
        cap = max_train // max(len(train_pairs_unique), 1)
        cap -= cap % _REPEAT_STEP
        if cap <= 0:
            raise ValueError(
                f'max_train_pairs={max_train} leaves under {_REPEAT_STEP} '
                f'examples per pair across {len(train_pairs_unique)} '
                'training pairs, which cannot be balanced.'
            )
        if cap < repeats:
            print(
                f'max_train_pairs={max_train}: reducing num_repeats '
                f'{repeats} -> {cap} (kept a multiple of {_REPEAT_STEP})'
            )
            repeats = cap

    repeats_eval = int(cfg.num_repeats_eval)
    _check_repeats(repeats_eval, 'num_repeats_eval')

    left = sorted(unseen)
    py_rng.shuffle(left)
    val_slice = len(left) // 2
    val_unseen_pairs = left[:val_slice]
    test_unseen_pairs = left[val_slice:]

    sched_rng = np.random.RandomState(base_seed)
    schedules = {
        'train': (
            _build_schedule(train_pairs_unique, repeats, sched_rng),
            base_seed + 1,
        ),
        # deviation 2: fresh scenes over *seen* pairs, for early stopping
        'val_seen': (
            _build_schedule(train_pairs_unique, repeats_eval, sched_rng),
            base_seed + 2,
        ),
        'val_unseen': (
            _build_schedule(val_unseen_pairs, repeats_eval, sched_rng),
            base_seed + 3,
        ),
        'test_unseen': (
            _build_schedule(test_unseen_pairs, repeats_eval, sched_rng),
            base_seed + 4,
        ),
    }

    print(
        f'sqoop rhs={rhs}: {len(train_pairs_unique)} train pairs '
        f'({len(schedules["train"][0])} train ex at {repeats}/pair), '
        f'{len(left)} unseen pairs '
        f'({len(schedules["val_unseen"][0])} val_unseen / '
        f'{len(schedules["test_unseen"][0])} test_unseen ex), '
        f'{len(schedules["val_seen"][0])} val_seen ex; '
        f'restrict_positive={restrict_positive}'
    )

    for name, (schedule, seed) in schedules.items():
        print(f'building {name} ({len(schedule)} examples)...')
        arrays = _gen_split(
            schedule, seed,
            num_objects=num_objects,
            img_size=int(cfg.img_size),
            min_obj=int(cfg.min_obj_size),
            max_obj=int(cfg.max_obj_size),
            restrict_positive=restrict_positive,
        )
        np.savez_compressed(out_dir / f'{name}.npz', allow_pickle=True, **arrays)
        print(f'saved {name}.npz to {out_dir}')

    # manifest (handles dataclass and DictConfig inputs -- past fix)
    if is_dataclass(cfg):
        manifest = asdict(cfg)
    else:
        manifest = OmegaConf.to_container(cfg, resolve=True)
    manifest['restrict_positive'] = restrict_positive # type: ignore
    manifest['num_repeats_effective'] = repeats # type: ignore
    OmegaConf.save(OmegaConf.create(manifest), out_dir / 'manifest.yaml')