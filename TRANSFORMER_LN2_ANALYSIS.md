# Why the SQOOP VQATransformer sits at ln2 — analysis, new hypotheses, and a one-night battery

Prepared from the repo zip (code read in full for the SQOOP path) plus four
measurements run locally on freshly generated SQOOP data. Everything marked
**[measured]** was run here; everything marked **[inferred]** is a reading of
the code or of your result table.

---

## 0. The single most useful thing in your result set

`conf/experiment/sqoop/baselines/iid_control.yaml` is a **perfectly matched
control** and you have not been using it as one.

Compare it line by line against the rhs=18 transformer cells:

| | rhs≤18 cells | iid_control (rhs=35) |
|---|---|---|
| model | `sqoop/transformer` defaults (h=128, L=4, ffn=4, patch 8, patch_emb 96, broadcast_cat, cls) | **identical** — no `model:` block at all, so the same defaults |
| optim | AdamW 3e-4, wd 0.01, warmup_cosine 2 000 | **identical** |
| train | 100k steps, bs 256, clip 1.0, **bf16**, gpu_cached | **identical** |
| data | rhs_variety 18 | rhs_variety 35 |

One run therefore controls, simultaneously, every hypothesis you have been
testing one at a time: **capacity, learning rate, warmup, precision, readout,
conditioning, encoder, patch size, clipping, weight decay, step budget.**
All of them were held fixed while the model went from ln2 to 0.672.

**[inferred] The cause is a property of the rhs≤18 training distribution
interacting with this architecture. It is not a hyperparameter and not a
wiring fault.** Every one of the eight eliminated hypotheses was eliminated a
second time, for free, by this run. That also means further model-side sweeps
have a low expected yield, which matters given four weeks.

The corollary is the experiment you have not run: **rhs = 34.** It is the
largest variety that still takes the ordinary (non-IID) code path — 1 224 of
1 260 pairs in train, 36 unseen pairs, `val_unseen`/`test_unseen` both built,
`_repeats_for(25 600, 18)` fine. So:

- rhs=34 **learns** → the effect is continuous in pair variety; bisect to find
  the knee, and you have a real, reportable finding about attention baselines
  and question variety.
- rhs=34 **is flat** → only the literal IID code path works, which means the
  "exception" is a property of `iid = not val_unseen_pairs` (a different
  schedule, different split config) rather than of variety — i.e. an artefact,
  and the honest write-up is three negatives.

Either answer closes the question. Cost: one 40-minute dataset build plus one
15-minute run.

---

## 1. Two of your standing beliefs are wrong in the code

### 1.1 The encoder is dead as a hypothesis, independently of the unlogged sweep

`conf/model/sqoop/syncnet.yaml`:

```yaml
encoder:
  name: patchify
  ch: 128
  patch_size: 8
```

and `VQASyncNet.__init__` raises unless `name == 'patchify'`. So the syncnet
runs the **identical single strided `Conv2d(3, 128, 8, stride=8)`** on the
identical 64 px images — and at module_dim 768 it reaches **train accuracy
0.9997** and test 0.873.

A linear per-patch projection therefore resolves 36 green glyphs straddling
8 px boundaries well enough to solve SQOOP to 4 nines on the training set.
You do not need to trust the inferred `encoder_name` column: the encoder
hypothesis is refuted by an experiment you already ran for another purpose.

Three places assert the opposite and should be deleted before they reach the
thesis:

- `src/models/vqa_transformer.py` lines ~91–100 ("the transformer and the
  syncnet both use patchify and both sit at ln2 or below" — the syncnet does
  not);
- the same claim in `conf/experiment/sqoop/diagnostics/transformer_encoder.yaml`;
- the readout comment at lines ~196–209 ("every pooled readout measured so far
  sits at exactly ln2 and cannot fit the training set"). The syncnet's readout
  is *also* a pooling — attention-weighted `content_head` over patches,
  `content_dim=8` per module — and it fits the training set. Pooling per se is
  not the problem either; what the syncnet has is (a) `F.normalize` on both
  query and key so patch magnitude drops out, (b) `beta=5` sharpening, (c) a
  question-derived query via `h_init(q)`, and (d) `pos_emb` added *before*
  `content_head`, so the pooled vector carries the attended glyph's position.

### 1.2 "Early stopping is off" is only half true

`Trainer.on_train_end` calls `self.early_stopping.load_best_model()`
**unconditionally**, before the test evaluation. `early_stop_patience =
1_000_000` disables the *stop*, not the *selection*. Every test number you
have reported comes from the argmin-eval-loss checkpoint, not the final one.

That is defensible, but it must be stated that way in the methods section, and
it changes the 200k-step question: "86 % of runs peak in the final 10 %" is a
statement about which checkpoint was *selected*, and selection over ~200 eval
points at σ≈0.009 has a real optimistic bias. If you keep it, report both the
selected and the final-step numbers for the headline table.

### 1.3 Two latent config bugs worth ten minutes

- **`restrict_positive` is unreachable and uncached.** It is read by
  `prepare_sqoop` via `_cfg_get(cfg, 'restrict_positive', False)`, but it is
  not a field on `SqoopDataConfig`, so under hydra's struct mode
  `dataset.restrict_positive=true` raises. Worse, it is **not part of
  `dataset.dir`** — so if you did add the field, toggling it would silently
  reuse the existing cached dataset and you would "measure" the leak fix
  against unchanged data. Add the field *and* add it to the `dir` template in
  the same commit.
- `dataset.dir` also omits `eval_split`/`test_split` — harmless (they only
  select files) but worth a comment so nobody adds a data-affecting field
  without extending `dir`.

---

## 2. Four measurements run here

Datasets generated with your own generator, unmodified:
`rhs18` (648 pairs, 72 reps/pair, 46 656 ex) and `rhs35` (1 260 pairs, 32
reps/pair, 40 320 ex), plus a "mini-SQOOP" (8 glyphs, 4 objects, otherwise
identical semantics). Positive rate 0.5000 in all of them, mean 8.1 rejection
attempts per example — your generator reproduces cleanly.

### 2.1 [measured] The visual pathway is ~9 % of the token, but that does not discriminate

At initialisation, under `broadcast_cat` on real SQOOP images
(4.1 % of pixels are non-black, 39 % of the 64 patches contain any ink):

```
visual block (96 dims)  rms 0.063     across-patch std 0.044
question block (32 dims) rms 0.639    across-patch std 0.000  (constant by construction)
pos-enc                  rms 0.028
after LayerNorm:        visual rms 0.226, question rms 1.961
```

LayerNorm cannot fix this — it normalises the concatenated vector, so it
preserves the 10:1 ratio between the two sub-blocks. The consequence at the
output:

| model | sd(logit gap ∣ image varies) | sd(logit gap ∣ question varies) | image share |
|---|---|---|---|
| transformer / broadcast_cat | 0.0036 | 0.196 | **1.8 %** |
| transformer / film | 0.0036 | 0.251 | 1.4 % |
| transformer / token | 0.027 | 0.007 | 79 % |
| transformer / flatten | 0.010 | 0.216 | 4.4 % |
| **syncnet** | 0.0008 | 0.274 | **0.3 %** |
| conv_lstm | 0.0051 | 0.0019 | 73 % |

**The syncnet is more question-dominated than the transformer and it learns.**
So "the image signal is drowned at init" is *not* sufficient, and I am
reporting it as a hypothesis killed rather than confirmed. (It is still worth
one free rider — a `GroupNorm` after `PatchifyEncoder` in `VQATransformer`, three
lines, matching what the syncnet already does — but do not spend a night on it.)

What the table *does* say is that the fastest available descent direction at
step 0 is **suppressing the question**, because the question drives ~0.2 of
logit-gap variance and carries exactly zero information (labels are balanced
inside every (pair, rel) cell by construction). Killing that noise buys ~0.01
nats. Your observed pin — 0.693143–0.693148, i.e. logit gaps of order 0.006 —
is what a model looks like *after* it has done that and found nothing to
replace it with.

### 2.2 [measured] The image-only leak is small; conv_lstm's 0.999 is not explained by it

The generator docstring flags an unmeasured asymmetry (negatives are
rejection-sampled under three simultaneous positional constraints, positives
under one). I fit a logistic probe on image-only features, per relation, with a
70/30 held-out split:

| features | held-out accuracy (mean over relations) |
|---|---|
| 8 global layout moments (ink centroid/spread/skew per axis, total ink) | 0.520 |
| 8×8 coarse ink map (64 features) | 0.518 |
| both | 0.514 |

Ink mass differs by ≤0.05 sd between classes. So the layout bias is real but
worth about a point, not the 0.999. **[measured]** That is a good sentence for
your limitations paragraph and it removes a worry rather than creating one.

The duplicate-shape leak (0.575) remains the one that matters, and it is the
reason hypothesis H5 below is the cheapest way to make the anomaly disappear.

### 2.3 [measured] The chance plateau is universal and long

On the full 36-glyph rhs=18 data at 46 656 examples, bs 64, lr 3e-4 (your
schedule), on CPU:

```
conv_lstm  step 1500: train_ce 0.69322  acc 0.490
syncnet    step 3500: train_ce 0.69360  acc 0.505
```

Both of the models that *do* solve SQOOP are pinned at ln2 at 3 500 steps.
This confirms, quantitatively, the warning already in your handover: **any
probe shorter than the escape time measures budget, not mechanism.** It also
means the "readout cls/mean/flatten all flat at 1 500 CPU steps" evidence
should be struck from the ruled-out list — it has no discriminating power.

### 2.4 [measured] mini-SQOOP reproduces the pin in 35 minutes on 2 CPU cores

"mini-SQOOP" keeps your generator's semantics exactly — hard negatives, exact
per-cell label balance, position-only labels, same image size and glyph sizes —
and changes only the vocabulary: **8 glyphs instead of 36, 4 objects instead of
5**, 39 936 examples at rhs=3. Glyph recognition becomes trivial; the
relational structure and the exactly-flat plateau are untouched.

Same optimiser, same clip, same warmup_cosine, bs 64, lr 3e-4:

| step | conv_lstm train CE / acc | VQATransformer train CE / acc |
|---|---|---|
| 250 | 0.69347 / 0.505 | 0.69752 / 0.494 |
| 1 500 | 0.69238 / 0.519 | 0.69336 / 0.504 |
| 1 750 | **0.69086 / 0.528** ← escapes | 0.69328 / 0.508 |
| 3 000 | 0.66728 / 0.587 | 0.69349 / 0.500 |
| 4 500 | 0.57904 / 0.698 | 0.69320 / 0.500 |
| 6 000 | **0.54395 / 0.733** | **0.69315 / 0.505** |

0.69315 is ln2 to five decimals. **The full-scale phenomenon reproduces
exactly** at 1/27 the data and 1/16 the steps, with a working positive control
in the same harness, for 35 minutes of CPU.

Three things follow.

1. **It is not glyph recognition.** With 8 easily-separable glyphs and 4
   objects the transformer is still pinned. Combined with §1.1 (the syncnet
   solves the 36-glyph version through the same patchify encoder) the
   perception story is dead twice over.
2. **You now have a 35-minute local reproduction.** Every future transformer
   hypothesis can be screened on CPU before it costs a GPU night, with
   conv_lstm's escape at ~1 700 steps as the built-in positive control that the
   earlier 1 500-step probes lacked. The scripts are in the bundle
   (`probe_gen_mini.py`, `probe_train.py`); `--arm` covers conv_lstm, syncnet
   and six transformer variants.
3. **The caveat that keeps H2 alive.** mini-SQOOP at rhs=3 is a *low-variety*
   split (24 pairs), so under H2 it is expected to fail. The matching
   high-variety run (`mini_rhs7`, all 56 pairs, the mini IID case) is the one
   that separates H1 from H2 locally, and it costs 20 minutes rather than a
   dataset build plus a GPU hour. Run that before the GPU battery.

`readout=flatten` was also pinned at ln2 in this harness (1.88M params,
0.69315 by step 750), which independently corroborates the GPU encoder-sweep
cell whose provenance you were unsure of.

---

## 3. Hypotheses, ranked, with the evidence that separates them

Everything below is consistent with all eight of your eliminations *and* with
the rhs=35 exception, which is the filter I applied.

### H1 — Escape-time × cosine schedule (highest prior)

SQOOP has an **exactly flat** chance plateau: generator deviation 8 makes the
question-only Bayes optimum exactly 0.5000 in every (pair, rel) cell, so a
question-only model has zero gradient, not a small one. Every model starts
there (§2.3). Escape requires the image pathway to break the symmetry.
conv_lstm and syncnet escape early because their readouts hand them a
first-order signal (position-indexed flatten; normalised content-addressed
attention with a question-derived query). The transformer must discover
selection through un-normalised dot-product attention whose across-patch
variation is ~2 % of the token norm. If its escape time exceeds ~30–40k steps,
`warmup_cosine` has already annealed the LR away and it can never escape —
**and the runs that "peak early" would be exactly the stuck ones, which is what
you observe.**

*Separating evidence, free:* the **step at which train CE departs ln2** for
(a) conv_lstm rhs=18, (b) syncnet rhs=18, (c) transformer rhs=35. If (a) and
(b) are at 5–20k and (c) is late (>40k), H1 is strongly supported.

*Experiment, 1 run:* transformer, rhs=18, `lr_scheduler=constant`, lr 1e-3,
`n_steps=300_000`, everything else identical. ~45 min. Constant LR is the point
— cosine is the thing under test.

### H2 — Variety-controlled symmetry breaking

Escape probability per step scales with the number of distinct (pair, rel)
cells contributing independent gradient directions: 144 at rhs=1, 2 592 at
rhs=18, 5 040 at rhs=35. Under H2 the transition is continuous in rhs and
rhs=35 is simply past the knee.

*Separating experiment:* **rhs = 34** (see §0). Continuous → H2. Only-35 →
artefact of the IID path, and there is nothing to explain.

### H3 — The exact per-cell label balance removed the ramp

This one is a genuine thesis paragraph if it lands. Your deviation 8 replaced a
globally-alternating labelling (which left each cell Binomially skewed to ~0.59)
with exact per-cell balance. The skew was a leak and removing it was correct for
the question-only control — but a ~0.59 skew is also a **smooth,
question-only, first-order descent direction**, i.e. a ramp off the plateau.
A model that follows it leaves the degenerate regime with non-trivial
representations and can then keep going; a model on an exactly flat plateau
cannot. Models with spatial inductive bias never needed the ramp; a bag-of-
patches transformer might.

*Separating experiment:* build one rhs=18 dataset with a deliberate mild
per-cell skew (0.55) and run the transformer. Escape → H3 confirmed, and the
finding is "exact label balance, which the control requires, is what makes the
landscape unnavigable for the attention baseline" — a much better sentence than
"the transformer did not train". Cost: one build + one run.

### H4 — Precision (low prior, nearly free)

Every transformer run in your table is bf16; the fp32 arms of `transformer_lr`
never executed. rhs=35 was also bf16, which argues against H4 — but the
untested interaction is bf16 × low-variety data, where the attention logits are
a large constant plus a ~2 % patch-dependent term. Run it as a rider, not as a
night.

### H5 — There is no exception to explain (cheapest way to close everything)

Read 0.672/0.673 at rhs=35 as the **leak ceiling**, not as learning:
duplicate-shape detection alone is worth 0.575–0.58, plus ~0.01–0.02 of layout
(§2.2), plus whatever glyph-count statistics a transformer picks up. Under H5
the transformer never optimises the relation on SQOOP at any rhs, and rhs=35
only differs in how much non-relational signal happens to be reachable.

*Separating experiment:* rerun rhs=35 with `restrict_positive=True` (after
fixing §1.3). If 0.673 → ~0.52, the anomaly evaporates and your SQOOP chapter
is clean: three defensible negatives, no unexplained exception. **This is the
highest-value-per-GPU-minute experiment in the list, because a confirmed H5
lets you stop.**

### H6 — Provenance (free, do it regardless)

`model.encoder_name` unlogged; 4 of 8 `transformer_lr` cells missing;
`model.readout` likely in the same state. Log every model dataclass field into
`wandb.config` explicitly and add one assertion in `tests_migration` that the
sweeper's cartesian product matches the number of finished runs. Ten minutes,
and it retroactively secures 34 runs' worth of claims.

### Demoted

- *CLS has no question-dependent query at layer 1 under `broadcast_cat`* —
  true (the question block is identical across all patch keys, so it adds a
  constant to every attention logit and cancels in the softmax; the CLS
  parameter itself never receives `q`), but `token_seq` gives CLS a question
  token to attend to and was also flat. Interesting for the write-up, not a
  cause.
- *Pooled readout cannot represent SQOOP* — refuted by the syncnet (§1.1).
- *Visual signal drowned at init* — refuted by §2.1.

---

## 3b. Screen on CPU first — it is free

Before the GPU battery, run these three on any spare CPU box. Each is ~20-35
minutes and each can kill a hypothesis:

```bash
python probe_gen_mini.py --rhs 7 --n 40000 --out probe/mini_rhs7.npz   # 90 s
python probe_train.py --arm tf_bcat_cls --data probe/mini_rhs7.npz --steps 6000
python probe_train.py --arm tf_bcat_cls --data probe/mini_rhs3.npz --steps 20000 \
       --lr 1e-3 --warmup 200            # long, high, constant-ish LR: H1
python probe_train.py --arm syncnet --data probe/mini_rhs3.npz --steps 6000
```

- mini rhs=7 (all 56 pairs) escapes but rhs=3 does not → **H2**, and the GPU
  rhs=34 run is confirmatory rather than exploratory.
- 20k steps at lr 1e-3 escapes where 6k at 3e-4 did not → **H1**, and the
  constant-LR block is the one that matters.
- neither → the plateau is structural for this architecture on this task, and
  the stopping rule below should be applied early rather than late.

## 4. The one-night battery, with a stopping rule written in advance

Two cards, one run per card. Everything at rhs=18 unless stated.

| # | Run | Tests | Cost |
|---|---|---|---|
| 0 | Pull `train_loss/cross_entropy` histories (no GPU) | H1 escape step | free |
| 1 | transformer, `lr_scheduler=constant`, lr 1e-3, 300k steps | H1 | ~45 min |
| 2 | transformer, rhs=34 | H2 | 40 min build + 15 min |
| 3 | transformer, rhs=35, `restrict_positive=True` | H5 | 40 min build + 15 min |
| 4 | transformer, fp32, 100k | H4 | 15 min |
| 5 | transformer + `GroupNorm(8, patch_emb_dim)` after the encoder | §2.1 rider | 15 min |

Total ≈ 3.5 h wall on two cards including builds. Add the **chance abort** you
already designed (`stop if train CE > 0.69 at step 20 000`) to every cell
except #1 — it makes the whole battery cheaper and it is zero-risk here,
because escape after 20k with a decaying LR is exactly what #1 exists to test.

**Stopping rule, committed before the runs:** if none of #1, #2, #3 leaves
0.6900 by step 40 000, the transformer question is closed. Write it as
*"the attention baseline does not optimise on SQOOP under any configuration we
tried; the rhs=35 point is consistent with the duplicate-shape leak ceiling
rather than with learning; the conv_lstm establishes that the task and the data
are fine."* That is a perfectly publishable negative and it costs you one
paragraph, not a chapter.

---

## 5. The wandb pulls to make, precisely

One CSV per run of per-step history, keys:

```
_step
train_loss/cross_entropy
train_callbacks/accuracy
train_optim/lr
train_optim/grad_norm
eval_loss/cross_entropy
eval_callbacks/accuracy
```

Runs (12 total):

1. `iid_control` transformer rhs=35 — **the most important single history in the project**
2. conv_lstm rhs=18, all 3 seeds
3. syncnet rhs=18, all 3 seeds
4. two transformer rhs=18 cells from `transformer_scale` (hidden 128 L4, hidden 384 L8)
5. the four `transformer_lr` cells that did run
6. one SOC transformer at patch 5 (for the 200k-truncation question)

What to read off each:

- **first step where `train_loss/cross_entropy` < 0.690** — the escape step.
  This is the number the whole H1/H2 argument turns on.
- `train_optim/grad_norm` on the plateau: if it is pinned at ≈1.0 the clip is
  binding and the effective LR is not what you think; if it is ≪1 the plateau
  is genuinely flat. Either way it is a fact worth a figure.
- for the rhs=35 run, whether CE falls smoothly from ~5k or steps down late.
  Smooth → real learning; a late step-down under a dying cosine → it barely
  escaped, which supports H1 hard.

---

## 6. Budget advice, since the schedule is writing-limited

Nothing in this failure threatens any of the three claims. The thesis question
is whether phase synchrony is load-bearing; the attention *baseline* failing to
optimise on one of three tasks is a limitations paragraph. Meanwhile the SOC
gate null is finished, 5 seeds, properly powered, and coalitions — the only
remaining route to a positive result — has **zero runs** with four weeks left
and a benchmarked cost of ~13.7 h for the full suggested pass.

Recommended allocation: **one night on the battery in §4, then stop
regardless of outcome, and give the rest to coalitions.** If §4 finds the
cause, you gain a good paragraph; if it does not, you gain a clean negative and
lose nothing. Either way the marginal thesis value of a third night on the
transformer is lower than the marginal value of the first night on the
topology-change experiment (`osc_dim {2,3,4} × 3 seeds`, ~3.3 h), which is the
only experiment in the project that can come out *positive*.
