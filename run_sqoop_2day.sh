#!/usr/bin/env bash
set -uo pipefail

# SQOOP, ~39 h on one card. Pair with run_soc_coalitions_2day.sh on the
# other card; that one never touches a SQOOP dataset, so the two cannot
# race a build.
#
# Steps 1-5 are the DIAGNOSTIC: why the transformer sits at exactly ln2.
# Steps 6-11 are the CONSOLIDATION and do not depend on how 1-5 come out.
#
# STOPPING RULE, DECIDED NOW: if neither step 1 nor step 2 puts
# train_loss/cross_entropy below 0.6900 by step 40,000, the transformer
# question is closed -- report it as an arm that does not optimise on
# SQOOP and move on. Do not open a third night on it.
#
# READ train_loss/cross_entropy BEFORE ANY ACCURACY.
#   0.0000  = memorised, its eval numbers describe nothing
#   0.69315 = ln2, the model never left chance
#
# -m appears on lines that override a seed list from the command line. A
# CLI sweep replaces that key in the yaml's hydra.sweeper.params and is
# crossed with the remaining axes; without -m hydra rejects the comma
# list as ambiguous. Lines with no comma override do not need it.

# ------------------------------------------------- 1. ESCAPE TIME
# 6 runs, ~4.5 h. THE SCHEDULE IS UNDER TEST, NOT THE LR. Every previous
# sweep used warmup_cosine, which anneals to ~6e-9 by 100k; these hold
# the LR up for 300k. SQOOP has an exactly flat chance plateau (labels
# balanced inside every (pair, rel) cell, so a question-only model has
# ZERO gradient) and every model starts on it -- conv_lstm is still at
# 0.69322 at 1,500 steps and the syncnet at 0.69360 at 3,500 on data they
# later solve. If the transformer's escape time exceeds ~40k the LR is
# already gone. READ: first step where train CE < 0.6900.
python main.py task=sqoop experiment=sqoop/diagnostics/transformer_escape

# ---------------------------------------------- 2. VARIETY BISECT
# 3 builds (~40 min each) + 3 runs, ~2.9 h. iid_control.yaml carries no
# model: block, so the rhs=35 run used the sqoop/transformer defaults
# with identical optim and train blocks -- it already controls capacity,
# lr, warmup, precision, readout, conditioning, encoder, patch size,
# clipping, weight decay and step budget. So the cause is in the DATA.
# rhs=34 is the largest variety still on the ordinary code path; rhs=35
# alone trips `iid = not val_unseen_pairs`.
#   24/30/34 escape -> continuous in variety, bisect for the knee
#   only 35 escapes -> the exception is the IID path, nothing to explain
# question_only warms each cache in 2 min AND gives that rhs's leakage
# gate, which has to be reported anyway.
python main.py task=sqoop experiment=sqoop/baselines/question_only dataset.rhs_variety=24
python main.py task=sqoop experiment=sqoop/baselines/question_only dataset.rhs_variety=30
python main.py task=sqoop experiment=sqoop/baselines/question_only dataset.rhs_variety=34
python main.py task=sqoop experiment=sqoop/diagnostics/transformer_variety

# -------------------------------------------------- 3. PRECISION
# 4 runs, ~1 h. The fp32 cells of diagnostics/transformer_lr never
# executed -- 4 of 8 present, all bf16. Low prior, since rhs=35 was also
# bf16 and escaped, but it closes a hole in the record for an hour.
python main.py task=sqoop experiment=sqoop/diagnostics/transformer_precision

# ------------------------------------------ 4. THE ONE EXCEPTION
# 5 runs, ~1.3 h. rhs=35 transformer is currently n=1. If the anomaly is
# one lucky seed, better to know now than in the write-up.
python main.py -m task=sqoop experiment=sqoop/baselines/iid_control \
  model=sqoop/transformer train.seed=0,1,2,3,4

# ---------------------------------------------- 5. rhs=35 CONTROLS
# 6 runs, ~0.7 h. What is reachable at rhs=35 WITHOUT relations. The
# duplicate-shape leak scores 0.575 and an image-only linear probe adds
# ~0.01-0.02 on top. If the transformer's 0.672 sits at that ceiling it
# never learned the relation at any rhs and the exception disappears --
# the cheapest good outcome available.
python main.py -m task=sqoop experiment=sqoop/baselines/iid_control \
  model=sqoop/question_only train.seed=0,1,2
python main.py -m task=sqoop experiment=sqoop/baselines/iid_control \
  model=sqoop/conv_lstm train.seed=0,1,2

# ================= CONSOLIDATION -- runs regardless of steps 1-5 ======
# All 5 seeds. 1 seed is screening. The within-arm sd at test_size=1000
# is ~0.009, so re-estimate sigma from these rather than reusing 0.022,
# which came from the old 200-scene test split.

# ------------------------------------------------- 6. GATE NULL
# 25 runs, ~6.2 h. The Sort-of-CLEVR gate null (all five modes inside
# 0.024 ternary, phase vs frozen p=0.29, phase_R 0.885 vs 0.450) is a
# null about Sort-of-CLEVR. Replicating it here -- different images,
# different question format, systematic rather than IID split -- makes it
# a null about the MECHANISM. Needs no new dataset.
python main.py task=sqoop experiment=sqoop/syncnet/gate_null

# ------------------------------------------------- 7. RHS CURVE
# 50 runs, ~9.8 h. The headline comparison, currently 3 seeds.
python main.py -m task=sqoop experiment=sqoop/baselines/rhs_seeded train.seed=0,1,2,3,4
python main.py -m task=sqoop experiment=sqoop/syncnet/rhs_seeded  train.seed=0,1,2,3,4

# ----------------------------------------------- 8. HONEST FLOOR
# 18 runs, ~2.6 h. Our conv_lstm is NOT Bahdanau et al.'s Conv+LSTM:
# ours keeps a learned absolute position embedding and a flatten readout
# and reaches 0.999, theirs is spatially weak and stays above 14% error
# at every #rhs. Report both bars, and stop calling ours "the no-routing
# floor". readout=pool is EXPECTED to sit at exactly 0.500 / ln2 -- that
# is proof the task is position-bound, not a broken run.
python main.py task=sqoop experiment=sqoop/baselines/conv_lstm_weak

# ------------------------------------------ 9. IID CONTROL, MATCHED
# 5 runs, ~1.3 h. The published rhs=35 syncnet cell (0.513) ran at
# partition=none, not quadrant, so it is comparable to nothing.
python main.py -m task=sqoop experiment=sqoop/baselines/iid_control_quadrant \
  train.seed=0,1,2,3,4

# --------------------------------------------- 10. SYNCNET CAPACITY
# 20 runs, ~5 h. "The gap is architectural, not capacity" is load-bearing
# and rests on 1 seed per module_dim. At 3.2M the syncnet fits train to
# 0.9997 and generalises no better; that needs error bars.
python main.py -m task=sqoop experiment=sqoop/syncnet/capacity train.seed=0,1,2,3,4

# ------------------------------------------------------- 11. TAIL
# ~3.3 h, drop if you are short. The first is only meaningful if step 1
# or 2 found an escape. The second is the dynamics null from the other
# direction: omega_init 0.5 -> 10 swings phase_R 0.97 -> 0.55 while
# accuracy moves 0.844 -> 0.852.
python main.py -m task=sqoop experiment=sqoop/baselines/transformer_rhs train.seed=0,1,2,3,4
python main.py -m task=sqoop experiment=sqoop/syncnet/dynamics train.seed=0,1,2
