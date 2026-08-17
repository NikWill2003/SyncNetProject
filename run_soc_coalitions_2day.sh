#!/usr/bin/env bash
set -uo pipefail

# Sort-of-CLEVR (steps 1-2, ~9 h) then coalitions (steps 3-9, ~39 h), on
# one card. Pair with run_sqoop_2day.sh on the other; that one owns every
# SQOOP dataset and this one owns the SOC and coalitions data, so the two
# cards cannot race a build (build_dataloaders is not concurrency-safe --
# two cold jobs on the same dir each build their own copy).
#
# WHY COALITIONS IS ON THE SORT-OF-CLEVR CARD
# Sort-of-CLEVR has ~9 h of work left worth doing: the gate null is
# finished at 5 seeds and properly powered, and the only open questions
# are whether it is a truncation artefact and whether the 0.867
# transformer survives more than one seed. That leaves ~39 h idle, and
# coalitions has ZERO runs -- it is the only task with temporal, sparse
# relevance and therefore the only one that can support the
# topology-change claim POSITIVELY. Delete steps 3-9 if you want this
# card to stay Sort-of-CLEVR only, but then coalitions never runs.
#
# COALITIONS FLOORS -- overall accuracy is dominated by free steps (61%
# of steps are a free copy in iid mode). ALWAYS report deltas, never raw:
#   constant 0.233 | copy-own-token 0.757 | NO-COMM ORACLE 0.786
# A model at or below 0.786 has NOT demonstrated communication.
#
# -m appears on lines that override a seed list from the command line. A
# CLI sweep replaces that key in the yaml's hydra.sweeper.params and is
# crossed with the remaining axes; without -m hydra rejects the comma
# list as ambiguous.

# --------------------------------------- 1. IS THE GATE NULL REAL?
# 10 runs, ~4.2 h. First, because it is the one result the thesis rests
# on. 86% of Sort-of-CLEVR runs peak in the final 10% of budget and the
# median best_step of the runs that learned is 98% of budget, while
# warmup_cosine anneals to ~6e-9 at n_steps -- so "still improving" and
# "the schedule ran out" are indistinguishable at 100k. This doubles the
# budget on the two arms the null turns on, at 5 seeds.
#   reproduces  -> the null is real, cite both budgets, done
#   separates   -> the null is underpowered in TIME, not seeds, and the
#                  whole SOC table needs rerunning before anything is
#                  written
python main.py experiment=sort_of_clevr/syncnet/gate_null_200k

# ------------------------------------------- 2. SOC TRANSFORMER
# 5 runs, ~5 h. ternary 0.867 (+0.322 over the 0.545 floor) is the
# strongest baseline number in the thesis and rests on ONE seed. Patch
# size dominates this arm -- 0.867 at patch 5 vs 0.573 at patch 15 -- so
# both are pinned here rather than swept.
python main.py -m experiment=sort_of_clevr/baselines/transformer_conditioning \
  model.q_conditioning=film model.patch_size=5 train.seed=0,1,2,3,4

# ===================== COALITIONS -- nothing has ever run here ========
# The task was blocked by a signature mismatch (CoalitionsBase.forward
# took (streams, commands, ...) while the Trainer calls model(batch)),
# fixed by adding forward(batch) delegating to forward_seq. That fix is
# validated in a notebook, not through the Trainer -- hence step 3.
#
# DATASET DIR HAZARD: conf/dataset/coalitions/default.yaml sets
#   dir: coalitions_${dataset.family}
# so the cache key encodes ONLY the family. Unlike sqoop, changing
# T_train/T_test/train_size/K/stream_mode/command_mode does NOT change
# the directory and the generator silently reuses old data.
# coalitions/length_gen.yaml sets dataset.dir explicitly for this reason.

# ------------------------------------------------------ 3. SMOKE
# 7 runs x 2k steps, ~15 min, and it builds the N=4 dataset. STOP AND
# LOOK AT THIS before trusting anything below: 7/7 finished, losses
# finite and moving, acc_joint / gate_auc / offpair_leak all present,
# oracle above mlp, no_comm at or below 0.786. Fifteen minutes here is
# the difference between results and 40 identical stack traces at 03:00.
# The first line runs one gate alone so the cold build happens once.
python main.py -m task=coalitions experiment=coalitions/smoke model.gate=oracle train.seed=0
python main.py task=coalitions experiment=coalitions/smoke

# -------------------------------------------- 4. GATE COMPARISON
# 35 runs, ~11.4 h. Seven gates, one axis, backbone fixed,
# message_proj=shared so the gate is the sole routing bottleneck. The
# claim is NOT "phase wins": it is that phase reaches the oracle's
# routing quality with M-1 dof where attention needs O(M^2), and that
# phase degrades on FRUSTRATED graphs where attention does not. Read
# per-graph, as deltas above 0.786. Never the mean.
python main.py task=coalitions experiment=coalitions/gate_null

# ------------------------------------- 5. IS THE GATE THE BOTTLENECK?
# 12 runs, ~4.2 h. The control that makes step 4 interpretable, and the
# exact lesson from Sort-of-CLEVR, where the readout's covert
# cross-module integration channel had to be removed (readout_mode=sum)
# before any gate comparison meant anything. message_proj=per_pair is
# the same hazard. If shared and per_pair are within noise, the gate was
# never the bottleneck and step 4 is not a result about routing -- which
# has to be said, not buried. Read this BEFORE writing up step 4.
python main.py task=coalitions experiment=coalitions/routing_ablation

# --------------------------------------- 6. TOPOLOGY CHANGE, N=4
# 15 runs, ~5.5 h. THE ONLY EXPERIMENT IN THE PROJECT THAT CAN COME OUT
# POSITIVE. A scalar phase gate (osc_dim=2) can only express coalitions
# realisable on the circle, so a frustrated graph should be unreachable
# and raising the dimension should repair it. The rho ladder makes this
# falsifiable in advance, which is rare and worth saying so:
#     STAR_A       rho(d=2) = 0.408   rho(d=3) = 0.021
#     every other  0.003 - 0.015 at BOTH dimensions
# So osc_dim=2 should fail ON STARS AND ONLY ON STARS, and osc_dim=3
# should repair it while doing nothing elsewhere; osc_dim=4 is the
# did-it-keep-helping control. Any of these kills the claim and should
# be reported plainly: osc_dim=2 fails on non-stars too, or osc_dim=3
# helps uniformly (both = capacity, not topology), or osc_dim=2 already
# matches oracle on stars (no frustration to relieve).
python main.py task=coalitions experiment=coalitions/osc_dim

# ------------------------------------------ 7. N=6 FRUSTRATION LADDER
# 1 build (~1.6x N=4) + 18 runs, ~10.8 h. N=4 has ONE frustrated graph so
# the claim rests on a single point. N=6 has FIVE (STAR3/4/5,
# BIPARTITE_2_3, BIPARTITE_3_3) and gives a GRADED ladder -- the
# difference between a suggestive figure and a dose-response curve. The
# comparison that matters is the INTERACTION: (phase - attention) as a
# function of rho(d=2) across graphs. Flat = null, sloped = the result.
# Runs before step 8 because it establishes the per-graph floors and
# ceilings that step 8 is read against.
python main.py -m task=coalitions experiment=coalitions/n6_ladder \
  model.gate=oracle train.seed=0 train.n_steps=2000
python main.py task=coalitions experiment=coalitions/n6_ladder

# --------------------------------------- 8. N=6 DIMENSION LADDER
# 12 runs, ~7 h. The sharpest form of the question: does the oscillator
# dimension at which a graph becomes learnable track the dimension at
# which its frustration rho collapses? The N=6 npz caches rho at dims
# [2,3,4,6], so the prediction is on disk before the runs. Expected
# signature: a STAIRCASE per graph, not a slope. If the two agree you
# predicted a learning threshold from a geometric quantity computed
# before training, which is the strongest sentence this thesis can make.
python main.py task=coalitions experiment=coalitions/n6_osc_dim

# ------------------------------------------------------- 9. TAIL
# ~10.6 h, drop if you are short. Capacity control (is the gate ordering
# just parameter count), length generalisation (T 32 -> 64, the
# routing/content decoupling claim in testable form), and a real
# Kuramoto phase update instead of the learned MLP one.
python main.py task=coalitions experiment=coalitions/capacity
python main.py -m task=coalitions experiment=coalitions/length_gen \
  model.gate=oracle train.seed=0 train.n_steps=2000
python main.py task=coalitions experiment=coalitions/length_gen
python main.py -m task=coalitions experiment=coalitions/gate_null \
  model.gate=phase model.phase_update=kuramoto train.seed=0,1,2
