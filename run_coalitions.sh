#!/usr/bin/env bash
set -uo pipefail

# ===================== COALITIONS =====================================
# Was blocked by a signature mismatch: CoalitionsBase.forward took
# (streams, commands, ...) while the Trainer calls model(batch). Fixed by
# adding forward(batch) delegating to forward_seq, with
# callbacks/metrics.py calling forward_seq directly. Every step below has
# been run end to end through the Trainer at 4 steps per cell -- all 7
# gates, osc_dim 2/3/4, kuramoto, per_pair, N=6, and the four new data
# axes (stream_mode, command_mode, readout_mode, K).
#
# ~108 h queued, ORDERED BY VALUE, not by cost. One card gets to roughly
# step 7 in two days. Steps 8+ are the next block, or the other card once
# SQOOP is done. Stop anywhere -- nothing below reads anything above.
#
# DATASET BUILDS ARE CHEAP HERE: 25 s for the default 20k/2k/2k at T=32 on
# two CPU cores, so a step that changes a data field costs nothing to
# prepare. (SQOOP's are 40 min; do not carry that intuition over.) The dir
# template now encodes every data field, so each variant gets its own
# directory and build_dataloaders builds it on first use.
#
# ALWAYS REPORT AGAINST THE FLOORS. Overall accuracy is dominated by free
# steps -- 61% of steps are a free copy in iid mode:
#   constant 0.233 | copy-own-token 0.757 | NO-COMM ORACLE 0.786
# A model at or below 0.786 has NOT demonstrated communication.
#
# READ PER-GRAPH, NOT THE MEAN. The callback emits acc_joint__STAR_A,
# acc_joint__FULL and so on per graph, plus gate_auc, gate_sep,
# offpair_leak and rho_err_corr. The topology prediction is an INTERACTION
# between gate and graph; the mean hides it completely.

# ---------------------------------------------------------- 1. SMOKE
# 7 runs x 2k steps, ~15 min, and it builds the N=4 dataset. STOP AND LOOK
# AT THIS: 7/7 finished, losses finite and moving, acc_joint / gate_auc /
# offpair_leak all present, oracle above mlp, no_comm at or below 0.786.
# The first line runs one gate alone so the cold build happens once.
python main.py -m task=coalitions experiment=coalitions/smoke model.gate=oracle train.seed=0
python main.py task=coalitions experiment=coalitions/smoke

# ------------------------------------------------ 2. GATE COMPARISON
# 35 runs, ~11.4 h. Seven gates, one axis, backbone fixed,
# message_proj=shared so the gate is the sole routing bottleneck. The claim
# is NOT "phase wins": it is that phase reaches the oracle's routing
# quality with M-1 dof where attention needs O(M^2), and that phase
# degrades on FRUSTRATED graphs where attention does not.
python main.py task=coalitions experiment=coalitions/gate_null

# --------------------------------------- 3. IS THE GATE THE BOTTLENECK?
# 12 runs, ~4.2 h. The control that makes step 2 interpretable, and the
# exact lesson from Sort-of-CLEVR, where the readout's covert cross-module
# integration channel had to be removed (readout_mode=sum) before any gate
# comparison meant anything. message_proj=per_pair is the same hazard. If
# shared and per_pair are within noise, the gate was never the bottleneck
# and step 2 is not a result about routing. Read this BEFORE writing up
# step 2.
python main.py task=coalitions experiment=coalitions/routing_ablation

# ----------------------------------------- 4. TOPOLOGY CHANGE, N=4
# 15 runs, ~5.5 h. THE ONLY EXPERIMENT IN THE PROJECT THAT CAN COME OUT
# POSITIVE. A scalar phase gate (osc_dim=2) can only express coalitions
# realisable on the circle, so a frustrated graph should be unreachable and
# raising the dimension should repair it. The rho ladder makes this
# falsifiable in advance:
#     STAR_A       rho(d=2) = 0.408   rho(d=3) = 0.021
#     every other  0.003 - 0.015 at BOTH dimensions
# So osc_dim=2 should fail ON STARS AND ONLY ON STARS, and osc_dim=3 should
# repair it while doing nothing elsewhere; osc_dim=4 is the
# did-it-keep-helping control. Any of these kills the claim and should be
# said plainly: osc_dim=2 fails on non-stars too, or osc_dim=3 helps
# uniformly (both = capacity, not topology), or osc_dim=2 already matches
# oracle on stars (no frustration to relieve).
python main.py task=coalitions experiment=coalitions/osc_dim

# ------------------------------------- 5. THE FRUSTRATED GRAPHS ALONE
# 24 runs, ~8.7 h. family=all dilutes the frustrated graphs with ones a
# scalar phase has no trouble on, and the per-graph metrics for the
# frustrated subset are then computed on whatever share of the batch
# happened to be that graph. Training on family=frustrated asks the sharp
# version: given a distribution made entirely of the cases the mechanism is
# supposed to fail on, does osc_dim=2 fail? It also removes the "did not
# bother" reading of a family=all null -- a model can only spend capacity
# on frustration if it sees it.
python main.py task=coalitions experiment=coalitions/frustrated_family

# ------------------------------------------------------ 6. DYNAMICS
# 18 runs, ~6.5 h. THIS ONE ATTACKS OUR OWN CLAIM AND SHOULD RUN EARLY.
# On Sort-of-CLEVR, omega_init 0.5 -> 10 swung phase_R 0.97 -> 0.55 while
# accuracy moved 0.844 -> 0.852: the oscillators synchronised, the degree
# of synchrony varied a lot, and the task did not care. That is the most
# damaging result in the project. Check it HERE, on the task where
# synchrony is supposed to be load-bearing, before believing any positive
# coalitions number. Flat accuracy with moving gate statistics is the null.
python main.py task=coalitions experiment=coalitions/dynamics

# --------------------------------------- 7. IS ROUTING EVEN NECESSARY?
# 24 runs + 1 build, ~8 h. The manipulation check for the benchmark
# itself. stream_mode=iid makes neighbour tokens unpredictable so a
# receiver must be told; stream_mode=rule makes them deterministic so it
# can predict instead. Under rule the gap to no_comm should COLLAPSE. If
# the gate ordering survives under rule, the gate comparison is not
# measuring routing and every number in step 2 means something else. Same
# logic that made the SQOOP question-only gate non-negotiable.
python main.py task=coalitions experiment=coalitions/stream_mode

# --------------------------------------- 8. N=6 FRUSTRATION LADDER
# 1 build + 18 runs, ~9.7 h (N=6 costs ~1.6x N=4). N=4 has ONE frustrated
# graph so the claim rests on a single point. N=6 has FIVE (STAR3/4/5,
# BIPARTITE_2_3, BIPARTITE_3_3) and gives a GRADED ladder -- the difference
# between a suggestive figure and a dose-response curve. The comparison
# that matters is the INTERACTION: (phase - attention) as a function of
# rho(d=2) across graphs. Flat = null, sloped = the result. Runs before
# step 9 because it establishes the per-graph floors and ceilings step 9 is
# read against.
python main.py -m task=coalitions experiment=coalitions/n6_ladder \
  model.gate=oracle train.seed=0 train.n_steps=2000
python main.py task=coalitions experiment=coalitions/n6_ladder

# ---------------------------------------- 9. N=6 DIMENSION LADDER
# 12 runs, ~7 h. The sharpest form of the question: does the oscillator
# dimension at which a graph becomes learnable track the dimension at which
# its frustration rho collapses? The N=6 npz caches rho at dims [2,3,4,6],
# so the prediction is on disk before the runs. Expected signature: a
# STAIRCASE per graph, not a slope. If the two agree you predicted a
# learning threshold from a geometric quantity computed before training,
# which is the strongest sentence this thesis can make.
python main.py task=coalitions experiment=coalitions/n6_osc_dim

# ------------------------------------------- 10. MESSAGE BOTTLENECK
# 24 runs, ~10 h. The M-1 vs O(M^2) argument in quantitative form. A claim
# about degrees of freedom should show up when the channel is tight and
# vanish when it is wide enough that inefficiency is free. msg_dim 4 -> 32
# with phase against attention; READ THE GAP AS A FUNCTION OF msg_dim. A
# constant gap is a capacity story. A gap that widens as msg_dim narrows is
# the efficiency story the thesis actually claims, and it is the experiment
# most likely to turn "phase is not worse" into "phase is better, here, for
# this reason".
python main.py task=coalitions experiment=coalitions/msg_bottleneck

# ------------------------------------ 11. COMMAND CHANNEL AND READOUT
# 24 runs + 3 builds, ~10 h. command_mode sparse = the coalition is
# announced once and must be HELD; dense = re-announced every step, so
# nothing has to be remembered. readout_mode instant = answer while the
# coalition is live; latch = answer up to readout_lag_max steps later, so
# the binding must survive the command going away.
# sparse+latch is the condition the thesis is really about: a coalition
# that persists without being continuously re-specified is the "partial
# commitment" story, and a phase gate has persistent state where an
# attention gate recomputes from scratch each step. If phase has an
# advantage anywhere it is most likely here, and it is currently untested.
python main.py task=coalitions experiment=coalitions/command_readout

# ------------------------------- 12. ROUTING/CONTENT ON THE CONTENT AXIS
# 24 runs + 1 build, ~8 h. K is how much there is to say. Widening it makes
# the content problem strictly harder while leaving the coalition structure
# identical. Claim two predicts gate quality (gate_auc, gate_sep,
# offpair_leak) FLAT in K while token accuracy falls. If gate quality
# degrades with K, routing and content are entangled here and claim two is
# not supported by this task. With step 13's length generalisation this is
# claim two on both axes -- content and time -- which is a far stronger
# paragraph than either alone. Report against the per-K chance level: the
# readout vocabulary is a function of (n_modules, K), so the head width
# changes with K.
python main.py task=coalitions experiment=coalitions/alphabet

# ----------------------------------------------------- 13. TAIL
# ~12 h, drop if you are short. Capacity control (is the gate ordering just
# parameter count -- the control that saved the SQOOP chapter from an
# over-claim), length generalisation (T 32 -> 64, claim two on the time
# axis), and a real Kuramoto phase update instead of the learned MLP one.
python main.py task=coalitions experiment=coalitions/capacity
python main.py task=coalitions experiment=coalitions/length_gen
python main.py -m task=coalitions experiment=coalitions/gate_null \
  model.gate=phase model.phase_update=kuramoto train.seed=0,1,2
