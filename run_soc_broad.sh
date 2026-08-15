#!/usr/bin/env bash
set -uo pipefail

# Sort-of-CLEVR broad pass -- 50 runs, ~13.0 h serial. Matched to the SQOOP
# broad pass: train_size 36,000 scenes x 3 subtypes x 10 questions =
# 1,080,000 examples, bs 256 x 100k steps = 23.7 epochs. Same numbers SQOOP
# runs at, so per-arm results are comparable across the two tasks.
#
# The sweeps are independent; to use both GPUs split the list across two
# shells with CUDA_VISIBLE_DEVICES. Keep question_only first in whichever
# shell runs it -- the dataset is built on the first cache miss.
#
# ALWAYS report accuracy as a delta above the prior-optimal floors
# (overall 0.506, non-rel 0.514, binary 0.459, ternary 0.545). Raw
# accuracy on this task is not interpretable on its own.
#
# Runtime note: at 75px, patch 5 gives 15x15 = 225 tokens and patch 15
# gives 5x5 = 25 -- measured 3.9x apart, with no divisor between (75 =
# 3x5x5). Resolution is swept deliberately in the two *conditioning*
# configs and pinned to 15 elsewhere, so capacity and depth sweeps are not
# silently 4x the cost of everything else.

# ------------------------------------------------------- 1. FLOOR   2 runs
# Question-only. Not a baseline -- it measures how much of the label the
# question alone carries. Compare against the prior-optimal floors above.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/question_only

# ---------------------------------------- 2. TRANSFORMER CAPACITY   6 runs
# hidden_dim x n_layers at fixed patch 15. A 10.8M transformer previously
# scored WORSE than a 3.3M one, so this is a control on whether the ceiling
# is capacity or conditioning.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/transformer_capacity

# ------------------------------------------ 3. TRANSFORMER DEPTH   6 runs
# share_layer_weights x n_layers: stacked vs looped.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/transformer_depth

# ----------------------------------- 4. TRANSFORMER CONDITIONING   6 runs
# q_conditioning x patch_size. The dominant axis: late `token` scored
# 0.784 / 0.547 ternary against per-token `broadcast_cat` at 0.949 / 0.865
# at matched ~3.3M params. Half these cells run at patch 5 (~59 min each),
# so this config alone is ~3.3 h.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/transformer_conditioning

# --------------------------------------- 5. SYNCNET CONDITIONING   6 runs
# The same axis on the syncnet, which always had FiLM before; this is what
# makes the two families comparable. Also ~3.3 h for the patch-5 cells.
python main.py task=sort_of_clevr experiment=sort_of_clevr/syncnet/conditioning

# -------------------------------------- 6. SYNCNET COMMUNICATION  10 runs
# partition x gate_mode -- optional vs necessary communication. Note SOC
# quadrants give UNIFORM relevance, so phase is expected to act as a switch
# rather than a router here; SQOOP quadrants are the sparse-relevance case.
python main.py task=sort_of_clevr experiment=sort_of_clevr/syncnet/communication

# ------------------------------------------- 7. SYNCNET DYNAMICS   8 runs
# omega_init x learn_omega -- does the trained model sit near cluster sync?
python main.py task=sort_of_clevr experiment=sort_of_clevr/syncnet/dynamics

# ------------------------------------------- 8. SYNCNET CAPACITY   6 runs
# module_dim x n_modules. Control: if accuracy tracks parameters rather
# than module structure, the coalition story is doing no work.
python main.py task=sort_of_clevr experiment=sort_of_clevr/syncnet/capacity
