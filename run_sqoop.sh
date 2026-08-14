#!/usr/bin/env bash
set -uo pipefail

# SQOOP broad pass -- 57 runs, ~13.8 h serial on one card.
#
# The eight sweeps are independent: none reads another's winner, so to
# use both GPUs just split the list by hand across two shells with
# CUDA_VISIBLE_DEVICES. Keep the gate first in whichever shell runs it --
# the dataset is built on the first cache miss, and starting two cold
# jobs at once has both generate their own copy.
#
# Read the gate before anything else, and read train_loss/cross_entropy
# before any accuracy: 0.0000 means the run memorised the training set
# and its eval numbers describe nothing.

# ------------------------------------------------------ 1. GATE
# 1 run. Not a baseline -- a leakage test of the generator. Labels are
# balanced exactly 50/50 within every (x, rel, y) cell, so question-only
# accuracy is 0.500 by construction; above ~0.51 and nothing below is
# worth reading. Also builds the rhs=18 dataset (~40 min, 0.6 GB).
python main.py task=sqoop experiment=sqoop/baselines/question_only

# ----------------------------------------------------- 2. FLOOR
# 2 runs -- the no-routing floor from Bahdanau et al., at two encoder
# widths. Everything else is a delta above this.
python main.py task=sqoop experiment=sqoop/baselines/conv_lstm

# --------------------------------------------- 3. TRANSFORMER
# 16 runs -- q_conditioning x patch_size x depth. On Sort-of-CLEVR
# conditioning was worth ~32 points of ternary accuracy at matched
# capacity, so it likely dominates the other axes here too.
python main.py task=sqoop experiment=sqoop/baselines/transformer

# ------------------------------------- 4. SYNCNET CONDITIONING
# 6 runs -- the same axis on the syncnet.
python main.py task=sqoop experiment=sqoop/syncnet/conditioning

# ------------------------------------ 5. SYNCNET COMMUNICATION
# 10 runs -- optional (partition=none) vs necessary (quadrant)
# communication x what controls the gate. Last round every gate_mode
# landed within 0.015 of the others, frozen and open included; this
# separates "the partition does the work" from "all five memorised".
python main.py task=sqoop experiment=sqoop/syncnet/communication

# ----------------------------------------- 6. SYNCNET DYNAMICS
# 8 runs -- does the trained model sit anywhere near cluster sync?
python main.py task=sqoop experiment=sqoop/syncnet/dynamics

# ------------------------------------------ 7. SYNCNET READOUT
# 8 runs -- readout_mode x msg_agg. Never swept, and concat lets the head
# integrate across modules directly, which can stand in for messages.
python main.py task=sqoop experiment=sqoop/syncnet/readout

# ----------------------------------------- 8. SYNCNET CAPACITY
# 6 runs -- n_modules x module_dim. A control: if accuracy tracks total
# parameters rather than module structure, the coalition story is doing
# no work.
python main.py task=sqoop experiment=sqoop/syncnet/capacity