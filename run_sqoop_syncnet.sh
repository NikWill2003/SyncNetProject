#!/usr/bin/env bash
set -uo pipefail

# SQOOP syncnet sweeps. Do not run before run_sqoop_baselines.sh step 1
# (the question-only gate) has passed -- a leaking generator makes all of
# this noise, and the rhs curve is the single result most sensitive to a
# per-question class skew.

# ------------------------------------------------- 1. CONDITIONING
# 6 runs at rhs=18. Sets q_conditioning for everything below; edit the
# winner into communication.yaml, dynamics.yaml and rhs.yaml.
python main.py task=sqoop experiment=sqoop/syncnet/conditioning

# ------------------------------------------------ 2. COMMUNICATION
# 10 runs -- optional vs necessary communication x what controls the gate.
# SQOOP quadrants give sparse per-sample relevance, which Sort-of-CLEVR
# quadrants do not; this is the sharper test of selectivity.
python main.py task=sqoop experiment=sqoop/syncnet/communication

# ----------------------------------------------------- 3. DYNAMICS
# 8 runs -- does the trained model sit anywhere near cluster sync?
python main.py task=sqoop experiment=sqoop/syncnet/dynamics

# ---------------------------------------------------- 4. RHS CURVE
# 5 runs -- THE headline. Plot against transformer_rhs and conv_lstm.
# 1 seed: shakedown only. Nothing here is rankable until 5+ seeds.
python main.py task=sqoop experiment=sqoop/syncnet/rhs
