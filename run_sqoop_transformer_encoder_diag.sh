#!/usr/bin/env bash
set -uo pipefail
# TRANSFORMER ENCODER DIAGNOSTIC -- 8 runs, ~2.2 h, rhs=18 (data exists).
#
# encoder {patchify, cnn} x q_conditioning {broadcast_cat, film}
#                         x readout {cls, flatten}
#
# Read train_loss/cross_entropy. ln2 = 0.69315.
#   any cnn cell below ~0.6  -> the encoder is the cause, and the syncnet
#                               uses patchify too, so every SQOOP
#                               syncnet-vs-conv_lstm number is confounded
#   all 8 flat               -> stop; report the transformer as failing to
#                               optimise on SQOOP and let conv_lstm carry
#                               the baseline

mkdir -p logs/sqoop_transformer_encoder_diag
python main.py task=sqoop experiment=sqoop/diagnostics/transformer_encoder \
  2>&1 | tee logs/sqoop_transformer_encoder_diag/transformer_encoder.log

echo
echo "failures:"
grep -lEi "Traceback|Error executing job" logs/sqoop_transformer_encoder_diag/*.log || echo "  none"
