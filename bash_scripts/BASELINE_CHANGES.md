# What changed for each baseline that was weak on SQOOP, and why

The dataset is held constant throughout (per-cell balance, duplicate cue as generated).
Every change below is to the model or its optimisation only.

| model | what changed | why |
|---|---|---|
| **Transformer** | tokens come from the **shared 6-layer stem** (was: a single stride-8 patchify conv); the tuned row additionally uses **effective batch 2048** (`grad_accum=8`) | a stride-8 patch is ~half a glyph, so letter identity was unrecoverable from tokens; with the stem it plateaued at .667, and averaging over a larger batch recovered the rest (.667 -> 1.000 at rhs=35) — the gradient-agreement mechanism |
| **Shared Workspace** | same shared stem; tuned row at effective batch 2048 | same perceptual argument; the batch runs are the pending test (its stem-only result is .52) |
| **RelNet** | pair over an **8x8 grid** (was: 5x5 max-pool of the 16x16 stem) | a 12.8 px cell can hold two 10-15 px glyphs, so their relation is unrepresentable; 8 px cells give each glyph its own cell |
| **Conv+LSTM** | effective batch 2048; lr 1e-4; no weight decay (probes) | on the balanced data the only early signal is weak partial features; larger batch (SNR), smaller step, and no decay each stop that signal being lost. NB: the original trained on a per-question label skew that the corrected dataset removes, so it *should* score lower here |

Paper-form (patchify) token models are kept as `baselines/ablations/*_patchify` for the appendix.
