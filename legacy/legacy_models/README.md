# Legacy models (reference only)

Superseded Sort-of-CLEVR models, kept so the results they produced stay
readable and reproducible. **They are not imported by `src/` and are not
registered in any MODELS dict.**

| file | what it was |
|---|---|
| `syncnet_v1.py` | AKOrN rotor field: Kuramoto dynamics over spatial locations, routing by rotor coherence |
| `syncnet_v2.py` | v1 + CTC gain routing and gated content integration (never run at scale) |
| `syncnet_v3.py` | parameter-matched no-oscillator control; best of the three (0.778 vs v1 0.693) |
| `recurrent_syncnet.py` | phase-as-module-state; also provided `ctm_syncnet` via `readout_mode='sync'` |

Superseded by `src/tasks/sort_of_clevr/models/syncnet.py`, which composes
the same ideas as configurable axes (partition / gate mode / message
aggregation / readout / conditioning).

**Their numbers predate the dataset corrections** (constant-answer
non-relational subtypes, the `999` nearest-neighbour sentinel, diagonal
object overlap), so they are not comparable to anything produced after
that fix without a rerun.

To run one, copy it back under `src/tasks/sort_of_clevr/models/`, restore
its relative imports, and add a `ModelSpec` to `models/__init__.py`.
