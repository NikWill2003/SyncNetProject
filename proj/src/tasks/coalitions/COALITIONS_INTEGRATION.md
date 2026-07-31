# Coalitions task — change log (validated design)

This documents the four fixes applied after the first sweep returned a null
result. They are already integrated into this task package; nothing outside
`src/tasks/coalitions/` and `conf/**/coalitions*` needs editing (the task
self-registers via `TASK: TaskSpec`).

## Run order

    # 1) GO/NO-GO: can the readout be learned given perfect routing?
    python prepare_dataset.py dataset=coalitions dataset.family=frustrated
    python main.py experiment=coalitions_oracle_check dataset.family=frustrated
    #    expect acc_joint -> ~1.0; if not, fix the task, don't sweep.

    # 2) Dissociation sweep (dir is keyed by family -> prepare each):
    python prepare_dataset.py dataset=coalitions dataset.family=clusterable
    python prepare_dataset.py dataset=coalitions dataset.family=frustrated
    python main.py -m experiment=coalitions_killswitch \
      model=coalitions/phase,coalitions/mlp,coalitions/recurrent,coalitions/oracle,coalitions/nocomm \
      dataset.family=clusterable,frustrated train.seed=0,1,2,3,4

    # 3) Dimension ladder (cleanest control: hold graph fixed, vary osc_dim):
    python main.py -m experiment=coalitions_killswitch model=coalitions/phase \
      dataset.family=frustrated model.osc_dim=2,3,4 train.seed=0,1,2

## The four fixes (defaults now baked in)

1. Readout too hard. mod-16 sum floored even the oracle at 0.24.
   -> integer SUM (no modulo) + K=4. Head size V = N*(K-1)+1 (=13 at N=4,K=4);
      the loss reads logits.shape[-1] so it adapts automatically.
      (data/generator.py combine(); data/constants.py readout_vocab_size;
       models/base.py + models/model.py head sizing)
2. Gate bypassable. Per-pair value projections let the model content-route
   around the gate (phase solved the star with gate_auc ~= 0.5).
   -> model.message_proj = 'shared' (sender-generic values; the gate G_ij is
      the only routing lever). 'per_pair' kept as an ablation.
      (models/base.py)
3. Routing dodgeable. Deterministic streams let modules infer neighbours from
   history. -> dataset.stream_mode = 'iid' + copy-self independent target.
      (data/generator.py; data/config)
4. Stale messages. Messages built from the previous hidden state carried t-1
   tokens; under iid even the oracle then failed. -> messages now source from
   the CURRENT token embedding ([tok_t ; h_{t-1}]).
      (models/base.py _messages + forward)

## Validation (single seed, ~450-700 steps, CPU, frustrated STAR, N=4/K=4/iid/shared)

    model                joint acc   gate_auc
    oracle               1.00        1.00
    mlp (unconstrained)  1.00        1.00
    phase d=4 (rho~0)    0.99        0.79
    phase d=2 (rho=0.41) 0.77        0.64
    no-comm              0.23        --   (floor: the integer sum concentrates)

The d=2 -> d=4 recovery on the SAME graph is the cleanest control: it holds the
readout fixed and varies only oscillator dimension, so the deficit tracks
rho(STAR,d), not a generic phase-gate weakness.

## Caveats before trusting numbers

- Single seed, short CPU runs: these validate the DESIGN, not effect sizes.
- The clusterable-at-ceiling half of the double dissociation is NOT yet
  confirmed (a quick 450-step phase d=2 run on clusterable hit only 0.475, but
  that family mixes FULL = 4-way sum with easy pairs and was undertrained).
  Confirm with adequate steps + the per-graph breakdown (acc_joint__PAIR_AB ...).
- no-comm floor is ~0.23 not chance because the sum concentrates; raise K a
  little for a wider range, but re-check the oracle stays near 1.0.

## rho (unchanged)

`python -m src.tasks.coalitions.data.rho`. Path/cycle are NOT frustrated at d=2
(they embed as arcs); the STAR K_{1,m} is. N=4 has one clean frustrated instance;
the N=6 catalogue adds K_{1,4}/K_{1,5}/K_{2,3}/K_{3,3} for a graded ladder.
Family tags come from measured rho: clusterable (~0), nonclique (~0 non-clique),
frustrated (>0).
