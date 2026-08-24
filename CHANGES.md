# Synchrony screens A and B

Two 48 h scripts, one per GPU, 100k steps, Sort-of-CLEVR. One seed
everywhere except the anchor cells (the ones everything else is compared
against), which also get seed 1. Every run carries the analysis bundle in
`conf/callbacks/sort_of_clevr/sync.yaml`:

    interventions      gate forced open / zero / frozen / shuffled, phases frozen /
                       shuffled, phase term removed from the read; accuracy drop
                       overall, per family, per subtype; heatmap figure
    t_variance         accuracy vs test-time T (0..16), overall / binary / ternary; figure
    binding_analysis   what modules read, how gates form, how phases move -- five
                       figures, each backed by scalar metrics (test_binding/*)

    bash run_sync_a.sh     # 108 runs, ~37 h: the existing SyncNet made competent, expressible,
                           #   on object tokens, stacked fixes (v3), Kuramoto regime map
    bash run_sync_b.sh     # 106 runs, ~39 h: PhaseBind and OscField, on the grid and on objects

Both need only the existing `sort_of_clevr` dataset. Run them concurrently.
Seed-1 repeats are plain `train.seed=1` overrides on the anchor experiments
(wandb records the override in the run notes).

## Merge

Drop the tree over the repo. Files touched:

    src/models/oscillators.py             NEW  one sphere-Kuramoto step, gate shapes, couplings
    src/models/object_tokens.py           NEW  six exact object descriptors read off the pixels
    src/models/phasebind.py               NEW  PhaseBind (token + module oscillators, phase-gated read)
    src/models/osc_field.py               NEW  OscField (AKOrN-style field, no modules)
    src/models/syncnet.py                 CHANGED  new axes, all defaulting to the old behaviour
    src/models/encoders.py                CHANGED  build_encoder ignores the objects-only key
    src/models/__init__.py                CHANGED  exports
    src/tasks/sort_of_clevr/models/{phasebind,osc_field}.py   NEW  task wrappers
    src/tasks/sort_of_clevr/models/syncnet.py                 CHANGED  passes the colour table
    src/tasks/sort_of_clevr/models/__init__.py                CHANGED  registers the two models
    src/tasks/sort_of_clevr/callbacks/interventions.py        NEW
    src/tasks/sort_of_clevr/callbacks/binding_analysis.py     NEW
    src/tasks/sort_of_clevr/callbacks/t_variance.py           CHANGED  binary/ternary curves + summary keys
    src/tasks/sort_of_clevr/callbacks/__init__.py             CHANGED  registers the new callbacks
    conf/callbacks/sort_of_clevr/sync.yaml                    NEW  default metrics + interventions + T sweep
    conf/model/sort_of_clevr/{phasebind,osc_field}.yaml       NEW
    conf/experiment/sort_of_clevr/sync_a/*.yaml               NEW  38 files
    conf/experiment/sort_of_clevr/sync_b/*.yaml               NEW  40 files
    run_sync_a.sh, run_sync_b.sh                              NEW
    tools/                                                    NEW  verification (below)

`syncnet.py` is backward compatible: `tools/check_syncnet_equivalence.py`
instantiates the original file (kept as `tools/syncnet_orig.py`) and the
patched one with the same seed under eight legacy configs and checks
identical parameter counts and bit-identical logits. Run it after merging:

    python tools/check_syncnet_equivalence.py     # expect 8 lines with max|dlogit| 0.0 (or ~1e-8)
    python tools/enumerate_sync_cells.py          # composes and trains one step on every cell (190)
    python tools/test_callbacks.py                # runs all three end-of-training callbacks on
                                                  #   each model family with a mock trainer,
                                                  #   figures under /tmp/cb_test/*/viz

None of them needs the dataset or a GPU. `compose_and_step.py` takes the
same overrides as `main.py` and builds + steps one config.

## What each cell asks

### A: existing SyncNet

| file | runs | question |
|---|---|---|
| ladder_0..4 | 5 | competence: canonical -> CNN -> 4 read queries -> content 64 -> per-module GRUs ("v2") |
| gate_null_v2 | 7 | the null again on v2, + `zero` (no messages, the missing floor) and `phase_io` (send/receive phases: directed gate that can realise a star on S^1) |
| objects_partition(_express) | 7 | **no patchify**: six object tokens, one module per object, every relation through a message; gates phase / phase_io / attn / open / zero, then sharpened and d=4 |
| objects_free | 2 | objects, no partition: do six modules take one object each? |
| segregation_v2 | 6 | partition=none on v2, with and without competition over modules |
| express_* | 7 | can the gate close: sharpened sigmoid(α cos + b), d ∈ {4, 8}, no self-message |
| pressure_* | 4 | open is free under mean aggregation: bus, budget, hard top-1 (phase / attn) |
| dyn_* | 11 | learned θ⁰ (with / without dynamics), dt, fixed ω, Hebbian / no coupling, stimulus / rotate drive, T |
| sync_readout, scale_free_v2 | 4 | CTM-style readout; 8 free modules |
| ladder_5_v2_200k | 1 | is v2 still climbing at 200k |
| objects_io_sharp | 1 | directed gate, sharpened, on objects |
| objects_pressure_* | 4 | bus / budget / hard top-1 on objects, where an open gate means the mean of five messages |
| objects_dyn_* | 8 | dt, learned θ⁰ (± dynamics), Hebbian ± stimulus, T, on objects |
| objects_free_M | 2 | 8 / 12 free modules for 6 objects: do surplus modules idle? |
| v3_quadrant, v3_objects | 12 | every cheap fix stacked (sharpened, zero-diag, budget, learned θ⁰, dt 0.5, Hebbian); gate null over it on the grid and on objects |
| segregation_v3 | 2 | partition=none + competition with the v3 settings |
| conditioning_v2 | 2 | where the question enters, on v2 |
| regime_map | 9 | dt × ω spread with ω fixed: where in Kuramoto space does global sync give way |
| seed 1 | 13 | ladder_4_v2, gate_null_v2, objects_partition repeated |

### B: new models

| file | runs | question |
|---|---|---|
| pb_full | 1 | PhaseBind on the CNN grid, six modules, no partition |
| pb_objects | 6 | PhaseBind on object tokens: free vs imposed assignment × phase / open / zero gate |
| pb_read, pb_norm, pb_hard | 6 | what the phase does at the read; competition; hard assignment |
| pb_gate | 3 | message gate open / zero / attention |
| pb_coupling_*, pb_stimulus | 6 | remove one coupling path at a time (token-token, module<->token, module-module); AKOrN stimulus |
| pb_objects_read, pb_quadrant | 5 | objects: content-only vs phase read × soft/hard; quadrant control |
| pb_dyn, pb_dt, pb_size | 14 | d × T; dt × learned/random module phase; M × readout |
| osc_field | 4 | field with no modules: phase used (sync vs content readout) × oscillators interact (conv vs none) |
| osc_field_{res,dyn,long,d2} | 8 | 38×38 / 10×10; stimulus × init; T 16; scalar oscillators |
| pb_objects_{readonly,coupling,modmod,dyn,dt,M} | 15 | on objects: pure-phase read, coupling paths, module coupling, d × T, dt, surplus modules |
| pb_quadrant_gate | 2 | with segregation imposed, is the gate load-bearing |
| pb_gate_shape, pb_msg, pb_lambda | 10 | thesis vs sharpened gate × self-message; budget/bus; phase weight in the read |
| pb_omega | 4 | token natural frequencies (none/feature) × spread |
| pb_encoder, pb_cond | 2 | patchify encoder; broadcast-concat conditioning |
| osc_field_{slots,width,kernel,dt,static,res_content} | 9 | 1/3 slots; wider field; 3×3 / 9×9 coupling; smaller dt; static phases (no coupling, no stimulus); 38×38 content readout |
| seed 1 | 11 | pb_full, pb_objects, osc_field repeated |

Several product sweeps contain a cell that repeats an earlier one exactly
((d=2, T=6) in pb_dyn, (stimulus, feature) in osc_field_dyn, (additive,
soft) in pb_objects_read, (global, on) in pb_objects_coupling, (true, true)
in pb_gate_shape): free seed-repeats, in addition to the seed-1 anchors.

## What to read (wandb summary keys)

    test_interventions/gate_zero_drop      messages used at all?          (0 => gate cannot matter)
    test_interventions/gate_open_drop      selectivity used?              (0 => open is as good)
    test_interventions/gate_frozen_drop    dynamics used?
    test_interventions/gate_shuffle_drop   gate input-dependent?
    test_interventions/tok_shuffle_drop    PhaseBind/OscField: binding by phase load-bearing?
    test_interventions/lambda0_drop        PhaseBind/OscField: phase term in the read load-bearing?
    *_ternary_drop                         the same, on ternary questions only

    eval_model/read_overlap    1 = modules read the same tokens (no segregation), 0 = disjoint
    eval_model/phase_R         ~1 = phases collapsed to global sync (gate == open)
    eval_model/gate_entropy    1 = every sender weighted equally
    eval_model/gate_tvar       does the gate change over the T steps (a schedule shows up here)
    eval_model/assign_purity   PhaseBind: 1 = every token owned by one module
    eval_model/module_use      PhaseBind: 1 = all modules used, 0 = one module owns everything
    eval_model/bind_R          PhaseBind: are the tokens a module reads in phase with each other
    eval_model/obj_found       objects runs: must be 6.0
    t_variance/acc_mean_T*     accuracy at test-time T in {0..16}: converging dynamics or unrolled depth
    t_variance/ternary_mean_T* the same on ternary questions

    test_binding/obj_coverage       fraction of the six objects that are some module's dominant object
    test_binding/module_purity      1 = each module reads one object
    test_binding/queried_attn       share of object-attention on the queried colour(s) (also per family)
    test_binding/coalition_score    gate mass on edges touching a queried object's module minus the rest
                                    (> 0: the gate forms the question's graph; also _binary / _ternary)
    test_binding/gate_qdep          per-sample deviation of the gate from the batch mean (input-dependence)
    test_binding/R_t{1..T}          order parameter after each step (sync onset)
    test_binding/gate_entropy_t{t}  incoming-channel entropy after each step
    test_binding/n_clusters_eff     participation ratio of the alignment matrix (1 = global sync)
    test_binding/within_obj_R       token phases: coherence within an object       (PhaseBind / OscField)
    test_binding/between_obj_align  token phases: alignment between objects        (binding = high within, low between)
    test_interventions/*_<subtype>_drop   drops per question subtype (nine)

Figures per run under {out_dir}/viz and wandb viz/*: interventions heatmap,
t_variance curves, gates_by_family, attention_maps, phase_dynamics,
module_object, token_phases.

Accuracies are one seed; the earlier cells put σ at ~0.01-0.02 on ternary,
so read differences under ~0.03 as noise and use the interventions
(within-run, paired) as the primary signal.

## Parameter counts

The screen is not parameter-matched (canonical 0.33M; v2 1.15M; PhaseBind
1.22M on the grid, 0.99M on objects; OscField 0.15-0.33M; RelNet 0.63M,
transformer 0.58M). That is fine for a mechanism screen; matching belongs
to the claims pass, once the screen says which cells deserve seeds.
