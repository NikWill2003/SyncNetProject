
# Hydra configuration

Entry points live in `scripts/` and are run as modules from the repo root,
so that `src` is importable and `outputs/` and `data/` land here:

```bash
python -m scripts.main
python -m scripts.debug
python -m scripts.prepare_dataset
```

Select exactly one task bundle:

```bash
python -m scripts.main task=sort_of_clevr
python -m scripts.main task=sqoop
```

Each task preset chooses a compatible dataset, model, loss and callback set while
keeping the final Python API unchanged (`cfg.dataset`, `cfg.model`, etc.).

Task-local overrides are namespaced:

```bash
python -m scripts.main task=sort_of_clevr model=sort_of_clevr/transformer
python -m scripts.main task=sqoop model=sqoop/relnet
python -m scripts.main task=sqoop experiment=sqoop/syncnet/capacity
```

Datasets are generated on first use: `build_dataloaders` calls the task's
`prepare` when the expected `.npz` splits are missing under
`${dataset.root}/${dataset.dir}`. `conf/prepare.yaml` is the config for a
standalone preparation entry point.

`optim/`, `lr_scheduler/`, and the encoder mechanism remain shared infrastructure.
Datasets, top-level models, losses, callbacks and experiments are separated by task.


## v16: the component refactor (Aug 2026)

`src/models/sync/` is rebuilt as a component library (components/{binding,
identity,medium,dynamics,readout,interventions} + field + conditioning),
composed by three registered models: `busnet` (M2, the canonical model),
`identity_busnet` (M3), `token_busnet` (the scenes-fed mechanism isolator;
SQOOP only). Verified in `verify/`: the composition is BIT-IDENTICAL to the
last trusted implementation (state-dict transplant against golden fixtures),
all component invariants hold, and the pixel/scenes modality wall is
asserted from both directions. P5: a model's config file contains that
model's experiment and nothing else -- no cross-model pins.

Removed models: phasebind, fieldsync, osc_field, and syncnet -- the last
replaced by `gated`, its recomposition from the same component library
(binding by fiat, private lines, scalar circle, votes). All era results are
archived with the old repo; era experiment configs under sync_a/ .. sync_d/
and the old syncnet/ experiment dirs will no longer compose -- retained as
the historical record of what was run, not as launchable configs. Fresh
launches use the v16/ experiment tier.

## Experiment layout (v16 refresh)

    experiment/<task>/
      _protocol.yaml        the frozen protocol; every tier composes on it
      baselines/
        thesis/             the comparison set, protocol-bound: in this
                            folder <=> quotable in the document
        scratch/            exploratory; never cited (see its README)
      sync/
        _family.yaml        protocol + the sync analysis bundle
        thesis/             the reported cells (canonical, identity, gated,
                            tokens); seeds at launch: train.seed=0,1,2
        ablations/          thesis-grade too: one axis moved per file,
                            named by the axis (medium_*, addresses_*,
                            identity_*, gated_*) -- separated from thesis/
                            only because each is a one-axis move off a
                            thesis cell
        scratch/            exploratory; never cited
      archive/              every pre-v16 era config, verbatim: the
                            historical record, not launchable

Tier semantics: THESIS (and ablations/) means protocol-bound and quotable
in the document; SCRATCH is where anything exploratory lives, and promotion
out of scratch is a copy plus a fresh run, never an edit. `bash_scripts/
prepare_datasets_parallel` warms every dataset the registered tiers need.
