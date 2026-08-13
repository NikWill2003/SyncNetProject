# Migration checks

Run from the repo root. No dataset needed -- all four use synthetic batches.

    python tests_migration/test_models.py         # build + forward every registered model
    python tests_migration/test_axes.py           # every sweep arm, both tasks (62 arms)
    python tests_migration/test_train_step.py     # forward -> loss -> backward -> callback
    python tests_migration/test_compose_sweeps.py # compose AND build all 256 sweep cells

The last one is the one worth running before committing a night of GPU
time: `--cfg job` proves a yaml parses, not that the model it names can
be constructed with those values.
