
# Reorganised Hydra configuration

Select exactly one task bundle:

```bash
python main.py task=sort_of_clevr
python main.py task=coalitions
```

Each task preset chooses a compatible dataset, model, loss and callback set while
keeping the final Python API unchanged (`cfg.dataset`, `cfg.model`, etc.).

Task-local overrides are namespaced:

```bash
python main.py task=sort_of_clevr model=sort_of_clevr/transformer
python main.py task=coalitions model=coalitions/attention
python main.py task=coalitions experiment=coalitions/ladder
```

Dataset preparation uses the same task selector:

```bash
python prepare_dataset.py task=sort_of_clevr
python prepare_dataset.py task=coalitions
```

`optim/`, `lr_scheduler/`, and the encoder mechanism remain shared infrastructure.
All datasets, top-level models, losses, callbacks and experiments are separated by task.
