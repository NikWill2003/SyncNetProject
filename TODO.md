- create 3 main syncnet models to test
- create the ablations/plots/extra metrics for the syncmodels
- callbacks -> review/restructure -> DONE
- simplify the debug -> DONE
- create experiments configs for the runs
- create a script for prepare all
- review dataset code
- reveiew core


├── analysis
│   ├── __init__.py
│   ├── interventions.py
│   ├── sync_metrics.py
│   ├── sync_viz.py
│   └── t_variance.py
├── core
│   ├── callbacks.py
│   ├── config.py
│   ├── contracts.py
│   ├── __init__.py
│   ├── optim.py
│   └── registry.py
├── datasets
│   ├── base.py
│   ├── __init__.py
│   ├── soc
│   │   ├── callbacks.py
│   │   ├── generator.py
│   │   ├── loader.py
│   │   ├── spec.py
│   │   └── translate.py
│   └── sqoop
│       ├── callbacks.py
│       ├── generator.py
│       ├── loader.py
│       ├── spec.py
│       └── translate.py
├── models
│   ├── baseline
│   │   ├── conv.py **DONE**
│   │   ├── film.py **DONE**
│   │   ├── question_only.py **DONE**
│   │   ├── relnet.py **DONE**
│   │   ├── shared_workspace.py **DONE**
│   │   └── transformer.py **DONE**
│   ├── common
│   │   ├── img_enc.py **DONE**
│   │   ├── pos_enc.py **DONE**
│   │   └── qst_enc.py **DONE**
│   ├── __init__.py
│   └── sync
│       ├── busnet.py
│       ├── common
│       │   ├── osc_core.py
│       │   └── oscillators.py
│       ├── fieldsync.py
│       ├── osc_field.py
│       ├── phasebind.py
│       └── syncnet.py
├── training
│   ├── debug.py **DONE**
│   ├── early_stopping.py **DONE**
│   ├── __init__.py **DONE**
│   ├── logging.py **DONE**
│   └── trainer.py **DONE**
├── utils.py -> **DONE**
