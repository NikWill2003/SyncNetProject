"""Instrumentation that depends on more than the contract.

`interventions` reads the Sort-of-CLEVR question taxonomy; `sync_metrics`,
`sync_viz` and `t_variance` read the models' trace dicts. They sit outside
both models/ and datasets/ because they span both.
"""

from dataclasses import replace

from ..core.callbacks import CallbackSpec

ANALYSIS_CALLBACKS: dict[str, CallbackSpec] = {}

for _module in ('sync_metrics', 'sync_viz', 'interventions', 't_variance'):
    try:
        _mod = __import__(f'{__name__}.{_module}', fromlist=['*'])
    except Exception:
        continue
    for _name in dir(_mod):
        _obj = getattr(_mod, _name)
        if isinstance(_obj, dict) and _name.endswith('_callbacks'):
            for _cb_name, _spec in _obj.items():
                # these read phase traces or override the dynamics, so a
                # model must declare 'sync' in `supported_callbacks`
                ANALYSIS_CALLBACKS[_cb_name] = replace(
                    _spec, requires=frozenset({'sync'}))
