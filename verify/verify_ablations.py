#!/usr/bin/env python3
"""Ablation audit: every ablation / sweep / seed cell must be the CANONICAL
model with exactly one component changed. Resolves the full config of each
cell and diffs it against canonical, then checks the differing keys are the
ones that component is allowed to touch. Anything else is a confound."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.core.registry import register_configs

CANON = {
    'soc':   ['task=sort_of_clevr', 'experiment=sort_of_clevr/sync/hybrids/identity_spatial', 'model.readout_prior=false'],
    'sqoop': ['task=sqoop', 'experiment=sqoop/sync/hybrids/identity_partition_cnnenc', 'model.readout_prior=false', 'dataset.rhs_variety=1'],
}
# cell -> (overrides, keys the component is allowed to change)
CELLS = {
    'no_bias':         (['model.claim_prior=null'],                                   {'model.claim_prior'}),
    'hard_partition':  (['model.claim_prior_init=partition', 'model.claim_prior_scale=50'], {'model.claim_prior_init', 'model.claim_prior_scale'}),
    'anchors_off':     (['model.per_module_anchors=false'],                           {'model.per_module_anchors'}),
    'cells_off':       (['model.private_cells=false'],                                {'model.private_cells'}),
    'silent':          (['model.medium=silent'],                                      {'model.medium'}),
    'static_addr':     (['model.addresses=static'],                                   {'model.addresses'}),
    'lines_attn':      (['model.medium=lines', 'model.gate=attn'],                    {'model.medium', 'model.gate'}),
    'with_prior_term': (['model.readout_prior=true'],                                 {'model.readout_prior'}),
    'seed_partition':  (['model.claim_prior_init=partition'],                         {'model.claim_prior_init'}),
    'seed_random':     (['model.claim_prior_init=random'],                            {'model.claim_prior_init'}),
    'T2':  (['model.t_bus=2'], {'model.t_bus'}),      'T16': (['model.t_bus=16'], {'model.t_bus'}),
    'd2':  (['model.phase_dim=2'], {'model.phase_dim'}), 'd8': (['model.phase_dim=8'], {'model.phase_dim'}),
    'M4':  (['model.n_modules=4'], {'model.n_modules'}), 'M12': (['model.n_modules=12'], {'model.n_modules'}),
}
STEM = {'soc': ('experiment=sort_of_clevr/sync/hybrids/identity_spatial_cnnenc', {'model.encoder', 'model.encoder.name', 'model.encoder.ch'})}

def flat(cfg):
    out = {}
    def walk(d, pre=''):
        for k, v in d.items():
            if isinstance(v, dict): walk(v, pre + k + '.')
            else: out[pre + k] = v
    walk(OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)); return out

def main() -> int:
    register_configs(); bad = 0; n = 0
    with initialize_config_dir(config_dir=os.path.abspath('conf'), version_base='1.3'):
        for task, base in CANON.items():
            c0 = flat(compose(config_name='config', overrides=base))
            cells = dict(CELLS)
            if task in STEM:
                exp, allowed = STEM[task]; cells['cnn_stem'] = ([exp], allowed)
            for name, (ov, allowed) in cells.items():
                if task == 'sqoop' and name == 'seed_partition': continue      # partition IS the sqoop canonical
                if name == 'cnn_stem':
                    ovs = [o for o in base if not o.startswith('experiment=')] + ov
                else:
                    ovs = base + ov
                c1 = flat(compose(config_name='config', overrides=ovs))
                diff = {k for k in set(c0) | set(c1) if c0.get(k) != c1.get(k) and not k.startswith('hydra')}
                extra = diff - allowed
                n += 1
                if extra:
                    bad += 1; print(f'  CONFOUND {task}/{name}: also changes {sorted(extra)}')
                elif not diff:
                    bad += 1; print(f'  NO-OP    {task}/{name}: identical to canonical')
    print(f'{n} cells audited: {bad} problems')
    return 1 if bad else 0

if __name__ == '__main__':
    raise SystemExit(main())
