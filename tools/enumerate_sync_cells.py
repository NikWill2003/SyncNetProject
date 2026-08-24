import sys, itertools, glob, os, time, torch
sys.path.insert(0, '.')
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from compose_and_step import run
import warnings; warnings.filterwarnings('ignore')

def cells(exp):
    with initialize_config_dir(config_dir=str(__import__('pathlib').Path('conf').resolve()), version_base='1.3'):
        cfg = compose(config_name='config', overrides=['task=sort_of_clevr', f'experiment={exp}'], return_hydra_config=True)
    sp = cfg.hydra.sweeper.params
    params = (OmegaConf.to_container(sp) if sp is not None else {}) or {}
    keys = list(params)
    vals = [[v.strip() for v in str(params[k]).split(',')] for k in keys]
    return [[f'{k}={v}' for k, v in zip(keys, combo)] for combo in itertools.product(*vals)] if keys else [[]]

total = 0; t_all = time.time()
for group in ['sync_a', 'sync_b']:
    for f in sorted(glob.glob(f'conf/experiment/sort_of_clevr/{group}/*.yaml')):
        name = os.path.basename(f)[:-5]
        if name.startswith('_'): continue
        exp = f'sort_of_clevr/{group}/{name}'
        cs = cells(exp)
        t0 = time.time(); params = set(); ok = 0
        for c in cs:
            try:
                cfg, model, out, sup = run(['task=sort_of_clevr', f'experiment={exp}'] + c)
                params.add(sum(p.numel() for p in model.parameters())); ok += 1
            except Exception as e:
                print(f'  FAIL {exp} {c}: {type(e).__name__}: {e}'); raise
        total += ok
        print(f'{exp:55s} runs={ok:2d}  params={sorted(params)}  ({time.time()-t0:.0f}s)')
print('TOTAL runs', total, f'({time.time()-t_all:.0f}s)')
