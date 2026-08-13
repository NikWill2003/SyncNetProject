"""Compose every experiment, expand its sweeper grid from the yaml, and
BUILD the model for every cell. `--cfg job` only proves the yaml parses."""
import sys, itertools, io, contextlib, yaml
sys.path.insert(0, '.')
from hydra import compose, initialize_config_dir
from pathlib import Path
from src.core.registry import register_configs, build_model
register_configs()

CONF = str(Path('conf').resolve())
EXPS = []
for task in ('sort_of_clevr', 'sqoop'):
    for x in sorted(Path(f'conf/experiment/{task}').rglob('*.yaml')):
        EXPS.append((task, str(x.relative_to('conf/experiment')).removesuffix('.yaml'), x))

fails, total = [], 0
with initialize_config_dir(config_dir=CONF, version_base='1.3'):
    for task, exp, path in EXPS:
        raw = yaml.safe_load(path.read_text()) or {}
        params = raw.get('hydra', {}).get('sweeper', {}).get('params', {}) or {}
        keys = list(params)
        grids = [str(params[k]).split(',') for k in keys]
        cells = list(itertools.product(*grids)) if keys else [()]
        n_bad = 0
        for cell in cells:
            ov = [f'task={task}', f'experiment={exp}'] + [
                f'{k}={v.strip()}' for k, v in zip(keys, cell)]
            try:
                c = compose('config', overrides=ov)
                with contextlib.redirect_stdout(io.StringIO()):
                    build_model(c)
            except Exception as e:
                n_bad += 1
                if n_bad <= 2:
                    fails.append((exp, ' '.join(ov[2:]), f'{type(e).__name__}: {e}'))
        total += len(cells)
        print(f'{"ok  " if not n_bad else f"FAIL({n_bad})"} {exp:50s} {len(cells):3d} cells')

print(f'\n{total} sweep cells; {len(fails)} distinct failures')
for exp, cell, err in fails:
    print(f'  {exp}\n    [{cell}]\n    -> {err}')
sys.exit(1 if fails else 0)
