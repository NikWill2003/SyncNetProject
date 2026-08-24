"""Exercise the end-of-training callbacks (interventions, t_variance,
binding_analysis) on synthetic scenes with a mock trainer, for each
synchrony model family. No dataset or GPU needed. Writes figures to
/tmp/cb_test/<name>/viz."""
import sys, types, logging, shutil
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import numpy as np, torch
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from compose_and_step import run, scene
from src import build_callbacks
from src.tasks.sort_of_clevr.data.constants import COLOURS

def batches(n_batches=3, bs=16, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        x = torch.from_numpy(np.stack([scene(rng) for _ in range(bs)])).permute(0, 3, 1, 2).float() / 255
        q = torch.zeros(bs, 18)
        for i in range(bs):
            fam = i % 3
            q[i, rng.integers(6)] = 1
            if fam == 2:
                q[i, 6 + rng.integers(6)] = 1
            q[i, 12 + fam] = 1
            q[i, 15 + rng.integers(3)] = 1
        out.append({'images': x, 'questions': q, 'answers': torch.randint(0, 10, (bs,))})
    return out

class MockAccelerator:
    is_main_process = True
    def get_tracker(self, *a, **k): return None

class MockTrainer:
    def __init__(self, cfg, model, out_dir):
        self.cfg = cfg; self.model = model; self.out_dir = out_dir
        self.test_dataloader = batches()
        self.logger = logging.getLogger('cb_test'); self.opt_step = 100
        self.accelerator = MockAccelerator()
        self.summaries = {}
    def log_info(self, msg, *args): print('   ', msg % args if args else msg)
    def summary(self, metrics, mode): self.summaries.update({f'{mode}_{k}': v for k, v in metrics.items()})

configs = {
    'syncnet_v2_quadrant': ['experiment=sort_of_clevr/sync_a/ladder_4_v2'],
    'syncnet_objects_partition': ['experiment=sort_of_clevr/sync_a/objects_partition', 'model.gate_mode=phase_io'],
    'syncnet_free_d4': ['experiment=sort_of_clevr/sync_a/segregation_v2', 'model.gate_mode=phase', 'model.read_norm=modules', 'model.osc_dim=4'],
    'phasebind_grid': ['experiment=sort_of_clevr/sync_b/pb_full'],
    'phasebind_objects': ['experiment=sort_of_clevr/sync_b/pb_objects', 'model.partition=none', 'model.gate_mode=phase'],
    'osc_field': ['experiment=sort_of_clevr/sync_b/osc_field', 'model.readout=sync', 'model.coupling=conv'],
    'osc_field_content': ['experiment=sort_of_clevr/sync_b/osc_field', 'model.readout=content', 'model.coupling=none'],
}
for name, ov in configs.items():
    cfg, model, out, sup = run(['task=sort_of_clevr'] + ov)
    cfg.wandb.enabled = False
    out_dir = f'/tmp/cb_test/{name}'; shutil.rmtree(out_dir, ignore_errors=True); Path(out_dir).mkdir(parents=True)
    tr = MockTrainer(cfg, model, out_dir)
    cbs = build_callbacks(cfg)
    print(f'== {name}')
    cbs.on_train_end(tr)
    keys = sorted(tr.summaries)
    ivs = [k for k in keys if 'interventions' in k and k.endswith('_drop') and not any(s in k for s in ['ternary', 'binary', 'shape', 'centre', 'half', 'count', 'band'])]
    bnd = [k for k in keys if 'binding' in k]
    print('    interventions:', {k.split('/')[-1]: round(v, 3) for k, v in tr.summaries.items() if k in ivs})
    print('    binding:', {k.split('/')[-1]: round(v, 3) for k, v in tr.summaries.items() if k in bnd})
    print('    figures:', sorted(p.name for p in Path(out_dir, 'viz').glob('*.png')))
