#!/usr/bin/env python3
"""Peak VRAM of one forward+backward+AdamW step for a cell at several micro-batch sizes, on
synthetic data (no dataset needed). Each size runs in its own process, so nothing from a
previous size is still on the card when the next is measured.
    python scripts/vram_check.py                              # relnet (8x8) and conv, batches 256..2048
    python scripts/vram_check.py --models relnet --bs 512,1024 --compile
    python scripts/vram_check.py --models syncnet
The printed per-run total adds the SQOOP training split (~13.3 GB) for gpu_cached mode.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELLS = {
    'relnet': ['task=sqoop', 'experiment=sqoop/baselines/thesis/relnet', 'model.pair_spatial=8'],
    'conv': ['task=sqoop', 'experiment=sqoop/baselines/thesis/conv'],
    'syncnet': ['task=sqoop', 'experiment=sqoop/sync/hybrids/identity_partition_cnnenc', 'model.readout_prior=false'],
}

WORKER = r'''
import os, sys, json, torch
sys.path.insert(0, %(root)r)
from hydra import compose, initialize_config_dir
from src.core.registry import build_loss_fn, build_model, register_configs
register_configs()
with initialize_config_dir(config_dir=os.path.join(%(root)r, 'conf'), version_base='1.3'):
    cfg = compose(config_name='config', overrides=%(overrides)r)
bs = %(bs)d
try:
    model = build_model(cfg).cuda()
    if %(compile)r: model = torch.compile(model)
    loss_fn = build_loss_fn(cfg); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    batch = {'images': torch.rand(bs, 3, 64, 64, device='cuda'), 'questions': torch.randint(0, 36, (bs, 3), device='cuda'), 'answers': torch.randint(0, 2, (bs,), device='cuda')}
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = loss_fn(model(batch), batch)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print('RESULT ' + json.dumps({'peak': torch.cuda.max_memory_allocated() / 2**30, 'reserved': torch.cuda.max_memory_reserved() / 2**30}))
except torch.cuda.OutOfMemoryError:
    print('RESULT ' + json.dumps({'oom': True}))
'''


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default='relnet,conv'); ap.add_argument('--bs', default='256,512,1024,2048'); ap.add_argument('--compile', action='store_true')
    args = ap.parse_args()
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    used = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30
    print(f'{torch.cuda.get_device_name(0)} ({total:.0f} GB, {used:.1f} GB in use by other processes); bf16 autocast, one fwd+bwd+AdamW step, synthetic 64x64 batches; one process per size')
    for name in args.models.split(','):
        for bs in (int(b) for b in args.bs.split(',')):
            p = subprocess.run([sys.executable, '-c', WORKER % {'root': ROOT, 'overrides': CELLS[name], 'bs': bs, 'compile': args.compile}], capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
            if not line:
                print(f'   {name:8s} bs {bs:5d}   FAILED: {(p.stderr.strip().splitlines() or ["?"])[-1][:100]}'); continue
            r = json.loads(line[0][7:])
            if r.get('oom'): print(f'   {name:8s} bs {bs:5d}   OOM on this card (alone, no split)')
            else: print(f'   {name:8s} bs {bs:5d}   peak {r["peak"]:6.2f} GB  (reserved {r["reserved"]:5.1f})   gpu_cached total {r["peak"] + 13.3:5.1f} GB per run')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
