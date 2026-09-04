#!/usr/bin/env python3
"""Fast wiring check for every ablation / sweep / seed cell: build the model,
run a real batch forward + backward, then every intervention override the
model declares. Catches a broken code path in seconds instead of GPU-hours.
    python scripts/check_ablation_wiring.py            # both tasks
    python scripts/check_ablation_wiring.py sqoop      # one task
"""
from __future__ import annotations
import os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hydra import compose, initialize_config_dir
from src.core.registry import build_dataloaders, build_loss_fn, build_model, register_configs

SOC = ['task=sort_of_clevr', 'dataset.train_size=64', 'dataset.test_size=32', 'dataset.nb_questions=2']
SQ  = ['task=sqoop', 'dataset.train_size=5184', 'dataset.test_size=5184', 'dataset.rhs_variety=1']
CSOC = ['experiment=sort_of_clevr/sync/hybrids/identity_spatial', 'model.readout_prior=false']
CSQ  = ['experiment=sqoop/sync/hybrids/identity_partition_cnnenc', 'model.readout_prior=false']
CELLS = {
 'soc': [('canonical', CSOC), ('no_bias', CSOC+['model.claim_prior=null']),
         ('hard_partition', CSOC+['model.claim_prior_init=partition','model.claim_prior_scale=50']),
         ('anchors_off', CSOC+['model.per_module_anchors=false']), ('cells_off', CSOC+['model.private_cells=false']),
         ('silent', CSOC+['model.medium=silent']), ('static_addr', CSOC+['model.addresses=static']),
         ('lines_attn', CSOC+['model.medium=lines','model.gate=attn']),
         ('cnn_stem', ['experiment=sort_of_clevr/sync/hybrids/identity_spatial_cnnenc','model.readout_prior=false']),
         ('with_prior_term', ['experiment=sort_of_clevr/sync/hybrids/identity_spatial']),
         ('seed_partition', CSOC+['model.claim_prior_init=partition']), ('seed_random', CSOC+['model.claim_prior_init=random']),
         ('T2', CSOC+['model.t_bus=2']), ('T16', CSOC+['model.t_bus=16']), ('d2', CSOC+['model.phase_dim=2']),
         ('d8', CSOC+['model.phase_dim=8']), ('M4', CSOC+['model.n_modules=4']), ('M12', CSOC+['model.n_modules=12'])],
 'sqoop': [('canonical', CSQ), ('no_bias', CSQ+['model.claim_prior=null']), ('hard_partition', CSQ+['model.claim_prior_scale=50']),
         ('anchors_off', CSQ+['model.per_module_anchors=false']), ('cells_off', CSQ+['model.private_cells=false']),
         ('silent', CSQ+['model.medium=silent']), ('static_addr', CSQ+['model.addresses=static']),
         ('lines_attn', CSQ+['model.medium=lines','model.gate=attn']),
         ('with_prior_term', ['experiment=sqoop/sync/hybrids/identity_partition_cnnenc']),
         ('seed_random', CSQ+['model.claim_prior_init=random']), ('T2', CSQ+['model.t_bus=2']), ('d2', CSQ+['model.phase_dim=2']),
         ('M12', CSQ+['model.n_modules=12']),
         ('transformer_tuned', ['experiment=sqoop/matched/transformer_cnnenc']), ('workspace_tuned', ['experiment=sqoop/matched/shared_workspace_cnnenc'])],
}

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    register_configs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ok = fail = 0
    for task, cells in CELLS.items():
        if which not in ('all', task): continue
        base = SOC if task == 'soc' else SQ
        batch = None
        for name, ov in cells:
            t0 = time.time()
            try:
                with initialize_config_dir(config_dir=os.path.abspath('conf'), version_base='1.3'):
                    cfg = compose(config_name='config', overrides=base + ov + ['train.train_bs=4', 'train.val_bs=4', 'train.seed=0', 'wandb.enabled=false', 'train.compile_model=false', 'train.mixed_precision=no'])
                model = build_model(cfg).to(device)
                if batch is None:
                    loaders = build_dataloaders(cfg, device); train = loaders[0] if isinstance(loaders, (tuple, list)) else loaders['train']
                    batch = next(iter(train)); batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                loss_fn = build_loss_fn(cfg)
                out = model(batch); res = loss_fn(out if isinstance(out, dict) else {'logits': out}, batch)
                loss = res[0] if isinstance(res, (tuple, list)) else res
                loss.backward()
                nparam = sum(p.numel() for p in model.parameters())
                ngrad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
                model.eval()
                with torch.no_grad():
                    for p_ in getattr(model, 'PHASE_OVERRIDES', ()): model(batch, phase_override=p_)
                    for g_ in getattr(model, 'GATE_OVERRIDES', ()): model(batch, gate_override=g_)
                print(f'ok   {task}/{name:18s} loss={float(loss):.3f} params={nparam:,} grads={ngrad} overrides={len(getattr(model,"PHASE_OVERRIDES",()))+len(getattr(model,"GATE_OVERRIDES",()))} ({time.time()-t0:.0f}s)', flush=True)
                ok += 1
            except Exception as e:
                print(f'FAIL {task}/{name}: {type(e).__name__}: {str(e)[:120]}', flush=True); traceback.print_exc(limit=2); fail += 1
    print(f'---- {ok} ok, {fail} failed'); return 1 if fail else 0

if __name__ == '__main__':
    raise SystemExit(main())
