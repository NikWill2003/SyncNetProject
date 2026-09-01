"""The hard gate: the fresh BusNet computes the canonical model, proven by
state-dict transplant. Golden fixtures were cut from the last verified
implementation (348,482 params, sync_d/field_stim6): its weights are mapped
onto the new composition, the fixture batch is run at the fixture seed, and
the logits must match BIT FOR BIT. One deliberate delta: the old model
carried a dead `module_embed` (unused on the field path); the new one does
not, so the transplant skips it and the parameter count differs by exactly
M * dm = 576.

Fully detached: imports src as a library; nothing imports this."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.sync.busnet import BusNet, BusNetConfig                      # noqa: E402

PREFIX_MAP = {
    'field_enc.': 'field_enc.', 'pos_emb': 'pos_emb', 'field.': 'field.',
    'grid_film_gamma.': 'pathways.grid_film_gamma.', 'grid_film_beta.': 'pathways.grid_film_beta.',
    'grid_norm.': 'pathways.grid_norm.', 'film_gamma.': 'pathways.film_gamma.',
    'film_beta.': 'pathways.film_beta.', 'norm.': 'pathways.norm.',
    'h_init.': 'pathways.h_init.', 'head_init.': 'pathways.head_init.',
    'head_embed': 'pathways.head_embed',
    'phase_slots.': 'binder.', 'anchor_to_phase.': 'anchor_to_phase.',
    'cell.': 'identity.cell.', 'msg_proj.': 'medium.msg_proj.', 'frame_ref': 'medium.frame_ref',
    'omega': 'dynamics.omega', 'K': 'dynamics.K', 'k_mlp.': 'dynamics.k_mlp.',
    'stim.': 'dynamics.stim.', 'gen.raw': 'dynamics.gen.raw',
    'head_out.': 'readout.head_out.', 'prior_head.': 'readout.prior_head.',
}
SKIP = ('module_embed', 'obj_tok')  # dead on the field path: unused embed + colour-tokenizer buffers


def map_key(old: str) -> str | None:
    if old in SKIP or any(old.startswith(s + '.') for s in SKIP):
        return None
    for a, b in sorted(PREFIX_MAP.items(), key=lambda kv: -len(kv[0])):
        if old == a or old.startswith(a):
            return b + old[len(a):]
    raise KeyError(f'unmapped golden key: {old}')


def main() -> int:
    gold = torch.load(ROOT / 'verify' / 'fixtures' / 'canonical_golden.pt', weights_only=False)
    model = BusNet(BusNetConfig(), 'sort_of_clevr', 10).eval()

    new_sd = model.state_dict()
    mapped, skipped = {}, []
    for k, v in gold['state_dict'].items():
        k = k.removeprefix('inner.')
        nk = map_key(k)
        (skipped.append(k) if nk is None else mapped.__setitem__(nk, v))
    missing = [k for k in new_sd if k not in mapped]
    assert not missing, f'new params with no golden source: {missing}'
    model.load_state_dict(mapped, strict=True)

    n_new = sum(p.numel() for p in model.parameters())
    print(f'params: golden {gold["param_count"]}, new {n_new} '
          f'(delta {gold["param_count"] - n_new} = dead module_embed), skipped {skipped}')

    batch = {'images': gold['images'], 'questions': gold['questions'],
             'answers': torch.zeros(gold['images'].shape[0], dtype=torch.long)}
    with torch.no_grad():
        torch.manual_seed(gold['forward_seed'])
        out = model(batch)
    diff = (out['logits'] - gold['logits']).abs().max().item()
    print(f'max |logit diff| = {diff}')
    assert diff == 0.0, 'NOT bit-identical'
    print('BIT-IDENTICAL: the composition computes the canonical model.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
