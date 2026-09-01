"""The modality wall, asserted from both directions. A model consumes
exactly one perceptual modality -- pixels or ground-truth scenes, never
both. (1) Every pixel model's logits are BIT-IDENTICAL with and without a
`scenes` key in the batch: no pixel path can have grown a dependence on
annotations. (2) The token model constructs no pixel machinery at all --
no encoder, no field, no positional embedding -- and runs from scenes
alone. Detached: imports src as a library; nothing imports this."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.sync.busnet import BusNet, BusNetConfig                      # noqa: E402
from src.models.sync.identity_busnet import (IdentityBusNet,                 # noqa: E402
                                             IdentityBusNetConfig)
from src.models.sync.gated import GatedNet, GatedNetConfig                   # noqa: E402
from src.models.sync.token_busnet import TokenBusNet, TokenBusNetConfig      # noqa: E402


def batch_for(dataset: str, B: int = 4, with_scenes: bool = False):
    img = 75 if dataset == 'sort_of_clevr' else 64
    torch.manual_seed(7)
    b = {'images': torch.rand(B, 3, img, img),
         'answers': torch.zeros(B, dtype=torch.long)}
    if dataset == 'sort_of_clevr':
        b['questions'] = torch.zeros(B, 18)
        b['questions'][torch.arange(B), torch.randint(0, 18, (B,))] = 1
    else:
        b['questions'] = torch.randint(0, 36, (B, 3))
    if with_scenes:
        b['scenes'] = torch.cat([torch.randint(0, 36, (B, 5, 1)),
                                 torch.randint(0, img, (B, 5, 2))], -1)
    return b


def main() -> int:
    for dataset in ('sort_of_clevr', 'sqoop'):
        vocab = None if dataset == 'sort_of_clevr' else 40
        for name, Model, Cfg in (('busnet', BusNet, BusNetConfig),
                                 ('identity_busnet', IdentityBusNet, IdentityBusNetConfig),
                                 ('gated', GatedNet, GatedNetConfig)):
            torch.manual_seed(0)
            m = Model.from_config(Cfg(q_onehot_vocab=vocab), dataset, 10).eval()
            b1 = batch_for(dataset)
            img = 75 if dataset == 'sort_of_clevr' else 64
            b2 = {**b1, 'scenes': torch.cat([torch.randint(0, 36, (4, 5, 1)),
                                             torch.randint(0, img, (4, 5, 2))], -1)}
            with torch.no_grad():
                torch.manual_seed(1)
                a = m(b1)['logits']
                torch.manual_seed(1)
                c = m(b2)['logits']                                          # same pixels, scenes key added
            assert torch.equal(a, c), f'{name}/{dataset}: scenes in the batch changed pixel logits'
            print(f'  ok  {name}/{dataset}: logits bit-identical with and without scenes')

    for ds, cfg, adim, nobj in (('sqoop', TokenBusNetConfig(q_onehot_vocab=40), 2, 5),
                                ('sort_of_clevr', TokenBusNetConfig(n_modules=6), 10, 6)):
        torch.manual_seed(0)
        tok = TokenBusNet.from_config(cfg, ds, adim).eval()
        for attr in ('field_enc', 'field', 'pos_emb', 'anchor_to_phase'):
            assert not hasattr(tok, attr), f'token/{ds} constructed pixel machinery: {attr}'
        b = batch_for(ds, with_scenes=True)
        if ds == 'sort_of_clevr':
            b['scenes'] = torch.cat([torch.randint(0, 75, (4, 6, 2)),
                                     torch.randint(0, 2, (4, 6, 1))], -1).float()
        with torch.no_grad():
            out = tok(b)
        assert out['logits'].shape == (4, adim)
        print(f'  ok  token_busnet/{ds}: no pixel machinery constructed; runs from scenes alone')
    print('\nsegregation holds in both directions')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
