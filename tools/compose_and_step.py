"""Compose a config exactly as main.py would, build the model and callbacks,
and run one train step on synthetic data. Usage: compose_test.py <overrides...>"""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src import register_configs, build_model, build_callbacks
from src.tasks.sort_of_clevr.data.constants import COLOURS

register_configs()
cols = list(COLOURS.values())

def scene(rng, img_size=75, obj_size=5):
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    centres = []
    for col in cols:
        while True:
            c = rng.integers(obj_size, img_size - obj_size, 2)
            if all(abs(c[0]-o[0]) >= 2*obj_size or abs(c[1]-o[1]) >= 2*obj_size for o in centres): break
        centres.append(c)
        if rng.random() < 0.5: img[c[1]-obj_size:c[1]+obj_size+1, c[0]-obj_size:c[0]+obj_size+1] = col
        else:
            yy, xx = np.mgrid[:img_size, :img_size]; img[(xx-c[0])**2 + (yy-c[1])**2 <= obj_size**2] = col
    return img

def run(overrides, steps=1):
    with initialize_config_dir(config_dir=str(__import__('pathlib').Path('conf').resolve()), version_base='1.3'):
        cfg = compose(config_name='config', overrides=overrides)
    OmegaConf.resolve(cfg)
    model = build_model(cfg)
    cbs = build_callbacks(cfg)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(np.stack([scene(rng) for _ in range(8)])).permute(0,3,1,2).float()/255
    q = torch.zeros(8, 18); q[:, 2] = 1; q[:, 9] = 1; q[torch.arange(8) % 3 == 0, 14] = 1; q[torch.arange(8) % 3 == 1, 13] = 1; q[torch.arange(8) % 3 == 2, 12] = 1; q[:, 16] = 1
    ans = torch.randint(0, 10, (8,))
    batch = {'images': x, 'questions': q, 'answers': ans}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        out = model(batch)
        loss = torch.nn.functional.cross_entropy(out['logits'], ans)
        opt.zero_grad(); loss.backward(); opt.step()
    inner = getattr(model, 'inner', model)
    sup = getattr(inner, 'GATE_OVERRIDES', frozenset()) | getattr(inner, 'PHASE_OVERRIDES', frozenset())
    return cfg, model, out, sorted(sup)

if __name__ == '__main__':
    cfg, model, out, sup = run(sys.argv[1:])
    n = sum(p.numel() for p in model.parameters())
    print(f"OK model={cfg.model.name} params={n} metrics={ {k: round(v,3) for k,v in out.get('metrics',{}).items()} } overrides={sup}")
