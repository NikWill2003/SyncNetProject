"""Minimal CPU reproduction harness: same models, same optimiser, same
clip/schedule as the repo's Trainer, on a small real SQOOP split.

Reports train cross-entropy (the number that matters) every `--log` steps.
ln2 = 0.693147.
"""
import sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/root/proj')
from src.models.vqa_transformer import VQATransformer
from src.models.question_encoders import TokenEmbedQuestionEncoder
from src.models.syncnet import VQASyncNet, VQASyncNetConfig
from src.models.conv_lstm import VQAConvLSTM
from src.models.encoders import build_encoder
from src.core.optim import build_lr_scheduler
from src.core.config import OptimConfig
from src.tasks.sqoop.data import constants as C

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True)
ap.add_argument('--data', default='/root/proj/probe/rhs18.npz')
ap.add_argument('--steps', type=int, default=4000)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--warmup', type=int, default=200)
ap.add_argument('--clip', type=float, default=1.0)
ap.add_argument('--log', type=int, default=200)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default='')
a = ap.parse_args()

torch.set_num_threads(2)
torch.manual_seed(a.seed)
np.random.seed(a.seed)

d = np.load(a.data)
imgs = torch.from_numpy(d['images']).permute(0, 3, 1, 2).contiguous()
qs = torch.from_numpy(d['questions']).long()
ans = torch.from_numpy(d['answers']).long()
N = imgs.shape[0]


def qenc():
    return TokenEmbedQuestionEncoder(C.VOCAB_SIZE, C.QUESTION_LEN, 32)


def transformer(**kw):
    base = dict(q_encoder=qenc(), img_size=64, answer_size=2, patch_size=8,
                patch_emb_dim=96, hidden_dim=128, n_heads=4, n_layers=4,
                ffn_mult=4, dropout=0.0, pos_enc='learnt_2d',
                q_conditioning='broadcast_cat', share_layer_weights=False,
                readout='cls')
    base.update(kw)
    m = VQATransformer(**base)
    return m, (lambda i, q: m(i, q))


def syncnet(**kw):
    cfg = VQASyncNetConfig(q_conditioning='film', partition='none',
                           encoder={'name': 'patchify', 'ch': 128,
                                    'patch_size': 8}, **kw)
    m = VQASyncNet(cfg, qenc(), 64, 2)
    return m, (lambda i, q: m(i, q)['logits'])


def convlstm(**kw):
    enc = build_encoder({'name': 'cnn', 'ch': 128, 'hidden': 64}, 64)
    m = VQAConvLSTM(qenc(), enc, 2, **kw)
    return m, (lambda i, q: m(i, q))


ARMS = {
    'conv_lstm':        lambda: convlstm(),
    'conv_lstm_pool':   lambda: convlstm(readout='pool'),
    'syncnet':          lambda: syncnet(),
    'tf_bcat_cls':      lambda: transformer(),
    'tf_bcat_flat':     lambda: transformer(readout='flatten'),
    'tf_token_cls':     lambda: transformer(q_conditioning='token',
                                            patch_emb_dim=128),
    'tf_film_cls':      lambda: transformer(q_conditioning='film',
                                            patch_emb_dim=128),
    'tf_cnn_bcat_cls':  lambda: transformer(
        encoder={'name': 'cnn', 'ch': 96, 'hidden': 64}),
    'tf_cnn_bcat_flat': lambda: transformer(
        encoder={'name': 'cnn', 'ch': 96, 'hidden': 64}, readout='flatten'),
    'tf_p4_bcat_cls':   lambda: transformer(patch_size=4),
}

model, fwd = ARMS[a.arm]()
nparam = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.0)
sched = build_lr_scheduler(opt, a.steps, OptimConfig(
    optimiser='adamw', lr=a.lr, weight_decay=0.0,
    lr_scheduler='warmup_cosine',
    lr_scheduler_params={'warmup_steps': a.warmup}))
lossf = nn.CrossEntropyLoss()

name = a.tag or a.arm
print(f'== {name} | {nparam/1e6:.2f}M params | N={N:,} | steps={a.steps} '
      f'bs={a.bs} lr={a.lr} data={a.data.split("/")[-1]}', flush=True)

model.train()
hist, run_ce, run_acc, t0 = [], [], [], time.time()
for step in range(1, a.steps + 1):
    idx = torch.randint(0, N, (a.bs,))
    im = imgs[idx].float().div_(255.)
    logits = fwd(im, qs[idx])
    y = ans[idx]
    loss = lossf(logits.float(), y)
    opt.zero_grad()
    loss.backward()
    if a.clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
    opt.step()
    sched.step()
    run_ce.append(float(loss))
    run_acc.append(float((logits.argmax(-1) == y).float().mean()))
    if step % a.log == 0:
        ce, acc = float(np.mean(run_ce[-a.log:])), float(np.mean(run_acc[-a.log:]))
        hist.append({'step': step, 'ce': ce, 'acc': acc})
        print(f'  step {step:6d}  train_ce {ce:.5f}  acc {acc:.4f}  '
              f'({(time.time()-t0)/step*1000:.0f} ms/step)', flush=True)

json.dump({'arm': name, 'params': nparam, 'data': a.data, 'lr': a.lr,
           'steps': a.steps, 'hist': hist},
          open(f'/root/proj/probe/run_{name}.json', 'w'), indent=1)
