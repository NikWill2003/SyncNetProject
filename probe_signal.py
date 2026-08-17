"""Init-time signal-propagation probe on real SQOOP images.

Question: at initialisation, how much of the model's output variance comes
from the IMAGE and how much from the QUESTION -- and how big is the
gradient that reaches the visual pathway relative to the question pathway?

A model whose logits are (near-)invariant to the image at init, and whose
image-pathway gradient is orders of magnitude below its question-pathway
gradient, sits in the question-only basin. On SQOOP the question-only
optimum is EXACTLY 0.5 accuracy / ln2 cross-entropy by construction, so
that basin is a perfectly flat attractor -- which is the observed symptom.
"""
import sys, json, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/root/proj')
from src.models.vqa_transformer import VQATransformer
from src.models.question_encoders import TokenEmbedQuestionEncoder
from src.models.syncnet import VQASyncNet, VQASyncNetConfig
from src.models.conv_lstm import VQAConvLSTM
from src.models.encoders import build_encoder
from src.tasks.sqoop.data import constants as C

torch.manual_seed(0)

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='/root/proj/probe/rhs18.npz')
ap.add_argument('--n', type=int, default=256)
args = ap.parse_args()

d = np.load(args.data)
imgs = torch.from_numpy(d['images'][:args.n]).permute(0, 3, 1, 2).float() / 255.
qs = torch.from_numpy(d['questions'][:args.n]).long()
ans = torch.from_numpy(d['answers'][:args.n]).long()
B = imgs.shape[0]
print(f'{B} examples, image mean {imgs.mean():.4f}, '
      f'frac nonzero pixels {(imgs > 0).float().mean():.4f}')


def qenc():
    return TokenEmbedQuestionEncoder(C.VOCAB_SIZE, C.QUESTION_LEN, 32)


def mk_transformer(**kw):
    base = dict(q_encoder=qenc(), img_size=64, answer_size=2, patch_size=8,
                patch_emb_dim=96, hidden_dim=128, n_heads=4, n_layers=4,
                ffn_mult=4, dropout=0.0, pos_enc='learnt_2d',
                q_conditioning='broadcast_cat', share_layer_weights=False,
                readout='cls')
    base.update(kw)
    m = VQATransformer(**base)
    return lambda i, q: m(i, q), m


def mk_syncnet():
    cfg = VQASyncNetConfig(q_conditioning='film', partition='none',
                           encoder={'name': 'patchify', 'ch': 128,
                                    'patch_size': 8})
    m = VQASyncNet(cfg, qenc(), 64, 2)
    return lambda i, q: m(i, q)['logits'], m


def mk_convlstm():
    enc = build_encoder({'name': 'cnn', 'ch': 128, 'hidden': 64}, 64)
    m = VQAConvLSTM(qenc(), enc, 2)
    return lambda i, q: m(i, q), m


MODELS = {
    'transformer/broadcast_cat': lambda: mk_transformer(),
    'transformer/film': lambda: mk_transformer(q_conditioning='film',
                                               patch_emb_dim=128),
    'transformer/token': lambda: mk_transformer(q_conditioning='token',
                                                patch_emb_dim=128),
    'transformer/flatten': lambda: mk_transformer(readout='flatten'),
    'transformer/cnn_enc': lambda: mk_transformer(
        encoder={'name': 'cnn', 'ch': 96, 'hidden': 64}),
    'syncnet': mk_syncnet,
    'conv_lstm': mk_convlstm,
}

rows = []
for name, ctor in MODELS.items():
    torch.manual_seed(0)
    fwd, m = ctor()
    m.eval()
    with torch.no_grad():
        # (a) fixed question, vary image
        q_fix = qs[:1].expand(B, -1)
        z_img = fwd(imgs, q_fix)
        d_img = (z_img[:, 1] - z_img[:, 0])
        # (b) fixed image, vary question
        i_fix = imgs[:1].expand(B, -1, -1, -1)
        z_q = fwd(i_fix, qs)
        d_q = (z_q[:, 1] - z_q[:, 0])
        # (c) both vary
        z_both = fwd(imgs, qs)
        d_both = (z_both[:, 1] - z_both[:, 0])

    # gradient reaching the visual vs question pathway at init
    m.train()
    m.zero_grad()
    logits = fwd(imgs, qs)
    loss = nn.functional.cross_entropy(logits.float(), ans)
    loss.backward()

    def gnorm(pred):
        tot = 0.0
        for n_, p in m.named_parameters():
            if p.grad is not None and pred(n_):
                tot += float(p.grad.detach().pow(2).sum())
        return tot ** 0.5

    g_vis = gnorm(lambda n_: 'encoder' in n_ and 'q_encoder' not in n_)
    g_q = gnorm(lambda n_: 'q_encoder' in n_ or n_.startswith('q_')
                or 'film' in n_ or 'h_init' in n_ or n_.startswith('lstm'))
    g_all = gnorm(lambda n_: True)

    rows.append(dict(
        model=name,
        sd_logit_vs_image=float(d_img.std()),
        sd_logit_vs_question=float(d_q.std()),
        image_share=float(d_img.std() / (d_img.std() + d_q.std() + 1e-12)),
        sd_logit_both=float(d_both.std()),
        loss0=float(loss),
        grad_visual=g_vis, grad_question=g_q, grad_total=g_all,
        grad_vis_over_q=g_vis / (g_q + 1e-12),
    ))
    print(f'{name:28s} sd(z|image)={d_img.std():.5f} '
          f'sd(z|question)={d_q.std():.5f} '
          f'img_share={rows[-1]["image_share"]:.3f} '
          f'|g_vis|={g_vis:.3e} |g_q|={g_q:.3e} '
          f'ratio={rows[-1]["grad_vis_over_q"]:.3e}')

json.dump(rows, open('/root/proj/probe/signal.json', 'w'), indent=1)


# ---- extra: inside the transformer, how big is the visual half of a
# token relative to the question half under broadcast_cat?
torch.manual_seed(0)
_, m = mk_transformer()
m.eval()
with torch.no_grad():
    feats = m.encoder(imgs)                       # (B, 96, 8, 8)
    tok = feats.flatten(2).transpose(1, 2)        # (B, 64, 96)
    q_seq = m.q_encoder(qs)
    q_flat = q_seq.flatten(1)
    qe = m.q_emb(q_flat).unsqueeze(1).expand(-1, tok.size(1), -1)
    cat = torch.cat([tok, qe], dim=-1)
    pos = m.pos_enc(cat)
    print('\n--- broadcast_cat token composition (rms per element) ---')
    print(f'  visual block (96 dims) rms  : {tok.pow(2).mean().sqrt():.5f}')
    print(f'  question block (32 dims) rms: {qe.pow(2).mean().sqrt():.5f}')
    print(f'  pos-enc rms                 : {(pos-cat).pow(2).mean().sqrt():.5f}')
    # across-token variation is what attention can actually use
    print(f'  visual across-patch std     : {tok.std(dim=1).mean():.5f}')
    print(f'  question across-patch std   : {qe.std(dim=1).mean():.5f}  (0 by construction)')
    ln = nn.LayerNorm(128)
    y = ln(pos)
    print(f'  after LayerNorm: visual rms {y[..., :96].pow(2).mean().sqrt():.5f}, '
          f'question rms {y[..., 96:].pow(2).mean().sqrt():.5f}')
    # what fraction of a patch is ink?
    ink = (imgs.sum(1) > 0).float()
    patch_ink = ink.unfold(1, 8, 8).unfold(2, 8, 8).mean(dim=(-1, -2))
    print(f'  patches with any ink: {(patch_ink > 0).float().mean():.3f} of 64')
