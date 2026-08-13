"""Mimic Trainer.train_step for every model: forward -> loss -> backward,
plus the accuracy callback. Catches contract breaks the shape test can't."""
import sys; sys.path.insert(0, '.')
import torch
from omegaconf import OmegaConf
from src.tasks.sqoop.models import MODELS as SQ
from src.tasks.sort_of_clevr.models import MODELS as SOC
from src.tasks.sqoop.config import SqoopDataConfig
from src.tasks.sort_of_clevr.config import SortOfClevrDataConfig
from src.tasks.sqoop.loss import build_cross_entropy as sq_loss
from src.tasks.sort_of_clevr.loss import build_cross_entropy as soc_loss
from src.tasks.sqoop.callbacks.metrics import _sqoop_accuracy
import src.tasks.sqoop.data.constants as SQC
import src.tasks.sort_of_clevr.data.constants as SOCC

torch.manual_seed(0); B = 8
sq_d, soc_d = SqoopDataConfig(), SortOfClevrDataConfig()
sq_batch = {'images': torch.rand(B,3,sq_d.img_size,sq_d.img_size),
            'questions': torch.stack([torch.randint(0,SQC.N_SHAPES,(B,)),
                                      torch.randint(SQC.N_SHAPES,SQC.VOCAB_SIZE,(B,)),
                                      torch.randint(0,SQC.N_SHAPES,(B,))],1),
            'answers': torch.randint(0,2,(B,))}
soc_batch = {'images': torch.rand(B,3,soc_d.img_size,soc_d.img_size),
             'questions': torch.rand(B,SOCC.QUESTION_SIZE),
             'answers': torch.randint(0,SOCC.ANSWER_SIZE,(B,))}

bad = 0
for label, models, d, batch, lb in [('SQOOP',SQ,sq_d,sq_batch,sq_loss),
                                    ('SOC',SOC,soc_d,soc_batch,soc_loss)]:
    loss_fn = lb()
    for name, spec in sorted(models.items()):
        m = spec.model_class.from_config(OmegaConf.structured(spec.config), d)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        out = m(batch)
        loss, metrics = loss_fn(out, batch)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        n_nograd = sum(1 for p in m.parameters() if p.requires_grad and p.grad is None)
        opt.step()
        cb = (_sqoop_accuracy(out['logits'], batch['answers'], batch['questions'])
              if label == 'SQOOP' else {})
        ok = torch.isfinite(loss) and torch.isfinite(gn)
        bad += (not ok)
        print(f'  {label:5s} {name:30s} loss {loss.item():.4f} gnorm {gn:.3f} '
              f'unused-params {n_nograd}'
              + (f' acc {cb["accuracy"]:.2f}' if cb else ''))
print('\nnon-finite:', bad)
sys.exit(1 if bad else 0)
