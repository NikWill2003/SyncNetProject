"""Construct every registered model, run a forward, check the output shape
and that the trainer's calling convention works."""
import sys; sys.path.insert(0, '.')
import torch
from omegaconf import OmegaConf
from src.tasks.sort_of_clevr.models import MODELS as SOC
from src.tasks.sqoop.models import MODELS as SQ
from src.tasks.sort_of_clevr.config import SortOfClevrDataConfig
from src.tasks.sqoop.config import SqoopDataConfig
import src.tasks.sort_of_clevr.data.constants as SOCC
import src.tasks.sqoop.data.constants as SQC

torch.manual_seed(0)
B = 4
soc_data = SortOfClevrDataConfig()
sq_data = SqoopDataConfig()

soc_batch = {
    'images': torch.rand(B, 3, soc_data.img_size, soc_data.img_size),
    'questions': torch.rand(B, SOCC.QUESTION_SIZE),
    'answers': torch.randint(0, SOCC.ANSWER_SIZE, (B,)),
}
sq_batch = {
    'images': torch.rand(B, 3, sq_data.img_size, sq_data.img_size),
    'questions': torch.stack([
        torch.randint(0, SQC.N_SHAPES, (B,)),
        torch.randint(SQC.N_SHAPES, SQC.VOCAB_SIZE, (B,)),
        torch.randint(0, SQC.N_SHAPES, (B,)),
    ], dim=1),
    'answers': torch.randint(0, 2, (B,)),
}

fail = 0
for label, models, data_cfg, batch, n_ans in [
        ('SOC', SOC, soc_data, soc_batch, SOCC.ANSWER_SIZE),
        ('SQOOP', SQ, sq_data, sq_batch, SQC.ANSWER_SIZE)]:
    for name, spec in sorted(models.items()):
        try:
            cfg = OmegaConf.structured(spec.config)
            model = spec.model_class.from_config(cfg, data_cfg)
            out = model(batch)                       # trainer convention
            assert isinstance(out, dict), type(out)
            assert out['logits'].shape == (B, n_ans), out['logits'].shape
            n = sum(p.numel() for p in model.parameters())
            extra = ''
            if hasattr(model, 'T'):
                o = model(batch, t_override=2)       # t_variance convention
                assert o['logits'].shape == (B, n_ans)
                extra = f' T={model.T} t_override ok'
            print(f'  {label:5s} {name:32s} {n/1e6:6.2f}M  logits {tuple(out["logits"].shape)}{extra}')
        except Exception as e:
            fail += 1
            print(f'  {label:5s} {name:32s} FAIL {type(e).__name__}: {e}')
print('failures:', fail)
sys.exit(1 if fail else 0)
