"""Every arm the sweeps will visit, on both tasks. A config that only
composes is not a config that runs -- this builds and forwards each."""
import sys, itertools; sys.path.insert(0, '.')
import torch
from omegaconf import OmegaConf
from src.tasks.sort_of_clevr.models import MODELS as SOC
from src.tasks.sqoop.models import MODELS as SQ
from src.tasks.sort_of_clevr.config import SortOfClevrDataConfig
from src.tasks.sqoop.config import SqoopDataConfig
import src.tasks.sort_of_clevr.data.constants as SOCC
import src.tasks.sqoop.data.constants as SQC

torch.manual_seed(0); B = 3
soc_data, sq_data = SortOfClevrDataConfig(), SqoopDataConfig()
soc_batch = {'images': torch.rand(B,3,soc_data.img_size,soc_data.img_size),
             'questions': torch.rand(B, SOCC.QUESTION_SIZE)}
sq_batch = {'images': torch.rand(B,3,sq_data.img_size,sq_data.img_size),
            'questions': torch.stack([
                torch.randint(0,SQC.N_SHAPES,(B,)),
                torch.randint(SQC.N_SHAPES,SQC.VOCAB_SIZE,(B,)),
                torch.randint(0,SQC.N_SHAPES,(B,))], 1)}

TASKS = {'SOC': (SOC, soc_data, soc_batch, SOCC.ANSWER_SIZE, 'sort_of_clevr'),
         'SQOOP': (SQ, sq_data, sq_batch, SQC.ANSWER_SIZE, 'sqoop')}

fails = []
def run(task, model_key, overrides, tag):
    models, data_cfg, batch, n_ans, _ = TASKS[task]
    spec = models[model_key]
    try:
        cfg = OmegaConf.structured(spec.config)
        for k, v in overrides.items():
            OmegaConf.update(cfg, k, v, merge=True)
        m = spec.model_class.from_config(cfg, data_cfg)
        out = m(batch)
        assert out['logits'].shape == (B, n_ans)
        assert torch.isfinite(out['logits']).all(), 'non-finite logits'
    except Exception as e:
        fails.append((task, tag, f'{type(e).__name__}: {e}'))
        print(f'  FAIL {task:5s} {tag:52s} {type(e).__name__}: {e}')
        return
    print(f'  ok   {task:5s} {tag}')

print('--- transformer q_conditioning x patch/share ---')
for task, mk, pdim in [('SOC','sort_of_clevr_transformer',96), ('SQOOP','sqoop_transformer',96)]:
    for qc in ['film','broadcast_cat','token','token_seq']:
        for share in [False, True]:
            run(task, mk, {'q_conditioning':qc,'patch_emb_dim':pdim,'share_layer_weights':share},
                f'q={qc} share={share}')

print('--- syncnet: conditioning x partition ---')
for task, mk in [('SOC','sort_of_clevr_syncnet'), ('SQOOP','sqoop_syncnet')]:
    for qc in ['film','broadcast_cat','token']:
        for part, nm in [('none',4), ('quadrant',4), ('views',3)]:
            run(task, mk, {'q_conditioning':qc,'partition':part,'n_modules':nm},
                f'q={qc} partition={part}')

print('--- syncnet: gate_mode x readout_mode x msg_agg ---')
for task, mk in [('SOC','sort_of_clevr_syncnet'), ('SQOOP','sqoop_syncnet')]:
    for gm in ['phase','attn','mlp','open','frozen']:
        run(task, mk, {'gate_mode':gm,'readout_mode':'sum'}, f'gate={gm} readout=sum')
    for ro in ['concat','sync','both','sum']:
        run(task, mk, {'readout_mode':ro}, f'readout={ro}')
    for agg in ['mean','bus']:
        run(task, mk, {'msg_agg':agg}, f'msg_agg={agg}')
    for extra in [{'use_module_embed':True}, {'learn_omega':False},
                  {'deterministic_phase':True}, {'module_dim':32,'n_modules':8}]:
        run(task, mk, extra, f'{extra}')

print(f'\n{len(fails)} failures')
sys.exit(1 if fails else 0)
