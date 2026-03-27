import sys
sys.path.insert(0, '/share/thickstun/zhihan/Eso-LMs')

import torch
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import algo
import dataloader

CKPT = '/share/thickstun/zhihan/checkpoints/esolm/owt-esolmb-alpha0-1-313000/checkpoints/14-250000.ckpt'
CACHE_DIR = '/share/thickstun/zhihan/data/blockdiffusion'
CONFIG_DIR = '/share/thickstun/zhihan/Eso-LMs/configs'

GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    config = compose(
        config_name='config',
        overrides=[
            'data=openwebtext-split',
            'model=small',
            'algo=esolm',
            'algo.alpha_0=1.0',
            'algo.batch_split=1.0',
            'algo.diffusion_shuffle=True',
            'algo.diffusion_attn_mode=causal',
            'algo.loss_type=low_var',
            'model.length=1024',
            f'data.cache_dir={CACHE_DIR}',
            f'eval.checkpoint_path={CKPT}',
            'trainer.devices=1',
            '+eval.num_imdce_orders=0',
        ]
    )

tokenizer = dataloader.get_tokenizer(config)
model = algo.EsoLM.load_from_checkpoint(
    CKPT,
    tokenizer=tokenizer,
    config=config,
    map_location='cuda',
)
model.eval()
print('Model loaded.')

config.eval.num_iw_orders = 10
print('using num_iw_orders =', config.eval.num_iw_orders)

ds = load_dataset("Rowan/hellaswag")
val = ds["validation"]

SEQ_LEN = config.model.length  # 1024
PAD_ID = tokenizer.eos_token_id  # use EOS as padding

def encode_and_pad(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:SEQ_LEN]
    ids = ids + [PAD_ID] * (SEQ_LEN - len(ids))
    return ids

num_examples = 10000
print(f"Evaluating on {num_examples} examples...")
correct = 0
correct_top2 = 0

with torch.no_grad():
    for i in range(num_examples):
        row = val[i]
        ctx = row['ctx']
        endings = row['endings']
        label = int(row['label'])

        logps = []
        for ending in endings:
            text = ctx + ' ' + ending
            ids = torch.tensor(encode_and_pad(text), dtype=torch.long).unsqueeze(0).cuda()
            valid = (ids != PAD_ID).float()
            loss = model._importance_weighted_loss_parallel(ids, valid_tokens=valid)
            # loss = model._importance_weighted_loss(ids, valid_tokens=valid)
            # logp = -loss_per_token * num_tokens = -loss * SEQ_LEN
            logp = -loss.loss.item() * SEQ_LEN
            logps.append(logp)

        logps_t = torch.tensor(logps)
        pred = int(logps_t.argmax().item())
        top2 = logps_t.topk(2).indices.tolist()
        correct += int(pred == label)
        correct_top2 += int(label in top2)

        n = i + 1
        rolling_acc = correct / n
        rolling_top2 = correct_top2 / n
        print(f"[{i}] label={label} pred={pred} {'CORRECT' if pred == label else 'WRONG'} top2={top2}  |  acc={rolling_acc:.2%}  top2_acc={rolling_top2:.2%}")
        print(f"  ctx: {ctx[:80]}...")
        for j, (e, lp) in enumerate(zip(endings, logps)):
            print(f"  [{j}] logp={lp:.2f}  {e}")
        print()

print(f"Accuracy:      {correct}/{num_examples} = {correct/num_examples:.2%}")
print(f"Top-2 accuracy: {correct_top2}/{num_examples} = {correct_top2/num_examples:.2%}")
