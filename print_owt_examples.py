import sys
sys.path.insert(0, '/share/thickstun/zhihan/Eso-LMs')

import datasets
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import algo
import dataloader

CKPT = '/share/thickstun/zhihan/checkpoints/esolm/owt-esolmb-alpha0-1-313000/checkpoints/14-250000.ckpt'
CACHE_DIR = '/share/thickstun/zhihan/data/blockdiffusion'
CONFIG_DIR = '/share/thickstun/zhihan/Eso-LMs/configs'

# Build config matching the training script
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
print('Model loaded successfully.')

# Load one example
dataset = datasets.load_from_disk(
    f'{CACHE_DIR}/openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat'
).with_format('torch')
x0 = dataset[0]['input_ids'].unsqueeze(0).cuda()  # [1, 1024]
valid_tokens = torch.ones_like(x0)

print(f'Example: {tokenizer.decode(x0[0])[:200]}\n')

N = 1000

def report(vals):
    t = torch.tensor(vals)
    print(f'  mean={t.mean():.4f}  std={t.std():.4f}\n')

# --- _loss (single ordering each call) ---
print(f'=== _loss ({N} calls, one ordering each) ===')
config.eval.num_iw_orders = 0
config.eval.num_imdce_orders = 0

# Wrap q_xt to capture the proportion of non-mask tokens each call
_orig_q_xt = model.q_xt
_last_prop = []
def _q_xt_recording(x0, alpha_t):
    xt = _orig_q_xt(x0, alpha_t)
    _last_prop.append((xt != model.mask_index).float().mean().item())
    return xt
model.q_xt = _q_xt_recording

vals = []
prop_non_mask = []
with torch.no_grad():
    for i in range(N):
        _last_prop.clear()
        loss = model._loss(x0, valid_tokens)
        vals.append(loss.loss.item())
        prop = _last_prop[0] if _last_prop else float('nan')
        prop_non_mask.append(prop)
        print(f'  [{i}] nll/tok = {vals[-1]:.4f}  prop_non_mask = {prop:.4f}')

model.q_xt = _orig_q_xt
report(vals)
prop_t = torch.tensor(prop_non_mask)
print(f'  prop_non_mask: mean={prop_t.mean():.4f}  std={prop_t.std():.4f}\n')

# --- _importance_weighted_loss (10 orderings averaged, called N times) ---
# print(f'=== _importance_weighted_loss ({N} calls, 10 orderings each) ===')
# config.eval.num_iw_orders = 100
# vals = []
# with torch.no_grad():
#     for i in range(N):
#         loss = model._importance_weighted_loss(x0)
#         vals.append(loss.loss.item())
#         print(f'  [{i}] nll/tok = {vals[-1]:.4f}')
# report(vals)

# --- _imdce_loss (10 orderings averaged, called N times) ---
print(f'=== _imdce_loss ({N} calls, 10 orderings each) ===')
config.eval.num_imdce_orders = 100
vals = []
with torch.no_grad():
    for i in range(N):
        loss = model._imdce_loss(x0)
        vals.append(loss.loss.item())
        print(f'  [{i}] nll/tok = {vals[-1]:.4f}')
report(vals)
