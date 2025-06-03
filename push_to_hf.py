import torch

from models.hf import EsoLM, EsoLMConfig

EsoLMConfig.register_for_auto_class()
EsoLM.register_for_auto_class('AutoModelForMaskedLM')

# name_or_path = 'sahoo-diffusion/Eso-LM-B-alpha-1'
# ckpt_path = (
#   '/share/kuleshov/ssahoo/elmo/'
#   'owt/elmo-owt-d3s2-as1-lowvar-10095/last.ckpt')

name_or_path = 'sahoo-diffusion/Eso-LM-B-alpha-0_25'
ckpt_path = (
  '/share/kuleshov/ssahoo/elmo/'
  'owt/elmo-owt-d3s2-as0d25-bs0d5-10389/last.ckpt')

model = EsoLM(EsoLMConfig(
    vocab_size=50258,
    model_length=1024,
    hidden_dim=768,
    cond_dim=128,
    n_blocks=12,
    n_heads=12,
    dropout=0.1,
    return_dict=False
))

model.config._name_or_path = name_or_path
model.config.auto_map = {
    'AutoConfig': f'{name_or_path}--configuration.EsoLMConfig',
    'AutoModelForMaskedLM': f'{name_or_path}--model.EsoLM',
}

# load ema params
ckpt = torch.load(ckpt_path, weights_only=False)
model = model.to('cuda')
model.load_state_dict(ckpt['state_dict'])
ema_params = ckpt['ema']['shadow_params']
for s_param, param in zip(ema_params, model.parameters()):
    if param.requires_grad:
        param.data.copy_(s_param.data)
model.push_to_hub(name_or_path, private=True)
