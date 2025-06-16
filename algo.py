import os
import collections
import copy
import pickle
import time

import fsspec
import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step, train_mode):
    del train_mode, current_accumulation_step
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]

  @torch.no_grad()
  def generate_samples(self, num_samples, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    kv_cache = self.config.sampling.kv_cache
    self.backbone.reset_kv_cache()
    for i in range(num_pred_tokens):
      output = self.backbone(
        x[:, :i + 1], sigma=None, x0=None, kv_cache=kv_cache)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    self.backbone.reset_kv_cache()
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    # zero-masking probabilities
    model_output[:, :, self.mask_index] = self.neg_infinity
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output.log_softmax(-1)
    return model_output

  def nll_per_token(self, log_x_theta, x0, loss_mask, alpha_t,
                    dalpha_t, low_var=False):
    return log_x_theta.gather(
      dim=-1,
      index=x0[:, :, None])[:, :, 0]
    # carry-over unmasking
    log_p_theta = log_p_theta * loss_mask
    if low_var:
      return -log_p_theta
    else:
      return dalpha_t / (1 - alpha_t) * log_p_theta    


class EsoLM(MDLM):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.alpha_0 = config.algo.alpha_0
    self.noise = trainer_base.LogLinear(self.alpha_0)

  def _loss(self, x0, valid_tokens, 
            current_accumulation_step=None, train_mode=False):
    batch_size = x0.shape[0]
    # batch size used for diffusion loss
    split_batch = int(
      self.config.algo.batch_split * batch_size)

    x0_reconstruction = x0[split_batch:]
    x0_diffusion = x0[:split_batch]
    valid_tokens_reconstruction = valid_tokens[split_batch:]
    valid_tokens_diffusion = valid_tokens[:split_batch]
    num_recons = valid_tokens_reconstruction.sum()
    num_diffusion = valid_tokens_diffusion.sum()

    do_sequential = self.config.algo.alpha_0 != 1
    do_diffusion = self.config.algo.alpha_0 != 0
    
    if do_sequential:
      assert num_recons > 0
      alpha_start = self.config.algo.alpha_0
      z0 = self.q_xt(x0_reconstruction, alpha_start)
      reconstruction_loss = (
        self._reconstruction_loss(x0_reconstruction, z0)
        * valid_tokens_reconstruction).sum()
      # artificially scale the reconstruction loss so that the
      # NLL is computed correctly.
      recons_loss_per_token = reconstruction_loss / num_recons
    else:
      recons_loss_per_token = torch.tensor([0.0]).to(x0.device)

    if do_diffusion:
      assert num_diffusion > 0
      diffusion_loss = (
        self.nll(x0_diffusion, None, 
                 current_accumulation_step, train_mode)
        * valid_tokens_diffusion).sum()
      diffusion_loss_per_token = diffusion_loss / num_diffusion
    else:
      diffusion_loss_per_token = torch.tensor([0.0]).to(x0.device)
      
    loss_per_token = (recons_loss_per_token
                      + diffusion_loss_per_token)

    if num_recons == 0:
      num_tokens = num_diffusion
    elif num_diffusion == 0:
      num_tokens = num_recons
    else:
      num_tokens = num_diffusion
    
    return trainer_base.Loss(
        loss=loss_per_token,
        nlls=loss_per_token * num_tokens,
        reconstruction_loss=recons_loss_per_token * num_tokens,
        num_tokens=num_tokens)
 
  def _reconstruction_loss(self, x0, z0):
    dummy_t0 = torch.zeros(1, z0.shape[0], dtype=self.dtype,
                           device=self.device)
    model_output_t0 = self.forward(
      z0, dummy_t0, x0=x0)
    reconstruction_loss = - torch.gather(
      input=model_output_t0,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    return reconstruction_loss
  
  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = int(self.config.loader.global_batch_size
              * self.config.algo.batch_split)
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(self.trainer.accumulate_grad_batches)[
        accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    return t

  def _tokens_unmasked_per_step(self, num_steps):
    remaining_tokens = self.num_tokens
    num_tokens_to_unmask = []
    dt = 1 / num_steps
    # Assumes a log-linear schedule.
    for t in np.linspace(1, dt, num_steps):
      _, alpha_t = self.noise(t)
      _, alpha_s = self.noise(t - dt)
      n_unmask = np.random.binomial(
        remaining_tokens, (alpha_s - alpha_t) / (1 - alpha_t))
      if n_unmask != 0:
        num_tokens_to_unmask.append(n_unmask)
        remaining_tokens -= n_unmask
    if remaining_tokens != 0 and self.alpha_0 == 1:
      num_tokens_to_unmask.append(remaining_tokens)
    return num_tokens_to_unmask

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None,
                       eps=1e-5):
    """
    Generate samples from the model (only supports Eso-LM (B)).
    """
    if num_steps is None:
      num_steps = self.config.sampling.steps

    unmask_k_tokens = self._tokens_unmasked_per_step(num_steps)
    num_diffusion_tokens = sum(unmask_k_tokens)
    
    # for tokens to be generated by diffusion, shuffle
    # for tokens to be generated by sequential, don't shuffle
    sort_idx = torch.rand(
      num_samples, self.num_tokens).argsort(
        descending=False).to(self.device)
    sort_idx[:, num_diffusion_tokens:] = (
      sort_idx[:, num_diffusion_tokens:].sort().values)

    x = self.prior_sample(num_samples, self.num_tokens)
    x = torch.gather(x, dim=1, index=sort_idx)

    unmask_k_tokens = unmask_k_tokens + [1] * (
      self.num_tokens - num_diffusion_tokens)
    assert sum(unmask_k_tokens) == self.num_tokens
    noise = torch.distributions.Gumbel(0, 1).sample(
      (num_samples, self.num_tokens,
       self.vocab_size)).to(self.device)
    unmasked_tokens = 0
    kv_cache = self.config.sampling.kv_cache
    start = time.perf_counter()
    self.backbone.reset_kv_cache()
    for i, k in enumerate(unmask_k_tokens):
      # attn_mode and cutoffs are important when kv caching is off
      if unmasked_tokens >= num_diffusion_tokens:
        # stop doing diffusion
        attn_mode = self.config.algo.sequential_attn_mode
        # prefix-lm masking is named differently for diffusion
        # and sequential phase
        if attn_mode == 'mixed':  # prefix-lm masking for sequential
          attn_mode = 'mixed2'  # prefix-lm masking for diffusion
        cutoffs = num_diffusion_tokens
      else:
        # keep doing diffusion
        attn_mode = self.config.algo.diffusion_attn_mode
        cutoffs = unmasked_tokens
      if i == 0:
        last_k_start = 0
      else:
        last_k_start = unmasked_tokens - unmask_k_tokens[i-1]
      log_p_x0 = self.backbone.forward_sample(
        zt=x,  # shape[1] is model.length
        sort_idx=sort_idx,  # shape[1] is model.length
        attn_mode=attn_mode,
        cutoffs=cutoffs,
        kv_cache=kv_cache,
        last_k_start=last_k_start,
        curr_k_start=unmasked_tokens,  # also last_k_end
        curr_k_end=unmasked_tokens+k)
      if self.config.sampling.use_float64:
        log_p_x0 = log_p_x0.to(torch.float64)
      log_p_x0[:, :, self.mask_index] = self.neg_infinity
      if self.config.sampling.p_nucleus < 1:
        # top_k_top_p_filtering takes in logits (normalized or
        # unnormalized) and returns logits (unnormalized)
        log_p_x0 = utils.top_k_top_p_filtering(
          log_p_x0, top_p=self.config.sampling.p_nucleus)
      # log_p_x0 is unnormalized, but that's okay
      # with the gumbel max trick because normalized and
      # unnormalized logits differ by a constant, i.e., 
      # the log normalizing constant, which doesn't
      # affect the argmax operation
      indices = slice(unmasked_tokens, unmasked_tokens + k)
      if kv_cache:
        y = (log_p_x0 + noise[:, indices, :]).argmax(-1)
      else:
        y = (log_p_x0[:, indices, :] + noise[:, indices, :]).argmax(-1)
      x[:, indices] = y
      unmasked_tokens += k
    end = time.perf_counter()
    print(f'Sampling duration: {end - start} seconds')
    self.backbone.reset_kv_cache()
    sort_idx_reversed = utils.get_reverse_indices(sort_idx)
    x = torch.gather(x, dim=1, index=sort_idx_reversed)
    return x
