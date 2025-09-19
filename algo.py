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
    if self.config.algo.prepend_token == 'bos':
      bs = x0.shape[0]
      bos = torch.ones(
        bs, 1, dtype=torch.long, 
        device=x0.device) * self.tokenizer.bos_token_id
      input_tokens = torch.cat([bos, x0[:, :-1]], dim=-1)
      output_tokens = x0
      valid_tokens = valid_tokens
    elif self.config.algo.prepend_token == 'mask':
      bs = x0.shape[0]
      mask = torch.ones(
        bs, 1, dtype=torch.long, 
        device=x0.device) * self.mask_index
      input_tokens = torch.cat([mask, x0[:, :-1]], dim=-1)
      output_tokens = x0
      valid_tokens = valid_tokens
    elif self.config.algo.prepend_token == 'none':
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
    if self.config.algo.prepend_token == 'bos':
      num_pred_tokens = self.num_tokens
    elif self.config.algo.prepend_token == 'mask':
      num_pred_tokens = self.num_tokens
    elif self.config.algo.prepend_token == 'none':
      num_pred_tokens = self.num_tokens - 1

    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)

    if self.config.algo.prepend_token == 'bos':
      x[:, 0] = self.tokenizer.bos_token_id
    elif self.config.algo.prepend_token == 'mask':
      x[:, 0] = self.mask_index
    elif self.config.algo.prepend_token == 'none':
      x[:, 0] = self.tokenizer.bos_token_id

    kv_cache = self.config.sampling.kv_cache
    self.backbone.reset_kv_cache()
    profile_throughput = self.config.sampling.profile_throughput
    if profile_throughput:
      torch.cuda.synchronize()
    start = time.perf_counter()
    nfe = 0
    for i in range(num_pred_tokens):
      output = self.backbone(
        x[:, :i + 1], sigma=None, x0=None, kv_cache=kv_cache)
      nfe += 1
      if not profile_throughput:
        output[:, :, self.mask_index] = self.neg_infinity
        if self.config.sampling.p_nucleus < 1:
          output = output.to(torch.float64)
          output = utils.top_k_top_p_filtering(
            output, top_p=self.config.sampling.p_nucleus)
        # generate noise on the fly to avoid memory issues
        u = torch.rand((x.shape[0], self.vocab_size),
                        dtype=torch.float64, device=self.device)
        noise = -torch.log(-torch.log(u))
        y = (output[:, -1, :] + noise).argmax(-1)
        x[:, i + 1] = y
    if profile_throughput:
      torch.cuda.synchronize()
    duration = time.perf_counter() - start
    print(f'Sampling duration: {duration} seconds')
    self.backbone.reset_kv_cache()
    if self.config.algo.prepend_token in ['bos', 'mask']:
      x = x[:, 1:]
    return x, float(nfe), duration

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

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    log_p_theta = log_x_theta.gather(
      dim=-1,
      index=x0[:, :, None])[:, :, 0]
    # carry-over unmasking
    loss_mask = xt == self.mask_index
    log_p_theta = log_p_theta * loss_mask
    if low_var:
      return -log_p_theta
    else:
      return dalpha_t / (1 - alpha_t) * log_p_theta
    
  def _sample_nfe(self, num_steps):
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
    assert remaining_tokens == 0
    return len(num_tokens_to_unmask)

  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert torch.all(t - dt >= 0)
    _, stay_chance_t = self.noise(t)
    _, stay_chance_s = self.noise(t - dt)
    move_chance_t = 1 - stay_chance_t
    move_chance_s = 1 - stay_chance_s
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0 is None:
      assert not self.config.algo.time_conditioning
      sigma_t = torch.zeros(x.shape[0], device=x.device)
      logits_p_x0 = self.backbone(x, sigma_t)
      logits_p_x0[:, :, self.mask_index] = self.neg_infinity
      if self.config.sampling.p_nucleus < 1:
        logits_p_x0 = logits_p_x0.to(torch.float64)
        logits_p_x0 = utils.top_k_top_p_filtering(
          logits_p_x0, top_p=self.config.sampling.p_nucleus)
      p_x0 = logits_p_x0.softmax(dim=-1)

    q_xs = p_x0 * (1 - mask_prob)
    q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
    logits_q_xs = torch.log(q_xs)

    u = torch.rand((x.shape[0], self.num_tokens, self.vocab_size), 
                    dtype=torch.float64, device=self.device)
    noise = -torch.log(-torch.log(u))

    x_sample = (logits_q_xs + noise).argmax(-1)
    copy_flag = (x != self.mask_index).to(x.dtype)
    x_new = copy_flag * x + (1 - copy_flag) * x_sample

    if not torch.allclose(x_new, x):
      return None, x_new
    else:
      return p_x0, x_new

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, **kwargs):
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x_accum = self.prior_sample(num_samples, self.num_tokens)
    ones = torch.ones((num_samples, 1), dtype=self.dtype,
                      device=self.device)
    dt = 1 / num_steps
    p_x0_cache = None
    timesteps = torch.linspace(1, dt, num_steps, 
                               device=self.device)
    profile_throughput = self.config.sampling.profile_throughput
    if profile_throughput:
      torch.cuda.synchronize()
    start = time.perf_counter()
    if profile_throughput:
      nfe = self._sample_nfe(num_steps)
      for i in range(nfe):
        sigma_t = torch.zeros(
          x_accum.shape[0], device=x_accum.device)
        _ = self.backbone(x_accum, sigma_t)
    else:
      if self.config.sampling.use_ar_order:
        sigma_t = torch.zeros(x_accum.shape[0], device=x_accum.device)
        for i in range(self.config.model.length):
          logits_p_x0 = self.backbone(x_accum, sigma_t)
          logits_p_x0 = logits_p_x0[:, i]  # [bs, vocab size]
          logits_p_x0[:, self.mask_index] = self.neg_infinity
          if self.config.sampling.p_nucleus < 1:
            logits_p_x0 = logits_p_x0.to(torch.float64)
            logits_p_x0 = utils.top_k_top_p_filtering(
              logits_p_x0, top_p=self.config.sampling.p_nucleus)
          u = torch.rand((x_accum.shape[0], self.vocab_size), 
                    dtype=torch.float64, device=self.device)
          noise = -torch.log(-torch.log(u))
          x_sample = (logits_p_x0 + noise).argmax(-1)
          x_accum[:, i] = x_sample
        nfe = self.config.model.length
      elif self.config.sampling.use_block_ar_order:
        sigma_t = torch.zeros(x_accum.shape[0], device=x_accum.device)
        subcontext_length = self.config.model.length // 4
        for pos_within_subcontext in range(subcontext_length):
          logits_p_x0 = self.backbone(x_accum, sigma_t)
          for num_subcontext in range(4):
            i = num_subcontext * subcontext_length + pos_within_subcontext
            logits_p_x0_ = logits_p_x0[:, i]  # [bs, vocab size]
            logits_p_x0_[:, self.mask_index] = self.neg_infinity
            if self.config.sampling.p_nucleus < 1:
              logits_p_x0_ = logits_p_x0_.to(torch.float64)
              logits_p_x0_ = utils.top_k_top_p_filtering(
                logits_p_x0_, top_p=self.config.sampling.p_nucleus)
            u = torch.rand((x_accum.shape[0], self.vocab_size), 
                      dtype=torch.float64, device=self.device)
            noise = -torch.log(-torch.log(u))
            x_sample = (logits_p_x0_ + noise).argmax(-1)
            x_accum[:, i] = x_sample
        nfe = subcontext_length
      else:
        nfe = 0
        for i in range(num_steps):
          if self.mask_index not in x_accum:
            break
          t = timesteps[i]
          if p_x0_cache is None:
            nfe += 1
          p_x0_cache, x_next = self._ddpm_caching_update(
              x=x_accum, t=t * ones, dt=dt, p_x0=p_x0_cache)
          x_accum = x_next
        assert self.mask_index not in x_accum
    if profile_throughput:
      torch.cuda.synchronize()
    end = time.perf_counter()
    duration = end - start
    print(duration)
    return x_accum, float(nfe), duration


class EsoLM(MDLM):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.alpha_0 = config.algo.alpha_0
    self.noise = trainer_base.LogLinear(self.alpha_0)

  def _sort_indices(
    self, indices, shuffle, keep_masks_unshuffled=False):
    masked = (indices == self.mask_index)
    if shuffle:
      offsets = torch.rand(
        indices.shape).to(indices.device) * 0.9
      if keep_masks_unshuffled:
        # induce left-to-right order within masked tokens
        # only for sequential part
        offsets[masked] = torch.linspace(
          0, 1, torch.sum(masked)).to(indices.device)
    else:
      offsets = torch.linspace(
        0, 0.9, indices.shape[1]).to(indices.device)
    sort_idx = (masked + offsets).argsort(descending=False)
    return sort_idx

  def _any_order_ar_loss(self, x0):
    # x0 has no mask, so sort_idx would be completely random
    offsets = torch.rand(1, self.num_tokens, device='cuda')
    sort_idx = offsets.argsort(descending=False)
    sort_idx = sort_idx.expand(x0.shape[0], self.num_tokens)
    x0 = torch.gather(x0, dim=1, index=sort_idx)
    z0 = self.q_xt(x0, 0)
    # our model is time invariant, so whatever t to pass
    dummy_t0 = torch.zeros(1, z0.shape[0], dtype=self.dtype,
                           device=self.device)
    output = self.forward(
      z0, dummy_t0, sort_idx, x0=x0)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    logp_per_token = output.gather(
      -1, x0[:, :, None])[:, :, 0]
    logp_per_seq = logp_per_token.sum(dim=1)
    return logp_per_seq

  def _importance_weighted_loss(self, x0):
    batch_size = x0.shape[0]
    num_orders = self.config.eval.num_iw_orders
    num_orderings_torch = torch.tensor(
      [num_orders], device='cuda')
    logp_per_seq_per_order = torch.zeros(
      (batch_size, num_orders), device='cuda')
    for i in range(num_orders):
      logp_per_seq_one_order = self._any_order_ar_loss(x0)
      assert logp_per_seq_one_order.shape[0] == batch_size
      logp_per_seq_per_order[:, i] = logp_per_seq_one_order
    logp_per_seq = (
      torch.log(1/num_orderings_torch) + 
      torch.logsumexp(logp_per_seq_per_order, dim=1))
    loss = - logp_per_seq.sum()  # as total nll
    num_tokens = batch_size * self.num_tokens
    loss_per_token = loss / num_tokens

    return trainer_base.Loss(
        loss=loss_per_token,
        nlls=loss_per_token * num_tokens,
        reconstruction_loss=0,
        num_tokens=num_tokens)

  def _loss(self, x0, valid_tokens, 
            current_accumulation_step=None, train_mode=False):
    if self.config.eval.num_iw_orders > 0:
      return self._importance_weighted_loss(x0)

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
      reconstruction_loss, sort_idx = (
        self._reconstruction_loss(x0_reconstruction, z0))
      valid_tokens_reconstruction = torch.gather(
        valid_tokens_reconstruction, dim=1, index=sort_idx)
      reconstruction_loss = (
        reconstruction_loss * valid_tokens_reconstruction).sum()
      # artificially scale the reconstruction loss so that the
      # NLL is computed correctly.
      recons_loss_per_token = reconstruction_loss / num_recons
    else:
      recons_loss_per_token = torch.tensor([0.0]).to(x0.device)

    if do_diffusion:
      assert num_diffusion > 0
      diffusion_loss, sort_idx = self.nll(
        x0_diffusion, None, current_accumulation_step, train_mode)
      valid_tokens_diffusion = torch.gather(
        valid_tokens_diffusion, dim=1, index=sort_idx)
      diffusion_loss = (
        diffusion_loss * valid_tokens_diffusion).sum()
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
    # sort inputs and targets before passing to the model
    sort_idx = self._sort_indices(
      z0, shuffle=self.config.algo.sequential_shuffle,
      keep_masks_unshuffled=True)
    z0 = torch.gather(z0, dim=1, index=sort_idx)
    x0 = torch.gather(x0, dim=1, index=sort_idx)
    # pass sort_idx into the model to also sort pos. embeddings   
    # _process_model_output performs zero-masking trick 
    model_output_t0 = self.forward(
      z0, dummy_t0, sort_idx, x0=x0)
    reconstruction_loss = - torch.gather(
      input=model_output_t0,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    # carry-over loss masking
    loss_mask = z0 == self.mask_index
    reconstruction_loss = reconstruction_loss * loss_mask
    return reconstruction_loss, sort_idx

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del output_tokens
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t)
    # sort inputs and targets before passing to the model
    sort_idx = self._sort_indices(
      xt, shuffle=self.config.algo.diffusion_shuffle)
    xt = torch.gather(xt, dim=1, index=sort_idx)
    x0 = torch.gather(x0, dim=1, index=sort_idx)
    # pass sort_idx into the model to also sort pos. embeddings
    # _process_model_output performs zero-masking trick
    log_x_theta = self.forward(xt, sigma=sigma, sort_idx=sort_idx)
    # nll_per_token performs carry-over loss masking
    return self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var'), sort_idx
  
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
    unmasked_tokens = 0
    self.backbone.reset_kv_cache()
    # orderings are different across batch, hence the need to reset
    self.backbone.reset_sorted_rotary_cache()
    profile_throughput = self.config.sampling.profile_throughput
    if profile_throughput:
      torch.cuda.synchronize()
    start = time.perf_counter()
    nfe = 0
    for i, k in enumerate(unmask_k_tokens):
      if i == 0:
        last_k_start = 0
      else:
        last_k_start = unmasked_tokens - unmask_k_tokens[i-1]
      logits = self.backbone.forward_sample(
        zt=x,  # shape[1] is model.length
        sort_idx=sort_idx,  # shape[1] is model.length
        last_k_start=last_k_start,
        curr_k_start=unmasked_tokens,  # also last_k_end
        curr_k_end=unmasked_tokens+k)
      nfe += 1
      if not profile_throughput:
        logits[:, :, self.mask_index] = self.neg_infinity
        if self.config.sampling.p_nucleus < 1:
          logits = logits.to(torch.float64)
          logits = utils.top_k_top_p_filtering(
            logits, top_p=self.config.sampling.p_nucleus)
        indices = slice(unmasked_tokens, unmasked_tokens + k)
        # generate noise on the fly to avoid memory issues
        u = torch.rand(num_samples, k, self.vocab_size, 
                       device=self.device, dtype=torch.float64)
        noise = -torch.log(-torch.log(u))
        y = (logits + noise).argmax(-1)
        x[:, indices] = y
      unmasked_tokens += k
    if profile_throughput:
      torch.cuda.synchronize()
    end = time.perf_counter()
    duration = end - start
    print(f'Sampling duration: {duration} seconds')
    self.backbone.reset_kv_cache()
    self.backbone.reset_sorted_rotary_cache()
    sort_idx_reversed = utils.get_reverse_indices(sort_idx)
    x = torch.gather(x, dim=1, index=sort_idx_reversed)
    return x, float(nfe), duration
