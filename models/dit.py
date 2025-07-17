import math
import typing

import einops
from functools import partial
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import lru_cache

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
import torch._inductor.config as inductor_cfg
inductor_cfg.triton.cudagraphs = True
inductor_cfg.coordinate_descent_tuning = True

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

import time


@lru_cache
def _causal_mask(b, h, q_idx, kv_idx):
  causal = q_idx >= kv_idx
  return causal


@lru_cache
def _get_causal_mask(seq_len):
  return create_block_mask(
    _causal_mask,
    B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


@lru_cache
def _bidirectional_mask(b, h, q_idx, kv_idx):
  bidirectional = q_idx == q_idx
  return bidirectional


@lru_cache
def _get_bidirectional_mask(seq_len):
  return create_block_mask(
    _bidirectional_mask,
    B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


@lru_cache
def _mixed_mask(b, h, q_idx, kv_idx, cutoffs):
  causal = q_idx >= kv_idx
  block_identity = q_idx >= cutoffs[b]
  return causal | block_identity


@lru_cache
def _get_mixed_mask(seq_len, cutoffs):
  return create_block_mask(
    partial(_mixed_mask, cutoffs=cutoffs),
    B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


@lru_cache
def _mixed2_mask(b, h, q_idx, kv_idx, cutoffs):
  causal = q_idx >= kv_idx
  block_identity = (q_idx < cutoffs[b]) & (kv_idx < cutoffs[b])
  return causal | block_identity


@lru_cache
def _get_mixed2_mask(seq_len, cutoffs):
  return create_block_mask(
    partial(_mixed2_mask, cutoffs=cutoffs),
    B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


def _block_diff_mask(b, h, q_idx, kv_idx, block_size=1, n=None):
  """
  Copied directly from BD3LM's codebase: https://github.com/kuleshov-group/bd3lms

  Constructs the specialized block diffusion attention mask for training
  composed of three masks:
  - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
  - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
  - **Block Causal Mask (M_BC)**: Attention to update x0

  Args:
      b, h: Batch and head indices (ignored for mask logic).
      q_idx, kv_idx: Query and Key indices.
      seq_len: Total sequence length.
      block_size: Defines the block structure.

  Returns:
      A boolean attention mask.
  """

  # Indicate whether token belongs to xt or x0
  x0_flag_q = (q_idx >= n)
  x0_flag_kv = (kv_idx >= n)

  # Compute block indices
  block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
  block_kv = torch.where(x0_flag_kv == 1,
                         (kv_idx - n) // block_size,
                         kv_idx // block_size)

  # **1. Block Diagonal Mask (M_BD) **
  block_diagonal = (
    block_q == block_kv) & (x0_flag_q == x0_flag_kv)

  # **2. Offset Block-Causal Mask (M_OBC) **
  offset_block_causal = ((block_q > block_kv)
                          & (x0_flag_kv == 1)
                          & (x0_flag_q == 0))

  # **3. Block-Causal Mask (M_BC) **
  block_causal = (block_q >= block_kv) & (
    x0_flag_kv == 1) & (x0_flag_q == 1)

  # **4. Combine Masks **
  return block_diagonal | offset_block_causal | block_causal


@lru_cache
def _get_seq_mask(seq_len):
  # here, seq_len means the length of zt only
  return create_block_mask(
    partial(_block_diff_mask, block_size=1, n=seq_len),
    B=None, H=None, Q_LEN=seq_len*2, KV_LEN=seq_len*2)


def _block_diff_mask_prefix_lm(b, h, q_idx, kv_idx, n, cutoffs):
  block_diff_mask_output = _block_diff_mask(
    b, h, q_idx, kv_idx, block_size=1, n=n)
  block_prefix_lm = (
    (n <= q_idx) & (q_idx < n + cutoffs[b])
    & (n <= kv_idx) & (kv_idx < n + cutoffs[b]))
  return block_diff_mask_output | block_prefix_lm


@lru_cache
def _get_seq_mask_prefix_lm(seq_len, cutoffs):
  # here, seq_len means the length of zt only
  return create_block_mask(
    partial(_block_diff_mask_prefix_lm, n=seq_len, cutoffs=cutoffs),
    B=None, H=None, Q_LEN=seq_len*2, KV_LEN=seq_len*2)


flex_attention_compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True, mode='reduce-overhead')
# flex_attention_compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True, mode='max-autotune-no-cudagraphs')
# flex_attention_compiled = flex_attention
# flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


def fused_flex_attention(q, k, v, mask=None):
  return flex_attention_compiled(q, k, v, block_mask=mask)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x, interleaved=False):
  """Copied and refactored from FlashAttention"""
  if interleaved:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return einops.rearrange(
      torch.stack((-x2, x1), dim=-1),
      "... d two -> ... (d two)",
      two=2)
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
  """
  Copied and refactored from FlashAttention
  x: (batch_size, seq_len, nheads, headdim)
  cos, sin: (seq_len, rotary_dim / 2) or (batch_size, seq_len, rotary_dim / 2)
  """
  ro_dim = cos.shape[-1] * 2
  assert ro_dim <= x.shape[-1]
  pattern = "... d -> ... 1 (2 d)"
  if interleaved:
    pattern =  "... d -> ... 1 (d 2)"
  cos = einops.repeat(cos, pattern)
  sin = einops.repeat(sin, pattern)
  return torch.cat(
      [x[..., :ro_dim] * cos
       + rotate_half(x[..., :ro_dim],
                     interleaved) * sin, x[..., ro_dim:]],
      dim=-1)


def _split_rotary(rotary_cos_sin, dtype):
  cos, sin = rotary_cos_sin
  cos = cos.to(dtype)
  sin = sin.to(dtype)
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return cos, sin


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = _split_rotary(rotary_cos_sin, dtype=qkv.dtype)
    q, k, v = qkv.chunk(3, dim=2)
    q = apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def split_and_apply_rotary_pos_emb_batch(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[:,:,0,0,:cos.shape[-1]//2]  # difference is here
    sin = sin[:,:,0,0,:sin.shape[-1]//2]  # difference is here
    q, k, v = qkv.chunk(3, dim=2)
    q = apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def flex_attention_multi_headed(q, k, v, mask):
  q = q.transpose(1, 2).contiguous()
  k = k.transpose(1, 2).contiguous()
  v = v.transpose(1, 2).contiguous()
  attention_output = fused_flex_attention(q, k, v, mask=mask)
  attention_output = attention_output.transpose(1, 2).contiguous()
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')

#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockCausal(nn.Module):
  def __init__(self, dim, n_heads, num_tokens, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.num_tokens = num_tokens

    self.dim = dim
    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    self.k_cache = None
    self.v_cache = None
    self.pointer = -1

  def _process_and_update_kv(self, k, v):
    if (self.k_cache is None
        and self.v_cache is None):
      self.k_cache = torch.empty(
        k.shape[0], self.num_tokens,  # in case of prepending
        self.n_heads, 
        self.dim // self.n_heads,
        device=k.device, dtype=k.dtype
      )
      self.v_cache = torch.empty_like(self.k_cache)
    self.pointer += 1
    self.k_cache[:, self.pointer:self.pointer+1] = k
    self.v_cache[:, self.pointer:self.pointer+1] = v
    past_k = self.k_cache[:, :self.pointer+1]
    past_v = self.v_cache[:, :self.pointer+1]
    return past_k, past_v

  @torch.no_grad()
  def _attention_with_kv_cache(self, qkv, rotary_cos_sin):
    # qkv: qkv values of the current position
    # qkv shape: [b 1 3 h d]
    # rotary_cos_sin: rotary embed of the current position
    # rotary_cos_sin shape: [1 1 3 1 d]
    assert qkv.shape[1] == 1
    # q or k or v shape: [b 1 1 h d]
    q, k, v = qkv.chunk(3, dim=2)
    # q or k or v shape: [b 1 h d]
    q = q.squeeze(dim=2)
    k = k.squeeze(dim=2)
    v = v.squeeze(dim=2)
    # apply rotary emb before kv caching
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = _split_rotary(rotary_cos_sin, q.dtype)
      q = apply_rotary_emb_torch(q, cos, sin)
      k = apply_rotary_emb_torch(k, cos, sin)
    k, v = self._process_and_update_kv(k=k, v=v)
    scale = q.shape[-1] ** 0.5
    # swap seq_len and num_heads
    # q shape:   [b h 1 d]
    # k/v shape: [b h s' d]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    x =  torch.matmul(attn_weights, v).transpose(1, 2)
    return x.view(x.shape[0], 1, self.dim)

  def forward(self, x, rotary_cos_sin, kv_cache=False, **kwargs):
    del kwargs
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    x_skip = x
    x = self.norm1(x)
    qkv = einops.rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    
    if kv_cache:
      x = self._attention_with_kv_cache(qkv.detach(), rotary_cos_sin)
    else:
      q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
      # recreate the mask every time (cheap) to fit different input length
      # different input length can happen during generation
      attn_mask = _get_causal_mask(x.shape[1])
      x = flex_attention_multi_headed(q, k, v, attn_mask)

    scale = torch.ones(1, device=x.device, dtype=x.dtype)
    x = bias_dropout_scale_fn(
      self.attn_out(x), None, scale, x_skip, self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, adaLN, num_tokens,
               cond_dim=None, mlp_ratio=4,
               dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.adaLN = adaLN
    self.num_tokens = num_tokens

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

    self.neg_infinity = -1000000.0

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    self.k_cache = None
    self.v_cache = None
    self.num_clean_cached = 0

  def _process_and_update_kv(self, k, v, num_clean):
    if num_clean == 0:
      # no caching if all we see is mask tokens
      return k, v
    else:
      if (self.k_cache is None 
          and self.v_cache is None):
        self.k_cache = torch.empty(
          k.shape[0], self.num_tokens, self.n_heads, 
          self.dim // self.n_heads,
          device=k.device, dtype=k.dtype
        )
        self.v_cache = torch.empty_like(self.k_cache)
        num_new = k.shape[1]
        self.k_cache[:, :num_new] = k
        self.v_cache[:, :num_new] = v
        self.num_clean_cached += num_clean
        return k, v
      else:
        num_new = k.shape[1]
        num_clean_cached_and_new = self.num_clean_cached + num_new
        self.k_cache[:, self.num_clean_cached:num_clean_cached_and_new] = k
        self.v_cache[:, self.num_clean_cached:num_clean_cached_and_new] = v
        self.num_clean_cached += num_clean
        k_so_far = self.k_cache[:, :num_clean_cached_and_new]
        v_so_far = self.v_cache[:, :num_clean_cached_and_new]
        return k_so_far, v_so_far

  @torch.no_grad()
  def _attention_with_kv_cache(self, qkv, rotary_cos_sin, 
                               num_clean, num_clean_and_mask):
    # qkv shape: 
    # [bs, num gen last + num to gen, 3, h, d]
    # rotary_cos_sin shape:
    # [1, num gen last + num to gen, 1, h, d]
    # num_clean: num gen last
    # num_clean_and_mask: num gen last + num to gen
    assert qkv.shape[1] == num_clean_and_mask
    # q or k or v shape: [b s' 1 h d]
    q, k, v = qkv.chunk(3, dim=2)
    # q or k or v shape: [b s' h d]
    q = q.squeeze(dim=2)
    k = k.squeeze(dim=2)
    v = v.squeeze(dim=2)
    # apply rotary emb before kv caching
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      cos = cos.to(qkv.dtype)
      sin = sin.to(qkv.dtype)
      cos = cos[:,:,0,0,:cos.shape[-1]//2]
      sin = sin[:,:,0,0,:sin.shape[-1]//2]
      q = apply_rotary_emb_torch(q, cos, sin)
      k = apply_rotary_emb_torch(k, cos, sin)
    k, v = self._process_and_update_kv(
      k=k, v=v, num_clean=num_clean)
    # new kv shape: 
    # [bs, 
    #  num gen before last + num gen last + num to gen, 
    #  h, d]
    scale = q.shape[-1] ** 0.5
    # shapes after transpose:
    # q: [bs, h, num gen last + num to gen, d]
    # k: [bs, h, num gen before last + num gen last + num to gen, d]
    # v: [bs, h, num gen before last + num gen last + num to gen, d]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # attn_scores shape: 
    # [bs, h, 
    #  num gen last + num to gen, 
    #  num gen before last + num gen last + num to gen]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    ones = torch.ones(
      num_clean_and_mask, num_clean_and_mask).to(qkv.device)
    # A contains very large negative values above the diagonal
    # - q attends to all v values over "num gen before last"
    # - q attends causally to v values within "num gen last
    #   + num to gen"
    A = self.neg_infinity * torch.triu(ones, diagonal=1)
    A = A.view(1, 1, num_clean_and_mask, num_clean_and_mask)
    attn_scores[:, :, :, -num_clean_and_mask:] += A
    attn_weights = F.softmax(attn_scores, dim=-1)
    # matmul shape: [bs, h, num gen last + num to gen, d] 
    # shape after tranpose: [bs, num gen last + num to gen, h, d]
    attn_output = torch.matmul(attn_weights, v).transpose(1, 2)
    return einops.rearrange(attn_output, 'b s h d -> b s (h d)')

  def forward(self, x, rotary_cos_sin, c=None, attn_mask=None,
              kv_cache=False, num_clean=None, num_clean_and_mask=None):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = x
    x = self.norm1(x)
    if self.adaLN:
      # self.adaLN_modulation(c): (128, 1536)
      # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
      # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
       gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = einops.rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads).contiguous()
    if kv_cache:
      x = self._attention_with_kv_cache(
        qkv.detach(), rotary_cos_sin,
        num_clean=num_clean, num_clean_and_mask=num_clean_and_mask)
    else:
      if rotary_cos_sin[0].shape[0] > 1:
        q, k, v = split_and_apply_rotary_pos_emb_batch(qkv, rotary_cos_sin)
      else:
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
      x = flex_attention_multi_headed(q, k, v, attn_mask)

    if self.adaLN:
      x = bias_dropout_scale_fn(self.attn_out(x),
                                None,
                                gate_msa,
                                x_skip,
                                self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    if x.ndim == 2:
      return self.embedding[x]
    assert x.ndim == 3
    return torch.einsum(
      "blv,ve->ble",
      torch.nn.functional.softmax(x, dim=-1).float(),
      self.embedding.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim,
               adaLN):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    x = self.norm_final(x)
    if self.adaLN:
      shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x


class DiT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.causal = config.algo.causal_attention
    self.adaLN = not self.causal
    self.config = config
    self.vocab_size = vocab_size
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    if not self.causal:
      self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_dim = dim // config.model.n_heads
    self.rotary_emb = Rotary(self.rotary_dim)

    blocks = []
    for _ in range(config.model.n_blocks):
      if self.causal:
        block = DDiTBlockCausal(
          dim=dim,
          n_heads=config.model.n_heads,
          num_tokens=config.model.length,
          dropout=config.model.dropout)
      else:
        block = DDiTBlock(
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          num_tokens=config.model.length,
          dropout=config.model.dropout)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    for block in self.blocks:
      block.reset_kv_cache()

  def forward(self, x, sigma, sort_idx=None, x0=None, kv_cache=False):
    assert x0 is None
    if kv_cache:
      seq_len_so_far = x.shape[1]
      x = x[:, -1:]
    x = self.vocab_embed(x)
    if self.causal:
      t_cond = None
    else:
      t_cond = F.silu(self.sigma_map(sigma))

    if kv_cache:
      # create dummy constant-length input to avoid recomputing cos and sin
      length = self.config.model.length
      if self.config.algo.prepend_token in ['bos', 'mask']:
        length += 1
      dummy_x = torch.zeros(1, length).to(x.device)
      rotary_cos_sin = self.rotary_emb(dummy_x)
      cos, sin = rotary_cos_sin
      cos = cos[:, seq_len_so_far:seq_len_so_far+1]
      sin = sin[:, seq_len_so_far:seq_len_so_far+1]
      rotary_cos_sin = (cos, sin)
    else:
      rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](
          x, rotary_cos_sin, c=t_cond, kv_cache=kv_cache)
      x = self.output_layer(x, c=t_cond)
    return x


class EsoLMDiT(DiT):
  def __init__(self, config, vocab_size: int, mask_index: int):
    super().__init__(config, vocab_size)
    # sequential not causal
    # this also makes sure that
    # - sigma_map was created
    # - DDiTBlock was used instead of DDiTBlockCausal
    assert not self.causal and self.adaLN
    self.mask_index = mask_index

    self.diffusion_attn_mode = config.algo.diffusion_attn_mode
    self.sequential_attn_mode = config.algo.sequential_attn_mode

    self.mdlm_mask = None
    self.seq_mask = None
  
  def _sort_rotary_cos_sin(self, rotary_cos_sin, sort_idx):
    # example cos shape: (1, 128, 3, 1, 32)
    # 128 for seq_len, 3 for qkv, 32 for head dim
    cos, sin = rotary_cos_sin
    bs = sort_idx.shape[0]
    cos = cos.expand(bs, -1, -1, -1, -1)
    sin = sin.expand(bs, -1, -1, -1, -1)
    cos = torch.gather(
      cos, dim=1, 
      index=sort_idx[:, :, None, None, None].expand(
        -1, -1, 3, -1, self.rotary_dim)).contiguous()
    sin = torch.gather(
      sin, dim=1, 
      index=sort_idx[:, :, None, None, None].expand(
        -1, -1, 3, -1, self.rotary_dim)).contiguous()
    return cos, sin

  def _get_attention_mask(self, seq_len, attn_mode=None,
                          cutoffs=None):
    if attn_mode == 'causal':
      if self.mdlm_mask is None:
        self.mdlm_mask = _get_causal_mask(seq_len)
      return self.mdlm_mask
    elif attn_mode == 'bidirectional':
      if self.mdlm_mask is None:
        self.mdlm_mask = _get_bidirectional_mask(seq_len)
      return self.mdlm_mask
    elif attn_mode == 'mixed':
      # causal over clean tokens
      # bidirectional over masked tokens
      return _get_mixed_mask(seq_len=seq_len,
                             cutoffs=cutoffs)
    elif attn_mode == 'mixed2':
      # bidirectional over clean tokens
      # causal over masked tokens
      return _get_mixed2_mask(seq_len=seq_len,
                              cutoffs=cutoffs)

  def _diffusion_features(self, zt, sort_idx,
                          attn_mode=None, cutoffs=None):
    if cutoffs is None:
      cutoffs = torch.sum(zt != self.mask_index, dim=1)
    if attn_mode is None:
      attn_mode = self.diffusion_attn_mode
    x = self.vocab_embed(zt)
    rotary_cos_sin = self.rotary_emb(x)
    rotary_cos_sin = self._sort_rotary_cos_sin(
      rotary_cos_sin, sort_idx)
    attention_mask = self._get_attention_mask(
      seq_len=zt.shape[1],
      attn_mode=attn_mode,
      cutoffs=cutoffs)
    return {'x': x,
            'rotary': rotary_cos_sin,
            'attention': attention_mask,
            'sorted_indices': sort_idx}

  def _sequential_features(self, zt, x0, sort_idx):
    seq_len = zt.shape[1]
    zt_and_x0 = torch.cat([zt, x0], dim=1)
    x = self.vocab_embed(zt_and_x0)
    rotary_cos_sin = self.rotary_emb(x[:, :seq_len])
    rotary_cos_sin = self._sort_rotary_cos_sin(
      rotary_cos_sin, sort_idx)
    cos, sin = rotary_cos_sin
    cos = torch.cat([cos, cos], dim=1)
    sin = torch.cat([sin, sin], dim=1)
    rotary_cos_sin = (cos, sin)
    
    if self.sequential_attn_mode == 'causal':
      if self.seq_mask is None:
        self.seq_mask = _get_seq_mask(seq_len)
      return {'x': x,
              'rotary': rotary_cos_sin,
              'attention': self.seq_mask,
              'sorted_indices': sort_idx}
    elif self.sequential_attn_mode == 'mixed':
      cutoffs = torch.sum(zt != self.mask_index, dim=1)
      return {'x': x,
              'rotary': rotary_cos_sin,
              'attention': _get_seq_mask_prefix_lm(
                seq_len, cutoffs=cutoffs),
              'sorted_indices': sort_idx}

  def forward(self, zt, sigma, sort_index, x0=None):
    diffusion_mode = x0 is None
    seq_len = zt.shape[1]

    if diffusion_mode:
      features = self._diffusion_features(zt, sort_index)
    else:
      features = self._sequential_features(zt, x0, sort_index)
    x = features['x']
    t_cond = F.silu(self.sigma_map(sigma))
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, features['rotary'], c=t_cond, 
                           attn_mask=features['attention'])
      x = self.output_layer(x, c=t_cond)

    if not diffusion_mode:
      x = x[:, :seq_len]
    return x

  def reset_sorted_rotary_cache(self):
    self.rotary_cos_sin_sorted = None

  @torch.no_grad()
  def forward_sample(self, zt, sort_idx,
                     last_k_start=None,
                     curr_k_start=None,
                     curr_k_end=None):
    # expect zt to be sorted already
    x = self.vocab_embed(zt[:, last_k_start:curr_k_end])

    if self.rotary_cos_sin_sorted is None:
      # avoid repeatedly sorting across steps
      rotary_cos_sin = self.rotary_emb(zt)
      self.rotary_cos_sin_sorted = self._sort_rotary_cos_sin(
        rotary_cos_sin, sort_idx)

    cos, sin = self.rotary_cos_sin_sorted
    rotary_cos_sin = (
      cos[:, last_k_start:curr_k_end], 
      sin[:, last_k_start:curr_k_end])

    zeros = torch.zeros(zt.shape[0], device=zt.device)
    t_cond = F.silu(self.sigma_map(zeros))

    num_clean = curr_k_start - last_k_start
    num_clean_and_mask = curr_k_end - last_k_start

    assert self.config.sampling.kv_cache
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](
          x, rotary_cos_sin, c=t_cond, 
          attn_mask=None,
          kv_cache=True, 
          num_clean=num_clean,
          num_clean_and_mask=num_clean_and_mask)
      x = self.output_layer(x, c=t_cond)
  
    x = x[:, num_clean:, :]
    return x
