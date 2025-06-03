import transformers


class EsoLMConfig(transformers.PretrainedConfig):
  """Hugging Face configuration class for EsoLM."""
  model_type = 'EsoLM'

  def __init__(
    self,
    vocab_size: int = 50258,
    mask_index: int = 50257,
    model_length: int = 1024,
    hidden_size: int = 768,
    cond_dim: int = 128,
    n_blocks: int = 12,
    n_heads: int = 12,
    dropout: float = 0.1,
    ** kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.mask_index = mask_index
    self.model_length = model_length
    self.hidden_size = hidden_size
    self.cond_dim = cond_dim
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.dropout = dropout
