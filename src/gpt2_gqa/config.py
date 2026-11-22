"""Model configuration (no torch import, so it can be used and tested anywhere)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    """GPT-2 style decoder with grouped-query attention.

    ``n_kv_head == n_head``  -> standard multi-head attention (GPT-2)
    ``1 < n_kv_head < n_head`` -> grouped-query attention (GQA)
    ``n_kv_head == 1``       -> multi-query attention (MQA)
    """

    block_size: int = 256     # maximum context length in tokens
    vocab_size: int = 1024
    n_layer: int = 6
    n_head: int = 6           # query heads
    n_kv_head: int = 2        # key/value heads (must divide n_head)
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_head < 1 or self.n_head % self.n_kv_head:
            raise ValueError(f"n_kv_head ({self.n_kv_head}) must divide n_head ({self.n_head})")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def group_size(self) -> int:
        """How many query heads share each key/value head."""
        return self.n_head // self.n_kv_head

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- analytic sizes ------------------------------------------------
    def num_params(self, include_position: bool = True) -> int:
        """Parameter count (token embedding is tied with the output layer, so it is counted once)."""
        c, d, b = self.n_embd, self.head_dim, int(self.bias)
        q = c * (self.n_head * d) + b * self.n_head * d
        kv = c * (2 * self.n_kv_head * d) + b * 2 * self.n_kv_head * d
        proj = (self.n_head * d) * c + b * c
        attn = q + kv + proj
        mlp = c * 4 * c + b * 4 * c + 4 * c * c + b * c
        norms = 2 * (c + b * c)
        total = self.n_layer * (attn + mlp + norms) + (c + b * c)  # final LayerNorm
        total += self.vocab_size * c
        if include_position:
            total += self.block_size * c
        return total

    def kv_cache_bytes_per_token(self, bytes_per_value: int = 2) -> int:
        """Memory the attention cache needs per generated token (keys and values, all layers)."""
        return 2 * self.n_layer * self.n_kv_head * self.head_dim * bytes_per_value


PRESETS = {
    # CPU smoke test: runs in about a minute.
    "tiny": dict(n_layer=2, n_head=4, n_kv_head=2, n_embd=128, block_size=128, dropout=0.0,
                 batch_size=32, max_iters=300, eval_interval=100, eval_iters=10, lr=2e-3, min_lr=2e-4, warmup_iters=20),
    # Default for a free Colab GPU.
    "small": dict(n_layer=6, n_head=6, n_kv_head=2, n_embd=384, block_size=256, dropout=0.2,
                  batch_size=64, max_iters=3000, eval_interval=250, eval_iters=50, lr=1e-3, min_lr=1e-4, warmup_iters=100),
    # GPT-2 (124M) shape, with GQA. Needs a large GPU and a much larger dataset than Shakespeare.
    "gpt2": dict(n_layer=12, n_head=12, n_kv_head=4, n_embd=768, block_size=1024, dropout=0.0,
                 batch_size=8, max_iters=5000, eval_interval=500, eval_iters=20, lr=6e-4, min_lr=6e-5, warmup_iters=200),
}
