"""GPT-2 style decoder-only transformer with grouped-query attention (GQA).

Follows the architecture and training recipe from Andrej Karpathy's "Let's reproduce GPT-2 (124M)":
pre-LayerNorm blocks, GELU(tanh) MLP, learned position embeddings, tied input/output embeddings,
residual-projection init scaled by 1/sqrt(2 * n_layer), fused scaled-dot-product (flash) attention.

The one architectural change is attention: ``n_kv_head`` key/value heads are shared by groups of
``n_head // n_kv_head`` query heads. With ``n_kv_head == n_head`` this is exactly GPT-2's multi-head attention.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (portable across torch versions)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.group_size = cfg.group_size
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=cfg.bias)
        self.kv_proj = nn.Linear(cfg.n_embd, 2 * cfg.n_kv_head * cfg.head_dim, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embd, bias=cfg.bias)
        self.c_proj.SCALE_INIT = True  # scaled residual init, see GPT._init_weights
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)        # (B, H, T, D)
        k, v = self.kv_proj(x).split(self.n_kv_head * self.head_dim, dim=2)
        k = k.reshape(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)               # (B, Hkv, T, D)
        v = v.reshape(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.group_size > 1:
            # Query heads [g*G, (g+1)*G) all read key/value head g.
            k = k.repeat_interleave(self.group_size, dim=1)                              # (B, H, T, D)
            v = v.repeat_interleave(self.group_size, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                                           is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.SCALE_INIT = True
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = GroupedQueryAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights_std(module: nn.Module, n_layer: int) -> float:
        std = 0.02
        if getattr(module, "SCALE_INIT", False):
            std *= (2 * n_layer) ** -0.5
        return std

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self._init_weights_std(module, self.config.n_layer))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, include_position: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())  # parameters() de-duplicates the tied weight
        return n if include_position else n - self.transformer.wpe.weight.numel()

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, return_all: bool = False):
        """Returns (logits, loss). Without targets, logits are for the last position only unless ``return_all``."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        logits = self.lm_head(x if return_all else x[:, [-1], :])
        return logits, None

    def configure_optimizers(self, weight_decay: float, lr: float, betas: tuple[float, float], device_type: str):
        """AdamW; weight decay only on matrices (weights, embeddings), not on biases or LayerNorm."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        extra = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressive sampling. temperature=0 is greedy decoding."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        self.train(was_training)
        return idx


def save_checkpoint(path: "str | Path", model: GPT, **extra) -> None:
    torch.save({"model": model.state_dict(), "config": model.config.to_dict(), **extra}, path)


def load_checkpoint(path: "str | Path", device: str = "cpu") -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}  # tolerate torch.compile prefixes
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt
