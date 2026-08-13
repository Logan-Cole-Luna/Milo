"""
Modern decoder-only LM (Llama-style) for from-scratch pretraining benchmarks.

Architecture (current standard practice, ~2024-2025):
  * RMSNorm (pre-norm)
  * Rotary position embeddings (RoPE)
  * SwiGLU feed-forward
  * No biases; weight-tied input/output embeddings (optional)

Sizes are configurable via GPTConfig; presets in config.py mirror common
benchmark scales (GPT-2 small/medium-ish) used in the optimizer literature.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304        # GPT-2 BPE (50257) padded to a multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    mlp_ratio: float = 8 / 3       # SwiGLU keeps ~4*d params with the gate
    rope_theta: float = 10000.0
    tie_embeddings: bool = True


def _rms_norm(x, weight, eps=1e-6):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x.to(dt)) * weight


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return _rms_norm(x, self.weight, self.eps)


def precompute_rope(head_dim, seq_len, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    cos = cos[:T].view(1, 1, T, D // 2)
    sin = sin[:T].view(1, 1, T, D // 2)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embd)
        hidden = 64 * ((hidden + 63) // 64)        # round to multiple of 64
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self._rope_cache = {}
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, T, device):
        if T not in self._rope_cache:
            self._rope_cache[T] = precompute_rope(
                self.cfg.n_embd // self.cfg.n_head, T, self.cfg.rope_theta, device)
        return self._rope_cache[T]

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = self._rope(T, idx.device)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            return logits, loss
        # inference: only last position
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n
