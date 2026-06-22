"""LM pretraining benchmark task: GPT-2-style transformer on FineWeb-Edu.

The primary arena for modern optimizer comparisons (Muon, SOAP, Sophia, and
AdEMAMix were all introduced on this task). Supports every optimizer in the
registry, including Sophia's Gauss-Newton-Bartlett Hessian loop and
schedule-free's train()/eval() protocol.

Model presets (untied embeddings by default so optimizer param-routing is
clean; pass --tie to tie them):
  small : 6L/6H/384d   (~52M params, ~30M non-embedding)  -- LR sweeps
  gpt2  : 12L/12H/768d (~163M params, 124M-class)         -- headline runs

Examples:
  python tasks/lm_pretrain.py --data shakespeare --model small --optimizer mion \
      --lr 0.003 --steps 200 --batch-size 8 --eval-every 50            # smoke
  python tasks/lm_pretrain.py --data fineweb --model small --optimizer muon \
      --lr 0.02 --tokens 4e8 --batch-size 16 --accum 2 --seed 1
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizers.registry import build_optimizer, parse_opt_kwargs  # noqa: E402
from tasks.common import (RunLogger, already_done, apply_lr, base_lrs,  # noqa: E402
                          cosine_with_warmup, count_params, maybe_compile,
                          run_id_from, set_seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ model
@dataclass
class GPTConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    ctx: int = 1024
    vocab: int = 50304  # padded for efficiency
    tie: bool = False


PRESETS = {"small": GPTConfig(),
           "gpt2": GPTConfig(n_layer=12, n_head=12, n_embd=768)}


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.mlp_up = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.mlp_down = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)
        self.n_head = c.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp_down(F.gelu(self.mlp_up(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok_embed = nn.Embedding(c.vocab, c.n_embd)
        self.pos_embed = nn.Embedding(c.ctx, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab, bias=False)
        if c.tie:
            self.lm_head.weight = self.tok_embed.weight
        self.apply(self._init)
        # GPT-2 style residual-proj scaling
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("mlp_down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.ln_f(x))


# ------------------------------------------------------------------ data
class BinData:
    def __init__(self, name, ctx):
        d = Path(__file__).resolve().parents[1] / "data"
        self.train = np.memmap(d / f"{name}_train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(d / f"{name}_val.bin", dtype=np.uint16, mode="r")
        self.ctx = ctx

    def batch(self, split, bs, generator):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.ctx - 1, (bs,), generator=generator)
        x = torch.stack([torch.from_numpy(data[i:i + self.ctx].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + self.ctx + 1].astype(np.int64)) for i in ix])
        if DEVICE == "cuda":
            return (x.pin_memory().to(DEVICE, non_blocking=True),
                    y.pin_memory().to(DEVICE, non_blocking=True))
        return x, y


# ------------------------------------------------------------------ train
@torch.no_grad()
def evaluate(model, data, bs, iters=40, generator=None):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = data.batch("val", bs, generator)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return tot / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fineweb", choices=["fineweb", "shakespeare"])
    ap.add_argument("--model", default="small", choices=list(PRESETS))
    ap.add_argument("--optimizer", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tokens", type=float, default=None, help="overrides --steps")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tie", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out-dir", default="results/lm")
    ap.add_argument("--opt-kwargs", default="", help='JSON, e.g. \'{"ns_steps":3}\'')
    args = ap.parse_args()

    cfg_model = PRESETS[args.model]
    cfg_model.ctx = args.ctx
    cfg_model.tie = args.tie
    tokens_per_step = args.batch_size * args.accum * args.ctx
    steps = args.steps or int(args.tokens / tokens_per_step)
    warmup = max(20, int(args.warmup_frac * steps))

    config = dict(task=f"lm_{args.data}", model=args.model, optimizer=args.optimizer,
                  lr=args.lr, wd=args.wd, seed=args.seed, steps=steps,
                  tokens=steps * tokens_per_step, batch=args.batch_size,
                  accum=args.accum, ctx=args.ctx, opt_kwargs=args.opt_kwargs)
    if already_done(args.out_dir, config):
        print("[skip] already in results.jsonl:", run_id_from(config))
        return

    set_seed(args.seed)
    data = BinData(args.data, args.ctx)
    model = GPT(cfg_model).to(DEVICE)
    print(f"params: {count_params(model)/1e6:.1f}M")
    model = maybe_compile(model, args.compile)

    opt_kw = parse_opt_kwargs(args.opt_kwargs)
    if args.optimizer == "sophia":
        opt_kw.setdefault("sophia_bs", tokens_per_step)
    optimizer, meta = build_optimizer(args.optimizer, model, args.lr, args.wd, **opt_kw)
    if meta["sf_train_eval"]:
        optimizer.train()
    lrs0 = base_lrs(optimizer)

    gen = torch.Generator().manual_seed(args.seed)
    logger = RunLogger(args.out_dir, run_id_from(config), config)
    torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None
    best_val, t_steps = float("inf"), []

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        if meta["use_schedule"]:
            apply_lr(optimizer, lrs0, cosine_with_warmup(step, steps, warmup))
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.accum):
            x, y = data.batch("train", args.batch_size, gen)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / args.accum).backward()
            loss_acc += loss.item() / args.accum
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(**meta["step_kwargs"])
        t_steps.append(time.perf_counter() - t0)

        # Sophia: Gauss-Newton-Bartlett Hessian estimate every k steps
        if meta["hessian_every"] and step % meta["hessian_every"] == meta["hessian_every"] - 1:
            x, _ = data.batch("train", args.batch_size, gen)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(x)
            samp = torch.distributions.Categorical(logits=logits.float()).sample()
            loss_h = F.cross_entropy(logits.float().view(-1, logits.size(-1)), samp.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss_h.backward()
            optimizer.update_hessian()
            optimizer.zero_grad(set_to_none=True)

        if step % args.eval_every == 0 or step == steps:
            if meta["sf_train_eval"]:
                optimizer.eval()
            val = evaluate(model, data, args.batch_size, generator=gen)
            if meta["sf_train_eval"]:
                optimizer.train()
            best_val = min(best_val, val)
            logger.log(step, f"{loss_acc:.4f}", f"{val:.4f}",
                       f"{optimizer.param_groups[0]['lr']:.2e}")
            print(f"step {step}/{steps} train {loss_acc:.4f} val {val:.4f} "
                  f"({1000*np.mean(t_steps[-50:]):.0f} ms/step)")
        elif step % 50 == 0:
            logger.log(step, f"{loss_acc:.4f}")

    logger.finish(final_val_loss=round(val, 4), best_val_loss=round(best_val, 4),
                  final_train_loss=round(loss_acc, 4),
                  ms_per_step=round(1000 * float(np.median(t_steps)), 1))


if __name__ == "__main__":
    main()
