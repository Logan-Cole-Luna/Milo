"""
From-scratch LM pretraining for optimizer benchmarking.

Reports the metrics modern optimizer papers expect:
  * val loss vs tokens          (sample efficiency)
  * val loss vs wall-clock      (real speed, incl. per-step overhead)
  * tokens / steps / time to a target val loss   (speedup headline)
  * throughput (tok/s) and optimizer-state memory

Usage:
  python -m experiments.lm.train --model small --optimizer MION --lr 0.02 \
      --tokens 1.2e9 --batch-size 32 --grad-accum 8 --ctx 1024 --seed 1
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.lm.model import GPT                       # noqa: E402
from experiments.lm.config import MODEL_PRESETS, OPTIMIZER_PARAMS, LEARNING_RATES, build_optimizer  # noqa: E402
from experiments.lm.data import FineWebData                # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    import math
    prog = (step - warmup) / max(1, total - warmup)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, val, iters=40):
    model.eval(); losses = []
    for _ in range(iters):
        x, y = val.batch()
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def optimizer_state_mb(opt):
    tot = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
    return tot / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=list(MODEL_PRESETS))
    ap.add_argument("--optimizer", required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tokens", default="1.2e9", help="total training tokens")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--target-val-loss", type=float, default=3.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="results/lm")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    cfg = MODEL_PRESETS[args.model]; cfg.block_size = args.ctx
    model = GPT(cfg).to(DEVICE)
    if args.compile and DEVICE == "cuda":
        model = torch.compile(model)

    lr = args.lr if args.lr is not None else LEARNING_RATES.get(args.optimizer.upper(), 1e-3)
    params = dict(OPTIMIZER_PARAMS.get(args.optimizer.upper(), {}))
    if args.weight_decay is not None and "weight_decay" in params:
        params["weight_decay"] = args.weight_decay
    opt = build_optimizer(args.optimizer, model, lr, params)

    tokens_per_step = args.batch_size * args.grad_accum * args.ctx
    total_steps = int(float(args.tokens) / tokens_per_step)
    warmup = int(args.warmup_frac * total_steps)

    train = FineWebData("train", args.ctx, args.batch_size, DEVICE)
    val = FineWebData("val", args.ctx, args.batch_size, DEVICE)

    n_params = model.num_params() if hasattr(model, "num_params") else \
        sum(p.numel() for p in model.parameters())
    print(f"LM pretrain | {args.model} (~{n_params/1e6:.0f}M) | {args.optimizer} lr={lr:g} "
          f"| {total_steps} steps × {tokens_per_step} tok = {float(args.tokens):.2e} tokens", flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log = {"optimizer": args.optimizer, "model": args.model, "lr": lr, "seed": args.seed,
           "tokens_per_step": tokens_per_step, "curve": [],
           "tokens_to_target": None, "steps_to_target": None, "secs_to_target": None,
           "target_val_loss": args.target_val_loss}
    t0 = time.time(); model.train(); step_times = []

    for step in range(total_steps):
        _lr_now = cosine_lr(step, total_steps, warmup, lr)
        for g in opt.param_groups:
            if "lr" in g:
                g["lr"] = _lr_now * g.get("lr_mult", 1.0)   # per-group LR multiplier (e.g. embeddings)
        st = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = train.batch()
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - st)

        if step % args.eval_every == 0 or step == total_steps - 1:
            vl = evaluate(model, val)
            toks = (step + 1) * tokens_per_step
            elapsed = time.time() - t0
            log["curve"].append({"step": step, "tokens": toks, "val_loss": vl,
                                  "secs": round(elapsed, 1)})
            print(f"  step {step:5d} | tok {toks/1e6:6.1f}M | val {vl:.4f} | "
                  f"{elapsed/60:5.1f} min | {tokens_per_step/np.mean(step_times[-50:]):.0f} tok/s",
                  flush=True)
            if log["tokens_to_target"] is None and vl <= args.target_val_loss:
                log["tokens_to_target"] = toks
                log["steps_to_target"] = step
                log["secs_to_target"] = round(elapsed, 1)
                print(f"  ★ reached target val {args.target_val_loss} at {toks/1e6:.0f}M tokens", flush=True)

    log["final_val_loss"] = log["curve"][-1]["val_loss"]
    log["tokens_per_sec"] = float(tokens_per_step / np.mean(step_times))
    log["sec_per_step"] = float(np.mean(step_times))
    log["opt_state_mb"] = round(optimizer_state_mb(opt), 1)
    log["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1) if DEVICE == "cuda" else None
    log["n_params_m"] = round(n_params / 1e6, 1)

    out = Path(args.out_dir) / f"lm_{args.model}_{args.optimizer.lower()}_seed{args.seed}.json"
    json.dump(log, open(out, "w"), indent=2)
    print(f"\n✓ final val {log['final_val_loss']:.4f} | {log['tokens_per_sec']:.0f} tok/s | "
          f"opt-state {log['opt_state_mb']}MB | -> {out}", flush=True)


if __name__ == "__main__":
    main()
