"""
Controlled optimizer cost benchmark: optimizer-state memory, peak GPU memory,
throughput (tok/s), and per-step latency — same model, fixed step count, all
optimizers. This is the resource half of the accuracy/resource tradeoff, and
plays to MION/Muon's memory advantage (one momentum buffer vs Adam's two states).

Usage:
  python -m benchmarks.optimizer_cost --model small --steps 30
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.lm.model import GPT                       # noqa: E402
from experiments.lm.config import MODEL_PRESETS, OPTIMIZER_PARAMS, OPTIMIZERS, build_optimizer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def opt_state_mb(opt):
    tot = sum(v.numel() * v.element_size()
              for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
    return tot / 1e6


def bench_one(name, cfg, batch, ctx, steps):
    torch.manual_seed(0)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    model = GPT(cfg).to(DEVICE)
    opt = build_optimizer(name, model, 1e-3, OPTIMIZER_PARAMS.get(name.upper(), {}))
    x = torch.randint(0, cfg.vocab_size, (batch, ctx), device=DEVICE)
    y = torch.randint(0, cfg.vocab_size, (batch, ctx), device=DEVICE)
    times = []
    for i in range(steps + 3):                              # 3 warmup
        t = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        if i >= 3:
            times.append(time.time() - t)
    sec = float(np.mean(times))
    return {
        "optimizer": name,
        "opt_state_mb": round(opt_state_mb(opt), 1),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if DEVICE == "cuda" else None,
        "sec_per_step": round(sec, 4),
        "tokens_per_sec": round(batch * ctx / sec, 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=list(MODEL_PRESETS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out-dir", default="results/cost")
    args = ap.parse_args()

    cfg = MODEL_PRESETS[args.model]; cfg.block_size = args.ctx
    nparams = GPT(cfg).num_params() / 1e6
    print(f"Cost benchmark | {args.model} (~{nparams:.0f}M) | batch {args.batch} × ctx {args.ctx} "
          f"| {args.steps} steps | {DEVICE}\n")
    rows = []
    for name in OPTIMIZERS:
        try:
            r = bench_one(name, cfg, args.batch, args.ctx, args.steps)
            rows.append(r)
            print(f"  {name:<10} state {r['opt_state_mb']:>7}MB | peak {r['peak_mem_mb']}MB | "
                  f"{r['sec_per_step']*1000:6.1f} ms/step | {r['tokens_per_sec']:.0f} tok/s", flush=True)
        except Exception as e:
            print(f"  {name:<10} FAILED: {type(e).__name__}: {e}", flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir) / f"cost_{args.model}.json"
    json.dump({"model": args.model, "n_params_m": round(nparams, 1), "rows": rows}, open(out, "w"), indent=2)
    print(f"\n✓ -> {out}")


if __name__ == "__main__":
    main()
