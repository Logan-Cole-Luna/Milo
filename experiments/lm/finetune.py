"""
Full fine-tuning of a 1B+ pretrained LLM (Qwen2.5-1.5B) on instruction data
(Alpaca), comparing optimizers. This is where optimizer-state memory matters most:
full FT of 1.5B params means Adam's two moment buffers alone cost ~12 GB, vs MION's
single buffer ~3 GB — the headline accuracy/resource trade-off at scale.

Reports val loss vs steps/tokens, tokens/sec, optimizer-state & peak memory.

  python -m experiments.lm.finetune --optimizer MION --lr 1e-4 --steps 600
"""
import argparse, json, os, sys, time, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.lm.config import build_optimizer, OPTIMIZER_PARAMS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.expanduser("~/scratch/models/qwen2.5-1.5b")
ALPACA = os.path.expanduser("~/scratch/datasets/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
# fine-tuning LR defaults (smaller than pretraining); per-optimizer
FT_LR = {
    "ADAMW": 3.91e-05,
    "ADAM_MINI": 3.91e-05,
    "LION": 1.21e-05,
    "SGD": 0.00797,
    "MUON": 0.000158,
    "SHAMPOO": 0.00797,
    "MILO": 8.73e-05,
    "MILO_LW": 8.73e-05,
    "MILOM": 1.31e-05,
    "MION": 2.05e-05,
}


def fmt_example(ex):
    instr, inp, out = ex["instruction"], ex.get("input", ""), ex["output"]
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{instr}\n\n### Response:\n{out}"


def load_data(tokenizer, max_len, n=None):
    import pyarrow.parquet as pq
    rows = pq.read_table(ALPACA).to_pylist()
    if n: rows = rows[:n]
    texts = [fmt_example(r) for r in rows]
    enc = tokenizer(texts, max_length=max_len, truncation=True, padding="max_length",
                    return_tensors="pt")
    ids = enc["input_ids"]
    labels = ids.clone()
    labels[enc["attention_mask"] == 0] = -100        # ignore pad in loss
    n_val = max(64, int(0.03 * len(ids)))
    return (ids[:-n_val], labels[:-n_val], enc["attention_mask"][:-n_val],
            ids[-n_val:], labels[-n_val:], enc["attention_mask"][-n_val:])


def opt_state_mb(opt):
    return sum(v.numel() * v.element_size() for st in opt.state.values()
               for v in st.values() if torch.is_tensor(v)) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-examples", type=int, default=None)
    ap.add_argument("--out-dir", default="results/ft1b")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    model.gradient_checkpointing_enable(); model.config.use_cache = False

    nparams = sum(p.numel() for p in model.parameters())
    lr = args.lr if args.lr is not None else FT_LR.get(args.optimizer.upper(), 1e-4)
    opt = build_optimizer(args.optimizer, model, lr, OPTIMIZER_PARAMS.get(args.optimizer.upper(), {}))
    print(f"FT {args.optimizer} | Qwen2.5 ({nparams/1e9:.2f}B) | lr={lr:g} | {args.steps} steps", flush=True)

    trX, trY, trM, vaX, vaY, vaM = load_data(tok, args.max_len, args.n_examples)
    def batch(X, Y, M, bs):
        i = torch.randint(0, len(X), (bs,))
        return X[i].to(DEVICE), Y[i].to(DEVICE), M[i].to(DEVICE)

    @torch.no_grad()
    def evaluate(iters=20):
        model.eval(); ls = []
        for k in range(iters):
            j = slice((k*16) % max(1, len(vaX)-16), (k*16) % max(1, len(vaX)-16) + 16)
            out = model(input_ids=vaX[j].to(DEVICE), attention_mask=vaM[j].to(DEVICE), labels=vaY[j].to(DEVICE))
            ls.append(out.loss.item())
        model.train(); return float(np.mean(ls))

    def cos_lr(s):
        if s < int(args.warmup_frac*args.steps): return lr*(s+1)/max(1,int(args.warmup_frac*args.steps))
        pr = (s-int(args.warmup_frac*args.steps))/max(1,args.steps-int(args.warmup_frac*args.steps))
        return 0.1*lr + 0.9*lr*0.5*(1+math.cos(math.pi*pr))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log = {"optimizer": args.optimizer, "lr": lr, "model": "qwen2.5-1.5b",
           "n_params_b": round(nparams/1e9, 3), "curve": []}
    t0 = time.time(); model.train(); step_times = []
    tok_per_step = args.batch_size * args.grad_accum * args.max_len

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cos_lr(step) * g.get("lr_mult", 1.0)
        st = time.time(); opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y, mm = batch(trX, trY, trM, args.batch_size)
            out = model(input_ids=x, attention_mask=mm, labels=y)
            (out.loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if DEVICE == "cuda": torch.cuda.synchronize()
        step_times.append(time.time()-st)
        if step % args.eval_every == 0 or step == args.steps-1:
            vl = evaluate()
            log["curve"].append({"step": step, "val_loss": vl, "secs": round(time.time()-t0,1)})
            print(f"  step {step:4d} | val {vl:.4f} | {(time.time()-t0)/60:.1f} min | "
                  f"{tok_per_step/np.mean(step_times[-20:]):.0f} tok/s", flush=True)

    log["final_val_loss"] = log["curve"][-1]["val_loss"]
    log["tokens_per_sec"] = float(tok_per_step/np.mean(step_times))
    log["opt_state_mb"] = round(opt_state_mb(opt), 1)
    log["peak_mem_mb"] = round(torch.cuda.max_memory_allocated()/1e6, 1) if DEVICE=="cuda" else None
    out = Path(args.out_dir)/f"ft_{args.optimizer.lower()}_seed{args.seed}.json"
    json.dump(log, open(out, "w"), indent=2)
    print(f"\n✓ {args.optimizer}: final val {log['final_val_loss']:.4f} | "
          f"opt-state {log['opt_state_mb']:.0f}MB | peak {log['peak_mem_mb']:.0f}MB -> {out}", flush=True)


if __name__ == "__main__":
    main()
