"""Paper figures from results directories.

  1. curves     : val-loss (or metric) vs step at each optimizer's best LR
  2. lr_sens    : final metric vs LR per optimizer (robustness -- reviewers
                  increasingly require this; cf. AlgoPerf, Muon appendix)
  3. pareto     : final metric vs total wall-clock (preconditioner overhead)

Usage:
  python analysis/plots.py results/lm_sweep --metric best_val_loss --lower
  python analysis/plots.py results/vision --metric best_acc
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STYLE = {"adamw": "#444444", "sgdm": "#999999", "muon": "#d62728",
         "soap": "#ff7f0e", "shampoo": "#8c564b", "sophia": "#9467bd",
         "lion": "#bcbd22", "milo": "#1f77b4", "milo_m": "#17becf",
         "mion": "#2ca02c"}


def load(d):
    p = Path(d) / "results.jsonl"
    return pd.DataFrame([json.loads(line) for line in open(p)])


def best_lr(df, metric, lower):
    out = {}
    for opt, g in df.groupby("optimizer"):
        by = g.groupby("lr")[metric].mean()
        out[opt] = by.idxmin() if lower else by.idxmax()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--metric", default="best_val_loss")
    ap.add_argument("--lower", action="store_true",
                    help="lower metric is better (losses)")
    args = ap.parse_args()
    d = Path(args.result_dir)
    df = load(d)
    bests = best_lr(df, args.metric, args.lower)
    figdir = Path("results/figures"); figdir.mkdir(parents=True, exist_ok=True)
    tag = d.name

    # ---- 1. training curves at best LR (seed-averaged)
    plt.figure(figsize=(7.5, 4.8))
    for opt, lr in sorted(bests.items()):
        sel = df[(df.optimizer == opt) & (df.lr == lr)]
        curves = []
        for _, row in sel.iterrows():
            cp = d / f"curve_{row.run_id}.csv"
            if cp.exists():
                c = pd.read_csv(cp)
                c = c[c.val_metric.notna() & (c.val_metric != "")]
                curves.append(c.set_index("step")["val_metric"].astype(float))
        if not curves:
            continue
        avg = pd.concat(curves, axis=1).mean(axis=1)
        plt.plot(avg.index, avg.values, label=f"{opt} (lr={lr:g})",
                 color=STYLE.get(opt), lw=1.7)
    plt.xlabel("step"); plt.ylabel(args.metric.replace("best_", "val "))
    if args.lower:
        plt.yscale("log")
    plt.legend(fontsize=8); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(figdir / f"{tag}_curves.png", dpi=160); plt.close()

    # ---- 2. LR sensitivity
    plt.figure(figsize=(7.5, 4.8))
    for opt, g in df.groupby("optimizer"):
        by = g.groupby("lr")[args.metric].agg(["mean", "std"]).sort_index()
        plt.errorbar(by.index, by["mean"], yerr=by["std"].fillna(0),
                     marker="o", ms=4, capsize=3, label=opt,
                     color=STYLE.get(opt), lw=1.4)
    plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel(args.metric)
    plt.legend(fontsize=8); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(figdir / f"{tag}_lr_sensitivity.png", dpi=160); plt.close()

    # ---- 3. quality vs wall-clock pareto
    plt.figure(figsize=(6.5, 4.8))
    for opt, lr in bests.items():
        sel = df[(df.optimizer == opt) & (df.lr == lr)]
        plt.scatter(sel.total_wall_s / 3600, sel[args.metric],
                    label=opt, color=STYLE.get(opt), s=42)
    plt.xlabel("wall-clock (h)"); plt.ylabel(args.metric)
    plt.legend(fontsize=8); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(figdir / f"{tag}_pareto.png", dpi=160); plt.close()
    print(f"wrote 3 figures to {figdir}/ ({tag}_*)")


if __name__ == "__main__":
    main()
