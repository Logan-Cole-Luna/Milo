"""Aggregate results.jsonl files into reviewer-ready tables.

For each (task, model, optimizer): selects the best LR by the task's primary
metric, then reports mean +/- std across seeds at that LR, plus ms/step and
peak memory. Emits markdown and LaTeX (booktabs).

Usage:
  python analysis/aggregate.py results/lm_sweep results/lm_final results/vision
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# primary metric per task prefix: (column, lower_is_better)
METRICS = {"lm": ("best_val_loss", True), "cifar10": ("best_acc", False),
           "glue": ("best_metric", False)}


def metric_for(task):
    for k, v in METRICS.items():
        if task.startswith(k):
            return v
    return ("final_train_loss", True)


def load(dirs):
    rows = []
    for d in dirs:
        p = Path(d) / "results.jsonl"
        if not p.exists():
            print(f"warning: {p} not found")
            continue
        rows += [json.loads(line) for line in open(p)]
    return pd.DataFrame(rows)


def main(dirs):
    df = load(dirs)
    if df.empty:
        sys.exit("no results found")
    out = []
    for (task, model), g in df.groupby(["task", "model"]):
        col, lower = metric_for(task)
        if col not in g:
            continue
        for opt, go in g.groupby("optimizer"):
            # best lr by mean metric across seeds (ablation variants keyed by opt_kwargs)
            kw_col = (go["opt_kwargs"].fillna("") if "opt_kwargs" in go
                      else pd.Series("", index=go.index))
            for kwargs, gk in go.groupby(kw_col):
                by_lr = gk.groupby("lr")[col].mean()
                best_lr = by_lr.idxmin() if lower else by_lr.idxmax()
                sel = gk[gk.lr == best_lr]
                out.append(dict(
                    task=task, model=model,
                    optimizer=opt + (f" [{kwargs}]" if kwargs else ""),
                    best_lr=best_lr, n_seeds=len(sel),
                    metric=f"{sel[col].mean():.4f} ± {sel[col].std():.4f}"
                           if len(sel) > 1 else f"{sel[col].mean():.4f}",
                    ms_per_step=round(sel["ms_per_step"].mean(), 1),
                    peak_mem_gb=round(sel["peak_mem_gb"].max(), 2),
                ))
    table = pd.DataFrame(out).sort_values(["task", "model", "metric"])
    print(table.to_markdown(index=False))
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv("results/tables/summary.csv", index=False)
    with open("results/tables/summary.tex", "w") as f:
        f.write(table.to_latex(index=False))
    print("\nwrote results/tables/summary.{csv,tex}")

    # paired bootstrap significance vs adamw at best LR (same seeds)
    print("\nSignificance vs AdamW (paired bootstrap over seeds, 95% CI of diff):")
    for (task, model), g in df.groupby(["task", "model"]):
        col, lower = metric_for(task)
        if col not in g or "adamw" not in set(g.optimizer):
            continue
        ga = g[g.optimizer == "adamw"]
        base_lr = ga.groupby("lr")[col].mean()
        base_lr = base_lr.idxmin() if lower else base_lr.idxmax()
        base = ga[ga.lr == base_lr].sort_values("seed")[col].values
        for opt, go in g[g.optimizer != "adamw"].groupby("optimizer"):
            by_lr = go.groupby("lr")[col].mean()
            blr = by_lr.idxmin() if lower else by_lr.idxmax()
            x = go[go.lr == blr].sort_values("seed")[col].values
            n = min(len(x), len(base))
            if n < 2:
                continue
            diffs = x[:n] - base[:n]
            boots = [np.mean(np.random.choice(diffs, n)) for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  {task}/{model} {opt:12s} Δ{col}={np.mean(diffs):+.4f} "
                  f"CI[{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/lm_sweep"])
