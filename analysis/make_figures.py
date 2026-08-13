"""
Generate paper-ready tables (LaTeX) and figures from the results/ tree.

Outputs:
  results/paper/tables/{lm,vision_converged,cost}.tex
  results/paper/figures/{lm_loss_vs_tokens,lm_loss_vs_walltime,
                         acc_vs_memory,mion_spectral_ablation}.png

Read-only aggregation + matplotlib (no training). Run anywhere.
"""
import json, glob, csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/logan03/Milo")
OUT = ROOT / "results/paper"; (OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
HILITE = {"MION", "MILO", "MILO_LW", "MILOM", "MION_ADAM", "MION_NOR", "MION_GATED"}   # our family


def load_lm():
    rows = [json.load(open(f)) for f in glob.glob(str(ROOT / "results/lm/*.json"))]
    return sorted(rows, key=lambda d: d["final_val_loss"])


def load_cost():
    fs = glob.glob(str(ROOT / "results/cost/*.json"))
    return json.load(open(fs[0])) if fs else None


def vision_converged():
    opts = ["MION","MION_NOR","MILOM","MILO","MILO_LW","MUON","SOAP","ADAMW","ADAM_MINI","LION",
            "SGD","ADAGRAD","RMSPROP_MOMENTUM","SHAMPOO"]
    exps = ["resnet34_cifar10","vgg11_cifar10","vit_tiny_cifar10"]
    out = {}
    for opt in opts:
        row = {}
        for e in exps:
            f = ROOT / f"experiments/vision/{e}/results_converged_{opt.lower()}/{e}_final_test_results.csv"
            v = None
            if f.exists():
                for r in csv.DictReader(open(f)):
                    o = r.get("optimizer") or list(r.values())[0]
                    if o.upper() == opt:
                        v = next((float(x) for k, x in r.items() if "acc" in k.lower() and x not in ("","nan")), None)
            row[e] = v
        out[opt] = row
    return out, exps


# ---------------- LaTeX tables ----------------
def tex_lm(rows):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Optimizer & Val loss & Tokens$\to$3.3 & tok/s & State (MB) \\", r"\midrule"]
    for d in rows:
        t = d.get("tokens_to_target"); t = f"{t/1e6:.0f}M" if t else "--"
        nm = d["optimizer"].replace("_", r"\_")
        nm = rf"\textbf{{{nm}}}" if d["optimizer"] in HILITE else nm
        lines.append(f"{nm} & {d['final_val_loss']:.3f} & {t} & {d['tokens_per_sec']:.0f} & {d['opt_state_mb']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables/lm.tex").write_text("\n".join(lines))


def tex_vision(vc, exps):
    hdr = " & ".join(["Optimizer"] + [e.replace("_cifar","").replace("_tiny","").upper() for e in exps])
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", hdr + r" \\", r"\midrule"]
    for opt, row in vc.items():
        nm = opt.replace("_", r"\_"); nm = rf"\textbf{{{nm}}}" if opt in HILITE else nm
        vals = " & ".join(f"{row[e]:.1f}" if isinstance(row[e], float) else "--" for e in exps)
        lines.append(f"{nm} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables/vision_converged.tex").write_text("\n".join(lines))


def tex_cost(cost):
    if not cost: return
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Optimizer & State (MB) & ms/step & tok/s \\", r"\midrule"]
    for r in sorted(cost["rows"], key=lambda x: x["opt_state_mb"]):
        nm = r["optimizer"].replace("_", r"\_"); nm = rf"\textbf{{{nm}}}" if r["optimizer"] in HILITE else nm
        lines.append(f"{nm} & {r['opt_state_mb']:.0f} & {r['sec_per_step']*1000:.0f} & {r['tokens_per_sec']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables/cost.tex").write_text("\n".join(lines))


# ---------------- figures ----------------
def fig_lm_curves(rows):
    for xkey, xlabel, fname in [("tokens", "Tokens", "lm_loss_vs_tokens"),
                                ("secs", "Wall-clock (s)", "lm_loss_vs_walltime")]:
        plt.figure(figsize=(6, 4.2))
        for d in rows:
            c = d["curve"]
            xs = [p[xkey] for p in c]; ys = [p["val_loss"] for p in c]
            lw = 2.4 if d["optimizer"] in HILITE else 1.0
            z = 5 if d["optimizer"] in HILITE else 1
            plt.plot(xs, ys, label=d["optimizer"], linewidth=lw, zorder=z,
                     alpha=1.0 if d["optimizer"] in HILITE else 0.6)
        plt.xlabel(xlabel); plt.ylabel("Val loss"); plt.ylim(3.0, 4.5)
        plt.title("LM pretraining (FineWeb-Edu, 124M)")
        plt.legend(fontsize=7, ncol=2); plt.tight_layout()
        plt.savefig(OUT / f"figures/{fname}.png", dpi=150); plt.close()


def fig_acc_vs_memory(rows):
    # MION's headline: low memory + competitive loss
    plt.figure(figsize=(6, 4.2))
    for d in rows:
        x = d["opt_state_mb"]; y = d["final_val_loss"]
        c = "tab:red" if d["optimizer"] in HILITE else "tab:gray"
        plt.scatter(x, y, c=c, s=70 if d["optimizer"] in HILITE else 40, zorder=5)
        plt.annotate(d["optimizer"], (x, y), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    plt.xlabel("Optimizer-state memory (MB)"); plt.ylabel("Final val loss")
    plt.title("Accuracy vs. memory (lower-left = better)")
    plt.tight_layout(); plt.savefig(OUT / "figures/acc_vs_memory.png", dpi=150); plt.close()


def fig_spectral_ablation():
    # MION spectral ON vs OFF across tasks (from ablation results)
    def acc_vis(tag, exp):
        f = ROOT / f"experiments/vision/{exp}/results_ablation_vision_{tag}/{exp}_final_test_results.csv"
        if not f.exists(): return None
        for r in csv.DictReader(open(f)):
            if (r.get("optimizer") or list(r.values())[0]).upper() == "MION":
                return next((float(x) for k, x in r.items() if "acc" in k.lower() and x not in ("","nan")), None)
    def acc_inet(tag):
        f = ROOT / f"results/mion_ablation/results_ablation_imagenet_{tag}/imagenet_100_ResNet34_mion.json"
        if not f.exists(): return None
        return float(np.mean([r["best_val_accuracy"] for r in json.load(open(f))]))
    tasks = ["ResNet34", "ViT", "ImageNet"]
    on = [acc_vis("spec_on","resnet34_cifar10"), acc_vis("spec_on","vit_tiny_cifar10"), acc_inet("spec_on")]
    off = [acc_vis("spec_off","resnet34_cifar10"), acc_vis("spec_off","vit_tiny_cifar10"), acc_inet("spec_off")]
    if any(v is None for v in on+off):
        print("  (spectral ablation data incomplete, skipping fig)"); return
    x = np.arange(len(tasks)); w = 0.35
    plt.figure(figsize=(6, 4.2))
    plt.bar(x - w/2, on, w, label="spectral ON (MION)", color="tab:red")
    plt.bar(x + w/2, off, w, label="spectral OFF (group-std)", color="tab:gray")
    plt.xticks(x, tasks); plt.ylabel("Accuracy (%)")
    plt.title("MION ablation: Newton-Schulz spectral path")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "figures/mion_spectral_ablation.png", dpi=150); plt.close()


if __name__ == "__main__":
    rows = load_lm()
    tex_lm(rows); fig_lm_curves(rows); fig_acc_vs_memory(rows)
    vc, exps = vision_converged(); tex_vision(vc, exps)
    cost = load_cost(); tex_cost(cost)
    fig_spectral_ablation()
    print("✓ tables ->", OUT / "tables")
    for p in sorted(glob.glob(str(OUT / "tables/*.tex"))): print("   ", os.path.basename(p))
    print("✓ figures ->", OUT / "figures")
    for p in sorted(glob.glob(str(OUT / "figures/*.png"))): print("   ", os.path.basename(p))
