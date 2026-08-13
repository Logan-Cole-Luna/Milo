"""
Generate figures for EVERY experiment into results/paper/figures/.

Covers: LM pretraining (loss-vs-tokens/walltime, acc-vs-memory, tokens-to-target,
throughput), cost benchmark, NLP finals, ImageNet finals, vision (5-epoch + 60-
epoch converged), MION ablations (ns_steps/rms_target/scale_factor/spectral),
controlled MION-vs-Muon, and the Optuna LR sweeps. Read-only + matplotlib.
"""
import json, glob, csv, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/home/logan03/Milo")
FIG = R / "results/paper/figures"; FIG.mkdir(parents=True, exist_ok=True)
FAM = {"MION", "MILO", "MILO_LW", "MILOM", "MION_ADAM", "MION_NOR", "MION_GATED"}
def col(o): return "tab:red" if o in FAM else "tab:blue"


def csv_acc(path, opt):
    if not os.path.exists(path): return None
    for r in csv.DictReader(open(path)):
        o = (r.get("optimizer") or list(r.values())[0]).upper()
        if o == opt:
            return next((float(v) for k, v in r.items() if "acc" in k.lower() and v not in ("", "nan")), None)
    return None

def jmean(path, key="best_val_accuracy"):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    return float(np.mean([r[key] for r in d])) if isinstance(d, list) else None

def barh(title, labels, vals, fname, xlabel, ann="%.1f"):
    order = np.argsort(vals)
    labels = [labels[i] for i in order]; vals = [vals[i] for i in order]
    plt.figure(figsize=(6, max(3, 0.4*len(labels))))
    plt.barh(range(len(labels)), vals, color=[col(l) for l in labels])
    plt.yticks(range(len(labels)), labels, fontsize=8)
    for i, v in enumerate(vals): plt.text(v, i, " "+ann % v, va="center", fontsize=7)
    plt.xlabel(xlabel); plt.title(title); plt.tight_layout()
    plt.savefig(FIG / fname, dpi=150); plt.close()


# ---------- LM ----------
def lm():
    rows = [json.load(open(f)) for f in glob.glob(str(R/"results/lm/*.json"))]
    if not rows: return
    rows.sort(key=lambda d: d["final_val_loss"])
    for xk, xl, fn in [("tokens","Tokens","lm_loss_vs_tokens"),("secs","Wall-clock (s)","lm_loss_vs_walltime")]:
        plt.figure(figsize=(6.2,4.3))
        for d in rows:
            c=d["curve"]; lwf=d["optimizer"] in FAM
            plt.plot([p[xk] for p in c],[p["val_loss"] for p in c],label=d["optimizer"],
                     lw=2.4 if lwf else 1.0, alpha=1 if lwf else .55, zorder=5 if lwf else 1)
        plt.xlabel(xl); plt.ylabel("Val loss"); plt.ylim(3.0,4.6)
        plt.title("LM pretraining (FineWeb-Edu, 124M)"); plt.legend(fontsize=7,ncol=2)
        plt.tight_layout(); plt.savefig(FIG/f"{fn}.png",dpi=150); plt.close()
    # acc vs memory
    plt.figure(figsize=(6,4.3))
    for d in rows:
        plt.scatter(d["opt_state_mb"], d["final_val_loss"], c=col(d["optimizer"]),
                    s=80 if d["optimizer"] in FAM else 40, zorder=5)
        plt.annotate(d["optimizer"], (d["opt_state_mb"], d["final_val_loss"]), fontsize=7,
                     xytext=(4,3), textcoords="offset points")
    plt.xlabel("Optimizer-state memory (MB)"); plt.ylabel("Final val loss")
    plt.title("LM: accuracy vs memory (lower-left best)"); plt.tight_layout()
    plt.savefig(FIG/"lm_acc_vs_memory.png",dpi=150); plt.close()
    # tokens-to-target
    tt=[(d["optimizer"], d["tokens_to_target"]/1e6) for d in rows if d.get("tokens_to_target")]
    if tt: barh("LM: tokens to reach val 3.3 (lower=better)", [a for a,_ in tt],[b for _,b in tt],
               "lm_tokens_to_target.png","Tokens (M)","%.0f")
    barh("LM: throughput", [d["optimizer"] for d in rows],[d["tokens_per_sec"] for d in rows],
         "lm_throughput.png","tokens/sec","%.0f")


# ---------- cost ----------
def cost():
    fs=glob.glob(str(R/"results/cost/*.json"))
    if not fs: return
    d=json.load(open(fs[0])); rows=d["rows"]
    barh(f"Optimizer-state memory ({d['model']}, ~{d['n_params_m']:.0f}M)",
         [r["optimizer"] for r in rows],[r["opt_state_mb"] for r in rows],
         "cost_memory.png","State memory (MB)","%.0f")
    plt.figure(figsize=(6,4.3))
    for r in rows:
        plt.scatter(r["tokens_per_sec"], r["opt_state_mb"], c=col(r["optimizer"]),
                    s=80 if r["optimizer"] in FAM else 40, zorder=5)
        plt.annotate(r["optimizer"],(r["tokens_per_sec"],r["opt_state_mb"]),fontsize=7,
                     xytext=(4,3),textcoords="offset points")
    plt.xlabel("tokens/sec"); plt.ylabel("State memory (MB)")
    plt.title("Cost: throughput vs memory"); plt.tight_layout()
    plt.savefig(FIG/"cost_throughput_vs_memory.png",dpi=150); plt.close()


# ---------- NLP / ImageNet finals ----------
def nlp():
    fs=glob.glob(str(R/"results/finals/results_final_nlp/bert_sst2_*.json"))
    if not fs: return
    labels=[]; vals=[]
    for f in fs:
        d=json.load(open(f)); labels.append(d[0]["optimizer"]); vals.append(np.mean([r["best_val_accuracy"] for r in d]))
    barh("NLP — BERT/SST-2 (val acc, 5 seeds)", labels, vals, "nlp_finals.png","Val accuracy (%)")

def imagenet():
    fs=glob.glob(str(R/"results/finals/results_final_imagenet/imagenet_100_*.json"))
    if not fs: return
    labels=[]; vals=[]
    for f in fs:
        d=json.load(open(f)); labels.append(d[0]["optimizer"]); vals.append(np.mean([r["best_val_accuracy"] for r in d]))
    barh("ImageNet — ResNet34/Tiny-ImageNet-200 (val acc, 3 seeds)", labels, vals,
         "imagenet_finals.png","Val accuracy (%)")


# ---------- vision heatmaps (5-epoch + converged) ----------
def vision_heatmap(kind):
    opts=["MION","MION_NOR","MILOM","MILO","MILO_LW","MUON","SOAP","ADAMW","ADAM_MINI","LION","SGD","ADAGRAD","RMSPROP_MOMENTUM","SHAMPOO"]
    if kind=="converged":
        exps=["resnet34_cifar10","vgg11_cifar10","vit_tiny_cifar10"]
        getter=lambda e,o: csv_acc(R/f"experiments/vision/{e}/results_converged_{o.lower()}/{e}_final_test_results.csv", o)
        title="Converged vision (60 epochs, tuned LR) — test acc"
    else:
        exps=["logistic","multilayer","resnet34_cifar10","resnet34_cifar100","vgg11_cifar10","vgg11_cifar100","vit_tiny_cifar10","vit_tiny_cifar100"]
        getter=lambda e,o: csv_acc(R/f"experiments/vision/{e}/results_nt/{e}_final_test_results.csv", o)
        title="Vision (5 epochs) — test acc"
    M=np.full((len(opts),len(exps)),np.nan)
    for i,o in enumerate(opts):
        for j,e in enumerate(exps):
            v=getter(e,o)
            if v is not None: M[i,j]=v
    plt.figure(figsize=(1.1*len(exps)+2, 0.45*len(opts)+1.5))
    im=plt.imshow(M, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="test acc (%)")
    plt.xticks(range(len(exps)),[e.replace("_cifar","").replace("_tiny","") for e in exps],rotation=40,ha="right",fontsize=7)
    plt.yticks(range(len(opts)),opts,fontsize=8)
    for i in range(len(opts)):
        for j in range(len(exps)):
            if not np.isnan(M[i,j]): plt.text(j,i,f"{M[i,j]:.0f}",ha="center",va="center",fontsize=6,color="w")
    plt.title(title); plt.tight_layout()
    plt.savefig(FIG/f"vision_{kind}_heatmap.png",dpi=150); plt.close()


# ---------- MION ablations ----------
def ablations():
    def vis(tag,e): return csv_acc(R/f"experiments/vision/{e}/results_ablation_vision_{tag}/{e}_final_test_results.csv","MION")
    def inet(tag):
        f=R/f"results/mion_ablation/results_ablation_imagenet_{tag}/imagenet_100_ResNet34_mion.json"
        return jmean(f) if f.exists() else None
    axes={
      "ns_steps":[("ns1",1),("ns3",3),("ns5",5),("ns7",7)],
      "rms_target":[("rms0p1",.1),("rms0p2",.2),("rms0p3",.3),("rms0p5",.5)],
      "scale_factor":[("sf0p0",0),("sf0p1",.1),("sf0p2",.2)],
    }
    for ax,cells in axes.items():
        plt.figure(figsize=(5.5,4))
        for e,lab in [("resnet34_cifar10","ResNet34"),("vit_tiny_cifar10","ViT")]:
            xs=[v for _,v in cells]; ys=[vis(t,e) for t,_ in cells]
            plt.plot(xs,ys,"o-",label=lab)
        xs=[v for _,v in cells]; yi=[inet(t) for t,_ in cells]
        if all(v is not None for v in yi): plt.plot(xs,yi,"s--",label="ImageNet")
        plt.xlabel(ax); plt.ylabel("test/val acc (%)"); plt.title(f"MION ablation: {ax}")
        plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG/f"mion_abl_{ax}.png",dpi=150); plt.close()
    # spectral on/off
    tasks=["ResNet34","ViT","ImageNet"]
    on=[vis("spec_on","resnet34_cifar10"),vis("spec_on","vit_tiny_cifar10"),inet("spec_on")]
    off=[vis("spec_off","resnet34_cifar10"),vis("spec_off","vit_tiny_cifar10"),inet("spec_off")]
    if all(v is not None for v in on+off):
        x=np.arange(3); w=.35
        plt.figure(figsize=(5.5,4))
        plt.bar(x-w/2,on,w,label="spectral ON (MION)",color="tab:red")
        plt.bar(x+w/2,off,w,label="spectral OFF (group-std)",color="tab:gray")
        plt.xticks(x,tasks); plt.ylabel("acc (%)"); plt.title("MION: Newton-Schulz spectral path")
        plt.legend(); plt.tight_layout(); plt.savefig(FIG/"mion_abl_spectral.png",dpi=150); plt.close()


# ---------- controlled MION vs Muon-aux ----------
def controlled():
    auxlrs=["1e-4","3e-4","1e-3","3e-3"]
    def vis(e,a): return csv_acc(R/f"experiments/vision/{e}/results_mvm_vision_auxlr{a}/{e}_final_test_results.csv","MION")
    def inet(a):
        f=R/f"results/controlled/results_mvm_imagenet_auxlr{a}/imagenet_100_ResNet34_mion.json"
        return jmean(f) if f.exists() else None
    base={"ResNet34":csv_acc(R/"experiments/vision/resnet34_cifar10/results_ablation_vision_spec_on/resnet34_cifar10_final_test_results.csv","MION"),
          "ViT":csv_acc(R/"experiments/vision/vit_tiny_cifar10/results_ablation_vision_spec_on/vit_tiny_cifar10_final_test_results.csv","MION"),
          "ImageNet":jmean(R/"results/mion_ablation/results_ablation_imagenet_spec_on/imagenet_100_ResNet34_mion.json")}
    plt.figure(figsize=(6,4.2))
    series={"ResNet34":[vis("resnet34_cifar10",a) for a in auxlrs],
            "ViT":[vis("vit_tiny_cifar10",a) for a in auxlrs],
            "ImageNet":[inet(a) for a in auxlrs]}
    xs=range(len(auxlrs))
    for k,ys in series.items():
        if all(v is not None for v in ys):
            line,=plt.plot(xs,ys,"o-",label=f"{k} (adam-aux)")
            if base[k] is not None: plt.axhline(base[k],ls=":",color=line.get_color(),alpha=.7)
    plt.xticks(list(xs),auxlrs); plt.xlabel("Adam-aux LR (dotted = MION group-std, 1 LR)")
    plt.ylabel("test/val acc (%)"); plt.title("Controlled: MION group-std vs Adam-aux")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG/"controlled_mion_vs_muon.png",dpi=150); plt.close()


# ---------- Optuna sweep example curves ----------
def optuna():
    # best LR per optimizer, per domain (one bar fig per domain)
    studies={}
    for f in glob.glob(str(R/"results/optuna/*.json")) + glob.glob(str(R/"results_optuna/*.json")):
        if "_tmp" in f: continue
        d=json.load(open(f)); tag=d.get("tag")
        if not tag or "best_lr" not in d: continue
        dom=tag.split("_")[0]
        studies.setdefault(dom,[]).append((d["optimizer"], d["best_lr"]))
    for dom,items in studies.items():
        # dedupe (vision has per-exp); keep median best_lr per optimizer
        byo={}
        for o,lr in items: byo.setdefault(o,[]).append(lr)
        labels=list(byo); vals=[float(np.median(v)) for v in byo.values()]
        plt.figure(figsize=(6,max(3,0.4*len(labels))))
        order=np.argsort(vals)
        labels=[labels[i] for i in order]; vals=[vals[i] for i in order]
        plt.barh(range(len(labels)),vals,color=[col(l) for l in labels])
        plt.yticks(range(len(labels)),labels,fontsize=8); plt.xscale("log")
        plt.xlabel("tuned LR (log)"); plt.title(f"Optuna best LR — {dom}")
        plt.tight_layout(); plt.savefig(FIG/f"optuna_best_lr_{dom}.png",dpi=150); plt.close()


def ft():
    rows = [json.load(open(f)) for f in glob.glob(str(R/"results/ft1b/*.json"))]
    if not rows: return
    rows.sort(key=lambda d: d["final_val_loss"])
    barh("1B FT (Qwen2.5-1.5B/Alpaca) — val loss (lower=better)",
         [d["optimizer"] for d in rows], [d["final_val_loss"] for d in rows],
         "ft_val_loss.png", "val loss", "%.3f")
    # memory vs loss scatter (MION's headline at scale)
    plt.figure(figsize=(6,4.3))
    for d in rows:
        plt.scatter(d["opt_state_mb"]/1000, d["final_val_loss"], c=col(d["optimizer"]),
                    s=80 if d["optimizer"] in FAM else 40, zorder=5)
        plt.annotate(d["optimizer"], (d["opt_state_mb"]/1000, d["final_val_loss"]), fontsize=7,
                     xytext=(4,3), textcoords="offset points")
    plt.xlabel("Optimizer-state memory (GB)"); plt.ylabel("Fine-tune val loss")
    plt.title("1B fine-tune: accuracy vs memory (lower-left best)"); plt.tight_layout()
    plt.savefig(FIG/"ft_acc_vs_memory.png", dpi=150); plt.close()


def rl():
    for env, tag in [("CartPole-v1", "cartpolev1"), ("Acrobot-v1", "acrobotv1")]:
        fs = glob.glob(str(R/f"results/rl/rl_{tag}_*.json"))
        if not fs: continue
        rows = [json.load(open(f)) for f in fs]
        rows.sort(key=lambda d: d["final_reward_mean"])
        labels = [d["optimizer"] for d in rows]; vals = [d["final_reward_mean"] for d in rows]
        errs = [d["final_reward_std"] for d in rows]
        plt.figure(figsize=(6, max(3, 0.4*len(labels))))
        plt.barh(range(len(labels)), vals, xerr=errs, color=[col(l) for l in labels], capsize=2)
        plt.yticks(range(len(labels)), labels, fontsize=8)
        plt.xlabel("final reward (3 seeds)"); plt.title(f"RL — {env} (higher=better)")
        plt.tight_layout(); plt.savefig(FIG/f"rl_{tag}.png", dpi=150); plt.close()


if __name__ == "__main__":
    for fn in [lm, cost, nlp, imagenet,
               lambda: vision_heatmap("nt"), lambda: vision_heatmap("converged"),
               ablations, controlled, optuna, ft, rl]:
        try: fn()
        except Exception as e: print(f"  ⚠ {getattr(fn,'__name__','fig')} failed: {e}")
    figs=sorted(glob.glob(str(FIG/"*.png")))
    print(f"✓ {len(figs)} figures in {FIG}")
    for p in figs: print("   ", os.path.basename(p))
