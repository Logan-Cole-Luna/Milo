"""
Optuna-based learning-rate search (continuous log-uniform range per optimizer).

One study per (domain, optimizer[, experiment]); each trial trains with a
sampled LR (1 seed, reduced epochs) and returns validation accuracy. The best
LR + full trial history are written to results_optuna/<tag>.json.

Usage:
  python optuna_sweep.py --domain nlp      --optimizer MILO            --trials 15
  python optuna_sweep.py --domain imagenet --optimizer SHAMPOO         --trials 15
  python optuna_sweep.py --domain vision   --optimizer MION --experiment VGG11_CIFAR10 --trials 15
"""
import argparse, json, os, sys, warnings
from pathlib import Path
import numpy as np
import optuna

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
warnings.filterwarnings("ignore")

# Continuous (lo, hi) log-uniform LR ranges per optimizer, per regime.
NLP_RANGE = {
    "SGD": (1e-4, 5e-2), "ADAMW": (5e-6, 5e-4), "ADAGRAD": (1e-5, 5e-3),
    "LION": (1e-6, 1e-4), "ADAM_MINI": (5e-6, 5e-4), "RMSPROP_MOMENTUM": (1e-5, 1e-3),
    "SHAMPOO": (1e-5, 5e-3), "SOAP": (1e-5, 1e-3), "MUON": (1e-3, 1e-1),
    "MILO": (1e-5, 3e-3), "MILO_LW": (1e-5, 3e-3), "MILOM": (1e-6, 1e-3), "MION": (1e-5, 3e-3),
}
CNN_RANGE = {
    "SGD": (1e-2, 5e-1), "ADAMW": (1e-4, 1e-2), "ADAGRAD": (1e-3, 1e-1),
    "LION": (3e-5, 3e-3), "ADAM_MINI": (1e-4, 1e-2), "RMSPROP_MOMENTUM": (1e-4, 1e-2),
    "SHAMPOO": (1e-4, 3e-2), "SOAP": (3e-4, 3e-2), "MUON": (3e-3, 1e-1),
    "MILO": (3e-3, 3e-1), "MILO_LW": (3e-3, 3e-1), "MILOM": (1e-3, 1e-1), "MION": (1e-3, 1e-1),
}
VIT_RANGE = {
    "SGD": (3e-3, 1e-1), "ADAMW": (5e-5, 3e-3), "ADAGRAD": (3e-4, 3e-2),
    "LION": (1e-5, 1e-3), "ADAM_MINI": (5e-5, 3e-3), "RMSPROP_MOMENTUM": (5e-5, 3e-3),
    "SHAMPOO": (1e-4, 1e-2), "SOAP": (1e-4, 1e-2), "MUON": (1e-3, 5e-2),
    "MILO": (1e-3, 1e-1), "MILO_LW": (1e-3, 1e-1), "MILOM": (1e-3, 5e-2), "MION": (1e-3, 5e-2),
}


def make_nlp_objective(opt, epochs):
    from experiments.nlp.nlp_experiment import run_bert_experiment
    from experiments.nlp.config import OPTIMIZER_PARAMS, BATCH_SIZE
    lo, hi = NLP_RANGE[opt]
    def objective(trial):
        lr = trial.suggest_float("lr", lo, hi, log=True)
        res = run_bert_experiment(optimizer_name=opt, optimizer_params=OPTIMIZER_PARAMS[opt],
                                  batch_size=BATCH_SIZE, epochs=epochs, learning_rate=lr,
                                  runs=1, results_dir=f"results_optuna/_tmp_nlp_{opt}")
        return float(np.mean([r["best_val_accuracy"] for r in res]))
    return objective


def make_imagenet_objective(opt, epochs):
    from experiments.imagenet.imagenet_experiment import run_imagenet_experiment
    from experiments.imagenet.config import OPTIMIZER_PARAMS, BATCH_SIZE, DATA_ROOT
    lo, hi = CNN_RANGE[opt]
    data_root = os.getenv("IMAGENET_DATA", DATA_ROOT)
    def objective(trial):
        lr = trial.suggest_float("lr", lo, hi, log=True)
        res = run_imagenet_experiment(model_name="ResNet34", batch_size=BATCH_SIZE,
                                      epochs=epochs, learning_rate=lr, optimizer_name=opt,
                                      optimizer_params=OPTIMIZER_PARAMS[opt], runs=1,
                                      data_root=data_root,
                                      results_dir=f"results_optuna/_tmp_inet_{opt}")
        return float(np.mean([r["best_val_accuracy"] for r in res]))
    return objective


def make_vision_objective(opt, experiment, epochs):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from experiments.vision.vision_experiment import get_model, get_dataloader
    from experiments.vision.config import OPTIMIZER_PARAMS, EXPERIMENT_CONFIGS, ARCH_FAMILY
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from smoke_quick import build_optimizer

    rng = VIT_RANGE if ARCH_FAMILY.get(experiment) == "vit" else CNN_RANGE
    lo, hi = rng[opt]
    cfg = EXPERIMENT_CONFIGS[experiment]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load once; reuse across trials. Subset for speed.
    full = get_dataloader(cfg["dataset_name"], cfg["transforms"], 128, train=True).dataset
    n = len(full); n_val = int(0.1 * n)
    sub = min(n - n_val, 8000)  # cap train subset for fast trials
    tr, va, _ = random_split(full, [sub, n_val, n - sub - n_val],
                             generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=128, shuffle=True, num_workers=4)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=2)

    def objective(trial):
        lr = trial.suggest_float("lr", lo, hi, log=True)
        torch.manual_seed(0)
        model = get_model(cfg["model_name"], cfg["model_args"]).to(device)
        opt_obj = build_optimizer(opt, model, lr, OPTIMIZER_PARAMS.get(opt, {}))
        crit = nn.CrossEntropyLoss()
        for _ in range(epochs):
            model.train()
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                opt_obj.zero_grad(); loss = crit(model(x), y)
                if not torch.isfinite(loss): raise optuna.TrialPruned()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt_obj.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
        return 100.0 * correct / total
    return objective


# LM LR search ranges (FineWeb-Edu). MILO-family wide; baselines literature-centered.
LM_RANGE = {
    "MILO": (3e-4, 3e-2), "MILO_LW": (3e-4, 3e-2), "MILOM": (1e-3, 1e-1),
    "MION": (1e-3, 1e-1), "MION_ADAM": (3e-3, 1e-1), "MION_V2": (1e-3, 1e-1),
    "MION_NOR": (1e-3, 1e-1),  # same range as MION
    "ADAMW": (1e-4, 3e-3), "ADAM_MINI": (1e-4, 3e-3), "LION": (3e-5, 1e-3),
    "SGD": (3e-2, 1.0), "MUON": (3e-3, 1e-1), "SOAP": (3e-4, 1e-2), "SHAMPOO": (3e-4, 3e-2),
}


FT_RANGE = {  # 1B fine-tuning LRs (small; orthogonalizers want smaller still)
    "ADAMW": (3e-6, 1e-4), "ADAM_MINI": (3e-6, 1e-4), "LION": (1e-6, 3e-5), "SGD": (1e-4, 1e-2),
    "MUON": (1e-5, 1e-3), "SHAMPOO": (1e-4, 1e-2),
    "MILO": (3e-6, 3e-4), "MILO_LW": (3e-6, 3e-4), "MILOM": (1e-5, 1e-3), "MION": (1e-5, 1e-3),
    "MION_GATED": (1e-5, 1e-3),  # same LR range as MION; ortho_strength (beta) is jointly tuned
}
RL_RANGE = {
    "SGD": (1e-3, 3e-2), "ADAMW": (3e-4, 1e-2), "ADAGRAD": (1e-3, 3e-2), "LION": (3e-5, 3e-3),
    "ADAM_MINI": (3e-4, 1e-2), "RMSPROP_MOMENTUM": (3e-4, 1e-2), "SHAMPOO": (1e-3, 3e-2),
    "SOAP": (3e-4, 1e-2), "MUON": (1e-3, 3e-2), "MILO": (1e-3, 3e-2), "MILO_LW": (1e-3, 3e-2),
    "MILOM": (1e-3, 3e-2), "MION": (1e-3, 3e-2),
}


def make_ft_objective(opt, steps):
    import subprocess
    lo, hi = FT_RANGE[opt]
    def objective(trial):
        lr = trial.suggest_float("lr", lo, hi, log=True)
        out = f"results_optuna/_tmp_ft_{opt}_{trial.number}"
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        if opt == "MION_GATED":
            # jointly search the orthogonalization-strength gate alongside lr:
            # beta=1 -> full MION (pretrain-like), beta=0 -> MiloM-like group-std (FT-like)
            beta = trial.suggest_float("beta", 0.0, 1.0)
            env["ORTHO_STRENGTH"] = f"{beta:g}"
        r = subprocess.run([sys.executable, "-u", "-m", "experiments.lm.finetune",
                            "--optimizer", opt, "--lr", f"{lr:g}", "--steps", str(steps),
                            "--out-dir", out], capture_output=True, text=True, env=env)
        f = Path(out) / f"ft_{opt.lower()}_seed1.json"
        if not f.exists():
            print(r.stderr[-1200:]); raise optuna.TrialPruned()
        return float(json.load(open(f))["final_val_loss"])
    return objective


def make_rl_objective(opt, env, episodes):
    import subprocess
    lo, hi = RL_RANGE[opt]
    def objective(trial):
        lr = trial.suggest_float("lr", lo, hi, log=True)
        out = f"results_optuna/_tmp_rl_{opt}_{trial.number}"
        r = subprocess.run([sys.executable, "-u", "-m", "experiments.rl.reinforce",
                            "--env", env, "--optimizer", opt, "--lr", f"{lr:g}",
                            "--episodes", str(episodes), "--seeds", "1", "--out-dir", out],
                           capture_output=True, text=True)
        f = Path(out) / f"rl_{env.replace('-','').lower()}_{opt.lower()}.json"
        if not f.exists():
            print(r.stderr[-1200:]); raise optuna.TrialPruned()
        return float(json.load(open(f))["final_reward_mean"])
    return objective


def make_lm_objective(opt, tokens):
    """Each trial trains the real LM (subprocess) at a sampled LR; returns val loss."""
    import subprocess
    lo, hi = LM_RANGE[opt]
    def objective(trial):
        lr = trial.suggest_float("lr", lo, hi, log=True)
        out_dir = f"results_optuna/_tmp_lm_{opt}_{trial.number}"
        cmd = [sys.executable, "-u", "-m", "experiments.lm.train",
               "--optimizer", opt, "--model", "small", "--lr", f"{lr:g}",
               "--tokens", str(tokens), "--batch-size", "16", "--grad-accum", "16",
               "--ctx", "1024", "--eval-every", "100", "--out-dir", out_dir, "--seed", "1"]
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        f = Path(out_dir) / f"lm_small_{opt.lower()}_seed1.json"
        if not f.exists():
            print(r.stdout[-1500:]); print(r.stderr[-1500:])
            raise optuna.TrialPruned()
        d = json.load(open(f))
        return float(d["final_val_loss"])
    return objective


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["nlp", "imagenet", "vision", "lm", "ft", "rl"])
    ap.add_argument("--optimizer", required=True)
    ap.add_argument("--experiment", default=None)   # vision: exp name; rl: env name
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lm-tokens", default="6e7", help="tokens per LM sweep trial")
    args = ap.parse_args()

    minimize = (args.domain in ("lm", "ft"))   # ft minimizes val loss; rl maximizes reward
    if args.domain == "nlp":
        epochs = args.epochs or 2
        objective = make_nlp_objective(args.optimizer, epochs); tag = f"nlp_{args.optimizer}"
    elif args.domain == "imagenet":
        epochs = args.epochs or 3
        objective = make_imagenet_objective(args.optimizer, epochs); tag = f"imagenet_{args.optimizer}"
    elif args.domain == "lm":
        objective = make_lm_objective(args.optimizer, args.lm_tokens)
        tag = f"lm_{args.optimizer}"
    elif args.domain == "ft":
        objective = make_ft_objective(args.optimizer, args.epochs or 150)
        tag = f"ft_{args.optimizer}"
    elif args.domain == "rl":
        env = args.experiment or "CartPole-v1"
        objective = make_rl_objective(args.optimizer, env, args.epochs or 300)
        tag = f"rl_{env.replace('-','').lower()}_{args.optimizer}"
    else:
        assert args.experiment, "vision requires --experiment"
        epochs = args.epochs or 4
        objective = make_vision_objective(args.optimizer, args.experiment, epochs)
        tag = f"vision_{args.experiment}_{args.optimizer}"

    print(f"Optuna LR search: {tag} | {args.trials} trials | {'minimize' if minimize else 'maximize'}", flush=True)
    study = optuna.create_study(direction="minimize" if minimize else "maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials)

    Path("results_optuna").mkdir(exist_ok=True)
    out = {
        "tag": tag, "domain": args.domain, "optimizer": args.optimizer,
        "experiment": args.experiment, "best_lr": study.best_params["lr"],
        "best_params": study.best_params,  # full HP dict (e.g. includes "beta" for MION_GATED)
        "best_value": study.best_value,
        "trials": [{**t.params, "value": t.value} for t in study.trials],
    }
    with open(f"results_optuna/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    metric = "val loss" if minimize else "val acc"
    print(f"\n✓ {tag}: best LR = {study.best_params['lr']:.3e}  "
          f"({metric} {study.best_value:.4f})  → results_optuna/{tag}.json", flush=True)
