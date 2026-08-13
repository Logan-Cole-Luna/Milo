"""
Lightweight smoke test: confirm every optimizer LOADS and RUNS on every
experiment, with 2 training steps on tiny synthetic batches.

Not a training run — just validates instantiation + forward/backward/step for
the full (experiment x optimizer) matrix in seconds per cell. Runs on CPU.

Usage:
    python hpc/experiments/smoke_quick.py [--domains vision imagenet nlp]
"""
import sys, os, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimizer imports (same as the runners)
from milo import milo
from optimizers.milo2 import MiloM, Mion
from optimizers.lion import Lion
from optimizers.adam_mini import AdamMini
from optimizers.rmsprop_momentum import RMSpropMomentum
from optimizers.shampoo import Shampoo
from optimizers.soap import SOAP
from optimizers.muon import MuonWithAuxAdam


def build_optimizer(name, model, lr, params):
    """Mirror the construction logic used in the experiment runners."""
    p = {k: v for k, v in params.items() if k != "lr"}
    n = name.upper()
    if n in ("MILO", "MILO_LW"):
        return milo(model.parameters(), lr=lr, **p)
    if n == "MILOM":
        return MiloM(model.parameters(), lr=lr, **p)
    if n in ("MION", "MION_NOR"):
        return Mion(model.parameters(), lr=lr, **p)
    if n == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, **p)
    if n == "ADAGRAD":
        return torch.optim.Adagrad(model.parameters(), lr=lr, **p)
    if n == "ADAMW":
        return torch.optim.AdamW(model.parameters(), lr=lr, **p)
    if n == "LION":
        return Lion(model.parameters(), lr=lr, **p)
    if n == "ADAM_MINI":
        return AdamMini(model.parameters(), lr=lr, **p)
    if n == "RMSPROP_MOMENTUM":
        return RMSpropMomentum(model.parameters(), lr=lr, **p)
    if n == "SHAMPOO":
        return Shampoo(model.parameters(), lr=lr, **p)
    if n == "SOAP":
        return SOAP(model.parameters(), lr=lr, **p)
    if n == "MUON":
        wd = p.get("weight_decay", 0)
        hidden = [w for w in model.parameters() if w.ndim >= 2]
        other = [w for w in model.parameters() if w.ndim < 2]
        groups = [
            dict(params=hidden, use_muon=True, lr=lr, momentum=0.95, weight_decay=wd),
            dict(params=other, use_muon=False, lr=lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=wd),
        ]
        return MuonWithAuxAdam(groups)
    raise ValueError(f"Unknown optimizer {name}")


def run_cell(model, optimizer, x, y):
    """Two training steps; returns True on success."""
    model.train()
    crit = nn.CrossEntropyLoss()
    for _ in range(2):
        optimizer.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return float(loss.item())


def smoke_matrix(label, model_fn, optimizers, lr_fn, params_fn, batch_fn):
    """Run every optimizer against a freshly-built model; report per cell."""
    print(f"\n{'='*64}\n  {label}\n{'='*64}")
    failures = []
    for opt_name in optimizers:
        try:
            model = model_fn().to(DEVICE)
            x, y = batch_fn()
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer = build_optimizer(opt_name, model, lr_fn(opt_name), params_fn(opt_name))
            loss = run_cell(model, optimizer, x, y)
            print(f"  ✓ {opt_name:<18} (loss {loss:.3f})")
        except Exception as e:
            print(f"  ✗ {opt_name:<18} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(opt_name)
    return failures


def smoke_vision():
    from experiments.supervised_learning.network import (
        LogisticRegressionModel, MLP, ResNet34, VGG11, ViT_Tiny)
    from experiments.vision.config import (
        OPTIMIZERS, OPTIMIZER_PARAMS, get_learning_rate)

    # (experiment_name, model factory, input shape, num_classes)
    experiments = [
        ("LOGISTIC",           lambda: LogisticRegressionModel(input_dim=784, num_classes=10), (8, 784), 10),
        ("MULTILAYER",         lambda: MLP(input_dim=784, hidden_dim=256, output_dim=10),       (8, 784), 10),
        ("RESNET34_CIFAR10",   lambda: ResNet34(num_classes=10),   (8, 3, 32, 32), 10),
        ("RESNET34_CIFAR100",  lambda: ResNet34(num_classes=100),  (8, 3, 32, 32), 100),
        ("VGG11_CIFAR10",      lambda: VGG11(num_classes=10),      (8, 3, 32, 32), 10),
        ("VGG11_CIFAR100",     lambda: VGG11(num_classes=100),     (8, 3, 32, 32), 100),
        ("VIT_TINY_CIFAR10",   lambda: ViT_Tiny(num_classes=10),   (8, 3, 32, 32), 10),
        ("VIT_TINY_CIFAR100",  lambda: ViT_Tiny(num_classes=100),  (8, 3, 32, 32), 100),
    ]
    all_fail = {}
    for exp_name, model_fn, shape, ncls in experiments:
        def batch_fn(s=shape, c=ncls):
            return torch.randn(*s), torch.randint(0, c, (s[0],))
        fails = smoke_matrix(
            f"VISION / {exp_name}", model_fn, OPTIMIZERS,
            lr_fn=lambda o, e=exp_name: get_learning_rate(e, o),
            params_fn=lambda o: OPTIMIZER_PARAMS.get(o, {}),
            batch_fn=batch_fn)
        if fails:
            all_fail[exp_name] = fails
    return all_fail


def smoke_imagenet():
    from experiments.supervised_learning.network import ResNet34
    from experiments.imagenet.config import OPTIMIZERS, OPTIMIZER_PARAMS, LEARNING_RATES
    def batch_fn():
        return torch.randn(8, 3, 64, 64), torch.randint(0, 200, (8,))
    fails = smoke_matrix(
        "IMAGENET / ResNet34 (Tiny-ImageNet-200)",
        lambda: ResNet34(num_classes=200), OPTIMIZERS,
        lr_fn=lambda o: LEARNING_RATES[o],
        params_fn=lambda o: OPTIMIZER_PARAMS.get(o, {}),
        batch_fn=batch_fn)
    return {"IMAGENET": fails} if fails else {}


def smoke_nlp():
    from transformers import AutoModelForSequenceClassification
    from experiments.nlp.config import OPTIMIZERS, OPTIMIZER_PARAMS, LEARNING_RATES

    def model_fn():
        return AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2)

    # BERT forward differs (takes input_ids); use a dedicated cell runner.
    print(f"\n{'='*64}\n  NLP / BERT-base (SST-2)\n{'='*64}")
    failures = []
    for opt_name in OPTIMIZERS:
        try:
            model = model_fn().to(DEVICE)
            ids = torch.randint(0, 30522, (4, 32)).to(DEVICE)
            mask = torch.ones(4, 32, dtype=torch.long).to(DEVICE)
            labels = torch.randint(0, 2, (4,)).to(DEVICE)
            optimizer = build_optimizer(opt_name, model, LEARNING_RATES[opt_name],
                                        OPTIMIZER_PARAMS.get(opt_name, {}))
            model.train()
            for _ in range(2):
                optimizer.zero_grad()
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            print(f"  ✓ {opt_name:<18} (loss {out.loss.item():.3f})")
        except Exception as e:
            print(f"  ✗ {opt_name:<18} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(opt_name)
    return {"NLP": failures} if failures else {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=["vision", "imagenet", "nlp"])
    args = ap.parse_args()

    print(f"Quick smoke — device: {DEVICE}")
    all_failures = {}
    if "vision" in args.domains:
        all_failures.update(smoke_vision())
    if "imagenet" in args.domains:
        all_failures.update(smoke_imagenet())
    if "nlp" in args.domains:
        all_failures.update(smoke_nlp())

    print(f"\n{'='*64}\n  SUMMARY\n{'='*64}")
    if all_failures:
        for where, opts in all_failures.items():
            print(f"  ✗ {where}: {opts}")
        sys.exit(1)
    print("  ✓ ALL optimizers loaded and ran on ALL experiments")
