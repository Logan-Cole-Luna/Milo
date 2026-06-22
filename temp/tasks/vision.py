"""Vision benchmark: ResNet-18 (CNN regime) and ViT-Tiny (transformer regime,
trained from scratch -- a known hard case for plain SGD) on CIFAR-10.

Examples:
  python tasks/vision.py --model resnet18 --optimizer milo_m --lr 0.01 --epochs 30 --seed 1
  python tasks/vision.py --model vit_tiny --optimizer mion --lr 0.003 --epochs 60 --seed 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizers.registry import build_optimizer, parse_opt_kwargs  # noqa: E402
from tasks.common import (RunLogger, already_done, apply_lr, base_lrs,  # noqa: E402
                          cosine_with_warmup, run_id_from, set_seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ models
def resnet18_cifar():
    from torchvision.models import resnet18
    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # CIFAR stem
    m.maxpool = nn.Identity()
    return m


class ViTTiny(nn.Module):
    """~5.4M param ViT: patch 4, dim 192, depth 9, heads 3 (DeiT-Ti-ish)."""

    def __init__(self, dim=192, depth=9, heads=3, patch=4, n_cls=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        n_patch = (32 // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patch + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0,
                                         activation="gelu", batch_first=True,
                                         norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_cls)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos_embed
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))


# ------------------------------------------------------------------ data
def loaders(batch, vit_aug):
    import torchvision as tv
    import torchvision.transforms as T
    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    aug = [T.RandomCrop(32, 4), T.RandomHorizontalFlip()]
    if vit_aug:
        aug += [T.RandAugment(num_ops=2, magnitude=9)]
    tf = T.Compose(aug + [T.ToTensor(), T.Normalize(mean, std)])
    tfe = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    root = Path(__file__).resolve().parents[1] / "data"
    tr = tv.datasets.CIFAR10(root, True, download=True, transform=tf)
    te = tv.datasets.CIFAR10(root, False, download=True, transform=tfe)
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True, num_workers=4,
                                        drop_last=True, persistent_workers=True),
            torch.utils.data.DataLoader(te, 1024, num_workers=2))


@torch.no_grad()
def test_acc(model, te):
    model.eval()
    correct = n = 0
    for x, y in te:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        n += len(y)
    model.train()
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet18", "vit_tiny"], required=True)
    ap.add_argument("--optimizer", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="results/vision")
    ap.add_argument("--opt-kwargs", default="")
    args = ap.parse_args()

    config = dict(task="cifar10", model=args.model, optimizer=args.optimizer,
                  lr=args.lr, wd=args.wd, seed=args.seed, epochs=args.epochs,
                  batch=args.batch, opt_kwargs=args.opt_kwargs)
    if already_done(args.out_dir, config):
        print("[skip]", run_id_from(config))
        return

    set_seed(args.seed)
    tr, te = loaders(args.batch, vit_aug=args.model == "vit_tiny")
    model = (resnet18_cifar() if args.model == "resnet18" else ViTTiny()).to(DEVICE)
    optimizer, meta = build_optimizer(args.optimizer, model, args.lr, args.wd,
                                      **parse_opt_kwargs(args.opt_kwargs))
    if meta["sf_train_eval"]:
        optimizer.train()
    lrs0 = base_lrs(optimizer)
    steps_total = args.epochs * len(tr)
    warmup = max(100, steps_total // 50)

    logger = RunLogger(args.out_dir, run_id_from(config), config)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step, best_acc, t_steps = 0, 0.0, []
    for epoch in range(args.epochs):
        for x, y in tr:
            step += 1
            t0 = time.perf_counter()
            if meta["use_schedule"]:
                apply_lr(optimizer, lrs0, cosine_with_warmup(step, steps_total, warmup))
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                loss = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            t_steps.append(time.perf_counter() - t0)
            if step % 50 == 0:
                logger.log(step, f"{loss.item():.4f}")
        if meta["sf_train_eval"]:
            optimizer.eval()
        acc = test_acc(model, te)
        if meta["sf_train_eval"]:
            optimizer.train()
        best_acc = max(best_acc, acc)
        logger.log(step, f"{loss.item():.4f}", f"{acc:.4f}")
        print(f"epoch {epoch+1}/{args.epochs} loss {loss.item():.4f} acc {acc*100:.2f}%")

    logger.finish(final_acc=round(acc, 4), best_acc=round(best_acc, 4),
                  ms_per_step=round(1000 * float(np.median(t_steps)), 1))


if __name__ == "__main__":
    main()
