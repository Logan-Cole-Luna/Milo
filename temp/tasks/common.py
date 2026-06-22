"""Shared utilities for all tasks: seeding, LR schedule, results logging,
wall-clock and memory instrumentation."""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(step, total_steps, warmup_steps, min_ratio=0.1):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * min(prog, 1.0)))


def apply_lr(optimizer, base_lrs, scale):
    for g, base in zip(optimizer.param_groups, base_lrs):
        g["lr"] = base * scale


def base_lrs(optimizer):
    return [g["lr"] for g in optimizer.param_groups]


class RunLogger:
    """Writes a per-run curve CSV and appends one summary line to
    <out>/results.jsonl when finish() is called."""

    def __init__(self, out_dir, run_id, config):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.config = config
        self.curve_path = self.out / f"curve_{run_id}.csv"
        self._f = open(self.curve_path, "w")
        self._f.write("step,train_loss,val_metric,lr,elapsed_s\n")
        self.t0 = time.perf_counter()
        self.step_times = []

    def log(self, step, train_loss, val_metric="", lr=""):
        self._f.write(f"{step},{train_loss},{val_metric},{lr},"
                      f"{time.perf_counter() - self.t0:.1f}\n")
        self._f.flush()

    def finish(self, **summary):
        self._f.close()
        peak_mem = (torch.cuda.max_memory_allocated() / 2**30
                    if torch.cuda.is_available() else 0.0)
        rec = dict(run_id=self.run_id, **self.config, **summary,
                   peak_mem_gb=round(peak_mem, 3),
                   total_wall_s=round(time.perf_counter() - self.t0, 1))
        with open(self.out / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[result]", json.dumps(rec))
        return rec


def run_id_from(config):
    keys = ["task", "model", "optimizer", "lr", "seed"]
    return "_".join(str(config.get(k, "")) for k in keys).replace("/", "-")


def already_done(out_dir, config):
    p = Path(out_dir) / "results.jsonl"
    if not p.exists():
        return False
    rid = run_id_from(config)
    for line in open(p):
        try:
            if json.loads(line).get("run_id") == rid:
                return True
        except json.JSONDecodeError:
            pass
    return False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maybe_compile(model, enable):
    if enable and hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception as e:  # pragma: no cover
            print("torch.compile failed, continuing eager:", e)
    return model
