"""Sequential sweep runner. Reads a YAML config describing a task command and
per-optimizer LR grids, expands the grid (optimizer x lr x seed), and runs
each as a subprocess. Re-running skips completed runs (each task script
checks results.jsonl), so sweeps are resumable after interruption.

Usage:
  python sweep/run_sweep.py configs/lm_sweep_small.yaml
  python sweep/run_sweep.py configs/lm_sweep_small.yaml --dry-run
  python sweep/run_sweep.py configs/cifar_resnet.yaml --only mion milo_m
"""

import argparse
import itertools
import shlex
import subprocess
import sys
import time

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="restrict to these optimizers")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = cfg["base_cmd"]                     # e.g. "python tasks/lm_pretrain.py --data fineweb ..."
    seeds = cfg.get("seeds", [1])
    fixed = cfg.get("fixed_args", "")
    runs = []
    for opt, spec in cfg["optimizers"].items():
        if args.only and opt not in args.only:
            continue
        lrs = spec["lrs"] if isinstance(spec, dict) else spec
        extra = spec.get("extra_args", "") if isinstance(spec, dict) else ""
        opt_seeds = spec.get("seeds", seeds) if isinstance(spec, dict) else seeds
        for lr, seed in itertools.product(lrs, opt_seeds):
            runs.append(f"{base} {fixed} --optimizer {opt} --lr {lr} "
                        f"--seed {seed} {extra}".strip())

    print(f"{len(runs)} runs queued from {args.config}")
    failures = []
    for i, cmd in enumerate(runs, 1):
        print(f"\n=== [{i}/{len(runs)}] {cmd}")
        if args.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(shlex.split(cmd))
        print(f"=== exit {r.returncode} in {time.time()-t0:.0f}s")
        if r.returncode != 0:
            failures.append(cmd)
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for c in failures:
            print(" ", c)
        sys.exit(1)


if __name__ == "__main__":
    main()
