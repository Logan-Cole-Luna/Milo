# Repository Structure

Reorganized around the two things that matter for the paper: **optimizers** and
**large benchmark experiments**.

```
Milo/
├── milo.py                      # canonical MILO (accelerated). LIVE import: `from milo import milo`
├── optimizers/                  # all other optimizers
│   ├── milo2.py                 # MiloM, Mion  (flagship)
│   ├── muon.py  soap.py  shampoo.py  lion.py  adam_mini.py  rmsprop_momentum.py ...
│
├── experiments/
│   ├── lm/                      # ★ from-scratch LM pretraining (FineWeb-Edu, Llama-style)
│   │   ├── model.py             #   RMSNorm + RoPE + SwiGLU decoder; size presets
│   │   ├── data.py              #   download (login) / tokenize (compute) / memmap loader
│   │   ├── config.py            #   model presets + per-optimizer params + build_optimizer
│   │   └── train.py             #   loss-vs-tokens / -walltime, speedup-to-target, tok/s, mem
│   ├── vision/                  # CIFAR / MNIST (ResNet, VGG, ViT)
│   ├── imagenet/                # Tiny-ImageNet-200 (ResNet34)
│   ├── nlp/                     # BERT/SST-2 fine-tuning
│   └── supervised_learning/     # shared network defs (network.py)
│
├── benchmarks/
│   └── optimizer_cost.py        # state-memory / peak-mem / tok-s / ms-step table
│
├── hpc/experiments/             # all SLURM submit scripts (run_*, submit_*, tokenize_*)
│
├── results/                     # ALL experiment outputs (gitignored). See subdirs below.
└── temp/                        # legacy milo-bench (kept for reference; not the main pipeline)
```

## results/ layout
- `results/finals/`        — tuned multi-seed final runs (the reported numbers)
- `results/mion_ablation/` — MION one-knob ablations (ns_steps, rms_target, spectral, scale_factor)
- `results/controlled/`    — MION vs Muon-aux controlled study
- `results/optuna/`        — LR-search studies (best LR per optimizer)
- `results/lm/`, `results/cost/` — LM pretraining + cost benchmark (in progress)
- `results/archive_old/`   — superseded early runs (results_nt_*)
- Vision per-experiment outputs remain under `experiments/vision/<exp>/results_*` for now
  (unifying these into `results/` requires changing the vision saver — planned, deferred to
  avoid breaking in-flight jobs).

## Known cleanup TODO (deferred — risk of breaking live imports)
- **milo.py duplication**: root `milo.py` (live) vs `optimizers/milo.py` and
  `milo_accelerated.py` (stale copies). Consolidate to one canonical module and update
  imports in a dedicated pass with full test coverage.
- Stale root files: `run_milo_comparison.py`, `milo_comparison_results.json`,
  `milo_layer_mapping.pt` — verify unused, then remove.
- `experiments/train-llm-from-scratch/` — superseded by `experiments/lm/`; archive.
