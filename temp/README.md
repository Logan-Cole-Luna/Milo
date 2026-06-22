# milo-bench: conference-grade optimizer benchmark suite

Benchmarking harness for the Milo optimizer family (original Milo, MiloM, Mion)
against the optimizers reviewers at NeurIPS/ICML/ICLR/AAAI expect to see, on
tasks representative of the modern training landscape. Designed to run a full
evidence package on a single RTX 5070 Ti (16 GB), scaling up trivially if more
compute becomes available.

## Repository layout

```
milo-bench/
├── setup.sh                    # one-shot environment setup (read the Blackwell note!)
├── requirements.txt
├── optimizers/
│   ├── registry.py             # unified factory: build_optimizer(name, model, lr, wd)
│   ├── milo2.py                # MiloM + Mion (ours)
│   ├── milo.py                 # <- DROP YOUR ORIGINAL milo.py HERE
│   ├── ademamix.py             # local AdEMAMix implementation
│   └── vendored/               # muon.py / soap.py / sophia.py (vendor script)
├── tasks/
│   ├── lm_pretrain.py          # GPT-2-style LM on FineWeb-Edu  (primary task)
│   ├── vision.py               # ResNet-18 + ViT-Tiny on CIFAR-10
│   ├── glue_finetune.py        # RoBERTa-base fine-tuning (SST-2/MRPC/RTE)
│   └── common.py
├── scripts/
│   ├── vendor_optimizers.sh    # pinned single-file upstream optimizers
│   ├── download_data.py        # FineWeb-Edu / shakespeare tokenization
│   ├── smoke_test.sh           # 10-min validation of every optimizer
│   └── run_ablations.sh        # MiloM/Mion component ablations
├── sweep/run_sweep.py          # resumable grid runner over YAML configs
├── configs/*.yaml              # LR grids per task
├── analysis/
│   ├── aggregate.py            # tables (md/LaTeX) + paired-bootstrap significance
│   └── plots.py                # curves, LR-sensitivity, wall-clock Pareto
└── tests/test_optimizers.py    # 30-step smoke test of every optimizer
```

## 1. Environment setup

> **RTX 5070 Ti is Blackwell (sm_120).** Stock PyPI torch wheels may lack
> sm_120 kernels — you need PyTorch ≥ 2.7 built against CUDA 12.8 and an
> NVIDIA driver ≥ 570. `setup.sh` handles this via the cu128 index.

```bash
git init milo-bench && cd milo-bench   # or unzip this folder
bash setup.sh                          # venv + torch cu128 + deps + vendoring
source .venv/bin/activate
cp /path/to/your/milo.py optimizers/milo.py   # original implementation
python tests/test_optimizers.py               # every optimizer: 30 steps, loss must drop
```

Vendored / installed optimizer sources (pinned in `scripts/vendor_optimizers.sh`):

| Optimizer | Source | Why reviewers expect it |
|---|---|---|
| SGD-M, AdamW, NAdamW, Adafactor | `torch.optim` | universal baselines |
| Lion | [lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch) | sign-momentum family |
| Schedule-Free AdamW | [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free) | AlgoPerf 2024 self-tuning winner |
| Prodigy | [konstmish/prodigy](https://github.com/konstmish/prodigy) | parameter-free baseline |
| Sophia-G | [Liuhong99/Sophia](https://github.com/Liuhong99/Sophia) | 2nd-order (Hessian-diag) for LMs |
| Muon | [KellerJordan/Muon](https://github.com/KellerJordan/Muon) | spectral; closest rival to Mion |
| SOAP | [nikhilvyas/SOAP](https://github.com/nikhilvyas/SOAP) | Shampoo-eigenbasis Adam |
| Distributed Shampoo | [facebookresearch/optimizers](https://github.com/facebookresearch/optimizers) | full Kronecker preconditioning |
| PSGD-Kron | [kron-torch](https://github.com/evanatyourservice/kron-torch) | Kronecker probabilistic SGD |
| AdEMAMix | local impl of [arXiv:2409.03137](https://arxiv.org/abs/2409.03137) | dual-EMA momentum |
| Milo / MiloM / Mion | this repo | ours |

The registry degrades gracefully: missing packages skip with a clear message,
so you can start sweeps before everything installs.

## 2. Data

```bash
python scripts/download_data.py --source shakespeare                      # smoke (instant)
python scripts/download_data.py --source fineweb --train-tokens 2e9       # ~4 GB on disk
# CIFAR-10 and GLUE download automatically on first task run.
```

## 3. Validate, then run

```bash
bash scripts/smoke_test.sh           # ~10 min: 60 LM steps per optimizer
```

**Phase 1 — LR sweeps** (small 52M LM, 0.4B tokens/run, 1 seed; ~25–40 min/run,
~16 optimizers × ~3 LRs ≈ 2–3 GPU-days; resumable — re-running skips finished runs):

```bash
python sweep/run_sweep.py configs/lm_sweep_small.yaml
python analysis/aggregate.py results/lm_sweep
python analysis/plots.py results/lm_sweep --metric best_val_loss --lower
```

**Phase 2 — headline LM runs**: copy each optimizer's winning LR into
`configs/lm_final.yaml`, then run 124M-class × 2B tokens × 3 seeds (~5–7 h/run;
prioritize adamw/muon/soap/sophia/milo/milo_m/mion ≈ 5 GPU-days):

```bash
python sweep/run_sweep.py configs/lm_final.yaml
```

**Phase 3 — breadth**: vision (both regimes) + fine-tuning (~2 GPU-days):

```bash
python sweep/run_sweep.py configs/cifar_resnet.yaml
python sweep/run_sweep.py configs/cifar_vit.yaml
python sweep/run_sweep.py configs/glue.yaml
```

**Phase 4 — ablations** (the section reviewers read first for a method paper):

```bash
bash scripts/run_ablations.sh
```
Covers: MiloM ordering legacy (vs original Milo itself), row vs flat grouping,
grafting blend strength α ∈ {0, 0.2, 0.5}, Mion Newton–Schulz steps ∈ {1,3,5}
(speed/quality), and rms_target ∈ {0.1, 0.2, 0.4}.

All results accumulate in per-task `results.jsonl` + per-run curve CSVs;
`aggregate.py` emits the markdown/LaTeX summary table (best LR, mean±std over
seeds, ms/step, peak VRAM) and paired-bootstrap CIs vs AdamW.

## 4. What this suite gives you for the paper (reviewer checklist)

- [x] **Per-optimizer tuning, not shared HPs** — independent LR grid per
      method, best-LR selection protocol stated (cf. AlgoPerf, Dahl et al. 2023).
- [x] **LR-sensitivity curves** — robustness across LRs is now a standard ask
      (and Mion/Muon's wide stable range is a selling point).
- [x] **Multiple seeds with mean±std and significance** — paired bootstrap vs
      AdamW at matched seeds.
- [x] **Both step-budget and wall-clock comparisons** — ms/step + total
      wall-clock Pareto plot; preconditioned methods must win *after* overhead.
- [x] **Memory accounting** — peak VRAM per optimizer (Shampoo/SOAP state
      multipliers vs Mion's single momentum buffer is a key table row).
- [x] **Modern task spread** — LM pretraining (primary arena), CNN regime,
      ViT-from-scratch (hard for SGD), and fine-tuning (different LR regime).
- [x] **Two model scales on the LM task** — 52M sweep scale + 124M headline
      scale; report whether best LRs transfer (µP-style evidence).
- [x] **Component ablations** isolating each design decision.
- [x] **Reproducibility** — pinned vendored baselines, seeds, resumable runner,
      every run's config serialized into results.jsonl.

## 5. 16 GB VRAM notes

- 124M LM: `--batch-size 8 --accum 8` (bf16 autocast) fits comfortably with
  AdamW/Muon/Mion. For SOAP/Shampoo drop to `--batch-size 4 --accum 16`.
- Shampoo: keep `precondition_frequency=25`, `max_preconditioner_dim=2048`
  (set in registry) or step time explodes.
- `--compile` gives ~1.3–1.6× on the LM task once warm; disable while debugging.
- If you hit OOM with tied embeddings disabled, pass `--tie` (also reduces the
  aux-param fraction for Muon/Mion — note it in the paper either way).

## 6. Known scope limits (state these in the paper's limitations)

- Single-GPU scale: no ImageNet-1k, no >1B-param models. If reviews demand it,
  `lm_pretrain.py` is DDP-trivial (the registry's Muon path already has the
  distributed variant vendored) and PACE/ICE A100 time covers a 350M run.
- Sophia is integrated for the LM task only (its Hessian estimator is
  CE-specific), matching its paper's scope.
- Prodigy/Schedule-Free are run in their parameter-free configurations, which
  is their intended comparison mode.

## 7. Suggested headline claims to test for

1. Mion matches/beats Muon at equal steps **without an auxiliary AdamW** and
   with one LR across all parameter types (unified rms_target).
2. MiloM > original Milo everywhere at equal cost → the ordering/grouping/
   decay fixes are responsible (ablation table).
3. Mion's overhead vs AdamW ≤ Muon's (~5%), far below SOAP/Shampoo, at
   comparable or better val loss — the wall-clock Pareto plot.
4. Wider stable-LR range than AdamW (LR-sensitivity figure).
