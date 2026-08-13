# Milo-Bench Setup & Execution Guide

## ⚠️ Critical Constraint: GPU Nodes Are Offline

- **Login Node**: Has internet, NO GPU
- **Compute Nodes**: Have GPUs, NO internet

Therefore: **Setup on login, benchmark on GPU**

---

## Phase 1: Setup Environment (LOGIN NODE - Interactive)

### Step 1: SSH to login node (if not already there)
```bash
# From your local machine or any node
ssh logan03@cedar.computecanada.ca
# (or whatever login node you use)
```

### Step 2: Run setup script on login node
```bash
cd ~/Milo
bash hpc/setup_milo_bench_login.sh
```

This will:
- Create Python venv in `temp/.venv` with internet access
- Download PyTorch with CUDA support
- Install all dependencies (transformers, datasets, etc.)
- Vendor optimizer implementations (Muon, SOAP, etc.)
- Copy your original `milo.py` 
- Download Shakespeare dataset for quick tests

**Expected time: 15-20 minutes**

### Step 3: Verify setup
```bash
# After setup completes, verify:
cd ~/Milo/temp
source .venv/bin/activate

# Check imports
python -c "import torch, transformers, datasets; print('✓ All imports work')"

# Activate is persistent (no need to re-run login setup)
```

---

## Phase 2: Run Benchmarks (GPU NODES - via SLURM)

### Step 1: Submit GPU benchmark job
```bash
cd ~/Milo
sbatch hpc/experiments/run_milo_bench_on_gpu.slurm
```

This submits a job that:
- ✓ Uses pre-setup environment from login node (offline)
- ✓ Runs on GPU node with full isolation
- ✓ Tests all optimizers with quick benchmarks
- ✓ Generates results in `temp/results/`

**Expected time: 30-45 minutes**

### Step 2: Monitor progress
```bash
# Watch job
squeue -u $(whoami) | grep milo_bench

# See output in real-time
tail -f logs/milo_bench_gpu_*.out
```

---

## Phase 3: Analyze Results (On Any Node)

### After GPU job completes:
```bash
cd ~/Milo/temp
source .venv/bin/activate

# Aggregate results
python analysis/aggregate.py results/

# Generate plots
python analysis/plots.py results/ --metric best_val_loss --lower
```

---

## Full Benchmark Runs (Optional - Much Longer)

### LM Sweep (2-3 hours per run)
```bash
# On GPU node or via SLURM
cd ~/Milo/temp
source .venv/bin/activate

python sweep/run_sweep.py configs/lm_sweep_small.yaml
```

### Vision Benchmarks (1-2 hours)
```bash
python sweep/run_sweep.py configs/cifar_resnet.yaml
python sweep/run_sweep.py configs/cifar_vit.yaml
```

### Fine-tuning (1-2 hours)
```bash
python sweep/run_sweep.py configs/glue.yaml
```

---

## Troubleshooting

### "Environment not setup"
```bash
# Setup hasn't run yet, do this first:
bash hpc/setup_milo_bench_login.sh
```

### "CUDA not available"
```bash
# Make sure you're running on a GPU node
# If job fails with this, ask scheduler for GPU nodes
sbatch --gpus-per-node=1 hpc/experiments/run_milo_bench_on_gpu.slurm
```

### "Import errors for specific optimizer"
```bash
# Some optimizers are optional and gracefully degrade
# Check logs - missing optimizer shouldn't stop the run
tail -f logs/milo_bench_gpu_*.err
```

### Network error during setup
```bash
# Make sure setup runs on LOGIN node, NOT compute node:
# ✓ Correct: bash hpc/setup_milo_bench_login.sh
# ✗ Wrong:  sbatch hpc/setup_milo_bench_login.sh (don't!)
```

---

## What Gets Set Up

```
~/Milo/temp/
├── .venv/                    # Isolated Python environment
│   └── lib/python3.x/site-packages/  # All packages
├── optimizers/
│   ├── milo.py              # Original (from ~/Milo/)
│   ├── milo2.py             # NEW: MiloM + Mion
│   ├── muon.py              # Vendored (offline)
│   ├── soap.py              # Vendored (offline)
│   ├── sophia.py            # Vendored (offline)
│   └── ... others
├── tasks/
│   ├── lm_pretrain.py       # GPT-2 on FineWeb
│   ├── vision.py            # ResNet + ViT
│   └── glue_finetune.py
├── results/                 # Benchmark outputs
│   └── results.jsonl        # Per-optimizer metrics
└── data/
    └── shakespeare/         # Cached dataset (offline)
```

All of this is **self-contained and offline-capable** after initial setup.

---

## Key Differences: Milo vs MiloM vs Mion

| Feature | Milo (Original) | MiloM (Improved) | Mion (Preconditioned) |
|---------|-----------------|-----------------|----------------------|
| Speed | Baseline | Same or faster | ~5% overhead |
| Memory | Single buffer | Same | Single buffer (like Milo) |
| Stability | Good | Better | Better |
| Best for | General use | All tasks | High-scale LM |

MiloM should **beat original Milo everywhere** with fixes to ordering/grouping.
Mion should **match Muon** while being faster than SOAP.

---

## Quick Start (Copy-Paste)

```bash
# On login node:
cd ~/Milo
bash hpc/setup_milo_bench_login.sh

# Then (can be on compute via SLURM):
sbatch hpc/experiments/run_milo_bench_on_gpu.slurm

# Monitor:
squeue -u $(whoami)
tail -f logs/milo_bench_gpu_*.out
```

Done! Results appear in `temp/results/` after GPU job completes.
