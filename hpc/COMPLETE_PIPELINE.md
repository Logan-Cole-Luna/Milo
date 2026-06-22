# Complete Experiment Pipeline: Baseline → Tuning → Tuned

Run all three optimization phases with a single command. Compare baseline vs tuning vs tuned hyperparameters.

## Quick Start

```bash
bash hpc/submit_all_phases.sh
```

This submits:
- **Phase 1 (PARALLEL):** Baseline + Tuning experiments
- **Phase 2 (SEQUENTIAL):** Tuned experiments (waits for Phase 1)

---

## What Runs

### Phase 1: Baseline + Tuning (All Parallel, ~24h)

**Baseline (3 jobs):**
- Vision: 6 models × 11 optimizers × 5 runs = 330 runs (~24h)
- ImageNet: 1 model × 11 optimizers × 3 runs = 33 runs (~12h)
- NLP: BERT × 11 optimizers × 5 runs = 55 runs (~22h)

**Tuning (1 job):**
- Hyperparameter search on all 3 domains (~8-10h)
- Finds optimal LR, momentum, weight_decay, etc. for each optimizer

### Phase 2: Tuned Experiments (Sequential, ~10-15h)

**Tuned (3 jobs, wait for Phase 1):**
- Vision: Same experiments with optimized hyperparameters
- ImageNet: Same experiments with optimized hyperparameters
- NLP: Same experiments with optimized hyperparameters

---

## Results Comparison

Each domain has results in three directories:

```
experiments/vision/
├── results_nt/              ← Baseline hyperparameters
└── results_nt_tuned/        ← Optimized hyperparameters
    └── vit_tiny_cifar100/
        ├── vision_resnet34_cifar10_milo.json
        ├── vision_resnet34_cifar10_adamw.json
        └── ...

experiments/nlp/
├── results_nt_bert/         ← Baseline hyperparameters
└── results_nt_bert_tuned/   ← Optimized hyperparameters

experiments/imagenet/
├── results_nt_imagenet200/  ← Baseline hyperparameters
└── results_nt_imagenet200_tuned/  ← Optimized hyperparameters
```

---

## Comparison Workflow

### Step 1: Run Pipeline
```bash
bash hpc/submit_all_phases.sh
```

### Step 2: Monitor Progress
```bash
# Watch job queue
watch squeue -u $(whoami)

# Tail logs
tail -f logs/vision_experiments_*.out
tail -f logs/nlp_experiments_*.out
tail -f logs/imagenet_experiments_*.out
```

### Step 3: After Completion (~34-39 hours)

Compare results:

**Vision:**
```bash
# Baseline vs Tuned accuracies
ls experiments/vision/results_nt*/resnet34_cifar100/*milo.json
# Compare mean accuracies in JSON files
```

**NLP:**
```bash
# Baseline vs Tuned accuracies
diff <(jq '.[] | .best_val_accuracy' experiments/nlp/results_nt_bert/bert_sst2_milo.json) \
     <(jq '.[] | .best_val_accuracy' experiments/nlp/results_nt_bert_tuned/bert_sst2_milo.json)
```

**ImageNet:**
```bash
# Baseline vs Tuned accuracies
ls experiments/imagenet/results_nt_imagenet200*/imagenet_*_milo.json
```

---

## Timeline Visualization

```
Hour 0                Hour 24               Hour 34-39
├─────────────────────┼─────────────────────┤
│ PHASE 1             │ PHASE 2             │
│ (Parallel)          │ (Sequential)        │
│                     │                     │
│ Baseline Vision ─────────────────┐       │
│ Baseline ImageNet ──────┐        │       │
│ Baseline NLP ────────────────────┼──────┐│
│ Tuning ──────────────────┼───────┼──────┤│
│                          │       │      ││
│                       ┌──▼──────▼──────▼┤
│                       │ Vision Tuned    │
│                       │ ImageNet Tuned  │
│                       │ NLP Tuned       │
│                       └─────────────────┘
```

---

## Configuration Files

The tuned hyperparameters are defined in each domain's config:

**Vision:** `experiments/vision/config.py`
```python
OPTIMIZER_PARAMS_TUNED = {
    "MILO": {"scale_factor": 0.25, ...},
    "ADAMW": {"betas": (0.95, 0.999), ...},
    ...
}
```

**NLP:** `experiments/nlp/config.py`
```python
OPTIMIZER_PARAMS_TUNED = {
    "MILO": {"scale_factor": 0.15, ...},
    ...
}
```

**ImageNet:** `experiments/imagenet/config.py`
```python
OPTIMIZER_PARAMS_TUNED = {
    "MILO": {"scale_factor": 0.2, ...},
    ...
}
```

You can manually update these based on tuning results, or implement automated parameter extraction from tuning output.

---

## Tuned Experiment Scripts

Each domain has a tuned variant:

- `experiments/vision/vision_experiment_tuned.py`
- `experiments/nlp/nlp_experiment_tuned.py`
- `experiments/imagenet/imagenet_experiment_tuned.py`

These scripts:
1. Import tuned parameters from config
2. Use `RESULTS_DIR_TUNED` for output
3. Run the same experiments with optimized hyperparameters

You can also run these independently:
```bash
python experiments/vision/vision_experiment_tuned.py
python experiments/nlp/nlp_experiment_tuned.py
python experiments/imagenet/imagenet_experiment_tuned.py
```

---

## Manual SLURM Submission (Alternative)

If you want more control, submit phases manually:

**Phase 1 - Baseline (all parallel):**
```bash
sbatch hpc/experiments/run_vision_experiments.slurm
sbatch hpc/experiments/run_imagenet_experiments.slurm
sbatch hpc/experiments/run_nlp_experiments.slurm
sbatch hpc/experiments/run_hyperparameter_tuning.slurm
```

Get job IDs from output, then:

**Phase 2 - Tuned (with dependencies):**
```bash
# After getting VISION_BASE, IMAGENET_BASE, NLP_BASE, TUNE_JID
VISION_BASE=123
IMAGENET_BASE=124
NLP_BASE=125
TUNE_JID=126

# Run tuned with dependency on tuning completion
sbatch --dependency=afterok:${TUNE_JID} \
    hpc/experiments/run_vision_experiments_tuned.slurm
sbatch --dependency=afterok:${TUNE_JID} \
    hpc/experiments/run_imagenet_experiments_tuned.slurm
sbatch --dependency=afterok:${TUNE_JID} \
    hpc/experiments/run_nlp_experiments_tuned.slurm
```

---

## Troubleshooting

**Tuned jobs not starting after tuning completes:**
```bash
# Check dependency status
squeue -j <TUNED_JOB_ID> --format=JobID,Name,State,Reason

# Check if tuning job succeeded
sacct -j <TUNING_JID> --format=JobID,State
```

**Missing results in _tuned directories:**
```bash
# Verify tuned scripts ran
tail -f logs/vision_tuned_*.out
tail -f logs/nlp_tuned_*.out
tail -f logs/imagenet_tuned_*.out
```

**Want to rerun just Phase 2:**
```bash
# Tuned experiments can be resubmitted independently
sbatch hpc/experiments/run_vision_experiments_tuned.slurm
sbatch hpc/experiments/run_nlp_experiments_tuned.slurm
sbatch hpc/experiments/run_imagenet_experiments_tuned.slurm
```

---

## Output Structure After Completion

```
experiments/
├── vision/
│   ├── results_nt/                    # Baseline results
│   │   └── resnet34_cifar100/
│   │       ├── vision_*_milo.json
│   │       ├── vision_*_adamw.json
│   │       └── ...
│   └── results_nt_tuned/              # Tuned results
│       └── resnet34_cifar100/
│           ├── vision_*_milo.json
│           ├── vision_*_adamw.json
│           └── ...
│
├── nlp/
│   ├── results_nt_bert/               # Baseline results
│   │   ├── bert_sst2_milo.json
│   │   ├── bert_sst2_adamw.json
│   │   └── ...
│   └── results_nt_bert_tuned/         # Tuned results
│       ├── bert_sst2_milo.json
│       ├── bert_sst2_adamw.json
│       └── ...
│
└── imagenet/
    ├── results_nt_imagenet200/        # Baseline results
    │   ├── imagenet_100_resnet34_milo.json
    │   ├── imagenet_100_resnet34_adamw.json
    │   └── ...
    └── results_nt_imagenet200_tuned/  # Tuned results
        ├── imagenet_100_resnet34_milo.json
        ├── imagenet_100_resnet34_adamw.json
        └── ...
```

Each JSON file contains:
```json
[
  {
    "run": 1,
    "optimizer": "MILO",
    "best_val_accuracy": 42.5,
    "training_time_seconds": 3600
  },
  ...
]
```

---

## Analysis Script (Optional)

To compare results programmatically:

```python
import json
import numpy as np

def compare_results(baseline_dir, tuned_dir, domain):
    optimizers = ["MILO", "ADAMW", "SGD", ...]
    
    for opt in optimizers:
        baseline_file = f"{baseline_dir}/results_*_{opt.lower()}.json"
        tuned_file = f"{tuned_dir}/results_*_{opt.lower()}.json"
        
        with open(baseline_file) as f:
            baseline = json.load(f)
        with open(tuned_file) as f:
            tuned = json.load(f)
        
        base_acc = np.mean([r["best_val_accuracy"] for r in baseline])
        tuned_acc = np.mean([r["best_val_accuracy"] for r in tuned])
        improvement = ((tuned_acc - base_acc) / base_acc) * 100
        
        print(f"{domain} {opt}: {base_acc:.2f}% → {tuned_acc:.2f}% ({improvement:+.1f}%)")
```

---

## Key Points

✓ All phases run with a single command: `bash hpc/submit_all_phases.sh`
✓ Phase 1 jobs run in parallel (~24h)
✓ Phase 2 jobs wait for Phase 1 with SLURM dependencies
✓ Results in separate directories for easy comparison
✓ Tuned hyperparameters in `OPTIMIZER_PARAMS_TUNED` (can be automated from tuning output)
✓ Total runtime: ~34-39 hours
✓ Baseline vs Tuned comparison built into result structure

