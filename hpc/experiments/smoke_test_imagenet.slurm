#!/usr/bin/env bash
# ============================================================
# Smoke Test: ImageNet Experiments (2 epochs, 1 run, all 13 optimizers)
#
# Validates all optimizers work with Tiny ImageNet-200 before
# the full 5-epoch, 3-run job completes.
#
# Submit: sbatch hpc/experiments/smoke_test_imagenet.slurm
# Monitor: tail -f logs/smoke_imagenet_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=smoke_imagenet
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/home/logan03/Milo/logs/smoke_imagenet_%A.out
#SBATCH --error=/home/logan03/Milo/logs/smoke_imagenet_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

IMAGENET_PATH="${IMAGENET_DATA:-${HOME}/scratch/datasets/tiny-imagenet-200}"
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "✗ Tiny ImageNet not found at $IMAGENET_PATH"
    exit 1
fi

echo "============================================================"
echo "  SMOKE TEST: ImageNet Experiments"
echo "  2 epochs, 1 run, all 13 optimizers, ResNet34 on Tiny ImageNet"
echo "============================================================"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "  Data: $IMAGENET_PATH"
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import experiments.imagenet.config as cfg
cfg.EPOCHS = 2
cfg.RUNS_PER_OPTIMIZER = 1

from experiments.imagenet.imagenet_experiment import run_imagenet_experiment

data_root = os.path.expanduser("~/scratch/datasets")
failed = []

# SHAMPOO excluded: distributed_shampoo requires Python>=3.12 (cluster runs 3.11.5)
smoke_optimizers = [o for o in cfg.OPTIMIZERS if o != "SHAMPOO"]
for optimizer_name in smoke_optimizers:
    print(f"\n{'='*60}")
    print(f"  Smoke testing: {optimizer_name}")
    print(f"{'='*60}")
    try:
        run_imagenet_experiment(
            model_name="ResNet34",
            batch_size=cfg.BATCH_SIZE,
            epochs=2,
            learning_rate=cfg.LEARNING_RATES[optimizer_name],
            optimizer_name=optimizer_name,
            optimizer_params=cfg.OPTIMIZER_PARAMS[optimizer_name],
            runs=1,
            data_root=data_root,
            results_dir="results_smoke_imagenet",
        )
        print(f"✓ {optimizer_name} passed")
    except Exception as e:
        print(f"✗ {optimizer_name} FAILED: {e}")
        import traceback; traceback.print_exc()
        failed.append(optimizer_name)

print("\n" + "="*60)
if failed:
    print(f"SMOKE TEST FAILED — {len(failed)} optimizers failed: {failed}")
    sys.exit(1)
else:
    print(f"✓ All {len(cfg.OPTIMIZERS)} optimizers passed ImageNet smoke test")
PYEOF

echo ""
echo "============================================================"
echo "  Smoke Test Complete: $(date)"
echo "============================================================"
