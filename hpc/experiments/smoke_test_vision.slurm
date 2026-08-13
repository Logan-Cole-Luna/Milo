#!/usr/bin/env bash
# ============================================================
# Smoke Test: Vision Experiments (2 epochs, 2 runs, subset of optimizers)
#
# Validates that all 13 optimizers instantiate and train correctly
# before the full 5-epoch, 5-run job completes.
#
# Submit: sbatch hpc/experiments/smoke_test_vision.slurm
# Monitor: tail -f logs/smoke_vision_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=smoke_vision
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/home/logan03/Milo/logs/smoke_vision_%A.out
#SBATCH --error=/home/logan03/Milo/logs/smoke_vision_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

echo "============================================================"
echo "  SMOKE TEST: Vision Experiments"
echo "  2 epochs, 2 runs, all 13 optimizers, ResNet34 CIFAR-10 only"
echo "============================================================"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# Override config — use LOGISTIC (fastest model) to validate all optimizers quickly
import experiments.vision.config as cfg
cfg.EPOCHS = 2
cfg.RUNS_PER_OPTIMIZER = 1
cfg.FINAL_SUITE = ["LOGISTIC"]
cfg.EXPERIMENTS = [cfg.FINAL_SUITE]

from experiments.vision.vision_experiment import run_supervised_experiments
run_supervised_experiments(["LOGISTIC"])

print("\n✓ Vision smoke test passed — all optimizers ran successfully")
PYEOF

echo ""
echo "============================================================"
echo "  Smoke Test Complete: $(date)"
echo "============================================================"
