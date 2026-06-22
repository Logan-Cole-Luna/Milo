#!/usr/bin/env bash
# ============================================================
# Vision Experiments - Tuned Hyperparameters
#
# Runs AFTER tuning phase completes.
# Uses optimized hyperparameters found during tuning.
#
# Submit: sbatch hpc/experiments/run_vision_experiments_tuned.slurm
# Monitor: tail -f logs/vision_tuned_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_vision_tuned
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/logan03/Milo/logs/vision_tuned_%A.out
#SBATCH --error=/home/logan03/Milo/logs/vision_tuned_%A.err
#SBATCH --mail-user=logan03luna@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

# Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "============================================================"
echo "  Vision Experiments - Tuned Hyperparameters"
echo "============================================================"
echo "Configuration:"
echo "  - 8 experiments (Logistic, MLP, ResNet34×2, VGG11×2, ViT-Tiny×2)"
echo "  - 11 optimizers with TUNED hyperparameters"
echo "  - 5 runs per optimizer"
echo "  - Compare with baseline results_nt/"
echo ""
echo "System Information:"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}"

python experiments/vision/vision_experiment_tuned.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Vision tuned experiments completed successfully"
else
    echo ""
    echo "✗ Vision tuned experiments failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Vision Tuned Experiments Complete"
echo "============================================================"
echo "Results:"
echo "  Baseline: experiments/vision/results_nt/"
echo "  Tuned:    experiments/vision/results_nt_tuned/"
echo "Finished: $(date)"
echo "============================================================"
