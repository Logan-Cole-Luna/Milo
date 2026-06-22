#!/usr/bin/env bash
# ============================================================
# Tiny ImageNet-200 - Tuned Hyperparameters
#
# Runs AFTER tuning phase completes.
# Uses optimized hyperparameters found during tuning.
#
# Submit: sbatch hpc/experiments/run_imagenet_experiments_tuned.slurm
# Monitor: tail -f logs/imagenet_tuned_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_imagenet_tuned
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/logan03/Milo/logs/imagenet_tuned_%A.out
#SBATCH --error=/home/logan03/Milo/logs/imagenet_tuned_%A.err
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
echo "  Tiny ImageNet-200 - Tuned Hyperparameters"
echo "============================================================"
echo "Configuration:"
echo "  - ResNet34 on Tiny ImageNet-200 (64×64, 200 classes)"
echo "  - 11 optimizers with TUNED hyperparameters"
echo "  - 3 runs per optimizer"
echo "  - Compare with baseline results_nt_imagenet200/"
echo ""
echo "System Information:"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}"

# Check if Tiny ImageNet is available
IMAGENET_PATH="${IMAGENET_DATA:-/scratch/datasets/tiny-imagenet-200}"
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "✗ Tiny ImageNet data not found at $IMAGENET_PATH"
    echo "  Please download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    exit 1
fi

python experiments/imagenet/imagenet_experiment_tuned.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ ImageNet tuned experiments completed successfully"
else
    echo ""
    echo "✗ ImageNet tuned experiments failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "  ImageNet Tuned Experiments Complete"
echo "============================================================"
echo "Results:"
echo "  Baseline: experiments/imagenet/results_nt_imagenet200/"
echo "  Tuned:    experiments/imagenet/results_nt_imagenet200_tuned/"
echo "Finished: $(date)"
echo "============================================================"
