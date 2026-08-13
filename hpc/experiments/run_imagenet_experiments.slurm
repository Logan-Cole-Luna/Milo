#!/usr/bin/env bash
# ============================================================
# Tiny ImageNet-200 Large-Scale Vision Experiments
#
# Submit: sbatch hpc/experiments/run_imagenet_experiments.slurm
# Monitor: tail -f logs/imagenet_experiments_*.out
#
# Note: Tiny ImageNet must be available at /scratch/datasets/tiny-imagenet-200
#       Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_imagenet
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/logan03/Milo/logs/imagenet_experiments_%A.out
#SBATCH --error=/home/logan03/Milo/logs/imagenet_experiments_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
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
echo "  PHASE 2: Large-Scale Vision (Tiny ImageNet-200)"
echo "============================================================"
echo "Configuration:"
echo "  - ResNet34 on Tiny ImageNet-200 (64×64, 200 classes)"
echo "  - Optimizers: 11 (MILO, MILO_LW, SGD, AdamW, AdaGrad, Lion, Adam-mini, RMSprop-Momentum, Shampoo, SOAP, Muon)"
echo "  - Runs: 3 per optimizer"
echo "  Subtotal: 1 experiment × 11 optimizers × 3 runs = 33 runs"
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
IMAGENET_PATH="${IMAGENET_DATA:-${HOME}/scratch/datasets/tiny-imagenet-200}"
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "✗ Tiny ImageNet data not found at $IMAGENET_PATH"
    echo "  Please download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    exit 1
fi

python experiments/imagenet/imagenet_experiment.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ ImageNet experiments completed successfully"
else
    echo ""
    echo "✗ ImageNet experiments failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "  ImageNet Experiments Complete"
echo "============================================================"
echo "Results: experiments/imagenet/results_nt_imagenet200/"
echo "Finished: $(date)"
echo "============================================================"
