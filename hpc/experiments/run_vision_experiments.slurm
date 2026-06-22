#!/usr/bin/env bash
# ============================================================
# Vision Experiments: CIFAR-10/100 with ResNet34, VGG11, ViT-Tiny
#
# Submit: sbatch hpc/experiments/run_vision_experiments.slurm
# Monitor: tail -f logs/vision_experiments_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_vision
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/logan03/Milo/logs/vision_experiments_%A.out
#SBATCH --error=/home/logan03/Milo/logs/vision_experiments_%A.err
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
echo "  PHASE 1: Vision Experiments (CIFAR with ViT)"
echo "============================================================"
echo "Configuration:"
echo "  - LOGISTIC (MNIST)"
echo "  - MULTILAYER (MNIST)"
echo "  - ResNet34 CIFAR-10/100 (2)"
echo "  - VGG11 CIFAR-10/100 (2)"
echo "  - ViT-Tiny CIFAR-10/100 (2)"
echo "  Subtotal: 8 experiments × 7 optimizers × 5 runs = 280 runs"
echo ""
echo "System Information:"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}"

python experiments/vision/vision_experiment.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Vision experiments completed successfully"
else
    echo ""
    echo "✗ Vision experiments failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Vision Experiments Complete"
echo "============================================================"
echo "Results: experiments/vision/results_nt/"
echo "Finished: $(date)"
echo "============================================================"
