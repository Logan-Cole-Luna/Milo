#!/usr/bin/env bash
# ============================================================
# Hyperparameter Tuning for All Models
#
# Optimizes learning rates and other hyperparameters for:
# - All vision models (ResNet34, VGG11, ViT-Tiny)
# - CIFAR-10 and CIFAR-100
#
# Submit: sbatch hpc/experiments/run_hyperparameter_tuning.slurm
# Monitor: tail -f logs/hyperparameter_tuning_*.out
#
# This will run 5 trials per optimizer per model to find
# optimal hyperparameters for the comprehensive suite
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_hyperparameter_tuning
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/logan03/Milo/logs/hyperparameter_tuning_%A.out
#SBATCH --error=/home/logan03/Milo/logs/hyperparameterf_tuning_%A.err
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
echo "  Hyperparameter Tuning for Vision Models"
echo "============================================================"
echo "Configuration:"
echo "  - Models: ResNet34, VGG11, ViT-Tiny"
echo "  - Datasets: CIFAR-10, CIFAR-100"
echo "  - Optimizers: MILO, MILO_LW, SGD, ADAMW, ADAGRAD, ADEMAMIX, SOAP"
echo "  - Trials per combo: 5"
echo "  - Task: Find optimal learning rates for each optimizer"
echo ""
echo "System Information:"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}"

python experiments/vision/vision_experiment.py \
    --experiments RESNET34_CIFAR10 RESNET34_CIFAR100 VGG11_CIFAR10 VGG11_CIFAR100 VIT_TINY_CIFAR10 VIT_TINY_CIFAR100

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Hyperparameter tuning completed successfully"
else
    echo ""
    echo "✗ Hyperparameter tuning failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Tuning Complete"
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
