#!/usr/bin/env bash
# ============================================================
# Milo — Full Supervised Learning Experiment Suite
#
# Fresh run of all experiments with:
# - Two MILO variants: network-wise and layer-wise
# - Corrected hyperparameters (scale_aware=True, max_group_size=5000)
# - 5 runs per optimizer for statistical validity
# - 6 model/dataset combinations
#
# Submit:
#   sbatch hpc/run_full_experiments.slurm
#
# Monitor:
#   tail -f logs/supervised_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_full_suite
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/logan03/Milo/logs/supervised_full_%A.out
#SBATCH --error=/home/logan03/Milo/logs/supervised_full_%A.err
#SBATCH --mail-user=logan03luna@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

# --- Full experiment suite configuration ---
# All 6 experiments with proper statistical rigor
EXPERIMENTS="LOGISTIC MULTILAYER RESNET34_CIFAR10 RESNET34_CIFAR100 VGG11_CIFAR10 VGG11_CIFAR100"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-5}"
RUNS_PER_OPTIMIZER="${RUNS_PER_OPTIMIZER:-5}"

echo "============================================================"
echo "  Milo — Full Supervised Learning Experiment Suite"
echo "============================================================"
echo "Configuration:"
echo "  Experiments:           ${EXPERIMENTS}"
echo "  Total experiments:     6"
echo "  Optimizers:            MILO, MILO_LW, SGD, ADAMW, ADAGRAD, ADEMAMIX, SOAP"
echo "  Runs per optimizer:    ${RUNS_PER_OPTIMIZER}"
echo "  Batch size:            ${BATCH_SIZE}"
echo "  Epochs:                ${EPOCHS}"
echo "  Total runs:            $((6 * 7 * RUNS_PER_OPTIMIZER)) (42 experiment runs)"
echo ""
echo "Key Changes:"
echo "  ✓ MILO scale_aware=True (was False - caused convergence issues)"
echo "  ✓ MILO max_group_size=5000 (was None/dynamic - over-aggressive)"
echo "  ✓ MILO use_cuda_kernels=True (performance optimization)"
echo "  ✓ 5 runs per optimizer (was 1 - enables statistical tests)"
echo ""
echo "System Information:"
echo "  Host:                  $(hostname)"
echo "  GPU:                   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID:                ${SLURM_JOB_ID}"
echo "  Started:               $(date)"
echo "============================================================"

cd "${PROJECT_ROOT}"

# Run the full supervised learning experiment suite
python experiments/supervised_learning/supervised_learning_experiment.py \
    --experiments ${EXPERIMENTS}

echo ""
echo "============================================================"
echo "Experiment suite completed: $(date)"
echo "Results location: experiments/supervised_learning/*/results_nt/"
echo "============================================================"
