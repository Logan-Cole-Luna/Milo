#!/usr/bin/env bash
# ============================================================
# Milo — Supervised Learning Experiment Suite
#
# Trains models from the supervised_learning/ experiment suite
# using specified optimizers on Compute Canada HPC.
#
# Submit:
#   sbatch hpc/supervised_learning.slurm
#
# Override examples:
#   EPOCHS=10 sbatch hpc/supervised_learning.slurm
#   EXPERIMENTS="LOGISTIC MULTILAYER" sbatch hpc/supervised_learning.slurm
#   OPTIMIZERS="SGD ADAMW" sbatch hpc/supervised_learning.slurm
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_supervised
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/logan03/Milo/logs/supervised_%A.out
#SBATCH --error=/home/logan03/Milo/logs/supervised_%A.err
#SBATCH --mail-user=logan03luna@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$SCRATCH/Milo}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

# --- Configuration variables (override with environment) ---
EXPERIMENTS="${EXPERIMENTS:-VGG11_CIFAR10 VGG11_CIFAR100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-5}"
RUNS_PER_OPTIMIZER="${RUNS_PER_OPTIMIZER:-1}"

echo "============================================================"
echo "  Milo — Supervised Learning Experiments"
echo "  Experiments:      ${EXPERIMENTS}"
echo "  Batch size:       ${BATCH_SIZE}"
echo "  Epochs:           ${EPOCHS}"
echo "  Runs per opt:     ${RUNS_PER_OPTIMIZER}"
echo "  Host:             $(hostname)"
echo "  GPU:              $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job:              ${SLURM_JOB_ID}"
echo "  Started:          $(date)"
echo "============================================================"

cd "${PROJECT_ROOT}"

# Run the supervised learning experiment
python experiments/supervised_learning/supervised_learning_experiment.py \
    --experiments ${EXPERIMENTS}

echo "Finished: $(date)"
