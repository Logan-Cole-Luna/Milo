#!/usr/bin/env bash
# ============================================================
# MILO vs MILO Accelerated - Full Supervised Learning
#
# Compares both MILO variants on the full experiment suite.
# Uses the configured FINAL_SUITE experiments.
#
# Submit:
#   sbatch hpc/milo_experiment_comparison.slurm
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_exp_comp
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/logan03/Milo/logs/exp_comparison_%A.out
#SBATCH --error=/home/logan03/Milo/logs/exp_comparison_%A.err
#SBATCH --mail-user=logan03luna@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$SCRATCH/Milo}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

echo "============================================================"
echo "  MILO vs MILO Accelerated - Full Experiment Suite"
echo "  Host:             $(hostname)"
echo "  GPU:              $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job:              ${SLURM_JOB_ID}"
echo "  Started:          $(date)"
echo "============================================================"

cd "${PROJECT_ROOT}"

# Run focused comparison script (VGG11 experiments only)
echo ""
echo "============================================================"
echo "Running direct MILO vs MILO Accelerated comparison"
echo "============================================================"
echo ""

python run_milo_comparison.py 2>&1

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
