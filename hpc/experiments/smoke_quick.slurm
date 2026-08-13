#!/usr/bin/env bash
# ============================================================
# Quick smoke: every optimizer x every experiment, 2 steps each.
# Confirms load + run for the full matrix. Fast (minutes on 1 GPU).
#
# Submit: sbatch hpc/experiments/smoke_quick.slurm
# Monitor: tail -f logs/smoke_quick_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=smoke_quick
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/logan03/Milo/logs/smoke_quick_%A.out
#SBATCH --error=/home/logan03/Milo/logs/smoke_quick_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "============================================================"
echo "  QUICK SMOKE: optimizer x experiment matrix (2 steps each)"
echo "  Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "============================================================"

cd "${PROJECT_ROOT}"
# -u: unbuffered stdout so per-cell progress streams live to the log
python -u hpc/experiments/smoke_quick.py --domains vision imagenet nlp

echo ""
echo "Finished: $(date)"
