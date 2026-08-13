#!/usr/bin/env bash
# Run ImageNet for a single optimizer (set via --export OPT_ONLY=NAME).
#   sbatch --export=ALL,OPT_ONLY=SHAMPOO -J inet_shampoo hpc/experiments/run_imagenet_one.slurm

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=inet_one
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=5:00:00
#SBATCH --output=/home/logan03/Milo/logs/inet_one_%x_%A.out
#SBATCH --error=/home/logan03/Milo/logs/inet_one_%x_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

IMAGENET_PATH="${IMAGENET_DATA:-${HOME}/scratch/datasets/tiny-imagenet-200}"
[ -d "$IMAGENET_PATH" ] || { echo "✗ Tiny ImageNet not found at $IMAGENET_PATH"; exit 1; }

echo "ImageNet run | OPT_ONLY=${OPT_ONLY:-<all>} | LR_GRID=${LR_GRID:-<config>} | Job ${SLURM_JOB_ID} | $(date)"
cd "${PROJECT_ROOT}"
if [ -n "${LR_GRID:-}" ]; then
  for lr in ${LR_GRID//,/ }; do
    echo "===== sweep ${OPT_ONLY} lr=${lr} ====="
    LR_OVERRIDE=$lr SWEEP_RUNS="${SWEEP_RUNS:-1}" \
      RESULTS_DIR_OVERRIDE="results_sweep_imagenet/${OPT_ONLY}_lr${lr}" \
      python -u experiments/imagenet/imagenet_experiment.py || echo "  ⚠ lr=${lr} failed"
  done
else
  python -u experiments/imagenet/imagenet_experiment.py
fi
echo "Done: $(date)"
