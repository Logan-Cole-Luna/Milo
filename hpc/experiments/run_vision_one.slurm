#!/usr/bin/env bash
# Run Vision for a single experiment (set via --export EXP_ONLY=NAME).
# Enables parallel per-experiment submission across the free cluster.
#   sbatch --export=ALL,EXP_ONLY=RESNET34_CIFAR10 -J vis_rn34c10 hpc/experiments/run_vision_one.slurm

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=vis_one
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/logan03/Milo/logs/vis_one_%x_%A.out
#SBATCH --error=/home/logan03/Milo/logs/vis_one_%x_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Vision run | EXP_ONLY=${EXP_ONLY:?must set EXP_ONLY} | OPT_ONLY=${OPT_ONLY:-<all>} | LR_GRID=${LR_GRID:-<config>} | Job ${SLURM_JOB_ID} | $(date)"
cd "${PROJECT_ROOT}"
if [ -n "${LR_GRID:-}" ]; then
  for lr in ${LR_GRID//,/ }; do
    echo "===== sweep ${EXP_ONLY}/${OPT_ONLY} lr=${lr} ====="
    LR_OVERRIDE=$lr SWEEP_RUNS="${SWEEP_RUNS:-1}" \
      RESULTS_DIR_OVERRIDE="results_sweep/${EXP_ONLY}_${OPT_ONLY}_lr${lr}" \
      python -u experiments/vision/vision_experiment.py --experiments "${EXP_ONLY}" || echo "  ⚠ lr=${lr} failed"
  done
else
  python -u experiments/vision/vision_experiment.py --experiments "${EXP_ONLY}"
fi
echo "Done: $(date)"
