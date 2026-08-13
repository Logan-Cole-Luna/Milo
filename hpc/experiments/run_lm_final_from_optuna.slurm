#!/usr/bin/env bash
# Final LM run using the LR from a prior Optuna sweep (results_optuna/lm_<OPT>.json).
# Chain via: sbatch --dependency=afterok:<jobid> --export=ALL,OPT=... this script.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=lm_final
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/logan03/Milo/logs/lm_final_%x_%A.out
#SBATCH --error=/home/logan03/Milo/logs/lm_final_%x_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${PROJECT_ROOT}"

LR=$(python -c "import json; print(json.load(open('results_optuna/lm_${OPT}.json'))['best_lr'])")
echo "LM final run | OPT=${OPT} MODEL=${MODEL:-small} TOKENS=${TOKENS:-1.2e9} tuned LR=${LR} | Job ${SLURM_JOB_ID} | $(date)"
python -u -m experiments.lm.train \
    --optimizer "${OPT}" --model "${MODEL:-small}" \
    --tokens "${TOKENS:-1.2e9}" --seed "${SEED:-1}" \
    --batch-size "${BATCH_SIZE:-16}" --grad-accum "${GRAD_ACCUM:-16}" --ctx "${CTX:-1024}" \
    --lr "${LR}" --out-dir "${OUT_DIR:-results/lm}"
echo "Done: $(date)"
