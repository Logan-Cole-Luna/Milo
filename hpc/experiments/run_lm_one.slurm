#!/usr/bin/env bash
# Train one LM optimizer run. Set OPT, MODEL, TOKENS, LR(optional), SEED via --export=ALL.
#   sbatch --export=ALL,OPT=MION,MODEL=small,TOKENS=1.2e9,SEED=1 -J lm_mion hpc/experiments/run_lm_one.slurm
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=lm_one
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/logan03/Milo/logs/lm_%x_%A.out
#SBATCH --error=/home/logan03/Milo/logs/lm_%x_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation
cd "${PROJECT_ROOT}"

# Per-device batch 16 (fits 40GB A100); accum 16 keeps effective tokens/step = 262k.
echo "LM run | OPT=${OPT} MODEL=${MODEL:-small} TOKENS=${TOKENS:-1.2e9} SEED=${SEED:-1} | Job ${SLURM_JOB_ID} | $(date)"
python -u -m experiments.lm.train \
    --optimizer "${OPT}" --model "${MODEL:-small}" \
    --tokens "${TOKENS:-1.2e9}" --seed "${SEED:-1}" \
    --batch-size "${BATCH_SIZE:-16}" --grad-accum "${GRAD_ACCUM:-16}" --ctx "${CTX:-1024}" \
    ${LR:+--lr "${LR}"} \
    --out-dir "${OUT_DIR:-results/lm}"
echo "Done: $(date)"
