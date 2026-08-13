#!/usr/bin/env bash
# Final 1B FT run using (lr[, beta]) from a prior Optuna sweep
# (results_optuna/ft_<OPT>.json). beta -> ORTHO_STRENGTH env var for MION_GATED.
# Chain via: sbatch --dependency=afterok:<jobid> --export=ALL,OPT=... this script.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=ft_final
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/home/logan03/Milo/logs/ft_final_%x_%A.out
#SBATCH --error=/home/logan03/Milo/logs/ft_final_%x_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${PROJECT_ROOT}"

read -r LR BETA <<< "$(python -c "
import json
d = json.load(open('results_optuna/ft_${OPT}.json'))
bp = d.get('best_params', {})
print(d['best_lr'], bp.get('beta', ''))
")"
if [ -n "${BETA}" ]; then
  export ORTHO_STRENGTH="${BETA}"
fi
echo "FT final run | OPT=${OPT} tuned LR=${LR} ORTHO_STRENGTH=${ORTHO_STRENGTH:-<default>} | Job ${SLURM_JOB_ID} | $(date)"
python -u -m experiments.lm.finetune --optimizer "${OPT}" --lr "${LR}" --steps "${STEPS:-600}"
echo "Done: $(date)"
