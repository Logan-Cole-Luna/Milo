#!/usr/bin/env bash
# One Optuna LR study (set DOMAIN, OPT, optionally EXP, TRIALS via --export=ALL).
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=optuna_one
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=/home/logan03/Milo/logs/optuna_%x_%A.out
#SBATCH --error=/home/logan03/Milo/logs/optuna_%x_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Optuna study | DOMAIN=${DOMAIN} OPT=${OPT} EXP=${EXP:-} TRIALS=${TRIALS:-15} | Job ${SLURM_JOB_ID} | $(date)"
cd "${PROJECT_ROOT}"
python -u hpc/experiments/optuna_sweep.py \
    --domain "${DOMAIN}" --optimizer "${OPT}" \
    ${EXP:+--experiment "${EXP}"} --trials "${TRIALS:-15}"
echo "Done: $(date)"
