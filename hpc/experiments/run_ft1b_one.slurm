#!/usr/bin/env bash
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=ft1b
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/home/logan03/Milo/logs/ft1b_%x_%A.out
#SBATCH --error=/home/logan03/Milo/logs/ft1b_%x_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL
set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${PROJECT_ROOT}"
echo "FT1B | OPT=${OPT} LR=${LR:-default} | Job ${SLURM_JOB_ID} | $(date)"
python -u -m experiments.lm.finetune --optimizer "${OPT}" --steps "${STEPS:-600}" ${LR:+--lr "$LR"}
echo "Done: $(date)"
