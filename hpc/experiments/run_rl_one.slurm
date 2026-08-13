#!/usr/bin/env bash
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=rl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=/home/logan03/Milo/logs/rl_%x_%A.out
#SBATCH --error=/home/logan03/Milo/logs/rl_%x_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL
set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
cd "${PROJECT_ROOT}"
echo "RL | ENV=${ENVNAME} OPT=${OPT} | Job ${SLURM_JOB_ID} | $(date)"
python -u -m experiments.rl.reinforce --env "${ENVNAME}" --optimizer "${OPT}" --episodes "${EPISODES:-600}" --seeds "${SEEDS:-3}"
echo "Done: $(date)"
