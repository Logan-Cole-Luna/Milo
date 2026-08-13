#!/usr/bin/env bash
# Optimizer cost benchmark (memory + throughput) on GPU. Set MODEL via --export.
#   sbatch --export=ALL,MODEL=small hpc/experiments/run_cost_benchmark.slurm
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=opt_cost
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/home/logan03/Milo/logs/opt_cost_%x_%A.out
#SBATCH --error=/home/logan03/Milo/logs/opt_cost_%x_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"
cd "${PROJECT_ROOT}"
python -u -m benchmarks.optimizer_cost --model "${MODEL:-small}" --steps "${STEPS:-30}"
