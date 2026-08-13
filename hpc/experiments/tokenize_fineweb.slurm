#!/usr/bin/env bash
# Tokenize downloaded FineWeb-Edu parquet -> uint16 .bin shards (CPU, no GPU).
# Run AFTER `data.py download` completes on the login node.
#   sbatch hpc/experiments/tokenize_fineweb.slurm
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=tok_fineweb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/logan03/Milo/logs/tok_fineweb_%A.out
#SBATCH --error=/home/logan03/Milo/logs/tok_fineweb_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"
# tiktoken GPT-2 vocab is pre-cached on the login node (compute nodes are offline)
export TIKTOKEN_CACHE_DIR="${HOME}/scratch/.cache/tiktoken"
cd "${PROJECT_ROOT}"
echo "Tokenizing FineWeb-Edu | target ${TARGET_TOKENS:-1.2e9} tokens | $(date)"
python -u experiments/lm/data.py tokenize --target-tokens "${TARGET_TOKENS:-1.2e9}"
echo "Done: $(date)"
