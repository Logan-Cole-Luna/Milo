#!/bin/bash
# ============================================================
# Milo — Compute Canada shared environment for SLURM jobs.
# Source this at the top of every job script:
#   source /path/to/Milo/hpc/cc_env.sh
#
# Prerequisites:
#   1. Run  bash hpc/setup_cc.sh  once on the login node
#   2. Ensure Python environment is set up with required packages
# ============================================================

# --- Modules (standard Compute Canada stack) ---
module --force purge
module load StdEnv/2023
module load cuda/12.2 cudnn/9.2.1.18
module load python/3.11.5 scipy-stack/2024a
module load gcc arrow/23.0.1

# --- Project root ---
# Uses SLURM_SUBMIT_DIR if submitted from Milo/, else falls back to current directory
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# --- Activate the persistent CC venv ---
source "${PROJECT_ROOT}/.venv_cc/bin/activate"

# --- HF offline mode (compute nodes have no internet) ---
# Prefer shared scratch cache if present, else fall back to repo-local cache.
DEFAULT_SHARED_HF_HOME="${HOME}/scratch/.cache/huggingface"
if [[ -d "${DEFAULT_SHARED_HF_HOME}/hub" ]]; then
    export HF_HOME="${HF_HOME:-${DEFAULT_SHARED_HF_HOME}}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${DEFAULT_SHARED_HF_HOME}/hub}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DEFAULT_SHARED_HF_HOME}/datasets}"
else
    export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${PROJECT_ROOT}/.hf_cache/hub}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/.hf_cache/datasets}"
fi
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_TOKEN=$(cat "$HOME/.hf_token" 2>/dev/null || true)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# --- Move into project directory ---
cd "${PROJECT_ROOT}"

echo "[cc_env] Project root: ${PROJECT_ROOT}"
echo "[cc_env] HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "[cc_env] Python: $(which python) ($(python --version 2>&1))"
echo "[cc_env] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
