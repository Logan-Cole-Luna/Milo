#!/bin/bash
# ============================================================
# Milo — Initial HPC setup on Compute Canada login node
#
# Run this once on the login node before submitting jobs:
#   bash hpc/setup_cc.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
VENV_DIR="${PROJECT_ROOT}/.venv_cc"

echo "============================================================"
echo "  Milo — Compute Canada Setup"
echo "  Project root: ${PROJECT_ROOT}"
echo "============================================================"

# Load modules
module --force purge
module load StdEnv/2023
module load cuda/12.2 cudnn/9.2.1.18
module load python/3.11.5 scipy-stack/2024a
module load gcc arrow/23.0.1

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    python -m venv "${VENV_DIR}"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

# Activate the virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements..."
if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    pip install -r "${PROJECT_ROOT}/requirements.txt"
    echo "Requirements installed from requirements.txt"
else
    echo "WARNING: No requirements.txt found. Installing common ML packages..."
    pip install torch transformers numpy scipy matplotlib seaborn tensorboard
fi

echo "============================================================"
echo "  Setup Complete!"
echo "  Next steps:"
echo "    1. Submit jobs with: sbatch hpc/supervised_learning.slurm"
echo "    2. Check job status with: squeue -u \$USER"
echo "    3. Check logs in: logs/"
echo "============================================================"
