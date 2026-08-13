#!/usr/bin/env bash
# ============================================================
# NLP Fine-tuning Experiments: BERT on SST-2
#
# Submit: sbatch hpc/experiments/run_nlp_experiments.slurm
# Monitor: tail -f logs/nlp_experiments_*.out
#
# Note: Requires BERT model and SST-2 dataset pre-cached
#       Run: bash hpc/prepare_offline_models.sh (on login node)
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_nlp
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home/logan03/Milo/logs/nlp_experiments_%A.out
#SBATCH --error=/home/logan03/Milo/logs/nlp_experiments_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

# Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "============================================================"
echo "  PHASE 3: NLP Fine-tuning (BERT on SST-2)"
echo "============================================================"
echo "Configuration:"
echo "  - BERT-base on SST-2 sentiment classification"
echo "  - Optimizers: 11 (MILO, MILO_LW, SGD, AdamW, AdaGrad, Lion, Adam-mini, RMSprop-Momentum, Shampoo, SOAP, Muon)"
echo "  - Runs: 5 per optimizer"
echo "  - Epochs: 3"
echo "  Subtotal: 1 experiment × 11 optimizers × 5 runs = 55 runs"
echo ""
echo "System Information:"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}"

# Install NLP dependencies if needed
python -c "import transformers" 2>/dev/null || {
    echo "Installing transformers library..."
    pip install -q transformers datasets evaluate accelerate
}

python experiments/nlp/nlp_experiment.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ NLP experiments completed successfully"
else
    echo ""
    echo "✗ NLP experiments failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "  NLP Experiments Complete"
echo "============================================================"
echo "Results: experiments/nlp/results_nt_bert/"
echo "Finished: $(date)"
echo "============================================================"
