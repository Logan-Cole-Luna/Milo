#!/usr/bin/env bash
# ============================================================
# NLP Fine-tuning - Tuned Hyperparameters
#
# Runs AFTER tuning phase completes.
# Uses optimized hyperparameters found during tuning.
#
# Submit: sbatch hpc/experiments/run_nlp_experiments_tuned.slurm
# Monitor: tail -f logs/nlp_tuned_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_nlp_tuned
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home/logan03/Milo/logs/nlp_tuned_%A.out
#SBATCH --error=/home/logan03/Milo/logs/nlp_tuned_%A.err
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
echo "  NLP Fine-tuning - Tuned Hyperparameters"
echo "============================================================"
echo "Configuration:"
echo "  - BERT-base on SST-2 sentiment classification"
echo "  - 11 optimizers with TUNED hyperparameters"
echo "  - 5 runs per optimizer"
echo "  - Compare with baseline results_nt_bert/"
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

python experiments/nlp/nlp_experiment_tuned.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ NLP tuned experiments completed successfully"
else
    echo ""
    echo "✗ NLP tuned experiments failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "  NLP Tuned Experiments Complete"
echo "============================================================"
echo "Results:"
echo "  Baseline: experiments/nlp/results_nt_bert/"
echo "  Tuned:    experiments/nlp/results_nt_bert_tuned/"
echo "Finished: $(date)"
echo "============================================================"
