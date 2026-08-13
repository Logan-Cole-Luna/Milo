#!/usr/bin/env bash
# ============================================================
# Milo — Comprehensive Experiment Suite (Option C)
#
# Runs all three experiment suites in coordinated fashion:
# 1. Vision experiments (CIFAR, including ViT)
# 2. Large-scale vision (ImageNet-100)
# 3. NLP fine-tuning (BERT on SST-2)
#
# Submit:
#   sbatch hpc/run_comprehensive_suite_c.slurm
#
# Monitor:
#   tail -f logs/comprehensive_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_comprehensive_c
#SBATCH --gpus-per-node=2    # Change to 2 or 4 for faster parallel execution
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=/home/logan03/Milo/logs/comprehensive_c_%A.out
#SBATCH --error=/home/logan03/Milo/logs/comprehensive_c_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

# Ensure offline mode (compute nodes have no internet)
# Models must be pre-cached by running: bash hpc/prepare_offline_models.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "============================================================"
echo "  Milo — Comprehensive Experiment Suite (Option C)"
echo "============================================================"
echo "Configuration:"
echo "  PHASE 1: Vision Experiments (CIFAR with ViT)"
echo "    - LOGISTIC, MULTILAYER (2)"
echo "    - ResNet34 CIFAR-10/100 (2)"
echo "    - VGG11 CIFAR-10/100 (2)"
echo "    - ViT-Tiny CIFAR-10/100 (2)"
echo "    - Subtotal: 8 experiments × 7 optimizers × 5 runs = 280 runs"
echo ""
echo "  PHASE 2: Large-Scale Vision (ImageNet-100)"
echo "    - ResNet50 on ImageNet-100 (1)"
echo "    - Optimizers: MILO, MILO_LW, SGD, AdamW (4)"
echo "    - Runs: 3 (reduced from 5 to manage time)"
echo "    - Subtotal: 1 experiment × 4 optimizers × 3 runs = 12 runs"
echo ""
echo "  PHASE 3: NLP Fine-tuning (BERT on SST-2)"
echo "    - BERT-base on SST-2 sentiment classification (1) ← NEW"
echo "    - Optimizers: MILO, MILO_LW, SGD, AdamW (4)"
echo "    - Runs: 5"
echo "    - Subtotal: 1 experiment × 4 optimizers × 5 runs = 20 runs"
echo ""
echo "  TOTAL: 10 experiments, 15 optimizer variants, 312 individual runs"
echo ""
echo "System Information:"
echo "  Host:                     $(hostname)"
echo "  GPU:                      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID:                   ${SLURM_JOB_ID}"
echo "  Started:                  $(date)"
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}"

# ============================================================
# Pre-Flight Check: Ensure Datasets are Available
# ============================================================
echo ""
echo "Checking for pre-downloaded datasets..."
echo ""

# Check MNIST (optional but preferred)
if [ ! -d "$HOME/scratch/datasets/mnist" ]; then
    echo "⚠ MNIST not pre-downloaded (will attempt download on compute node)"
    echo "  To pre-download: bash hpc/download_all_datasets.sh"
fi

# Check CIFAR-10
if [ ! -d "$HOME/scratch/datasets/cifar10" ]; then
    echo "⚠ CIFAR-10 not pre-downloaded (will attempt download on compute node)"
fi

# Check CIFAR-100
if [ ! -d "$HOME/scratch/datasets/cifar100" ]; then
    echo "⚠ CIFAR-100 not pre-downloaded (will attempt download on compute node)"
fi

echo ""

# ============================================================
# PHASE 1: Vision Experiments (CIFAR, with ViT)
# ============================================================
echo ""
echo "=============== PHASE 1: Vision Experiments ==============="
echo "Starting vision experiments (LOGISTIC, MULTILAYER, ResNets, VGG11, ViT-Tiny)"
echo "Time: $(date)"
echo ""

python experiments/supervised_learning/supervised_learning_experiment.py \
    --experiments LOGISTIC MULTILAYER RESNET34_CIFAR10 RESNET34_CIFAR100 \
                 VGG11_CIFAR10 VGG11_CIFAR100 VIT_TINY_CIFAR10 VIT_TINY_CIFAR100

if [ $? -eq 0 ]; then
    echo "✓ PHASE 1 completed successfully"
else
    echo "✗ PHASE 1 failed with error code $?"
    exit 1
fi

echo ""
echo "=============== PHASE 2: Large-Scale Vision ==============="
echo "Starting ImageNet-100 experiments (ResNet50)"
echo "Time: $(date)"
echo ""

# Check if ImageNet data is available
IMAGENET_PATH="${IMAGENET_DATA:-/scratch/datasets/imagenet}"
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "⚠ ImageNet data not found at $IMAGENET_PATH"
    echo "  Skipping Phase 2 (ImageNet-100 not available)"
    echo "  Note: Full results will only include CIFAR/ViT experiments"
else
    python experiments/supervised_learning/imagenet_100_experiment.py

    if [ $? -eq 0 ]; then
        echo "✓ PHASE 2 completed successfully"
    else
        echo "⚠ PHASE 2 had errors but continuing to Phase 3"
    fi
fi

echo ""
echo "=============== PHASE 3: NLP Fine-tuning ==============="
echo "Starting BERT fine-tuning on SST-2"
echo "Time: $(date)"
echo ""

# Install NLP dependencies if needed
python -c "import transformers" 2>/dev/null || {
    echo "Installing transformers library..."
    pip install -q transformers datasets evaluate accelerate
}

python experiments/nlp_experiment.py

if [ $? -eq 0 ]; then
    echo "✓ PHASE 3 completed successfully"
else
    echo "⚠ PHASE 3 had errors"
fi

echo ""
echo "============================================================"
echo "  Comprehensive Suite Completed"
echo "============================================================"
echo ""
echo "Results Locations:"
echo "  Vision (CIFAR + ViT):  experiments/supervised_learning/*/results_nt/"
echo "  Large-scale Vision:    experiments/supervised_learning/results_nt_imagenet100/"
echo "  NLP Fine-tuning:       experiments/results_nt_bert/"
echo ""
echo "Next Steps:"
echo "  1. Check all result directories for .csv files"
echo "  2. Run analysis script to aggregate results"
echo "  3. Generate comparison plots"
echo ""
echo "Finished: $(date)"
echo "============================================================"
