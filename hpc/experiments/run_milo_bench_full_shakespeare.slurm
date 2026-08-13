#!/usr/bin/env bash
# ============================================================
# Run Full Milo-Bench with Shakespeare Data
#
# LM + Vision benchmarks for MiloM, Mion, and baselines
# Uses Shakespeare data for LM (no FineWeb download needed)
#
# Submit: sbatch hpc/experiments/run_milo_bench_full_shakespeare.slurm
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_bench_full
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/logan03/Milo/logs/milo_bench_full_%A.out
#SBATCH --error=/home/logan03/Milo/logs/milo_bench_full_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"

TEMP_DIR="${PROJECT_ROOT}/temp"
mkdir -p "${PROJECT_ROOT}/logs/"

echo "============================================================"
echo "  Milo-Bench Full Suite (Shakespeare + Vision)"
echo "============================================================"
echo ""
echo "Benchmarking MiloM, Mion, and 14 baseline optimizers"
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

cd "$TEMP_DIR"

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')" || echo "⚠ TorchVision not available"
echo ""

# ============================================================
# LM Sweep with Shakespeare (modified for smaller data)
# ============================================================
echo "============================================================"
echo "  Phase 1: Language Model Pretraining (Shakespeare)"
echo "============================================================"
echo ""

if [ -f "sweep/run_sweep.py" ] && [ -f "configs/lm_sweep_small_shakespeare.yaml" ]; then
    echo "Running LM benchmark with Shakespeare data..."
    timeout 14400 python sweep/run_sweep.py configs/lm_sweep_small_shakespeare.yaml || echo "⚠ LM benchmark completed or timed out"
    echo ""
else
    echo "⚠ LM Shakespeare config not found, skipping LM benchmark"
fi

echo ""

# ============================================================
# Vision Benchmarks (CIFAR-10)
# ============================================================
echo "============================================================"
echo "  Phase 2: Vision Benchmarks"
echo "============================================================"
echo ""

# CIFAR-10 ResNet
if [ -f "configs/cifar_resnet.yaml" ]; then
    echo "Running: CIFAR-10 ResNet-18"
    timeout 7200 python sweep/run_sweep.py configs/cifar_resnet.yaml || echo "⚠ ResNet sweep completed or timed out"
    echo ""
else
    echo "⚠ CIFAR ResNet config not found"
fi

# CIFAR-10 ViT
if [ -f "configs/cifar_vit.yaml" ]; then
    echo "Running: CIFAR-10 ViT"
    timeout 7200 python sweep/run_sweep.py configs/cifar_vit.yaml || echo "⚠ ViT sweep completed or timed out"
    echo ""
else
    echo "⚠ CIFAR ViT config not found"
fi

echo ""

# ============================================================
# Results Summary
# ============================================================
echo "============================================================"
echo "  Results Summary"
echo "============================================================"
echo ""

if [ -d "results" ]; then
    RESULT_COUNT=$(find results -type f | wc -l)
    echo "Total results files: $RESULT_COUNT"
    echo ""
    echo "Sample results:"
    find results -type f -name "*.csv" -o -name "*.json" | head -20
else
    echo "No results found"
fi

echo ""
echo "============================================================"
echo "  Milo-Bench Complete!"
echo "============================================================"
echo "Finished: $(date)"
