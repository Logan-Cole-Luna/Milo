#!/usr/bin/env bash
# ============================================================
# Run Full Milo-Bench Training Suite
#
# Comprehensive benchmarks for MiloM and Mion:
# - LM pretraining sweep
# - Vision benchmarks (ResNet + ViT)
# - Fine-tuning evaluation
#
# Submit: sbatch hpc/experiments/run_milo_bench_full.slurm
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

export PROJECT_ROOT="/home/logan03/Milo"
TEMP_DIR="${PROJECT_ROOT}/temp"

echo "============================================================"
echo "  Milo-Bench Full Training Suite"
echo "============================================================"
echo ""
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

cd "$TEMP_DIR"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
echo ""

# ============================================================
# LM Pretraining (FineWeb-Edu)
# ============================================================
echo "============================================================"
echo "  Phase 1: Language Model Pretraining"
echo "============================================================"
echo ""

if [ -f "sweep/run_sweep.py" ] && [ -f "configs/lm_sweep_small.yaml" ]; then
    echo "Running: python sweep/run_sweep.py configs/lm_sweep_small.yaml"
    echo "This trains MiloM, Mion, and baselines on language modeling..."
    echo ""
    timeout 18000 python sweep/run_sweep.py configs/lm_sweep_small.yaml || echo "⚠ LM sweep completed or timed out"
    echo ""
    echo "LM Results: $(find results -name '*.csv' | wc -l) files"
else
    echo "⚠ LM sweep config not found"
fi

echo ""

# ============================================================
# Vision Benchmarks (CIFAR-10 ResNet + ViT)
# ============================================================
echo "============================================================"
echo "  Phase 2: Vision Benchmarks"
echo "============================================================"
echo ""

# CIFAR-10 ResNet
if [ -f "configs/cifar_resnet.yaml" ]; then
    echo "Running: CIFAR-10 ResNet-18"
    timeout 7200 python sweep/run_sweep.py configs/cifar_resnet.yaml || echo "⚠ CIFAR ResNet sweep completed or timed out"
    echo ""
else
    echo "⚠ CIFAR ResNet config not found"
fi

# CIFAR-10 ViT
if [ -f "configs/cifar_vit.yaml" ]; then
    echo "Running: CIFAR-10 ViT"
    timeout 7200 python sweep/run_sweep.py configs/cifar_vit.yaml || echo "⚠ CIFAR ViT sweep completed or timed out"
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
    RESULT_SIZE=$(du -sh results 2>/dev/null | cut -f1 || echo "N/A")
    echo "Total results files: $RESULT_COUNT"
    echo "Results directory size: $RESULT_SIZE"
    echo ""
    echo "Result locations:"
    find results -type d | head -10
else
    echo "No results directory yet"
fi

echo ""
echo "============================================================"
echo "  Benchmark Complete!"
echo "============================================================"
echo "Finished: $(date)"
