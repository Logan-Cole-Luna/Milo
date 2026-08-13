#!/usr/bin/env bash
# ============================================================
# Run Milo-Bench Vision Benchmarks Only
#
# Simpler, faster validation:
# - CIFAR-10 ResNet-18 (15-30 min)
# - CIFAR-10 ViT-Small (30-60 min)
# - Tests MiloM, Mion, and baselines
#
# Submit: sbatch hpc/experiments/run_milo_bench_vision_only.slurm
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=milo_bench_vision
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/home/logan03/Milo/logs/milo_bench_vision_%A.out
#SBATCH --error=/home/logan03/Milo/logs/milo_bench_vision_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="/home/logan03/Milo"
TEMP_DIR="${PROJECT_ROOT}/temp"

echo "============================================================"
echo "  Milo-Bench Vision Benchmarks"
echo "============================================================"
echo ""
echo "Testing MiloM, Mion, and baselines on CIFAR-10"
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

cd "$TEMP_DIR"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo ""

# ============================================================
# CIFAR-10 ResNet-18
# ============================================================
if [ -f "configs/cifar_resnet.yaml" ]; then
    echo "============================================================"
    echo "  CIFAR-10 ResNet-18"
    echo "============================================================"
    echo ""
    timeout 3600 python sweep/run_sweep.py configs/cifar_resnet.yaml || echo "⚠ ResNet sweep completed or timed out"
    echo ""
else
    echo "⚠ CIFAR ResNet config not found"
fi

# ============================================================
# CIFAR-10 ViT
# ============================================================
if [ -f "configs/cifar_vit.yaml" ]; then
    echo "============================================================"
    echo "  CIFAR-10 ViT"
    echo "============================================================"
    echo ""
    timeout 3600 python sweep/run_sweep.py configs/cifar_vit.yaml || echo "⚠ ViT sweep completed or timed out"
    echo ""
else
    echo "⚠ CIFAR ViT config not found"
fi

# ============================================================
# Results Summary
# ============================================================
echo "============================================================"
echo "  Results"
echo "============================================================"
echo ""

if [ -d "results" ]; then
    echo "Results saved to: results/"
    find results -name "*.csv" -o -name "*.json" | head -20
else
    echo "No results found"
fi

echo ""
echo "Finished: $(date)"
