#!/usr/bin/env bash
# ============================================================
# Smoke Test: Milo-Bench (100 LM steps + 1 vision epoch, all optimizers)
#
# Validates that all optimizers run in the milo-bench framework
# before the full sweep job completes.
#
# Submit: sbatch hpc/experiments/smoke_test_milo_bench.slurm
# Monitor: tail -f logs/smoke_milo_bench_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=smoke_milo_bench
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/logan03/Milo/logs/smoke_milo_bench_%A.out
#SBATCH --error=/home/logan03/Milo/logs/smoke_milo_bench_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
TEMP_DIR="${PROJECT_ROOT}/temp"
mkdir -p "${PROJECT_ROOT}/logs/"

echo "============================================================"
echo "  SMOKE TEST: Milo-Bench"
echo "  100 LM steps (Shakespeare) + 1 vision epoch per optimizer"
echo "============================================================"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

cd "$TEMP_DIR"

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
echo ""

# ── LM smoke: 100 steps on Shakespeare, one optimizer at a time ──
echo "============================================================"
echo "  LM Smoke: 100 steps on Shakespeare (all optimizers)"
echo "============================================================"

# shampoo excluded: distributed_shampoo requires Python>=3.12 (cluster runs 3.11.5)
OPTIMIZERS="adamw nadamw sgdm adafactor lion muon soap kron milo milo_m mion"
LM_FAILED=""

for opt in $OPTIMIZERS; do
    echo -n "  Testing LM optimizer: $opt ... "
    if timeout 120 python tasks/lm_pretrain.py \
        --data shakespeare --model small \
        --optimizer "$opt" --lr 0.001 \
        --steps 100 --batch-size 4 --accum 1 --ctx 128 \
        --out-dir results/smoke_lm \
        --seed 1 > /tmp/lm_smoke_${opt}.log 2>&1; then
        echo "✓"
    else
        echo "✗ FAILED"
        LM_FAILED="${LM_FAILED} ${opt}"
        tail -5 /tmp/lm_smoke_${opt}.log
    fi
done

echo ""

# ── Vision smoke: 1 epoch on CIFAR-10 ResNet-18, one optimizer at a time ──
echo "============================================================"
echo "  Vision Smoke: 1 epoch CIFAR-10 ResNet-18 (all optimizers)"
echo "============================================================"

VIS_FAILED=""

for opt in $OPTIMIZERS; do
    echo -n "  Testing Vision optimizer: $opt ... "
    if timeout 180 python tasks/vision.py \
        --model resnet18 --optimizer "$opt" --lr 0.001 \
        --epochs 1 --batch 128 \
        --out-dir results/smoke_vision \
        --seed 1 > /tmp/vis_smoke_${opt}.log 2>&1; then
        echo "✓"
    else
        echo "✗ FAILED"
        VIS_FAILED="${VIS_FAILED} ${opt}"
        tail -5 /tmp/vis_smoke_${opt}.log
    fi
done

echo ""
echo "============================================================"
echo "  Smoke Test Summary"
echo "============================================================"

EXIT_CODE=0
if [ -n "${LM_FAILED}" ]; then
    echo "✗ LM failures: ${LM_FAILED}"
    EXIT_CODE=1
else
    echo "✓ All LM optimizers passed"
fi

if [ -n "${VIS_FAILED}" ]; then
    echo "✗ Vision failures: ${VIS_FAILED}"
    EXIT_CODE=1
else
    echo "✓ All vision optimizers passed"
fi

echo ""
echo "Finished: $(date)"
exit $EXIT_CODE
