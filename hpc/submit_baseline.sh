#!/bin/bash
# ============================================================
# Submit Baseline Experiments Only (no tuning)
#
# Submits all 3 baseline experiments in parallel:
# - Vision (6 models × 11 optimizers × 5 runs)
# - ImageNet (1 model × 11 optimizers × 3 runs)
# - NLP (BERT × 11 optimizers × 5 runs)
#
# Usage:
#   bash hpc/submit_baseline.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Submitting Baseline Experiments (No Tuning)"
echo "============================================================"
echo ""

# Submit all baseline experiments in parallel
echo "1. Vision Baseline (6 models × 11 optimizers × 5 runs)..."
VISION_JID=$(sbatch hpc/experiments/run_vision_experiments.slurm | awk '{print $NF}')
echo "   Job ID: $VISION_JID"

echo "2. ImageNet Baseline (1 model × 11 optimizers × 3 runs)..."
IMAGENET_JID=$(sbatch hpc/experiments/run_imagenet_experiments.slurm | awk '{print $NF}')
echo "   Job ID: $IMAGENET_JID"

echo "3. NLP Baseline (BERT × 11 optimizers × 5 runs)..."
NLP_JID=$(sbatch hpc/experiments/run_nlp_experiments.slurm | awk '{print $NF}')
echo "   Job ID: $NLP_JID"

echo ""
echo "============================================================"
echo "  Baseline Jobs Submitted (Running in Parallel)"
echo "============================================================"
echo ""
echo "Job IDs for reference:"
echo "  Vision:   $VISION_JID"
echo "  ImageNet: $IMAGENET_JID"
echo "  NLP:      $NLP_JID"
echo ""
echo "Monitor with:"
echo "  squeue -u $(whoami)"
echo "  tail -f logs/*.out"
echo ""
echo "Next: When baselines complete, you can run tuning:"
echo "  sbatch hpc/experiments/run_hyperparameter_tuning.slurm"
echo "============================================================"
