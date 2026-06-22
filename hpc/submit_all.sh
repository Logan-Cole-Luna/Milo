#!/bin/bash
# ============================================================
# Submit Baseline Experiments (No Tuning)
#
# Submits all 3 baseline experiment phases in parallel:
# - Vision (6 models × 11 optimizers × 5 runs)
# - ImageNet (1 model × 11 optimizers × 3 runs)
# - NLP (BERT × 11 optimizers × 5 runs)
#
# Usage:
#   bash hpc/submit_all.sh              # Baseline only
#   bash hpc/submit_baseline_then_tuning.sh  # Baseline → Tuning (chained)
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Submitting Comprehensive Experiment Suite"
echo "============================================================"
echo ""

# Submit all experiments
echo "1. Vision Experiments (6 models × 12 optimizers × 5 runs)..."
VISION_JID=$(sbatch hpc/experiments/run_vision_experiments.slurm | awk '{print $NF}')
echo "   Job ID: $VISION_JID"

echo "2. ImageNet Experiments (1 model × 12 optimizers × 3 runs)..."
IMAGENET_JID=$(sbatch hpc/experiments/run_imagenet_experiments.slurm | awk '{print $NF}')
echo "   Job ID: $IMAGENET_JID"

echo "3. NLP Experiments (BERT × 12 optimizers × 5 runs)..."
NLP_JID=$(sbatch hpc/experiments/run_nlp_experiments.slurm | awk '{print $NF}')
echo "   Job ID: $NLP_JID"

echo ""
echo "============================================================"
echo "  Baseline Jobs Submitted!"
echo "============================================================"
echo ""
echo "Monitor with:"
echo "  squeue -u $(whoami)"
echo "  tail -f logs/*.out"
echo ""
echo "Job Summary (running in PARALLEL):"
echo "  Vision:   $VISION_JID   (~24 hours)"
echo "  ImageNet: $IMAGENET_JID   (~12 hours)"
echo "  NLP:      $NLP_JID   (~22 hours)"
echo ""
echo "To chain with tuning after baselines complete:"
echo "  bash hpc/submit_baseline_then_tuning.sh"
echo "============================================================"
