#!/bin/bash
# ============================================================
# Submit Baseline Experiments, Then Tuning (Chained)
#
# Phase 1: All baselines run in parallel
# Phase 2: Tuning runs after ALL baselines complete
#
# Usage:
#   bash hpc/submit_baseline_then_tuning.sh
#
# This ensures tuning doesn't start until all baseline results exist.
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Submitting Baseline → Tuning Pipeline"
echo "============================================================"
echo ""
echo "PHASE 1: Submitting all baseline experiments (parallel)..."
echo ""

# Submit all baseline experiments in parallel
VISION_JID=$(sbatch hpc/experiments/run_vision_experiments.slurm | awk '{print $NF}')
echo "  Vision:   Job $VISION_JID (~24 hours)"

IMAGENET_JID=$(sbatch hpc/experiments/run_imagenet_experiments.slurm | awk '{print $NF}')
echo "  ImageNet: Job $IMAGENET_JID (~12 hours)"

NLP_JID=$(sbatch hpc/experiments/run_nlp_experiments.slurm | awk '{print $NF}')
echo "  NLP:      Job $NLP_JID (~22 hours)"

echo ""
echo "PHASE 2: Submitting tuning (waits for all baselines)..."
# Tuning depends on ALL baselines completing
TUNING_JID=$(sbatch \
    --dependency=afterok:${VISION_JID}:${IMAGENET_JID}:${NLP_JID} \
    hpc/experiments/run_hyperparameter_tuning.slurm | awk '{print $NF}')
echo "  Tuning:   Job $TUNING_JID (runs after baselines complete)"

echo ""
echo "============================================================"
echo "  Pipeline Submitted Successfully"
echo "============================================================"
echo ""
echo "Timeline:"
echo "  PHASE 1 (Baselines):"
echo "    Vision   (~0-24h)  │"
echo "    ImageNet (~0-12h)  ├─→ All must complete before Phase 2"
echo "    NLP      (~0-22h)  │"
echo "           │"
echo "           └→ PHASE 2 (Tuning):"
echo "              Hyperparameter search (~8-12h after Phase 1)"
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f logs/*.out"
echo ""
echo "Job IDs:"
echo "  Baseline Vision:   $VISION_JID"
echo "  Baseline ImageNet: $IMAGENET_JID"
echo "  Baseline NLP:      $NLP_JID"
echo "  Tuning:            $TUNING_JID (depends on all above)"
echo "============================================================"
