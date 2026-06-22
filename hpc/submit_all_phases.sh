#!/bin/bash
# ============================================================
# Submit Complete Experiment Pipeline
#
# Phase 1 (PARALLEL):
#   - Baseline: 3 domains (Vision, ImageNet, NLP)
#   - Tuning: 3 domains (Vision, ImageNet, NLP)
#
# Phase 2 (After Phase 1):
#   - Tuned: 3 domains with optimal hyperparameters
#
# This enables comparison: Baseline vs Tuning vs Tuned
#
# Usage:
#   bash hpc/submit_all_phases.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Complete Experiment Pipeline: Baseline + Tuning + Tuned"
echo "============================================================"
echo ""
echo "PHASE 1: Submitting Baseline + Tuning (PARALLEL)..."
echo ""

# Phase 1: Submit all baselines in parallel
echo "  Baselines:"
VISION_BASE=$(sbatch hpc/experiments/run_vision_experiments.slurm | awk '{print $NF}')
echo "    Vision baseline        Job $VISION_BASE (~24h)"

IMAGENET_BASE=$(sbatch hpc/experiments/run_imagenet_experiments.slurm | awk '{print $NF}')
echo "    ImageNet baseline      Job $IMAGENET_BASE (~12h)"

NLP_BASE=$(sbatch hpc/experiments/run_nlp_experiments.slurm | awk '{print $NF}')
echo "    NLP baseline           Job $NLP_BASE (~22h)"

echo ""
echo "  Tuning (starts immediately, runs in parallel with baselines):"
VISION_TUNE=$(sbatch hpc/experiments/run_hyperparameter_tuning.slurm --job-name=milo_vision_tuning | awk '{print $NF}')
echo "    Vision tuning          Job $VISION_TUNE (~8-10h)"

# Note: The tuning script runs all domains. For parallel domain-specific tuning,
# you'd need separate tuning scripts. For now, one tuning job handles all.
# If you want separate tuning jobs, uncomment below and modify run_hyperparameter_tuning.slurm

# IMAGENET_TUNE=$(sbatch hpc/experiments/run_imagenet_tuning.slurm | awk '{print $NF}')
# echo "    ImageNet tuning        Job $IMAGENET_TUNE (~6-8h)"
#
# NLP_TUNE=$(sbatch hpc/experiments/run_nlp_tuning.slurm | awk '{print $NF}')
# echo "    NLP tuning             Job $NLP_TUNE (~6-8h)"

echo ""
echo "PHASE 2: Submitting Tuned Experiments (waits for Phase 1)..."
echo ""
echo "  These will run AFTER all baselines + tuning complete:"

# Phase 2: Tuned experiments run after all Phase 1 jobs complete
# They depend on tuning job completing successfully
VISION_TUNED=$(sbatch --dependency=afterok:${VISION_TUNE} \
    hpc/experiments/run_vision_experiments_tuned.slurm | awk '{print $NF}')
echo "    Vision tuned           Job $VISION_TUNED (waits for tuning)"

IMAGENET_TUNED=$(sbatch --dependency=afterok:${VISION_TUNE} \
    hpc/experiments/run_imagenet_experiments_tuned.slurm | awk '{print $NF}')
echo "    ImageNet tuned         Job $IMAGENET_TUNED (waits for tuning)"

NLP_TUNED=$(sbatch --dependency=afterok:${VISION_TUNE} \
    hpc/experiments/run_nlp_experiments_tuned.slurm | awk '{print $NF}')
echo "    NLP tuned              Job $NLP_TUNED (waits for tuning)"

echo ""
echo "============================================================"
echo "  Pipeline Submitted Successfully"
echo "============================================================"
echo ""
echo "Timeline:"
echo "  ┌─ PHASE 1 (All PARALLEL, ~24h max):"
echo "  │  ├─ Vision baseline      (24h)    │"
echo "  │  ├─ ImageNet baseline    (12h)    ├─ All start immediately"
echo "  │  ├─ NLP baseline         (22h)    │"
echo "  │  └─ Tuning               (8-10h)  │"
echo "  │"
echo "  └─ PHASE 2 (After Phase 1, ~10-15h):"
echo "     ├─ Vision tuned         (12h, depends on tuning)"
echo "     ├─ ImageNet tuned       (6h,  depends on tuning)"
echo "     └─ NLP tuned            (10h, depends on tuning)"
echo ""
echo "Total runtime: ~34-39 hours"
echo ""
echo "Comparison available:"
echo "  - Baseline:  experiments/{domain}/results_nt/"
echo "  - Tuning:    experiments/{domain}/results_tuned/ (from hyperparameter tuning)"
echo "  - Tuned:     experiments/{domain}/results_nt_tuned/ (optimal hyperparams)"
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f logs/*.out"
echo ""
echo "Job IDs:"
echo ""
echo "  Phase 1 (Parallel, ~24h max):"
echo "    Vision baseline:   $VISION_BASE (~24h)"
echo "    ImageNet baseline: $IMAGENET_BASE (~12h)"
echo "    NLP baseline:      $NLP_BASE (~22h)"
echo "    Tuning:            $VISION_TUNE (~8-10h)"
echo ""
echo "  Phase 2 (After Phase 1, ~10-15h):"
echo "    Vision tuned:      $VISION_TUNED (depends on tuning)"
echo "    ImageNet tuned:    $IMAGENET_TUNED (depends on tuning)"
echo "    NLP tuned:         $NLP_TUNED (depends on tuning)"
echo ""
echo "Comparison:"
echo "  - Baseline results:  experiments/{domain}/results_nt*/"
echo "  - Tuned results:     experiments/{domain}/results_nt*_tuned/"
echo ""
echo "Total runtime: ~34-39 hours"
echo "============================================================"
