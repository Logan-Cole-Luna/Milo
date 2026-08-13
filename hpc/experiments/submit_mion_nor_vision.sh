#!/usr/bin/env bash
# MION-Nor converged-vision check: rerun the 60-epoch showcase suite for
# MION_NOR only (other optimizers' numbers are already final), reusing MION's
# tuned LR as a starting point (per lit_review plan; row-norm preserves the
# pre-normalization global RMS, so the optimal LR shouldn't shift much).
set -euo pipefail
cd /home/logan03/Milo
export EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-60}" SWEEP_RUNS="${SWEEP_RUNS:-3}"
unset LR_GRID MION_PARAMS_JSON || true
n=0
for exp in RESNET34_CIFAR10 VGG11_CIFAR10 VIT_TINY_CIFAR10; do
  export EXP_ONLY="$exp" OPT_ONLY="MION_NOR" RESULTS_DIR_OVERRIDE="results_converged_mion_nor"
  sbatch --export=ALL -J "cv_${exp,,}_mion_nor" hpc/experiments/run_vision_one.slurm >/dev/null && n=$((n+1))
done
echo "Submitted $n MION_NOR converged-vision jobs (${EPOCHS_OVERRIDE} epochs, ${SWEEP_RUNS} seeds)."
