#!/usr/bin/env bash
# Converged vision: showcase tasks trained long (tuned LRs), per-(exp,opt) parallel.
set -euo pipefail
cd /home/logan03/Milo
OPTS=(MILO MILO_LW MILOM MION SGD ADAMW ADAGRAD LION ADAM_MINI RMSPROP_MOMENTUM SHAMPOO SOAP MUON)
export EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-60}" SWEEP_RUNS="${SWEEP_RUNS:-3}"
unset LR_GRID MION_PARAMS_JSON || true
n=0
for exp in RESNET34_CIFAR10 VGG11_CIFAR10 VIT_TINY_CIFAR10; do
  for opt in "${OPTS[@]}"; do
    export EXP_ONLY="$exp" OPT_ONLY="$opt" RESULTS_DIR_OVERRIDE="results_converged_${opt,,}"
    sbatch --export=ALL -J "cv_${exp,,}_${opt,,}" hpc/experiments/run_vision_one.slurm >/dev/null && n=$((n+1))
  done
done
echo "Submitted $n converged-vision jobs (${EPOCHS_OVERRIDE} epochs, ${SWEEP_RUNS} seeds)."
