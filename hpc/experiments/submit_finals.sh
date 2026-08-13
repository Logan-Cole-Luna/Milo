#!/usr/bin/env bash
# Final reporting runs at Optuna-tuned LRs (from config LEARNING_RATES).
# Full seeds/epochs per config. NLP/ImageNet: one job per optimizer (parallel,
# isolates slow optimizers). Vision: one job per experiment (all 13 opts at their
# tuned per-family LRs). Results in results_final_*.
set -euo pipefail
cd /home/logan03/Milo

OPTS=(MILO MILO_LW MILOM MION SGD ADAMW ADAGRAD LION ADAM_MINI RMSPROP_MOMENTUM SHAMPOO SOAP MUON)
n=0

echo "=== NLP finals (per optimizer) ==="
unset EXP_ONLY LR_GRID || true
for opt in "${OPTS[@]}"; do
  export OPT_ONLY="$opt" RESULTS_DIR_OVERRIDE="results_final_nlp"
  sbatch --export=ALL -J "fin_nlp_${opt,,}" hpc/experiments/run_nlp_one.slurm >/dev/null && n=$((n+1))
done

echo "=== ImageNet finals (per optimizer) ==="
for opt in "${OPTS[@]}"; do
  export OPT_ONLY="$opt" RESULTS_DIR_OVERRIDE="results_final_imagenet"
  sbatch --export=ALL -J "fin_inet_${opt,,}" hpc/experiments/run_imagenet_one.slurm >/dev/null && n=$((n+1))
done

echo "=== Vision finals (per experiment, all optimizers at tuned LRs) ==="
unset OPT_ONLY || true
for exp in LOGISTIC MULTILAYER RESNET34_CIFAR10 RESNET34_CIFAR100 VGG11_CIFAR10 VGG11_CIFAR100 VIT_TINY_CIFAR10 VIT_TINY_CIFAR100; do
  export EXP_ONLY="$exp" RESULTS_DIR_OVERRIDE="results_final_vision"
  sbatch --export=ALL -J "fin_vis_${exp,,}" hpc/experiments/run_vision_one.slurm >/dev/null && n=$((n+1))
done

echo "Submitted $n final jobs."
