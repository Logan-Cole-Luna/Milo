#!/usr/bin/env bash
# Launch Optuna LR studies: one per (domain, optimizer[, experiment]).
# Continuous log-uniform LR search (ranges defined in optuna_sweep.py).
set -euo pipefail
cd /home/logan03/Milo

OPTS=(MILO MILO_LW MILOM MION SGD ADAMW ADAGRAD LION ADAM_MINI RMSPROP_MOMENTUM SHAMPOO SOAP MUON)
export TRIALS="${TRIALS:-15}"
n=0

echo "=== NLP Optuna studies ==="
unset EXP || true
for opt in "${OPTS[@]}"; do
  export DOMAIN=nlp OPT="$opt"
  sbatch --export=ALL -J "op_nlp_${opt,,}" hpc/experiments/run_optuna_one.slurm >/dev/null && n=$((n+1))
done

echo "=== ImageNet Optuna studies ==="
for opt in "${OPTS[@]}"; do
  export DOMAIN=imagenet OPT="$opt"
  sbatch --export=ALL -J "op_inet_${opt,,}" hpc/experiments/run_optuna_one.slurm >/dev/null && n=$((n+1))
done

echo "=== Vision Optuna studies (3 representative experiments) ==="
for exp in RESNET34_CIFAR10 VGG11_CIFAR10 VIT_TINY_CIFAR10; do
  for opt in "${OPTS[@]}"; do
    export DOMAIN=vision OPT="$opt" EXP="$exp"
    sbatch --export=ALL -J "op_${exp,,}_${opt,,}" hpc/experiments/run_optuna_one.slurm >/dev/null && n=$((n+1))
  done
done

echo "Submitted $n Optuna studies."
