#!/usr/bin/env bash
# Launch a per-optimizer LR sweep across all three domains.
# 1 seed per (optimizer, lr) to pick the best LR cheaply; final full runs follow.
#
# NOTE: grids contain commas, and `sbatch --export=K=V,...` treats commas as
# variable separators. So we EXPORT the per-job vars into the environment and
# submit with `--export=ALL`, which preserves commas inside values.
set -euo pipefail
cd /home/logan03/Milo

OPTS=(MILO MILO_LW MILOM MION SGD ADAMW ADAGRAD LION ADAM_MINI RMSPROP_MOMENTUM SHAMPOO SOAP MUON)

declare -A NLP=(
  [SGD]="1e-2,3e-3,1e-3" [ADAMW]="5e-5,2e-5,1e-5" [ADAGRAD]="1e-3,3e-4,1e-4"
  [LION]="3e-5,1e-5,3e-6" [ADAM_MINI]="5e-5,2e-5,1e-5" [RMSPROP_MOMENTUM]="3e-4,1e-4,3e-5"
  [SHAMPOO]="1e-3,3e-4,1e-4" [SOAP]="3e-4,1e-4,3e-5" [MUON]="2e-2,1e-2,3e-3"
  [MILO]="3e-4,1e-4,3e-5" [MILO_LW]="3e-4,1e-4,3e-5" [MILOM]="1e-4,3e-5,1e-5" [MION]="3e-4,1e-4,3e-5"
)
declare -A CNN=(
  [SGD]="0.2,0.1,0.03" [ADAMW]="3e-3,1e-3,3e-4" [ADAGRAD]="3e-2,1e-2,3e-3"
  [LION]="1e-3,3e-4,1e-4" [ADAM_MINI]="3e-3,1e-3,3e-4" [RMSPROP_MOMENTUM]="1e-3,3e-4,1e-4"
  [SHAMPOO]="1e-2,1e-3,1e-4" [SOAP]="1e-2,3e-3,1e-3" [MUON]="3e-2,1e-2,3e-3"
  [MILO]="0.1,0.03,0.01" [MILO_LW]="0.1,0.03,0.01" [MILOM]="0.05,0.02,0.01" [MION]="0.05,0.02,0.01"
)
declare -A VIT=(
  [SGD]="0.05,0.02,0.01" [ADAMW]="1e-3,3e-4,1e-4" [ADAGRAD]="1e-2,3e-3,1e-3"
  [LION]="3e-4,1e-4,3e-5" [ADAM_MINI]="1e-3,3e-4,1e-4" [RMSPROP_MOMENTUM]="3e-4,1e-4,3e-5"
  [SHAMPOO]="3e-3,1e-3,3e-4" [SOAP]="3e-3,1e-3,3e-4" [MUON]="2e-2,1e-2,3e-3"
  [MILO]="0.03,0.01,0.005" [MILO_LW]="0.03,0.01,0.005" [MILOM]="0.02,0.01,0.005" [MION]="0.02,0.01,0.005"
)

export SWEEP_RUNS=1
n=0

echo "=== NLP sweep ==="
unset EXP_ONLY || true
for opt in "${OPTS[@]}"; do
  export OPT_ONLY="$opt" LR_GRID="${NLP[$opt]}"
  sbatch --export=ALL -J "sw_nlp_${opt,,}" hpc/experiments/run_nlp_one.slurm >/dev/null && n=$((n+1))
done

echo "=== ImageNet sweep ==="
for opt in "${OPTS[@]}"; do
  export OPT_ONLY="$opt" LR_GRID="${CNN[$opt]}"
  sbatch --export=ALL -J "sw_inet_${opt,,}" hpc/experiments/run_imagenet_one.slurm >/dev/null && n=$((n+1))
done

echo "=== Vision sweep (3 representative experiments) ==="
for exp in RESNET34_CIFAR10 VGG11_CIFAR10 VIT_TINY_CIFAR10; do
  if [ "$exp" = "VIT_TINY_CIFAR10" ]; then declare -n G=VIT; else declare -n G=CNN; fi
  for opt in "${OPTS[@]}"; do
    export EXP_ONLY="$exp" OPT_ONLY="$opt" LR_GRID="${G[$opt]}"
    sbatch --export=ALL -J "sw_${exp,,}_${opt,,}" hpc/experiments/run_vision_one.slurm >/dev/null && n=$((n+1))
  done
  unset -n G
done

echo "Submitted $n sweep jobs."
