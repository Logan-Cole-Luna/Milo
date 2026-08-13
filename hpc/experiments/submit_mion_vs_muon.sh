#!/usr/bin/env bash
# Controlled MION vs Muon-style aux study.
# Both use MION's Newton-Schulz matrix path (identical). Only the NON-matrix path
# differs:
#   - MION (baseline): group-std aux, SINGLE LR  -> already have it as the
#     ablation "spec_on" cell (results_ablation_*_spec_on).
#   - MION-adam: AdamW aux with its OWN tuned aux_lr (give Muon's design its best
#     shot). Sweep aux_lr; matrix path uses MION's tuned LR from config.
# 3 seeds, MION's strongest showcases. Results -> results_mvm_*/.
set -euo pipefail
cd /home/logan03/Milo
export SWEEP_RUNS=3 OPT_ONLY=MION
unset LR_GRID || true

AUX_LRS=(1e-4 3e-4 1e-3 3e-3)   # AdamW-aux LR grid (canonical Muon aux ~3e-4)

n=0
echo "=== Vision: MION-adam-aux sweep (ResNet34, ViT) ==="
for exp in RESNET34_CIFAR10 VIT_TINY_CIFAR10; do
  for alr in "${AUX_LRS[@]}"; do
    export EXP_ONLY="$exp" \
           MION_PARAMS_JSON="{\"aux_mode\":\"adam\",\"aux_lr\":${alr}}" \
           RESULTS_DIR_OVERRIDE="results_mvm_vision_auxlr${alr}"
    sbatch --export=ALL -J "mvm_${exp,,}_aux${alr}" hpc/experiments/run_vision_one.slurm >/dev/null && n=$((n+1))
  done
done

echo "=== ImageNet: MION-adam-aux sweep ==="
unset EXP_ONLY || true
for alr in "${AUX_LRS[@]}"; do
  export MION_PARAMS_JSON="{\"aux_mode\":\"adam\",\"aux_lr\":${alr}}" \
         RESULTS_DIR_OVERRIDE="results_mvm_imagenet_auxlr${alr}"
  sbatch --export=ALL -J "mvm_inet_aux${alr}" hpc/experiments/run_imagenet_one.slurm >/dev/null && n=$((n+1))
done

echo "Submitted $n MION-vs-Muon jobs (baseline = ablation spec_on cells)."
