#!/usr/bin/env bash
# MION ablations: vary one hyperparameter at a time from the tuned baseline
# (ns_steps=5, rms_target=0.2, scale_factor=0.0, spectral=True), 3 seeds each,
# on MION's strongest showcases (ResNet34-CIFAR10, ViT-CIFAR10, Tiny-ImageNet).
# Each cell uses MION's tuned LR (from config). Results -> results_ablation_*/.
set -euo pipefail
cd /home/logan03/Milo
export SWEEP_RUNS=3

# tag -> MION_PARAMS_JSON override (single axis)
declare -A CELLS=(
  [ns1]='{"ns_steps":1}'   [ns3]='{"ns_steps":3}'   [ns5]='{"ns_steps":5}'   [ns7]='{"ns_steps":7}'
  [rms0p1]='{"rms_target":0.1}' [rms0p2]='{"rms_target":0.2}' [rms0p3]='{"rms_target":0.3}' [rms0p5]='{"rms_target":0.5}'
  [spec_on]='{"spectral":true}' [spec_off]='{"spectral":false}'
  [sf0p0]='{"scale_factor":0.0}' [sf0p1]='{"scale_factor":0.1}' [sf0p2]='{"scale_factor":0.2}'
)

n=0
echo "=== Vision MION ablations (ResNet34, ViT) ==="
unset LR_GRID || true
export OPT_ONLY=MION
for exp in RESNET34_CIFAR10 VIT_TINY_CIFAR10; do
  for tag in "${!CELLS[@]}"; do
    export EXP_ONLY="$exp" MION_PARAMS_JSON="${CELLS[$tag]}" \
           RESULTS_DIR_OVERRIDE="results_ablation_vision_${tag}"
    sbatch --export=ALL -J "abl_${exp,,}_${tag}" hpc/experiments/run_vision_one.slurm >/dev/null && n=$((n+1))
  done
done

echo "=== ImageNet MION ablations ==="
unset EXP_ONLY || true
export OPT_ONLY=MION
for tag in "${!CELLS[@]}"; do
  export MION_PARAMS_JSON="${CELLS[$tag]}" RESULTS_DIR_OVERRIDE="results_ablation_imagenet_${tag}"
  sbatch --export=ALL -J "abl_inet_${tag}" hpc/experiments/run_imagenet_one.slurm >/dev/null && n=$((n+1))
done

echo "Submitted $n MION ablation jobs."
