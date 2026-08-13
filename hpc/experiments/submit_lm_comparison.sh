#!/usr/bin/env bash
# LM pretraining comparison: all optimizers on FineWeb-Edu, one job each (parallel).
# Requires tokenized shards (tokenize_fineweb.slurm) to exist first.
#   bash hpc/experiments/submit_lm_comparison.sh            # small, 1.2B tokens, seed 1
#   MODEL=small TOKENS=1.2e9 SEEDS="1 2" bash ...           # multi-seed
set -euo pipefail
cd /home/logan03/Milo

OPTS=(MILO MILO_LW MILOM MION MION_ADAM ADAMW LION MUON SOAP SHAMPOO ADAM_MINI SGD)
MODEL="${MODEL:-small}"; TOKENS="${TOKENS:-1.2e9}"; SEEDS="${SEEDS:-1}"
n=0
for seed in $SEEDS; do
  for opt in "${OPTS[@]}"; do
    export OPT="$opt" MODEL="$MODEL" TOKENS="$TOKENS" SEED="$seed" OUT_DIR="results/lm"
    sbatch ${DEP:+--dependency=afterok:$DEP} --export=ALL -J "lm_${opt,,}_s${seed}" \
        hpc/experiments/run_lm_one.slurm >/dev/null && n=$((n+1))
  done
done
echo "Submitted $n LM jobs (model=$MODEL tokens=$TOKENS seeds='$SEEDS')."
