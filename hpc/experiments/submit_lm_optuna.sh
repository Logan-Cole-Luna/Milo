#!/usr/bin/env bash
# Optuna LR search for the LM task (FineWeb-Edu), one study per optimizer.
# Short-token trials (default 60M) pick best LR; final runs use full budget.
set -euo pipefail
cd /home/logan03/Milo
OPTS=(MILO MILO_LW MILOM MION MION_ADAM ADAMW LION MUON SOAP SHAMPOO ADAM_MINI SGD)
export TRIALS="${TRIALS:-8}"
unset EXP || true
n=0
for opt in "${OPTS[@]}"; do
  export DOMAIN=lm OPT="$opt"
  sbatch --export=ALL -J "oplm_${opt,,}" hpc/experiments/run_optuna_one.slurm >/dev/null && n=$((n+1))
done
echo "Submitted $n LM Optuna studies (${TRIALS} trials each)."
