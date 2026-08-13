#!/usr/bin/env bash
# MION-Nor (NorMuon-style row-norm, closes G1/G8 per lit_review): tune LR on LM
# (FineWeb-Edu, 60M-token trials), then run the final 124M/1.2B-token comparison
# using the tuned LR. Queued as a SLURM dependency chain (one submission).
set -euo pipefail
cd /home/logan03/Milo
export DOMAIN=lm OPT=MION_NOR TRIALS="${TRIALS:-8}"
opt_job=$(sbatch --export=ALL -J oplm_mion_nor --parsable hpc/experiments/run_optuna_one.slurm)
echo "Submitted MION_NOR LM Optuna sweep: job $opt_job"

export MODEL="${MODEL:-small}" TOKENS="${TOKENS:-1.2e9}" SEED="${SEED:-1}" OUT_DIR="results/lm"
final_job=$(sbatch --dependency=afterok:"$opt_job" --export=ALL -J lm_mion_nor_final --parsable \
    hpc/experiments/run_lm_final_from_optuna.slurm)
echo "Queued MION_NOR final LM run (depends on $opt_job): job $final_job"
