#!/usr/bin/env bash
# MION-Gated / "MION-FT" (tunable orthogonalization-strength gate, per lit_review
# G2/memo-2.3): jointly tune (lr, beta=ortho_strength) on the Qwen2.5-1.5B/Alpaca
# FT benchmark, then run the final 600-step comparison at the tuned point.
# Queued as a SLURM dependency chain (one submission).
set -euo pipefail
cd /home/logan03/Milo
export DOMAIN=ft OPT=MION_GATED TRIALS="${TRIALS:-15}"
opt_job=$(sbatch --export=ALL -J opft_mion_gated --parsable hpc/experiments/run_optuna_one.slurm)
echo "Submitted MION_GATED FT (lr, beta) Optuna sweep: job $opt_job"

export STEPS="${STEPS:-600}"
final_job=$(sbatch --dependency=afterok:"$opt_job" --export=ALL -J ft_mion_gated_final --parsable \
    hpc/experiments/run_ft_final_from_optuna.slurm)
echo "Queued MION_GATED final FT run (depends on $opt_job): job $final_job"
