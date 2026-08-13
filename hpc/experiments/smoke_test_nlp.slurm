#!/usr/bin/env bash
# ============================================================
# Smoke Test: NLP Experiments (1 epoch, 1 run, all 13 optimizers)
#
# Validates that all optimizers instantiate and fine-tune BERT
# correctly before the full 3-epoch, 5-run job completes.
#
# Submit: sbatch hpc/experiments/smoke_test_nlp.slurm
# Monitor: tail -f logs/smoke_nlp_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=smoke_nlp
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/home/logan03/Milo/logs/smoke_nlp_%A.out
#SBATCH --error=/home/logan03/Milo/logs/smoke_nlp_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs/"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "============================================================"
echo "  SMOKE TEST: NLP Experiments"
echo "  1 epoch, 1 run, all 13 optimizers, BERT on SST-2"
echo "============================================================"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments', 'nlp'))

# Override config before importing experiment runner
import experiments.nlp.config as cfg
cfg.EPOCHS = 1
cfg.RUNS_PER_OPTIMIZER = 1

from experiments.nlp.nlp_experiment import run_bert_experiment

failed = []
for optimizer_name in cfg.OPTIMIZERS:
    print(f"\n{'='*60}")
    print(f"  Smoke testing: {optimizer_name}")
    print(f"{'='*60}")
    try:
        run_bert_experiment(
            optimizer_name=optimizer_name,
            optimizer_params=cfg.OPTIMIZER_PARAMS[optimizer_name],
            batch_size=cfg.BATCH_SIZE,
            epochs=1,
            learning_rate=cfg.LEARNING_RATES[optimizer_name],
            runs=1,
            results_dir="results_smoke_nlp",
        )
        print(f"✓ {optimizer_name} passed")
    except Exception as e:
        print(f"✗ {optimizer_name} FAILED: {e}")
        import traceback; traceback.print_exc()
        failed.append(optimizer_name)

print("\n" + "="*60)
if failed:
    print(f"SMOKE TEST FAILED — {len(failed)} optimizers failed: {failed}")
    sys.exit(1)
else:
    print(f"✓ All {len(cfg.OPTIMIZERS)} optimizers passed NLP smoke test")
PYEOF

echo ""
echo "============================================================"
echo "  Smoke Test Complete: $(date)"
echo "============================================================"
