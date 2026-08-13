#!/bin/bash
# ============================================================
# Submit Milo-Bench Setup & Validation Pipeline
#
# Orchestrates:
# 1. Environment setup (venv, PyTorch, dependencies)
# 2. Original Milo.py integration
# 3. Smoke tests (LM + Vision)
# 4. Results analysis
#
# Usage:
#   bash hpc/submit_milo_bench.sh
#
# This submits jobs with dependencies so smoke tests
# automatically start after setup completes.
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Milo-Bench Pipeline Submission"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Submit setup job
# ============================================================
echo "Step 1: Submitting setup job..."
SETUP_JID=$(sbatch hpc/experiments/setup_milo_bench.slurm | awk '{print $NF}')
echo "  Setup Job ID: $SETUP_JID"
echo ""

# ============================================================
# Step 2: Submit smoke tests (depends on setup)
# ============================================================
echo "Step 2: Submitting smoke test job (waits for setup)..."
SMOKE_JID=$(sbatch --dependency=afterok:${SETUP_JID} hpc/experiments/run_milo_bench_smoke_test.slurm | awk '{print $NF}')
echo "  Smoke Test Job ID: $SMOKE_JID (depends on $SETUP_JID)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "  Milo-Bench Pipeline Submitted!"
echo "============================================================"
echo ""
echo "Timeline:"
echo "  Phase 1: Setup environment       [Job $SETUP_JID]"
echo "           └─ venv, PyTorch, deps  (~5-15 min)"
echo ""
echo "  Phase 2: Smoke tests             [Job $SMOKE_JID]"
echo "           ├─ LM (60 steps)        (waits for Phase 1)"
echo "           ├─ Vision (CIFAR-10)    (~20-30 min total)"
echo "           └─ Results validation"
echo ""
echo "Monitoring:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/setup_milo_bench_*.out"
echo "  tail -f logs/milo_bench_smoke_*.out"
echo ""
echo "Job details:"
echo "  Setup:      $SETUP_JID"
echo "  Smoke test: $SMOKE_JID (depends on setup)"
echo ""
echo "After completion:"
echo "  cd temp/"
echo "  source .venv/bin/activate"
echo "  python analysis/aggregate.py results/"
echo "  python analysis/plots.py results/ --metric best_val_loss --lower"
echo ""
echo "============================================================"
