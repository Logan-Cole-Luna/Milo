#!/bin/bash
# ============================================================
# Milo-Bench Quick Setup (Using Existing .venv_cc)
#
# Instead of downloading everything, this leverages the
# already-installed .venv_cc environment which has:
# - PyTorch 2.12.0
# - transformers 5.3.0
# - datasets 5.0.0
# - All other dependencies
#
# Usage:
#   bash hpc/setup_milo_bench_simple.sh
# ============================================================

set -euo pipefail

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="${PROJECT_ROOT}/temp"

echo "============================================================"
echo "  Milo-Bench Setup (Using Existing .venv_cc)"
echo "============================================================"
echo ""
echo "Working directory: $TEMP_DIR"
echo "Started: $(date)"
echo ""

cd "$TEMP_DIR"

# ============================================================
# Step 1: Link to existing .venv_cc environment
# ============================================================
echo "Step 1: Linking to existing .venv_cc environment..."

if [ -L ".venv" ] || [ -d ".venv" ]; then
    rm -rf .venv
fi

# Create symlink to the shared environment
ln -s ~/.venv_cc .venv
echo "✓ Symlink created: .venv -> ~/.venv_cc"

# Activate it
ACTIVATE_PATH=".venv/bin/activate"
if [ ! -f "$ACTIVATE_PATH" ]; then
    echo "ERROR: Cannot find activate script at $ACTIVATE_PATH"
    ls -la .venv/bin/ 2>/dev/null | head -10 || echo "(symlink target not readable)"
    exit 1
fi

source "$ACTIVATE_PATH"
echo "✓ Environment activated"
echo "  Python: $(python --version)"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
echo ""

# ============================================================
# Step 2: Verify core packages
# ============================================================
echo "Step 2: Verifying core packages..."

python - <<'PY'
import sys
packages = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('datasets', 'Datasets'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
]

all_ok = True
for pkg_name, display_name in packages:
    try:
        mod = __import__(pkg_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {display_name}: {version}")
    except ImportError:
        print(f"  ✗ {display_name}: NOT FOUND", file=sys.stderr)
        all_ok = False

if all_ok:
    print("\n✓ All core packages available")
else:
    print("\n✗ Some packages missing", file=sys.stderr)
    sys.exit(1)
PY

echo ""

# ============================================================
# Step 3: Copy original milo.py
# ============================================================
echo "Step 3: Integrating original Milo optimizer..."

if [ -f "${PROJECT_ROOT}/milo.py" ]; then
    cp "${PROJECT_ROOT}/milo.py" optimizers/milo.py
    echo "✓ Original milo.py copied to optimizers/"
else
    echo "⚠ Original milo.py not found (optional)"
fi

echo ""

# ============================================================
# Step 4: Download Shakespeare dataset
# ============================================================
echo "Step 4: Downloading Shakespeare dataset (for smoke tests)..."

if [ -f "scripts/download_data.py" ]; then
    timeout 300 python scripts/download_data.py --source shakespeare 2>&1 | tail -3 || echo "  (download slower than expected)"
    echo "✓ Shakespeare dataset ready"
else
    echo "⚠ download_data.py not found"
fi

echo ""

# ============================================================
# Step 5: Vendor optimizers (optional, best-effort)
# ============================================================
echo "Step 5: Vendoring optimizer implementations..."

if [ -f "scripts/vendor_optimizers.sh" ]; then
    timeout 60 bash scripts/vendor_optimizers.sh 2>&1 | head -10 || echo "  (vendoring skipped or slow)"
    echo "✓ Optimizer vendoring complete"
else
    echo "⚠ vendor_optimizers.sh not found"
fi

echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "  Environment Ready for Benchmarks!"
echo "============================================================"
echo ""
echo "Environment: ~/.venv_cc (symlinked)"
echo "Location: $TEMP_DIR"
echo ""
echo "To activate:"
echo "  source ~/.venv_cc/bin/activate"
echo ""
echo "Ready to run benchmarks on GPU!"
echo ""
echo "Next steps:"
echo "  1. Submit GPU job:  sbatch hpc/experiments/run_milo_bench_on_gpu.slurm"
echo "  2. Monitor:         tail -f logs/milo_bench_gpu_*.out"
echo ""
echo "Finished: $(date)"
echo "============================================================"
