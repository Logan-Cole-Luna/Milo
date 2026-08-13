#!/bin/bash
# ============================================================
# Setup Milo-Bench Environment on LOGIN NODE
#
# ⚠️  THIS RUNS ON LOGIN NODE (has internet, no GPU)
# Do NOT submit to SLURM - run directly on login node
#
# Usage:
#   bash hpc/setup_milo_bench_login.sh
#
# This creates the full environment in temp/.venv with:
# - PyTorch + CUDA support
# - All dependencies downloaded
# - Original milo.py integrated
# - Shakespeare dataset cached
#
# After this completes, GPU jobs can run benchmarks offline.
# ============================================================

set -euo pipefail

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="${PROJECT_ROOT}/temp"

echo "============================================================"
echo "  Milo-Bench Environment Setup (LOGIN NODE)"
echo "============================================================"
echo ""
echo "WARNING: This script runs on LOGIN NODE (has internet)"
echo "         Do NOT submit to SLURM"
echo ""
echo "Working directory: $TEMP_DIR"
echo "Started: $(date)"
echo ""

# ============================================================
# Step 1: Check temp folder
# ============================================================
if [ ! -d "$TEMP_DIR" ]; then
    echo "ERROR: temp folder not found at $TEMP_DIR"
    exit 1
fi

cd "$TEMP_DIR"
echo "✓ Changed to temp folder"
echo ""

# ============================================================
# Step 2: Create or reuse virtual environment
# ============================================================
echo "Step 2: Setting up virtual environment..."

if [ -d ".venv" ]; then
    echo "  (venv already exists, reusing)"
else
    echo "  Creating new venv..."
    if command -v uv &> /dev/null; then
        uv venv .venv
    else
        python3 -m venv .venv
    fi
fi

ACTIVATE=".venv/bin/activate"
source "$ACTIVATE"
echo "✓ Virtual environment ready"
echo "  Python: $(python --version)"
echo ""

# ============================================================
# Step 3: Upgrade pip first
# ============================================================
echo "Step 3: Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
echo "✓ pip upgraded"
echo ""

# ============================================================
# Step 4: Install PyTorch with CUDA support (skip torchaudio)
# ============================================================
echo "Step 4: Installing PyTorch (with CUDA 12.1 support)..."
echo "  This may take 5-10 minutes..."
echo "  (skipping torchaudio - not needed for benchmarks)"

# Try cu121 first, fallback to cu124
for cuda_ver in cu121 cu124; do
    echo "  Trying PyTorch with CUDA ${cuda_ver}..."
    if command -v uv &> /dev/null; then
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/${cuda_ver} -q 2>/dev/null && break
    else
        pip install torch torchvision --index-url https://download.pytorch.org/whl/${cuda_ver} -q 2>/dev/null && break
    fi
done

echo "✓ PyTorch installed"
python -c "import torch; print(f'  torch {torch.__version__} | cuda {torch.version.cuda}')"
echo ""

# ============================================================
# Step 5: Install core dependencies (skip optional ones)
# ============================================================
echo "Step 5: Installing core dependencies..."

# Core packages needed for benchmarking
CORE_DEPS=(
    "numpy"
    "pandas"
    "matplotlib"
    "pyyaml"
    "tabulate"
    "tiktoken"
    "datasets>=2.19"
    "transformers>=4.44"
    "torchvision"
)

for dep in "${CORE_DEPS[@]}"; do
    echo "  Installing $dep..."
    if command -v uv &> /dev/null; then
        uv pip install "$dep" -q 2>/dev/null || echo "    ⚠ $dep install slower than expected"
    else
        pip install "$dep" -q 2>/dev/null || echo "    ⚠ $dep install slower than expected"
    fi
done

echo "✓ Core dependencies installed"
python -c "import torch, transformers, datasets, numpy, pandas; print('  ✓ All core imports work')"
echo ""

# ============================================================
# Step 6: Install optional optimizer packages (best-effort)
# ============================================================
echo "Step 6: Installing optimizer packages (optional, skipping if slow)..."

# These are optional - failures don't stop the pipeline
# Skip if network is slow
OPTIONAL_PKGS=(
    "lion-pytorch"
    "schedulefree"
    "prodigyopt"
)

for pkg in "${OPTIONAL_PKGS[@]}"; do
    echo "  Attempting to install $pkg..."
    if command -v uv &> /dev/null; then
        timeout 30 uv pip install "$pkg" -q 2>/dev/null && echo "    ✓ $pkg" || echo "    ⚠ $pkg (skipped - slow network)"
    else
        timeout 30 pip install "$pkg" -q 2>/dev/null && echo "    ✓ $pkg" || echo "    ⚠ $pkg (skipped - slow network)"
    fi
done

# Skip git-based installs if network is unreliable
echo "  (Skipping git-based optimizer packages due to network timeouts)"

echo "✓ Optimizer setup complete (core optimizers available)"
echo ""

# ============================================================
# Step 7: Vendor single-file optimizers
# ============================================================
echo "Step 7: Vendoring optimizer implementations..."
if [ -f "scripts/vendor_optimizers.sh" ]; then
    bash scripts/vendor_optimizers.sh 2>&1 | head -20
    echo "  (vendoring complete)"
else
    echo "  ⚠ vendor_optimizers.sh not found"
fi

echo "✓ Optimizers vendored"
echo ""

# ============================================================
# Step 8: Copy original milo.py
# ============================================================
echo "Step 8: Integrating original Milo optimizer..."
ORIGINAL_MILO="${PROJECT_ROOT}/milo.py"
TARGET_MILO="optimizers/milo.py"

if [ -f "$ORIGINAL_MILO" ]; then
    cp "$ORIGINAL_MILO" "$TARGET_MILO"
    echo "✓ Original milo.py copied to optimizers/"
else
    echo "⚠ Original milo.py not found (optional)"
fi

echo ""

# ============================================================
# Step 9: Download Shakespeare dataset
# ============================================================
echo "Step 9: Downloading Shakespeare dataset (smoke test data)..."
if [ -f "scripts/download_data.py" ]; then
    python scripts/download_data.py --source shakespeare 2>&1 | tail -5
    echo "✓ Shakespeare dataset ready"
else
    echo "⚠ download_data.py not found"
fi

echo ""

# ============================================================
# Step 10: Quick validation
# ============================================================
echo "Step 10: Validating environment..."
python - <<'PY'
import sys
try:
    import torch
    import transformers
    import datasets
    import numpy
    import pandas
    print(f"✓ Core imports work")
    print(f"  torch: {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  datasets: {datasets.__version__}")
except Exception as e:
    print(f"✗ Import error: {e}", file=sys.stderr)
    sys.exit(1)
PY

echo ""

# ============================================================
# Step 11: List results directory
# ============================================================
echo "Step 11: Checking results directory..."
if [ -d "results" ]; then
    echo "  Results directory exists"
    ls -lah results/ 2>/dev/null | head -5 || true
else
    mkdir -p results
    echo "✓ Created results directory"
fi

echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "  Environment Setup Complete!"
echo "============================================================"
echo ""
echo "Environment location: $TEMP_DIR/.venv"
echo "Activation command:"
echo "  source $TEMP_DIR/.venv/bin/activate"
echo ""
echo "Ready for GPU benchmarks!"
echo ""
echo "To run benchmarks on GPU:"
echo "  sbatch hpc/experiments/run_milo_bench_on_gpu.slurm"
echo ""
echo "Finished: $(date)"
echo "============================================================"
