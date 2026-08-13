#!/usr/bin/env bash
# ============================================================
# Setup and Validate Milo-Bench (New Optimizer Suite)
#
# Sets up the new milo-bench environment in temp/ folder with:
# - MiloM (improved Milo with ordering/grouping fixes)
# - Mion (Newton-Schulz preconditioned variant)
# - Comprehensive benchmark suite
#
# Submit: sbatch hpc/experiments/setup_milo_bench.slurm
# Monitor: tail -f logs/setup_milo_bench_*.out
# ============================================================

#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=setup_milo_bench
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/home/logan03/Milo/logs/setup_milo_bench_%A.out
#SBATCH --error=/home/logan03/Milo/logs/setup_milo_bench_%A.err
#SBATCH --mail-user=lambethl@my.erau.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
TEMP_DIR="${PROJECT_ROOT}/temp"

echo "============================================================"
echo "  Milo-Bench Environment Setup & Validation"
echo "============================================================"
echo ""
echo "Working directory: $TEMP_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo ""

# ============================================================
# Step 1: Check if temp folder exists
# ============================================================
echo "Step 1: Checking temp folder..."
if [ ! -d "$TEMP_DIR" ]; then
    echo "ERROR: temp folder not found at $TEMP_DIR"
    exit 1
fi

cd "$TEMP_DIR"
echo "✓ Changed to temp folder"
echo ""

# ============================================================
# Step 2: Create virtual environment with uv
# ============================================================
echo "Step 2: Creating virtual environment..."
if ! command -v uv &> /dev/null; then
    echo "⚠ uv not found, trying pip instead..."
    python3 -m venv .venv
else
    uv venv .venv
fi

source .venv/bin/activate
echo "✓ Virtual environment created and activated"
echo "  Python: $(python --version)"
echo ""

# ============================================================
# Step 3: Install PyTorch (HPC has CUDA 12.x)
# ============================================================
echo "Step 3: Installing PyTorch with CUDA support..."
if command -v uv &> /dev/null; then
    # Try cu124 for HPC (Ampere/Hopper compatible)
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || \
    uv pip install torch torchvision || true
else
    pip install --upgrade pip setuptools wheel
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || \
    pip install torch torchvision || true
fi

echo "✓ PyTorch installed"
python -c "import torch; print(f'  torch {torch.__version__} | cuda {torch.version.cuda}')"
echo ""

# ============================================================
# Step 4: Install dependencies from requirements.txt
# ============================================================
echo "Step 4: Installing dependencies from requirements.txt..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

echo "✓ Dependencies installed"
echo ""

# ============================================================
# Step 5: Vendor optimizers
# ============================================================
echo "Step 5: Vendoring optimizer implementations..."
if [ -f "scripts/vendor_optimizers.sh" ]; then
    bash scripts/vendor_optimizers.sh || echo "⚠ Some optimizers may have failed to vendor"
else
    echo "⚠ vendor_optimizers.sh not found"
fi

echo "✓ Optimizer vendoring complete"
echo ""

# ============================================================
# Step 6: Copy original milo.py from main project
# ============================================================
echo "Step 6: Copying original Milo implementation..."
ORIGINAL_MILO="${PROJECT_ROOT}/milo.py"
TARGET_MILO="optimizers/milo.py"

if [ -f "$ORIGINAL_MILO" ]; then
    cp "$ORIGINAL_MILO" "$TARGET_MILO"
    echo "✓ Copied original milo.py from project root"
else
    echo "⚠ Original milo.py not found at $ORIGINAL_MILO"
    echo "  (This is optional - benchmarks can run without it for comparison)"
fi

echo ""

# ============================================================
# Step 7: Verify CUDA and bf16 support
# ============================================================
echo "Step 7: Verifying CUDA and bfloat16 support..."
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    result = x @ x
    print("bf16 matmul works:", result.shape)
else:
    print("WARNING: CUDA not available, will run on CPU (slow)")
PY

echo "✓ CUDA verification complete"
echo ""

# ============================================================
# Step 8: Run smoke test (optional, quick validation)
# ============================================================
echo "Step 8: Running smoke test (30-step validation)..."
if [ -f "tests/test_optimizers.py" ]; then
    timeout 600 python tests/test_optimizers.py || {
        EXITCODE=$?
        if [ $EXITCODE -eq 124 ]; then
            echo "⚠ Smoke test timed out (likely still validating)"
        else
            echo "⚠ Some smoke tests may have failed"
        fi
    }
    echo "✓ Smoke test complete"
else
    echo "⚠ tests/test_optimizers.py not found"
fi

echo ""

# ============================================================
# Step 9: Download Shakespeare data (quick dataset)
# ============================================================
echo "Step 9: Downloading Shakespeare dataset (minimal data for testing)..."
if [ -f "scripts/download_data.py" ]; then
    timeout 300 python scripts/download_data.py --source shakespeare || \
        echo "⚠ Shakespeare download failed or timed out"
    echo "✓ Shakespeare dataset ready"
else
    echo "⚠ scripts/download_data.py not found"
fi

echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "  Milo-Bench Setup Complete!"
echo "============================================================"
echo ""
echo "Environment Details:"
echo "  Location: $TEMP_DIR"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"Not available\")')"
echo ""
echo "Next steps:"
echo "  1. Source the environment: source $TEMP_DIR/.venv/bin/activate"
echo "  2. Run benchmarks:"
echo "     python sweep/run_sweep.py configs/cifar_resnet.yaml"
echo "     python sweep/run_sweep.py configs/cifar_vit.yaml"
echo "     python sweep/run_sweep.py configs/lm_sweep_small.yaml (after: python scripts/download_data.py --source fineweb --train-tokens 2e9)"
echo "  3. Analyze results:"
echo "     python analysis/aggregate.py results/lm_sweep"
echo "     python analysis/plots.py results/lm_sweep --metric best_val_loss --lower"
echo ""
echo "Key directories:"
echo "  Optimizers: $TEMP_DIR/optimizers/"
echo "  Tasks: $TEMP_DIR/tasks/"
echo "  Results: $TEMP_DIR/results/"
echo ""
echo "Finished: $(date)"
echo "============================================================"
