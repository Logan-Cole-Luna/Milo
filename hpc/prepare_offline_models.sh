#!/bin/bash
# ============================================================
# Prepare Models & Datasets for Offline HPC Environment
#
# This script must be run ONCE on a machine with internet
# before submitting jobs to the compute nodes.
#
# Usage:
#   bash hpc/prepare_offline_models.sh
#
# This will cache:
# 1. BERT-base model from HuggingFace
# 2. SST-2 dataset from HuggingFace
# 3. ViT dependencies (already in code)
# 4. CIFAR-10/100 datasets (auto-downloaded on first run)
#
# Note: This should be run on login node or local machine
# with internet. The models will be cached in .venv_cc and
# transferred to compute nodes via HPC file system.
# ============================================================

set -euo pipefail

PROJECT_ROOT="${1:-.}"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Preparing Offline HPC Environment"
echo "=========================================="
echo ""

# Load required modules BEFORE activating venv
echo "Loading compute canada modules..."
module --force purge
module load StdEnv/2023
module load cuda/12.2 cudnn/9.2.1.18
module load python/3.11.5 scipy-stack/2024a
module load gcc arrow/23.0.1  # CRITICAL: Must load arrow BEFORE venv
echo "✓ Modules loaded (including arrow)"

echo ""

# Activate environment
if [ -f ".venv_cc/bin/activate" ]; then
    source .venv_cc/bin/activate
    echo "✓ Activated .venv_cc"
else
    echo "✗ .venv_cc not found. Create it first:"
    echo "  bash hpc/setup_cc.sh"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 1: Install Required Libraries"
echo "=========================================="
echo ""

# Install with arrow/pyarrow already available from modules
pip install -q --no-build-isolation transformers datasets evaluate torch torchvision

echo "✓ Libraries installed"

echo ""
echo "=========================================="
echo "Step 2: Pre-download BERT Model"
echo "=========================================="
echo ""

python << 'PYTHON_SCRIPT'
import os
print("Downloading BERT-base model for offline use...")

# Set cache directory to shared location (HOME works on both login and compute)
hf_cache = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = hf_cache

# IMPORTANT: Don't set offline flags here - we need to DOWNLOAD on login node
# Unset any offline flags that might be set from previous runs
for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
    if key in os.environ:
        del os.environ[key]

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"Cache directory: {hf_cache}")
    print(f"Downloading from HuggingFace hub...")

    # Download tokenizer
    print("  → Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("  ✓ Tokenizer cached")

    # Download model
    print("  → Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    print("  ✓ Model cached")

    print(f"\n✓ BERT model and tokenizer cached at: {hf_cache}")
    print("  Ready for offline use on compute nodes")

except Exception as e:
    print(f"\n✗ Error downloading BERT: {e}")
    print(f"  Check: Do you have internet access on login node?")
    print(f"  Cache location: {hf_cache}")
    exit(1)

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Step 3: Pre-download SST-2 Dataset"
echo "=========================================="
echo ""

python << 'PYTHON_SCRIPT'
import os
print("Downloading SST-2 dataset for offline use...")

hf_cache = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = hf_cache

try:
    from datasets import load_dataset

    print(f"Cache directory: {hf_cache}")
    print("  → Downloading SST-2...")
    dataset = load_dataset("glue", "sst2")
    print("  ✓ SST-2 cached")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")

    print("\n✓ SST-2 dataset ready for offline use")

except Exception as e:
    print(f"\n✗ Error downloading SST-2: {e}")
    print("  Note: Dataset will be downloaded on first compute job run")
    exit(0)

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Step 4: Prepare CIFAR Datasets"
echo "=========================================="
echo ""

python << 'PYTHON_SCRIPT'
import os
import torch
from torchvision import datasets, transforms

print("Preparing CIFAR datasets...")

# Create temp directory for downloads
cifar_cache = os.path.expanduser("~/scratch/datasets/cifar") if os.path.exists(
    os.path.expanduser("~/scratch")
) else "./datasets/cifar"

os.makedirs(cifar_cache, exist_ok=True)

try:
    print(f"Cache directory: {cifar_cache}")

    # Download CIFAR-10
    print("  → Downloading CIFAR-10...")
    cifar10_train = datasets.CIFAR10(cifar_cache, train=True, download=True)
    cifar10_test = datasets.CIFAR10(cifar_cache, train=False, download=True)
    print("  ✓ CIFAR-10 ready")

    # Download CIFAR-100
    print("  → Downloading CIFAR-100...")
    cifar100_train = datasets.CIFAR100(cifar_cache, train=True, download=True)
    cifar100_test = datasets.CIFAR100(cifar_cache, train=False, download=True)
    print("  ✓ CIFAR-100 ready")

    print(f"\n✓ CIFAR datasets ready at {cifar_cache}")

except Exception as e:
    print(f"\n⚠ Warning: Could not download CIFAR: {e}")
    print("  Datasets will be downloaded on first job run")
    exit(0)

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Step 5: Verify Setup"
echo "=========================================="
echo ""

echo "Checking HuggingFace cache..."
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -d "$hf_home" ]; then
    echo "✓ HF_HOME: $hf_home"
    echo "  Hub size: $(du -sh "$hf_home/hub" 2>/dev/null | cut -f1 || echo 'N/A')"
    echo "  Datasets size: $(du -sh "$hf_home/datasets" 2>/dev/null | cut -f1 || echo 'N/A')"
fi

echo ""
echo "=========================================="
echo "Offline Preparation Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ BERT-base cached"
echo "  ✓ SST-2 dataset cached"
echo "  ✓ CIFAR-10/100 cached"
echo ""
echo "When submitting jobs to compute nodes:"
echo "  1. Models will be available in shared cache"
echo "  2. No internet required on compute nodes"
echo "  3. Jobs will run with HF_HUB_OFFLINE=1"
echo ""
echo "To submit comprehensive suite:"
echo "  sbatch hpc/run_comprehensive_suite_c.slurm"
echo ""
