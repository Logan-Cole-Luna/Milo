#!/bin/bash
# ============================================================
# Download All Datasets for Offline HPC
#
# MUST be run BEFORE submitting comprehensive suite
# Run on machine with internet (login node or local)
#
# Usage:
#   bash hpc/download_all_datasets.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="${1:-.}"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Downloading All Datasets for Offline HPC"
echo "=========================================="
echo ""

# Activate environment
if [ -f ".venv_cc/bin/activate" ]; then
    source .venv_cc/bin/activate
    echo "✓ Activated .venv_cc"
else
    echo "✗ .venv_cc not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Downloading datasets..."
echo "=========================================="
echo ""

python << 'PYTHON_SCRIPT'
import os
import torch
from torchvision import datasets, transforms

# Create dataset directories
data_dirs = {
    'mnist': os.path.expanduser('~/scratch/datasets/mnist'),
    'cifar10': os.path.expanduser('~/scratch/datasets/cifar10'),
    'cifar100': os.path.expanduser('~/scratch/datasets/cifar100'),
}

for name, path in data_dirs.items():
    os.makedirs(path, exist_ok=True)

print("Downloading datasets (this may take several minutes)...\n")

# Download MNIST
print("1. Downloading MNIST...")
try:
    datasets.MNIST(root=data_dirs['mnist'], train=True, download=True)
    datasets.MNIST(root=data_dirs['mnist'], train=False, download=True)
    print("   ✓ MNIST train and test downloaded")
except Exception as e:
    print(f"   ⚠ MNIST download failed: {e}")
    print("   (Continue without it - CIFAR experiments don't need MNIST)")

# Download CIFAR-10
print("2. Downloading CIFAR-10...")
try:
    datasets.CIFAR10(root=data_dirs['cifar10'], train=True, download=True)
    datasets.CIFAR10(root=data_dirs['cifar10'], train=False, download=True)
    print("   ✓ CIFAR-10 train and test downloaded")
except Exception as e:
    print(f"   ⚠ CIFAR-10 failed: {e}")

# Download CIFAR-100
print("3. Downloading CIFAR-100...")
try:
    datasets.CIFAR100(root=data_dirs['cifar100'], train=True, download=True)
    datasets.CIFAR100(root=data_dirs['cifar100'], train=False, download=True)
    print("   ✓ CIFAR-100 train and test downloaded")
except Exception as e:
    print(f"   ⚠ CIFAR-100 failed: {e}")

print("")
print("==========================================")
print("Dataset download complete!")
print("==========================================")
print("")
print("Datasets located at:")
for name, path in data_dirs.items():
    if os.path.exists(path):
        size = os.popen(f"du -sh {path} 2>/dev/null | cut -f1").read().strip()
        print(f"  {name}: {path} ({size})")
    else:
        print(f"  {name}: {path} (NOT DOWNLOADED)")

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Ready for offline execution!"
echo "=========================================="
