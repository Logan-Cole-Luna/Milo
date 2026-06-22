#!/bin/bash
# ============================================================
# Download Tiny ImageNet for Large-Scale Experiments
#
# MUST be run on login node with internet access
# Usage:
#   bash hpc/download_tiny_imagenet.sh
#
# Downloads ~230MB and extracts to ~/scratch/datasets/tiny-imagenet-200
# ============================================================

set -euo pipefail

PROJECT_ROOT="${1:-.}"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Downloading Tiny ImageNet"
echo "=========================================="
echo ""

# Create dataset directory
mkdir -p ~/scratch/datasets

DATASET_DIR=~/scratch/datasets/tiny-imagenet-200

if [ -d "$DATASET_DIR" ]; then
    echo "✓ Tiny ImageNet already downloaded at $DATASET_DIR"
    echo ""
    ls -lh "$DATASET_DIR" | head -5
    exit 0
fi

echo "Downloading Tiny ImageNet (200 classes, ~230MB)..."
echo "This may take 5-10 minutes..."
echo ""

# Download
cd ~/scratch/datasets

if ! command -v wget &> /dev/null; then
    echo "Using curl instead of wget..."
    curl -L -O http://cs231n.stanford.edu/tiny-imagenet-200.zip
else
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
fi

echo ""
echo "✓ Download complete"
echo ""

echo "Extracting..."
unzip -q tiny-imagenet-200.zip
rm tiny-imagenet-200.zip

echo ""
echo "=========================================="
echo "Tiny ImageNet Ready!"
echo "=========================================="
echo ""

ls -lh "$DATASET_DIR"
du -sh "$DATASET_DIR"

echo ""
echo "Location: $DATASET_DIR"
echo "Training samples: $(find $DATASET_DIR/train -type f -name '*.JPEG' | wc -l)"
echo "Validation samples: $(find $DATASET_DIR/val -type f -name '*.JPEG' | wc -l)"
echo ""
echo "Ready for offline HPC experiments!"
