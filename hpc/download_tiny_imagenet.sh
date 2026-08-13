#!/bin/bash
# ============================================================
# Download Tiny ImageNet-200 to HPC Scratch
#
# Run on LOGIN NODE (has internet)
# Usage: bash hpc/download_tiny_imagenet.sh
# ============================================================

set -euo pipefail

SCRATCH_DIR="/home/logan03/scratch/datasets"
DATASET_DIR="$SCRATCH_DIR/tiny-imagenet-200"

echo "============================================================"
echo "  Downloading Tiny ImageNet-200"
echo "============================================================"
echo ""
echo "Destination: $DATASET_DIR"
echo ""

mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# Check if already exists
if [ -d "$DATASET_DIR" ]; then
    echo "✓ Already downloaded at: $DATASET_DIR"
    du -sh "$DATASET_DIR"
    exit 0
fi

ZIP_FILE="tiny-imagenet-200.zip"

echo "Downloading from Stanford (240 MB, ~5-10 min)..."
if command -v wget &> /dev/null; then
    wget -O "$ZIP_FILE" http://cs231n.stanford.edu/tiny-imagenet-200.zip 2>&1 | grep -E "saved|failed|ERROR" || echo "Downloading..."
else
    curl -L -o "$ZIP_FILE" http://cs231n.stanford.edu/tiny-imagenet-200.zip
fi

if [ ! -f "$ZIP_FILE" ]; then
    echo "ERROR: Download failed!"
    exit 1
fi

echo "✓ Downloaded"
echo ""
echo "Extracting (may take 1-2 min)..."
unzip -q "$ZIP_FILE"
rm "$ZIP_FILE"

echo ""
echo "============================================================"
echo "✓ Download Complete!"
echo "============================================================"
echo ""
echo "Location: $DATASET_DIR"
echo "Size: $(du -sh "$DATASET_DIR" 2>/dev/null)"
echo ""
echo "Ready for ImageNet baseline experiments!"
