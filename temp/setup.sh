#!/usr/bin/env bash
# One-shot environment setup. Tested target: RTX 5070 Ti (Blackwell, sm_120),
# Ubuntu 22.04/24.04, CUDA driver >= 570.
set -e

uv venv .venv
source .venv/bin/activate

# IMPORTANT: RTX 50-series (Blackwell) requires PyTorch >= 2.7 built with
# CUDA 12.8. The default PyPI wheel may not include sm_120 kernels.
uv pip install torch --index-url https://download.pytorch.org/whl/cu132

uv pip install -r requirements.txt

# Meta's Distributed Shampoo reference implementation
uv pip install "git+https://github.com/facebookresearch/optimizers.git" || \
  echo "WARNING: Shampoo install failed -- 'shampoo' optimizer will be unavailable"

# Vendor single-file optimizers (pinned upstreams)
bash scripts/vendor_optimizers.sh

# Sanity: torch sees the GPU and sm_120 kernels work
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if torch.cuda.is_available():
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul ok:", (x @ x).shape)
PY

echo
echo "Done. Next steps:"
echo "  1. cp /path/to/your/milo.py optimizers/milo.py     # original Milo"
echo "  2. python scripts/download_data.py --source shakespeare"
echo "  3. bash scripts/smoke_test.sh"
echo "  4. python scripts/download_data.py --source fineweb --train-tokens 2e9"
echo "  5. python sweep/run_sweep.py configs/lm_sweep_small.yaml"
