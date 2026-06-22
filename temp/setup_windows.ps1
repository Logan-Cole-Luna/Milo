# One-shot environment setup for Windows PowerShell
# RTX 5070 Ti (Blackwell, sm_120), Windows 11, CUDA driver >= 570
$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment..." -ForegroundColor Green
uv venv .venv --clear
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }

Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\.venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) { throw "venv activation failed" }

Write-Host "Installing PyTorch with CUDA 13.2 support (RTX 5070 Ti Blackwell)..." -ForegroundColor Green
uv pip install torch --index-url https://download.pytorch.org/whl/cu132
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Green
uv pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements.txt install failed" }

Write-Host "Attempting to install Distributed Shampoo..." -ForegroundColor Green
$oldErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
uv pip install "git+https://github.com/facebookresearch/optimizers.git" 2>&1 | Out-Null
$ErrorActionPreference = $oldErrorAction
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Shampoo install failed -- 'shampoo' optimizer will be unavailable" -ForegroundColor Yellow
}

Write-Host "Vendoring single-file optimizers..." -ForegroundColor Green
bash scripts/vendor_optimizers.sh
if ($LASTEXITCODE -ne 0) { throw "vendor_optimizers.sh failed" }

Write-Host "Validating PyTorch CUDA setup..." -ForegroundColor Green
python -c @"
import torch
print(f'torch {torch.__version__} | cuda {torch.version.cuda} | device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
if torch.cuda.is_available():
    x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
    print('bf16 matmul ok:', x.shape)
else:
    print('WARNING: CUDA not available')
"@

Write-Host "`nDone! Next steps:" -ForegroundColor Green
Write-Host "  1. cp /path/to/your/milo.py optimizers/milo.py     # original Milo"
Write-Host "  2. python scripts/download_data.py --source shakespeare"
Write-Host "  3. bash scripts/smoke_test.sh"
Write-Host "  4. python scripts/download_data.py --source fineweb --train-tokens 2e9"
Write-Host "  5. python sweep/run_sweep.py configs/lm_sweep_small.yaml"
