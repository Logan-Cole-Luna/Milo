#!/usr/bin/env python3
"""
Test script for MILO CUDA optimizers.
Validates basic functionality and performance of CUDA-accelerated MILO variants.
"""
import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from milo import milo
from torch.optim import SGD, AdamW
from experiments.supervised_learning.config import EXPERIMENT_CONFIGS, OPTIMIZER_PARAMS, LR

# Try to import CUDA optimizers
try:
    from milo import milo
    print("✓ MILO imported successfully")
    MILO_AVAILABLE = True
except ImportError as e:
    print(f"✗ MILO import failed: {e}")
    MILO_AVAILABLE = False

def create_test_model():
    """Create a simple CNN for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def test_optimizer(optimizer_name, optimizer_class, model, data, target, **kwargs):
    """Test an optimizer with given data."""
    print(f"\n--- Testing {optimizer_name} ---")
    
    # Reset model parameters
    for param in model.parameters():
        if param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    # Create optimizer
    try:
        optimizer = optimizer_class(model.parameters(), lr=0.01, **kwargs)
        print(f"✓ {optimizer_name} optimizer created")
    except Exception as e:
        print(f"✗ {optimizer_name} optimizer creation failed: {e}")
        return None
    
    # Test forward pass
    try:
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        print(f"✓ Forward pass successful, loss: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return None
    
    # Test backward pass and optimizer step
    try:
        start_time = time.time()
        
        for step in range(5):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            # Measure step time
            step_start = time.time()
            optimizer.step()
            step_time = time.time() - step_start
            
            if step == 0:
                print(f"✓ First optimization step successful, step time: {step_time*1000:.2f}ms")
        
        total_time = time.time() - start_time
        avg_step_time = total_time / 5
        
        print(f"✓ {optimizer_name} completed 5 steps")
        print(f"  Average step time: {avg_step_time*1000:.2f}ms")
        print(f"  Final loss: {loss.item():.4f}")
        
        return {
            'avg_step_time': avg_step_time,
            'final_loss': loss.item(),
            'success': True
        }
        
    except Exception as e:
        print(f"✗ Optimization step failed: {e}")
        return {'success': False, 'error': str(e)}

def make_dataloader(dataset_name: str, transform, batch_size: int, device: torch.device):
    if dataset_name.upper() == 'CIFAR10':
        ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name.upper() == 'CIFAR100':
        ds = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif dataset_name.upper() == 'MNIST':
        ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    # Keep it small and fast: subset first 4096 samples
    subset_size = min(4096, len(ds))
    ds = torch.utils.data.Subset(ds, list(range(subset_size)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type=='cuda'))
    return loader


def build_milo_from_config(name: str, params: dict):
    def ctor(model_params, lr):
        return milo(model_params, lr=lr, **params)
    return ctor


def benchmark_optimizers():
    """Benchmark optimizers with a mini training pipeline using config settings."""
    print("=" * 60)
    print("MILO CUDA Optimizer Benchmark")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    # Choose an experiment config (match your training runs)
    exp_key = 'VGG11_CIFAR10'
    exp_cfg = EXPERIMENT_CONFIGS[exp_key]
    transform = exp_cfg['transform']
    dataset_name = exp_cfg['dataset_name']
    num_classes = exp_cfg['model_args'].get('num_classes', 10)
    batch_size = 128
    steps_limit = 100  # run limited steps for speed while capturing pipeline cost

    # DataLoader
    loader = make_dataloader(dataset_name, transform, batch_size, device)
    
    # Test optimizers
    # Prepare optimizer constructors using experiment defaults
    milo_params = OPTIMIZER_PARAMS['MILO'].copy()
    milo_lw_params = OPTIMIZER_PARAMS['MILO_LW'].copy()
    base_lr = LR['VGG11_CIFAR10']

    optimizers_to_test = [
        ("MILO (baseline)", build_milo_from_config('MILO', milo_params), {}),
        ("MILO_LW", build_milo_from_config('MILO_LW', milo_lw_params), {}),
        ("SGD", lambda params, **kw: SGD(params, lr=base_lr, momentum=OPTIMIZER_PARAMS['SGD']['momentum'], nesterov=OPTIMIZER_PARAMS['SGD']['nesterov'], weight_decay=OPTIMIZER_PARAMS['SGD']['weight_decay']), {}),
        ("ADAMW", lambda params, **kw: AdamW(params, lr=base_lr, betas=OPTIMIZER_PARAMS['ADAMW']['betas'], eps=OPTIMIZER_PARAMS['ADAMW']['eps'], weight_decay=OPTIMIZER_PARAMS['ADAMW']['weight_decay']), {}),
    ]
    
    if MILO_AVAILABLE:
        optimizers_to_test.extend([
            ("MILO CUDA", milo, {'use_cuda_kernels': True}),
            ("MILO CUDA (fallback)", milo, {'force_cuda_fallback': True}),
        ])
    
    results = {}
    
    criterion = nn.CrossEntropyLoss()

    def train_n_steps(model, optimizer, loader, steps):
        model.train()
        seen = 0
        total_time = 0.0
        data_time = 0.0
        step_time = 0.0
        end = time.time()
        for i, (x, y) in enumerate(loader):
            data_time += time.time() - end
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            t1 = time.time()
            optimizer.step()
            step_time += time.time() - t1
            total_time += time.time() - t0
            seen += x.size(0)
            if (i + 1) >= steps:
                break
            end = time.time()
        return {
            'total_time': total_time,
            'data_time': data_time,
            'opt_time': step_time,
            'samples': seen,
            'avg_step_time': total_time / max(1, steps),
        }

    for name, opt_ctor, kwargs in optimizers_to_test:
        model = create_test_model().to(device)
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            if model.fc.out_features != num_classes:
                model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
        optimizer = opt_ctor(model.parameters(), lr=base_lr)

        print(f"\n--- Pipeline Benchmark: {name} ---")
        stats = train_n_steps(model, optimizer, loader, steps_limit)
        print(f"  Data time: {stats['data_time']*1000/steps_limit:.2f} ms/step")
        print(f"  Opt time:  {stats['opt_time']*1000/steps_limit:.2f} ms/step")
        print(f"  Total:     {stats['avg_step_time']*1000:.2f} ms/step")
        results[name] = {
            'avg_step_time': stats['avg_step_time'],
            'final_loss': None,
            'success': True,
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    successful_optimizers = [name for name, result in results.items() if result.get('success', False)]
    
    if successful_optimizers:
        print(f"✓ {len(successful_optimizers)} optimizers working correctly")
        
        # Sort by step time
        sorted_results = sorted(
            [(name, result) for name, result in results.items() if result.get('success', False)],
            key=lambda x: x[1]['avg_step_time']
        )
        
        print("\nPerformance Ranking (fastest to slowest):")
        for i, (name, result) in enumerate(sorted_results, 1):
            step_time = result['avg_step_time'] * 1000
            print(f"{i:2d}. {name:25s} {step_time:6.2f}ms/step")
        
        # Calculate speedups
        if len(sorted_results) > 1:
            baseline_time = sorted_results[-1][1]['avg_step_time']  # slowest as baseline
            print(f"\nSpeedup vs slowest ({sorted_results[-1][0]}):")
            for name, result in sorted_results[:-1]:
                speedup = baseline_time / result['avg_step_time']
                print(f"  {name:25s} {speedup:4.1f}x faster")
    
    else:
        print("✗ No optimizers working correctly")
    
    failed_optimizers = [name for name, result in results.items() if not result.get('success', False)]
    if failed_optimizers:
        print(f"\n✗ {len(failed_optimizers)} optimizers failed:")
        for name in failed_optimizers:
            error = results[name].get('error', 'Unknown error')
            print(f"  - {name}: {error}")

def test_cuda_compilation():
    """Test if CUDA kernels can be compiled and loaded."""
    print("\n" + "=" * 60)
    print("CUDA COMPILATION TEST")
    print("=" * 60)
    
    try:
        from milo_cuda_ops import MiloCudaOps
        print("✓ CUDA operations module imported successfully")
        
        # Test if we can create the operations
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            # Create test tensors
            test_grad = torch.randn(100, 50).cuda()
            param_sizes = torch.tensor([100, 50], dtype=torch.int32).cuda()
            param_offsets = torch.tensor([0, 100], dtype=torch.int32).cuda()
            layer_indices = torch.tensor([0, 1], dtype=torch.int32).cuda()
            
            try:
                MiloCudaOps.fused_gradient_normalization(
                    [test_grad],
                    param_sizes,
                    param_offsets,
                    layer_indices,
                    eps=1e-5,
                    scale_factor=0.2,
                    scale_aware=True,
                    layer_wise=True
                )
                print("✓ CUDA gradient normalization kernel working")
            except Exception as e:
                print(f"✗ CUDA gradient normalization kernel failed: {e}")
            
        else:
            print("! CUDA not available, skipping kernel tests")
            
    except ImportError as e:
        print(f"✗ CUDA operations module import failed: {e}")
        print("  This is expected if CUDA kernels haven't been compiled yet")

if __name__ == "__main__":
    test_cuda_compilation()
    benchmark_optimizers()
