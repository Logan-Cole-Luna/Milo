#!/usr/bin/env python3
"""
Direct comparison of MILO base vs MILO accelerated on VGG11 experiments.
Runs both optimizers on the same models and datasets, timing each run.
"""
import sys
import os
import time
import json
import importlib
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import training utilities
from experiments.train_utils import run_training, evaluate_model
from experiments.supervised_learning.network import VGG11
from experiments.supervised_learning.config import (
    BATCH_SIZE, EPOCHS, EXPERIMENTS, LR, OPTIMIZER_PARAMS
)

def get_optimizer_class(optimizer_name):
    """Import and return the optimizer class."""
    if optimizer_name == "milo_base":
        from milo import milo
        return milo, "MILO (Base)"
    elif optimizer_name == "milo_accelerated":
        from milo_accelerated import milo
        return milo, "MILO Accelerated"
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def setup_data(dataset_name, batch_size=128):
    """Load dataset and return DataLoader."""
    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, num_classes

def run_comparison():
    """Run MILO base vs MILO accelerated comparison."""
    results = {
        "milo_base": {},
        "milo_accelerated": {}
    }

    experiments = ["VGG11_CIFAR10", "VGG11_CIFAR100"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running {len(experiments)} experiments with 2 optimizers = {len(experiments) * 2} total runs\n")

    # Create a seed map for experiments to ensure both optimizers use the same initialization
    exp_seed_map = {exp: 42 + i for i, exp in enumerate(experiments)}

    for opt_variant in ["milo_base", "milo_accelerated"]:
        print(f"\n{'='*70}")
        print(f"  {opt_variant.upper()}")
        print(f"{'='*70}")

        optimizer_class, display_name = get_optimizer_class(opt_variant)
        results[opt_variant]["display_name"] = display_name
        results[opt_variant]["experiments"] = {}

        for exp_name in experiments:
            print(f"\n{exp_name}:")

            # Parse experiment config
            if "CIFAR10" in exp_name:
                dataset_name = "CIFAR10"
                num_classes = 10
            else:
                dataset_name = "CIFAR100"
                num_classes = 100

            # Load data
            train_loader, test_loader, _ = setup_data(dataset_name, BATCH_SIZE)

            # CRITICAL: Set seed before creating model to ensure both optimizers use identical initialization
            torch.manual_seed(exp_seed_map[exp_name])

            # Create model
            model = VGG11(num_classes=num_classes).to(device)

            # Create optimizer
            lr = LR[exp_name]
            optimizer_params = OPTIMIZER_PARAMS["MILO"].copy()
            optimizer_params["lr"] = lr
            optimizer = optimizer_class(model.parameters(), **optimizer_params)

            # Loss function
            criterion = nn.CrossEntropyLoss()

            # Train
            print(f"  Training for {EPOCHS} epochs...")
            start_time = time.time()

            for epoch in range(EPOCHS):
                model.train()
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    if (batch_idx + 1) % 50 == 0:
                        print(f"    Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

                # Evaluate
                model.eval()
                correct = 0
                total = 0
                test_loss = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)
                        test_loss += loss.item()

                        _, predicted = output.max(1)
                        correct += predicted.eq(target).sum().item()
                        total += target.size(0)

                accuracy = 100 * correct / total
                avg_test_loss = test_loss / len(test_loader)
                print(f"    Epoch {epoch+1} - Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")

            elapsed = time.time() - start_time

            # Store results
            results[opt_variant]["experiments"][exp_name] = {
                "time": elapsed,
                "accuracy": accuracy,
                "loss": avg_test_loss
            }

            print(f"  Completed in {elapsed:.2f}s")

            # Cleanup
            del model, optimizer
            torch.cuda.empty_cache()

    return results

def print_comparison(results):
    """Print comparison table."""
    print(f"\n\n{'='*70}")
    print("  COMPARISON RESULTS")
    print(f"{'='*70}")

    milo_base_results = results["milo_base"]
    milo_accel_results = results["milo_accelerated"]

    print(f"\n{'Experiment':<25} {'MILO (Base)':<20} {'MILO Accel':<20} {'Speedup':<10}")
    print("-" * 75)

    total_base = 0
    total_accel = 0

    for exp_name in milo_base_results["experiments"]:
        base_time = milo_base_results["experiments"][exp_name]["time"]
        accel_time = milo_accel_results["experiments"][exp_name]["time"]
        speedup = base_time / accel_time if accel_time > 0 else 0

        total_base += base_time
        total_accel += accel_time

        print(f"{exp_name:<25} {base_time:>8.2f}s        {accel_time:>8.2f}s        {speedup:>7.2f}x")

    print("-" * 75)
    total_speedup = total_base / total_accel if total_accel > 0 else 0
    print(f"{'TOTAL':<25} {total_base:>8.2f}s        {total_accel:>8.2f}s        {total_speedup:>7.2f}x")

    # Save to JSON
    output_file = "milo_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    print("MILO vs MILO Accelerated Comparison")
    print("=" * 70)

    results = run_comparison()
    print_comparison(results)
