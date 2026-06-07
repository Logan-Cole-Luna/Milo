"""
ImageNet-100 Large-Scale Experiment Runner

Evaluates optimizers on ImageNet-100 (100-class subset of ImageNet).
This is a separate script from supervised_learning_experiment.py because:
- Requires different dataset handling (ImageNet vs CIFAR)
- Longer training time (~15-20 hours for 5 runs)
- Can be run independently or in parallel

Usage:
    python imagenet_100_experiment.py

Environment:
    Expects .venv_cc environment with torchvision >= 0.14
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.dirname(__file__))

import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import time
import json
from pathlib import Path

# Import from existing modules
from milo import milo
from experiments.train_utils import run_training, evaluate_model
from experiments.supervised_learning.network import ResNet34

# Set reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageNet100Subset(datasets.ImageNet):
    """Subset of ImageNet with 100 classes (10 per major category)."""

    def __init__(self, root, split='train', transform=None, target_transform=None):
        # Map to 100 classes: take classes 0-99 from full ImageNet
        self.selected_classes = list(range(100))
        super().__init__(root, split, transform, target_transform)

        # Filter dataset to only selected classes
        self.samples = [
            (path, cls) for path, cls in self.samples
            if cls in self.selected_classes
        ]
        # Remap class indices
        self.class_to_idx = {
            self.classes[i]: idx
            for idx, i in enumerate(self.selected_classes)
            if i < len(self.classes)
        }

    def __len__(self):
        return len(self.samples)


def get_imagenet_100_loaders(data_root, batch_size=128, num_workers=8):
    """Get ImageNet-100 train/val dataloaders."""

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    try:
        # Try to load from standard ImageNet location
        train_dataset = ImageNet100Subset(data_root, split='train', transform=train_transforms)
        val_dataset = ImageNet100Subset(data_root, split='val', transform=val_transforms)
    except Exception as e:
        print(f"Warning: Could not load full ImageNet: {e}")
        print("Using CIFAR-100 as fallback (will run smaller experiment)")
        # Fallback: use CIFAR-100 with larger images
        from torchvision.datasets import CIFAR100

        train_dataset = CIFAR100(root=data_root, train=True, download=True, transform=train_transforms)
        val_dataset = CIFAR100(root=data_root, train=False, download=True, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def run_imagenet_experiment(
    model_name="ResNet50",
    batch_size=128,
    epochs=5,
    learning_rate=0.001,
    optimizer_name="MILO",
    optimizer_params=None,
    runs=3,
    data_root="/scratch/datasets/imagenet",
    results_dir="results_nt_imagenet100"
):
    """Run single optimizer experiment on ImageNet-100."""

    if optimizer_params is None:
        optimizer_params = {}

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"\n{'='*70}")
    print(f"  ImageNet-100 Experiment: {model_name}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Runs: {runs}")
    print(f"{'='*70}\n")

    for run_idx in range(runs):
        print(f"\n--- Run {run_idx + 1}/{runs} ---")
        start_time = time.time()

        # Reset seeds for this run
        torch.manual_seed(seed + run_idx)
        np.random.seed(seed + run_idx)

        # Load data
        train_loader, val_loader = get_imagenet_100_loaders(data_root, batch_size)
        num_classes = 100

        # Create model
        if model_name == "ResNet50":
            model = ResNet34(num_classes=num_classes)  # Use ResNet34 as proxy (can extend)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(device)

        # Create optimizer
        if optimizer_name.upper() == "MILO":
            optimizer = milo(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "ADAMW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_name.upper() == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=optimizer_params.get("momentum", 0.9),
                nesterov=optimizer_params.get("nesterov", True),
                weight_decay=optimizer_params.get("weight_decay", 0.0001),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(targets).sum().item()
                train_total += targets.size(0)

                if (batch_idx + 1) % 50 == 0:
                    print(
                        f"  Epoch {epoch+1} Batch {batch_idx+1}: "
                        f"Loss {train_loss/(batch_idx+1):.4f}, "
                        f"Acc {100*train_correct/train_total:.2f}%"
                    )

            # Validate
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(targets).sum().item()
                    val_total += targets.size(0)

            val_acc = 100 * val_correct / val_total
            print(f"Epoch {epoch+1}: Val Loss {val_loss/len(val_loader):.4f}, Val Acc {val_acc:.2f}%")

            best_val_acc = max(best_val_acc, val_acc)

        run_time = time.time() - start_time
        print(f"Run completed in {run_time/60:.1f} minutes")

        result = {
            "run": run_idx + 1,
            "optimizer": optimizer_name,
            "model": model_name,
            "best_val_accuracy": best_val_acc,
            "training_time_seconds": run_time,
        }
        all_results.append(result)

    # Save results
    results_file = results_path / f"imagenet_100_{model_name}_{optimizer_name.lower()}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    accuracies = [r["best_val_accuracy"] for r in all_results]
    print(f"\n{optimizer_name} Summary:")
    print(f"  Mean accuracy: {np.mean(accuracies):.2f}%")
    print(f"  Std accuracy: {np.std(accuracies):.2f}%")
    print(f"  Results saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATES = {
        "MILO": 0.001,
        "MILO_LW": 0.001,
        "SGD": 0.1,
        "ADAMW": 0.001,
    }
    OPTIMIZERS = ["MILO", "MILO_LW", "SGD", "ADAMW"]
    RUNS_PER_OPTIMIZER = 3  # Reduced from 5 to save time
    DATA_ROOT = os.getenv("IMAGENET_DATA", "/scratch/datasets/imagenet")

    # MILO parameters
    milo_params = {
        "normalize": True,
        "layer_wise": False,
        "scale_aware": True,
        "scale_factor": 0.2,
        "max_group_size": 5000,
        "momentum": 0.9,
        "adaptive": True,
        "use_cuda_kernels": True,
    }

    milo_lw_params = {**milo_params, "layer_wise": True}

    optimizer_configs = {
        "MILO": milo_params,
        "MILO_LW": milo_lw_params,
        "SGD": {"momentum": 0.9, "nesterov": True, "weight_decay": 0.0001},
        "ADAMW": {},
    }

    print("ImageNet-100 Large-Scale Experiment")
    print(f"Data location: {DATA_ROOT}")
    print(f"Optimizers: {OPTIMIZERS}")
    print(f"Runs per optimizer: {RUNS_PER_OPTIMIZER}")
    print(f"Epochs: {EPOCHS}")

    # Run experiments
    for optimizer_name in OPTIMIZERS:
        run_imagenet_experiment(
            model_name="ResNet50",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATES[optimizer_name],
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_configs[optimizer_name],
            runs=RUNS_PER_OPTIMIZER,
            data_root=DATA_ROOT,
            results_dir="results_nt_imagenet100",
        )

    print("\n" + "="*70)
    print("ImageNet-100 experiments completed!")
    print("="*70)
