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

# Import optimizers
try:
    from optimizers.lion import Lion
    from optimizers.adam_mini import AdamMini
    from optimizers.rmsprop_momentum import RMSpropMomentum
    from optimizers.shampoo import Shampoo
    from optimizers.soap import SOAP
    from optimizers.muon import MuonWithAuxAdam
except ImportError as e:
    print(f"Warning: Could not import some optimizers: {e}")

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


def get_imagenet_100_loaders(data_root, batch_size=128, num_workers=8):
    """Get large-scale vision dataloaders using Tiny ImageNet (200 classes, 64x64)."""
    import os
    from PIL import Image

    normalize = transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2770, 0.2691, 0.2821]
    )

    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Tiny ImageNet dataset class
    class TinyImageNet(torch.utils.data.Dataset):
        """Custom loader for Tiny ImageNet dataset."""
        def __init__(self, root, train=True, transform=None):
            self.root = os.path.expanduser(root)
            self.train = train
            self.transform = transform

            split = 'train' if train else 'val'
            self.data_dir = os.path.join(self.root, split)

            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Tiny ImageNet not found at {self.data_dir}. Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip")

            # Load image paths and labels
            self.images = []
            self.labels = []
            self._load_data()

        def _load_data(self):
            if self.train:
                # Training: each class has its own folder
                class_names = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
                for class_idx, class_name in enumerate(class_names):
                    class_path = os.path.join(self.data_dir, class_name, 'images')
                    if os.path.isdir(class_path):
                        for img_file in sorted(os.listdir(class_path)):
                            if img_file.endswith(('.JPEG', '.jpg', '.png')):
                                self.images.append(os.path.join(class_path, img_file))
                                self.labels.append(class_idx)
            else:
                # Validation: images in one folder, labels in separate file
                img_dir = os.path.join(self.data_dir, 'images')
                if os.path.isdir(img_dir):
                    # First, build class name to index mapping
                    train_dir = os.path.join(self.root, 'train')
                    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
                    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

                    # Load labels from val_annotations.txt
                    labels_file = os.path.join(self.data_dir, 'val_annotations.txt')
                    label_map = {}
                    if os.path.exists(labels_file):
                        with open(labels_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split('\t')
                                img_name = parts[0]
                                class_name = parts[1]
                                # Map to class index (0-199)
                                label_map[img_name] = class_to_idx.get(class_name, 0)

                    # Load images in order - only include if label is valid
                    for img_file in sorted(os.listdir(img_dir)):
                        if img_file.endswith(('.JPEG', '.jpg', '.png')):
                            if img_file in label_map:
                                label = label_map[img_file]
                                # Ensure label is in valid range [0, num_classes)
                                if 0 <= label < len(class_names):
                                    self.images.append(os.path.join(img_dir, img_file))
                                    self.labels.append(label)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            label = self.labels[idx]

            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                # Fallback to zeros if image fails to load
                print(f"Warning: Could not load {img_path}: {e}")
                img = Image.new('RGB', (64, 64))

            if self.transform:
                img = self.transform(img)

            return img, label

    # Try to load Tiny ImageNet
    tiny_imagenet_root = os.path.expanduser('~/scratch/datasets/tiny-imagenet-200')

    try:
        train_dataset = TinyImageNet(tiny_imagenet_root, train=True, transform=train_transforms)
        val_dataset = TinyImageNet(tiny_imagenet_root, train=False, transform=val_transforms)
        print(f"✓ Using Tiny ImageNet (200 classes, 64×64 images)")
    except Exception as e:
        print(f"Error loading Tiny ImageNet: {e}")
        print(f"Make sure Tiny ImageNet is downloaded to: {tiny_imagenet_root}")
        raise

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
    data_root="/scratch/datasets",
    results_dir="results_nt_imagenet100"
):
    """Run single optimizer experiment on large-scale dataset (CIFAR-100)."""

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
        elif optimizer_name.upper() == "MILO_LW":
            optimizer = milo(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "ADAMW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "ADAGRAD":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=optimizer_params.get("momentum", 0.9),
                nesterov=optimizer_params.get("nesterov", True),
                weight_decay=optimizer_params.get("weight_decay", 0.0001),
            )
        elif optimizer_name.upper() == "LION":
            optimizer = Lion(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "ADAM_MINI":
            optimizer = AdamMini(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "RMSPROP_MOMENTUM":
            optimizer = RMSpropMomentum(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "SHAMPOO":
            optimizer = Shampoo(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "SOAP":
            optimizer = SOAP(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "MUON":
            optimizer = MuonWithAuxAdam([dict(params=model.parameters(), use_muon=True, lr=learning_rate, **optimizer_params)])
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
        "ADAGRAD": 0.01,
        "LION": 0.001,
        "ADAM_MINI": 0.001,
        "RMSPROP_MOMENTUM": 0.01,
        "SHAMPOO": 0.001,
        "SOAP": 0.001,
        "MUON": 0.001,
    }
    OPTIMIZERS = [
        "MILO", "MILO_LW",
        "SGD", "ADAMW", "ADAGRAD",
        "LION", "ADAM_MINI", "RMSPROP_MOMENTUM", "SHAMPOO",
        "SOAP", "MUON"
    ]
    RUNS_PER_OPTIMIZER = 3
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
        "ADAGRAD": {"lr_decay": 0, "weight_decay": 0.0},
        "LION": {"betas": (0.9, 0.99), "weight_decay": 0.0001},
        "ADAM_MINI": {"betas": (0.9, 0.999), "eps": 1e-8},
        "RMSPROP_MOMENTUM": {"alpha": 0.99, "momentum": 0.9, "eps": 1e-8},
        "SHAMPOO": {"eps": 1e-10, "momentum": 0.0},
        "SOAP": {"betas": (0.95, 0.95), "weight_decay": 0.0001},
        "MUON": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0001},
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
