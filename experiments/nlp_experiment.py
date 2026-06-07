"""
BERT Fine-tuning Experiment for NLP

Evaluates optimizers on GLUE SST-2 (Sentiment Classification task).
This is a separate script because:
- NLP task requires HuggingFace transformers
- Different training pipeline than vision
- Independent of vision experiments

Usage:
    python nlp_experiment.py

Requirements:
    pip install transformers datasets evaluate

Environment:
    Expects .venv_cc with HuggingFace libraries

NOTE: HPC Offline Mode
    - Pre-download BERT model locally before running
    - HF_HOME env var should point to local cache
    - Run on login node first to cache datasets/models
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path
import time

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    import evaluate
except ImportError as e:
    print(f"Error: Required transformers library not found: {e}")
    print("Install with: pip install transformers datasets evaluate")
    sys.exit(1)

# Import optimizers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from milo import milo
except ImportError:
    print("Warning: Could not import milo, will use standard PyTorch optimizers")
    milo = None

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sst2_dataset(max_length=128):
    """Load SST-2 (Stanford Sentiment Treebank) dataset."""
    print("Loading SST-2 dataset...")

    # Load from HuggingFace datasets
    dataset = load_dataset("glue", "sst2")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess(example):
        return tokenizer(
            example["sentence"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    # Apply tokenization
    dataset = dataset.map(preprocess, batched=True, remove_columns=["sentence", "idx"])
    dataset = dataset.rename_column("label", "labels")

    return dataset, tokenizer


def train_epoch(model, train_loader, optimizer, device, use_milo=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (standard for NLP)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Tracking
        total_loss += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100 * total_correct / total_samples
    return avg_loss, avg_acc


def evaluate_model(model, eval_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)

    avg_loss = total_loss / len(eval_loader)
    avg_acc = 100 * total_correct / total_samples
    return avg_loss, avg_acc


def run_bert_experiment(
    optimizer_name="AdamW",
    optimizer_params=None,
    batch_size=32,
    epochs=3,
    learning_rate=2e-5,
    runs=5,
    results_dir="results_nt_bert",
):
    """Run BERT fine-tuning experiment."""

    if optimizer_params is None:
        optimizer_params = {}

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"\n{'='*70}")
    print(f"  BERT Fine-tuning: SST-2 Sentiment Classification")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Runs: {runs}")
    print(f"  Epochs: {epochs}")
    print(f"{'='*70}\n")

    # Load dataset once (reuse for all runs)
    dataset, tokenizer = load_sst2_dataset()

    for run_idx in range(runs):
        print(f"\n--- Run {run_idx + 1}/{runs} ---")
        run_start = time.time()

        # Reset seeds
        torch.manual_seed(seed + run_idx)
        np.random.seed(seed + run_idx)

        # Load fresh model
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        model.to(device)

        # Create dataloaders
        train_loader = DataLoader(
            dataset["train"], batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            dataset["validation"], batch_size=batch_size, shuffle=False
        )

        # Create optimizer
        if optimizer_name.upper() == "MILO" and milo is not None:
            optimizer = milo(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "MILO_LW" and milo is not None:
            optimizer = milo(model.parameters(), lr=learning_rate, **optimizer_params)
        elif optimizer_name.upper() == "ADAMW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_name.upper() == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=optimizer_params.get("momentum", 0.9),
                weight_decay=optimizer_params.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Training loop
        best_val_acc = 0
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, use_milo=(milo is not None)
            )
            val_loss, val_acc = evaluate_model(model, val_loader, device)

            print(
                f"  Epoch {epoch+1}: "
                f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, "
                f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss

        run_time = time.time() - run_start
        print(f"Run completed in {run_time/60:.1f} minutes")

        result = {
            "run": run_idx + 1,
            "optimizer": optimizer_name,
            "task": "SST-2",
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            "training_time_seconds": run_time,
        }
        all_results.append(result)

    # Save results
    results_file = results_path / f"bert_sst2_{optimizer_name.lower()}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    accuracies = [r["best_val_accuracy"] for r in all_results]
    print(f"\n{optimizer_name} Summary (SST-2):")
    print(f"  Mean accuracy: {np.mean(accuracies):.2f}%")
    print(f"  Std accuracy: {np.std(accuracies):.2f}%")
    print(f"  Results saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATES = {
        "MILO": 1e-4,
        "MILO_LW": 1e-4,
        "SGD": 1e-3,
        "ADAMW": 2e-5,  # Standard BERT LR
    }
    OPTIMIZERS = ["MILO", "MILO_LW", "SGD", "ADAMW"]
    RUNS_PER_OPTIMIZER = 5

    # MILO parameters
    milo_params = {
        "normalize": True,
        "layer_wise": False,
        "scale_aware": True,
        "scale_factor": 0.2,
        "max_group_size": 5000,
        "momentum": 0.9,
        "adaptive": True,
        "use_cuda_kernels": False,  # Transformers may not benefit from CUDA kernels
    }

    milo_lw_params = {**milo_params, "layer_wise": True}

    optimizer_configs = {
        "MILO": milo_params,
        "MILO_LW": milo_lw_params,
        "SGD": {"momentum": 0.9, "weight_decay": 0.01},
        "ADAMW": {},
    }

    print("BERT Fine-tuning Experiment on SST-2")
    print(f"Optimizers: {OPTIMIZERS}")
    print(f"Runs per optimizer: {RUNS_PER_OPTIMIZER}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    # Run experiments
    for optimizer_name in OPTIMIZERS:
        try:
            run_bert_experiment(
                optimizer_name=optimizer_name,
                optimizer_params=optimizer_configs[optimizer_name],
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATES[optimizer_name],
                runs=RUNS_PER_OPTIMIZER,
                results_dir="results_nt_bert",
            )
        except Exception as e:
            print(f"\nError running {optimizer_name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "="*70)
    print("BERT fine-tuning experiments completed!")
    print("="*70)
