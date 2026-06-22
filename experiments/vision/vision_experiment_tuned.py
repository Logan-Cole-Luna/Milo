"""
Vision Experiments - Tuned Variant

Runs the same vision experiments but with optimized hyperparameters from tuning phase.
Compares against baseline results.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Import base experiment
from vision_experiment import (
    run_training, device, seed, get_model, get_dataloader, create_train_experiment_fn,
    EXPERIMENTS, BATCH_SIZE, EPOCHS, RUNS_PER_OPTIMIZER, EXPERIMENT_CONFIGS,
    OPTIMIZERS, VAL_SPLIT_RATIO, TEST_SPLIT_RATIO, LR
)

# Import tuned config
from config import (
    OPTIMIZER_PARAMS_TUNED,
    RESULTS_DIR_TUNED,
    VISUALS_DIR_TUNED,
)

if __name__ == "__main__":
    # Use tuned parameters instead of baseline
    OPTIMIZER_PARAMS = OPTIMIZER_PARAMS_TUNED
    RESULTS_DIR = RESULTS_DIR_TUNED
    VISUALS_DIR = VISUALS_DIR_TUNED

    import random
    import numpy as np
    import torch

    print("\n" + "="*70)
    print("  VISION EXPERIMENTS - TUNED HYPERPARAMETERS")
    print("="*70)
    print(f"\nUsing optimized hyperparameters from tuning phase")
    print(f"Results will be saved to: {RESULTS_DIR}/")
    print(f"Visuals will be saved to: {VISUALS_DIR}/")
    print(f"Experiments: {EXPERIMENTS}")
    print(f"Optimizers: {OPTIMIZERS}")
    print(f"Runs per optimizer: {RUNS_PER_OPTIMIZER}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}\n")

    # Run experiments with tuned parameters
    for experiment_type in EXPERIMENTS[0]:
        print(f"\n{'='*70}")
        print(f"  Running: {experiment_type} (TUNED)")
        print(f"{'='*70}")

        config = EXPERIMENT_CONFIGS[experiment_type]
        train_loader_instance = get_dataloader(
            config["dataset_name"],
            config["transforms"],
            BATCH_SIZE,
            train=True,
        )

        # Use the same training function as baseline but with tuned params
        train_fn = create_train_experiment_fn(experiment_type, train_loader_instance)

        base_dir = os.path.dirname(__file__)
        results_dir = os.path.join(base_dir, experiment_type.lower(), RESULTS_DIR)
        visuals_dir = os.path.join(base_dir, experiment_type.lower(), VISUALS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)

        print(f"Results: {results_dir}")
        print(f"Visuals: {visuals_dir}\n")

        # Run experiments with tuned parameters
        for optimizer_name in OPTIMIZERS:
            try:
                base_lr = LR.get(experiment_type, 0.001)
                train_fn(
                    optimizer_name=optimizer_name,
                    base_lr=base_lr,
                    results_dir=results_dir,
                    visuals_dir=visuals_dir,
                )
            except Exception as e:
                print(f"\nError with {experiment_type} - {optimizer_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*70)
    print("VISION TUNED EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults locations:")
    print(f"  Baseline: experiments/vision/results_nt/")
    print(f"  Tuned:    experiments/vision/results_nt_tuned/")
    print(f"\nCompare baseline vs tuned in results JSONs")
    print("="*70)
