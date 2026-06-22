"""
ImageNet Experiments - Tuned Variant

Runs Tiny ImageNet-200 with optimized hyperparameters from tuning phase.
Compares against baseline results.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Import base experiment function
from imagenet_experiment import run_imagenet_experiment

# Import tuned config
from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATES,
    OPTIMIZERS,
    RUNS_PER_OPTIMIZER,
    OPTIMIZER_PARAMS_TUNED,
    RESULTS_DIR_TUNED,
    DATASET_NAME,
    DATA_ROOT as CONFIG_DATA_ROOT,
    MODEL_NAME,
)

if __name__ == "__main__":
    DATA_ROOT = os.getenv("IMAGENET_DATA", CONFIG_DATA_ROOT)

    print("\n" + "="*70)
    print("  TINY IMAGENET-200 - TUNED HYPERPARAMETERS")
    print("="*70)
    print(f"\nUsing optimized hyperparameters from tuning phase")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Data location: {DATA_ROOT}")
    print(f"Results directory: {RESULTS_DIR_TUNED}/")
    print(f"Optimizers: {OPTIMIZERS}")
    print(f"Runs per optimizer: {RUNS_PER_OPTIMIZER}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}\n")

    # Run experiments with tuned parameters
    for optimizer_name in OPTIMIZERS:
        try:
            run_imagenet_experiment(
                model_name=MODEL_NAME,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATES[optimizer_name],
                optimizer_name=optimizer_name,
                optimizer_params=OPTIMIZER_PARAMS_TUNED[optimizer_name],
                runs=RUNS_PER_OPTIMIZER,
                data_root=DATA_ROOT,
                results_dir=RESULTS_DIR_TUNED,
            )
        except Exception as e:
            print(f"\nError running {optimizer_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("IMAGENET TUNED EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults locations:")
    print(f"  Baseline: experiments/imagenet/results_nt_imagenet200/")
    print(f"  Tuned:    experiments/imagenet/results_nt_imagenet200_tuned/")
    print(f"\nCompare baseline vs tuned in results JSONs")
    print("="*70)
