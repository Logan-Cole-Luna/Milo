"""
NLP Experiments - Tuned Variant

Runs BERT fine-tuning with optimized hyperparameters from tuning phase.
Compares against baseline results.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Import base experiment function
from nlp_experiment import run_bert_experiment

# Import tuned config
from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATES,
    OPTIMIZERS,
    RUNS_PER_OPTIMIZER,
    OPTIMIZER_PARAMS_TUNED,
    RESULTS_DIR_TUNED,
)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  BERT FINE-TUNING - TUNED HYPERPARAMETERS")
    print("="*70)
    print(f"\nUsing optimized hyperparameters from tuning phase")
    print(f"Dataset: SST-2 Sentiment Classification")
    print(f"Results directory: {RESULTS_DIR_TUNED}/")
    print(f"Optimizers: {OPTIMIZERS}")
    print(f"Runs per optimizer: {RUNS_PER_OPTIMIZER}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}\n")

    # Run experiments with tuned parameters
    for optimizer_name in OPTIMIZERS:
        try:
            run_bert_experiment(
                optimizer_name=optimizer_name,
                optimizer_params=OPTIMIZER_PARAMS_TUNED[optimizer_name],
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATES[optimizer_name],
                runs=RUNS_PER_OPTIMIZER,
                results_dir=RESULTS_DIR_TUNED,
            )
        except Exception as e:
            print(f"\nError running {optimizer_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("BERT TUNED EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults locations:")
    print(f"  Baseline: experiments/nlp/results_nt_bert/")
    print(f"  Tuned:    experiments/nlp/results_nt_bert_tuned/")
    print(f"\nCompare baseline vs tuned in results JSONs")
    print("="*70)
