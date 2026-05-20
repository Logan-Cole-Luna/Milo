#!/usr/bin/env python3
"""
Re-generate plots from saved experiment results.

- Training metrics: uses custom_plotting.CustomPlotter to recreate train_* plots
- Validation metrics: reads raw_runs_validation_data_*.json and plots
- Iteration/walltime: reads *_iteration_logs.json and plots train loss vs steps and vs walltime

Usage:
  python experiments/replot_from_results.py \
    --results-dir experiments/supervised_learning/logistic/results \
    --output-dir experiments/supervised_learning/logistic/test_plots \
    --exclude-optimizers MILO_TUNED MILO_LW_TUNED
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure local imports work
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from plotting import plot_seaborn_style_with_error_bars, setup_plot_style
from custom_plotting import CustomPlotter, create_default_config


def _calc_stats_across_runs(runs: List[List[float]]) -> Tuple[List[float], List[float]]:
    """Compute mean and std error across runs (trim to min length)."""
    if not runs:
        return [], []
    arrays = [np.asarray(r, dtype=float) for r in runs if r is not None]
    if not arrays:
        return [], []
    min_len = min(len(a) for a in arrays)
    if min_len == 0:
        return [], []
    arrays = [a[:min_len] for a in arrays]
    stacked = np.stack(arrays)
    mean = np.mean(stacked, axis=0)
    if len(arrays) > 1:
        std = np.std(stacked, axis=0, ddof=1)
        std_err = std / np.sqrt(len(arrays))
    else:
        std_err = np.zeros_like(mean)
    return mean.tolist(), std_err.tolist()


def find_file_by_glob(base_dir: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(base_dir, pattern))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Re-generate plots from saved results")
    parser.add_argument("--results-dir", required=True, help="Directory with saved results (CSV/JSON)")
    parser.add_argument("--output-dir", required=True, help="Directory to save re-generated plots")
    parser.add_argument("--include-optimizers", nargs="+", help="Optimizers to include")
    parser.add_argument("--exclude-optimizers", nargs="+", default=[], help="Optimizers to exclude")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    include_opts = set(args.include_optimizers) if args.include_optimizers else None
    exclude_opts = set(args.exclude_optimizers) if args.exclude_optimizers else set()

    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()

    # 1) Training metrics via CustomPlotter
    train_csv = find_file_by_glob(results_dir, "*_training_metrics.csv")
    if train_csv:
        cfg = create_default_config()
        if include_opts is not None:
            cfg['include_optimizers'] = list(include_opts)
        if exclude_opts:
            cfg['exclude_optimizers'] = list(exclude_opts)
        plotter = CustomPlotter(cfg)
        df = plotter.load_results(train_csv)
        # Save plots to output_dir
        plotter.plot_multiple_metrics(df, output_dir)
        # Infer experiment id from csv filename
        experiment_file_id = Path(train_csv).stem.replace("_training_metrics", "")
    else:
        print(f"Warning: No training metrics CSV found in {results_dir}")
        experiment_file_id = None

    # 2) Validation metrics from raw_runs_validation_data_*.json
    raw_val_json = find_file_by_glob(results_dir, "raw_runs_validation_data_*.json")
    if raw_val_json:
        with open(raw_val_json, 'r') as f:
            raw_val = json.load(f)
        # Structure: { 'val_losses': {opt: [[...], ...]}, 'val_accuracies': {...}, 'val_f1_scores': {...}, 'val_aucs': {...} }
        metrics_map = {
            'val_losses': ("Validation Loss", "Loss", "val_loss"),
            'val_accuracies': ("Validation Accuracy", "Accuracy (%)", "val_accuracy"),
            'val_f1_scores': ("Validation F1 Score", "F1 Score", "val_f1_score"),
            'val_aucs': ("Validation AUC", "AUC Score", "val_auc"),
        }
        # Determine base title id
        base_title = (experiment_file_id or "experiment").upper()
        # Compute stats per optimizer
        for key, (title_base, y_label, fname_base) in metrics_map.items():
            per_opt_runs: Dict[str, List[List[float]]] = raw_val.get(key, {})
            means: Dict[str, List[float]] = {}
            errs: Dict[str, List[float]] = {}
            for opt_name, runs in per_opt_runs.items():
                if include_opts is not None and opt_name not in include_opts:
                    continue
                if opt_name in exclude_opts:
                    continue
                mean, std_err = _calc_stats_across_runs(runs)
                if mean:
                    means[opt_name] = mean
                    errs[opt_name] = std_err
            if means:
                plot_seaborn_style_with_error_bars(
                    means,
                    errs,
                    list(range(1, len(next(iter(means.values()))) + 1)),
                    f"{title_base} for {base_title}",
                    f"{fname_base}_{experiment_file_id or 'replot'}",
                    y_label,
                    output_dir,
                    xlabel="Epoch"
                )
            else:
                print(f"Warning: No validation data to plot for {key}")
    else:
        print(f"Warning: No validation raw JSON found in {results_dir}")

    # 3) Iteration-level logs (loss vs steps & walltime)
    if experiment_file_id:
        iter_json = os.path.join(results_dir, f"{experiment_file_id}_iteration_logs.json")
    else:
        iter_json = find_file_by_glob(results_dir, "*_iteration_logs.json")
    if iter_json and os.path.exists(iter_json):
        with open(iter_json, 'r') as f:
            payload = json.load(f)
        all_logs = payload.get('iteration_logs', {})
        # Loss vs Steps
        iter_means: Dict[str, List[float]] = {}
        iter_errs: Dict[str, List[float]] = {}
        iter_x: Dict[str, List[int]] = {}
        wall_x: Dict[str, List[float]] = {}
        for opt_name, runs in all_logs.items():
            if include_opts is not None and opt_name not in include_opts:
                continue
            if opt_name in exclude_opts:
                continue
            # Each run is a dict with 'batch_losses' and 'cumulative_walltime'
            batch_losses_runs = [r.get('batch_losses', []) for r in runs]
            mean_loss, err_loss = _calc_stats_across_runs(batch_losses_runs)
            if not mean_loss:
                continue
            iter_means[opt_name] = mean_loss
            iter_errs[opt_name] = err_loss
            iter_x[opt_name] = list(range(1, len(mean_loss) + 1))
            # Walltime X as mean across runs
            walltime_runs = [r.get('cumulative_walltime', []) for r in runs]
            wall_mean, _ = _calc_stats_across_runs(walltime_runs)
            wall_x[opt_name] = wall_mean
        if iter_means:
            # Plot Loss vs Iteration (Steps)
            plot_seaborn_style_with_error_bars(
                iter_means,
                iter_errs,
                iter_x,
                f"Training Loss vs. Iteration for {(experiment_file_id or 'experiment').upper()}",
                f"train_loss_iteration_{experiment_file_id or 'replot'}",
                "Loss (Training)",
                output_dir,
                xlabel="Iteration (Step)",
                yscale='log'
            )
            # Plot Loss vs Walltime
            plot_seaborn_style_with_error_bars(
                iter_means,
                iter_errs,
                wall_x,
                f"Training Loss vs. Wall-clock Time for {(experiment_file_id or 'experiment').upper()}",
                f"train_loss_walltime_{experiment_file_id or 'replot'}",
                "Loss (Training)",
                output_dir,
                xlabel="Time (s)",
                yscale='log'
            )
        else:
            print("Warning: No iteration logs to plot.")
    else:
        print(f"Warning: No iteration logs JSON found in {results_dir}")

    print(f"Replot complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
