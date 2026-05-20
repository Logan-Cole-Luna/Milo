import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import time
import json
from .plotting import (
    plot_seaborn_style_with_error_bars,
    plot_resource_usage,
    plot_layer_runtime_breakdown,
    plot_forward_backward_runtime_breakdown
)
from .train_utils import (
    calculate_statistics,
    perform_statistical_tests,
    save_statistics_report,
    save_experimental_settings,
    print_system_info  
)

try:
    from .utils.resource_tracker import ResourceTracker, save_resource_info
    resource_tracking_available = True
except ImportError:
    resource_tracking_available = False
    print("Warning: ResourceTracker not available. Compute resource tracking will be disabled.")

def run_experiments(train_experiment, results_dir, visuals_dir, epochs,
                    optimizer_names=["MILO", "MILO_LW", "SGD", "ADAMW", "ADAGRAD", "NOVOGRAD"],
                    loss_title="Loss vs. Epoch", loss_ylabel="Average Loss", 
                    acc_title="Accuracy vs. Epoch", acc_ylabel="Accuracy (%)",
                    plot_filename="validation_curves", csv_filename="validation_metrics.csv", 
                    experiment_title="Experiment (Validation)", cost_xlimit=None, f1_title="F1 Score vs. Epoch",
                    num_runs=1, experiment_settings=None):
    """Run experiments, plot validation metrics, and save final test results.

    Args:
        train_experiment: Function to run a single experiment for an optimizer.
        results_dir: Directory to save results (CSVs, JSONs).
        visuals_dir: Directory to save plots.
        epochs: Number of epochs to train for.
        optimizer_names: List of optimizer names to test.
        loss_title: Title for the loss plot.
        loss_ylabel: Y-axis label for the loss plot.
        acc_title: Title for the accuracy plot.
        acc_ylabel: Y-axis label for the accuracy plot.
        plot_filename: Base filename for saving plots.
        csv_filename: Filename for saving validation metrics CSV.
        experiment_title: Title for the overall experiment.
        cost_xlimit: Optional x-axis limit for cost/iteration plots.
        f1_title: Title for the F1 score plot.
        num_runs: Number of times to run each optimizer experiment for averaging.
        experiment_settings: Optional dictionary of initial experimental settings.

    Returns:
        tuple: Dictionaries of averaged validation losses, accuracies, F1 scores, AUCs, 
               iteration costs, and wall times for each optimizer.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    print_system_info()

    # Dictionaries to store validation metric histories (averaged over runs)
    val_losses, val_accs, val_f1s, val_aucs = {}, {}, {}, {}
    # Dictionaries to store training metric histories (averaged over runs)
    train_losses, train_accs, train_f1s, train_aucs = {}, {}, {}, {}
    iter_results, wall_results = {}, {}
    grad_norms_history = {}
    layer_names_dict = {}
    steps_per_epoch_dict = {}

    # Lists to store validation metric histories from ALL runs for error bars/stats
    all_runs_val_losses = {opt: [] for opt in optimizer_names}
    all_runs_val_accs = {opt: [] for opt in optimizer_names}
    all_runs_val_f1s = {opt: [] for opt in optimizer_names}
    all_runs_val_aucs = {opt: [] for opt in optimizer_names}
    all_runs_iter_costs = {opt: [] for opt in optimizer_names}
    all_runs_walltimes = {opt: [] for opt in optimizer_names}
    all_runs_gradients = {opt: [] for opt in optimizer_names}
    all_runs_steps_per_epoch = {opt: [] for opt in optimizer_names}

    all_runs_train_losses = {opt: [] for opt in optimizer_names}
    all_runs_train_accs = {opt: [] for opt in optimizer_names}
    all_runs_train_f1s = {opt: [] for opt in optimizer_names}
    all_runs_train_aucs = {opt: [] for opt in optimizer_names}

    # Dictionary to store layer runtime histories
    all_runs_layer_runtimes = {opt: [] for opt in optimizer_names}
    all_runs_layer_runtimes_back = {opt: [] for opt in optimizer_names}  # Backward runtimes
    # Dictionary to store component-level runtime histories (forward, backward, optimizer, overhead)
    all_runs_component_runtimes = {opt: [] for opt in optimizer_names}
    all_layer_names = None  # to hold consistent layer order

    all_runs_test_results = []

    # Optional iteration-level logging (populated only if train_experiment returns them)
    all_runs_iteration_logs = {opt: [] for opt in optimizer_names}
    all_runs_validation_walltimes = {opt: [] for opt in optimizer_names}
    all_runs_epoch_end_walltimes = {opt: [] for opt in optimizer_names}

    # Initialize resource tracking data AFTER warm-up
    resource_info_list = []
    experiment_start_time = time.time()  
    # Initialize settings collection
    collected_settings = {}
    if experiment_settings:
        collected_settings = experiment_settings.copy()

    for opt in optimizer_names:
        run_layer_names = None
        optimizer_resource_info = []

        # Execute training multiple times for the given optimizer
        for run_idx in range(num_runs):
            print(f"Starting {opt} - Run {run_idx+1}/{num_runs}")

            resource_tracker = None
            if resource_tracking_available:
                resource_tracker = ResourceTracker().start()

            # Initialize variables for the run to ensure they exist in all paths
            val_loss_hist, val_acc_hist, val_f1_hist, val_auc_hist = None, None, None, None
            # Dummy placeholders include layer_runtime_history and layer_names to prevent NameError
            iter_cost, walltime, grad_norms, layer_runtime_history, layer_bwd_runtime_history, layer_names, ln, test_metrics, steps_per_epoch, train_metrics_run = None, None, None, None, None, None, None, None, None, None
            run_specific_settings_value = None
            try:
                iter_cost, walltime, grad_norms, layer_runtime_history, layer_bwd_runtime_history, layer_names, ln, test_metrics, steps_per_epoch, train_metrics_run = None, None, None, None, None, None, None, None, None, None
                request_settings_now = False
                # Condition 1: First ever settings fetch (first optimizer, first run, no initial experiment_settings, no optimizer_params in collected_settings yet)
                is_first_ever_settings_fetch = (
                    run_idx == 0 and 
                    opt == optimizer_names[0] and 
                    experiment_settings is None and 
                    collected_settings.get('optimizer_params') is None
                )
                # Condition 2: Settings for the current optimizer are missing in collected_settings, and it's the first run for this optimizer.
                are_current_optimizer_settings_missing = (
                    opt not in collected_settings.get('optimizer_params', {}) and 
                    run_idx == 0
                )

                if is_first_ever_settings_fetch or are_current_optimizer_settings_missing:
                    request_settings_now = True
                
                results_to_unpack = None
                if request_settings_now:
                    out_data = train_experiment(opt, return_settings=True)
                    if isinstance(out_data, tuple) and len(out_data) >= 12:
                        # If last item is a settings dict, separate it
                        if isinstance(out_data[-1], dict) and (('model' in out_data[-1]) or ('optimizer_params' in out_data[-1])):
                            run_specific_settings_value = out_data[-1]
                            results_to_unpack = out_data[:-1]
                        else:
                            results_to_unpack = out_data
                            print(f"Warning: Requested settings for {opt}, but train_experiment did not return settings as last item.")
                    else:
                        raise ValueError(f"train_experiment for {opt} (settings requested) returned {len(out_data) if isinstance(out_data, tuple) else 'non-tuple'}. Expected >=12 items. Data: {out_data}")
                else:
                    out_data = train_experiment(opt, return_settings=False)
                    if not isinstance(out_data, tuple):
                        raise ValueError(f"train_experiment for {opt} (no settings requested) returned non-tuple. Data: {out_data}")
                    results_to_unpack = out_data

                # Unpack core results with backward compatibility (11, 14, or 17 core items)
                core = results_to_unpack
                core_len = len(core)
                if core_len < 11:
                    raise ValueError(f"Unexpected number of core results from train_experiment for {opt}: {core_len}. Expected at least 11.")

                val_loss_hist = core[0]
                val_acc_hist = core[1]
                val_f1_hist = core[2]
                val_auc_hist = core[3]
                iter_cost = core[4]
                walltime = core[5]
                grad_norms = core[6]
                pos = 7

                # Optional layer/component runtimes (present in 14+ item payloads)
                layer_runtime_history = None
                layer_bwd_runtime_history = None
                component_runtime_history = None
                if core_len >= 14:
                    layer_runtime_history = core[pos]; pos += 1
                    layer_bwd_runtime_history = core[pos]; pos += 1
                    component_runtime_history = core[pos]; pos += 1

                layer_names = core[pos]; pos += 1
                test_metrics = core[pos]; pos += 1
                steps_per_epoch = core[pos]; pos += 1
                train_metrics_run = core[pos]; pos += 1

                # Optional iteration-level logging triplet
                iteration_logs = None
                validation_walltimes = None
                epoch_end_walltimes = None
                remaining = core_len - pos
                if remaining >= 3:
                    iteration_logs = core[pos]; pos += 1
                    validation_walltimes = core[pos]; pos += 1
                    epoch_end_walltimes = core[pos]; pos += 1
                # Assign layer_names to ln and run_layer_names for downstream usage
                ln = layer_names
                run_layer_names = layer_names

                # Store settings if fetched
                if run_specific_settings_value:
                    # Update common settings if this is the first comprehensive fetch and no initial experiment_settings were provided
                    if is_first_ever_settings_fetch and experiment_settings is None:
                        for key, value in run_specific_settings_value.items():
                            if key != 'optimizer_params': 
                                collected_settings[key] = value
                        
                    # Store/update optimizer-specific parameters
                    if 'optimizer_params' in run_specific_settings_value:
                        if 'optimizer_params' not in collected_settings:
                            collected_settings['optimizer_params'] = {}
                        collected_settings['optimizer_params'][opt] = run_specific_settings_value['optimizer_params']
                    elif request_settings_now: 
                         print(f"Warning: Settings fetched for {opt} but 'optimizer_params' key was missing in the returned settings dict: {run_specific_settings_value}")


            except Exception as e:
                print(f"Fatal error during train_experiment call or unpacking for optimizer {opt}, run {run_idx+1}: {type(e).__name__} - {e}")
                raise 

            if resource_tracking_available and resource_tracker:
                resource_tracker.stop()
                run_info = resource_tracker.get_info()
                run_info.update({
                    "optimizer": opt,
                    "run_index": run_idx,
                    "epochs": epochs,
                    "experiment_title": experiment_title,
                })
                optimizer_resource_info.append(run_info)

            if val_loss_hist is None or test_metrics is None or steps_per_epoch is None:
                 raise ValueError(f"Experiment run for optimizer {opt} (Run {run_idx+1}) failed to return valid results (check steps_per_epoch).") # Include run_idx in error

            # Store history from this run (append backward layer timings too)
            if val_loss_hist is not None:
                all_runs_val_losses[opt].append(val_loss_hist)
                all_runs_val_accs[opt].append(val_acc_hist)
                all_runs_val_f1s[opt].append(val_f1_hist)
                all_runs_val_aucs[opt].append(val_auc_hist)
                all_runs_iter_costs[opt].append(iter_cost)
                all_runs_walltimes[opt].append(walltime)
                all_runs_gradients[opt].append(grad_norms)
                all_runs_layer_runtimes[opt].append(layer_runtime_history)
                # store backward timings
                all_runs_layer_runtimes_back[opt].append(layer_bwd_runtime_history)
                # store component histories
                all_runs_component_runtimes[opt].append(component_runtime_history)
                all_runs_steps_per_epoch[opt].append(steps_per_epoch)

                # Store training history
                all_runs_train_losses[opt].append(train_metrics_run['train_loss'])
                all_runs_train_accs[opt].append(train_metrics_run.get('train_accuracy', []))
                all_runs_train_f1s[opt].append(train_metrics_run.get('train_f1_score', []))
                all_runs_train_aucs[opt].append(train_metrics_run.get('train_auc', []))
            run_layer_names = ln 

            # Store final test metrics for this run
            test_result_row = {
                "optimizer": opt,
                "run": run_idx + 1,
                "test_loss": test_metrics.get('loss', np.nan),
                "test_accuracy": test_metrics.get('accuracy', np.nan),
                "test_f1_score": test_metrics.get('f1_score', np.nan),
                "test_auc": test_metrics.get('auc', np.nan),
                "test_eval_time_seconds": test_metrics.get('eval_time_seconds', np.nan)
            }
            all_runs_test_results.append(test_result_row)
            # Store optional iteration-level logging structures per run
            if 'iteration_logs' in locals() and iteration_logs is not None:
                all_runs_iteration_logs[opt].append(iteration_logs)
            if 'validation_walltimes' in locals() and validation_walltimes is not None:
                all_runs_validation_walltimes[opt].append(validation_walltimes)
            if 'epoch_end_walltimes' in locals() and epoch_end_walltimes is not None:
                all_runs_epoch_end_walltimes[opt].append(epoch_end_walltimes)
        # End of inner loop (runs per optimizer)
        # Capture layer names from first optimizer/run to use in plotting
        if all_layer_names is None and run_layer_names is not None:
            all_layer_names = run_layer_names

        # --- Averaging Validation Metrics Across Runs ---
        loss_stats = calculate_statistics(all_runs_val_losses[opt])
        acc_stats = calculate_statistics(all_runs_val_accs[opt])
        f1_stats = calculate_statistics(all_runs_val_f1s[opt])
        auc_stats = calculate_statistics(all_runs_val_aucs[opt]) 

        val_losses[opt] = loss_stats['mean']
        val_accs[opt] = acc_stats['mean']
        val_f1s[opt] = f1_stats['mean']
        val_aucs[opt] = auc_stats['mean']

        # --- Averaging Training Metrics Across Runs ---
        train_loss_stats = calculate_statistics(all_runs_train_losses[opt])
        train_acc_stats = calculate_statistics(all_runs_train_accs[opt])
        train_f1_stats = calculate_statistics(all_runs_train_f1s[opt])
        train_auc_stats = calculate_statistics(all_runs_train_aucs[opt])

        train_losses[opt] = train_loss_stats['mean']
        train_accs[opt] = train_acc_stats['mean']
        train_f1s[opt] = train_f1_stats['mean']
        train_aucs[opt] = train_auc_stats['mean']

        # Average iteration costs and walltimes
        avg_iter_costs = calculate_statistics(all_runs_iter_costs[opt])['mean']
        avg_wall = calculate_statistics(all_runs_walltimes[opt])['mean']

        # Check consistency and get average steps_per_epoch
        if not all(s == all_runs_steps_per_epoch[opt][0] for s in all_runs_steps_per_epoch[opt]):
            print(f"Warning: Inconsistent steps_per_epoch across runs for optimizer {opt}. Using average.")
            avg_steps_per_epoch = int(np.mean(all_runs_steps_per_epoch[opt]))
        else:
            avg_steps_per_epoch = all_runs_steps_per_epoch[opt][0]

        # Average gradient norms per layer
        avg_gradients = {}
        if all_runs_gradients[opt] and run_layer_names:
            for layer in run_layer_names:
                layer_runs_grads = [run_grads.get(layer, []) for run_grads in all_runs_gradients[opt]]
                # Ensure all lists have the same length (pad if necessary, though unlikely)
                min_len = min(len(g) for g in layer_runs_grads) if layer_runs_grads else 0
                if min_len > 0:
                    layer_runs_grads_trimmed = [g[:min_len] for g in layer_runs_grads]
                    avg_layer_grads = np.mean(np.array(layer_runs_grads_trimmed), axis=0)
                    avg_gradients[layer] = [float(x) for x in avg_layer_grads]
                else:
                    avg_gradients[layer] = []

        iter_results[opt] = avg_iter_costs
        wall_results[opt] = avg_wall
        grad_norms_history[opt] = avg_gradients
        layer_names_dict[opt] = run_layer_names
        steps_per_epoch_dict[opt] = avg_steps_per_epoch 

        resource_info_list.extend(optimizer_resource_info) 
        print(f"Averaged validation metrics logged for optimizer {opt} over {num_runs} runs")
    # End of outer loop (optimizers)

    # --- Prepare Error Bars for Validation Plots ---
    loss_std_err = {opt: calculate_statistics(all_runs_val_losses[opt])['std_err'] for opt in optimizer_names}
    acc_std_err = {opt: calculate_statistics(all_runs_val_accs[opt])['std_err'] for opt in optimizer_names}
    f1_std_err = {opt: calculate_statistics(all_runs_val_f1s[opt])['std_err'] for opt in optimizer_names}
    auc_std_err = {opt: calculate_statistics(all_runs_val_aucs[opt])['std_err'] for opt in optimizer_names} 

    # --- Prepare Error Bars for Training Metrics ---
    train_loss_std_err = {opt: calculate_statistics(all_runs_train_losses[opt])['std_err'] for opt in optimizer_names}
    train_acc_std_err = {opt: calculate_statistics(all_runs_train_accs[opt])['std_err'] for opt in optimizer_names}
    train_f1_std_err = {opt: calculate_statistics(all_runs_train_f1s[opt])['std_err'] for opt in optimizer_names}
    train_auc_std_err = {opt: calculate_statistics(all_runs_train_aucs[opt])['std_err'] for opt in optimizer_names}

    # --- Prepare Error Bars for Iteration/Walltime Plots ---
    iter_cost_std_err = {opt: calculate_statistics(all_runs_iter_costs[opt])['std_err'] for opt in optimizer_names}

    # --- Statistical Analysis on Final Validation Metrics ---
    final_val_loss_data = {opt: [run[-1] for run in all_runs_val_losses[opt]] for opt in optimizer_names}
    final_val_acc_data = {opt: [run[-1] for run in all_runs_val_accs[opt]] for opt in optimizer_names}
    final_val_f1_data = {opt: [run[-1] for run in all_runs_val_f1s[opt]] for opt in optimizer_names}
    final_val_auc_data = {opt: [run[-1] for run in all_runs_val_aucs[opt]] for opt in optimizer_names}

    loss_sig_tests = perform_statistical_tests(final_val_loss_data, 'final_validation_loss')
    acc_sig_tests = perform_statistical_tests(final_val_acc_data, 'final_validation_accuracy')
    f1_sig_tests = perform_statistical_tests(final_val_f1_data, 'final_validation_f1_score')
    auc_sig_tests = perform_statistical_tests(final_val_auc_data, 'final_validation_auc') 

    # Prepare final validation statistics
    final_val_loss_stats = {opt: calculate_statistics([[run[-1]] for run in all_runs_val_losses[opt]]) for opt in optimizer_names}
    final_val_acc_stats = {opt: calculate_statistics([[run[-1]] for run in all_runs_val_accs[opt]]) for opt in optimizer_names}
    final_val_f1_stats = {opt: calculate_statistics([[run[-1]] for run in all_runs_val_f1s[opt]]) for opt in optimizer_names}
    final_val_auc_stats = {opt: calculate_statistics([[run[-1]] for run in all_runs_val_aucs[opt]]) for opt in optimizer_names}

    # Create comprehensive statistics report (based on validation metrics)
    stats_data = {
        'validation_loss': {
            'n_runs': num_runs,
            'final_values': {opt: v['mean'][0] for opt, v in final_val_loss_stats.items()}, 
            'final_stats': final_val_loss_stats,
            'significance_tests': loss_sig_tests
        },
        'validation_accuracy': {
            'n_runs': num_runs,
            'final_values': {opt: v['mean'][0] for opt, v in final_val_acc_stats.items()},
            'final_stats': final_val_acc_stats,
            'significance_tests': acc_sig_tests
        },
        'validation_f1_score': {
            'n_runs': num_runs,
            'final_values': {opt: v['mean'][0] for opt, v in final_val_f1_stats.items()},
            'final_stats': final_val_f1_stats,
            'significance_tests': f1_sig_tests
        },
        'validation_auc': { 
            'n_runs': num_runs,
            'final_values': {opt: v['mean'][0] for opt, v in final_val_auc_stats.items()},
            'final_stats': final_val_auc_stats,
            'significance_tests': auc_sig_tests
        }
    }
    save_statistics_report(stats_data, results_dir, plot_filename)

    # Save experimental settings
    if collected_settings:
        common_settings = {
            "experiment_title": experiment_title,
            "epochs": epochs,
            "num_runs": num_runs,
        }
        for opt in collected_settings:
            if isinstance(collected_settings[opt], dict):
                collected_settings[opt].update(common_settings)
            else:
                # If settings for an optimizer weren't a dict, initialize it
                collected_settings[opt] = common_settings.copy()
        save_experimental_settings(collected_settings, results_dir, f"experimental_settings_{plot_filename}")

    # --- Plotting Validation Metrics ---
    # Extract base experiment name (remove ' (Validation)' if present)
    base_experiment_name = experiment_title.replace(" (Validation)", "")
    # Use plot_filename as the experiment identifier part of the filename
    experiment_file_id = plot_filename.replace("_validation_curves", "")

    # Plot Validation Loss vs Epoch
    plot_seaborn_style_with_error_bars(
        val_losses,
        loss_std_err,
        range(1, epochs+1),
        f"Validation Loss for {base_experiment_name}",
        f"val_loss_{experiment_file_id}",
        "Loss",
        visuals_dir,
        xlabel="Epoch"
    )

    # Plot Validation Accuracy vs Epoch
    plot_seaborn_style_with_error_bars(
        val_accs,
        acc_std_err,
        range(1, epochs+1),
        f"Validation Accuracy for {base_experiment_name}",
        f"val_accuracy_{experiment_file_id}",
        "Accuracy (%)",
        visuals_dir,
        xlabel="Epoch"
    )

    # Plot Validation F1 Score vs Epoch
    plot_seaborn_style_with_error_bars(
        val_f1s,
        f1_std_err,
        range(1, epochs+1),
        f"Validation F1 Score for {base_experiment_name}",
        f"val_f1_score_{experiment_file_id}",
        "F1 Score",
        visuals_dir,
        xlabel="Epoch"
    )

    # Plot Validation AUC vs Epoch
    plot_seaborn_style_with_error_bars(
        val_aucs,
        auc_std_err,
        range(1, epochs+1),
        f"Validation AUC for {base_experiment_name}",
        f"val_auc_{experiment_file_id}",
        "AUC Score",
        visuals_dir,
        xlabel="Epoch"
    )

    # --- Plotting Training Metrics ---
    # Plot Training Loss vs Epoch
    plot_seaborn_style_with_error_bars(
        train_losses,
        train_loss_std_err,
        range(1, epochs + 1),
        f"Training Loss for {base_experiment_name}",
        f"train_loss_{experiment_file_id}",
        "Average Training Loss",
        visuals_dir,
        xlabel="Epoch"
    )

    # Plot Training Accuracy vs Epoch
    plot_seaborn_style_with_error_bars(
        train_accs,
        train_acc_std_err,
        range(1, epochs + 1),
        f"Training Accuracy for {base_experiment_name}",
        f"train_accuracy_{experiment_file_id}",
        "Training Accuracy (%)",
        visuals_dir,
        xlabel="Epoch"
    )

    # Plot Training F1 Score vs Epoch
    plot_seaborn_style_with_error_bars(
        train_f1s,
        train_f1_std_err,
        range(1, epochs + 1),
        f"Training F1 Score for {base_experiment_name}",
        f"train_f1_score_{experiment_file_id}",
        "Training F1 Score",
        visuals_dir,
        xlabel="Epoch"
    )

    # Plot Training AUC vs Epoch
    plot_seaborn_style_with_error_bars(
        train_aucs,
        train_auc_std_err,
        range(1, epochs + 1),
        f"Training AUC for {base_experiment_name}",
        f"train_auc_{experiment_file_id}",
        "Training AUC Score",
        visuals_dir,
        xlabel="Epoch"
    )

    # --- Plotting Iteration/Step-based and Walltime-based Training Loss ---
    iter_x_values = {}
    max_total_steps = 0 
    for opt in optimizer_names:
        if opt in iter_results and iter_results[opt] is not None:
            num_steps = len(iter_results[opt])
            iter_x_values[opt] = list(range(1, num_steps + 1))
            if num_steps > max_total_steps:
                max_total_steps = num_steps
        else:
            iter_x_values[opt] = []

    # Plot Training Loss vs. Iteration (Steps)
    plot_seaborn_style_with_error_bars(
        iter_results,  # y-values (dict: opt_name -> list of mean losses per iteration)
        iter_cost_std_err, # error for y-values (dict: opt_name -> list of std_err for losses)
        iter_x_values, # x-values (dict: opt_name -> list of iteration/step numbers)
        f"Training Loss vs. Iteration for {base_experiment_name}",
        f"train_loss_iteration_{experiment_file_id}",
        "Loss (Training)",
        visuals_dir,
        xlabel="Iteration (Step)",
        xlimit=cost_xlimit if cost_xlimit is not None else max_total_steps, #
        yscale='log' 
    )

    # Plot Training Loss vs. Wall-clock Time
    # wall_results is a dict: opt_name -> list of mean cumulative walltimes per iteration
    # iter_results is a dict: opt_name -> list of mean losses per iteration
    plot_seaborn_style_with_error_bars(
        iter_results,  # y-values (dict: opt_name -> list of mean losses per iteration)
        iter_cost_std_err, # error for y-values (dict: opt_name -> list of std_err for losses)
        wall_results,  # x-values (dict: opt_name -> list of mean cumulative walltimes)
        f"Training Loss vs. Wall-clock Time for {base_experiment_name}",
        f"train_loss_walltime_{experiment_file_id}",
        "Loss (Training)", 
        visuals_dir,
        xlabel="Time (s)",
        # xlimit for walltime plots is handled by the plotting function if x_values is a dict (it plots all available data)
        yscale='log'
    )

    # --- Save Raw Validation Run Data ---
    raw_runs_data = {
        'val_losses': {opt: [list(run) for run in all_runs_val_losses[opt]] for opt in optimizer_names},
        'val_accuracies': {opt: [list(run) for run in all_runs_val_accs[opt]] for opt in optimizer_names},
        'val_f1_scores': {opt: [list(run) for run in all_runs_val_f1s[opt]] for opt in optimizer_names},
        'val_aucs': {opt: [list(run) for run in all_runs_val_aucs[opt]] for opt in optimizer_names}
    }
    with open(os.path.join(results_dir, f"raw_runs_validation_data_{plot_filename}.json"), 'w') as f:
        json.dump(raw_runs_data, f, default=lambda x: list(x) if isinstance(x, np.ndarray) else str(x))

    # --- Save Training Metrics CSV (Averaged) ---
    df_train_metrics_with_err = []
    for opt in optimizer_names:
        num_epochs_train = len(train_losses.get(opt, []))
        for i in range(num_epochs_train):
            df_train_metrics_with_err.append({
                "optimizer": opt,
                "epoch": i + 1,
                "train_loss": train_losses[opt][i] if i < len(train_losses.get(opt, [])) else np.nan,
                "train_accuracy": train_accs[opt][i] if i < len(train_accs.get(opt, [])) else np.nan,
                "train_f1_score": train_f1s[opt][i] if i < len(train_f1s.get(opt, [])) else np.nan,
                "train_auc": train_aucs[opt][i] if i < len(train_aucs.get(opt, [])) else np.nan,
                "train_loss_std_err": train_loss_std_err[opt][i] if i < len(train_loss_std_err.get(opt, [])) else np.nan,
                "train_accuracy_std_err": train_acc_std_err[opt][i] if i < len(train_acc_std_err.get(opt, [])) else np.nan,
                "train_f1_score_std_err": train_f1_std_err[opt][i] if i < len(train_f1_std_err.get(opt, [])) else np.nan,
                "train_auc_std_err": train_auc_std_err[opt][i] if i < len(train_auc_std_err.get(opt, [])) else np.nan
            })
    train_csv_filename = os.path.join(results_dir, f"{experiment_file_id}_training_metrics.csv")
    pd.DataFrame(df_train_metrics_with_err).to_csv(train_csv_filename, index=False)
    print(f"Averaged training metrics saved to {train_csv_filename}")

    # --- Save Resource Information ---
    resource_df = pd.DataFrame()
    if resource_tracking_available and resource_info_list:
        total_duration = time.time() - experiment_start_time
        for info in resource_info_list:
            info["total_experiment_duration"] = round(total_duration, 2)
        resource_csv_path = os.path.join(results_dir, f"compute_resources_{experiment_file_id}.csv")
        resource_df = save_resource_info(resource_info_list, resource_csv_path)

    # --- Plot Resource Usage ---
    if resource_tracking_available and resource_df is not None and not resource_df.empty:
        plot_resource_usage(resource_df, visuals_dir, experiment_file_id, base_experiment_name)
    elif resource_tracking_available and resource_df is None:
         print("Warning: resource_df was None, skipping resource usage plotting.")

    # --- Plot Layer-wise Runtime Breakdown ---
    if all_layer_names is not None:
        plot_layer_runtime_breakdown(
            all_runs_layer_runtimes,
            all_layer_names,
            visuals_dir,
            f"Layer-wise Runtime Breakdown for {base_experiment_name}"
        )
        # Plot forward vs backward breakdown
        plot_forward_backward_runtime_breakdown(
            all_runs_layer_runtimes,
            all_runs_layer_runtimes_back,
            all_layer_names,
            visuals_dir,
            f"Forward vs Backward Runtime Breakdown for {base_experiment_name}"
        )

    # --- Plot Accuracy/Runtime and Loss/Runtime Ratios ---
    try:
        # Compute avg runtime per optimizer from wall_results
        avg_runtime = {opt: (wall_results.get(opt, [])[-1] if wall_results.get(opt) else np.nan) for opt in optimizer_names}
        # Compute final val accuracy and loss
        final_acc = {opt: val_accs.get(opt, [np.nan])[-1] for opt in optimizer_names}
        final_loss = {opt: val_losses.get(opt, [np.nan])[-1] for opt in optimizer_names}
        # Compute ratios
        acc_runtime_ratio = {opt: final_acc[opt] / avg_runtime[opt] if avg_runtime[opt] else np.nan for opt in optimizer_names}
        loss_runtime_ratio = {opt: final_loss[opt] / avg_runtime[opt] if avg_runtime[opt] else np.nan for opt in optimizer_names}
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.bar(acc_runtime_ratio.keys(), acc_runtime_ratio.values())
        ax1.set_title(f"Accuracy/Runtime Ratio for {base_experiment_name}")
        ax1.set_ylabel('Acc (%) per sec')
        ax1.set_xticklabels(list(acc_runtime_ratio.keys()), rotation=45)
        ax2.bar(loss_runtime_ratio.keys(), loss_runtime_ratio.values())
        ax2.set_title(f"Loss/Runtime Ratio for {base_experiment_name}")
        ax2.set_ylabel('Loss per sec')
        ax2.set_xticklabels(list(loss_runtime_ratio.keys()), rotation=45)
        plt.tight_layout()
        ratio_plot_path = os.path.join(visuals_dir, f"accuracy_loss_runtime_ratio_{experiment_file_id}.png")
        fig.savefig(ratio_plot_path)
        plt.close(fig)
        print(f"Saved accuracy/loss-runtime ratio plot to {ratio_plot_path}")
    except Exception as e:
        print(f"Warning: Failed to plot accuracy/loss runtime ratios: {e}")

    # --- Save Raw Layer Runtime Histories ---
    try:
        raw_runtime_data = {}
        for opt in optimizer_names:
            raw_runtime_data[opt] = {
                'forward': all_runs_layer_runtimes.get(opt, []),
                'backward': all_runs_layer_runtimes_back.get(opt, [])
            }
        json_path = os.path.join(results_dir, f"{experiment_file_id}_layer_runtimes.json")
        with open(json_path, 'w') as jf:
            json.dump(raw_runtime_data, jf, indent=4)
        print(f"Saved raw layer runtime histories to {json_path}")
    except Exception as e:
        print(f"Warning: Failed to save raw layer runtime histories: {e}")
    # --- Plot Component-level Runtimes per Epoch ---
    try:
        comp_names = ['forward', 'backward', 'optimizer', 'overhead']
        # Prepare data structures
        for comp in comp_names:
            comp_mean = {}
            comp_err = {}
            for opt in optimizer_names:
                # collect list of per-run arrays
                runs_comp = [run[comp] for run in all_runs_component_runtimes.get(opt, [])]
                stats = calculate_statistics(runs_comp)
                comp_mean[opt] = stats['mean'] if stats else []
                comp_err[opt] = stats['std_err'] if stats else []
            # Plot
            plot_seaborn_style_with_error_bars(
                comp_mean,
                comp_err,
                range(1, epochs+1),
                f"{comp.title()} Runtime per Epoch for {base_experiment_name}",
                f"{comp}_runtime_epoch_{experiment_file_id}",
                f"{comp.title()} Time (s)",
                visuals_dir,
                xlabel="Epoch"
            )
    except Exception as e:
        print(f"Warning: Failed to plot component-level runtimes: {e}")

    # --- Save Iteration-level Logs (if available) ---
    try:
        # Only save if any optimizer has data
        has_iteration_logs = any(all_runs_iteration_logs.get(opt) for opt in optimizer_names)
        if has_iteration_logs:
            iter_log_path = os.path.join(results_dir, f"{experiment_file_id}_iteration_logs.json")
            payload = {
                'iteration_logs': all_runs_iteration_logs,
                'validation_walltimes': all_runs_validation_walltimes,
                'epoch_end_walltimes': all_runs_epoch_end_walltimes,
            }
            with open(iter_log_path, 'w') as jf:
                json.dump(payload, jf, default=lambda x: list(x) if isinstance(x, np.ndarray) else x)
            print(f"Saved iteration-level logs to {iter_log_path}")
    except Exception as e:
        print(f"Warning: Failed to save iteration-level logs: {e}")

    # --- Save Final Test Results (per-run) ---
    try:
        if all_runs_test_results:
            test_df = pd.DataFrame(all_runs_test_results)
            test_csv_path = os.path.join(results_dir, f"{experiment_file_id}_final_test_results.csv")
            test_df.to_csv(test_csv_path, index=False)
            print(f"Saved final test results to {test_csv_path}")
    except Exception as e:
        print(f"Warning: Failed to save final test results CSV: {e}")

    # Return averaged validation metrics history (optional, maybe not needed by caller)
    return val_losses, val_accs, val_f1s, val_aucs, iter_results, wall_results
