import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import time
import argparse
from typing import Dict, List, Any, Tuple
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator
from milo import milo
from novograd import NovoGrad
# Updated import: Use plotting.py instead of visualize.py
from utils.plotting import (
    plot_seaborn_style_with_error_bars,
    plot_gradient_norms_layerwise,
    plot_gradient_imbalance_ratio,
    plot_average_gradient_norm
)
# Assuming GradientTracker is defined elsewhere, e.g., in utils or a dedicated file
# from utils.gradient_tracker import GradientTracker # Example import
# Placeholder GradientTracker if not available
class GradientTracker:
    def __init__(self):
        self.gradients = {}
        self.layer_names = None
        self.history = {} # To store norms per eval step

    def update(self, model):
        if self.layer_names is None:
            self.layer_names = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is not None]

        current_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = param.grad.norm().item()
                current_grads[name] = norm
        # Store latest gradients for potential immediate use if needed
        self.gradients = current_grads

    def track_at_eval(self, eval_step):
        """Stores the currently tracked gradients associated with an evaluation step."""
        # For simplicity, store the last computed norms.
        # A more robust approach might average norms since the last eval step.
        if self.gradients: # Only store if gradients have been computed
             self.history[eval_step] = self.gradients.copy()


    def get_data(self) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
        """Returns the history of gradient norms per evaluation step and layer names."""
        return self.history, self.layer_names

    def reset(self):
        """Resets the tracker for a new run."""
        self.gradients = {}
        self.layer_names = None
        self.history = {}


def calculate_statistics(data_list: List[List[float]]) -> Dict[str, List[float]]:
    """
    Calculates mean and standard error across runs for lists of potentially different lengths.
    Pads shorter lists with the value of their last element.
    """
    if not data_list:
        return {'mean': [], 'std_err': []}

    # Find the maximum length
    max_len = max(len(run) for run in data_list if run) if any(data_list) else 0
    if max_len == 0:
         return {'mean': [], 'std_err': []}


    # Pad shorter lists
    padded_data = []
    for run in data_list:
        if not run: # Handle empty runs
             padded_data.append([np.nan] * max_len)
             continue
        last_val = run[-1] if run else np.nan
        padded_run = run + [last_val] * (max_len - len(run))
        padded_data.append(padded_run)

    data_array = np.array(padded_data)

    # Calculate mean and standard error, ignoring NaNs
    mean_values = np.nanmean(data_array, axis=0).tolist()
    std_dev_values = np.nanstd(data_array, axis=0)
    # Calculate standard error, handle cases with only one run (std_err=0)
    n_runs = np.sum(~np.isnan(data_array), axis=0) # Count non-NaNs per step
    std_err_values = np.divide(std_dev_values, np.sqrt(n_runs), out=np.zeros_like(std_dev_values, dtype=float), where=n_runs>0).tolist()


    return {'mean': mean_values, 'std_err': std_err_values}


def train_with_optimizer(optimizer_name: str, run_index: int, num_runs: int) -> Dict[str, Any]:
    """
    Train a transformer model using the specified optimizer for a single run.

    Args:
        optimizer_name: Name of the optimizer to use.
        run_index: The index of the current run (0-based).
        num_runs: Total number of runs.

    Returns:
        Dictionary containing raw training metrics for this run.
    """
    print(f"\n{'='*40}\nTraining {optimizer_name} - Run {run_index + 1}/{num_runs}\n{'='*40}")

    # --- Seed for reproducibility across runs (optional but recommended) ---
    # seed = config.get('seed', 42) + run_index # Vary seed per run
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

    # Initialize the model
    model = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size'],
        N_BLOCKS=config['n_blocks']
    ).to(config['device'])

    # Print model parameters only for the first run
    if run_index == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters in the model: {total_params:,}")

    # Initialize metrics tracking for this run
    metrics_data = {
        'train_losses': [], # Losses from estimate_loss (eval mode)
        'dev_losses': [],   # Losses from estimate_loss (eval mode)
        'eval_steps': [],   # Steps at which evaluation was performed
        'wall_times': [],   # Wall time recorded at each iteration
        'iteration_losses': [] # Loss recorded at each training iteration
    }

    # Initialize gradient tracker for this run
    gradient_tracker = GradientTracker()

    # Setup optimizer
    optimizer_params = config['optimizer_params'][optimizer_name]

    if optimizer_name == "MILO" or optimizer_name == "MILO_LW":
        optimizer = milo(model.parameters(), **optimizer_params)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == "ADAGRAD": # Add Adagrad condition
        optimizer = torch.optim.Adagrad(model.parameters(), **optimizer_params)
    elif optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name == "ADAMW":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    elif optimizer_name == "NOVOGRAD":
        optimizer = NovoGrad(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if run_index == 0:
        print(f"Using optimizer: {optimizer_name} with parameters: {optimizer_params}")

    # Training tracking
    losses = [] # Stores recent iteration losses for progress bar
    AVG_WINDOW = 64

    # Helper function to estimate loss
    @torch.no_grad()
    def estimate_loss(steps: int):
        out = {}
        model.eval()

        for split in ['train', 'dev']:
            data_path = config['train_path'] if split == 'train' else config['dev_path']
            # Use a fresh iterator for each evaluation
            try:
                batch_iterator_eval = get_batch_iterator(
                    data_path, config['t_batch_size'], config['t_context_length'], device=config['device']
                )
                losses_eval = torch.zeros(steps)
                actual_steps = 0
                for k in range(steps):
                    try:
                        xb, yb = next(batch_iterator_eval)
                        _, loss = model(xb, yb)
                        losses_eval[k] = loss.item()
                        actual_steps += 1
                    except StopIteration:
                        print(f"Warning: Eval iterator for {split} ended early at step {k}.")
                        break
                if actual_steps > 0:
                    out[split] = losses_eval[:actual_steps].mean().item() # Use .item()
                else:
                    out[split] = np.nan # Indicate if no eval steps completed
            except Exception as e:
                 print(f"Error during evaluation for {split}: {e}")
                 out[split] = np.nan


        model.train()
        return out

    # Training loop
    batch_iterator = get_batch_iterator(
        config['train_path'],
        config['t_batch_size'],
        config['t_context_length'],
        device=config['device']
    )

    # Track start time for walltime tracking
    start_time = time.time()

    # Training loop with progress bar
    pbar = tqdm(range(config['t_train_steps']), desc=f"{optimizer_name} Run {run_index+1}", leave=False)
    for step in pbar:
        try:
            # Get batch
            xb, yb = next(batch_iterator)

            # Forward pass
            _, loss = model(xb, yb)

            # Record metrics
            current_loss = loss.item()
            losses.append(current_loss)
            metrics_data['iteration_losses'].append(current_loss)
            metrics_data['wall_times'].append(time.time() - start_time)

            pbar.set_postfix({"loss": f"{np.mean(losses[-AVG_WINDOW:]):.4f}"})

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Track gradients (before optimizer step)
            gradient_tracker.update(model)

            optimizer.step()

            # Evaluation
            if step % config['t_eval_steps'] == 0 or step == config['t_train_steps'] - 1:
                evaluation_losses = estimate_loss(config['t_eval_iters'])
                train_loss = evaluation_losses.get('train', np.nan)
                dev_loss = evaluation_losses.get('dev', np.nan)

                # Only print for the first run to avoid clutter
                if run_index == 0 and not np.isnan(train_loss) and not np.isnan(dev_loss):
                     print(f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")

                metrics_data['train_losses'].append(train_loss)
                metrics_data['dev_losses'].append(dev_loss)
                metrics_data['eval_steps'].append(step)
                # Track gradients at evaluation points
                gradient_tracker.track_at_eval(step)


            # Learning rate decay
            if config.get('t_lr_decay_step') is not None and step == config['t_lr_decay_step']:
                if run_index == 0:
                    print('Decaying learning rate')
                new_lr = config.get('t_lr_decayed', optimizer_params.get('lr', 0.01) / 10) # Default decay if not specified
                for g in optimizer.param_groups:
                    g['lr'] = new_lr

        except StopIteration:
            print(f"Warning: Training data iterator finished early at step {step}.")
            break
        except Exception as e:
             print(f"Error during training step {step} for {optimizer_name} Run {run_index+1}: {e}")
             # Optionally break or continue depending on desired robustness
             break


    # Get final gradient norm data for this run
    gradient_norms_history, layer_names = gradient_tracker.get_data()

    # --- Model Saving (Optional - Save last run's model or best model) ---
    if run_index == num_runs - 1: # Save only the model from the last run
        optimizer_out_path = config['t_out_path'].replace('.pt', f'_{optimizer_name}.pt')
        os.makedirs(os.path.dirname(optimizer_out_path), exist_ok=True)
        try:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # Include final metrics if needed, though they are also returned
                    'final_train_loss': metrics_data['train_losses'][-1] if metrics_data['train_losses'] else np.nan,
                    'final_dev_loss': metrics_data['dev_losses'][-1] if metrics_data['dev_losses'] else np.nan,
                },
                optimizer_out_path
            )
            print(f"Saved final model for {optimizer_name} to {optimizer_out_path}")
        except Exception as e:
            print(f"Error saving model for {optimizer_name}: {e}")


    # Return raw metrics for this run
    return {
        'optimizer_name': optimizer_name,
        'run_index': run_index,
        'train_losses': metrics_data['train_losses'],
        'dev_losses': metrics_data['dev_losses'],
        'eval_steps': metrics_data['eval_steps'],
        'iteration_losses': metrics_data['iteration_losses'],
        'wall_times': metrics_data['wall_times'],
        'gradient_norms_history': gradient_norms_history, # Dict: {eval_step: {layer: norm}}
        'layer_names': layer_names,
    }

def create_comparative_visualizations(
    all_runs_results: Dict[str, List[Dict[str, Any]]],
    num_runs: int,
    visuals_dir: str,
    experiment_title: str = "Optimizer Comparison" # Keep default or allow override
) -> None:
    """
    Create comparative visualizations with error bars for all trained optimizers based on multiple runs.

    Args:
        all_runs_results: Dictionary mapping optimizer names to a list of their training result dicts (one dict per run).
        num_runs: The number of runs performed for each optimizer.
        visuals_dir: Directory to save visualizations.
        experiment_title: Base title for the plots.
    """
    os.makedirs(visuals_dir, exist_ok=True)
    optimizer_names = list(all_runs_results.keys())

    # --- Process data for plotting ---
    agg_train_losses, agg_dev_losses, agg_iteration_losses, agg_wall_times = {}, {}, {}, {}
    all_runs_gradients = {opt: [] for opt in optimizer_names} # List of gradient histories per run
    consistent_layer_names = None
    eval_steps = None # Assume eval_steps are consistent across runs and optimizers for simplicity

    for opt, runs_data in all_runs_results.items():
        if not runs_data: continue # Skip if an optimizer had no successful runs

        # Collect metrics from each run
        runs_train_losses = [run.get('train_losses', []) for run in runs_data]
        runs_dev_losses = [run.get('dev_losses', []) for run in runs_data]
        runs_iter_losses = [run.get('iteration_losses', []) for run in runs_data]
        runs_wall_times = [run.get('wall_times', []) for run in runs_data]
        runs_gradients = [run.get('gradient_norms_history', {}) for run in runs_data] # List of dicts {step: {layer: norm}}

        # Store gradient history for all runs
        all_runs_gradients[opt] = runs_gradients

        # Calculate mean and std_err for plot data
        agg_train_losses[opt] = calculate_statistics(runs_train_losses)
        agg_dev_losses[opt] = calculate_statistics(runs_dev_losses)
        agg_iteration_losses[opt] = calculate_statistics(runs_iter_losses)
        # For wall time, we plot loss vs time. We need mean/stderr for loss (Y) and mean time (X)
        # Calculate mean wall times corresponding to iteration losses
        mean_wall_times = calculate_statistics(runs_wall_times)['mean']
        agg_wall_times[opt] = {'mean_time': mean_wall_times, 'loss_stats': calculate_statistics(runs_iter_losses)}


        # Get eval_steps and layer_names (assume consistency)
        if eval_steps is None and runs_data[0].get('eval_steps'):
            eval_steps = runs_data[0]['eval_steps']
        if consistent_layer_names is None and runs_data[0].get('layer_names'):
            consistent_layer_names = runs_data[0]['layer_names']

    # --- Plotting ---
    # Use experiment_title for base, but allow specific titles for plots
    plot_filename_base = experiment_title.lower().replace(" ", "_")
    num_eval_steps = len(eval_steps) if eval_steps else 0

    # 1. Training loss comparison (vs. Evaluation Step)
    if any(d['mean'] for d in agg_train_losses.values()):
        train_loss_means = {opt: d['mean'] for opt, d in agg_train_losses.items()}
        train_loss_stderr = {opt: d['std_err'] for opt, d in agg_train_losses.items()}
        plot_seaborn_style_with_error_bars(
            data=train_loss_means,
            data_std_err=train_loss_stderr,
            x_values=eval_steps if eval_steps else range(len(next(iter(train_loss_means.values()), []))), # Use eval_steps if available
            title=f"{experiment_title}: Training Loss (Eval)",
            filename=f"{plot_filename_base}_train_loss",
            y_label="Average Loss",
            visuals_dir=visuals_dir,
            xlabel="Evaluation Step"
        )
    else:
        print("Warning: No valid training loss data available for plotting.")

    # 2. Validation loss comparison (vs. Evaluation Step)
    if any(d['mean'] for d in agg_dev_losses.values()):
        dev_loss_means = {opt: d['mean'] for opt, d in agg_dev_losses.items()}
        dev_loss_stderr = {opt: d['std_err'] for opt, d in agg_dev_losses.items()}
        plot_seaborn_style_with_error_bars(
            data=dev_loss_means,
            data_std_err=dev_loss_stderr,
            x_values=eval_steps if eval_steps else range(len(next(iter(dev_loss_means.values()), []))),
            title=f"{experiment_title}: Validation Loss",
            filename=f"{plot_filename_base}_validation_loss",
            y_label="Average Loss",
            visuals_dir=visuals_dir,
            xlabel="Evaluation Step"
        )
    else:
        print("Warning: No valid validation loss data available for plotting.")

    # 3. Iteration loss comparison (vs. Iteration/Steps)
    if any(d['mean'] for d in agg_iteration_losses.values()):
        iter_loss_means = {opt: d['mean'] for opt, d in agg_iteration_losses.items()}
        iter_loss_stderr = {opt: d['std_err'] for opt, d in agg_iteration_losses.items()}
        max_iters = max(len(m) for m in iter_loss_means.values() if m) if any(iter_loss_means.values()) else 0

        if max_iters > 0:
             plot_seaborn_style_with_error_bars(
                 data=iter_loss_means,
                 data_std_err=iter_loss_stderr,
                 x_values=range(1, max_iters + 1),
                 # Use specific title provided by user
                 title="Training Loss vs Steps for LLM Experiment",
                 # Update filename to reflect specific plot
                 filename=f"{plot_filename_base}_loss_vs_steps",
                 y_label="Loss",
                 visuals_dir=visuals_dir,
                 xlabel="Iteration" # Keep xlabel as Iteration or change to Steps if preferred
                 # yscale='log' # Optional
             )
        else:
             print("Warning: No iteration loss data to plot vs steps.")
    else:
        print("Warning: No valid iteration loss data available for plotting vs steps.")

    # 4. Wall time vs loss comparison
    if any(d['mean_time'] for d in agg_wall_times.values()):
        wall_loss_means = {opt: d['loss_stats']['mean'] for opt, d in agg_wall_times.items()}
        wall_loss_stderr = {opt: d['loss_stats']['std_err'] for opt, d in agg_wall_times.items()}
        wall_mean_times = {opt: d['mean_time'] for opt, d in agg_wall_times.items()}

        # Prepare data for plotting, ensuring lengths match based on available data points
        walltime_plot_data = {}
        walltime_plot_err = {}
        walltime_plot_x = {}
        for opt in optimizer_names:
            # Check if data exists for this optimizer
            if opt in wall_loss_means and opt in wall_mean_times:
                mean_loss = wall_loss_means[opt]
                mean_time = wall_mean_times[opt]
                std_err = wall_loss_stderr.get(opt, []) # Get std_err, default to empty list

                # Determine the minimum length based on available data
                min_len = min(len(mean_loss), len(mean_time), len(std_err) if std_err else len(mean_loss))

                if min_len > 0:
                    walltime_plot_data[opt] = mean_loss[:min_len]
                    walltime_plot_x[opt] = mean_time[:min_len]
                    # Only include error if it was available and had sufficient length
                    if opt in wall_loss_stderr and len(wall_loss_stderr[opt]) >= min_len:
                        walltime_plot_err[opt] = std_err[:min_len]
                    else:
                        # If error data is missing or too short, don't pass it for this optimizer
                        # The plotting function should handle missing keys in data_std_err
                        pass # walltime_plot_err[opt] will not be set
                else:
                    print(f"Warning: Insufficient data points for optimizer {opt} in walltime plot.")
            else:
                print(f"Warning: Missing loss or time data for optimizer {opt} in walltime plot.")

        # Check if there's any data prepared for the walltime plot
        if walltime_plot_data:
            plot_seaborn_style_with_error_bars(
                data=walltime_plot_data,
                data_std_err=walltime_plot_err, # Pass potentially incomplete dict
                x_values=walltime_plot_x, # Pass dict of x-values (times)
                # Use specific title provided by user
                title="Training Loss vs Walltime for LLM Experiment",
                # Update filename to reflect specific plot
                filename=f"{plot_filename_base}_loss_vs_walltime",
                y_label="Loss",
                visuals_dir=visuals_dir,
                xlabel="Time (s)" # Set xlabel to Time (s)
            )
        else:
             print("Warning: No data prepared for walltime vs loss plot.")

    else:
        print("Warning: No valid wall time data available for plotting.")

    # 5. Gradient norm visualizations
    # Requires processing all_runs_gradients into the format expected by plotting functions:
    # List[Dict[str, Dict[str, List[float]]]] -> List over runs[ Dict Optimizer -> Dict Layer -> List Norms per epoch/eval_step ]
    if consistent_layer_names and eval_steps and any(all_runs_gradients.values()):
        processed_gradient_data = [] # List over runs
        valid_runs_count = 0
        for i in range(num_runs):
            run_data = {} # Dict Optimizer -> Dict Layer -> List Norms
            valid_run = False
            for opt in optimizer_names:
                 # Check if this run exists for this optimizer
                 if i < len(all_runs_gradients.get(opt, [])):
                     opt_run_grads_history = all_runs_gradients[opt][i] # Dict: {eval_step: {layer: norm}}
                     if opt_run_grads_history: # Check if history is not empty
                         opt_data = {layer: [] for layer in consistent_layer_names}
                         for step in eval_steps:
                             step_grads = opt_run_grads_history.get(step, {})
                             for layer in consistent_layer_names:
                                 opt_data[layer].append(step_grads.get(layer, np.nan)) # Use NaN for missing data points
                         run_data[opt] = opt_data
                         valid_run = True # Mark run as valid if at least one optimizer has data
            if valid_run:
                 processed_gradient_data.append(run_data)
                 valid_runs_count += 1


        if valid_runs_count > 0:
            print(f"Plotting gradient norms based on {valid_runs_count} valid runs.")
            try:
                plot_gradient_norms_layerwise(
                    all_gradient_norms=processed_gradient_data,
                    optimizer_names=optimizer_names,
                    layer_names=consistent_layer_names,
                    visuals_dir=visuals_dir,
                    plot_filename=plot_filename_base,
                    experiment_title=experiment_title,
                    epochs=num_eval_steps # Use number of eval steps as 'epochs'
                )
            except Exception as e:
                print(f"Error plotting layer-wise gradient norms: {e}")

            try:
                plot_gradient_imbalance_ratio(
                    all_gradient_norms=processed_gradient_data,
                    optimizer_names=optimizer_names,
                    layer_names=consistent_layer_names,
                    visuals_dir=visuals_dir,
                    plot_filename=plot_filename_base,
                    experiment_title=experiment_title,
                    epochs=num_eval_steps
                )
            except Exception as e:
                print(f"Error plotting gradient imbalance ratio: {e}")

            try:
                plot_average_gradient_norm(
                    all_gradient_norms=processed_gradient_data,
                    optimizer_names=optimizer_names,
                    layer_names=consistent_layer_names,
                    visuals_dir=visuals_dir,
                    plot_filename=plot_filename_base,
                    experiment_title=experiment_title,
                    epochs=num_eval_steps
                )
            except Exception as e:
                print(f"Error plotting average gradient norm: {e}")
        else:
             print("Warning: No valid gradient norm data collected across runs for plotting.")

    else:
        print("Warning: Skipping gradient norm plots due to missing layer names, eval steps, or gradient data.")


    # 6. Save metrics to CSV
    try:
        results_dir = os.path.join(os.path.dirname(visuals_dir), "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_filename = os.path.join(results_dir, f"{plot_filename_base}_metrics.csv")

        df_metrics_list = []
        for opt in optimizer_names:
            # Use lengths of mean values, assuming they are consistent after padding in calculate_statistics
            num_eval_points = len(agg_dev_losses.get(opt, {}).get('mean', []))
            num_iter_points = len(agg_iteration_losses.get(opt, {}).get('mean', []))

            max_len = max(num_eval_points, num_iter_points)

            for i in range(max_len):
                 row = {"optimizer": opt}
                 # Eval metrics
                 if i < num_eval_points:
                     row["eval_step"] = eval_steps[i] if eval_steps and i < len(eval_steps) else i
                     row["train_loss_mean"] = agg_train_losses.get(opt, {}).get('mean', [])[i] if i < len(agg_train_losses.get(opt, {}).get('mean', [])) else np.nan
                     row["train_loss_stderr"] = agg_train_losses.get(opt, {}).get('std_err', [])[i] if i < len(agg_train_losses.get(opt, {}).get('std_err', [])) else np.nan
                     row["dev_loss_mean"] = agg_dev_losses.get(opt, {}).get('mean', [])[i] if i < len(agg_dev_losses.get(opt, {}).get('mean', [])) else np.nan
                     row["dev_loss_stderr"] = agg_dev_losses.get(opt, {}).get('std_err', [])[i] if i < len(agg_dev_losses.get(opt, {}).get('std_err', [])) else np.nan
                 else:
                     row["eval_step"] = np.nan
                     row["train_loss_mean"] = np.nan
                     row["train_loss_stderr"] = np.nan
                     row["dev_loss_mean"] = np.nan
                     row["dev_loss_stderr"] = np.nan

                 # Iteration metrics
                 if i < num_iter_points:
                     row["iteration"] = i + 1
                     row["iter_loss_mean"] = agg_iteration_losses.get(opt, {}).get('mean', [])[i] if i < len(agg_iteration_losses.get(opt, {}).get('mean', [])) else np.nan
                     row["iter_loss_stderr"] = agg_iteration_losses.get(opt, {}).get('std_err', [])[i] if i < len(agg_iteration_losses.get(opt, {}).get('std_err', [])) else np.nan
                     row["wall_time_mean"] = agg_wall_times.get(opt, {}).get('mean_time', [])[i] if i < len(agg_wall_times.get(opt, {}).get('mean_time', [])) else np.nan
                 else:
                     row["iteration"] = np.nan
                     row["iter_loss_mean"] = np.nan
                     row["iter_loss_stderr"] = np.nan
                     row["wall_time_mean"] = np.nan

                 df_metrics_list.append(row)


        metrics_df = pd.DataFrame(df_metrics_list)
        metrics_df.to_csv(csv_filename, index=False)
        print(f"Saved aggregated metrics to {csv_filename}")

    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")


def main():
    """Main function to train with multiple optimizers across multiple runs and create comparative visualizations."""
    parser = argparse.ArgumentParser(description="Train transformer models with multiple optimizers")
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs='+',
        default=config['optimizers_to_train'],
        help="List of optimizers to train and compare"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=config.get('num_runs', 1), # Get from config or default to 1
        help="Number of times to run training for each optimizer"
    )
    parser.add_argument(
        "--experiment_title",
        type=str,
        default="Optimizer Comparison",
        help="Base title for plots and filenames"
    )
    args = parser.parse_args()

    if args.num_runs < 1:
        print("Error: --num_runs must be at least 1.")
        sys.exit(1)

    # Create visualization directory based on config
    visuals_dir = config.get('visuals_dir', 'visuals') # Use top-level visuals dir from config
    os.makedirs(visuals_dir, exist_ok=True)

    # Store results from all runs for all optimizers
    all_runs_results = {opt: [] for opt in args.optimizers}

    # Train with each optimizer for the specified number of runs
    for optimizer_name in args.optimizers:
        if optimizer_name not in config['optimizer_params']:
            print(f"Warning: Optimizer {optimizer_name} not found in config. Skipping.")
            continue

        for run_idx in range(args.num_runs):
            try:
                # Train the model for one run
                results = train_with_optimizer(optimizer_name, run_idx, args.num_runs)
                all_runs_results[optimizer_name].append(results)
            except Exception as e:
                print(f"Error during run {run_idx+1} for optimizer {optimizer_name}: {e}")
                # Decide whether to continue with other runs/optimizers or stop
                # For now, just print the error and continue
                all_runs_results[optimizer_name].append(None) # Add placeholder for failed run


    # Filter out optimizers with no successful runs before visualization
    successful_results = {opt: [r for r in runs if r is not None]
                          for opt, runs in all_runs_results.items() if any(r is not None for r in runs)}


    # Create comparative visualizations if any results were obtained
    if successful_results:
        create_comparative_visualizations(successful_results, args.num_runs, visuals_dir, args.experiment_title)
        print(f"\nComparative visualizations saved to {visuals_dir}")
        results_dir = os.path.join(os.path.dirname(visuals_dir), "results")
        print(f"Aggregated metrics CSV saved in {results_dir}")
    else:
        print("\nNo successful training runs completed. Cannot create visualizations.")

if __name__ == "__main__":
    main()
