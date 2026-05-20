import torch
import time
import numpy as np
import pandas as pd
import itertools
import os
import sys
import platform
import psutil
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from scipy import stats
import json
import warnings
import platform as _platform

# Suppress the PyTorch full backward hook warning globally to avoid the runtime warning message
warnings.filterwarnings("ignore", ".*Full backward hook is firing.*")

def get_layer_names(model):
    """
    Extract meaningful layer names from model
    
    Args:
        model: The model from which to extract layer names.
        
    Returns:
        list: A list of layer names.
    """
    layer_names = []
    for name, _ in model.named_parameters():
        if '.' in name:
            layer_name = name.split('.')[0]
            if layer_name not in layer_names:
                layer_names.append(layer_name)
        else:
            if name not in layer_names:
                layer_names.append(name)
    return layer_names

def compute_gradient_norms(model, layer_names):
    """
    Compute gradient norms for each layer
    
    Args:
        model: The model for which to compute gradient norms.
        layer_names: A list of layer names.
        
    Returns:
        dict: A dictionary mapping layer names to lists of gradient norms.
    """
    gradient_norms = {}
    for layer in layer_names:
        gradient_norms[layer] = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer = name.split('.')[0] if '.' in name else name
            if layer in gradient_norms:
                # Convert tensor to scalar immediately to avoid keeping reference to full gradient tensor
                gradient_norms[layer].append(float(param.grad.norm().item()))
    
    return gradient_norms

def evaluate_model(model, data_loader, criterion, device, task_type="classification"):
    """
    Evaluate the model on a given dataset.
    
    Args:
        model: The model to evaluate.
        data_loader: The data loader for the dataset.
        criterion: The loss function.
        device: The device to use for evaluation.
        task_type: The type of task (e.g., "classification").
        
    Returns:
        dict: A dictionary of evaluation metrics.
    """
    model.eval() 
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            if task_type == "classification":
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

    if total == 0:  # Handle empty dataloader case
        if task_type == "classification":
            return {'loss': 0.0, 'accuracy': 0.0, 'f1_score': 0.0, 'auc': 0.0}
        else:
            return {'loss': 0.0}

    avg_loss = running_loss / total

    if task_type == "classification":
        accuracy = 100.0 * correct / total
        all_targets_np = np.array(all_targets)
        all_preds_np = np.array(all_preds)
        all_probs_np = np.array(all_probs)

        f1 = f1_score(all_targets_np, all_preds_np, average='macro', zero_division=0)

        auc = -1.0  # Placeholder
        num_classes = all_probs_np.shape[1]
        unique_targets = np.unique(all_targets_np)

        if len(unique_targets) == 2 and num_classes == 2:
            auc = roc_auc_score(all_targets_np, all_probs_np[:, 1])
        elif len(unique_targets) > 1 and num_classes > 2:
            try:
                auc = roc_auc_score(all_targets_np, all_probs_np, multi_class='ovr', average='macro', labels=np.arange(num_classes))
            except ValueError as e:
                print(f"Could not compute AUC (targets shape: {all_targets_np.shape}, probs shape: {all_probs_np.shape}, unique targets: {unique_targets}): {e}")
                auc = -1.0
        else:
            auc = -1.0

        return {'loss': avg_loss, 'accuracy': accuracy, 'f1_score': f1, 'auc': auc}
    else:
        return {'loss': avg_loss}

def run_training(model, train_loader, val_loader, optimizer, criterion, device, epochs, scheduler=None, layer_names=None):
    """
    Generic training loop with validation at each epoch.
    Returns validation metrics, cumulative wall times, gradient norms, iteration costs, the trained model,
    steps per epoch, training metrics, and detailed iteration-level logs.
    
    Args:
        model: The model to train.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        optimizer: The optimizer to use for training.
        criterion: The loss function.
        device: The device to use for training.
        epochs: The number of epochs to train for.
        scheduler: An optional learning rate scheduler.
        layer_names: An optional list of layer names for gradient norm tracking.
        
    Returns:
        tuple: A 13-element tuple containing:
            - val_metrics_hist: Dictionary with epoch-level validation metrics
            - cumulative_times: List of cumulative wall times per epoch
            - gradient_norms_history: Dictionary of gradient norms per layer
            - iter_costs: List of training losses per iteration
            - layer_runtime_history: Forward pass timing per layer
            - layer_bwd_runtime_history: Backward pass timing per layer  
            - component_runtime_history: Component-level timing
            - model: The trained model
            - steps_per_epoch: Number of steps per epoch
            - train_metrics_hist: Dictionary with epoch-level training metrics
            - iteration_logs: Dictionary with step-level data (step_numbers, epoch_numbers, 
                            batch_losses, cumulative_walltime, batch_walltime, learning_rates, batch_sizes)
            - validation_walltimes: List of cumulative wall times at validation points
            - epoch_end_walltimes: List of cumulative wall times at end of each epoch
    """
    val_metrics_hist = {'epoch': [], 'val_loss': [], 'val_accuracy': [], 'val_f1_score': [], 'val_auc': []}
    train_metrics_hist = {'epoch': [], 'train_loss': [], 'train_accuracy': [], 'train_f1_score': [], 'train_auc': []}
    gradient_norms_history = {layer: [] for layer in layer_names} if layer_names else {}
    epoch_times = []
    training_start = time.time()

    iter_costs = []  # Store training loss per iteration
    iter_times = []   # Store time per iteration
    
    # NEW: Detailed iteration-level logging
    iteration_logs = {
        'step_numbers': [],           # Global step counter
        'epoch_numbers': [],          # Which epoch each step belongs to
        'batch_losses': [],           # Loss for each mini-batch
        'cumulative_walltime': [],    # Cumulative time since training start
        'batch_walltime': [],         # Time taken for each batch
        'learning_rates': [],         # Learning rate at each step
        'batch_sizes': []             # Actual batch size (may vary for last batch)
    }
    
    # NEW: Validation walltime tracking
    validation_walltimes = []  # Walltime when each validation occurred
    epoch_end_walltimes = []   # Walltime when each epoch ended
    
    global_step = 0
    
    # Setup component-level timing
    fwd_times = []
    bwd_times = []
    opt_times = []
    overhead_times = []
    component_runtime_history = {'forward': [], 'backward': [], 'optimizer': [], 'overhead': []}

    # Setup per-layer runtime measurement
    if layer_names:
        # Prepare timers and counts for forward
        timers = {layer: 0.0 for layer in layer_names}
        counts = {layer: 0 for layer in layer_names}
        # History per epoch for forward
        layer_runtime_history = {layer: [] for layer in layer_names}
        hook_data = {}
        hooks = []
        # Prepare timers and counts for backward
        bwd_timers = {layer: 0.0 for layer in layer_names}
        bwd_counts = {layer: 0 for layer in layer_names}
        # History per epoch for backward
        layer_bwd_runtime_history = {layer: [] for layer in layer_names}
        bwd_hook_data = {}
        # Define hook factories
        def make_pre_hook(layer):
            def pre_hook(module, inputs):
                hook_data[layer] = time.time()
            return pre_hook
        def make_post_hook(layer):
            def post_hook(module, inputs, output):
                start = hook_data.get(layer)
                if start is not None:
                    elapsed = time.time() - start
                    timers[layer] += elapsed
                    counts[layer] += 1
            return post_hook
        def make_bwd_pre_hook(layer):
            def pre_bwd_hook(module, *args):
                # args can be (grad_input,) or (grad_input, grad_output)
                bwd_hook_data[layer] = time.time()
            return pre_bwd_hook
        def make_bwd_post_hook(layer):
            def post_bwd_hook(module, grad_input, grad_output):
                start = bwd_hook_data.get(layer)
                if start is not None:
                    elapsed = time.time() - start
                    bwd_timers[layer] += elapsed
                    bwd_counts[layer] += 1
            return post_bwd_hook
        # Register hooks on top-level modules
        is_windows = (_platform.system() == 'Windows')
        for layer in layer_names:
            module = getattr(model, layer, None)
            # Register hooks only on submodules (skip Tensors/Parameters like cls_token/pos_embed)
            if module is not None and isinstance(module, torch.nn.Module):
                # Forward timing hooks
                hooks.append(module.register_forward_pre_hook(make_pre_hook(layer)))
                hooks.append(module.register_forward_hook(make_post_hook(layer)))
                # Register backward timing hooks only if module has trainable parameters
                # On Windows, skip full backward hooks to avoid potential autograd hangs.
                if not is_windows:
                    if any(hasattr(module, 'parameters') and p.requires_grad for p in module.parameters()):
                        try:
                            hooks.append(module.register_full_backward_pre_hook(make_bwd_pre_hook(layer)))
                            hooks.append(module.register_full_backward_hook(make_bwd_post_hook(layer)))
                        except AttributeError:
                            # Fallback for older PyTorch: register backward hook (may not support pre-hook)
                            try:
                                hooks.append(module.register_backward_hook(lambda mod, grad_in, grad_out: None))
                                print(f"Warning: Full backward hooks not supported; backward timing may be unavailable for layer {layer}.")
                            except Exception:
                                pass
    else:
        layer_runtime_history = None
        layer_bwd_runtime_history = None
     
    steps_per_epoch = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        num_train_batches = 0

        for inputs, targets in train_loader:
            # Batch start
            batch_start = time.time()
            global_step += 1
            batch_size = targets.size(0)
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Forward pass timing
            fwd_start = time.time()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            fwd_elapsed = time.time() - fwd_start
            fwd_times.append(fwd_elapsed)
            
            # Backward pass timing
            bwd_start = time.time()
            loss.backward()
            bwd_elapsed = time.time() - bwd_start
            bwd_times.append(bwd_elapsed)
            
            # Optimizer step timing
            opt_start = time.time()
            optimizer.step()
            opt_elapsed = time.time() - opt_start
            opt_times.append(opt_elapsed)
            
            # Overhead (data movement, other calls)
            overhead_elapsed = time.time() - batch_start - (fwd_elapsed + bwd_elapsed + opt_elapsed)
            overhead_times.append(overhead_elapsed)

            running_train_loss += loss.item()
            num_train_batches += 1

            # Original iteration tracking
            iter_costs.append(loss.item())
            iter_elapsed_time = time.time() - batch_start
            iter_times.append(iter_elapsed_time)
            
            # NEW: Detailed iteration logging
            cumulative_time = time.time() - training_start
            
            # Get current learning rate
            if hasattr(optimizer, 'param_groups'):
                current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = None
                
            iteration_logs['step_numbers'].append(global_step)
            iteration_logs['epoch_numbers'].append(epoch)
            iteration_logs['batch_losses'].append(loss.item())
            iteration_logs['cumulative_walltime'].append(cumulative_time)
            iteration_logs['batch_walltime'].append(iter_elapsed_time)
            iteration_logs['learning_rates'].append(current_lr)
            iteration_logs['batch_sizes'].append(batch_size)

        # --- Calculate Training Metrics for the Epoch ---
        # Record component-level average runtimes for this epoch
        if fwd_times:
            component_runtime_history['forward'].append(sum(fwd_times)/len(fwd_times))
            component_runtime_history['backward'].append(sum(bwd_times)/len(bwd_times))
            component_runtime_history['optimizer'].append(sum(opt_times)/len(opt_times))
            component_runtime_history['overhead'].append(sum(overhead_times)/len(overhead_times))
        else:
            # No iterations, append zeros
            for key in component_runtime_history:
                component_runtime_history[key].append(0.0)
        # Reset iteration component lists
        fwd_times.clear(); bwd_times.clear(); opt_times.clear(); overhead_times.clear()

        avg_train_loss = running_train_loss / num_train_batches if num_train_batches > 0 else 0
        train_eval_metrics = evaluate_model(model, train_loader, criterion, device) # Evaluate on train set
        train_loss = train_eval_metrics['loss'] # Use evaluated loss for consistency
        train_accuracy = train_eval_metrics['accuracy']
        train_f1 = train_eval_metrics['f1_score']
        train_auc = train_eval_metrics['auc']

        train_metrics_hist['epoch'].append(epoch)
        train_metrics_hist['train_loss'].append(train_loss) # Append evaluated train loss
        train_metrics_hist['train_accuracy'].append(train_accuracy)
        train_metrics_hist['train_f1_score'].append(train_f1)
        train_metrics_hist['train_auc'].append(train_auc)
        # --- End Training Metrics Calculation ---

        # --- Calculate Validation Metrics ---
        validation_start_time = time.time() - training_start  # Walltime when validation starts
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        validation_end_time = time.time() - training_start    # Walltime when validation ends
        
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        val_f1 = val_metrics['f1_score']
        val_auc = val_metrics['auc']

        # --- Gradient Norm Calculation (per epoch) ---
        if layer_names:
            epoch_gradient_norms = compute_gradient_norms(model, layer_names)
            for layer in layer_names:
                if layer in epoch_gradient_norms and epoch_gradient_norms[layer]:
                    layer_norms = epoch_gradient_norms[layer]
                    avg_layer_norm = sum(layer_norms) / len(layer_norms) if layer_norms else 0.0
                    gradient_norms_history[layer].append(float(avg_layer_norm))
                else:
                    gradient_norms_history[layer].append(0.0) 

        # Record layer runtimes for this epoch
        if layer_names:
            for layer in layer_names:
                if counts[layer] > 0:
                    avg_time = timers[layer] / counts[layer]
                else:
                    avg_time = 0.0
                layer_runtime_history[layer].append(avg_time)
            # Reset forward timers and counts for next epoch
            for layer in layer_names:
                timers[layer] = 0.0
                counts[layer] = 0
            # Record backward runtimes for this epoch
            for layer in layer_names:
                if bwd_counts[layer] > 0:
                    avg_bwd = bwd_timers[layer] / bwd_counts[layer]
                else:
                    avg_bwd = 0.0
                layer_bwd_runtime_history[layer].append(avg_bwd)
            # Reset backward timers and counts
            for layer in layer_names:
                bwd_timers[layer] = 0.0
                bwd_counts[layer] = 0

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Append validation metrics
        val_metrics_hist['epoch'].append(epoch)
        val_metrics_hist['val_loss'].append(val_loss)
        val_metrics_hist['val_accuracy'].append(val_accuracy)
        val_metrics_hist['val_f1_score'].append(val_f1)
        val_metrics_hist['val_auc'].append(val_auc)
        
        # NEW: Track validation walltime
        validation_walltimes.append(validation_start_time)
        epoch_end_walltimes.append(time.time() - training_start)

        # Update print statement
        print(f"Epoch {epoch}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f}, | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

    total_time = time.time() - training_start
    cumulative_times = list(np.cumsum(iter_times))
    steps_per_epoch = len(train_loader) 
    print(f"Total training time: {total_time:.2f} seconds")
    # Remove hooks
    if layer_names:
        for h in hooks:
            h.remove()
    # Return including per-layer and component-level runtime history
    return (
        val_metrics_hist,
        cumulative_times,
        gradient_norms_history,
        iter_costs,
        layer_runtime_history,
        layer_bwd_runtime_history,
        component_runtime_history,
        model,
        steps_per_epoch,
        train_metrics_hist,
        iteration_logs,
        validation_walltimes,
        epoch_end_walltimes
    )

# Function to print system hardware info
def print_system_info():
    """
    Print system hardware information.
    
    Args:
        None
        
    Returns:
        None
    """
    print("=== System Information ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Count: {psutil.cpu_count(logical=True)}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    
    # Check GPU information if CUDA is available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
    else:
        print("GPU: No CUDA-compatible GPU found")
    print("==========================\\n")

def calculate_statistics(data_arrays):
    """
    Calculate mean, standard deviation, standard error, and 95% confidence interval.
    
    Args:
        data_arrays: List of arrays, each representing one run's data.
        
    Returns:
        dict: Statistics including mean, std_dev, std_err, conf_interval, and n_runs.
    """
    if not data_arrays:
        return None
        
    data_arrays = [np.array(arr) for arr in data_arrays]
    min_length = min(len(arr) for arr in data_arrays)
    data_arrays = [arr[:min_length] for arr in data_arrays]
    stacked_data = np.stack(data_arrays)
    mean = np.mean(stacked_data, axis=0)
    n_runs = len(data_arrays)
    
    if n_runs > 1:
        std_dev = np.std(stacked_data, axis=0, ddof=1)
        std_err = std_dev / np.sqrt(n_runs)
        t_value = stats.t.ppf(0.975, n_runs - 1)
        conf_interval = t_value * std_err
    else:
        std_dev = np.zeros_like(mean)
        std_err = np.zeros_like(mean)
        conf_interval = np.zeros_like(mean)
    
    return {
        'mean': mean,
        'std_dev': std_dev,
        'std_err': std_err,
        'conf_interval': conf_interval,
        'n_runs': n_runs
    }

def perform_statistical_tests(optimizer_data, metric_name='final_accuracy'):
    """
    Perform statistical significance tests between pairs of optimizers.
    
    Args:
        optimizer_data: Dict mapping optimizer names to lists of final metric values.
        metric_name: Name of the metric being tested.
        
    Returns:
        DataFrame: Table of p-values and significance indicators.
    """
    optimizers = list(optimizer_data.keys())
    results = []
    for opt1, opt2 in itertools.combinations(optimizers, 2):
        data1 = optimizer_data[opt1]
        data2 = optimizer_data[opt2]
        if not data1 or not data2:
            print(f"Warning: Skipping statistical test between {opt1} and {opt2} due to missing data.")
            continue
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        significance = ""
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        mean1, mean2 = np.mean(data1), np.mean(data2)
        if 'loss' in metric_name.lower():
            better = opt1 if mean1 < mean2 else opt2
        else:
            better = opt1 if mean1 > mean2 else opt2
        results.append({
            'Optimizer A': opt1,
            'Optimizer B': opt2,
            'Mean A': mean1,
            'Mean B': mean2,
            'Better': better,
            'p-value': p_value,
            'Significant': significance,
            'Metric': metric_name
        })
    return pd.DataFrame(results)

def save_statistics_report(stats_data, results_dir, filename):
    """
    Save a comprehensive statistical report in Markdown and CSV formats.

    Args:
        stats_data: Dictionary containing statistics for each metric.
        results_dir: Directory to save the report.
        filename: Base filename for the report.
        
    Returns:
        None
    """
    report_path = os.path.join(results_dir, f"{filename}_statistics_report.md")
    with open(report_path, 'w') as f:
        f.write("# Statistical Analysis Report\n\n")
        f.write("## Methodology\n\n")
        f.write("Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.\n")
        f.write("Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.\n\n") # Clarify scope
        for metric_name, metric_data in stats_data.items():
            # Replace underscores with spaces and title case for display
            display_metric_name = metric_name.replace('_', ' ').title()
            f.write(f"### {display_metric_name} Statistics\n\n")
            if 'n_runs' in metric_data:
                f.write(f"Number of runs: {metric_data['n_runs']}\n\n")
            # Use final_stats which contains the full statistics dictionary for the *final* value
            if 'final_stats' in metric_data:
                final_stats = metric_data['final_stats']
                f.write("**Final Epoch Statistics:**\n\n") # Clarify these are final epoch stats
                f.write("| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |\n")
                f.write("|-----------|------|---------|-----------|--------------|--------------|\n")
                # Iterate through the final_stats dictionary
                for opt, stats_val in final_stats.items():
                    # Check if stats_val is a dict and has the expected keys
                    if isinstance(stats_val, dict) and all(k in stats_val for k in ['mean', 'std_dev', 'std_err', 'conf_interval']):
                        mean_val = stats_val['mean'][0] if isinstance(stats_val['mean'], np.ndarray) else stats_val['mean']
                        std_dev_val = stats_val['std_dev'][0] if isinstance(stats_val['std_dev'], np.ndarray) else stats_val['std_dev']
                        std_err_val = stats_val['std_err'][0] if isinstance(stats_val['std_err'], np.ndarray) else stats_val['std_err']
                        ci_lower = mean_val - (stats_val['conf_interval'][0] if isinstance(stats_val['conf_interval'], np.ndarray) else stats_val['conf_interval'])
                        ci_upper = mean_val + (stats_val['conf_interval'][0] if isinstance(stats_val['conf_interval'], np.ndarray) else stats_val['conf_interval'])
                        f.write(f"| {opt} | {mean_val:.4f} | {std_dev_val:.4f} | {std_err_val:.4f} | {ci_lower:.4f} | {ci_upper:.4f} |\n")
                    else:
                         f.write(f"| {opt} | N/A | N/A | N/A | N/A | N/A |\n") # Handle cases where stats might be missing
                f.write("\n")
            if 'significance_tests' in metric_data and not metric_data['significance_tests'].empty:
                f.write("#### Pairwise Significance Tests (Final Epoch)\n\n") # Clarify scope
                f.write(metric_data['significance_tests'].to_markdown(index=False))
                f.write("\n\n")
    print(f"Statistical report saved to {report_path}")

def save_experimental_settings(settings, results_dir, filename):
    """
    Save experimental settings in JSON and CSV formats.
    
    Args:
        settings: Dictionary of settings.
        results_dir: Directory to save the settings.
        filename: Base filename.
        
    Returns:
        None
    """
    json_path = os.path.join(results_dir, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(settings, f, indent=2, default=str)
    flattened_settings = []
    for opt, params in settings.items():
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_params[f"{key}_{sub_key}"] = str(sub_value)
            else:
                flat_params[key] = str(value)
        flat_params['optimizer'] = opt
        flattened_settings.append(flat_params)
    csv_path = os.path.join(results_dir, f"{filename}.csv")
    pd.DataFrame(flattened_settings).to_csv(csv_path, index=False)
    print(f"Experimental settings saved to {json_path} and {csv_path}")
