import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

MARKERS = ['o', 's', 'X', 'P', 'D', '^', 'v', '<', '>', '*', '+']

def setup_plot_style():
    """Sets up a standardized plot style using seaborn and matplotlib rcParams."""
    sns.set(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14

def plot_average_gradient_norm(
    all_gradient_norms, optimizer_names, layer_names, visuals_dir, plot_filename, experiment_title, epochs, steps_per_epoch
):
    """
    Plots the average gradient norm across ALL layers over steps for each optimizer.

    Args:
        all_gradient_norms (list): List (over runs) of dicts (optimizer -> layer -> list of norms per epoch).
        optimizer_names (list): List of optimizer names.
        layer_names (list): List of layer names.
        visuals_dir (str): Directory to save the plot.
        plot_filename (str): Base filename for the plot.
        experiment_title (str): Title for the experiment.
        epochs (int): Number of epochs.
        steps_per_epoch (int): Number of steps (batches) per epoch.
    """
    if not all_gradient_norms or not layer_names:
        print("No gradient norm data or layer names provided for average norm plotting.")
        return

    plot_data = []
    num_runs = len(all_gradient_norms)
    for run_idx, run_norms in enumerate(all_gradient_norms):
        for opt_name in optimizer_names:
            if opt_name in run_norms:
                # Collect norms for all layers for this optimizer and run
                all_layers_norms_epoch = [[] for _ in range(epochs)]  # List of lists (epochs x layers)
                valid_epoch_counts = True
                for layer_name in layer_names:
                    if layer_name in run_norms[opt_name]:
                        norms_over_epochs = run_norms[opt_name][layer_name]
                        if len(norms_over_epochs) == epochs:
                            for epoch, norm in enumerate(norms_over_epochs):
                                if norm is not None and np.isfinite(norm):
                                     all_layers_norms_epoch[epoch].append(norm)
                        else:
                            valid_epoch_counts = False
                            break # Mismatched epochs for a layer
                    else:
                         # Layer missing for this optimizer/run, can't compute average reliably
                         valid_epoch_counts = False
                         break

                if valid_epoch_counts:
                    # Calculate average norm across layers for each epoch
                    for epoch in range(epochs):
                        step = (epoch + 1) * steps_per_epoch # Calculate step number
                        if all_layers_norms_epoch[epoch]: # Check if list is not empty
                            avg_norm = np.mean(all_layers_norms_epoch[epoch])
                            plot_data.append({
                                "Run": run_idx,
                                "Step": step, # Use Step instead of Epoch
                                "Optimizer": opt_name,
                                "Average Gradient Norm": avg_norm
                            })
                        else:
                             # Append NaN if no valid norms for this epoch
                             plot_data.append({
                                 "Run": run_idx,
                                 "Step": step, # Use Step instead of Epoch
                                 "Optimizer": opt_name,
                                 "Average Gradient Norm": np.nan
                             })

    if not plot_data:
        print("No valid data for average gradient norm plot.")
        return

    df = pd.DataFrame(plot_data).dropna(subset=['Average Gradient Norm']) # Drop NaN averages

    if df.empty:
        print("Average gradient norm data frame is empty after handling missing values.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Use "tab10" palette
    palette = sns.color_palette("tab10", n_colors=len(optimizer_names))
    opt_colors = {opt: color for opt, color in zip(optimizer_names, palette)}

    sns.lineplot(
        data=df,
        x="Step", # Use Step for x-axis
        y="Average Gradient Norm",
        hue="Optimizer",
        palette=opt_colors,
        ax=ax,
        errorbar=('ci', 95)
    )
    ax.set_title(f"{experiment_title}: Average Gradient Norm Across All Layers vs Steps") # Update title
    ax.set_ylabel("Average Gradient Norm")
    ax.set_xlabel("Steps") # Update x-axis label
    ax.set_yscale('log')  # Log scale is usually appropriate
    ax.legend(title="Optimizer")

    plt.tight_layout()
    filepath = os.path.join(visuals_dir, f"{plot_filename}_gradient_norm_average_steps.png") # Update filename
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved average gradient norm plot (vs Steps) to {filepath}")


def plot_gradient_imbalance_ratio(all_gradient_norms, optimizer_names, layer_names, visuals_dir, plot_filename, experiment_title, epochs, steps_per_epoch):
    """
    Plots the ratio of the last layer's gradient norm to the first layer's gradient norm over steps.

    Args:
        all_gradient_norms (list): List (over runs) of dicts (optimizer -> layer -> list of norms per epoch).
        optimizer_names (list): List of optimizer names.
        layer_names (list): List of layer names.
        visuals_dir (str): Directory to save the plot.
        plot_filename (str): Base filename for the plot.
        experiment_title (str): Title for the experiment.
        epochs (int): Number of epochs.
        steps_per_epoch (int): Number of steps (batches) per epoch.
    """
    if not all_gradient_norms or not layer_names or len(layer_names) < 2:
        print("Skipping gradient imbalance ratio plot: requires at least two layers.")
        return

    first_layer = layer_names[0]
    last_layer = layer_names[-1]
    epsilon = 1e-12  # Small value to prevent division by zero

    plot_data = []
    num_runs = len(all_gradient_norms)
    for run_idx, run_norms in enumerate(all_gradient_norms):
        for opt_name in optimizer_names:
            if opt_name in run_norms and first_layer in run_norms[opt_name] and last_layer in run_norms[opt_name]:
                first_layer_norms = run_norms[opt_name][first_layer]
                last_layer_norms = run_norms[opt_name][last_layer]

                if len(first_layer_norms) == epochs and len(last_layer_norms) == epochs:
                    for epoch in range(epochs):
                        step = (epoch + 1) * steps_per_epoch # Calculate step number
                        norm_first = first_layer_norms[epoch]
                        norm_last = last_layer_norms[epoch]
                        # Ensure norms are valid numbers before calculating ratio
                        if norm_first is not None and norm_last is not None and np.isfinite(norm_first) and np.isfinite(norm_last):
                            ratio = norm_last / (norm_first + epsilon)
                            plot_data.append({
                                "Run": run_idx,
                                "Step": step, # Use Step instead of Epoch
                                "Optimizer": opt_name,
                                "Imbalance Ratio": ratio
                            })
                        else:
                             # Append NaN if norms are invalid, will be dropped later
                             plot_data.append({
                                 "Run": run_idx,
                                 "Step": step, # Use Step instead of Epoch
                                 "Optimizer": opt_name,
                                 "Imbalance Ratio": np.nan
                             })

    if not plot_data:
        print("No valid data for gradient imbalance ratio plot.")
        return

    df = pd.DataFrame(plot_data).dropna(subset=['Imbalance Ratio']) # Drop rows with NaN ratios

    if df.empty:
        print("Gradient imbalance ratio data frame is empty after handling missing values.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Use "tab10" palette
    palette = sns.color_palette("tab10", n_colors=len(optimizer_names))
    opt_colors = {opt: color for opt, color in zip(optimizer_names, palette)}

    sns.lineplot(
        data=df,
        x="Step", # Use Step for x-axis
        y="Imbalance Ratio",
        hue="Optimizer",
        palette=opt_colors,
        ax=ax,
        errorbar=('ci', 95)
    )
    ax.set_title(f"{experiment_title}: Gradient Imbalance Ratio (Last/First Layer) vs Steps") # Update title
    ax.set_ylabel(f"Ratio (Norm({last_layer}) / Norm({first_layer}))")
    ax.set_xlabel("Steps") # Update x-axis label
    ax.set_yscale('log')  # Often useful for ratios
    ax.legend(title="Optimizer")
    ax.axhline(1.0, color='grey', linestyle='--', label='Balanced (Ratio=1)')  # Add reference line

    plt.tight_layout()
    filepath = os.path.join(visuals_dir, f"{plot_filename}_gradient_imbalance_ratio_steps.png") # Update filename
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved gradient imbalance ratio plot (vs Steps) to {filepath}")


def plot_gradient_norms_layerwise(all_gradient_norms, optimizer_names, layer_names, visuals_dir, plot_filename, experiment_title, epochs, steps_per_epoch):
    """
    Plots the average gradient norm per layer over steps for each optimizer using line plots.
    Handles single-layer cases. Renamed from visualize_gradient_norms.

    Args:
        all_gradient_norms (list): List (over runs) of dicts (optimizer -> layer -> list of norms per epoch).
        optimizer_names (list): List of optimizer names.
        layer_names (list): List of layer names.
        visuals_dir (str): Directory to save the plot.
        plot_filename (str): Base filename for the plot.
        experiment_title (str): Title for the experiment.
        epochs (int): Number of epochs.
        steps_per_epoch (int): Number of steps (batches) per epoch.
    """
    if not all_gradient_norms or not layer_names:
        print("No gradient norm data or layer names provided for plotting.")
        return

    # Prepare data for DataFrame
    plot_data = []
    num_runs = len(all_gradient_norms)
    for run_idx, run_norms in enumerate(all_gradient_norms):
        for opt_name in optimizer_names:
            if opt_name in run_norms:
                for layer_name in layer_names:
                    if layer_name in run_norms[opt_name]:
                        norms_over_epochs = run_norms[opt_name][layer_name]
                        # Ensure norms_over_epochs list matches the number of epochs
                        if len(norms_over_epochs) == epochs:
                            for epoch, norm in enumerate(norms_over_epochs):
                                step = (epoch + 1) * steps_per_epoch # Calculate step number
                                plot_data.append({
                                    "Run": run_idx,
                                    "Step": step, # Use Step instead of Epoch
                                    "Optimizer": opt_name,
                                    "Layer": layer_name,
                                    "Gradient Norm": norm if norm is not None else np.nan # Handle potential None values
                                })
                        else:
                             print(f"Warning: Mismatch in epoch count for {opt_name}, layer {layer_name}, run {run_idx}. Expected {epochs}, got {len(norms_over_epochs)}. Skipping.")


    if not plot_data:
        print("No valid gradient norm data collected for plotting.")
        return

    df = pd.DataFrame(plot_data).dropna(subset=['Gradient Norm']) # Drop rows where norm was None/NaN

    if df.empty:
        print("Gradient norm data frame is empty after handling missing values.")
        return

    # Determine the number of layers and prepare for subplots
    n_layers = len(layer_names)
    n_cols = min(3, n_layers) # Adjust number of columns, max 3 or n_layers if fewer
    n_rows = (n_layers + n_cols - 1) // n_cols
    # Use squeeze=False to ensure axes is always 2D, even if n_rows/n_cols is 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), sharex=True, squeeze=False)
    axes = axes.flatten() # Flatten to easily iterate

    # Define a consistent color palette using "tab10"
    palette = sns.color_palette("tab10", n_colors=len(optimizer_names))
    opt_colors = {opt: color for opt, color in zip(optimizer_names, palette)}


    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        # Filter potentially empty DataFrame safely
        layer_df = df[df["Layer"] == layer_name] if "Layer" in df.columns else pd.DataFrame()


        if not layer_df.empty:
            # Plot average gradient norm over steps for the current layer
            sns.lineplot(
                data=layer_df,
                x="Step", # Use Step for x-axis
                y="Gradient Norm",
                hue="Optimizer",
                palette=opt_colors,
                ax=ax,
                errorbar=('ci', 95) # Show 95% confidence interval across runs
            )
            ax.set_title(f"Layer: {layer_name}")
            ax.set_ylabel("Avg. Gradient Norm")
            ax.set_xlabel("Steps") # Update x-axis label
            ax.set_yscale('log') # Use log scale if norms vary widely
            # Only show legend on the first subplot if multiple layers, or always if one layer
            if i == 0 or n_layers == 1:
                 ax.legend(title="Optimizer")
            else:
                 ax.legend().set_visible(False)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{experiment_title}: Average Gradient Norm per Layer over Steps", fontsize=16) # Update title
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout slightly
    filepath = os.path.join(visuals_dir, f"{plot_filename}_gradient_norms_layerwise_steps.png") # Update filename
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved layer-wise gradient norm line plot (vs Steps) to {filepath}")


def plot_resource_usage(resource_df, visuals_dir, experiment_file_id, base_experiment_name):
    """Generates bar plots for key resource usage metrics."""
    if resource_df.empty:
        print("Resource data frame is empty. Skipping resource plots.")
        return

    # Ensure numeric types for aggregation
    numeric_cols_to_check = [
        'duration_seconds',
        'peak_gpu_memory_allocated_gb',
        'memory_increase_gb',
        'end_gpu_utilization_percent' # Use end utilization as a proxy
    ]
    existing_numeric_cols = [] # Store columns that actually exist
    for col in numeric_cols_to_check:
        if col in resource_df.columns:
            resource_df[col] = pd.to_numeric(resource_df[col], errors='coerce')
            existing_numeric_cols.append(col) # Add existing column to the list
        else:
            print(f"Warning: Resource metric column '{col}' not found. Skipping related plot.")

    # Drop rows where key metrics are NaN after conversion, only using existing columns
    if existing_numeric_cols: # Only drop if there are columns to check
        resource_df.dropna(subset=existing_numeric_cols, how='any', inplace=True)

    if resource_df.empty:
        print("Resource data frame is empty after handling NaNs. Skipping resource plots.")
        return

    # --- Plot Peak GPU Memory --- #
    if 'peak_gpu_memory_allocated_gb' in resource_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=resource_df, x='optimizer', y='peak_gpu_memory_allocated_gb', ci='sd', capsize=.2, palette="viridis")
        plt.title(f'Peak GPU Memory Usage for {base_experiment_name}')
        plt.ylabel('Peak GPU Memory (GB)')
        plt.xlabel('Optimizer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = os.path.join(visuals_dir, f"peak_gpu_memory_{experiment_file_id}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved peak GPU memory plot to {plot_filename}")

    # --- Plot Total Duration --- #
    if 'duration_seconds' in resource_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=resource_df, x='optimizer', y='duration_seconds', ci='sd', capsize=.2, palette="viridis")
        plt.title(f'Total Training Duration for {base_experiment_name}')
        plt.ylabel('Duration (seconds)')
        plt.xlabel('Optimizer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = os.path.join(visuals_dir, f"training_duration_{experiment_file_id}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved training duration plot to {plot_filename}")

    # --- Plot GPU Utilization (End) --- #
    if 'end_gpu_utilization_percent' in resource_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=resource_df, x='optimizer', y='end_gpu_utilization_percent', ci='sd', capsize=.2, palette="viridis")
        plt.title(f'GPU Utilization (End of Training) for {base_experiment_name}')
        plt.ylabel('GPU Utilization (%)')
        plt.xlabel('Optimizer')
        plt.ylim(0, 105) # Set y-axis limit 0-100%
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = os.path.join(visuals_dir, f"gpu_utilization_{experiment_file_id}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved GPU utilization plot to {plot_filename}")

    # --- Plot CPU RAM Increase --- #
    if 'memory_increase_gb' in resource_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=resource_df, x='optimizer', y='memory_increase_gb', ci='sd', capsize=.2, palette="viridis")
        plt.title(f'CPU RAM Increase During Training for {base_experiment_name}')
        plt.ylabel('CPU RAM Increase (GB)')
        plt.xlabel('Optimizer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = os.path.join(visuals_dir, f"cpu_ram_increase_{experiment_file_id}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved CPU RAM increase plot to {plot_filename}")


def plot_seaborn_style_with_error_bars(
    data, data_std_err, x_values, title, filename, y_label, visuals_dir, xlabel="Epoch", xlimit=None, yscale=None
):
    """
    Create a standardized seaborn plot with error bars showing standard error.
    For walltime plots (xlabel="Time (s)"), adds end markers and layers lines
    based on finish time (longest runs plotted first/bottom).
    Includes unique markers for optimizers and ensures legend order.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    optimizer_names_original_order = list(data.keys()) # Keep original order for legend

    # --- Color Assignment (Consistent) --- 
    try:
        # Use "tab10" palette for potentially more distinct colors
        palette = sns.color_palette("tab10", n_colors=len(optimizer_names_original_order))
    except: # Handle cases where len(data) might fail or be large
        palette = sns.color_palette("tab10")
    # Map optimizer name to color BEFORE sorting for plotting order
    opt_colors = {name: palette[i % len(palette)] for i, name in enumerate(optimizer_names_original_order)}

    # --- Determine Plotting Order (for Walltime) --- 
    is_walltime_plot = (xlabel.lower() == "time (s)")
    plot_order = optimizer_names_original_order # Default order

    if is_walltime_plot and isinstance(x_values, dict):
        # Get final walltime for each optimizer that has data
        final_times = []
        for opt in optimizer_names_original_order:
            # Check if x_values[opt] exists and is not empty
            opt_x_vals = x_values.get(opt)
            if opt_x_vals is not None and hasattr(opt_x_vals, '__len__') and len(opt_x_vals) > 0:
                # Ensure corresponding y data exists and lengths match
                opt_y_vals = data.get(opt)
                if opt_y_vals is not None and hasattr(opt_y_vals, '__len__') and len(opt_y_vals) == len(opt_x_vals):
                    final_times.append((opt, opt_x_vals[-1]))
                else:
                    print(f"Warning: Mismatched data lengths for {opt} in walltime plot. Excluding from sorting.")
            else:
                print(f"Warning: No valid x_values found for {opt} in walltime plot. Excluding from sorting.")

        if final_times:
            # Sort by final walltime, descending (longest time first -> plotted first/bottom)
            final_times.sort(key=lambda item: item[1], reverse=True)
            plot_order = [opt for opt, _ in final_times]
        else:
            print("Warning: Could not determine walltime plot order. Using default.")

    # --- Plotting Loop --- 
    for optimizer_name in plot_order: # Iterate using the determined order
        # Skip if optimizer somehow wasn't in data
        if optimizer_name not in data:
            continue

        metrics = data[optimizer_name]
        color = opt_colors[optimizer_name] # Get consistent color

        # Assign marker based on original optimizer_names list
        try:
            marker_index = optimizer_names_original_order.index(optimizer_name)
            current_marker = MARKERS[marker_index % len(MARKERS)]
        except ValueError: # Should not happen if data.keys() is used for optimizer_names_original_order
            current_marker = MARKERS[0]

        # --- Prepare x_vals, y_vals, err_vals --- 
        x_vals, y_vals, err_vals = [], [], None # Initialize

        if isinstance(x_values, dict):
            # Handle dict x_values (e.g., walltime)
            current_x_vals = x_values.get(optimizer_name, [])
            # Ensure metrics and current_x_vals have the same length for plotting
            min_len = min(len(metrics), len(current_x_vals))
            y_vals = metrics[:min_len]
            x_vals = current_x_vals[:min_len]
            if optimizer_name in data_std_err:
                # Ensure error array also exists and has sufficient length
                if data_std_err[optimizer_name] is not None and len(data_std_err[optimizer_name]) >= min_len:
                    err_vals = data_std_err[optimizer_name][:min_len]
                else:
                    err_vals = None # Set err_vals to None if std_err data is missing or too short
            else:
                err_vals = None
        else:
            # Handle range or list x_values (e.g., epochs, iterations/steps)
            max_len = len(metrics)
            current_x_range = x_values # Use the original x_values passed

            # Apply xlimit if applicable (only for non-dict x_values)
            if xlimit is not None:
                if isinstance(current_x_range, range):
                    # Adjust max_len based on xlimit for range
                    # Ensure xlimit is treated as upper bound (exclusive for range, inclusive for slicing)
                    effective_xlimit_idx = xlimit - current_x_range.start # Assuming step is 1
                    if effective_xlimit_idx < len(current_x_range):
                        max_len = min(max_len, effective_xlimit_idx) # Slice up to index before xlimit
                elif isinstance(current_x_range, (list, np.ndarray)):
                    # Find index corresponding to xlimit for list/array
                    x_vals_np = np.array(current_x_range)
                    # Find the index of the last element <= xlimit
                    valid_indices = np.where(x_vals_np <= xlimit)[0]
                    if len(valid_indices) > 0:
                        max_len = min(max_len, valid_indices[-1] + 1) # Include the element at xlimit
                    else:
                        max_len = 0 # No points within xlimit

            y_vals = metrics[:max_len]
            # Generate or slice x_vals based on max_len
            if isinstance(current_x_range, range):
                x_vals = list(current_x_range)[:max_len] # Convert range slice to list
            elif isinstance(current_x_range, (list, np.ndarray)):
                x_vals = current_x_range[:max_len]
            else: # Fallback if x_values type is unexpected
                x_vals = list(range(1, max_len + 1))

            if optimizer_name in data_std_err:
                # Ensure error array also exists and has sufficient length
                if data_std_err[optimizer_name] is not None and len(data_std_err[optimizer_name]) >= max_len:
                    err_vals = data_std_err[optimizer_name][:max_len]
                else:
                    err_vals = None # Set err_vals to None if std_err data is missing or too short
            else:
                err_vals = None

        # Plot line with error bands
        if len(x_vals) > 0 and len(y_vals) > 0: # Ensure there's data to plot
            # Determine zorder based on plot_order (higher index = higher zorder = plotted on top)
            # Add 1 because zorder=0 is default background
            current_zorder = plot_order.index(optimizer_name) + 1

            # Adaptive markevery for cleaner plots with many points
            num_points = len(x_vals)
            if num_points < 50:
                plot_markevery = 1  # Mark all points if there are few
            else:
                plot_markevery = max(1, num_points // 15) # Aim for around 15 markers

            sns.lineplot(x=x_vals, y=y_vals, label=optimizer_name, color=color, 
                         marker=current_marker, markersize=7, markevery=plot_markevery, # Use adaptive markevery
                         linewidth=2, alpha=0.9, ax=ax, zorder=current_zorder)

            # Add error bands if available
            if err_vals is not None and len(err_vals) == len(y_vals):
                ax.fill_between(
                    x_vals,
                    np.array(y_vals) - np.array(err_vals),
                    np.array(y_vals) + np.array(err_vals),
                    color=color,
                    alpha=0.2,
                    zorder=current_zorder - 0.5 # Slightly behind the line
                )

            # Add end marker for walltime plot
            if is_walltime_plot:
                # Ensure marker is plotted with high zorder to be visible
                # Use a zorder higher than any line/band
                marker_zorder = len(plot_order) + 1
                ax.plot(x_vals[-1], y_vals[-1], marker='o', markersize=7, color=color, markeredgecolor='black', markeredgewidth=0.5, linestyle='None', zorder=marker_zorder)

        else:
            print(f"Warning: No valid data points to plot for optimizer {optimizer_name} in '{title}'.")

    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.title(title, fontweight='bold', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)

    if xlimit is not None and not is_walltime_plot: # Apply xlimit unless it's walltime
        # Adjust xlim start based on x_values type
        xlim_start = 0
        if isinstance(x_values, range):
            xlim_start = x_values.start
        elif isinstance(x_values, (list, np.ndarray)) and len(x_values) > 0:
            xlim_start = x_values[0]
        plt.xlim(left=xlim_start, right=xlimit)

    if yscale:
        plt.yscale(yscale)

    # Add error bar explanation in the caption
    plt.figtext(0.5, 0.01, "Error bands represent standard error of the mean across runs. Markers indicate run end on walltime plots.",
                ha='center', fontsize=9, fontstyle='italic')

    # Ensure legend order matches optimizer_names_original_order
    handles, labels = ax.get_legend_handles_labels()
    hl_dict = dict(zip(labels, handles))
    
    ordered_handles = []
    ordered_labels = []
    for name in optimizer_names_original_order: # Use the original order from data.keys()
        if name in hl_dict:
            ordered_handles.append(hl_dict[name])
            ordered_labels.append(name)

    if ordered_handles: # Only create legend if there are handles
        legend = ax.legend(ordered_handles, ordered_labels, title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
        if legend: # Check if legend exists
            legend.get_title().set_fontweight('bold')
    else: # Fallback if no specific handles were ordered (e.g. if optimizer_names_original_order was empty or no lines plotted)
        legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
        if legend:
            legend.get_title().set_fontweight('bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 1]) # Adjust layout slightly for figtext
    filepath_png = os.path.join(visuals_dir, filename + '.png')
    filepath_pdf = os.path.join(visuals_dir, filename + '.pdf')
    plt.savefig(filepath_png, bbox_inches='tight')
    plt.savefig(filepath_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filepath_png} and {filepath_pdf}")
