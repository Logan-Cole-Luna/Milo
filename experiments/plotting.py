import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

MARKERS = ['o', 's', 'X', 'P', 'D', '^', 'v', '<', '>', '*', '+']

def setup_plot_style():
    """
    Sets up a standardized plot style using seaborn and matplotlib rcParams.
    
    Args:
        None
        
    Returns:
        None
    """
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

def plot_resource_usage(resource_df, visuals_dir, experiment_file_id, base_experiment_name):
    """
    Generates bar plots for key resource usage metrics.
    
    Args:
        resource_df: DataFrame containing resource usage data.
        visuals_dir: Directory to save the plots.
        experiment_file_id: Unique identifier for the experiment file.
        base_experiment_name: Base name for the experiment.
        
    Returns:
        None
    """
    if resource_df.empty:
        print("Resource data frame is empty. Skipping resource plots.")
        return

    # Ensure numeric types for aggregation
    numeric_cols_to_check = [
        'duration_seconds',
        'peak_gpu_memory_allocated_gb',
        'memory_increase_gb',
        'end_gpu_utilization_percent' 
    ]
    existing_numeric_cols = []
    for col in numeric_cols_to_check:
        if col in resource_df.columns:
            resource_df[col] = pd.to_numeric(resource_df[col], errors='coerce')
            existing_numeric_cols.append(col) 
        else:
            print(f"Warning: Resource metric column '{col}' not found. Skipping related plot.")

    if existing_numeric_cols: 
        resource_df.dropna(subset=existing_numeric_cols, how='any', inplace=True)

    if resource_df.empty:
        print("Resource data frame is empty after handling NaNs. Skipping resource plots.")
        return

    # --- Plot Peak GPU Memory --- #
    if 'peak_gpu_memory_allocated_gb' in resource_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=resource_df, x='optimizer', y='peak_gpu_memory_allocated_gb', errorbar='sd', capsize=.2, palette="viridis", hue='optimizer', legend=False)
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
        sns.barplot(data=resource_df, x='optimizer', y='duration_seconds', errorbar='sd', capsize=.2, palette="viridis", hue='optimizer', legend=False)
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
        sns.barplot(data=resource_df, x='optimizer', y='end_gpu_utilization_percent', errorbar='sd', capsize=.2, palette="viridis", hue='optimizer', legend=False)
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
        sns.barplot(data=resource_df, x='optimizer', y='memory_increase_gb', errorbar='sd', capsize=.2, palette="viridis", hue='optimizer', legend=False)
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
    
    Args:
        data: Dict mapping optimizer names to lists of metric values (mean).
        data_std_err: Dict mapping optimizer names to lists of standard error values.
        x_values: List, range, or dict mapping optimizer names to x-axis values.
        title: Title of the plot.
        filename: Filename for saving the plot (without extension).
        y_label: Label for the y-axis.
        visuals_dir: Directory to save the plots.
        xlabel: Label for the x-axis (default: "Epoch").
        xlimit: Optional upper limit for the x-axis.
        yscale: Optional scale for the y-axis (e.g., "log").
        
    Returns:
        None
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    optimizer_names_original_order = list(data.keys())

    # --- Color Assignment --- 
    try:
        palette = sns.color_palette("tab10", n_colors=len(optimizer_names_original_order))
    except: 
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
    for optimizer_name in plot_order: 
        if optimizer_name not in data:
            continue

        metrics = data[optimizer_name]
        color = opt_colors[optimizer_name] 

        # Assign marker based on original optimizer_names list
        try:
            marker_index = optimizer_names_original_order.index(optimizer_name)
            current_marker = MARKERS[marker_index % len(MARKERS)]
        except ValueError: 
            current_marker = MARKERS[0]

        # --- Prepare x_vals, y_vals, err_vals --- 
        x_vals, y_vals, err_vals = [], [], None 

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
                    zorder=current_zorder - 0.5 
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

    if xlimit is not None and not is_walltime_plot: 
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

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    filepath_png = os.path.join(visuals_dir, filename + '.png')
    plt.savefig(filepath_png, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filepath_png}")
