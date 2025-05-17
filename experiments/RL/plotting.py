import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D

# Publication-quality plot settings
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
# Use tab10 color palette
palette = sns.color_palette("tab10")
# Define markers to match base plotting.py
MARKERS = ['o', 's', 'X', 'P', 'D', '^', 'v', '<', '>', '*', '+']

# Helper function to detect and replace sharp dips (potential outliers)
def replace_sharp_dips_with_nan(data, threshold=2.0):
    """
    Replaces sharp dips in data with NaN.
    A dip at index i is considered sharp if data[i] is significantly lower
    than both data[i-1] and data[i+1] on a log scale.

    Args:
        data (np.array): The input data series (e.g., loss values).
        threshold (float): The minimum log10 difference to consider a dip sharp.

    Returns:
        np.array: Data with sharp dips replaced by np.nan.
    """
    if len(data) < 3:
        return data # Not enough points to detect dips

    data_log = np.log10(np.maximum(data, 1e-5)) # Use log10, avoid log(0)
    data_out = data.copy()

    for i in range(1, len(data) - 1):
        log_diff_prev = data_log[i-1] - data_log[i]
        log_diff_next = data_log[i+1] - data_log[i]

        # Check if it's a sharp dip compared to both neighbors
        if log_diff_prev > threshold and log_diff_next > threshold:
            data_out[i] = np.nan # Replace the dip with NaN

    return data_out

def visualize_trajectory(trajectory, start_pos, goal_pos, episode, optimizer_name):
    """Visualize the agent's trajectory in 3D space for a single episode.

    Args:
        trajectory (np.array): Array of 3D points representing the agent's path.
        start_pos (np.array): Starting position of the agent.
        goal_pos (np.array): Goal position.
        episode (int): The episode number.
        optimizer_name (str): Name of the optimizer used.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert trajectory to numpy array
    trajectory = np.array(trajectory)
    
    # Determine marker style based on optimizer name's position in common optimizer names
    common_optimizers = ['SGD', 'ADAM', 'AdamW', 'NOVOGRAD', 'MILO']
    marker_idx = common_optimizers.index(optimizer_name) if optimizer_name in common_optimizers else 0
    marker = MARKERS[marker_idx % len(MARKERS)]
    
    # Plot trajectory with appropriate marker
    ax.plot(
        trajectory[:, 0], 
        trajectory[:, 1], 
        trajectory[:, 2], 
        label='Trajectory',
        marker=marker,
        markevery=max(1, len(trajectory) // 15)  # Show about 15 markers along trajectory
    )
    
    # Plot start and goal positions
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='red', marker='o', s=100, label='Start')
    ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], c='green', marker='*', s=100, label='Goal')
    
    # Set labels and title
    ax.set_xlabel('X', fontweight='bold')
    ax.set_ylabel('Y', fontweight='bold')
    ax.set_zlabel('Z', fontweight='bold')
    ax.set_title(f'Agent Trajectory - Episode {episode} ({optimizer_name})', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
    
    plt.savefig(f'testing/experiments/RL/visuals/trajectory_episode_{episode}_{optimizer_name}.png', bbox_inches='tight')
    plt.close()

def average_trajectories(trajectories_list):
    """
    Average a list of trajectories by resampling them to a common length and averaging the points.

    Args:
        trajectories_list (list of np.array): A list of trajectories, where each trajectory is an array of 3D points.

    Returns:
        np.array or None: The averaged trajectory as a NumPy array, or None if the input list is empty or contains no valid trajectories.
    """
    if not trajectories_list:
        return None
        
    # Get max length for resampling
    max_len = 100  # Fixed length for consistent comparison
    
    resampled_trajectories = []
    
    for traj in trajectories_list:
        if len(traj) == 0:
            continue
        
        # Simple resampling to max_len points
        indices = np.linspace(0, len(traj) - 1, max_len)
        resampled = np.array([traj[int(i)] if i < len(traj) else traj[-1] for i in indices])
        resampled_trajectories.append(resampled)
        
    if resampled_trajectories:
        # Average the resampled trajectories
        avg_trajectory = np.mean(resampled_trajectories, axis=0)
        return avg_trajectory
    else:
        return None

def plot_training_rewards(all_rewards, optimizer_names, save_path='reward_comparison.png'):
    """Plot training rewards for each optimizer, smoothed for better visualization.

    Args:
        all_rewards (dict): A dictionary mapping optimizer names to lists of reward values per episode.
        optimizer_names (list): A list of optimizer names to include in the plot.
        save_path (str): Path to save the generated plot image.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    
    for i, opt_name in enumerate(optimizer_names):
        reward_values = all_rewards[opt_name]
        # Smoothed rewards for better visualization
        smoothed_rewards = np.array(reward_values)
        window_size = 30  # Increased window size for smoother curve
        if len(smoothed_rewards) > window_size:
            smoothed_rewards = np.convolve(smoothed_rewards, np.ones(window_size)/window_size, mode='valid')
            # Create x-axis based on actual episodes (accounting for convolution window)
            episodes_x = np.linspace(window_size//2, len(reward_values) - window_size//2, len(smoothed_rewards))
        else:
            episodes_x = np.arange(len(smoothed_rewards))
            
        sns.lineplot(x=episodes_x, y=smoothed_rewards, label=opt_name, color=palette[i], linewidth=2, alpha=0.9)
    
    plt.xlabel('Training Episode', fontweight='bold')
    plt.ylabel('Average Reward', fontweight='bold')
    plt.title('Training Reward', fontweight='bold', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
    legend.get_title().set_fontweight('bold')
    
    # Add subtle background shading for readability
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF for publication
    plt.close()

def plot_training_rewards_with_error(all_rewards, rewards_std_err, optimizer_names, save_path='reward_comparison.png', smoothing_window=50):
    """Plot training rewards for each optimizer with error bands and smoothing.

    Args:
        all_rewards (dict): Dictionary mapping optimizer names to lists of mean reward values per episode.
        rewards_std_err (dict): Dictionary mapping optimizer names to lists of standard error values for rewards.
        optimizer_names (list): List of optimizer names to plot.
        save_path (str): Path to save the plot.
        smoothing_window (int): Window size for smoothing the reward curves.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    current_palette = sns.color_palette("tab10", n_colors=len(optimizer_names))
    title = 'Training Reward' # Base title

    for i, opt_name in enumerate(optimizer_names):
        if opt_name not in all_rewards or len(all_rewards[opt_name]) == 0:
             print(f"Warning: No reward data for optimizer {opt_name}. Skipping.")
             continue

        reward_values = np.array(all_rewards[opt_name])
        episodes_x = np.arange(len(reward_values))
        err_vals = rewards_std_err.get(opt_name)

        # Ensure error values match length
        if err_vals is not None and len(err_vals) != len(reward_values):
            print(f"Warning: Length mismatch for reward error values for {opt_name}. Skipping error bars.")
            err_vals = None
        elif err_vals is not None:
             err_vals = np.array(err_vals) # Ensure it's a numpy array

        # Apply smoothing (Pandas rolling mean handles NaNs, though less likely in rewards)
        plot_y_vals = reward_values
        plot_err_vals = err_vals
        current_title = title # Use temp title
        if smoothing_window and isinstance(smoothing_window, int) and smoothing_window > 0 and len(reward_values) > smoothing_window:
            y_series = pd.Series(plot_y_vals)
            plot_y_vals = y_series.rolling(window=smoothing_window, min_periods=1).mean().to_numpy()
            if plot_err_vals is not None:
                 err_series = pd.Series(plot_err_vals)
                 plot_err_vals = err_series.rolling(window=smoothing_window, min_periods=1).mean().to_numpy()
            # Title already includes "Smoothed"
            current_title += " (Smoothed)"

        # Plot line (smoothed or original) with appropriate marker
        marker = MARKERS[i % len(MARKERS)]
        # Determine marker frequency - show fewer markers on longer sequences
        markevery = max(1, len(episodes_x) // 15)  # About 15 markers regardless of length
        
        sns.lineplot(
            x=episodes_x, 
            y=plot_y_vals, 
            label=opt_name, 
            color=current_palette[i], 
            linewidth=2, 
            alpha=0.9,
            marker=marker,
            markevery=markevery,
            markersize=7
        )

        # --- Add error bands ---
        if plot_err_vals is not None:
            valid_indices = ~np.isnan(plot_y_vals) & ~np.isnan(plot_err_vals)
            if np.any(valid_indices):
                lower_bound = plot_y_vals[valid_indices] - plot_err_vals[valid_indices]
                upper_bound = plot_y_vals[valid_indices] + plot_err_vals[valid_indices]
                ax.fill_between(
                    episodes_x[valid_indices],
                    lower_bound,
                    upper_bound,
                    color=current_palette[i],
                    alpha=0.15, # Make error bands slightly transparent
                    linewidth=0
                )
        # -----------------------

    plt.xlabel('Training Episode', fontweight='bold')
    plt.ylabel('Average Reward', fontweight='bold')
    plt.title(current_title, fontweight='bold', fontsize=13) # Use base title
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add error bar explanation
    plt.figtext(0.5, 0.01, "Shaded areas represent standard error of the mean across runs", # Updated text
                ha='center', fontsize=9, fontstyle='italic')

    legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
    if legend:
        legend.get_title().set_fontweight('bold')

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust layout to make room for the footnote
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF for publication
    plt.close()

def plot_iteration_cost(all_step_losses, losses_std_err, optimizer_names, visuals_dir, filename="iteration_loss_comparison", title="Training Loss vs. Iteration", smoothing_window=200, dip_threshold=2.0):
    """Plot training loss against iteration count with error bands, smoothing, dip handling, and interpolation.

    Args:
        all_step_losses (dict): Dict mapping optimizer names to lists of loss values per iteration.
        losses_std_err (dict): Dict mapping optimizer names to lists of standard error for losses.
        optimizer_names (list): List of optimizer names to plot.
        visuals_dir (str): Directory to save the plot.
        filename (str): Base filename for the saved plot.
        title (str): Title of the plot.
        smoothing_window (int): Window size for smoothing loss curves.
        dip_threshold (float): Threshold for detecting and replacing sharp dips in loss data.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    current_palette = sns.color_palette("tab10", n_colors=len(optimizer_names))
    original_title = title # Store original title
    min_loss_overall = float('inf') # Track min loss for y-limit
    is_smoothed = False # Track if smoothing is applied

    max_iterations = 0
    for opt_name in optimizer_names:
        if opt_name in all_step_losses:
            max_iterations = max(max_iterations, len(all_step_losses[opt_name]))

    for i, opt_name in enumerate(optimizer_names):
        current_iter_title = original_title # Reset title for each optimizer line
        if opt_name not in all_step_losses or len(all_step_losses[opt_name]) == 0:
            print(f"Warning: No iteration loss data for optimizer {opt_name}. Skipping.")
            continue

        # --- Dip handling & Interpolation ---
        loss_values = np.array(all_step_losses[opt_name])
        iterations = np.arange(len(loss_values))
        err_vals = losses_std_err.get(opt_name)
        loss_values_processed = loss_values.copy()
        interpolated = False
        if dip_threshold is not None and dip_threshold > 0:
            loss_values_no_dips = replace_sharp_dips_with_nan(loss_values, dip_threshold)
            if np.isnan(loss_values_no_dips).any():
                loss_series = pd.Series(loss_values_no_dips)
                # Interpolate linearly, filling NaNs from both directions
                loss_values_processed = loss_series.interpolate(method='linear', limit_direction='both').to_numpy()
                interpolated = True
                if " (Interpolated)" not in current_iter_title:
                    current_iter_title += " (Interpolated)"
        # -----------------------------------

        # Ensure error values match length and interpolate if needed
        err_vals_processed = None
        if err_vals is not None and len(err_vals) == len(loss_values):
            err_vals_processed = np.array(err_vals)
            if interpolated:
                 err_vals_processed[np.isnan(loss_values_no_dips)] = np.nan
                 err_series = pd.Series(err_vals_processed)
                 err_vals_processed = err_series.interpolate(method='linear', limit_direction='both').to_numpy()
        elif err_vals is not None:
             print(f"Warning: Length mismatch for iteration error values for {opt_name}. Skipping error bars.")

        # Apply smoothing if requested (using the default or passed value)
        plot_y_vals = loss_values_processed
        plot_err_vals = err_vals_processed
        if smoothing_window and isinstance(smoothing_window, int) and smoothing_window > 0 and len(loss_values_processed) > smoothing_window:
            y_series = pd.Series(plot_y_vals)
            plot_y_vals = y_series.rolling(window=smoothing_window, min_periods=1).mean().to_numpy()
            if plot_err_vals is not None:
                 err_series = pd.Series(plot_err_vals)
                 plot_err_vals = err_series.rolling(window=smoothing_window, min_periods=1).mean().to_numpy()
            is_smoothed = True # Mark that smoothing was applied
        # --------------------------------------------------------

        # Track min loss after processing for y-limit setting
        min_loss_overall = min(min_loss_overall, np.nanmin(plot_y_vals[plot_y_vals > 0])) # Ignore non-positive for log

        # Plot line (smoothed or original)
        sns.lineplot(x=iterations, y=plot_y_vals, label=opt_name, color=current_palette[i], linewidth=1.5, alpha=0.85, ax=ax)

        # Add error bands if available
        if plot_err_vals is not None:
            valid_indices = ~np.isnan(plot_y_vals) & ~np.isnan(plot_err_vals)
            if np.any(valid_indices):
                lower_bound = np.maximum(plot_y_vals[valid_indices] - plot_err_vals[valid_indices], 1e-12)
                upper_bound = plot_y_vals[valid_indices] + plot_err_vals[valid_indices]
                ax.fill_between(
                    iterations[valid_indices],
                    lower_bound,
                    upper_bound,
                    color=current_palette[i],
                    alpha=0.15,
                    linewidth=0
                )

    # Construct final title based on processing steps
    final_title = original_title
    if interpolated: # Check if interpolation happened for any optimizer
        if " (Interpolated)" not in final_title:
             final_title += " (Interpolated)"

    plt.xlabel('Training Iteration (Step)', fontweight='bold')
    plt.ylabel('Loss (log scale)', fontweight='bold')
    plt.title(final_title, fontweight='bold', fontsize=13)
    plt.yscale('log')
    # Set a dynamic lower y-limit based on observed minimum loss, but not too low
    if min_loss_overall != float('inf') and min_loss_overall > 0:
         ax.set_ylim(bottom=max(1e-5, min_loss_overall * 0.1)) # E.g., 1 order below min, capped at 1e-5
    else:
         ax.set_ylim(bottom=1e-5) # Default lower limit

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.figtext(0.5, 0.01, "Shaded areas represent standard error of the mean across runs",
                ha='center', fontsize=9, fontstyle='italic')
    legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
    if legend:
        legend.get_title().set_fontweight('bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    filepath_png = os.path.normpath(os.path.join(visuals_dir, filename + '.png'))
    filepath_pdf = os.path.normpath(os.path.join(visuals_dir, filename + '.pdf'))
    plt.savefig(filepath_png, bbox_inches='tight')
    plt.savefig(filepath_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filepath_png} and {filepath_pdf}")

def plot_walltime_cost(all_step_losses, all_step_times, losses_std_err, optimizer_names, visuals_dir, filename="walltime_loss_comparison", title="Training Loss vs. Wall-Clock Time", smoothing_window=200, dip_threshold=2.0):
    """Plot training loss against wall-clock time with error bars, smoothing, dip handling, and interpolation.

    Args:
        all_step_losses (dict): Dict mapping optimizer names to lists of loss values per iteration.
        all_step_times (dict): Dict mapping optimizer names to lists of cumulative wall-clock time per iteration.
        losses_std_err (dict): Dict mapping optimizer names to lists of standard error for losses.
        optimizer_names (list): List of optimizer names to plot.
        visuals_dir (str): Directory to save the plot.
        filename (str): Base filename for the saved plot.
        title (str): Title of the plot.
        smoothing_window (int): Window size for smoothing loss curves.
        dip_threshold (float): Threshold for detecting and replacing sharp dips in loss data.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    current_palette = sns.color_palette("tab10", n_colors=len(optimizer_names))
    original_title = title # Store original title
    min_loss_overall = float('inf') # Track min loss for y-limit
    is_smoothed = False # Track if smoothing is applied

    # --- Map optimizer to color before determining plot order ---
    opt_colors = {name: current_palette[i % len(current_palette)] for i, name in enumerate(optimizer_names)}

    # --- Determine Plotting Order based on end times ---
    # Get final walltime for each optimizer that has data
    final_times = []
    for opt_name in optimizer_names:
        if opt_name in all_step_times and len(all_step_times[opt_name]) > 0:
            final_times.append((opt_name, all_step_times[opt_name][-1]))
    
    # Sort by final walltime, descending (longest time first -> plotted first/bottom)
    if final_times:
        final_times.sort(key=lambda item: item[1], reverse=True)
        plot_order = [opt for opt, _ in final_times]
    else:
        plot_order = optimizer_names # Default order
    
    # --- Plotting loop ---
    for opt_name in plot_order: # Use the determined order
        current_wall_title = original_title # Reset title
        if opt_name not in all_step_losses or opt_name not in all_step_times or len(all_step_losses[opt_name]) == 0:
            print(f"Warning: No walltime/loss data for optimizer {opt_name}. Skipping.")
            continue

        # --- Dip handling & Interpolation ---
        loss_values = np.array(all_step_losses[opt_name])
        time_values = np.array(all_step_times[opt_name])
        err_vals = losses_std_err.get(opt_name)

        # Ensure lengths match for time, loss before dip handling
        min_len = min(len(loss_values), len(time_values))
        loss_values = loss_values[:min_len]
        time_values = time_values[:min_len]
        if err_vals is not None:
            if len(err_vals) >= min_len:
                 err_vals = np.array(err_vals[:min_len])
            else:
                print(f"Warning: Length mismatch for walltime error values for {opt_name} before processing. Skipping error bars.")
                err_vals = None

        loss_values_processed = loss_values.copy()
        interpolated = False
        if dip_threshold is not None and dip_threshold > 0:
            loss_values_no_dips = replace_sharp_dips_with_nan(loss_values, dip_threshold)
            if np.isnan(loss_values_no_dips).any():
                loss_series = pd.Series(loss_values_no_dips)
                loss_values_processed = loss_series.interpolate(method='linear', limit_direction='both').to_numpy()
                interpolated = True
                if " (Interpolated)" not in current_wall_title:
                    current_wall_title += " (Interpolated)"
        err_vals_processed = None
        if err_vals is not None:
            err_vals_processed = err_vals.copy()
            if interpolated:
                 err_vals_processed[np.isnan(loss_values_no_dips)] = np.nan
                 err_series = pd.Series(err_vals_processed)
                 err_vals_processed = err_series.interpolate(method='linear', limit_direction='both').to_numpy()

        # Apply smoothing if requested (using the default or passed value)
        plot_y_vals = loss_values_processed
        plot_err_vals = err_vals_processed
        plot_x_vals = time_values # Use time as x-axis

        if smoothing_window and isinstance(smoothing_window, int) and smoothing_window > 0 and len(loss_values_processed) > smoothing_window:
            # Smooth y-values (loss)
            y_series = pd.Series(plot_y_vals)
            plot_y_vals = y_series.rolling(window=smoothing_window, min_periods=1).mean().to_numpy()
            # Smooth errors similarly if they exist
            if plot_err_vals is not None:
                 err_series = pd.Series(plot_err_vals)
                 plot_err_vals = err_series.rolling(window=smoothing_window, min_periods=1).mean().to_numpy()
            # Smooth x-values (time) to align with smoothed y-values
            x_series = pd.Series(plot_x_vals)
            plot_x_vals = x_series.rolling(window=smoothing_window, min_periods=1).mean().to_numpy()

            is_smoothed = True # Mark that smoothing was applied
        # --------------------------------------------------------

        # Track min loss after processing for y-limit setting
        min_loss_overall = min(min_loss_overall, np.nanmin(plot_y_vals[plot_y_vals > 0]))

        # Get the color for the current optimizer (consistent)
        color = opt_colors[opt_name]
        
        # Determine zorder based on plot_order (higher index = higher zorder = plotted on top)
        current_zorder = plot_order.index(opt_name) + 1

        # Plot line (smoothed or original)
        sns.lineplot(x=plot_x_vals, y=plot_y_vals, label=opt_name, color=color, 
                    linewidth=1.5, alpha=0.85, ax=ax, zorder=current_zorder)

        # Add error bands if available
        if plot_err_vals is not None:
            valid_indices = ~np.isnan(plot_y_vals) & ~np.isnan(plot_err_vals) & ~np.isnan(plot_x_vals)
            if np.any(valid_indices):
                lower_bound = np.maximum(plot_y_vals[valid_indices] - plot_err_vals[valid_indices], 1e-12)
                upper_bound = plot_y_vals[valid_indices] + plot_err_vals[valid_indices]
                ax.fill_between(
                    plot_x_vals[valid_indices],
                    lower_bound,
                    upper_bound,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    zorder=current_zorder - 0.5 # Slightly behind the line
                )
                
        # Add end marker to show where each optimizer's run ended
        marker_zorder = len(plot_order) + 1 # Higher zorder to be on top
        ax.plot(plot_x_vals[-1], plot_y_vals[-1], marker='o', markersize=7, 
               color=color, markeredgecolor='black', markeredgewidth=0.5, 
               linestyle='None', zorder=marker_zorder)

    # Construct final title based on processing steps
    final_title = original_title

    plt.xlabel('Wall-Clock Time (s)', fontweight='bold')
    plt.ylabel('Loss (log scale)', fontweight='bold')
    plt.title(final_title, fontweight='bold', fontsize=13)
    plt.yscale('log')
    # Set a dynamic lower y-limit
    if min_loss_overall != float('inf') and min_loss_overall > 0:
         ax.set_ylim(bottom=max(1e-5, min_loss_overall * 0.1))
    else:
         ax.set_ylim(bottom=1e-5)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.figtext(0.5, 0.01, "Shaded areas represent standard error of the mean across runs. Markers indicate run end points.",
                ha='center', fontsize=9, fontstyle='italic')
    legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
    if legend:
        legend.get_title().set_fontweight('bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    filepath_png = os.path.join(visuals_dir, filename + '.png')
    filepath_pdf = os.path.join(visuals_dir, filename + '.pdf')
    plt.savefig(filepath_png, bbox_inches='tight')
    plt.savefig(filepath_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filepath_png} and {filepath_pdf}")

def plot_success_rates(success_rates, optimizer_names, num_eval_trials, std_errors=None, save_path='success_rates.png'):
    """Plot success rates for each optimizer as a bar chart with error bars.

    Args:
        success_rates (dict): Dictionary mapping optimizer names to their average success rates.
        optimizer_names (list): List of optimizer names to plot.
        num_eval_trials (int): Number of evaluation trials used to calculate success rates.
        std_errors (dict, optional): Dictionary mapping optimizer names to standard error of success rates.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    
    # Convert to arrays for plotting
    opt_array = np.array(optimizer_names)
    success_array = np.array([success_rates[opt] for opt in optimizer_names])
    
    # Default to zero error if not provided
    if std_errors is None:
        error_array = np.zeros_like(success_array)
    else:
        error_array = np.array([std_errors[opt] for opt in optimizer_names])
    
    # Create bar plot with error bars using tab10 palette
    bars = ax.bar(
        opt_array, 
        success_array,
        yerr=error_array,
        capsize=5,
        color=sns.color_palette("tab10", n_colors=len(optimizer_names)), # Use tab10
        alpha=0.85
    )
    
    # Add value labels on top of each bar
    for i, bar in enumerate(bars):
        value = success_array[i]
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + error_array[i] + 0.02,
            f"{value:.2%}",
            ha="center", 
            fontweight='bold',
            fontsize=10
        )
    
    plt.xlabel('Optimizer', fontweight='bold')
    plt.ylabel('Success Rate', fontweight='bold')
    plt.title(f'Success Rate by Optimizer ({num_eval_trials} Evaluation Trials)', fontweight='bold', fontsize=13)
    plt.ylim(0, min(1.1, max(success_array + error_array + 0.05)))  # Dynamic ylim with padding
    
    # Add error bar explanation
    plt.figtext(0.5, 0.01, "Error bars represent standard error of the mean across runs", 
                ha='center', fontsize=9, fontstyle='italic')
    
    # Remove top and right spines for cleaner look
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

def plot_averaged_trajectories_3d(all_eval_trajectories, success_rates, optimizer_names, start_pos, goal_pos, save_path='averaged_trajectories_comparison_seaborn.png'):
    """Plot averaged 3D trajectories for each optimizer with success rates in the legend.

    Args:
        all_eval_trajectories (dict): Dictionary mapping optimizer names to lists of evaluation trajectories.
        success_rates (dict): Dictionary mapping optimizer names to their average success rates.
        optimizer_names (list): List of optimizer names to plot.
        start_pos (np.array): Starting position for trajectories.
        goal_pos (np.array): Goal position for trajectories.
        save_path (str): Path to save the plot.
    """
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    for i, opt_name in enumerate(optimizer_names):
        trajectories = all_eval_trajectories[opt_name]
        avg_traj = average_trajectories([t for t in trajectories if len(t) > 0])
        if avg_traj is not None:
            ax.plot(avg_traj[:, 0], avg_traj[:, 1], avg_traj[:, 2],
                    label=f"{opt_name} (Success: {success_rates[opt_name]:.2%})", # Corrected f-string
                    linewidth=2.5, color=palette[i], alpha=0.9) # Use tab10 palette
    ax.scatter(start_pos[0], start_pos[1], start_pos[2],
               c='blue', marker='o', s=150, label='Start', edgecolor='k')
    ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
               c='green', marker='*', s=200, label='Goal', edgecolor='k')
    ax.set_xlabel('X', fontweight='bold', labelpad=10)
    ax.set_ylabel('Y', fontweight='bold', labelpad=10)
    ax.set_zlabel('Z', fontweight='bold', labelpad=10)
    ax.set_title('Average Agent Trajectories by Optimizer', fontweight='bold', fontsize=13)
    ax.legend(frameon=True, fancybox=True, framealpha=0.95, facecolor='white', title='Optimizer Performance')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

def plot_averaged_trajectories_topdown(all_eval_trajectories, success_rates, optimizer_names, start_pos, goal_pos, save_path='averaged_trajectories_topdown.png'):
    """Plot top-down view of averaged trajectories for each optimizer with success rates and direction arrows.

    Args:
        all_eval_trajectories (dict): Dictionary mapping optimizer names to lists of evaluation trajectories.
        success_rates (dict): Dictionary mapping optimizer names to their average success rates.
        optimizer_names (list): List of optimizer names to plot.
        start_pos (np.array): Starting position for trajectories (X, Y components used).
        goal_pos (np.array): Goal position for trajectories (X, Y components used).
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax_topdown = plt.gca()
    
    # Use the globally defined 'tab10' palette
    current_palette = sns.color_palette("tab10", n_colors=len(optimizer_names))

    # Plot averaged trajectories for each optimizer (only X and Y coordinates)
    for i, opt_name in enumerate(optimizer_names):
        trajectories = all_eval_trajectories[opt_name]
        avg_traj = average_trajectories([t for t in trajectories if len(t) > 0])
        if avg_traj is not None:
            plt.plot(avg_traj[:, 0], avg_traj[:, 1], 
                    label=f"{opt_name} (Success: {success_rates[opt_name]:.2%})", # Corrected f-string
                    linewidth=2.5, color=current_palette[i], alpha=0.9, # Use tab10 palette
                    marker=MARKERS[i % len(MARKERS)], markersize=4, markevery=10)  # Use consistent markers

    # Add start and goal positions (only X and Y)
    plt.scatter(start_pos[0], start_pos[1], c='blue', marker='o', s=150, label='Start', edgecolor='k', zorder=10)
    plt.scatter(goal_pos[0], goal_pos[1], c='green', marker='*', s=200, label='Goal', edgecolor='k', zorder=10)

    # Set labels and title
    plt.xlabel('X Position', fontweight='bold')
    plt.ylabel('Y Position', fontweight='bold')
    plt.title('Top-Down View of Average Agent Trajectories', fontweight='bold', fontsize=13)
    
    # Improve legend
    legend = plt.legend(title='Optimizer Performance', frameon=True, fancybox=True, 
                      framealpha=0.95, facecolor='white', loc='best')
    legend.get_title().set_fontweight('bold')
    
    ax_topdown.set_aspect('equal')
    ax_topdown.grid(True, linestyle='--', alpha=0.6)
    
    # Add direction arrows at intervals
    for i, opt_name in enumerate(optimizer_names):
        trajectories = all_eval_trajectories[opt_name]
        avg_traj = average_trajectories([t for t in trajectories if len(t) > 0])
        if avg_traj is not None and len(avg_traj) > 20:
            # Add direction arrows (every 20 points)
            arrow_indices = np.arange(10, len(avg_traj)-10, 20)
            for idx in arrow_indices:
                dx = avg_traj[idx+5, 0] - avg_traj[idx, 0]
                dy = avg_traj[idx+5, 1] - avg_traj[idx, 1]
                plt.arrow(avg_traj[idx, 0], avg_traj[idx, 1], dx, dy, 
                        head_width=0.03, head_length=0.05, fc=current_palette[i], ec=current_palette[i], alpha=0.6) # Use tab10 palette
    
    # Remove top and right spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

def plot_training_losses(all_losses, optimizer_names, save_path='loss_comparison.png'):
    """Plot training losses (vs episode) for each optimizer, smoothed for better visualization.

    Args:
        all_losses (dict): A dictionary mapping optimizer names to lists of loss values per episode.
        optimizer_names (list): A list of optimizer names to include in the plot.
        save_path (str): Path to save the generated plot image.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    
    # Use the globally defined 'tab10' palette
    current_palette = sns.color_palette("tab10", n_colors=len(optimizer_names))

    for i, opt_name in enumerate(optimizer_names):
        loss_values = all_losses[opt_name]
        # Smoothed loss for better visualization
        smoothed_loss = np.array(loss_values)
        window_size = 30  # Increased window size for smoother curve
        if len(smoothed_loss) > window_size:
            smoothed_loss = np.convolve(smoothed_loss, np.ones(window_size)/window_size, mode='valid')
            # Create x-axis based on actual episodes (accounting for convolution window)
            episodes_x = np.linspace(window_size//2, len(loss_values) - window_size//2, len(smoothed_loss))
        else:
            episodes_x = np.arange(len(smoothed_loss))
            
        # Determine marker frequency - show fewer markers on longer sequences
        markevery = max(1, len(episodes_x) // 15)  # About 15 markers regardless of length
        marker = MARKERS[i % len(MARKERS)]
        
        # Plot with consistent markers
        sns.lineplot(
            x=episodes_x, 
            y=smoothed_loss, 
            label=opt_name, 
            color=current_palette[i], 
            linewidth=1.5, 
            alpha=0.85,
            marker=marker,
            markevery=markevery,
            markersize=6
        )
    
    plt.xlabel('Training Episode', fontweight='bold')
    plt.ylabel('Loss (log scale)', fontweight='bold')
    plt.title('Training Loss', fontweight='bold', fontsize=13)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
    if legend:
        legend.get_title().set_fontweight('bold')
    
    # Add subtle background shading for readability
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF for publication
    plt.close()

def plot_training_losses_with_error(all_losses, losses_std_err, optimizer_names, save_path='loss_comparison.png', dip_threshold=2.0, smoothing_window=100):
    """Plot training losses (vs episode) with error bands, smoothing, dip handling, and interpolation.

    Args:
        all_losses (dict): Dictionary mapping optimizer names to lists of mean loss values per episode.
        losses_std_err (dict): Dictionary mapping optimizer names to lists of standard error values for losses.
        optimizer_names (list): List of optimizer names to plot.
        save_path (str): Path to save the plot.
        dip_threshold (float): Threshold for detecting and replacing sharp dips in loss data.
        smoothing_window (int): Window size for smoothing the loss curves.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    current_palette = sns.color_palette("tab10", n_colors=len(optimizer_names))
    base_title = 'Training Loss' # Base title
    min_loss_overall = float('inf') # Track min loss for y-limit

    for i, opt_name in enumerate(optimizer_names):
        title = base_title # Reset title for each line
        if opt_name not in all_losses or len(all_losses[opt_name]) == 0:
             print(f"Warning: No episode loss data for optimizer {opt_name}. Skipping.")
             continue

        loss_values = np.array(all_losses[opt_name])
        episodes_x = np.arange(len(loss_values))
        err_vals = losses_std_err.get(opt_name)

        # --- Dip handling & Interpolation ---
        loss_values_processed = loss_values.copy()
        interpolated = False
        if dip_threshold is not None and dip_threshold > 0:
            loss_values_no_dips = replace_sharp_dips_with_nan(loss_values, dip_threshold)
            if np.isnan(loss_values_no_dips).any():
                loss_series = pd.Series(loss_values_no_dips)
                # Interpolate linearly, filling NaNs from both directions
                loss_values_processed = loss_series.interpolate(method='linear', limit_direction='both').to_numpy()
                interpolated = True
                if " (Interpolated)" not in title:
                    title += " (Interpolated)"
        # -----------------------------------

        # Ensure error values match length and interpolate if needed
        err_vals_processed = None
        if err_vals is not None and len(err_vals) == len(loss_values):
            err_vals_processed = np.array(err_vals)
            if interpolated:
                 # Apply NaN mask from original dip detection before interpolating errors
                 err_vals_processed[np.isnan(loss_values_no_dips)] = np.nan
                 err_series = pd.Series(err_vals_processed)
                 err_vals_processed = err_series.interpolate(method='linear', limit_direction='both').to_numpy()
        elif err_vals is not None:
             print(f"Warning: Length mismatch for episode error values for {opt_name}. Skipping error bars.")

        # Apply smoothing (to processed data)
        plot_y_vals = loss_values_processed
        plot_err_vals = err_vals_processed
        if smoothing_window and isinstance(smoothing_window, int) and smoothing_window > 0 and len(loss_values_processed) > smoothing_window:
            y_series = pd.Series(plot_y_vals)
            plot_y_vals = y_series.rolling(window=smoothing_window, min_periods=1).mean().to_numpy()
            if plot_err_vals is not None:
                 err_series = pd.Series(plot_err_vals)
                 plot_err_vals = err_series.rolling(window=smoothing_window, min_periods=1).mean().to_numpy()

        # Track min loss after processing for y-limit setting
        min_loss_overall = min(min_loss_overall, np.nanmin(plot_y_vals[plot_y_vals > 0]))

        # Select marker and determine marker frequency
        marker = MARKERS[i % len(MARKERS)]
        markevery = max(1, len(episodes_x) // 15)  # About 15 markers regardless of length
        
        # Plot line with appropriate marker
        sns.lineplot(
            x=episodes_x, 
            y=plot_y_vals, 
            label=opt_name, 
            color=current_palette[i], 
            linewidth=1.5, 
            alpha=0.85,
            marker=marker,
            markevery=markevery,
            markersize=6,
            ax=ax
        )

        # Add error bands if available
        if plot_err_vals is not None:
            valid_indices = ~np.isnan(plot_y_vals) & ~np.isnan(plot_err_vals)
            if np.any(valid_indices):
                lower_bound = np.maximum(plot_y_vals[valid_indices] - plot_err_vals[valid_indices], 1e-12)
                upper_bound = plot_y_vals[valid_indices] + plot_err_vals[valid_indices]
                ax.fill_between(
                    episodes_x[valid_indices],
                    lower_bound,
                    upper_bound,
                    color=current_palette[i],
                    alpha=0.15,
                    linewidth=0
                )

    plt.xlabel('Training Episode', fontweight='bold')
    plt.ylabel('Loss (log scale)', fontweight='bold')
    # Use the potentially modified title (Smoothed, Interpolated)
    plt.title(title, fontweight='bold', fontsize=13)
    plt.yscale('log')
    # Set a dynamic lower y-limit
    if min_loss_overall != float('inf') and min_loss_overall > 0:
         ax.set_ylim(bottom=max(1e-5, min_loss_overall * 0.1))
    else:
         ax.set_ylim(bottom=1e-5)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.figtext(0.5, 0.01, "Shaded areas represent standard error of the mean across runs",
                ha='center', fontsize=9, fontstyle='italic')
    legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
    if legend:
        legend.get_title().set_fontweight('bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path} and {save_path.replace('.png', '.pdf')}")
