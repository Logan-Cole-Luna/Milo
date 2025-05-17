import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import sys, os
import csv
import json
from scipy import stats
import time
import random 

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.dirname(__file__))  
from experiments.RL.point_navigation_env import PointNavigationEnv 
from milo import milo
from network import ComplexPolicyNetwork
from experiments.hyperparameter_tuning_utils import tune_hyperparameters
from config import (
    GAMMA, GRADIENT_CLIP, MAX_STEPS, EPISODES 
)
from experiments.utils.resource_tracker import ResourceTracker, save_resource_info
resource_tracking_available = True
from experiments.experiment_runner import (perform_statistical_tests, 
                                                    save_statistics_report,
                                                    save_experimental_settings)
from scipy import stats
stats_functions_available = True
from experiments.RL.plotting import (
    plot_training_rewards_with_error,
    plot_training_losses_with_error, plot_success_rates, 
    plot_averaged_trajectories_topdown, plot_averaged_trajectories_3d, 
    plot_iteration_cost, plot_walltime_cost
)

from config import (
    GAMMA, EPISODES, LOG_INTERVAL, START_POS, GOAL_POS,
    INITIAL_EXPLORATION, FINAL_EXPLORATION, EXPLORATION_DECAY, GRADIENT_CLIP,
    DISTANCE_THRESHOLD, RANDOM_GOAL, MAX_STEPS, PARAM_GRID,
    OPTIMIZER_PARAMS, NUM_RUNS, CSV_FILENAME, OPTIMIZERS
)

from novograd import NovoGrad

# Set base_dir based on RANDOM_GOAL flag
if RANDOM_GOAL:
    base_dir = 'experiments/RL/rand/'
else:
    base_dir = 'experiments/RL/stand/'

visuals_dir = os.path.join(base_dir, 'visuals')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(visuals_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Create directory for statistics
stats_dir = os.path.join(base_dir, 'statistics')
os.makedirs(stats_dir, exist_ok=True)

def select_action(model, state, exploration_noise):
    """Select action from policy with exploration noise"""
    state = torch.from_numpy(state).float()
    with torch.no_grad():
        action_mean = model(state)
    
    # Add noise for exploration, scaled by the exploration parameter
    noise = np.random.normal(0, exploration_noise, size=action_mean.shape)
    action = action_mean.numpy() + noise
    return action

def train_episode(model, optimizer, env, gamma, exploration_noise, start_time, global_step_counter):
    """Train the model for one episode.

    Args:
        model: The policy model.
        optimizer: The optimizer for the model.
        env: The RL environment.
        gamma: Discount factor.
        exploration_noise: Current exploration noise level.
        start_time: Start time of the current run (for wall-clock time tracking).
        global_step_counter: Counter for total steps across all episodes in the run.

    Returns:
        tuple: episode_reward, trajectory, episode_loss, step_losses, step_times, 
               step_iterations, updated_global_step_counter.
    """
    state, _ = env.reset()
    states = []
    actions_taken = []
    rewards = []
    trajectory = [state[:3].copy()]
    step_losses = []
    step_times = []
    step_iterations = []

    done = False
    truncated = False
    episode_steps = 0
    episode_reward = 0

    while not done and not truncated and episode_steps < env.max_steps:
        current_step_time = time.time() - start_time
        states.append(state)
        state_tensor = torch.FloatTensor(state)

        # Get action from model and add exploration noise
        with torch.no_grad():
            action_mean = model(state_tensor)

        action_noise = torch.normal(mean=0.0, std=exploration_noise, size=action_mean.shape)
        action = action_mean + action_noise
        actions_taken.append(action)

        # Take action in environment
        next_state, reward, done, truncated, info = env.step(action.detach().numpy())

        rewards.append(reward)
        episode_reward += reward
        trajectory.append(next_state[:3].copy())
        state = next_state
        episode_steps += 1
        global_step_counter += 1 # Increment global step counter

    # --- Policy Update --- (Performed once per episode in this setup)
    if states: # Only update if steps were taken
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.stack(actions_taken)

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards_tensor = torch.FloatTensor(discounted_rewards)

        # Normalize rewards for stability
        if len(discounted_rewards_tensor) > 1:
            discounted_rewards_tensor = (discounted_rewards_tensor - discounted_rewards_tensor.mean()) / (discounted_rewards_tensor.std() + 1e-9)
        elif len(discounted_rewards_tensor) == 1:
             discounted_rewards_tensor = torch.zeros_like(discounted_rewards_tensor) # Avoid NaN if only one step

        # Forward pass to get predicted actions
        optimizer.zero_grad()
        predicted_actions = model(states_tensor)

        # Calculate loss (mean squared error weighted by discounted rewards)
        losses = torch.sum((actions_tensor - predicted_actions)**2, dim=1)
        weighted_losses = losses * discounted_rewards_tensor
        loss = weighted_losses.mean()

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        # Record the single episode loss for each step taken in this episode
        current_episode_loss = loss.item()
        step_losses = [current_episode_loss] * episode_steps
        step_times = [current_step_time] * episode_steps # Approximate time for each step as end-of-episode time
        step_iterations = list(range(global_step_counter - episode_steps + 1, global_step_counter + 1))

    else: # Handle case where episode ends immediately
         current_episode_loss = np.nan
         step_losses = []
         step_times = []
         step_iterations = []

    return episode_reward, np.array(trajectory), current_episode_loss, step_losses, step_times, step_iterations, global_step_counter

def run_experiment(optimizer_name, run_idx=0):
    """Train a model with the specified optimizer once and return settings, including step-wise data.

    Args:
        optimizer_name (str): Name of the optimizer to use.
        run_idx (int): Index of the current run (for resource tracking).

    Returns:
        tuple: model, total_rewards, final_trajectory, episode_losses, 
               all_step_losses_run, all_step_times_run, all_step_iterations_run, 
               resource_info, settings.
    """
    start_time = time.time() # Start timer for the run
    global_step_counter = 0 # Initialize global step counter for this run

    # Track resources if available
    if resource_tracking_available:
        resource_tracker = ResourceTracker().start()
    
    # Create environment with random goals for training
    env = PointNavigationEnv(start_pos=START_POS, goal_pos=GOAL_POS, random_goal=RANDOM_GOAL, distance_threshold=DISTANCE_THRESHOLD, max_steps=MAX_STEPS)
    model = ComplexPolicyNetwork(env.observation_space, env.action_space)
    
    # Print model size information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params} total parameters, {trainable_params} trainable parameters")
    
    # Define a function to create the environment for tuning
    # Use fixed goal for tuning evaluation consistency
    def create_tuning_env():
        return PointNavigationEnv(start_pos=START_POS, goal_pos=GOAL_POS, 
                                  random_goal=False, # Use fixed goal for eval within tuning
                                  distance_threshold=DISTANCE_THRESHOLD, 
                                  max_steps=MAX_STEPS)

    # Tune hyperparameters using RL-specific settings
    tuning_episodes_per_trial = 10 # Number of training episodes within each Optuna trial
    tuning_eval_episodes = 3     # Number of evaluation episodes within each Optuna trial
    tuning_num_trials = 10       # Number of Optuna trials per optimizer

    print(f"Starting hyperparameter tuning for {optimizer_name} ({tuning_num_trials} trials)...")
    best_hyperparams = tune_hyperparameters(
        model_fn=lambda: ComplexPolicyNetwork(env.observation_space, env.action_space),
        optimizer_names=[optimizer_name],
        param_grid=PARAM_GRID,
        device=torch.device('cpu'), # Assuming CPU for tuning, adjust if needed
        experiment_name="RL",
        task_type="rl", # *** Correct task type ***
        epochs=tuning_episodes_per_trial, # Use 'epochs' to mean training episodes per trial
        num_trials=tuning_num_trials,
        # SL args (set to None for RL)
        train_loader=None,
        val_ratio=None,
        criterion=None,
        # RL args
        env_fn=create_tuning_env, # Pass the function to create env instances
        gamma=GAMMA, 
        clip_grad=GRADIENT_CLIP,
        episodes_to_eval=tuning_eval_episodes, 
        max_steps_per_episode=MAX_STEPS 
    )[optimizer_name]
    
    print(f"Finished tuning. Using best hyperparameters for {optimizer_name}: {best_hyperparams}")
    
    # Handle None case and parse 'betas' if it's a string
    if best_hyperparams is None:
        best_hyperparams = {}
    
    # Convert 'betas' string back to tuple if necessary
    if 'betas' in best_hyperparams and isinstance(best_hyperparams['betas'], str):
        try:
            best_hyperparams['betas'] = tuple(map(float, best_hyperparams['betas'].split(',')))
        except ValueError:
            print(f"Warning: Could not parse betas string '{best_hyperparams['betas']}'. Using default betas.")
            # Remove the invalid string entry so it doesn't cause issues later
            del best_hyperparams['betas'] 

    # Create optimizer parameters, starting with defaults and updating with tuned ones
    params = OPTIMIZER_PARAMS.get(optimizer_name.upper(), {}).copy() 
    # Handle None case from tuning before updating
    if best_hyperparams is None:
        print(f"Warning: Hyperparameter tuning for {optimizer_name} did not yield results. Using defaults.")
        best_hyperparams = {}
    else:
        # Convert 'betas' string back to tuple if necessary (moved parsing here)
        if 'betas' in best_hyperparams and isinstance(best_hyperparams['betas'], str):
            try:
                best_hyperparams['betas'] = tuple(map(float, best_hyperparams['betas'].split(',')))
            except ValueError:
                print(f"Warning: Could not parse betas string '{best_hyperparams['betas']}' from tuning. Removing from params.")
                del best_hyperparams['betas'] # Remove invalid entry

    params.update(best_hyperparams) # Update with tuned params
    
    # Store experimental settings (using the final 'params' dictionary)
    settings = {
        "model": "ComplexPolicyNetwork",
        "model_architecture": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space)
        },
        "optimizer_params": params, # Ensure the final params are stored
        "environment": {
            "name": "PointNavigation3D",
            "start_position": list(START_POS),
            "goal_position": list(GOAL_POS),
            "random_goal": RANDOM_GOAL,
            "distance_threshold": DISTANCE_THRESHOLD,
            "max_steps": MAX_STEPS
        },
        "training_params": {
            "gamma": GAMMA,
            "episodes": EPISODES,
            "gradient_clip": GRADIENT_CLIP,
            "initial_exploration": INITIAL_EXPLORATION,
            "final_exploration": FINAL_EXPLORATION,
            "exploration_decay": EXPLORATION_DECAY
        },
        "tuning_params": { # Add info about the tuning process itself
             "tuning_episodes_per_trial": tuning_episodes_per_trial,
             "tuning_eval_episodes": tuning_eval_episodes,
             "tuning_num_trials": tuning_num_trials
        },
        "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # Removed "scheduler_params"
    }
    
    # Initialize the optimizer using the final 'params' dictionary
    optimizer_class = None
    optimizer_name_upper = optimizer_name.upper()

    # Custom optimizers
    if optimizer_name_upper == "MILO":
        optimizer_class = milo
    elif optimizer_name_upper == "MILO_LW":
        optimizer_class = milo 
    elif optimizer_name_upper == "NOVOGRAD":
        optimizer_class = NovoGrad
    # Standard PyTorch optimizers (check explicitly by uppercase name)
    elif optimizer_name_upper == "ADAM":
        optimizer_class = torch.optim.Adam
    elif optimizer_name_upper == "ADAMW":
        optimizer_class = torch.optim.AdamW
    elif optimizer_name_upper == "SGD":
        optimizer_class = torch.optim.SGD
    elif optimizer_name_upper == "ADAGRAD":
        optimizer_class = torch.optim.Adagrad
    else:
        
        
        # Try getattr as a fallback for other potential torch optimizers
        try:
            # Attempt to get the optimizer class using the provided name (respecting case)
            optimizer_class = getattr(torch.optim, optimizer_name)
        except AttributeError:
             # If the original case fails, try the uppercase version
             try:
                 optimizer_class = getattr(torch.optim, optimizer_name_upper)
             except AttributeError:
                 # If both fail, raise the error
                 raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim or custom optimizers.")

    # Final check to ensure optimizer_class was assigned
    if optimizer_class is None:
         # This check should now pass for ADAGRAD
         raise ValueError(f"Optimizer class for '{optimizer_name}' could not be determined.")

    # Create the optimizer instance using the assigned class and parameters
    optimizer = optimizer_class(model.parameters(), **params)

    total_rewards = []
    trajectories = []
    episode_losses = [] # Loss per episode
    all_step_losses_run = [] # Loss per training step (iteration) for the entire run
    all_step_times_run = [] # Wall-clock time per training step for the entire run
    all_step_iterations_run = [] # Global iteration number per training step

    exploration_noise = INITIAL_EXPLORATION
    
    for episode in range(EPISODES):
        # Decay exploration noise
        exploration_noise = max(FINAL_EXPLORATION, INITIAL_EXPLORATION * (EXPLORATION_DECAY ** episode))
        
        # Pass start_time and global_step_counter
        reward, trajectory, episode_loss, step_losses, step_times, step_iterations, global_step_counter = train_episode(
            model, optimizer, env, GAMMA, exploration_noise, start_time, global_step_counter
        )

        total_rewards.append(reward)
        trajectories.append(trajectory)
        if not np.isnan(episode_loss):
             episode_losses.append(episode_loss)
        all_step_losses_run.extend(step_losses)
        all_step_times_run.extend(step_times)
        all_step_iterations_run.extend(step_iterations)

        # Log progress at regular intervals
        if episode % LOG_INTERVAL == 0 or episode == EPISODES-1:
            avg_reward = np.mean(total_rewards[-10:]) if len(total_rewards) >= 10 else np.mean(total_rewards)
            final_pos = trajectory[-1]
            goal_pos_final = env.goal_pos
            goal_reached = np.linalg.norm(final_pos - goal_pos_final) < env.distance_threshold
            success_status = "GOAL REACHED" if goal_reached else "goal not reached"
            print(f"Episode {episode}: Reward = {reward:.2f}, Avg Reward = {avg_reward:.2f}, Loss = {episode_loss:.6f}, Exploration = {exploration_noise:.4f}, {success_status}")
    
    env.close()
    
    # Stop resource tracking and collect info
    resource_info = None
    if resource_tracking_available:
        resource_tracker.stop()
        resource_info = resource_tracker.get_info()
        resource_info.update({
            "optimizer": optimizer_name,
            "run_index": run_idx,
            "episodes": EPISODES
        })
    
    # Return collected step data along with other results
    return model, total_rewards, trajectories[-1], episode_losses, all_step_losses_run, all_step_times_run, all_step_iterations_run, resource_info, settings

def eval_model_trials(model, num_trials=10):
    """Evaluate a trained model with multiple trials and plot trajectories on a single 3D plot.

    Args:
        model: The trained policy model to evaluate.
        num_trials: Number of evaluation trials to run.

    Returns:
        tuple: List of trajectories, success rate, list of rewards.
    """
    trajectories = []
    success_count = 0
    rewards = []
    
    print(f"Evaluation goal position: {GOAL_POS}")
    
    for trial in range(num_trials):
        env = PointNavigationEnv(start_pos=START_POS, goal_pos=GOAL_POS, random_goal=False, distance_threshold=DISTANCE_THRESHOLD, max_steps=MAX_STEPS)
        state, _ = env.reset()
        trajectory = [state[:3].copy()]
        done = False
        truncated = False
        trial_reward = 0
        steps = 0
        
        while not done and not truncated and steps < env.max_steps:
            with torch.no_grad():
                action = model(torch.FloatTensor(state))
            next_state, reward, done, truncated, info = env.step(action.numpy())
            trial_reward += reward
            trajectory.append(next_state[:3].copy())
            state = next_state
            steps += 1
            
            if info.get('goal_reached', False):
                success_count += 1
                print(f"Trial {trial+1}: Goal reached in {steps} steps!")
                break
        
        if steps >= env.max_steps or truncated:
            print(f"Trial {trial+1}: Max steps reached without finding goal")
        
        trajectory = np.array(trajectory)
        trajectories.append(trajectory)
        rewards.append(trial_reward)
    
    success_rate = success_count / num_trials
    print(f"Success rate: {success_rate:.2%} ({success_count}/{num_trials})")
    
    return trajectories, success_rate, rewards

if __name__ == '__main__':
    """Main execution block for RL experiments."""
    # --- Set Seeds for Reproducibility ---
    SEED = 42 # You can choose any integer
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        # Potentially add for cudNN determinism, though it can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    optimizer_names = OPTIMIZERS
    # Dictionaries to store data across runs
    all_runs_rewards = {opt: [] for opt in optimizer_names}
    all_runs_episode_losses = {opt: [] for opt in optimizer_names} # Renamed for clarity
    all_runs_step_losses = {opt: [] for opt in optimizer_names} # Store step losses per run
    all_runs_step_times = {opt: [] for opt in optimizer_names} # Store step times per run
    all_runs_step_iterations = {opt: [] for opt in optimizer_names} # Store step iterations per run
    all_eval_trajectories = {opt: [] for opt in optimizer_names}
    all_success_rates = {opt: [] for opt in optimizer_names}
    all_resource_info = []  # Store resource information
    all_settings = {opt: {} for opt in optimizer_names}  # Store experimental settings
    
    # Final run data for detailed plots
    final_run_data = {opt: {} for opt in optimizer_names}
    
    # Results for CSV
    results = []
    
    print("\nRunning reinforcement learning experiments...")
    
    for opt in optimizer_names:
        print(f"\nRunning {NUM_RUNS} experiments for {opt} optimizer...")
        # Temporary lists for the current optimizer's runs
        opt_run_rewards = []
        opt_run_episode_losses = []
        opt_run_step_losses = []
        opt_run_step_times = []
        opt_run_step_iterations = []
        opt_run_success_rates = []

        for run in range(NUM_RUNS):
            print(f"\nRun {run+1}/{NUM_RUNS} for {opt}...")
            # Get step data from run_experiment
            model, rewards, final_traj, episode_losses, step_losses, step_times, step_iterations, resource_info, settings = run_experiment(opt, run)
            
            # Evaluate the model with fixed common goal
            eval_trajectories, success_rate, _ = eval_model_trials(model, num_trials=5)
            
            # Store results for this run temporarily
            opt_run_rewards.append(rewards)
            opt_run_episode_losses.append(episode_losses)
            opt_run_step_losses.append(step_losses)
            opt_run_step_times.append(step_times)
            opt_run_step_iterations.append(step_iterations)
            all_eval_trajectories[opt].extend(eval_trajectories) # Keep extending eval trajectories
            opt_run_success_rates.append(success_rate)

            # Store resource info if available
            if resource_info is not None:
                all_resource_info.append(resource_info)
            
            # Store settings for first run
            if run == 0:
                all_settings[opt] = settings
            
            # Save the last run's data for detailed plotting (if needed)
            if run == NUM_RUNS - 1:
                final_run_data[opt]['rewards'] = rewards
                final_run_data[opt]['episode_losses'] = episode_losses
                final_run_data[opt]['step_losses'] = step_losses
                final_run_data[opt]['step_times'] = step_times
                final_run_data[opt]['eval_trajectories'] = eval_trajectories

        # Store all runs for the current optimizer
        all_runs_rewards[opt] = opt_run_rewards
        all_runs_episode_losses[opt] = opt_run_episode_losses
        all_runs_step_losses[opt] = opt_run_step_losses
        all_runs_step_times[opt] = opt_run_step_times
        all_runs_step_iterations[opt] = opt_run_step_iterations
        all_success_rates[opt] = opt_run_success_rates

    # --- Statistical Calculations --- 
    if stats_functions_available:
        # --- Calculate stats for episode rewards --- 
        avg_rewards_per_episode = {}
        std_err_rewards_per_episode = {}
        for opt in optimizer_names:
            # Ensure all runs have the same number of episodes (or handle potential differences)
            min_episodes_rewards = min(len(run) for run in all_runs_rewards[opt])
            # Stack rewards for episodes up to the minimum length
            reward_runs_array = np.array([run[:min_episodes_rewards] for run in all_runs_rewards[opt]])
            avg_rewards_per_episode[opt] = np.mean(reward_runs_array, axis=0)
            std_err_rewards_per_episode[opt] = stats.sem(reward_runs_array, axis=0) if NUM_RUNS > 1 else np.zeros(min_episodes_rewards)

        # --- Calculate stats for episode losses --- 
        avg_losses_per_episode = {}
        std_err_losses_per_episode = {}
        for opt in optimizer_names:
            # Ensure all runs have the same number of episodes
            min_episodes_losses = min(len(run) for run in all_runs_episode_losses[opt])
            # Stack losses for episodes up to the minimum length
            loss_runs_array = np.array([run[:min_episodes_losses] for run in all_runs_episode_losses[opt]])
            avg_losses_per_episode[opt] = np.mean(loss_runs_array, axis=0)
            std_err_losses_per_episode[opt] = stats.sem(loss_runs_array, axis=0) if NUM_RUNS > 1 else np.zeros(min_episodes_losses)

        # --- Calculate stats for step losses (iteration cost) --- 
        avg_losses_per_step = {}
        std_err_losses_per_step = {}
        avg_times_per_step = {} # Also calculate average time for walltime plot
        max_total_steps = 0
        for opt in optimizer_names:
             # Find the minimum number of total steps across runs for this optimizer
             min_steps = min(len(run) for run in all_runs_step_losses[opt])
             max_total_steps = max(max_total_steps, min_steps)

             # Stack step losses and times up to the minimum length
             step_loss_runs_array = np.array([run[:min_steps] for run in all_runs_step_losses[opt]])
             step_time_runs_array = np.array([run[:min_steps] for run in all_runs_step_times[opt]])

             avg_losses_per_step[opt] = np.mean(step_loss_runs_array, axis=0)
             avg_times_per_step[opt] = np.mean(step_time_runs_array, axis=0) # Average time at each step
             std_err_losses_per_step[opt] = stats.sem(step_loss_runs_array, axis=0) if NUM_RUNS > 1 else np.zeros(min_steps)

        # --- Calculate final success rate statistics --- 
        success_stats = {
            opt: {
                'mean': np.mean(all_success_rates[opt]),
                'std_dev': np.std(all_success_rates[opt], ddof=1) if len(all_success_rates[opt]) > 1 else 0,
                'std_err': stats.sem(all_success_rates[opt]) if len(all_success_rates[opt]) > 1 else 0
            } for opt in optimizer_names
        }
        
        # Perform statistical significance tests on success rates
        success_sig_tests = perform_statistical_tests(
            {opt: all_success_rates[opt] for opt in optimizer_names}, 
            'success_rate'
        )
        
        # Create statistics report data
        stats_data = {
            'success_rate': {
                'n_runs': NUM_RUNS,
                'final_values': success_stats,
                'significance_tests': success_sig_tests
            }
            # Add reward/loss stats if needed
        }
        
        # Save statistical report
        save_statistics_report(stats_data, stats_dir, "rl_experiment")
    else:
        # Fallback if stats functions not available: use last run data and zero error
        avg_rewards_per_episode = {opt: final_run_data[opt]['rewards'] for opt in optimizer_names}
        std_err_rewards_per_episode = {opt: np.zeros_like(final_run_data[opt]['rewards']) for opt in optimizer_names}
        avg_losses_per_episode = {opt: final_run_data[opt]['episode_losses'] for opt in optimizer_names}
        std_err_losses_per_episode = {opt: np.zeros_like(final_run_data[opt]['episode_losses']) for opt in optimizer_names}
        # Fallback for step data
        avg_losses_per_step = {opt: final_run_data[opt]['step_losses'] for opt in optimizer_names}
        std_err_losses_per_step = {opt: np.zeros_like(final_run_data[opt]['step_losses']) for opt in optimizer_names}
        avg_times_per_step = {opt: final_run_data[opt]['step_times'] for opt in optimizer_names}
        success_stats = {opt: {'std_err': 0} for opt in optimizer_names} # Default error for plotting
    
    # Save experimental settings
    if stats_functions_available:
        save_experimental_settings(all_settings, results_dir, "rl_experimental_settings")
    
    # Save resource info if available
    if resource_tracking_available and all_resource_info:
        resource_csv_path = os.path.join(results_dir, "compute_resources_rl.csv")
        save_resource_info(all_resource_info, resource_csv_path)
    
    # Calculate average success rates for plotting
    avg_success_rates = {opt: np.mean(all_success_rates[opt]) for opt in optimizer_names}
    
    # --- Plotting --- 
    print("\nGenerating plots...")

    # Plot rewards vs episode
    plot_training_rewards_with_error(
        avg_rewards_per_episode,
        std_err_rewards_per_episode,
        optimizer_names,
        save_path=os.path.join(visuals_dir, 'reward_comparison.png')
    )

    # Plot loss vs episode
    plot_training_losses_with_error( # This call should now work
        avg_losses_per_episode,
        std_err_losses_per_episode,
        optimizer_names,
        save_path=os.path.join(visuals_dir, 'loss_comparison_episode.png') # Renamed file
    )

    # Plot loss vs iteration (step)
    plot_iteration_cost(
        avg_losses_per_step,
        std_err_losses_per_step,
        optimizer_names,
        visuals_dir,
        filename="iteration_loss_comparison"
    )

    # Plot loss vs wall-clock time
    plot_walltime_cost(
        avg_losses_per_step, # Use step losses
        avg_times_per_step, # Use averaged step times
        std_err_losses_per_step, # Use step loss errors
        optimizer_names,
        visuals_dir,
        filename="walltime_loss_comparison"
    )

    # Plot success rates
    plot_success_rates(
        avg_success_rates, 
        optimizer_names, 
        5, # Assuming 5 eval trials
        {opt: success_stats[opt]['std_err'] for opt in optimizer_names},
        save_path=os.path.join(visuals_dir, 'success_rates.png')
    )
    
    # Plot trajectory visualizations (using final run eval trajectories for clarity)
    plot_averaged_trajectories_3d( # Corrected function call
        {opt: final_run_data[opt]['eval_trajectories'] for opt in optimizer_names}, 
        avg_success_rates, 
        optimizer_names, 
        START_POS, 
        GOAL_POS, 
        save_path=os.path.join(visuals_dir, 'averaged_trajectories_comparison.png')
    )
    
    plot_averaged_trajectories_topdown(
        {opt: final_run_data[opt]['eval_trajectories'] for opt in optimizer_names}, 
        avg_success_rates, 
        optimizer_names, 
        START_POS, 
        GOAL_POS, 
        save_path=os.path.join(visuals_dir, 'averaged_trajectories_topdown.png')
    )
    
    # Save raw run data for future analysis
    raw_runs_data = {
        'rewards': {opt: [list(run) for run in all_runs_rewards[opt]] for opt in optimizer_names},
        'losses': {opt: [list(run) for run in all_runs_episode_losses[opt]] for opt in optimizer_names},
        'step_losses': {opt: [list(run) for run in all_runs_step_losses[opt]] for opt in optimizer_names},
        'step_times': {opt: [list(run) for run in all_runs_step_times[opt]] for opt in optimizer_names},
        'success_rates': {opt: all_success_rates[opt] for opt in optimizer_names}
    }
    
    with open(os.path.join(results_dir, "raw_rl_runs_data.json"), 'w') as f:
        json.dump(raw_runs_data, f, default=lambda x: list(x) if isinstance(x, np.ndarray) else str(x))
    
    # Write final CSV with statistics
    csv_path = os.path.join(results_dir, CSV_FILENAME)
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        headers = ["Optimizer", "Avg Reward", "Success Rate", "Success Rate Std Err", "Avg Final Loss", "Compute Time (s)"]
        csv_writer.writerow(headers)
        
        for opt in optimizer_names:
            avg_reward = np.mean([np.mean(run) for run in all_runs_rewards[opt]])
            success_rate = avg_success_rates[opt]
            success_std_err = success_stats[opt]['std_err'] if stats_functions_available else 0
            avg_final_loss = np.mean([run[-1] for run in all_runs_episode_losses[opt]])
            
            # Get compute time if available
            compute_time = "-"
            if resource_tracking_available and all_resource_info:
                opt_resources = [r for r in all_resource_info if r.get('optimizer') == opt]
                if opt_resources:
                    compute_time = np.mean([r.get('duration_seconds', 0) for r in opt_resources])
                    
            csv_writer.writerow([opt, avg_reward, success_rate, success_std_err, avg_final_loss, compute_time])
    
    print(f"\nResults saved to {csv_path}")
    print(f"Plots saved to {visuals_dir}")
    print(f"Raw data saved to {results_dir}")

