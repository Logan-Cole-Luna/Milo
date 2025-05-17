import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import sys
import optuna  # Import Optuna
import numpy as np # Ensure numpy is imported
# Add imports needed for RL evaluation
import gymnasium as gym 
# Assuming select_action and train_episode logic might be needed or adapted
# If those are complex, might need to import them or replicate simplified versions.
# For now, let's assume a simplified RL training/evaluation loop within the function.

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from milo import milo
from novograd import NovoGrad
from torch.utils.data import random_split, DataLoader

# --- Add RL Evaluation Function ---
def evaluate_rl_agent(model, env_fn, device, num_episodes=3, max_steps_per_episode=100):
    """Evaluates an RL agent for a few episodes and returns the average reward.
    
    Args:
        model: The RL agent (policy model) to evaluate.
        env_fn: A function that returns a new instance of the RL environment.
        device: The device to run the model on (e.g., 'cpu', 'cuda').
        num_episodes: Number of episodes to run for evaluation.
        max_steps_per_episode: Maximum number of steps allowed per episode.
        
    Returns:
        float: The average reward obtained over the evaluation episodes.
    """
    total_rewards = []
    env = env_fn() # Create a fresh environment instance

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        while not done and not truncated and steps < max_steps_per_episode:
            state_tensor = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                action = model(state_tensor) # Get action from policy (no exploration noise during eval)
            
            action_np = action.cpu().numpy() 
            
            next_state, reward, done, truncated, _ = env.step(action_np)
            episode_reward += reward
            state = next_state
            steps += 1
        total_rewards.append(episode_reward)
    
    env.close()
    return np.mean(total_rewards) if total_rewards else 0.0


# --- Optuna Objective Function ---
def objective(trial, model_fn, optimizer_name, param_grid, 
              # SL args
              train_loader, val_loader, criterion, 
              # RL args
              env_fn, gamma, clip_grad, episodes_to_eval, max_steps_per_episode,
              # Common args
              device, task_type, epochs): # 'epochs' will mean episodes_to_train for RL
    """Objective function for Optuna optimization.
    
    This function is called by Optuna for each trial to evaluate a set of hyperparameters.
    It configures the model and optimizer, trains it (for SL or RL), and returns a metric
    that Optuna will try to maximize.
    
    Args:
        trial: An Optuna `Trial` object used to suggest hyperparameter values.
        model_fn: A function that returns a new instance of the model to be trained.
        optimizer_name: The name of the optimizer to use (e.g., "Adam", "MILO").
        param_grid: A dictionary defining the search space for hyperparameters of the optimizer.
        train_loader: DataLoader for training data (for SL tasks).
        val_loader: DataLoader for validation data (for SL tasks).
        criterion: The loss function (for SL tasks).
        env_fn: A function that returns a new instance of the RL environment (for RL tasks).
        gamma: Discount factor for RL.
        clip_grad: Gradient clipping value for RL training.
        episodes_to_eval: Number of episodes to use for evaluation within an RL trial.
        max_steps_per_episode: Maximum steps per episode for RL evaluation within a trial.
        device: The device to run training on.
        task_type: The type of task ('classification', 'regression', 'rl').
        epochs: Number of training epochs (for SL) or training episodes (for RL) for the trial.
        
    Returns:
        float: The metric to be maximized by Optuna (e.g., validation accuracy, mean reward).
    """
    model = model_fn().to(device)

    params = {}
    for param_name, values in param_grid.items():
        if isinstance(values, list):  
            # Special handling for betas: Optuna suggests the tuple directly
            if param_name == 'betas':
                # Directly use the tuple suggested by Optuna
                params[param_name] = trial.suggest_categorical(param_name, values)
            else:
                # Process other categorical values, ensuring they are basic types
                processed_values = []
                for v in values:
                    if isinstance(v, (int, float, str, bool, type(None))):
                        processed_values.append(v)
                    else:
                        # If not a basic type, convert to string for Optuna compatibility
                        print(f"Warning: Non-standard type {type(v)} for {param_name}. Converting to string.")
                        processed_values.append(str(v))
                params[param_name] = trial.suggest_categorical(param_name, processed_values)
        elif isinstance(values, tuple) and len(values) == 2:  
            params[param_name] = trial.suggest_float(param_name, values[0], values[1])
        elif isinstance(values, tuple) and len(values) == 3 and values[2] == 'log': 
            params[param_name] = trial.suggest_float(param_name, values[0], values[1], log=True)
        else:
            print(f"Warning: Unsupported parameter format for {param_name}: {values}. Skipping suggestion.")

    optimizer_class = None
    optimizer_name_upper = optimizer_name.upper()
    if optimizer_name_upper == "MILO":
        optimizer_class = milo
    elif optimizer_name_upper == "MILO_LW":
        optimizer_class = milo
    elif optimizer_name_upper == "NOVOGRAD":
        optimizer_class = NovoGrad
    else:
        try:
            if optimizer_name_upper == "SGD":
                optimizer_class = torch.optim.SGD
            elif optimizer_name_upper == "ADAM":
                optimizer_class = torch.optim.Adam
            elif optimizer_name_upper == "ADAMW":
                optimizer_class = torch.optim.AdamW
            elif optimizer_name_upper == "ADAGRAD":
                optimizer_class = torch.optim.Adagrad
            else:
                optimizer_class = getattr(torch.optim, optimizer_name, None)

            if optimizer_class is None:
                raise AttributeError
        except AttributeError:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim or custom optimizers.")

    try:
        optimizer = optimizer_class(model.parameters(), **params)
    except TypeError as e:
        print(f"Error creating optimizer {optimizer_name} with params {params}: {e}")
        raise optuna.exceptions.TrialPruned() # Prune trial if optimizer creation fails

    # Train and evaluate based on task type
    if task_type == "classification" or task_type == "regression":
        # Use existing supervised learning evaluation
        if not criterion: # Make sure criterion is passed for SL tasks
             raise ValueError("Criterion must be provided for classification/regression tasks.")
        metric = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, device, task_type, epochs)
    elif task_type == "rl":
        # Use the new RL evaluation function
        if not env_fn:
             raise ValueError("env_fn must be provided for RL tasks.")
        # Use 'epochs' argument as the number of training episodes for the trial
        episodes_to_train = epochs 
        metric = train_and_evaluate_rl(model, optimizer, env_fn, device, gamma, clip_grad, 
                                       episodes_to_train, episodes_to_eval, max_steps_per_episode)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Handle NaN or Inf results from evaluation to prune trial
    if np.isnan(metric) or np.isinf(metric):
        print(f"Warning: Trial resulted in NaN or Inf metric ({metric}). Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    return metric # Optuna maximizes this value (average reward for RL)


def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, device, task_type, epochs=10):
    """Train model and evaluate on validation data (for SL tasks).
    
    For classification, it returns the mean validation accuracy over all epochs.
    For regression, it returns the negative mean squared error of the last epoch.
    
    Args:
        model: The model to train and evaluate.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        optimizer: The optimizer to use for training.
        criterion: The loss function.
        device: The device to run training on.
        task_type: 'classification' or 'regression'.
        epochs: Number of epochs to train for.
        
    Returns:
        float: Mean validation accuracy (for classification) or negative MSE (for regression).
    """
    model.train()
    if task_type == "classification":
        acc_history = [] 
        for epoch in range(epochs):
            # Training phase
            model.train()
            for inputs, labels in train_loader:
                if isinstance(inputs, np.ndarray): inputs = torch.from_numpy(inputs)
                if isinstance(labels, np.ndarray): labels = torch.from_numpy(labels)
                inputs = inputs.clone().detach().float().to(device)
                labels = labels.clone().detach().long().to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation phase (run every epoch, accumulate results)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if isinstance(inputs, np.ndarray): inputs = torch.from_numpy(inputs)
                    if isinstance(labels, np.ndarray): labels = torch.from_numpy(labels)
                    inputs = inputs.clone().detach().float().to(device)
                    labels = labels.clone().detach().long().to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            # Calculate and store accuracy on validation set
            acc = correct / total if total > 0 else 0
            acc_history.append(acc)

        # Return mean validation accuracy over all epochs (area under accuracy curve)
        return float(np.mean(acc_history))
    elif task_type == "regression":
        final_avg_mse = float('inf') # Store MSE of the last epoch

        for epoch in range(epochs):
            # Training phase
            model.train()
            for inputs, labels in train_loader:
                if isinstance(inputs, np.ndarray): inputs = torch.from_numpy(inputs)
                if isinstance(labels, np.ndarray): labels = torch.from_numpy(labels)
                inputs = inputs.clone().detach().float().to(device)
                labels = labels.clone().detach().float().to(device) # Regression uses float labels
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation phase (run every epoch, but only return the last one)
            model.eval()
            total_mse = 0.0
            num_batches = 0
            # Check if val_loader is usable (not None and not empty)
            if val_loader and len(val_loader) > 0:
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        if isinstance(inputs, np.ndarray): inputs = torch.from_numpy(inputs)
                        if isinstance(labels, np.ndarray): labels = torch.from_numpy(labels)
                        inputs = inputs.clone().detach().float().to(device)
                        labels = labels.clone().detach().float().to(device) # Regression uses float labels
                        outputs = model(inputs)
                        total_mse += nn.functional.mse_loss(outputs, labels).item()
                        num_batches += 1
            
            # Handle case where validation couldn't run (empty loader)
            if num_batches > 0:
                final_avg_mse = total_mse / num_batches
            else:
                print("Warning: No validation batches found. Returning 0 for objective.")
                return 0.0

        # Return negative mean MSE of the *last* epoch if validation occurred
        return float(-final_avg_mse) 
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def train_and_evaluate_rl(model, optimizer, env_fn, device, gamma, clip_grad, 
                          episodes_to_train, episodes_to_eval, max_steps_per_episode):
    """
    Trains and evaluates an RL agent for hyperparameter tuning.
    Trains for 'episodes_to_train' episodes, then evaluates average reward 
    over 'episodes_to_eval' deterministic episodes.
    
    Args:
        model: The RL policy model.
        optimizer: The optimizer for the policy model.
        env_fn: Function to create a new environment instance.
        device: Device to run training on.
        gamma: Discount factor for calculating returns.
        clip_grad: Value for gradient clipping.
        episodes_to_train: Number of episodes to train the agent for in this trial.
        episodes_to_eval: Number of episodes to evaluate the trained agent on.
        max_steps_per_episode: Maximum steps per episode during training and evaluation.
        
    Returns:
        float: The average reward obtained during the evaluation phase.
    """
    model.to(device)
    total_rewards_eval = []

    # --- Training Phase ---
    model.train()
    env_train = env_fn() # Create environment for training
    exploration_noise = 0.1 # Use a fixed moderate exploration for tuning training

    for episode in range(episodes_to_train):
        state, _ = env_train.reset()
        done = False
        truncated = False
        episode_steps = 0
        rewards_train = []
        states_train = []
        actions_train = []

        while not done and not truncated and episode_steps < max_steps_per_episode:
            state_tensor = torch.FloatTensor(state).to(device)
            states_train.append(state)
            
            with torch.no_grad():
                action_mean = model(state_tensor)
            
            action_noise_tensor = torch.normal(mean=0.0, std=exploration_noise, size=action_mean.shape).to(device)
            action = action_mean + action_noise_tensor
            actions_train.append(action)

            next_state, reward, done, truncated, _ = env_train.step(action.cpu().detach().numpy())
            rewards_train.append(reward)
            state = next_state
            episode_steps += 1

        # --- Policy Update ---
        if not states_train: # Skip if episode ended immediately
             continue

        states_tensor = torch.FloatTensor(np.array(states_train)).to(device)
        actions_tensor = torch.stack(actions_train)
        
        discounted_rewards = []
        R = 0
        for r in reversed(rewards_train):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards_tensor = torch.FloatTensor(discounted_rewards).to(device)
        if len(discounted_rewards_tensor) > 1:
             discounted_rewards_tensor = (discounted_rewards_tensor - discounted_rewards_tensor.mean()) / (discounted_rewards_tensor.std() + 1e-9)

        optimizer.zero_grad()
        predicted_actions = model(states_tensor)
        losses = torch.sum((actions_tensor - predicted_actions)**2, dim=1)
        weighted_losses = losses * discounted_rewards_tensor
        loss = weighted_losses.mean()
        
        if torch.isnan(loss):
            print("Warning: NaN loss detected during tuning training. Skipping update.")
            continue 

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
    env_train.close() 

    # --- Evaluation Phase ---
    model.eval()
    env_eval = env_fn() # Create environment for evaluation

    for _ in range(episodes_to_eval):
        state, _ = env_eval.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        while not done and not truncated and steps < max_steps_per_episode:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = model(state_tensor) # Deterministic action
            next_state, reward, done, truncated, _ = env_eval.step(action.cpu().numpy())
            episode_reward += reward
            state = next_state
            steps += 1
        total_rewards_eval.append(episode_reward)
        
    env_eval.close() # Close the evaluation environment

    # Return the average reward from the evaluation phase
    avg_reward = np.mean(total_rewards_eval) if total_rewards_eval else -float('inf')
    # Handle potential NaN/Inf from evaluation if something went wrong
    if np.isnan(avg_reward) or np.isinf(avg_reward):
        print(f"Warning: Invalid average reward ({avg_reward}) during evaluation. Returning -inf.")
        return -float('inf')
        
    return float(avg_reward)


# --- Main Tuning Function ---
def tune_hyperparameters(model_fn, optimizer_names, param_grid, device,
                         experiment_name="Experiment", task_type="classification",
                         epochs=5, num_trials=5, 
                         # SL specific args
                         train_loader=None, val_ratio=0.2, criterion=None,
                         # RL specific args
                         env_fn=None, gamma=0.99, clip_grad=1.0, 
                         episodes_to_eval=5, max_steps_per_episode=500):
    """
    Tune hyperparameters for multiple optimizers using Optuna.
    Handles supervised learning (classification/regression) and RL tasks.
    
    Args:
        model_fn: Function that creates a new model instance.
        optimizer_names: List of optimizer names to tune.
        param_grid: Dictionary of hyperparameter grids for each optimizer.
        device: Device to train on ('cpu' or 'cuda').
        experiment_name: Name for saving hyperparameters.
        task_type: 'classification', 'regression', or 'rl'.
        epochs: Number of epochs (SL) or training episodes per trial (RL).
        num_trials: Number of Optuna trials per optimizer.
        
        # SL Args
        train_loader: DataLoader for training data (required for SL).
        val_ratio: Fraction of training data for validation (SL only).
        criterion: Loss function (e.g., nn.CrossEntropyLoss) (required for SL).

        # RL Args
        env_fn: Function that creates a new environment instance (required for RL).
        gamma: Discount factor for RL.
        clip_grad: Gradient clipping value for RL training within objective.
        episodes_to_eval: How many episodes to run for evaluation in each trial (RL).
        max_steps_per_episode: Max steps per episode during RL tuning evaluation.

    Returns:
        Dictionary of best hyperparameters for each optimizer.
    """
    hyperparams_dir = os.path.join(os.path.dirname(__file__), 'hyperparameters')
    os.makedirs(hyperparams_dir, exist_ok=True)

    best_hyperparams = {}
    train_subset_loader = None
    val_subset_loader = None

    # Setup based on task type
    if task_type in ["classification", "regression"]:
        if not isinstance(train_loader, DataLoader):
             raise TypeError(f"train_loader must be a PyTorch DataLoader for task_type '{task_type}'.")
        if not criterion:
             raise ValueError(f"criterion must be provided for task_type '{task_type}'.")
             
        # Create validation split for SL tasks
        train_dataset = train_loader.dataset
        num_train_total = len(train_dataset)
        num_val = int(num_train_total * val_ratio)
        num_train = num_train_total - num_val
        
        if num_train <= 0 or num_val <= 0:
             raise ValueError("val_ratio results in zero samples for train or validation set.")

        train_subset, val_subset = random_split(
            train_dataset, [num_train, num_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_subset_loader = DataLoader(
            train_subset, batch_size=train_loader.batch_size, shuffle=True, 
            num_workers=getattr(train_loader, 'num_workers', 0)
        )
        val_subset_loader = DataLoader(
            val_subset, batch_size=train_loader.batch_size, shuffle=False,
            num_workers=getattr(train_loader, 'num_workers', 0)
        )
    elif task_type == "rl":
        if not env_fn:
            raise ValueError("env_fn must be provided for task_type 'rl'.")
        print("RL task type detected. Skipping DataLoader setup.")
        # train_loader and val_loader remain None, criterion is not needed
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    for optimizer_name in optimizer_names:
        if optimizer_name not in param_grid or not param_grid[optimizer_name]: # Check if grid exists and is not empty
            print(f"Skipping hyperparameter tuning for {optimizer_name} as it's not in param_grid or grid is empty.")
            best_hyperparams[optimizer_name] = {}
            continue

        # Check if hyperparameters already exist
        hyperparam_file = os.path.join(hyperparams_dir, f"{experiment_name}_{optimizer_name}_hyperparameters.json")
        if os.path.exists(hyperparam_file):
            try:
                with open(hyperparam_file, 'r') as f:
                    loaded_params = json.load(f)
                    # Basic check if loaded params look valid (e.g., is a dict)
                    if isinstance(loaded_params, dict):
                         print(f"Loading existing hyperparameters for {optimizer_name} from {hyperparam_file}")
                         best_hyperparams[optimizer_name] = loaded_params
                         continue 
                    else:
                         print(f"Warning: Invalid format in {hyperparam_file}. Re-tuning.")
            except json.JSONDecodeError:
                 print(f"Warning: Could not decode JSON from {hyperparam_file}. Re-tuning.")


        print(f"Tuning hyperparameters for {optimizer_name}...")
        study = optuna.create_study(direction="maximize") # Maximize reward (RL) or accuracy (SL classification), or -MSE (SL regression)

        # Define the objective function for Optuna, passing necessary args
        obj_func = lambda trial: objective(
            trial=trial,
            model_fn=model_fn,
            optimizer_name=optimizer_name,
            param_grid=param_grid[optimizer_name],
            # SL args (passed even if None for RL, objective handles it)
            train_loader=train_subset_loader, 
            val_loader=val_subset_loader, 
            criterion=criterion,
            # RL args (passed even if None for SL, objective handles it)
            env_fn=env_fn, 
            gamma=gamma, 
            clip_grad=clip_grad,
            episodes_to_eval=episodes_to_eval,
            max_steps_per_episode=max_steps_per_episode,
            device=device,
            task_type=task_type,
            epochs=epochs # Represents training episodes per trial for RL
        )

        try:
            study.optimize(obj_func, n_trials=num_trials)
        except optuna.exceptions.TrialPruned as e:
             print(f"A trial was pruned during optimization for {optimizer_name}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during Optuna optimization for {optimizer_name}: {e}")
             best_hyperparams[optimizer_name] = {}
             print(f"Skipping saving hyperparameters for {optimizer_name} due to optimization error.")
             continue 

        # Check if any trials completed successfully
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
             print(f"Warning: No trials completed successfully for {optimizer_name}. Cannot determine best parameters.")
             best_hyperparams[optimizer_name] = {}
        else:
             best_params = study.best_trial.params
             best_value = study.best_trial.value
             print(f"Best {optimizer_name} Trial Value: {best_value:.4f}, Best Params: {best_params}")
             best_hyperparams[optimizer_name] = best_params

             # Save the best hyperparameters
             try:
                 with open(hyperparam_file, 'w') as f:
                     json.dump(best_params, f, indent=4) 
                     print(f"Saved hyperparameters for {optimizer_name} to {hyperparam_file}")
             except TypeError as e:
                  print(f"Error saving hyperparameters for {optimizer_name} to JSON: {e}. Params: {best_params}")
             except IOError as e:
                  print(f"Error writing hyperparameters file {hyperparam_file}: {e}")


    return best_hyperparams
