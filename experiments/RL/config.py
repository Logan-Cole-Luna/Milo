"""
Configuration file for Reinforcement Learning (RL) experiments.

Purpose:
This file defines all settings for running RL experiments, specifically for tasks
like point navigation in a 3D environment. It covers agent hyperparameters,
environment parameters, optimizer choices, and configurations for potential
hyperparameter tuning of these optimizers in the RL context.

Options:
- LEARNING_RATE: Base learning rate for the optimizers.
- GAMMA: Discount factor for future rewards in RL.
- EPISODES: Total number of training episodes.
- LOG_INTERVAL: How often to log training progress.
- START_POS, GOAL_POS: Default start and goal positions for the agent.
- INITIAL_EXPLORATION, FINAL_EXPLORATION, EXPLORATION_DECAY: Parameters for
  epsilon-greedy exploration strategy.
- GRADIENT_CLIP: Maximum norm for gradient clipping to prevent exploding gradients.
- DISTANCE_THRESHOLD: Proximity to goal to consider an episode successful.
- RANDOM_GOAL: Boolean, if True, the goal position is randomized at each reset.
- MAX_STEPS: Maximum number of steps allowed per episode.
- NUM_RUNS: Number of independent training runs for each optimizer configuration
  to ensure statistical robustness of results.
- CSV_FILENAME: Name of the CSV file where experiment results will be saved.
- OPTIMIZERS: A list of optimizer names (e.g., "MILO", "SGD", "ADAMW") to be used
  for training the RL agent's policy or value network.
- PARAM_GRID: A dictionary defining the search space (ranges or specific values) for
  hyperparameters of each optimizer, intended for use with a hyperparameter tuning
  process (currently, most entries are commented out, implying fixed defaults are used).
- OPTIMIZER_PARAMS: Default hyperparameter settings for each optimizer. These are used
  if hyperparameter tuning is not active or if specific parameters are not included
  in the PARAM_GRID for tuning.

Experiments:
This configuration supports training an RL agent, likely using a policy gradient or
Q-learning based method, to learn a task such as navigating from a starting point to a
target in a simulated 3D environment. Different optimizers are employed to update the
parameters of the agent's neural network. The performance of these optimizers is then
compared based on metrics like cumulative reward over episodes, success rate in reaching
the goal, and convergence speed. The setup also allows for (though not fully enabled
in the provided snippet) hyperparameter optimization to find the best settings for each
optimizer within the RL task.
"""

import numpy as np

# Hyperparameters
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPISODES = 500
LOG_INTERVAL = EPISODES * 0.25
START_POS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
GOAL_POS = np.array([1, 1, 1], dtype=np.float32)
INITIAL_EXPLORATION = 0.2
FINAL_EXPLORATION = 0.01
EXPLORATION_DECAY = 0.998
GRADIENT_CLIP = 0.5
DISTANCE_THRESHOLD = 0.05
RANDOM_GOAL = False 
MAX_STEPS = 500
NUM_RUNS = 5
CSV_FILENAME = "rl_results.csv"

# --- Optimizers to Use ---
OPTIMIZERS = ["MILO", "MILO_LW", "SGD", "ADAMW", "ADAGRAD", "NOVOGRAD"]

# --- Parameter Grids for Hyperparameter Tuning ---
PARAM_GRID = {
    'SGD': {
        #'lr': (0.005, 0.1, 'log'),  # Log-uniform range for learning rate
        #'momentum': (0.45, 0.99),    # Uniform range for momentum (often high for SGD)
        #'weight_decay': (0.0005, 0.01, 'log'), # Log-uniform range for weight decay
    },
    'ADAGRAD': { # Added ADAGRAD tuning grid
        #'lr': (0.005, 0.1, 'log'),
        #'lr_decay': (0.0, 0.1),       # Uniform range for lr_decay
        #'weight_decay': (0.0005, 0.01, 'log'), # Log-uniform range for weight decay
        #'eps': (1e-10, 1e-6, 'log'),  # Log-uniform range for epsilon
    },
    'ADAMW': {
        #'lr': (0.005, 0.1, 'log'),
        #'weight_decay': (0.0005, 0.01, 'log'),
        #'betas': [(0.9, 0.98), (0.95, 0.99)],
        #'eps': (1e-9, 1e-6, 'log'),
    },
    'NOVOGRAD': {
        #'lr': (0.005, 0.1, 'log'),
        #'betas': [(0.9, 0.98), (0.95, 0.99)],
        #'weight_decay': (0.0005, 0.01, 'log'),
        #'grad_averaging': [False, True],
    },
    'MILO': {
        #'lr': (0.005, 0.1),                    
        #'weight_decay': (1e-6, 1e-2, 'log'),            
        #'momentum': (0.0, 0.99)
    },
    'MILO_LW': {
        #'lr': (0.005, 0.1),                     
        #'weight_decay': (1e-6, 1e-2, 'log'),            
        #'momentum': (0.0, 0.99)
    }
}

# --- Optimizer Parameter Settings (Unified Base) ---
OPTIMIZER_PARAMS = {
    "SGD": {"momentum": 0.9, "nesterov": True, "weight_decay": 0.0001}, # Added default weight_decay
    "ADAGRAD": {"lr_decay": 0, "weight_decay": 0.0, "eps": 1e-8}, # Added ADAGRAD defaults
    "ADAMW": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}, # Default weight_decay for AdamW
    "MILO": {
        #"lr": 0.05,
        "normalize": True,
        "layer_wise": False,
        "scale_aware": True,
        "scale_factor": 0.2,
        "nesterov": False,
        "adaptive": True,
        "momentum": 0.9,
        #"weight_decay": 0.001,
        "profile_time": False,
        'max_group_size': None,
        "use_cached_mapping": False,
        "foreach": True
    },
    "MILO_LW": {
        #"lr": 0.05,
        "normalize": True,
        "layer_wise": True,
        "scale_aware": True,
        "scale_factor": 0.2,
        "nesterov": False,
        "adaptive": True,
        "momentum": 0.9,
        #"weight_decay": 0.001,
        "profile_time": False,
        'max_group_size': None,
        "use_cached_mapping": True,
        "foreach": True
    },
    "NOVOGRAD": {
        "betas": (0.9, 0.99),
        "weight_decay": 0.001,
        "grad_averaging": True
    }
}
