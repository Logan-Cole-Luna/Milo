"""
Unified configuration for supervised learning experiments (e.g., image classification).

Purpose:
This file defines all settings for running supervised learning tasks to compare various
optimization algorithms. It covers model selection, dataset choices, training parameters,
hyperparameter tuning setups, and output directory specifications.

Options:
- EXPERIMENTS: List of experiment setups to run (e.g., "LOGISTIC", "RESNET18").
- PERFORM_HYPERPARAMETER_TUNING: Boolean to enable/disable hyperparameter search.
- BATCH_SIZE, EPOCHS, RUNS_PER_OPTIMIZER: Standard training controls.
- TRIALS, VAL_SPLIT_RATIO, TEST_SPLIT_RATIO: Parameters for tuning and data splitting.
- RESULTS_DIR_TUNING, VISUALS_DIR_TUNING, etc.: Output directory names.
- LR: Base learning rates for different experiment types.
- OPTIMIZERS: List of optimizer names to evaluate (e.g., "MILO", "SGD", "ADAMW").
- PARAM_GRID: Search space for hyperparameter tuning for each optimizer.
- OPTIMIZER_PARAMS: Default parameters for optimizers (used if tuning is off).
- SCHEDULER_PARAMS: Configuration for learning rate schedulers.
- EXPERIMENT_CONFIGS: Detailed dictionary for each experiment type, specifying model,
  dataset, model arguments, data transformations, and plot titles.

Experiments:
Facilitates image classification tasks (e.g., MNIST, CIFAR10/100) using various neural
network models (Logistic Regression, MLP, DeepCNN, ResNet18). The primary goal is to
benchmark the performance of different optimizers, with an optional automated
hyperparameter tuning phase to find optimal settings for each.
"""
import os
from torchvision import transforms

# --- Experiment Settings ---
# Options: ["LOGISTIC", "MULTILAYER", "DEEPCNN", "RESNET18"]
EXPERIMENTS = [
    "RESNET18"]

PERFORM_HYPERPARAMETER_TUNING = True
BATCH_SIZE = 128
EPOCHS = 10
RUNS_PER_OPTIMIZER = 5

TRIALS = 5 # Hyperparameter tuning trials
VAL_SPLIT_RATIO = 0.15 
TEST_SPLIT_RATIO = 0.10 

# --- Directory Names ---
RESULTS_DIR_TUNING = "results"
VISUALS_DIR_TUNING = "visuals"
RESULTS_DIR_NO_TUNING = "results_nt"
VISUALS_DIR_NO_TUNING = "visuals_nt"

# --- Base Learning Rate ---
LR = {
    "LOGISTIC": 0.05,
    "MULTILAYER": 0.05,
    "DEEPCNN": 0.005,
    "RESNET18": 0.005,
}

# --- Optimizers to Use ---
OPTIMIZERS = ["MILO", "MILO_LW", "SGD", "ADAMW", "ADAGRAD", "NOVOGRAD"]

# --- Parameter Grids for Hyperparameter Tuning ---
PARAM_GRID = {
    'SGD': {
        'lr': (0.005, 0.1, 'log'),  # Log-uniform range for learning rate
        'momentum': (0.45, 0.99),    # Uniform range for momentum (often high for SGD)
        'weight_decay': (0.0005, 0.01, 'log'), # Log-uniform range for weight decay
    },
    'ADAGRAD': {
        'lr': (0.005, 0.1, 'log'),
        'lr_decay': (0.0, 0.1),       # Uniform range for lr_decay
        'weight_decay': (0.0005, 0.01, 'log'), # Log-uniform range for weight decay
        'eps': (1e-10, 1e-6, 'log'),  # Log-uniform range for epsilon
    },
    'ADAMW': {
        'lr': (0.005, 0.1, 'log'),
        'weight_decay': (0.0005, 0.01, 'log'),
        'betas': [(0.9, 0.98), (0.95, 0.99)],
        'eps': (1e-9, 1e-6, 'log'),
    },
    'NOVOGRAD': {
        'lr': (0.005, 0.1, 'log'),
        'betas': [(0.9, 0.98), (0.95, 0.99)],
        'weight_decay': (0.0005, 0.01, 'log'),
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
    "SGD": {"momentum": 0.9, "nesterov": True, "weight_decay": 0.0001}, 
    "ADAGRAD": {"lr_decay": 0, "weight_decay": 0.0, "eps": 1e-10},
    "ADAMW": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}, 
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

# --- Scheduler Parameters (Unified) ---
# Using no scheduler for comparative experiments to avoid bias
SCHEDULER_PARAMS = {
    opt: {"scheduler": "None", "params": {}} for opt in OPTIMIZERS
}



# --- Experiment Configurations ---
EXPERIMENT_CONFIGS = {
    "LOGISTIC": {
        "model_name": "LogisticRegressionModel",
        "dataset_name": "MNIST",
        "model_args": {},
        "transform": transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.view(-1))
        ]),
        "plot_titles": {
            "loss": "LogReg: Loss vs. Epoch",
            "accuracy": "LogReg: Accuracy vs. Epoch",
            "f1": "LogReg: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "MULTILAYER": {
        "model_name": "MLP",
        "dataset_name": "MNIST",
        "model_args": {},
        "transform": transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.view(-1))
        ]),
        "plot_titles": {
            "loss": "MLP: Loss vs. Epoch",
            "accuracy": "MLP: Accuracy vs. Epoch",
            "f1": "MLP: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "DEEPCNN": {
        "model_name": "DeepCNN",
        "dataset_name": "CIFAR10",
        "model_args": {},
        "transform": transforms.Compose([
            transforms.ToTensor()  # Removed random augmentations for consistency
        ]),
        "plot_titles": {
            "loss": "Deep CNN: Loss vs. Epoch",
            "accuracy": "Deep CNN: Accuracy vs. Epoch",
            "f1": "Deep CNN: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    "RESNET18": { # Updated ResNet18 configuration
        "model_name": "ResNet18",
        "dataset_name": "CIFAR100", # Changed dataset to CIFAR100
        "model_args": {"num_classes": 100}, # Updated num_classes for CIFAR100
        "transform": transforms.Compose([
            transforms.ToTensor() # Use the same simple transform
        ]),
        "plot_titles": {
            "loss": "ResNet18 (CIFAR100): Loss vs. Epoch", # Updated title
            "accuracy": "ResNet18 (CIFAR100): Accuracy vs. Epoch", # Updated title
            "f1": "ResNet18 (CIFAR100): F1 Score vs. Epoch" # Updated title
        },
        "cost_xlimit": None
    }
}

