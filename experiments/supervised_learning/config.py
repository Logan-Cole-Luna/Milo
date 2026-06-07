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

# Recommended experiment subsets for different purposes:
# Option A: Current suite (CNN only)
#FINAL_SUITE = ["LOGISTIC", "MULTILAYER", "RESNET34_CIFAR10", "RESNET34_CIFAR100", "VGG11_CIFAR10", "VGG11_CIFAR100"]

# Option C: Comprehensive suite (includes Vision Transformers)
FINAL_SUITE = [
    "LOGISTIC", "MULTILAYER",
    "RESNET34_CIFAR10", "RESNET34_CIFAR100",
    "VGG11_CIFAR10", "VGG11_CIFAR100",
    "VIT_TINY_CIFAR10", "VIT_TINY_CIFAR100"
]


EXPERIMENTS = [FINAL_SUITE]

# --- Optimizers to Use ---
OPTIMIZERS = ["MILO", "MILO_LW", "SGD", "ADAMW", "ADAGRAD", "ADEMAMIX", "SOAP"]


PERFORM_HYPERPARAMETER_TUNING = False
BATCH_SIZE = 128
EPOCHS = 5
RUNS_PER_OPTIMIZER = 5  # Updated from 1 to 5 for statistical validity

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
    "ADVANCED_MLP": 0.01,
    "WIDE_MLP": 0.01,
    "DEEPCNN": 0.005,
    "MODERN_CNN": 0.001,
    "ATTENTION_CNN": 0.001,
    "HYBRID_CNN_TRANSFORMER": 0.0005,
    "WIDE_RESNET": 0.001,
    "SIMPLE_VIT": 0.0005,
    "RESNET18": 0.005,
    "RESNET34_CIFAR10": 0.001,
    "RESNET34_CIFAR100": 0.001,
    "VGG11_CIFAR10": 0.001,
    "VGG11_CIFAR100": 0.001,
    "VIT_TINY_CIFAR10": 0.0005,
    "VIT_TINY_CIFAR100": 0.0005,
}



# "MILO_TUNED", "MILO_LW_TUNED",
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
    'ADALAYER': {
        'lr': (0.001, 0.1, 'log'),
        'betas': [(0.9, 0.98), (0.95, 0.999)],
        'weight_decay': (1e-5, 1e-2, 'log'),
    },
    'ADAM_MINI': {
        'lr': (0.001, 0.1, 'log'),
        'betas': [(0.9, 0.98), (0.95, 0.999)],
        'weight_decay': (1e-5, 1e-2, 'log'),
    },
    'MUON': {
        'lr': (0.001, 0.1, 'log'),
        'betas': [(0.9, 0.98), (0.95, 0.999)],
        'weight_decay': (1e-5, 1e-2, 'log'),
    },
    'SOAP': {
        'lr': (0.001, 0.01, 'log'),
        'betas': [(0.9, 0.95), (0.95, 0.98)],
        'weight_decay': (1e-4, 1e-2, 'log'),
        'precondition_frequency': [5, 10, 20],
        'eps': (1e-10, 1e-6, 'log'),
    },
    'ADEMAMIX': {
        'lr': (0.0001, 0.01, 'log'),
        'betas': [(0.9, 0.999, 0.9999), (0.95, 0.999, 0.9999)],
        'alpha': (2.0, 10.0),
        'weight_decay': (1e-4, 1e-1, 'log'),
        'eps': (1e-10, 1e-6, 'log'),
    },
    'MILO': {
        'lr': (0.005, 0.1, 'log'),  # Learning rate with log-uniform distribution
        'momentum': (0.8, 0.95),    # Momentum for gradient averaging
        'eps': (1e-8, 1e-5, 'log'), # Numerical stability epsilon
        'adaptive_eps': (1e-10, 1e-6, 'log'), # Adaptive scaling epsilon
        'weight_decay': (1e-5, 1e-2, 'log'), # L2 regularization
        'clip_norm': [None, 1.0, 2.0, 5.0], # Gradient clipping values
    },
    'MILO_LW': {
        'lr': (0.005, 0.1, 'log'),  # Learning rate with log-uniform distribution
        'momentum': (0.8, 0.95),    # Momentum for gradient averaging
        'scale_factor': (0.1, 0.3), # Scale-aware normalization mixing factor
        'eps': (1e-8, 1e-5, 'log'), # Numerical stability epsilon
        'adaptive_eps': (1e-10, 1e-6, 'log'), # Adaptive scaling epsilon
        'weight_decay': (1e-5, 1e-2, 'log'), # L2 regularization
        'clip_norm': [None, 1.0, 2.0, 5.0], # Gradient clipping values
        'scale_aware': [True, False], # Whether to use scale-aware normalization
    }
}

# --- Optimizer Parameter Settings (Unified Base) ---
OPTIMIZER_PARAMS = {
    "SGD": {"momentum": 0.9, "nesterov": True, "weight_decay": 0.0001}, 
    "ADAGRAD": {"lr_decay": 0, "weight_decay": 0.0, "eps": 1e-10},
    "ADAMW": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}, 
    "MILO": {
        "verbose_profile": False,
        "normalize": True,
        "layer_wise": False,  # Network-wide grouping
        "scale_aware": True,  # CRITICAL: Enable blending to prevent vanishing updates
        "scale_factor": 0.2,  # Blend 20% raw gradient with 80% normalized
        "nesterov": False,
        "adaptive": True,     # RMSprop-style adaptive scaling
        "momentum": 0.9,
        "profile_time": False,
        'max_group_size': 5000,  # Fixed-size groups (was None/dynamic - caused over-normalization)
        "use_cached_mapping": True,
        "foreach": True,
        'use_cuda_kernels': True,  # Enable CUDA optimization
        'normalize_interval': 1
    },
    "MILO_LW": {
        "normalize": True,
        "layer_wise": True,   # Layer-wise grouping for architecture-aware normalization
        "scale_aware": True,  # CRITICAL: Enable blending to prevent vanishing updates
        "scale_factor": 0.2,  # Blend 20% raw gradient with 80% normalized
        "nesterov": False,
        "adaptive": True,     # RMSprop-style adaptive scaling
        "momentum": 0.9,
        "profile_time": False,
        'max_group_size': 5000,  # Fixed-size groups (was None/dynamic - caused over-normalization)
        "use_cached_mapping": True,
        "foreach": True,
        'use_cuda_kernels': True,  # Enable CUDA optimization
        'normalize_interval': 1
    },
    "NOVOGRAD": {
        "betas": (0.9, 0.99),
        "weight_decay": 0.001,
        "grad_averaging": True
    },
    "ADAM_MINI": {
        "betas": (0.9, 0.999), 
        "eps": 1e-8, 
        "weight_decay": 0
        },
    "MUON": {
        "betas": (0.9, 0.999), 
        "eps": 1e-8, 
        "weight_decay": 0.01
        },
    "SOAP": {
        "betas": (0.95, 0.95),
        "weight_decay": 0.01,
        "precondition_frequency": 10,
        "max_precond_dim": 10000,
        "merge_dims": False,
        "precondition_1d": False,
        "normalize_grads": False,
        "eps": 1e-8
    },
    "ADEMAMIX": {
        "betas": (0.9, 0.999, 0.9999),
        "alpha": 8.0,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "beta3_warmup": 0,
        "alpha_warmup": 0
    },
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
    
    "ADVANCED_MLP": {
        "model_name": "AdvancedMLP_Deep",
        "dataset_name": "CIFAR10",
        "model_args": {"input_dim": 32*32*3, "num_classes": 10},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(-1))
        ]),
        "plot_titles": {
            "loss": "Advanced MLP: Loss vs. Epoch",
            "accuracy": "Advanced MLP: Accuracy vs. Epoch", 
            "f1": "Advanced MLP: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "WIDE_MLP": {
        "model_name": "AdvancedMLP_Wide", 
        "dataset_name": "CIFAR10",
        "model_args": {"input_dim": 32*32*3, "num_classes": 10},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(-1))
        ]),
        "plot_titles": {
            "loss": "Wide MLP: Loss vs. Epoch",
            "accuracy": "Wide MLP: Accuracy vs. Epoch",
            "f1": "Wide MLP: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "WIDE_RESNET": {
        "model_name": "WideResNet16_8",
        "dataset_name": "CIFAR10", 
        "model_args": {"num_classes": 10},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "plot_titles": {
            "loss": "Wide ResNet-16-8: Loss vs. Epoch",
            "accuracy": "Wide ResNet-16-8: Accuracy vs. Epoch",
            "f1": "Wide ResNet-16-8: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "SIMPLE_VIT": {
        "model_name": "SimpleViT_Tiny",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "plot_titles": {
            "loss": "Simple ViT: Loss vs. Epoch",
            "accuracy": "Simple ViT: Accuracy vs. Epoch",
            "f1": "Simple ViT: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "DEEPCNN": {
        "model_name": "DeepCNN",
        "dataset_name": "CIFAR10",
        "model_args": {},
        "transform": transforms.Compose([
            transforms.ToTensor() 
        ]),
        "plot_titles": {
            "loss": "Deep CNN: Loss vs. Epoch",
            "accuracy": "Deep CNN: Accuracy vs. Epoch",
            "f1": "Deep CNN: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "MODERN_CNN": {
        "model_name": "ModernCNN_Small",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "plot_titles": {
            "loss": "Modern CNN: Loss vs. Epoch",
            "accuracy": "Modern CNN: Accuracy vs. Epoch",
            "f1": "Modern CNN: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "ATTENTION_CNN": {
        "model_name": "AttentionCNN_Small",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "plot_titles": {
            "loss": "Attention CNN: Loss vs. Epoch",
            "accuracy": "Attention CNN: Accuracy vs. Epoch",
            "f1": "Attention CNN: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "HYBRID_CNN_TRANSFORMER": {
        "model_name": "HybridCNNTransformer_Small",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "plot_titles": {
            "loss": "Hybrid CNN-Transformer: Loss vs. Epoch",
            "accuracy": "Hybrid CNN-Transformer: Accuracy vs. Epoch",
            "f1": "Hybrid CNN-Transformer: F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "RESNET18": { # Updated ResNet18 configuration
        "model_name": "ResNet18",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100},
        "transform": transforms.Compose([
            transforms.ToTensor() # Use the same simple transform
        ]),
        "plot_titles": {
            "loss": "ResNet18 (CIFAR100): Loss vs. Epoch", 
            "accuracy": "ResNet18 (CIFAR100): Accuracy vs. Epoch",
            "f1": "ResNet18 (CIFAR100): F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "RESNET34_CIFAR10": {
        "model_name": "ResNet34",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "plot_titles": {
            "loss": "ResNet34 (CIFAR10): Loss vs. Epoch", 
            "accuracy": "ResNet34 (CIFAR10): Accuracy vs. Epoch",
            "f1": "ResNet34 (CIFAR10): F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "RESNET34_CIFAR100": {
        "model_name": "ResNet34",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        "plot_titles": {
            "loss": "ResNet34 (CIFAR100): Loss vs. Epoch", 
            "accuracy": "ResNet34 (CIFAR100): Accuracy vs. Epoch",
            "f1": "ResNet34 (CIFAR100): F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "VGG11_CIFAR10": {
        "model_name": "VGG11",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "plot_titles": {
            "loss": "VGG11 (CIFAR10): Loss vs. Epoch", 
            "accuracy": "VGG11 (CIFAR10): Accuracy vs. Epoch",
            "f1": "VGG11 (CIFAR10): F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },
    
    "VGG11_CIFAR100": {
        "model_name": "VGG11",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        "plot_titles": {
            "loss": "VGG11 (CIFAR100): Loss vs. Epoch",
            "accuracy": "VGG11 (CIFAR100): Accuracy vs. Epoch",
            "f1": "VGG11 (CIFAR100): F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },

    "VIT_TINY_CIFAR10": {
        "model_name": "ViT_Tiny",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10, "img_size": 32},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "plot_titles": {
            "loss": "ViT-Tiny (CIFAR10): Loss vs. Epoch",
            "accuracy": "ViT-Tiny (CIFAR10): Accuracy vs. Epoch",
            "f1": "ViT-Tiny (CIFAR10): F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    },

    "VIT_TINY_CIFAR100": {
        "model_name": "ViT_Tiny",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100, "img_size": 32},
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        "plot_titles": {
            "loss": "ViT-Tiny (CIFAR100): Loss vs. Epoch",
            "accuracy": "ViT-Tiny (CIFAR100): Accuracy vs. Epoch",
            "f1": "ViT-Tiny (CIFAR100): F1 Score vs. Epoch"
        },
        "cost_xlimit": None
    }
}

