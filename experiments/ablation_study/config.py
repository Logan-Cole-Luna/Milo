"""
Configuration settings for the MILO/MILO_LW ablation study.

Purpose:
This file configures an ablation study for the MILO and MILO_LW optimizers.
An ablation study systematically varies hyperparameters of an optimizer to understand
the impact of each component (e.g., normalization, adaptive scaling, momentum) on
its performance across different models and datasets.

Options:
- BATCH_SIZE, EPOCHS, RUNS_PER_OPTIMIZER: Standard training control parameters.
- SKIP_EXISTING_ABLATION_GROUPS: Boolean to skip already completed ablation runs.
- OPTIMIZER_PARAMS: Baseline (default) configurations for "MILO" and "MILO_LW"
  which serve as a reference for the ablation variations.
- ABLATION_PARAMS: Defines which hyperparameters of MILO and MILO_LW will be
  varied and the list of alternative values to test against their baseline.
  For example, 'normalize': [True, False] will test MILO with and without normalization.
- EXPERIMENT_CONFIGS: A list specifying the combinations of models (e.g., "MLP",
  "ResNet18") and datasets (e.g., "MNIST", "CIFAR10") on which the ablation
  study will be performed. Each entry includes model name, dataset name,
  model-specific arguments, and necessary data transformations.

Experiments:
The script using this configuration will systematically test different variants of
MILO and MILO_LW. For each optimizer, it iterates through the parameters defined in
ABLATION_PARAMS, changing one hyperparameter at a time from its baseline value to
each of the specified alternatives. This process is repeated for every model-dataset
pair in EXPERIMENT_CONFIGS. The goal is to isolate the impact of each specific
hyperparameter (e.g., `normalize`, `adaptive`, `layer_wise`, `scale_aware`) on the
optimizer's training performance and final results.
"""
from torchvision import transforms # <-- Add this import

# --- Experiment Settings ---
BATCH_SIZE = 128
EPOCHS = 10 
RUNS_PER_OPTIMIZER = 2 
SKIP_EXISTING_ABLATION_GROUPS = True # Set to False to re-run all groups

# --- Base Optimizer Parameters ---
# Base configurations for MILO and MILO_LW used as starting points for ablation.
OPTIMIZER_PARAMS = {
    "MILO": {
        "lr": 0.01,
        "normalize": True,
        "adaptive": True,
        "momentum": 0.9,
        "eps": 1e-5,
        "weight_decay": 0.0001,
        "use_cached_mapping": True,
        "profile_time": False,
        "layer_wise": False,
        "scale_aware": False,
    },
    "MILO_LW": {
        "lr": 0.01,
        "normalize": True,
        "adaptive": True,
        "momentum": 0.9,
        "eps": 1e-5,
        "weight_decay": 0.0001,
        "use_cached_mapping": True,
        "profile_time": False,
        "layer_wise": True,
        "scale_aware": True,
        "scale_factor": 0.2,
    },
}

# --- Ablation Parameters ---
# Dictionary defining which parameters to vary for each optimizer type
# and the values to test against the baseline.
ABLATION_PARAMS = {
    "MILO": {
        "lr": [0.1, 0.01, 0.001],
        "momentum": [0.0, 0.45, 0.9],
        "weight_decay": [0.001, 0.0],
        "eps": [1e-5, 0.005, 0.05],
        "normalize": [True, False],
        "adaptive": [True, False]
    },
    "MILO_LW": {
        "lr": [0.1, 0.01, 0.001],
        "momentum": [0.0, 0.45, 0.9], 
        "weight_decay": [0.001, 0.0],
        "eps": [1e-5, 0.005, 0.05],
        "normalize": [True, False],
        "adaptive": [True, False],
        "scale_aware": [True, False],
        "scale_factor": [0.1, 0.2, 0.5] # Only relevant if scale_aware=True in the baseline/variation
    }
}

# --- Model and Dataset Configurations ---
# List of (model_name, dataset_name, model_args, dataset_transform) tuples to test
EXPERIMENT_CONFIGS = [
    ("MLP", "MNIST", {}, # No specific args for MLP
     transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])),

    #("DeepCNN", "CIFAR10", {}, # No specific args for DeepCNN
    # transforms.Compose([transforms.RandomCrop(32, padding=4),
    #                     transforms.RandomHorizontalFlip(),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])), # CIFAR-10 normalization

    ("ResNet18", "CIFAR10", {"num_classes": 10}, 
     transforms.Compose([transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
]
