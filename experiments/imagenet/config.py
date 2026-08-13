"""
ImageNet Experiments Configuration

Configures optimizer evaluation on Tiny ImageNet-200 (large-scale vision task).
"""

# --- Training Parameters ---
BATCH_SIZE = 128
EPOCHS = 5
RUNS_PER_OPTIMIZER = 3

# --- Output Directories ---
RESULTS_DIR = "results_nt_imagenet200"
RESULTS_DIR_TUNED = "results_nt_imagenet200_tuned"

# --- Learning Rates per Optimizer (Optuna-tuned) ---
LEARNING_RATES = {
    "MILO": 0.0208,
    "MILO_LW": 0.00392,
    "MILOM": 0.00338,
    "MION": 0.00101,
    "SGD": 0.0126,
    "ADAMW": 0.000104,
    "ADAGRAD": 0.00104,
    "LION": 3.11e-05,
    "ADAM_MINI": 0.000131,
    "RMSPROP_MOMENTUM": 0.000131,
    "SHAMPOO": 0.0252,
    "SOAP": 0.00171,
    "MUON": 0.00368,
}

# --- Optimizers to Evaluate ---
OPTIMIZERS = [
    # Original + Improved MILO variants
    "MILO", "MILO_LW", "MILOM", "MION",
    # Baselines
    "SGD", "ADAMW", "ADAGRAD",
    # Modern optimizers (2024-2026)
    "LION", "ADAM_MINI", "RMSPROP_MOMENTUM", "SHAMPOO",
    # Advanced optimizers
    "SOAP", "MUON"
]

# --- Optimizer Parameters ---
OPTIMIZER_PARAMS = {
    "MILO": {
        "normalize": True,
        "layer_wise": False,
        "scale_aware": True,
        "scale_factor": 0.2,
        "max_group_size": 5000,
        "momentum": 0.9,
        "adaptive": True,
        "use_cuda_kernels": True,
    },
    "MILO_LW": {
        "normalize": True,
        "layer_wise": True,
        "scale_aware": True,
        "scale_factor": 0.2,
        "max_group_size": 5000,
        "momentum": 0.9,
        "adaptive": True,
        "use_cuda_kernels": True,
    },
    "MILOM": {
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "scale_factor": 0.2,
        "rms_target": 0.2,
        "group_size": None
    },
    "MION": {
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "scale_factor": 0.0,
        "rms_target": 0.2,
        "ns_steps": 5
    },
    "SGD": {
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0001
    },
    "ADAMW": {},
    "ADAGRAD": {
        "lr_decay": 0,
        "weight_decay": 0.0
    },
    "LION": {
        "betas": (0.9, 0.99),
        "weight_decay": 0.0001
    },
    "ADAM_MINI": {
        "betas": (0.9, 0.999),
        "eps": 1e-8
    },
    "RMSPROP_MOMENTUM": {
        "alpha": 0.99,
        "momentum": 0.9,
        "eps": 1e-8
    },
    "SHAMPOO": {
        "eps": 1e-10,
        "momentum": 0.0,
        "weight_decay": 0.01
    },
    "SOAP": {
        "betas": (0.95, 0.95),
        "weight_decay": 0.0001
    },
    "MUON": {
        "weight_decay": 0.0001
    },
}

# --- Tuned Optimizer Parameters (from hyperparameter tuning) ---
OPTIMIZER_PARAMS_TUNED = {
    "MILO": {
        "normalize": True,
        "layer_wise": False,
        "scale_aware": True,
        "scale_factor": 0.2,
        "max_group_size": 5000,
        "momentum": 0.95,
        "adaptive": True,
        "use_cuda_kernels": True,
    },
    "MILO_LW": {
        "normalize": True,
        "layer_wise": True,
        "scale_aware": True,
        "scale_factor": 0.2,
        "max_group_size": 5000,
        "momentum": 0.95,
        "adaptive": True,
        "use_cuda_kernels": True,
    },
    "MILOM": {
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.005,
        "eps": 1e-8,
        "scale_factor": 0.2,
        "rms_target": 0.2,
        "group_size": None
    },
    "MION": {
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.005,
        "eps": 1e-8,
        "scale_factor": 0.0,
        "rms_target": 0.2,
        "ns_steps": 5
    },
    "SGD": {
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.00005
    },
    "ADAMW": {"weight_decay": 0.0005},
    "ADAGRAD": {
        "lr_decay": 0.01,
        "weight_decay": 0.0001
    },
    "LION": {
        "betas": (0.95, 0.98),
        "weight_decay": 0.00005
    },
    "ADAM_MINI": {
        "betas": (0.95, 0.999),
        "eps": 1e-8
    },
    "RMSPROP_MOMENTUM": {
        "alpha": 0.98,
        "momentum": 0.95,
        "eps": 1e-8
    },
    "SHAMPOO": {
        "eps": 1e-10,
        "momentum": 0.05,
        "weight_decay": 0.005
    },
    "SOAP": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.00005
    },
    "MUON": {
        "weight_decay": 0.00005
    },
}

# --- Dataset Configuration ---
DATASET_NAME = "Tiny ImageNet-200"
NUM_CLASSES = 200
IMAGE_SIZE = 64
DATA_ROOT = "/scratch/datasets/tiny-imagenet-200"

# --- Model Configuration ---
MODEL_NAME = "ResNet34"
