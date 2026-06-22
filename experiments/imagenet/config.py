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

# --- Learning Rates per Optimizer ---
LEARNING_RATES = {
    "MILO": 0.001,
    "MILO_LW": 0.001,
    "SGD": 0.1,
    "ADAMW": 0.001,
    "ADAGRAD": 0.01,
    "LION": 0.001,
    "ADAM_MINI": 0.001,
    "RMSPROP_MOMENTUM": 0.01,
    "SHAMPOO": 0.001,
    "SOAP": 0.001,
    "MUON": 0.001,
}

# --- Optimizers to Evaluate ---
OPTIMIZERS = [
    "MILO", "MILO_LW",
    "SGD", "ADAMW", "ADAGRAD",
    "LION", "ADAM_MINI", "RMSPROP_MOMENTUM", "SHAMPOO",
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
        "betas": (0.9, 0.999),
        "eps": 1e-8,
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
        "betas": (0.95, 0.999),
        "eps": 1e-8,
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
