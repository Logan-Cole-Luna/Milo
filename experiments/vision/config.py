"""
Vision Experiments Configuration

Configures optimizer evaluation on CIFAR-10/100 with multiple architectures.
"""

from torchvision import transforms

# --- Experiment Settings ---
FINAL_SUITE = [
    "LOGISTIC", "MULTILAYER",
    "RESNET34_CIFAR10", "RESNET34_CIFAR100",
    "VGG11_CIFAR10", "VGG11_CIFAR100",
    "VIT_TINY_CIFAR10", "VIT_TINY_CIFAR100"
]

EXPERIMENTS = [FINAL_SUITE]

# --- Optimizers to Evaluate ---
OPTIMIZERS = [
    # Core + Improved MILO variants
    "MILO", "MILO_LW", "MILOM", "MION",
    # Baseline optimizers
    "SGD", "ADAMW", "ADAGRAD",
    # Modern optimizers (2024-2026)
    "LION", "ADAM_MINI", "RMSPROP_MOMENTUM", "SHAMPOO",
    # Advanced optimizers
    "SOAP", "MUON"
]

# --- Training Parameters ---
BATCH_SIZE = 128
EPOCHS = 5
RUNS_PER_OPTIMIZER = 5
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.10

# --- Output Directories ---
RESULTS_DIR = "results_nt"
VISUALS_DIR = "visuals_nt"
RESULTS_DIR_TUNED = "results_nt_tuned"
VISUALS_DIR_TUNED = "visuals_nt_tuned"

# --- Base Learning Rates per Experiment (legacy fallback) ---
LR = {
    "LOGISTIC": 0.05,
    "MULTILAYER": 0.05,
    "RESNET34_CIFAR10": 0.001,
    "RESNET34_CIFAR100": 0.001,
    "VGG11_CIFAR10": 0.001,
    "VGG11_CIFAR100": 0.001,
    "VIT_TINY_CIFAR10": 0.0005,
    "VIT_TINY_CIFAR100": 0.0005,
}

# --- Per-optimizer learning rates, grouped by architecture family ---
# Each optimizer class has very different optimal LR scales, so using a single
# shared LR per experiment is unfair to baselines (SGD wants ~0.1, Lion ~1e-4,
# AdamW ~1e-3). These per-optimizer values are architecture-aware: small linear
# models (MNIST), deep CNNs (ResNet/VGG on CIFAR), and from-scratch ViTs each
# have their own LR regime. Values are sensible defaults from the milo-bench
# sweeps and standard practice (not per-run tuned).
ARCH_FAMILY = {
    "LOGISTIC": "linear",
    "MULTILAYER": "linear",
    "RESNET34_CIFAR10": "cnn",
    "RESNET34_CIFAR100": "cnn",
    "VGG11_CIFAR10": "cnn",
    "VGG11_CIFAR100": "cnn",
    "VIT_TINY_CIFAR10": "vit",
    "VIT_TINY_CIFAR100": "vit",
}

LEARNING_RATES = {
    "linear": {
        "SGD": 0.1, "ADAMW": 0.01, "ADAGRAD": 0.05,
        "LION": 1e-3, "ADAM_MINI": 0.01, "RMSPROP_MOMENTUM": 0.01,
        "SHAMPOO": 0.01, "SOAP": 0.01, "MUON": 0.05,
        "MILO": 0.05, "MILO_LW": 0.05, "MILOM": 0.05, "MION": 0.05, "MION_NOR": 0.05,
    },
    # Optuna-tuned (cnn = geomean of ResNet34+VGG11 sweeps; vit = ViT sweep)
    "cnn": {
        "SGD": 0.0674, "ADAMW": 0.000165, "ADAGRAD": 0.00103,
        "LION": 5.69e-05, "ADAM_MINI": 0.000236, "RMSPROP_MOMENTUM": 0.000102,
        "SHAMPOO": 0.0285, "SOAP": 0.00135, "MUON": 0.0143,
        "MILO": 0.0033, "MILO_LW": 0.00344, "MILOM": 0.00101, "MION": 0.00205,
        "MION_NOR": 0.00205,  # placeholder = MION's tuned LR; retune via optuna_sweep.py
    },
    "vit": {
        "SGD": 0.0345, "ADAMW": 0.000257, "ADAGRAD": 0.000615,
        "LION": 3.06e-05, "ADAM_MINI": 0.000258, "RMSPROP_MOMENTUM": 5.04e-05,
        "SHAMPOO": 0.00959, "SOAP": 0.00134, "MUON": 0.00433,
        "MILO": 0.00101, "MILO_LW": 0.00101, "MILOM": 0.00101, "MION": 0.00184,
        "MION_NOR": 0.00184,  # placeholder = MION's tuned LR; retune via optuna_sweep.py
    },
}


def get_learning_rate(experiment_type, optimizer_name):
    """Return the per-optimizer LR for an experiment, with safe fallbacks."""
    import os
    _override = os.getenv("LR_OVERRIDE")
    if _override:
        return float(_override)
    fam = ARCH_FAMILY.get(experiment_type)
    if fam is not None and optimizer_name in LEARNING_RATES.get(fam, {}):
        return LEARNING_RATES[fam][optimizer_name]
    # Fallback: legacy per-experiment base LR
    return LR.get(experiment_type, 0.001)

# --- Optimizer Parameters ---
OPTIMIZER_PARAMS = {
    "SGD": {"momentum": 0.9, "nesterov": True, "weight_decay": 0.0001},
    "ADAGRAD": {"lr_decay": 0, "weight_decay": 0.0, "eps": 1e-10},
    "ADAMW": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
    "MILO": {
        "verbose_profile": False,
        "normalize": True,
        "layer_wise": False,
        "scale_aware": True,
        "scale_factor": 0.2,
        "nesterov": False,
        "adaptive": True,
        "momentum": 0.9,
        "profile_time": False,
        "max_group_size": 5000,
        "use_cached_mapping": True,
        "foreach": True,
        "use_cuda_kernels": True,
        "normalize_interval": 1
    },
    "MILO_LW": {
        "normalize": True,
        "layer_wise": True,
        "scale_aware": True,
        "scale_factor": 0.2,
        "nesterov": False,
        "adaptive": True,
        "momentum": 0.9,
        "profile_time": False,
        "max_group_size": 5000,
        "use_cached_mapping": True,
        "foreach": True,
        "use_cuda_kernels": True,
        "normalize_interval": 1
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
    "MION_NOR": {
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "scale_factor": 0.0,
        "rms_target": 0.2,
        "ns_steps": 5,
        "row_norm": True
    },
    "LION": {
        "betas": (0.9, 0.99),
        "weight_decay": 0.01
    },
    "ADAM_MINI": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01
    },
    "RMSPROP_MOMENTUM": {
        "alpha": 0.99,
        "momentum": 0.9,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "centered": False
    },
    "SHAMPOO": {
        "eps": 1e-10,
        "momentum": 0.0,
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
    "MUON": {
        "weight_decay": 0.01
    },
}

# --- Tuned Optimizer Parameters (from hyperparameter tuning) ---
# These override OPTIMIZER_PARAMS when using tuned mode
OPTIMIZER_PARAMS_TUNED = {
    "SGD": {"momentum": 0.95, "nesterov": True, "weight_decay": 0.0005},
    "ADAGRAD": {"lr_decay": 0.01, "weight_decay": 0.001, "eps": 1e-8},
    "ADAMW": {"betas": (0.95, 0.999), "eps": 1e-8, "weight_decay": 0.005},
    "MILO": {
        "normalize": True,
        "layer_wise": False,
        "scale_aware": True,
        "scale_factor": 0.25,
        "nesterov": False,
        "adaptive": True,
        "momentum": 0.95,
        "max_group_size": 5000,
        "use_cached_mapping": True,
        "foreach": True,
        "use_cuda_kernels": True,
    },
    "MILO_LW": {
        "normalize": True,
        "layer_wise": True,
        "scale_aware": True,
        "scale_factor": 0.25,
        "nesterov": False,
        "adaptive": True,
        "momentum": 0.95,
        "max_group_size": 5000,
        "use_cached_mapping": True,
        "foreach": True,
        "use_cuda_kernels": True,
    },
    "MILOM": {
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.005,
        "eps": 1e-8,
        "scale_factor": 0.25,
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
    "MION_NOR": {
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.005,
        "eps": 1e-8,
        "scale_factor": 0.0,
        "rms_target": 0.2,
        "ns_steps": 5,
        "row_norm": True
    },
    "LION": {
        "betas": (0.95, 0.98),
        "weight_decay": 0.005
    },
    "ADAM_MINI": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.005
    },
    "RMSPROP_MOMENTUM": {
        "alpha": 0.98,
        "momentum": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.005,
    },
    "SHAMPOO": {
        "eps": 1e-10,
        "momentum": 0.1,
        "weight_decay": 0.005
    },
    "SOAP": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.005,
        "precondition_frequency": 10,
        "max_precond_dim": 10000,
    },
    "MUON": {
        "weight_decay": 0.005
    },
}

# --- Scheduler Parameters ---
SCHEDULER_PARAMS = {
    "scheduler": "None"
}

# --- Data Transforms ---
# MNIST: Flatten to 1D, normalize
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# CIFAR: Standard augmentation and normalization
CIFAR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# --- Experiment Configurations ---
EXPERIMENT_CONFIGS = {
    "LOGISTIC": {
        "model_name": "LogisticRegressionModel",
        "dataset_name": "MNIST",
        "model_args": {"input_dim": 784, "num_classes": 10},
        "transforms": MNIST_TRANSFORM,
        "plot_title": "Logistic Regression on MNIST",
    },
    "MULTILAYER": {
        "model_name": "MLP",
        "dataset_name": "MNIST",
        "model_args": {"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
        "transforms": MNIST_TRANSFORM,
        "plot_title": "MLP on MNIST",
    },
    "RESNET34_CIFAR10": {
        "model_name": "ResNet34",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transforms": CIFAR_TRANSFORM,
        "plot_title": "ResNet34 on CIFAR-10",
    },
    "RESNET34_CIFAR100": {
        "model_name": "ResNet34",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100},
        "transforms": CIFAR_TRANSFORM,
        "plot_title": "ResNet34 on CIFAR-100",
    },
    "VGG11_CIFAR10": {
        "model_name": "VGG11",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transforms": CIFAR_TRANSFORM,
        "plot_title": "VGG11 on CIFAR-10",
    },
    "VGG11_CIFAR100": {
        "model_name": "VGG11",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100},
        "transforms": CIFAR_TRANSFORM,
        "plot_title": "VGG11 on CIFAR-100",
    },
    "VIT_TINY_CIFAR10": {
        "model_name": "ViT_Tiny",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transforms": CIFAR_TRANSFORM,
        "plot_title": "ViT-Tiny on CIFAR-10",
    },
    "VIT_TINY_CIFAR100": {
        "model_name": "ViT_Tiny",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100},
        "transforms": CIFAR_TRANSFORM,
        "plot_title": "ViT-Tiny on CIFAR-100",
    },
}
