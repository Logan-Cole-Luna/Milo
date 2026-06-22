"""
Vision Experiments Configuration

Configures optimizer evaluation on CIFAR-10/100 with multiple architectures.
"""

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
    # Core MILO variants
    "MILO", "MILO_LW",
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

# --- Base Learning Rates per Experiment ---
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
        "betas": (0.9, 0.999),
        "eps": 1e-8,
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
        "betas": (0.95, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.005
    },
}

# --- Scheduler Parameters ---
SCHEDULER_PARAMS = {
    "scheduler": "None"
}

# --- Experiment Configurations ---
EXPERIMENT_CONFIGS = {
    "LOGISTIC": {
        "model_name": "LogisticRegressionModel",
        "dataset_name": "MNIST",
        "model_args": {"input_dim": 784, "num_classes": 10},
        "transforms": None,
        "plot_title": "Logistic Regression on MNIST",
    },
    "MULTILAYER": {
        "model_name": "MLP",
        "dataset_name": "MNIST",
        "model_args": {"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
        "transforms": None,
        "plot_title": "MLP on MNIST",
    },
    "RESNET34_CIFAR10": {
        "model_name": "ResNet34",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transforms": None,
        "plot_title": "ResNet34 on CIFAR-10",
    },
    "RESNET34_CIFAR100": {
        "model_name": "ResNet34",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100},
        "transforms": None,
        "plot_title": "ResNet34 on CIFAR-100",
    },
    "VGG11_CIFAR10": {
        "model_name": "VGG11",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transforms": None,
        "plot_title": "VGG11 on CIFAR-10",
    },
    "VGG11_CIFAR100": {
        "model_name": "VGG11",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100},
        "transforms": None,
        "plot_title": "VGG11 on CIFAR-100",
    },
    "VIT_TINY_CIFAR10": {
        "model_name": "ViT_Tiny",
        "dataset_name": "CIFAR10",
        "model_args": {"num_classes": 10},
        "transforms": None,
        "plot_title": "ViT-Tiny on CIFAR-10",
    },
    "VIT_TINY_CIFAR100": {
        "model_name": "ViT_Tiny",
        "dataset_name": "CIFAR100",
        "model_args": {"num_classes": 100},
        "transforms": None,
        "plot_title": "ViT-Tiny on CIFAR-100",
    },
}
