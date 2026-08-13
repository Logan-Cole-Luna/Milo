"""
NLP Experiments Configuration

Configures optimizer evaluation on BERT fine-tuning for sentiment classification.
"""

# --- Training Parameters ---
BATCH_SIZE = 32
EPOCHS = 3
RUNS_PER_OPTIMIZER = 5

# --- Output Directories ---
RESULTS_DIR = "results_nt_bert"
RESULTS_DIR_TUNED = "results_nt_bert_tuned"

# --- Learning Rates per Optimizer (Optuna-tuned; SOAP=known-good fallback) ---
LEARNING_RATES = {
    "MILO": 0.000308,
    "MILO_LW": 0.000274,
    "MILOM": 3.79e-06,
    "MION": 2.43e-05,
    "SGD": 0.000264,
    "ADAMW": 2.8e-05,
    "ADAGRAD": 0.000369,
    "LION": 4.21e-06,
    "ADAM_MINI": 6.53e-06,
    "RMSPROP_MOMENTUM": 1.01e-05,
    "SHAMPOO": 0.00414,
    "SOAP": 0.0001,
    "MUON": 0.00102,
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
        "use_cuda_kernels": False,
    },
    "MILO_LW": {
        "normalize": True,
        "layer_wise": True,
        "scale_aware": True,
        "scale_factor": 0.2,
        "max_group_size": 5000,
        "momentum": 0.9,
        "adaptive": True,
        "use_cuda_kernels": False,
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
        "weight_decay": 0.01
    },
    "ADAMW": {},
    "ADAGRAD": {
        "lr_decay": 0,
        "weight_decay": 0.0
    },
    "LION": {
        "betas": (0.9, 0.99),
        "weight_decay": 0.01
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
        "update_freq": 16  # amortize the large-matrix inverse on BERT
    },
    "SOAP": {
        "betas": (0.95, 0.95),
        "weight_decay": 0.01
    },
    "MUON": {
        "weight_decay": 0.01
    },
}

# --- Tuned Optimizer Parameters (from hyperparameter tuning) ---
OPTIMIZER_PARAMS_TUNED = {
    "MILO": {
        "normalize": True,
        "layer_wise": False,
        "scale_aware": True,
        "scale_factor": 0.15,
        "max_group_size": 5000,
        "momentum": 0.95,
        "adaptive": True,
        "use_cuda_kernels": False,
    },
    "MILO_LW": {
        "normalize": True,
        "layer_wise": True,
        "scale_aware": True,
        "scale_factor": 0.15,
        "max_group_size": 5000,
        "momentum": 0.95,
        "adaptive": True,
        "use_cuda_kernels": False,
    },
    "MILOM": {
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.005,
        "eps": 1e-8,
        "scale_factor": 0.15,
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
        "weight_decay": 0.005
    },
    "ADAMW": {"eps": 1e-8, "weight_decay": 0.001},
    "ADAGRAD": {
        "lr_decay": 0.01,
        "weight_decay": 0.001
    },
    "LION": {
        "betas": (0.95, 0.98),
        "weight_decay": 0.005
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
        "momentum": 0.1,
        "update_freq": 16  # amortize the large-matrix inverse on BERT
    },
    "SOAP": {
        "betas": (0.9, 0.98),
        "weight_decay": 0.005
    },
    "MUON": {
        "weight_decay": 0.005
    },
}

# --- Model Configuration ---
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2
DATASET = "SST-2 (Stanford Sentiment Treebank)"
MAX_LENGTH = 128

# --- Gradient Clipping ---
GRADIENT_CLIP_NORM = 1.0
