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

# --- Learning Rates per Optimizer ---
LEARNING_RATES = {
    "MILO": 1e-4,
    "MILO_LW": 1e-4,
    "SGD": 1e-3,
    "ADAMW": 2e-5,
    "ADAGRAD": 1e-3,
    "LION": 1e-4,
    "ADAM_MINI": 1e-4,
    "RMSPROP_MOMENTUM": 1e-3,
    "SHAMPOO": 1e-4,
    "SOAP": 1e-4,
    "MUON": 1e-4,
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
        "momentum": 0.0
    },
    "SOAP": {
        "betas": (0.95, 0.95),
        "weight_decay": 0.01
    },
    "MUON": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
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
        "momentum": 0.1
    },
    "SOAP": {
        "betas": (0.9, 0.98),
        "weight_decay": 0.005
    },
    "MUON": {
        "betas": (0.95, 0.999),
        "eps": 1e-8,
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
