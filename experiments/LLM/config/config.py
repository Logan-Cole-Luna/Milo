# --- Configuration ---

# Define vocabulary size and transformer configuration (Small model for testing)
VOCAB_SIZE = 50304          # Number of unique tokens in the vocabulary
CONTEXT_LENGTH = 64         # Maximum sequence length for the model
N_EMBED = 64                # Dimension of the embedding space (reduced from 128)
N_HEAD = 4                  # Number of attention heads in each transformer block (reduced from 8)
N_BLOCKS = 1                # Number of transformer blocks in the model

# Paths to training and development datasets
TRAIN_PATH = "data/train/pile_train.h5"  # File path for the training dataset
DEV_PATH = "data/val/pile_dev.h5"      # File path for the validation dataset

# Transformer training parameters
T_BATCH_SIZE = 32           # Number of samples per training batch (reduced from 32)
T_CONTEXT_LENGTH = 16       # Context length for training batches
T_TRAIN_STEPS = 3000       # Total number of training steps (reduced from 200000)
T_EVAL_STEPS = 500          # Frequency (in steps) to perform evaluation (reduced from 1000)
T_EVAL_ITERS = 125           # Number of iterations to evaluate the model (reduced from 250)
T_LR_DECAY_STEP = 1500      # Step at which to decay the learning rate (reduced from 50000)
T_LR = 0.05               # Initial learning rate for training
T_LR_DECAYED = 0.005         # Learning rate after decay
T_OUT_PATH = "models/transformer_tiny.pt"  # Path to save the trained model

# Number of runs for statistical analysis
NUM_RUNS = 5 # Number of times to run training for each optimizer

# List of optimizers to train and compare
OPTIMIZERS_TO_TRAIN = ["MILO", "MILO_LW", "SGD", "ADAMW", "ADAGRAD", "NOVOGRAD"]

# Optimizer configuration
OPTIMIZER = "NOVOGRAD"         # Options: "ADAMW", "SGD", "ADAM", "MILO", "MILO_LW", "NOVOGRAD", "ADAGRAD"
# --- Optimizer Parameter Settings (Unified Base) ---
OPTIMIZER_PARAMS = {
    "SGD": {"momentum": 0.9, "nesterov": True, "weight_decay": 0.0001},
    "ADAGRAD": {"lr_decay": 0, "weight_decay": 0.0, "eps": 1e-8},
    "ADAMW": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.001},
    "NOVOGRAD": {"betas": (0.9, 0.999), "weight_decay": 0.001, "grad_averaging": True},
    "MILO": {
        #"lr": 0.05,
        "normalize": True,
        "layer_wise": False,
        "scale_aware": True,
        "scale_factor": 1.0,
        "nesterov": False,
        "adaptive": True,
        "momentum": 0.9,
        #"weight_decay": 0.001,
        "profile_time": False,
        'max_group_size': None,
        "use_cached_mapping": False,
        "foreach": True,
    },
    "MILO_LW": {
        #"lr": 0.05,
        "normalize": True,
        "layer_wise": True,
        "scale_aware": True,
        "scale_factor": 1.0,
        "nesterov": False,
        "adaptive": True,
        "momentum": 0.9,
        #"weight_decay": 0.001,
        "profile_time": False,
        'max_group_size': None,
        "use_cached_mapping": True,
        "foreach": True,
    },
}


# Device configuration
DEVICE = 'cuda'

# Visualization folder
VISUALS_DIR = "visuals"

# Store all configurations in a dictionary for easy access and modification
default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,
    't_out_path': T_OUT_PATH,
    'device': DEVICE,
    'num_runs': NUM_RUNS,
    'optimizer': OPTIMIZER,
    'optimizer_params': OPTIMIZER_PARAMS,
    'optimizers_to_train': OPTIMIZERS_TO_TRAIN,
    'visuals_dir': VISUALS_DIR
}