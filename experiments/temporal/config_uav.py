'''
Configuration for UAV detection training (train_uav_detector.py).
'''
import torch

# --- General Settings ---
EXPERIMENT_NAME_BASE = "uav_detection_v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Settings ---
DATA_ROOT = r"C:\data\processed_anti_uav_v2"
TRAIN_SUBSET_FRACTION = 0.7

# --- Training Hyperparameters (Global defaults, can be overridden in EXPERIMENT_CONFIGURATIONS) ---
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4

# --- Validation and Visualization (Global defaults, can be overridden) ---
VALIDATE_EVERY_EPOCH_FACTOR = 0.2 # Validate every 20% of epochs (e.g., every 10 epochs if EPOCHS=50)
VISUALIZATION_SAMPLES = 5 # Number of samples to visualize

# --- Overfitting Settings (Global defaults, can be overridden) ---
OVERFIT = False 
OVERFIT_N_SAMPLES = 32

# --- Dataloader Settings (Global defaults) ---
NUM_WORKERS = 0
PIN_MEMORY = False  # Disabled to prevent CUDA resource mapping errors on Windows
PERSISTENT_WORKERS = False  # Disabled to avoid issues with worker threads holding resources
PREFETCH_FACTOR = 4

# --- Results Directory ---
RESULTS_BASE_DIR = "experiments/temporal/results"

# --- Experiment Runner Settings (Global defaults) ---
OPTIMIZERS = ["AdamW"]
RUNS_PER_OPTIMIZER = 1
# SCHEDULER_PARAMS can be global or part of each exp_config if schedulers change
SCHEDULER_PARAMS = {opt: {"scheduler": "CosineAnnealingLR", "params": {}} for opt in OPTIMIZERS}

# --- Plotting Titles (Global defaults, can be overridden in EXPERIMENT_CONFIGURATIONS) ---
LOSS_PLOT_TITLE = "Loss vs. Epoch (UAV Detection)"
IOU_PLOT_TITLE = "IoU vs. Epoch (UAV Detection)"
LOSS_PLOT_YLABEL = "Average Loss"
IOU_PLOT_YLABEL = "Average IoU"

# --- Experiment Configurations ---
EXPERIMENT_CONFIGURATIONS = [
    {
        "name": "single_frame_default_resnet18",
        "MODEL_TYPE": "single_frame",
        "MODEL_ARCHITECTURE": "resnet18",
        "SEQ_LEN": 1,
    },
    {
        "name": "temporal_s5_default_resnet18",
        "MODEL_TYPE": "temporal",
        "MODEL_ARCHITECTURE": "resnet18",
        "SEQ_LEN": 5,
    },
    #{
    #    "name": "single_frame_resnet18",
    #    "MODEL_TYPE": "single_frame",
    #    "MODEL_ARCHITECTURE": "resnet18",
    #    "SEQ_LEN": 1,
    #    "BATCH_SIZE": 64, # Example: Can still override other params
    #    "EPOCHS": 30
    #},
    #{
    #    "name": "temporal_s3_resnet18",
    #    "MODEL_TYPE": "temporal",
    #    "MODEL_ARCHITECTURE": "resnet18",
    #    "SEQ_LEN": 3,
    #    "EPOCHS": 20, # Example
    #},
    # Add more configurations as needed:
    # {
    #     "name": "temporal_s3_fast_run",
    #     "MODEL_TYPE": "temporal",
    #     "SEQ_LEN": 3,
    #     "EPOCHS": 5,
    #     "VISUALIZATION_SAMPLES": 2,
    # },
]
