'''
Configuration for UAV detection training (train_uav_detector.py).
'''
import torch

# --- General Settings ---
EXPERIMENT_NAME = "stand_experiment_01"  # e.g., "temporal_experiment_01"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Settings ---
DATA_ROOT = "C:/data/processed_anti_uav"  # Corresponds to --data-root (REQUIRED)
# Fraction of the training dataset to use (e.g., 0.25 for 25%%). Value should be in (0, 1.0].
TRAIN_SUBSET_FRACTION = 1.0 # Corresponds to --train-subset-fraction

# --- Model Settings ---
# Type of model to train ("single_frame" or "temporal").
MODEL_TYPE = "single_frame"
# Sequence length for temporal model (if model-type is temporal).
SEQ_LEN = 5 

# --- Training Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4

# --- Validation and Visualization ---
# Run validation every N epochs.
VALIDATE_EVERY = (EPOCHS / 2)
# Number of samples to visualize.
VISUALIZATION_SAMPLES = 10

# --- Overfitting Settings ---
# If True, run in overfitting mode on a small subset of data.
OVERFIT = False
# Number of samples to use for overfitting.
OVERFIT_N_SAMPLES = 32

# --- Dataloader Settings (from train_uav_detector.py) ---
NUM_WORKERS = 12
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

# --- Results Directory ---
# Base directory for saving results (logs, models, visualizations)
RESULTS_BASE_DIR = "results"

# --- Experiment Runner Settings (newly added) ---
# List of optimizers to use. For now, AdamW is the primary one used in train_uav_detector.
OPTIMIZERS = ["AdamW"]
RUNS_PER_OPTIMIZER = 1 # Number of times to run the experiment for each optimizer for averaging
SCHEDULER_PARAMS = {opt: {"scheduler": "CosineAnnealingLR", "params": {}} for opt in OPTIMIZERS}

# --- Plotting Titles (newly added) ---
LOSS_PLOT_TITLE = "Loss vs. Epoch (UAV Detection)"
IOU_PLOT_TITLE = "IoU vs. Epoch (UAV Detection)"
LOSS_PLOT_YLABEL = "Average Loss"
IOU_PLOT_YLABEL = "Average IoU"
