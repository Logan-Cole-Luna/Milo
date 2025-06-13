#!/usr/bin/env python3
"""
Main training script for UAV object detection models (single-frame and temporal).

This script handles:
- Configuration loading (via config_uav.py).
- Dataset and DataLoader setup (using uav_dataset.py).
- Model initialization (ResNet50Regressor or TemporalResNet50Regressor from network.py).
- Training and validation loops.
- Metric calculation (IoU using uav_metrics.py).
- Visualization of predictions.
- Integration with experiment_runner.py for managing multiple experiment runs
  (e.g., with different optimizers).

The script can be run directly to perform training and evaluation based on the
settings in config_uav.py. It pre-loads datasets and then uses `experiment_runner`
to iterate through specified optimizers or experiment configurations.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
import time
import json
import functools # Add functools import
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
import torch.onnx # Add ONNX import
from tqdm import tqdm
import numpy as np # For averaging metrics
import pandas as pd # For saving metrics to CSV
import matplotlib.pyplot as plt # For plotting (though plotting.py handles actual plotting)
import seaborn as sns # For styling plots
import functools # Add functools for partial

# Add project root to sys.path to allow importing from experiments and other top-level modules
project_root = Path(__file__).resolve().parents[2] # Assuming train_uav_detector.py is in experiments/temporal/
sys.path.append(str(project_root))

# Import models from network.py (assuming it's in the same directory or accessible via PYTHONPATH)
from network import SingleFrameResNetRegressor, TemporalResNetRegressor # Updated class names

# Local imports from the current 'temporal' package
import config_uav as C
from uav_dataset import FrameBBOXDataset
from uav_metrics import bbox_iou

# Imports from the \\'experiments\\' module
from experiments.train_utils import get_layer_names # evaluate_model might need adaptation for regression
from experiments.experiment_runner import run_experiments # This will be the main driver
from experiments.plotting import plot_seaborn_style_with_error_bars, setup_plot_style, plot_resource_usage
from experiments.utils.resource_tracker import ResourceTracker, save_resource_info # Optional resource tracking

# --- IoU Threshold for F1 Score Calculation ---
IOU_THRESHOLD_F1 = 0.5

# --- Setup Visualization Style (from supervised_learning_experiment.py) ---
setup_plot_style() # Applies seaborn styling etc.

# --- Albumentations Pipelines (defined globally) ---
train_tf = A.Compose([
    A.Resize(224, 224, interpolation=cv2.INTER_LINEAR), # Simplified: direct resize
    A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ToTensorV2(),
])
val_tf = A.Compose([
    A.Resize(224, 224, interpolation=cv2.INTER_LINEAR), # Simplified: direct resize
    A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

# Determine device (already done, but ensure it\'s consistent)
device = torch.device(C.DEVICE)

# --- Reproducibility: Fix random seeds globally (from supervised_learning_experiment.py) ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# Ensure deterministic behavior where possible
torch.backends.cudnn.deterministic = True
# Enable cuDNN autotuner to find optimal algorithms for fixed input sizes
torch.backends.cudnn.benchmark = True

# ─────────────────────────────── Dataset ──────────────────────────────────────
# The FrameBBOXDataset class has been moved to uav_dataset.py
# from .uav_dataset import FrameBBOXDataset # This line is now at the top with other local imports

# ─────────────────────────────── Metrics ─────────────────────────────────────
# The bbox_iou function has been moved to uav_metrics.py
# from .uav_metrics import bbox_iou # This line is now at the top with other local imports

# ─────────────────────────────── Training ────────────────────────────────────
def run_epoch(model, loader, optimizer, scaler, device, stage,
              criterion_l1, criterion_iou, scheduler=None, overfit=False, model_type="single_frame"):
    """
    Runs a single epoch of training or evaluation.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the current stage (train/val/test).
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. None for eval.
        scaler (GradScaler): Gradient scaler for mixed-precision training.
        device (torch.device): Device to run computations on (e.g., 'cuda', 'cpu').
        stage (str): Current stage, one of ['Train', 'Val', 'Test'].
        criterion_l1 (torch.nn.Module): L1 loss criterion (e.g., SmoothL1Loss).
        criterion_iou (callable): Function to compute IoU, expected to return IoU values
                                (higher is better). The training loss uses 1 - IoU.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        overfit (bool): If True, enables overfitting mode (e.g., no scheduler step).
        model_type (str): Type of model, e.g., "single_frame" or "temporal".

    Returns:
        dict: A dictionary containing metrics for the epoch.
              For training: {"loss": avg_loss, "iou": avg_iou, "f1": avg_f1,
                             "iter_costs": per_batch_losses, 
                             "batch_times": per_batch_processing_times}
              For validation/test: {"loss": avg_loss, "iou": avg_iou, "f1": avg_f1}
    """
    training = (stage == "Train")
    # Use no_grad in evaluation to skip CPU graph overhead
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    model.train(training) # Set model mode

    total_loss = 0.0
    total_iou  = 0.0
    epoch_tp = 0
    epoch_fp = 0
    epoch_fn = 0
    n_samples_processed = 0 # Total samples processed to correctly average IoU and Loss

    iter_costs_epoch = [] # Collect per-batch losses for training
    batch_times_epoch = [] # Collect per-batch processing times for training

    pbar = tqdm(loader, desc=stage, leave=False)
    with grad_ctx:
        for batch_idx, batch_data in enumerate(pbar):
            start_time_batch = time.time()

            if model_type == "temporal":
                # Assuming batch_data is (sequences, targets_bbox)
                # sequences shape: (batch, seq_len, C, H, W), targets_bbox shape: (batch, 4)
                inputs, targets_bbox = batch_data
                inputs = inputs.to(device, non_blocking=True)
                targets_bbox = targets_bbox.to(device, non_blocking=True)
            else: # single_frame
                # Assuming batch_data is (images, targets_bbox)
                # images shape: (batch, C, H, W), targets_bbox shape: (batch, 4)
                inputs, targets_bbox = batch_data
                inputs = inputs.to(device, non_blocking=True)
                targets_bbox = targets_bbox.to(device, non_blocking=True)

            current_batch_size = inputs.size(0)

            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(inputs) # Predicted bounding boxes
                loss_l1 = criterion_l1(outputs, targets_bbox)
                
                # Calculate IoU - criterion_iou is expected to be bbox_iou directly
                # It should return a tensor of IoU values for the batch, shape (batch_size,)
                iou_scores_for_batch = criterion_iou(outputs, targets_bbox) 
                
                # Ensure iou_scores_for_batch is a tensor, handle potential scalar return for batch_size=1
                if not isinstance(iou_scores_for_batch, torch.Tensor):
                    iou_scores_for_batch = torch.tensor([iou_scores_for_batch], device=outputs.device)
                elif iou_scores_for_batch.ndim == 0: # If it's a scalar tensor
                    iou_scores_for_batch = iou_scores_for_batch.unsqueeze(0)

                batch_iou_loss = (1.0 - iou_scores_for_batch).mean() # Loss is 1 - IoU
                total_combined_loss = loss_l1 + batch_iou_loss # Example: combine losses

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_combined_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                iter_costs_epoch.append(total_combined_loss.item())
                batch_times_epoch.append(time.time() - start_time_batch)

            total_loss += total_combined_loss.item() * current_batch_size
            total_iou  += iou_scores_for_batch.sum().item() # Sum of IoUs in the batch
            n_samples_processed += current_batch_size
            
            # F1 Score calculation parts
            tp_batch = (iou_scores_for_batch >= IOU_THRESHOLD_F1).sum().item()
            # For each item in the batch, if IoU < threshold, it's both a False Positive (bad prediction)
            # and a False Negative (ground truth not found accurately)
            # This assumes one prediction per ground truth.
            fp_batch = (iou_scores_for_batch < IOU_THRESHOLD_F1).sum().item()
            fn_batch = (iou_scores_for_batch < IOU_THRESHOLD_F1).sum().item()
            
            epoch_tp += tp_batch
            epoch_fp += fp_batch
            epoch_fn += fn_batch

            pbar.set_postfix(loss=f"{total_combined_loss.item():.4f}", iou=f"{iou_scores_for_batch.mean().item():.4f}")

    if training and scheduler and not overfit:
        scheduler.step()

    avg_loss = total_loss / n_samples_processed if n_samples_processed > 0 else 0.0
    avg_iou  = total_iou / n_samples_processed if n_samples_processed > 0 else 0.0
    
    # Calculate epoch-level F1 score
    precision_epoch = epoch_tp / (epoch_tp + epoch_fp) if (epoch_tp + epoch_fp) > 0 else 0.0
    recall_epoch = epoch_tp / (epoch_tp + epoch_fn) if (epoch_tp + epoch_fn) > 0 else 0.0
    avg_f1_epoch = 2 * (precision_epoch * recall_epoch) / (precision_epoch + recall_epoch) if (precision_epoch + recall_epoch) > 0 else 0.0
    
    if training:
        return {"loss": avg_loss, "iou": avg_iou, "f1": avg_f1_epoch,
                "iter_costs": iter_costs_epoch, "batch_times": batch_times_epoch}
    else:
        return {"loss": avg_loss, "iou": avg_iou, "f1": avg_f1_epoch}


def visualize_samples(model, data_root, split, exp_dir, n, device, transform_to_apply,
                      is_overfit_visualization=False, overfit_pool_size=0,
                      model_type="single_frame", seq_len=1, out_subdir=None):
    """
    Visualizes model predictions on a few sample images and saves them.

    Draws predicted bounding boxes (green) and ground truth bounding boxes (red)
    on the images and saves them to a specified directory.

    Args:
        model (torch.nn.Module): The trained model to use for predictions.
        data_root (str or Path): Root directory of the dataset.
        split (str): Data split to visualize (e.g., 'val', 'test', or 'train' for overfit).
        exp_dir (Path): Base directory for saving experiment artifacts (visualizations
                        will be saved in a subdirectory here).
        n (int): Number of samples to visualize.
        device (torch.device): Device to run model inference on.
        transform_to_apply (callable): Albumentations transform to apply to images before inference
                                     (typically the validation transform).
        is_overfit_visualization (bool): If True, and split is 'train', samples from the
                                       beginning of the train set (potential overfit pool).
        overfit_pool_size (int): If `is_overfit_visualization` is True, this specifies the
                                 size of the pool from which to sample (e.g., C.OVERFIT_N_SAMPLES).
        model_type (str): Type of the model ("single_frame" or "temporal").
        seq_len (int): Sequence length, used if `model_type` is "temporal".
        out_subdir (str, optional): Name of the subdirectory within `exp_dir/predictions`
                                    to save these specific visualizations.
                                    (e.g., "val_epoch_10", "test_final").
    """
    img_dir = Path(data_root)/"images"/split
    lbl_dir = Path(data_root)/"labels"/split
    # Create base predictions directory
    base_pred_dir = exp_dir/"predictions"
    # Subdirectory for this visualization (e.g., 'val_10', 'val_final')
    out_dir = base_pred_dir/(out_subdir if out_subdir else "") if out_subdir else base_pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files_in_split_sorted = sorted(list(img_dir.glob("*.jpg")))

    candidate_pool = all_files_in_split_sorted
    if is_overfit_visualization and split == "train": # Ensure overfit pool is from train split
        if overfit_pool_size > 0 and overfit_pool_size < len(all_files_in_split_sorted):
            candidate_pool = all_files_in_split_sorted[:overfit_pool_size]
            print(f"[INFO] Viz (Overfit): Using first {len(candidate_pool)} images from '{split}' split as candidate pool.")
        else:
            print(f"[INFO] Viz (Overfit): Using all {len(candidate_pool)} images from '{split}' split as candidate pool (overfit_pool_size={overfit_pool_size}).")

    if not candidate_pool:
        print(f"[WARN] Viz: No images found in the candidate pool for split {split} (path: {img_dir}).")
        return

    num_to_sample = min(n, len(candidate_pool))
    if num_to_sample <= 0:
        print(f"[WARN] Viz: Number of samples to visualize is {num_to_sample}. Skipping.")
        return
    
    # sample_paths_current_frames are Path objects to the *current* frame to be visualized
    sample_paths_current_frames = random.sample(candidate_pool, num_to_sample) 
    model.eval()

    for img_p_path_obj in tqdm(sample_paths_current_frames, desc="Viz", leave=False): # This is the current frame Path obj
        im_bgr = cv2.imread(str(img_p_path_obj)) # Load current frame in BGR for drawing
        if im_bgr is None:
            print(f"[WARN] Viz: Failed to load image {img_p_path_obj}. Skipping.")
            continue

        if model_type == "temporal":
            # For temporal, we need to load seq_len frames ending with img_p_path_obj
            # Re-fetch all sorted paths for robust indexing, could be optimized if img_dir content is static
            all_potential_image_paths_for_seq = sorted(list(img_dir.glob("*.jpg")))
            
            try:
                current_idx = all_potential_image_paths_for_seq.index(img_p_path_obj)
            except ValueError:
                print(f"[WARN] Viz: Current image {img_p_path_obj} not found in sorted list of {split} dir. Skipping.")
                continue

            if current_idx < seq_len - 1:
                # print(f"[INFO] Viz: Not enough preceding frames for {img_p_path_obj} (idx {current_idx}, seq_len {seq_len}). Skipping.")
                continue # Skip if not enough history
            
            img_paths_sequence_objects = all_potential_image_paths_for_seq[current_idx - seq_len + 1 : current_idx + 1]
            
            # Check if the sequence is from the same video (prefix check)
            expected_prefix = "_".join(img_paths_sequence_objects[-1].stem.split('_')[:-1]) # Assumes filenaming like videoID_frameXXXX
            consistent = all(p.stem.startswith(expected_prefix) for p in img_paths_sequence_objects)
            if not consistent:
                # print(f"[INFO] Viz: Frame sequence for {img_p_path_obj} is not from the same video. Skipping.")
                continue # Skip if sequence broken

            frames_for_model = []
            for p_obj in img_paths_sequence_objects:
                img_rgb_seq_frame = cv2.cvtColor(cv2.imread(str(p_obj)), cv2.COLOR_BGR2RGB)
                transformed_seq_frame = transform_to_apply(image=img_rgb_seq_frame)
                frames_for_model.append(transformed_seq_frame["image"])
            
            input_tensor = torch.stack(frames_for_model).unsqueeze(0).to(device) # [1, T, C, H, W]

        else: # single_frame model type
            img_rgb_for_model = cv2.cvtColor(im_bgr.copy(), cv2.COLOR_BGR2RGB) # RGB for model
            transformed = transform_to_apply(image=img_rgb_for_model)
            input_tensor = transformed["image"].unsqueeze(0).to(device) # [1, C, H, W]
        # Common logic for prediction and drawing using im_bgr (loaded current frame)
        lbl_p_ground_truth = lbl_dir / f"{img_p_path_obj.stem}.txt"
        
        with torch.no_grad():
            pred_bbox_norm = model(input_tensor)[0].cpu().numpy() # [4] (cx, cy, w, h)

        h_img, w_img = im_bgr.shape[:2]
        px_norm, py_norm, pw_norm, ph_norm = pred_bbox_norm
        
        # Convert normalized predicted bbox to pixel coordinates
        px1 = int((px_norm - pw_norm / 2) * w_img)
        py1 = int((py_norm - ph_norm / 2) * h_img)
        px2 = int((px_norm + pw_norm / 2) * w_img)
        py2 = int((py_norm + ph_norm / 2) * h_img)
        cv2.rectangle(im_bgr, (px1, py1), (px2, py2), (0, 0, 255), 2) # Red for prediction

        if lbl_p_ground_truth.exists() and lbl_p_ground_truth.stat().st_size > 0:
            try:
                with open(lbl_p_ground_truth, 'r') as f_gt:
                    line = f_gt.readline().strip()
                    if line: # Check if line is not empty
                        gt_values = list(map(float, line.split()))
                        
                        gtx_norm, gty_norm, gtw_norm, gth_norm = 0,0,0,0
                        if len(gt_values) == 5: # class_id, cx, cy, w, h
                            _, gtx_norm, gty_norm, gtw_norm, gth_norm = gt_values
                        elif len(gt_values) == 4: # cx, cy, w, h (no class_id)
                            gtx_norm, gty_norm, gtw_norm, gth_norm = gt_values
                        else:
                            print(f"[WARN] Viz: Label file {lbl_p_ground_truth} has unexpected number of values: {len(gt_values)}. Skipping GT drawing for this image.")
                            gt_values = [] # Ensure it doesn't proceed to draw

                        if gt_values: # If values were successfully parsed
                            # Convert normalized GT bbox to pixel coordinates
                            gx1 = int((gtx_norm - gtw_norm / 2) * w_img)
                            gy1 = int((gty_norm - gth_norm / 2) * h_img)
                            gx2 = int((gtx_norm + gtw_norm / 2) * w_img)
                            gy2 = int((gty_norm + gth_norm / 2) * h_img)
                            cv2.rectangle(im_bgr, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2) # Green for actual
            except Exception as e:
                print(f"[WARN] Viz: Error parsing or drawing GT for label file {lbl_p_ground_truth}: {e}")
        
        out_path_prediction_img = out_dir / img_p_path_obj.name
        cv2.imwrite(str(out_path_prediction_img), im_bgr)


# ─── Wrapper for experiment_runner compatibility (New Function) ───────────────
def train_uav_experiment_iteration(optimizer_name, base_visuals_dir, train_ds, val_ds, test_ds,
                                   current_config_name, current_model_type, current_model_architecture,
                                   current_seq_len, current_batch_size, current_epochs, current_lr,
                                   current_overfit_active=False, current_overfit_n_samples=0,
                                   return_settings=False):
    """
    Runs a single iteration of the UAV detection training experiment for a given optimizer
    and a specific experiment configuration.

    This function is designed to be compatible with the structure expected by
    `experiment_runner.py`. It handles model initialization, uses pre-loaded datasets
    for DataLoaders, executes the training and validation loops, performs final testing,
    and collects all necessary metrics for plotting and reporting by `experiment_runner`.

    Args:
        optimizer_name (str): Name of the optimizer to use (e.g., \"ADAMW\").
        base_visuals_dir (Path): Base directory where visualization outputs for this
                                 experiment run should be saved. (This will be config_name/optimizer_name specific path)
        train_ds (Dataset): Pre-loaded training dataset (potentially filtered for overfit).
        val_ds (Dataset): Pre-loaded validation dataset (potentially filtered for overfit).
        test_ds (Dataset): Pre-loaded test dataset (potentially filtered for overfit).
        current_config_name (str): Name of the current experiment configuration (e.g., \\\"single_frame_default\\\").
        current_model_type (str): Model type for this run (\\\"single_frame\\\" or \\\"temporal\\\").
        current_model_architecture (str): ResNet architecture for this run (e.g., \\\"resnet18\\\", \\\"resnet50\\\").
        current_seq_len (int): Sequence length for this run.
        current_batch_size (int): Batch size for this run.
        current_epochs (int): Number of epochs for this run.
        current_lr (float): Learning rate for this run.
        current_overfit_active (bool): If True, enables overfitting mode for this iteration.
        current_overfit_n_samples (int): Number of samples to use for overfitting, if active.
        return_settings (bool): If True, also returns a dictionary of settings specific
                                to this iteration (e.g., optimizer parameters).
                                This is used by `experiment_runner`.

    Returns:
        tuple: A tuple containing various metrics and results arrays required by
               `experiment_runner.py`. The structure is:
               (val_loss_h, val_accuracy_h, val_f1_h, val_auc_h,
                iter_costs_h, walltime_h, grad_norms_h, layer_names_h,
                test_metrics, steps_per_epoch_h, train_metrics_hist)
               If `return_settings` is True, an additional settings dictionary is appended.
               For this UAV task, 'accuracy' related fields (val_accuracy_h, etc.)
               are populated with IoU scores. 'f1_score' is now calculated.
    """
    print(f"\n--- Starting a new run for Optimizer: {optimizer_name}, Config: {current_config_name} ---")
    print(f"    Model Type: {current_model_type}, Architecture: {current_model_architecture}, SeqLen: {current_seq_len}, Batch: {current_batch_size}, Epochs: {current_epochs}, LR: {current_lr}")

    # DataLoaders - use current_batch_size
    loader_kwargs = dict(
        num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY,
        persistent_workers=C.PERSISTENT_WORKERS # Removed prefetch_factor from here
    )
    if C.NUM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = C.PREFETCH_FACTOR
        
    # Increase train batch size (up to 2x) to better fill GPU
    train_bs = min(len(train_ds), current_batch_size)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=train_bs, **loader_kwargs)

    val_bs = min(len(val_ds), current_batch_size)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=val_bs, **loader_kwargs)
    
    test_bs = min(len(test_ds), current_batch_size * 2)
    test_loader_final = DataLoader(test_ds, shuffle=False, batch_size=test_bs, **loader_kwargs) 

    # Model - use current_model_type, current_model_architecture, and current_seq_len
    if current_model_type == "temporal":
        model = TemporalResNetRegressor(model_architecture=current_model_architecture,
                                        seq_len=current_seq_len,
                                        overfit=current_overfit_active).to(device)
    else: # single_frame
        model = SingleFrameResNetRegressor(model_architecture=current_model_architecture,
                                           overfit=current_overfit_active).to(device)
    
    layer_names = get_layer_names(model)

    criterion_l1 = nn.SmoothL1Loss()

    if optimizer_name.upper() == "ADAMW":
        optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=0.0 if current_overfit_active else 1e-4)
    else:
        print(f"[WARN] Optimizer {optimizer_name} not explicitly configured, using AdamW with LR={current_lr}.")
        optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=0.0 if current_overfit_active else 1e-4)

    scheduler = None
    scheduler_config = C.SCHEDULER_PARAMS.get(optimizer_name, {})
    if not current_overfit_active and scheduler_config.get("scheduler") == "CosineAnnealingLR":
        t_max_scheduler = scheduler_config.get("params", {}).get("T_max", current_epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_scheduler)
    
    scaler = GradScaler(device=device.type)

    history_train_loss = []
    history_train_iou = []
    history_train_f1 = [] # Added for F1
    history_val_loss_steps = [] 
    history_val_iou_steps = []
    history_val_f1_steps = [] # Added for F1
    val_epochs_recorded = []
    all_iter_costs_for_run = [] 
    all_batch_times_for_run = []
    best_val_iou = 0.0
    
    # Create checkpoints directory for this specific optimizer run
    checkpoints_dir = base_visuals_dir / optimizer_name / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    validate_every_n_epochs = max(1, int(current_epochs * C.VALIDATE_EVERY_EPOCH_FACTOR))
    #if current_overfit_active:
    #    validate_every_n_epochs = 1

    for epoch in range(1, current_epochs + 1):
        print(f"\nEpoch {epoch}/{current_epochs} for {optimizer_name} ({current_config_name})")
        
        train_epoch_results = run_epoch(model, train_loader, optimizer, scaler, device, "Train",
                                        criterion_l1, bbox_iou, scheduler, current_overfit_active, current_model_type)
        history_train_loss.append(train_epoch_results["loss"])
        history_train_iou.append(train_epoch_results["iou"])
        history_train_f1.append(train_epoch_results["f1"]) # Store F1
        all_iter_costs_for_run.extend(train_epoch_results["iter_costs"])
        all_batch_times_for_run.extend(train_epoch_results["batch_times"])

        if epoch % validate_every_n_epochs == 0:
            val_metrics_epoch = run_epoch(model, val_loader, None, scaler, device, "Val",
                                          criterion_l1, bbox_iou, None, current_overfit_active, current_model_type)
            history_val_loss_steps.append(val_metrics_epoch["loss"])
            history_val_iou_steps.append(val_metrics_epoch["iou"])
            history_val_f1_steps.append(val_metrics_epoch["f1"]) # Store F1
            val_epochs_recorded.append(epoch)
            print(f"  Val   → loss {val_metrics_epoch['loss']:.4f}, IoU {val_metrics_epoch['iou']:.4f}, F1 {val_metrics_epoch['f1']:.4f}")

            if val_metrics_epoch["iou"] > best_val_iou: # Assuming best model is still based on IoU
                best_val_iou = val_metrics_epoch["iou"]
                print(f"  ↑ New best validation IoU: {best_val_iou:.4f}")
                # Save PyTorch checkpoint
                checkpoint_pth_path = checkpoints_dir / f"model_{current_config_name}_{optimizer_name}_epoch{epoch}_valiou{best_val_iou:.4f}.pth"
                torch.save(model.state_dict(), checkpoint_pth_path)
                print(f"    PyTorch Checkpoint saved to {checkpoint_pth_path}")

                # Save ONNX checkpoint
                onnx_checkpoint_filename = f"model_{current_config_name}_{optimizer_name}_epoch{epoch}_valiou{best_val_iou:.4f}.onnx"
                onnx_checkpoint_path = checkpoints_dir / onnx_checkpoint_filename
                try:
                    model.eval() # Ensure model is in eval mode for ONNX export
                    if current_model_type == "temporal":
                        dummy_input = torch.randn(1, current_seq_len, 3, 224, 224, device=device)
                        input_names = ['input_sequence']
                        output_names = ['output_bbox']
                        dynamic_axes={'input_sequence': {0: 'batch_size', 1: 'sequence_length'}, 'output_bbox': {0: 'batch_size'}}
                    else: # single_frame
                        dummy_input = torch.randn(1, 3, 224, 224, device=device)
                        input_names = ['input_image']
                        output_names = ['output_bbox']
                        dynamic_axes={'input_image': {0: 'batch_size'}, 'output_bbox': {0: 'batch_size'}}

                    torch.onnx.export(model,
                                      dummy_input,
                                      str(onnx_checkpoint_path),
                                      export_params=True,
                                      opset_version=11, 
                                      do_constant_folding=True,
                                      input_names=input_names,
                                      output_names=output_names,
                                      dynamic_axes=dynamic_axes
                                     )
                    print(f"    ONNX Checkpoint saved to: {onnx_checkpoint_path}")
                except Exception as e:
                    print(f"    [WARN] Failed to save ONNX checkpoint: {e}")
                # Restore model to train mode if it was training before (run_epoch handles this for the main loop)
                # However, since this is inside the validation block, model should be in eval.
                # If further training epochs follow, model.train() will be called by run_epoch.
            
            if C.VISUALIZATION_SAMPLES > 0:
                # base_visuals_dir is already config_results_dir/visuals. We need to create a sub-folder for the optimizer within it for these viz
                viz_exp_dir_optimizer_specific = base_visuals_dir / optimizer_name
                out_subdir_viz = f"val_epoch_{epoch}" if epoch < current_epochs else f"val_final"
                visualize_samples(model, C.DATA_ROOT, "train" if current_overfit_active else "val", viz_exp_dir_optimizer_specific, C.VISUALIZATION_SAMPLES, device, val_tf,
                                  is_overfit_visualization=current_overfit_active,
                                  overfit_pool_size=current_overfit_n_samples if current_overfit_active else 0,
                                  model_type=current_model_type, seq_len=current_seq_len,
                                  out_subdir=out_subdir_viz)
        
    # --- Final Test Evaluation ---
    print(f"\n[INFO] Preparing for final test run for {optimizer_name} ({current_config_name})...")
    
    test_eval_start_time = time.time()
    test_metrics_final = run_epoch(model, test_loader_final, None, scaler, device, "Test",
                                   criterion_l1, bbox_iou, None, C.OVERFIT, current_model_type)
    test_eval_time = time.time() - test_eval_start_time
    print(f"  Test  → loss {test_metrics_final['loss']:.4f}, IoU {test_metrics_final['iou']:.4f}, F1 {test_metrics_final['f1']:.4f}, Eval time: {test_eval_time:.2f}s")

    # --- Save ONNX Model ---
    # base_visuals_dir is effectively config_results_dir/visuals. We want to save ONNX in config_results_dir/optimizer_name/
    onnx_save_dir = base_visuals_dir.parent / optimizer_name 
    onnx_save_dir.mkdir(parents=True, exist_ok=True)
    onnx_filename = f"{current_config_name}_{optimizer_name}_model.onnx"
    onnx_path = onnx_save_dir / onnx_filename

    try:
        model.eval() # Ensure model is in eval mode
        # Create dummy input based on model type
        if current_model_type == "temporal":
            dummy_input = torch.randn(1, current_seq_len, 3, 224, 224, device=device)
            input_names = ['input_sequence']
            output_names = ['output_bbox']
        else: # single_frame
            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            input_names = ['input_image']
            output_names = ['output_bbox']

        torch.onnx.export(model,
                          dummy_input,
                          str(onnx_path),
                          export_params=True,
                          opset_version=11, # Or a version compatible with your target environment
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes={'input_sequence': {0: 'batch_size', 1: 'sequence_length'}, # if temporal
                                        'input_image': {0: 'batch_size'}, # if single_frame
                                        'output_bbox': {0: 'batch_size'}} if current_model_type == "temporal" else \
                                       {'input_image': {0: 'batch_size'},
                                        'output_bbox': {0: 'batch_size'}}
                         )
        print(f"  [INFO] ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"  [WARN] Failed to save ONNX model: {e}")


    if C.VISUALIZATION_SAMPLES > 0:
        viz_split_final = "train" if current_overfit_active else "test"
        viz_exp_dir_test_optimizer_specific = base_visuals_dir / optimizer_name
        out_subdir_test_viz = f"test_final"
        visualize_samples(model, C.DATA_ROOT, viz_split_final, viz_exp_dir_test_optimizer_specific, C.VISUALIZATION_SAMPLES, device, val_tf,
                          is_overfit_visualization=current_overfit_active,
                          overfit_pool_size=current_overfit_n_samples if current_overfit_active else 0,
                          model_type=current_model_type, seq_len=current_seq_len,
                          out_subdir=out_subdir_test_viz)
    
    val_loss_h = history_val_loss_steps
    val_iou_h = history_val_iou_steps 
    val_f1_h = history_val_f1_steps # Use stored F1
    val_auc_h = [0.0] * len(history_val_iou_steps) # AUC remains 0.0
    iter_costs_h = all_iter_costs_for_run
    cumulative_batch_times_h = list(np.cumsum(all_batch_times_for_run))
    grad_norms_h = {} 
    layer_names_h = layer_names
    test_metrics_for_runner = {
        'loss': test_metrics_final['loss'], 'accuracy': test_metrics_final['iou'], 
        'f1_score': test_metrics_final['f1'], 'auc': 0.0, # Use stored F1, AUC remains 0.0
        'eval_time_seconds': test_eval_time
    }
    steps_per_epoch_h = len(train_loader)
    train_metrics_hist_adapted = {
        'epoch': list(range(1, current_epochs + 1)), 'train_loss': history_train_loss,
        'train_iou': history_train_iou, 'train_accuracy': history_train_iou, # accuracy is IoU
        'train_f1_score': history_train_f1, 'train_auc': [0.0] * len(history_train_iou) # Use stored F1, AUC remains 0.0
    }
    results_tuple = (
        val_loss_h, val_iou_h, val_f1_h, val_auc_h,
        iter_costs_h, cumulative_batch_times_h, grad_norms_h, layer_names_h,
        test_metrics_for_runner, steps_per_epoch_h, train_metrics_hist_adapted
    )
    if return_settings:
        settings_for_runner = {
            "optimizer_params": C.SCHEDULER_PARAMS.get(optimizer_name, {}),
            "config_name": current_config_name, "model_type": current_model_type,
            "model_architecture": current_model_architecture, # Added
            "seq_len": current_seq_len, "batch_size": current_batch_size,
            "epochs": current_epochs, "learning_rate": current_lr,
        }
        return results_tuple + (settings_for_runner,)
    else:
        return results_tuple

if __name__ == "__main__":
    print(f"[INFO] Starting UAV Detection Training Script using configurations from config_uav.py")
    print(f"[INFO] Base experiment name: {C.EXPERIMENT_NAME_BASE}")
    print(f"[INFO] Found {len(C.EXPERIMENT_CONFIGURATIONS)} experiment configurations to run.")

    # --- Iterate through experiment configurations defined in config_uav.py ---
    all_configs_results = []
    for exp_config in C.EXPERIMENT_CONFIGURATIONS:
        config_name = exp_config["name"]
        model_type = exp_config["MODEL_TYPE"]
        # Get model_architecture, default to resnet50 if not specified for backward compatibility
        model_architecture = exp_config.get("MODEL_ARCHITECTURE", "resnet50") 
        seq_len = exp_config["SEQ_LEN"]
        
        # Override global settings if specified in the current exp_config
        batch_size = exp_config.get("BATCH_SIZE", C.BATCH_SIZE)
        epochs = exp_config.get("EPOCHS", C.EPOCHS)
        lr = exp_config.get("LEARNING_RATE", C.LEARNING_RATE)
        overfit_this_config = exp_config.get("OVERFIT", C.OVERFIT) # Allow per-config overfit setting
        overfit_n_samples_this_config = exp_config.get("OVERFIT_N_SAMPLES", C.OVERFIT_N_SAMPLES)
        visualization_samples_this_config = exp_config.get("VISUALIZATION_SAMPLES", C.VISUALIZATION_SAMPLES)

        print(f"\\n════════════════════════════════════════════════════════════════════════════════")
        print(f"[INFO] Starting Experiment Configuration: {config_name}")
        print(f"       Type: {model_type}, Arch: {model_architecture}, SeqLen: {seq_len}, Batch: {batch_size}, Epochs: {epochs}, LR: {lr}")
        print(f"       Overfitting: {overfit_this_config} (with {overfit_n_samples_this_config} samples if True)")
        print(f"       Visualization Samples: {visualization_samples_this_config}")
        print(f"════════════════════════════════════════════════════════════════════════════════")

        # --- Dataset Initialization for Current Configuration ---
        is_temporal_config = (model_type == "temporal")
        effective_seq_len = seq_len if is_temporal_config else 1
        
        load_n_for_overfit = None
        # train_subset_fraction for FrameBBOXDataset; 1.0 means use all (after load_n_samples if applicable)
        train_fraction_to_use = 1.0 

        if overfit_this_config:
            print(f"  [OVERFIT MODE] Configuring datasets to load up to {overfit_n_samples_this_config} samples using 'load_n_samples'.")
            load_n_for_overfit = overfit_n_samples_this_config
        else: 
            if C.TRAIN_SUBSET_FRACTION < 1.0:
                print(f"  Configuring train dataset to use {C.TRAIN_SUBSET_FRACTION*100}% of data via 'train_subset_fraction'.")
                train_fraction_to_use = C.TRAIN_SUBSET_FRACTION
        
        print(f"  Initializing datasets for {config_name} (Temporal: {is_temporal_config}, SeqLen: {effective_seq_len}, OverfitLoadN: {load_n_for_overfit}, TrainFrac: {train_fraction_to_use if not overfit_this_config else 'N/A'})")

        train_ds_for_run = FrameBBOXDataset(
            root=C.DATA_ROOT, split='train', transform=train_tf,
            seq_len=effective_seq_len, is_temporal=is_temporal_config,
            load_n_samples=load_n_for_overfit, 
            train_subset_fraction=train_fraction_to_use if not overfit_this_config else 1.0 
        )
        if overfit_this_config:
            # Overfitting: reuse the train dataset for val and test to avoid reloading
            val_ds_for_run = train_ds_for_run
            test_ds_for_run = train_ds_for_run
        else:
            val_ds_for_run = FrameBBOXDataset(
                root=C.DATA_ROOT, split='val', transform=val_tf,
                seq_len=effective_seq_len, is_temporal=is_temporal_config,
                load_n_samples=load_n_for_overfit
            )
            test_ds_for_run = FrameBBOXDataset(
                root=C.DATA_ROOT, split='test', transform=val_tf,
                seq_len=effective_seq_len, is_temporal=is_temporal_config,
                load_n_samples=load_n_for_overfit
            )
        
        # Report if the number of loaded samples is less than requested during overfit.
        if overfit_this_config and load_n_for_overfit is not None and load_n_for_overfit > 0:
            if len(train_ds_for_run) < load_n_for_overfit:
                 print(f"    [INFO] Train dataset: Loaded {len(train_ds_for_run)} of {load_n_for_overfit} requested samples. (May be fewer files available or dataset limit)")
            if len(val_ds_for_run) < load_n_for_overfit:
                 print(f"    [INFO] Val dataset: Loaded {len(val_ds_for_run)} of {load_n_for_overfit} requested samples. (May be fewer files available or dataset limit)")
            if len(test_ds_for_run) < load_n_for_overfit:
                 print(f"    [INFO] Test dataset: Loaded {len(test_ds_for_run)} of {load_n_for_overfit} requested samples. (May be fewer files available or dataset limit)")
        
        print(f"  Effective train dataset size for this run: {len(train_ds_for_run)}")
        print(f"  Effective val dataset size for this run: {len(val_ds_for_run)}")
        print(f"  Effective test dataset size for this run: {len(test_ds_for_run)}")

        # --- Define paths and titles for this specific configuration ---
        config_results_dir = Path(C.RESULTS_BASE_DIR) / C.EXPERIMENT_NAME_BASE / config_name
        config_visuals_dir = config_results_dir / "visuals"
        config_visuals_dir.mkdir(parents=True, exist_ok=True)

        # --- Prepare partial function for experiment_runner --- 
        # Pass all current config-specific parameters to the iteration function
        partial_train_iteration_func = functools.partial(
             train_uav_experiment_iteration,
             base_visuals_dir=config_visuals_dir, # Add base_visuals_dir here
             train_ds=train_ds_for_run,
             val_ds=val_ds_for_run,
             test_ds=test_ds_for_run,
             current_config_name=config_name,
             current_model_type=model_type,
             current_model_architecture=model_architecture, # Pass architecture
             current_seq_len=seq_len,
             current_batch_size=batch_size,
             current_epochs=epochs,
             current_lr=lr,
             current_overfit_active=overfit_this_config,
             current_overfit_n_samples=overfit_n_samples_this_config
         )

        # Allow overriding plot titles from config
        loss_plot_title = exp_config.get("LOSS_PLOT_TITLE", C.LOSS_PLOT_TITLE)
        iou_plot_title = exp_config.get("IOU_PLOT_TITLE", C.IOU_PLOT_TITLE) # Assuming IoU is primary accuracy metric
        # Add F1 plot title if desired, e.g., in config_uav.py
        f1_plot_title = exp_config.get("F1_PLOT_TITLE", f"F1 Score vs. Epoch ({config_name})")


        # --- Run experiments for this configuration using experiment_runner --- 
        results_for_config = run_experiments(
            train_experiment=partial_train_iteration_func,
            optimizer_names=C.OPTIMIZERS,                       # Renamed from optimizers_to_test
            num_runs=C.RUNS_PER_OPTIMIZER,                      # Renamed from num_runs_per_optimizer
            epochs=epochs, 
            results_dir=config_results_dir, 
            visuals_dir=config_visuals_dir, 
            experiment_title=f"{C.EXPERIMENT_NAME_BASE}_{config_name}", # Renamed from base_experiment_name
            # scheduler_params=C.SCHEDULER_PARAMS, # Removed
            loss_title=loss_plot_title,                         # Unpacked from loss_plot_params
            loss_ylabel=C.LOSS_PLOT_YLABEL,                     # Unpacked from loss_plot_params
            acc_title=iou_plot_title,                           # Unpacked from accuracy_plot_params
            acc_ylabel=C.IOU_PLOT_YLABEL,                       # Unpacked from accuracy_plot_params
            f1_title=f1_plot_title,                             # Added
            # save_config_to_json=True, # Removed
            experiment_settings={**C.__dict__, **exp_config}    # Renamed from config_dict_to_save
        )
        all_configs_results.append(results_for_config)

    print("\n[INFO] All experiment configurations completed.")
