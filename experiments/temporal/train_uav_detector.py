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
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
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
from network import ResNet50Regressor, TemporalResNet50Regressor # This is from temporal/network.py
import experiments.temporal.config_uav as C

# Local imports from the current 'temporal' package
from uav_dataset import FrameBBOXDataset
from uav_metrics import bbox_iou

# Imports from the \'experiments\' module
from experiments.train_utils import get_layer_names # evaluate_model might need adaptation for regression
from experiments.experiment_runner import run_experiments # This will be the main driver
from experiments.plotting import plot_seaborn_style_with_error_bars, setup_plot_style, plot_resource_usage
from experiments.utils.resource_tracker import ResourceTracker, save_resource_info # Optional resource tracking

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
torch.backends.cudnn.benchmark = False # Benchmark can be True if input sizes don't change, for speed

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
              For training: {"loss": avg_loss, "iou": avg_iou, 
                             "iter_costs": per_batch_losses, 
                             "batch_times": per_batch_processing_times}
              For validation/test: {"loss": avg_loss, "iou": avg_iou}
    """
    training = (stage == "Train")
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_iou  = 0.0
    n = 0
    iter_costs_epoch = [] # Collect per-batch losses for training
    batch_times_epoch = [] # Collect per-batch processing times for training

    pbar = tqdm(loader, desc=stage, leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        batch_start_time = time.time() # Start time for batch processing

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=scaler.is_enabled()):
            preds = model(inputs) 
            loss_l1 = criterion_l1(preds, targets)
            loss_iou_val = criterion_iou(preds, targets) 
            loss_iou = 1.0 - loss_iou_val.mean() 
            loss = loss_l1 + loss_iou
        
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter_costs_epoch.append(loss.item()) # Store loss for this iteration
        
        batch_end_time = time.time() # End time for batch processing
        if training: # Only record batch times during training for walltime plots
            batch_times_epoch.append(batch_end_time - batch_start_time)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        
        iou_metric_val = criterion_iou(preds.detach(), targets.detach()).mean().item()
        total_iou += iou_metric_val * batch_size 
        n += batch_size
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou_metric_val:.4f}")
        
        if overfit and batch_idx >= 0: 
            pass 

    if training and scheduler and not overfit:
        scheduler.step()

    avg_loss = total_loss / n if n > 0 else 0.0
    avg_iou  = total_iou / n if n > 0 else 0.0
    
    if training:
        return {"loss": avg_loss, "iou": avg_iou, "iter_costs": iter_costs_epoch, "batch_times": batch_times_epoch}
    else:
        return {"loss": avg_loss, "iou": avg_iou}


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
        cv2.rectangle(im_bgr, (px1, py1), (px2, py2), (0, 255, 0), 2) # Green for prediction

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
                            cv2.rectangle(im_bgr, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2) # Red for ground truth
            except Exception as e:
                print(f"[WARN] Viz: Error parsing or drawing GT for label file {lbl_p_ground_truth}: {e}")
        
        out_path_prediction_img = out_dir / img_p_path_obj.name
        cv2.imwrite(str(out_path_prediction_img), im_bgr)


# ─── Wrapper for experiment_runner compatibility (New Function) ───────────────
def train_uav_experiment_iteration(optimizer_name, base_visuals_dir, train_ds, val_ds, test_ds,
                                   # New parameters for current configuration
                                   current_config_name, current_model_type, current_seq_len,
                                   current_batch_size, current_epochs, current_lr,
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
        current_config_name (str): Name of the current experiment configuration (e.g., \"single_frame_default\").
        current_model_type (str): Model type for this run (\"single_frame\" or \"temporal\").
        current_seq_len (int): Sequence length for this run.
        current_batch_size (int): Batch size for this run.
        current_epochs (int): Number of epochs for this run.
        current_lr (float): Learning rate for this run.
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
               are populated with IoU scores.
    """
    print(f"\n--- Starting a new run for Optimizer: {optimizer_name}, Config: {current_config_name} ---")
    print(f"    Model: {current_model_type}, SeqLen: {current_seq_len}, Batch: {current_batch_size}, Epochs: {current_epochs}, LR: {current_lr}")

    # DataLoaders - use current_batch_size
    loader_kwargs = dict(
        batch_size=current_batch_size, num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY,
        persistent_workers=C.PERSISTENT_WORKERS, prefetch_factor=C.PREFETCH_FACTOR
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # Model - use current_model_type and current_seq_len
    if current_model_type == "temporal":
        model = TemporalResNet50Regressor(seq_len=current_seq_len, overfit=C.OVERFIT).to(device)
    else: # single_frame
        model = ResNet50Regressor(overfit=C.OVERFIT).to(device)
    
    layer_names = get_layer_names(model)

    criterion_l1 = nn.SmoothL1Loss()

    if optimizer_name.upper() == "ADAMW":
        optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=0.0 if C.OVERFIT else 1e-4)
    else:
        print(f"[WARN] Optimizer {optimizer_name} not explicitly configured, using AdamW with LR={current_lr}.")
        optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=0.0 if C.OVERFIT else 1e-4)

    scheduler = None
    scheduler_config = C.SCHEDULER_PARAMS.get(optimizer_name, {})
    if not C.OVERFIT and scheduler_config.get("scheduler") == "CosineAnnealingLR":
        t_max_scheduler = scheduler_config.get("params", {}).get("T_max", current_epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_scheduler)
    
    scaler = GradScaler(device=device.type)

    history_train_loss = []
    history_train_iou = []
    history_val_loss_steps = [] 
    history_val_iou_steps = []
    val_epochs_recorded = []
    all_iter_costs_for_run = [] 
    all_batch_times_for_run = []
    best_val_iou = 0.0
    
    validate_every_n_epochs = max(1, int(current_epochs * C.VALIDATE_EVERY_EPOCH_FACTOR))
    if C.OVERFIT:
        validate_every_n_epochs = 1

    for epoch in range(1, current_epochs + 1):
        print(f"\nEpoch {epoch}/{current_epochs} for {optimizer_name} ({current_config_name})")
        
        train_epoch_results = run_epoch(model, train_loader, optimizer, scaler, device, "Train",
                                        criterion_l1, bbox_iou, scheduler, C.OVERFIT, current_model_type)
        history_train_loss.append(train_epoch_results["loss"])
        history_train_iou.append(train_epoch_results["iou"])
        all_iter_costs_for_run.extend(train_epoch_results["iter_costs"])
        all_batch_times_for_run.extend(train_epoch_results["batch_times"])

        if epoch % validate_every_n_epochs == 0 or epoch == current_epochs:
            val_metrics_epoch = run_epoch(model, val_loader, None, scaler, device, "Val",
                                          criterion_l1, bbox_iou, None, C.OVERFIT, current_model_type)
            history_val_loss_steps.append(val_metrics_epoch["loss"])
            history_val_iou_steps.append(val_metrics_epoch["iou"])
            val_epochs_recorded.append(epoch)
            print(f"  Val   → loss {val_metrics_epoch['loss']:.4f}, IoU {val_metrics_epoch['iou']:.4f}")

            if val_metrics_epoch["iou"] > best_val_iou:
                best_val_iou = val_metrics_epoch["iou"]
                print(f"  ↑ New best validation IoU: {best_val_iou:.4f}")
            
            if C.VISUALIZATION_SAMPLES > 0:
                # base_visuals_dir is already config_visuals_dir from the main loop
                # We need to create a sub-folder for the optimizer within it for these viz
                viz_exp_dir_optimizer_specific = base_visuals_dir / optimizer_name 
                out_subdir_viz = f"val_epoch_{epoch}" if epoch < current_epochs else f"val_final"
                visualize_samples(model, C.DATA_ROOT, "val", viz_exp_dir_optimizer_specific, C.VISUALIZATION_SAMPLES, device, val_tf,
                                  is_overfit_visualization=C.OVERFIT,
                                  overfit_pool_size=C.OVERFIT_N_SAMPLES if C.OVERFIT else 0,
                                  model_type=current_model_type, seq_len=current_seq_len,
                                  out_subdir=out_subdir_viz)
        
    # --- Final Test Evaluation ---
    print(f"\n[INFO] Preparing for final test run for {optimizer_name} ({current_config_name})...")
    test_loader_final = DataLoader(test_ds, shuffle=False, **loader_kwargs) 
    
    test_eval_start_time = time.time()
    test_metrics_final = run_epoch(model, test_loader_final, None, scaler, device, "Test",
                                   criterion_l1, bbox_iou, None, C.OVERFIT, current_model_type)
    test_eval_time = time.time() - test_eval_start_time
    print(f"  Test  → loss {test_metrics_final['loss']:.4f}, IoU {test_metrics_final['iou']:.4f}, Eval time: {test_eval_time:.2f}s")

    if C.VISUALIZATION_SAMPLES > 0:
        viz_split_final = "train" if C.OVERFIT else "test"
        viz_exp_dir_test_optimizer_specific = base_visuals_dir / optimizer_name
        out_subdir_test_viz = f"test_final"
        visualize_samples(model, C.DATA_ROOT, viz_split_final, viz_exp_dir_test_optimizer_specific, C.VISUALIZATION_SAMPLES, device, val_tf,
                          is_overfit_visualization=C.OVERFIT,
                          overfit_pool_size=C.OVERFIT_N_SAMPLES if C.OVERFIT else 0,
                          model_type=current_model_type, seq_len=current_seq_len,
                          out_subdir=out_subdir_test_viz)
    
    val_loss_h = history_val_loss_steps
    val_iou_h = history_val_iou_steps 
    val_f1_h = [0.0] * len(history_val_iou_steps) 
    val_auc_h = [0.0] * len(history_val_iou_steps)
    iter_costs_h = all_iter_costs_for_run
    cumulative_batch_times_h = list(np.cumsum(all_batch_times_for_run))
    grad_norms_h = {} 
    layer_names_h = layer_names
    test_metrics_for_runner = {
        'loss': test_metrics_final['loss'], 'accuracy': test_metrics_final['iou'], 
        'f1_score': 0.0, 'auc': 0.0, 'eval_time_seconds': test_eval_time
    }
    steps_per_epoch_h = len(train_loader)
    train_metrics_hist_adapted = {
        'epoch': list(range(1, current_epochs + 1)), 'train_loss': history_train_loss,
        'train_iou': history_train_iou, 'train_accuracy': history_train_iou, 
        'train_f1_score': [0.0] * len(history_train_iou), 'train_auc': [0.0] * len(history_train_iou)
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

    overall_results_root_dir = Path(C.RESULTS_BASE_DIR) / C.EXPERIMENT_NAME_BASE
    overall_results_root_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Base results directory: {overall_results_root_dir}")

    for config_idx, config in enumerate(C.EXPERIMENT_CONFIGURATIONS):
        config_name = config.get("name", f"config_{config_idx}")
        print(f"\n\n{'='*80}")
        print(f"[INFO] Running Experiment Configuration: {config_name} ({config_idx+1}/{len(C.EXPERIMENT_CONFIGURATIONS)})")
        print(f"[INFO] Configuration details: {config}")
        print(f"{'='*80}")

        # --- Determine configuration-specific parameters ---
        current_model_type = config.get("MODEL_TYPE")
        current_seq_len = config.get("SEQ_LEN")
        current_batch_size = config.get("BATCH_SIZE", C.BATCH_SIZE) # Using C.BATCH_SIZE as fallback
        current_epochs = config.get("EPOCHS", C.EPOCHS)             # Using C.EPOCHS as fallback
        current_lr = config.get("LEARNING_RATE", C.LEARNING_RATE) # Using C.LEARNING_RATE as fallback

        if current_model_type is None or current_seq_len is None:
            raise ValueError(f"Configuration '{config_name}' is missing 'MODEL_TYPE' or 'SEQ_LEN'. These must be defined.")

        config_results_dir = overall_results_root_dir / config_name
        config_visuals_dir = config_results_dir / "visuals"
        config_results_dir.mkdir(parents=True, exist_ok=True)

        experiment_settings_log = {
            "experiment_group_name": C.EXPERIMENT_NAME_BASE,
            "config_name": config_name,
            "model_type": current_model_type,
            "sequence_length": current_seq_len if current_model_type == "temporal" else "N/A",
            "data_root": C.DATA_ROOT,
            "batch_size": current_batch_size,
            "epochs": current_epochs,
            "learning_rate": current_lr,
            "device": str(device),
            "overfit_mode": C.OVERFIT,
            "overfit_samples": C.OVERFIT_N_SAMPLES if C.OVERFIT else "N/A",
            "train_subset_fraction": C.TRAIN_SUBSET_FRACTION if not C.OVERFIT else "N/A (Overfitting)",
            "optimizers_tested": C.OPTIMIZERS,
            "runs_per_optimizer": C.RUNS_PER_OPTIMIZER,
            "date_run": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_details": config
        }
        settings_file = config_results_dir / f"{config_name}_experiment_settings.json"
        with open(settings_file, 'w') as f:
            json.dump(experiment_settings_log, f, indent=4)
        print(f"[INFO] Experiment settings for '{config_name}' saved to {settings_file}")

        print(f"\n[INFO] Loading datasets for configuration: {config_name} (Model: {current_model_type}, SeqLen: {current_seq_len})...")
        dataset_kwargs_main = {
            "root": C.DATA_ROOT,
            "seq_len": current_seq_len,
            "is_temporal": current_model_type == "temporal"
        }

        if C.OVERFIT:
            train_ds_full_main = FrameBBOXDataset(split="train", transform=train_tf, load_n_samples=C.OVERFIT_N_SAMPLES, **dataset_kwargs_main)
            num_actual_overfit_samples_main = len(train_ds_full_main)
            overfit_indices_main = list(range(num_actual_overfit_samples_main))
            train_ds_global = Subset(train_ds_full_main, overfit_indices_main)
            val_test_ds_for_overfitting_main = FrameBBOXDataset(split="train", transform=val_tf, load_n_samples=C.OVERFIT_N_SAMPLES, **dataset_kwargs_main)
            val_ds_global = Subset(val_test_ds_for_overfitting_main, overfit_indices_main)
            test_ds_global = Subset(val_test_ds_for_overfitting_main, overfit_indices_main)
            print(f"[INFO] Overfitting mode: Loaded {len(train_ds_global)} samples for train/val/test.")
        else:
            train_ds_global = FrameBBOXDataset(split="train", transform=train_tf, train_subset_fraction=C.TRAIN_SUBSET_FRACTION, **dataset_kwargs_main)
            val_ds_global   = FrameBBOXDataset(split="val",   transform=val_tf, train_subset_fraction=1.0, **dataset_kwargs_main)
            test_ds_global  = FrameBBOXDataset(split="test",  transform=val_tf, train_subset_fraction=1.0, **dataset_kwargs_main)
            print(f"[INFO] Loaded datasets: Train ({len(train_ds_global)}), Val ({len(val_ds_global)}), Test ({len(test_ds_global)})." )
        print(f"[INFO] Datasets loaded for configuration '{config_name}'.")

        resource_tracker = None # Placeholder for resource tracking logic

        print(f"\n[INFO] Starting experiment runs for configuration '{config_name}' via experiment_runner for optimizers: {C.OPTIMIZERS}")
        
        loss_plot_title = f"{C.LOSS_PLOT_TITLE} ({config_name})"
        iou_plot_title = f"{C.IOU_PLOT_TITLE} ({config_name})" # This will be passed as acc_title
        loss_plot_ylabel = C.LOSS_PLOT_YLABEL
        iou_plot_ylabel = C.IOU_PLOT_YLABEL # This will be passed as acc_ylabel
        
        # Use functools.partial since \'import functools\' is used at the top
        partial_train_func = functools.partial(train_uav_experiment_iteration, 
                                     base_visuals_dir=config_visuals_dir, # This is config_results_dir/visuals/
                                     train_ds=train_ds_global,
                                     val_ds=val_ds_global,
                                     test_ds=test_ds_global,
                                     current_config_name=config_name,
                                     current_model_type=current_model_type,
                                     current_seq_len=current_seq_len,
                                     current_batch_size=current_batch_size,
                                     current_epochs=current_epochs,
                                     current_lr=current_lr)

        run_experiments(
            partial_train_func, # 1. train_experiment (positional)
            config_results_dir, # 2. results_dir (positional)
            config_visuals_dir, # 3. visuals_dir (positional)
            current_epochs,     # 4. epochs (positional)
            optimizer_names=C.OPTIMIZERS,
            num_runs=C.RUNS_PER_OPTIMIZER,
            experiment_title=config_name,
            plot_filename=f"{config_name}_metrics",
            csv_filename=f"{config_name}_metrics.csv",
            loss_title=loss_plot_title,
            acc_title=iou_plot_title, # Renamed from accuracy_plot_title
            loss_ylabel=loss_plot_ylabel,
            acc_ylabel=iou_plot_ylabel, # Renamed from accuracy_plot_ylabel
            # Removed: scheduler_params_per_optimizer, resource_plot_filename, resource_tracker
        )
        print(f"[INFO] Completed all runs for configuration: {config_name}")

    print(f"\n\n{'='*80}")
    print(f"[INFO] All experiment configurations completed.")
    print(f"[INFO] Base results saved in: {overall_results_root_dir}")
    print(f"{'='*80}")
