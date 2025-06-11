#!/usr/bin/env python3
import sys # Add sys for path manipulation
import os # Add os for path manipulation
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
# Removed argparse as it's no longer used.
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset # Ensure Subset is imported
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
import config_uav as C # Import the new config file

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
class FrameBBOXDataset(Dataset):
    """Frame‐level UAV bbox regression with Albumentations."""
    def __init__(self, root, split, transform=None, load_n_samples=None, seq_len=1, is_temporal=False, train_subset_fraction=1.0): # Added train_subset_fraction
        self.img_dir = Path(root) / "images" / split
        self.lbl_dir = Path(root) / "labels" / split
        self.transform = transform
        self.samples = []
        self.seq_len = seq_len
        self.is_temporal = is_temporal
        self.train_subset_fraction = train_subset_fraction # Store it

        print(f"[INFO] Dataset '{split}': Globbing and pre-filtering label files from {self.lbl_dir}...")
        # Pre-filter label files: get stems of all non-empty .txt files in label directory
        valid_label_files = {p.stem: p for p in self.lbl_dir.glob("*.txt") if p.stat().st_size > 0}
        if not valid_label_files:
            print(f"[WARN] Dataset '{split}': No non-empty label files found in {self.lbl_dir}. No samples will be loaded.")
            # self.samples will remain empty, subsequent checks will handle this.
        else:
            print(f"[INFO] Dataset '{split}': Found {len(valid_label_files)} non-empty label files to consider.")

        print(f"[INFO] Dataset '{split}': Globbing images from {self.img_dir}...")
        all_potential_image_paths = sorted(self.img_dir.glob("*.jpg"))

        if split == "train" and self.train_subset_fraction < 1.0 and self.train_subset_fraction > 0:
            original_len = len(all_potential_image_paths)
            subset_len = int(original_len * self.train_subset_fraction)
            if subset_len == 0 and original_len > 0: # Ensure at least 1 sample if original had some and fraction is tiny
                subset_len = 1
            if subset_len > 0 : # Only slice if subset_len is meaningful
                all_potential_image_paths = all_potential_image_paths[:subset_len]
                print(f"[INFO] Dataset '{split}': Applied train_subset_fraction {self.train_subset_fraction}. Considering first {len(all_potential_image_paths)} of {original_len} potential image paths.")
            else:
                print(f"[WARN] Dataset '{split}': train_subset_fraction {self.train_subset_fraction} resulted in 0 samples. Using full dataset for train split (if any images exist).")
        elif split == "train" and (self.train_subset_fraction <= 0 or self.train_subset_fraction > 1.0) and self.train_subset_fraction != 1.0:
            print(f"[WARN] Dataset '{split}': train_subset_fraction ({self.train_subset_fraction}) is out of (0, 1.0] range. Using full dataset for train split (if any images exist).")
        
        paths_to_check = all_potential_image_paths
        if load_n_samples is not None:
            paths_to_check = all_potential_image_paths[:load_n_samples]
            print(f"[INFO] Dataset '{split}': Will iterate up to the first {len(paths_to_check)} sorted image files (due to load_n_samples={load_n_samples}) and check against pre-filtered labels.")
        
        if not valid_label_files: # If no labels were found initially, skip image iteration
            print(f"[INFO] Dataset '{split}': Skipping image iteration as no valid label files were pre-filtered.")
            # The check for self.samples emptiness later will raise the RuntimeError if needed.
        else:
            print(f"[INFO] Dataset '{split}': Processing {len(paths_to_check)} potential image files to find valid samples with corresponding labels...")
            
            for i, current_img_p in enumerate(paths_to_check):
                if self.is_temporal:
                    current_idx_in_all_paths = i # This index is within paths_to_check

                    if current_idx_in_all_paths < self.seq_len - 1:
                        continue 

                    # Find the actual index of currentImg_p in the (potentially subsetted by train_subset_fraction) all_potential_image_paths
                    # This is to correctly slice the sequence.
                    try:
                        actual_current_idx = all_potential_image_paths.index(current_img_p) # Corrected typo: currentImg_p -> current_img_p
                    except ValueError:
                        # Should not happen if paths_to_check is a sublist of all_potential_image_paths
                        print(f"[WARN] Dataset '{split}': Temporal sequence error, current image not in main list. Skipping.")
                        continue

                    if actual_current_idx < self.seq_len - 1:
                        continue

                    start_idx = actual_current_idx - self.seq_len + 1
                    frame_sequence_paths = all_potential_image_paths[start_idx : actual_current_idx + 1]

                    if not frame_sequence_paths:
                        continue

                    expected_prefix = "_".join(frame_sequence_paths[-1].stem.split('_')[:-1])
                    consistent_sequence = all(p.stem.startswith(expected_prefix) for p in frame_sequence_paths)
                    
                    if not consistent_sequence:
                        continue

                    # Check label for the current frame (the last one in the sequence)
                    last_frame_stem = frame_sequence_paths[-1].stem
                    if last_frame_stem in valid_label_files:
                        lbl_p_current = valid_label_files[last_frame_stem]
                        try:
                            contents = lbl_p_current.read_text().split()
                            if len(contents) == 5:
                                bbox_current = torch.tensor(list(map(float, contents[1:])), dtype=torch.float32)
                                self.samples.append((frame_sequence_paths, bbox_current))
                        except Exception as e:
                            print(f"[WARN] Dataset '{split}': Error reading label {lbl_p_current.name} for temporal sample: {e}")
                    # else: no valid label for this temporal sequence's target frame
                else: # Original non-temporal logic
                    img_stem = current_img_p.stem
                    if img_stem in valid_label_files:
                        lbl_p = valid_label_files[img_stem]
                        try:
                            contents = lbl_p.read_text().split()
                            if len(contents) == 5:
                                _, cx, cy, w, h = map(float, contents)
                                bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)
                                self.samples.append((current_img_p, bbox))
                            # else: print(f"[WARN] Dataset '{split}': Skipping {lbl_p} due to unexpected content length: {len(contents)}")
                        except ValueError:
                            # print(f"[WARN] Dataset '{split}': Skipping {lbl_p} due to ValueError during parsing.")
                            pass # Continue to next file
                    # else: no valid label for this image
                
                if (i + 1) % 10000 == 0 and len(paths_to_check) > 20000 :
                    print(f"[INFO] Dataset '{split}': Iterated {i+1}/{len(paths_to_check)} potential image files. Found {len(self.samples)} valid samples so far.")

        if not self.samples:
            # Enhanced error message
            reason = "no non-empty label files were found" if not valid_label_files else f"no images with corresponding valid labels were found among the {len(paths_to_check)} images checked"
            if load_n_samples is not None and len(paths_to_check) < load_n_samples:
                reason += f" (checked up to {len(paths_to_check)} due to available images/subset fraction)"
            
            base_error_msg = f"No valid annotated frames found for split '{split}' because {reason}."
            if not valid_label_files:
                 base_error_msg += f" Searched labels in {self.lbl_dir}."
            else:
                 base_error_msg += f" Searched images in {self.img_dir} and labels in {self.lbl_dir}."

            raise RuntimeError(base_error_msg)
        
        print(f"[INFO] Dataset '{split}': Successfully loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_temporal:
            img_paths_sequence, bbox_current = self.samples[idx]
            frames = []
            for img_p in img_paths_sequence:
                img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented = self.transform(image=img)
                frames.append(augmented["image"])
            
            # Stack frames to form the cuboid [T, C, H, W]
            frame_cuboid = torch.stack(frames) 
            return frame_cuboid, bbox_current # Return cuboid and bbox for the *current* frame
        else:
            img_p, bbox = self.samples[idx]
            # load BGR→RGB
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # apply Albumentations
            augmented = self.transform(image=img)
            img_t = augmented["image"]               # Tensor [C,H,W]
            return img_t, bbox


# ─────────────────────────────── Metrics ─────────────────────────────────────
def bbox_iou(pred_cxcywh, target_cxcywh, eps=1e-6):
    """IoU between normalized [cx,cy,w,h]."""
    def to_xyxy(bbox_cxcywh_tensor):
        cx, cy, w, h = bbox_cxcywh_tensor.unbind(dim=-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)

    pred_xyxy = to_xyxy(pred_cxcywh)
    target_xyxy = to_xyxy(target_cxcywh)

    # Calculate intersection coordinates
    ix1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
    iy1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
    ix2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
    iy2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])

    # Calculate intersection area
    inter_w = (ix2 - ix1).clamp(min=0)
    inter_h = (iy2 - iy1).clamp(min=0)
    intersection = inter_w * inter_h

    # Calculate union area
    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0) * \
                (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=0)
    target_area = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp(min=0) * \
                  (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp(min=0)
    
    union = pred_area + target_area - intersection + eps
    
    return intersection / union


# ─────────────────────────────── Training ────────────────────────────────────
def run_epoch(model, loader, optimizer, scaler, device, stage,
              criterion_l1, criterion_iou, scheduler=None, overfit=False, model_type="single_frame"): 
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
def train_uav_experiment_iteration(optimizer_name, base_visuals_dir, train_ds, val_ds, test_ds, return_settings=False): # Added train_ds, val_ds, test_ds
    """
    Runs a single iteration of the UAV detection training experiment for a given optimizer.
    This function is designed to be compatible with the structure expected by experiment_runner.
    It will handle model initialization, data loading (from pre-loaded datasets), training loop, and metric collection.
    """
    print(f"\\n--- Starting a new run for Optimizer: {optimizer_name} ---")
    
    is_temporal_model = C.MODEL_TYPE == "temporal"

    # DataLoaders
    loader_kwargs = dict(
        batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY,
        persistent_workers=C.PERSISTENT_WORKERS, prefetch_factor=C.PREFETCH_FACTOR
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # Model
    if C.MODEL_TYPE == "temporal":
        model = TemporalResNet50Regressor(seq_len=C.SEQ_LEN, overfit=C.OVERFIT).to(device)
    else:
        model = ResNet50Regressor(overfit=C.OVERFIT).to(device)
    
    layer_names = get_layer_names(model) # For potential gradient norm tracking

    # Losses, Optimizer, Scheduler, Scaler
    criterion_l1 = nn.SmoothL1Loss() # Primary loss for regression

    # Optimizer setup (AdamW is the default in original script)
    # This part might need to be more flexible if supporting multiple optimizers from C.OPTIMIZERS
    if optimizer_name.upper() == "ADAMW":
        optimizer = optim.AdamW(model.parameters(), lr=C.LEARNING_RATE, weight_decay=0.0 if C.OVERFIT else 1e-4)
    else:
        print(f"[WARN] Optimizer {optimizer_name} not explicitly configured, using AdamW with default UAV settings.")
        optimizer = optim.AdamW(model.parameters(), lr=C.LEARNING_RATE, weight_decay=0.0 if C.OVERFIT else 1e-4)

    scheduler = None
    if not C.OVERFIT and C.SCHEDULER_PARAMS.get(optimizer_name, {}).get("scheduler") == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.EPOCHS)
    
    scaler = GradScaler(device=device.type)

    # --- Training Loop (adapted from original main, now per run) ---
    history_train_loss = []
    history_train_iou = []
    history_val_loss_steps = [] 
    history_val_iou_steps = []
    val_epochs_recorded = []

    all_iter_costs_for_run = [] 
    all_batch_times_for_run = []

    best_val_iou = 0.0

    for epoch in range(1, C.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{C.EPOCHS} for {optimizer_name}")
        
        train_epoch_results = run_epoch(model, train_loader, optimizer, scaler, device, "Train",
                                        criterion_l1, bbox_iou, scheduler, C.OVERFIT, C.MODEL_TYPE)
        history_train_loss.append(train_epoch_results["loss"])
        history_train_iou.append(train_epoch_results["iou"])
        all_iter_costs_for_run.extend(train_epoch_results["iter_costs"])
        all_batch_times_for_run.extend(train_epoch_results["batch_times"]) # Collect per-batch times

        if epoch % C.VALIDATE_EVERY == 0 or epoch == C.EPOCHS:
            val_metrics_epoch = run_epoch(model, val_loader, None, scaler, device, "Val",
                                          criterion_l1, bbox_iou, None, C.OVERFIT, C.MODEL_TYPE)
            history_val_loss_steps.append(val_metrics_epoch["loss"]) # Append for each validation step
            history_val_iou_steps.append(val_metrics_epoch["iou"])   # Append for each validation step
            val_epochs_recorded.append(epoch) # Record the epoch number for this validation
            print(f"  Val   → loss {val_metrics_epoch['loss']:.4f}, IoU {val_metrics_epoch['iou']:.4f}")

            if val_metrics_epoch["iou"] > best_val_iou:
                best_val_iou = val_metrics_epoch["iou"]
                print(f"  ↑ New best validation IoU: {best_val_iou:.4f}")
            
            # UAV-specific visualization (can be adapted or made conditional)
            if C.VISUALIZATION_SAMPLES > 0:
                viz_exp_dir = base_visuals_dir 
                out_subdir_viz = f"{optimizer_name}/val_epoch_{epoch}" if epoch < C.EPOCHS else f"{optimizer_name}/val_final"
                visualize_samples(model, C.DATA_ROOT, "val", viz_exp_dir, C.VISUALIZATION_SAMPLES, device, val_tf,
                                  is_overfit_visualization=C.OVERFIT,
                                  overfit_pool_size=C.OVERFIT_N_SAMPLES if C.OVERFIT else 0,
                                  model_type=C.MODEL_TYPE, seq_len=C.SEQ_LEN,
                                  out_subdir=out_subdir_viz)
        
    # --- Final Test Evaluation (after all epochs for this run) ---
    print("\\n[INFO] Preparing for final test run for this iteration...")
    test_loader_final = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    
    test_eval_start_time = time.time()
    test_metrics_final = run_epoch(model, test_loader_final, None, scaler, device, "Test",
                                   criterion_l1, bbox_iou, None, C.OVERFIT, C.MODEL_TYPE)
    test_eval_time = time.time() - test_eval_start_time
    print(f"  Test  → loss {test_metrics_final['loss']:.4f}, IoU {test_metrics_final['iou']:.4f}, Eval time: {test_eval_time:.2f}s")

    if C.VISUALIZATION_SAMPLES > 0:
        viz_split_final = "train" if C.OVERFIT else "test"
        # Use base_visuals_dir here as well
        viz_exp_dir_test = base_visuals_dir
        out_subdir_test_viz = f"{optimizer_name}/test_final"
        visualize_samples(model, C.DATA_ROOT, viz_split_final, viz_exp_dir_test, C.VISUALIZATION_SAMPLES, device, val_tf,
                          is_overfit_visualization=C.OVERFIT,
                          overfit_pool_size=C.OVERFIT_N_SAMPLES if C.OVERFIT else 0,
                          model_type=C.MODEL_TYPE, seq_len=C.SEQ_LEN,
                          out_subdir=out_subdir_test_viz)
    
    val_loss_h = history_val_loss_steps
    val_iou_h = history_val_iou_steps # Using IoU for accuracy metrics
    val_f1_h = [0.0] * len(history_val_iou_steps) # Ensure this uses history_val_iou_steps
    val_auc_h = [0.0] * len(history_val_iou_steps) # Ensure this uses history_val_iou_steps

    iter_costs_h = all_iter_costs_for_run

    cumulative_batch_times_h = list(np.cumsum(all_batch_times_for_run)) # Cumulative time per batch
    grad_norms_h = {} 
    layer_names_h = layer_names
    
    test_metrics_for_runner = {
        'loss': test_metrics_final['loss'],
        'accuracy': test_metrics_final['iou'], # Using IoU for accuracy
        'f1_score': 0.0, # Placeholder
        'auc': 0.0,      # Placeholder
        'eval_time_seconds': test_eval_time
    }
    
    steps_per_epoch_h = len(train_loader)
    
    train_metrics_hist_adapted = {
        'epoch': list(range(1, C.EPOCHS + 1)),
        'train_loss': history_train_loss,
        'train_iou': history_train_iou,
        'train_accuracy': history_train_iou, # Using IoU for accuracy
        'train_f1_score': [0.0] * len(history_train_iou),
        'train_auc': [0.0] * len(history_train_iou)
    }

    results_tuple = (
        val_loss_h, val_iou_h, val_f1_h, val_auc_h,
        iter_costs_h, cumulative_batch_times_h, grad_norms_h, layer_names_h, # Use cumulative_batch_times_h
        test_metrics_for_runner, steps_per_epoch_h, train_metrics_hist_adapted
    )

    if return_settings:
        optimizer_specific_params = C.SCHEDULER_PARAMS.get(optimizer_name, {})
        settings_for_runner = {
            "optimizer_params": optimizer_specific_params,
        }
        return results_tuple + (settings_for_runner,)
    else:
        return results_tuple

if __name__=="__main__":
    print(f"[INFO] Using configuration from config_uav.py: {C.EXPERIMENT_NAME or 'DefaultExp'}")

    # Import functools at the top of the file.
    from functools import partial # Ensure this is here or at the top

    # --- Setup results directories (similar to supervised_learning_experiment.py) ---
    base_exp_name = C.EXPERIMENT_NAME or f"uav_exp_{time.strftime('%Y%m%d_%H%M%S')}"
    results_dir_main = Path(C.RESULTS_BASE_DIR) / base_exp_name
    visuals_dir_main = results_dir_main / "visuals"
    
    results_dir_main.mkdir(parents=True, exist_ok=True)
    visuals_dir_main.mkdir(parents=True, exist_ok=True)

    # --- Experiment Settings for logging (from supervised_learning_experiment.py) ---
    experiment_settings_log = {
        "experiment_name": base_exp_name,
        "model_type": C.MODEL_TYPE,
        "sequence_length": C.SEQ_LEN if C.MODEL_TYPE == "temporal" else "N/A",
        "data_root": C.DATA_ROOT,
        "batch_size": C.BATCH_SIZE,
        "epochs": C.EPOCHS,
        "learning_rate": C.LEARNING_RATE,
        "device": str(device),
        "overfit_mode": C.OVERFIT,
        "overfit_samples": C.OVERFIT_N_SAMPLES if C.OVERFIT else "N/A",
        "train_subset_fraction": C.TRAIN_SUBSET_FRACTION if not C.OVERFIT else "N/A (Overfitting)",
        "optimizers_tested": C.OPTIMIZERS, # From config_uav.py
        "runs_per_optimizer": C.RUNS_PER_OPTIMIZER, # From config_uav.py
        "date_run": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    # Save experiment settings
    settings_file = results_dir_main / f"{base_exp_name}_experiment_settings.json"
    with open(settings_file, 'w') as f:
        json.dump(experiment_settings_log, f, indent=4)
    print(f"[INFO] Experiment settings saved to {settings_file}")

    # --- Pre-load Datasets ---
    print("\\n[INFO] Loading datasets...")
    is_temporal_model_main = C.MODEL_TYPE == "temporal" # Use a distinct variable name if needed, or rely on C
    dataset_kwargs_main = {
        "root": C.DATA_ROOT,
        "seq_len": C.SEQ_LEN if is_temporal_model_main else 1,
        "is_temporal": is_temporal_model_main
    }

    if C.OVERFIT:
        # Ensure train_tf and val_tf are accessible here (they are global)
        train_ds_full_main = FrameBBOXDataset(split="train", transform=train_tf, load_n_samples=C.OVERFIT_N_SAMPLES, **dataset_kwargs_main)
        num_actual_overfit_samples_main = len(train_ds_full_main)
        overfit_indices_main = list(range(num_actual_overfit_samples_main))
        
        train_ds_global = Subset(train_ds_full_main, overfit_indices_main)
        
        # For validation and test in overfit mode, use the same subset of training data
        val_test_ds_for_overfitting_main = FrameBBOXDataset(split="train", transform=val_tf, load_n_samples=C.OVERFIT_N_SAMPLES, **dataset_kwargs_main)
        val_ds_global = Subset(val_test_ds_for_overfitting_main, overfit_indices_main)
        test_ds_global = Subset(val_test_ds_for_overfitting_main, overfit_indices_main)
        print(f"[INFO] Overfitting mode: Loaded {len(train_ds_global)} samples for train/val/test from the first {num_actual_overfit_samples_main} train samples.")
    else:
        train_ds_global = FrameBBOXDataset(split="train", transform=train_tf, train_subset_fraction=C.TRAIN_SUBSET_FRACTION, **dataset_kwargs_main)
        val_ds_global   = FrameBBOXDataset(split="val",   transform=val_tf, train_subset_fraction=1.0, **dataset_kwargs_main)
        test_ds_global  = FrameBBOXDataset(split="test",  transform=val_tf, train_subset_fraction=1.0, **dataset_kwargs_main)
        print(f"[INFO] Loaded datasets: Train ({len(train_ds_global)} samples), Val ({len(val_ds_global)} samples), Test ({len(test_ds_global)} samples).")
    print("[INFO] Datasets loaded globally.")

    # --- Call the experiment_runner ---
    resource_tracker = None
    if 'ResourceTracker' in globals() and getattr(C, 'TRACK_RESOURCES', False):
        resource_tracker = ResourceTracker()
        print("[INFO] Resource tracking enabled.")
    else:
        print("[INFO] Resource tracking disabled or not configured.")

    print(f"\\n[INFO] Starting experiments via experiment_runner for: {C.OPTIMIZERS}")
    
    # Titles for plots (can be customized in config_uav.py)
    loss_plot_title = getattr(C, "LOSS_PLOT_TITLE", "Loss vs. Epoch (UAV)")
    iou_plot_title = getattr(C, "IOU_PLOT_TITLE", "IoU vs. Epoch (UAV)") # Using IoU for "accuracy" plot
    loss_plot_ylabel = getattr(C, "LOSS_PLOT_YLABEL", "Average Loss")
    iou_plot_ylabel = getattr(C, "IOU_PLOT_YLABEL", "Average IoU")

    
    partial_train_func = partial(train_uav_experiment_iteration, 
                                 base_visuals_dir=visuals_dir_main,
                                 train_ds=train_ds_global,
                                 val_ds=val_ds_global,
                                 test_ds=test_ds_global)

    all_optimizer_results = run_experiments(
        partial_train_func, # Pass the partially applied function
        results_dir=results_dir_main,
        visuals_dir=visuals_dir_main,
        epochs=C.EPOCHS, # experiment_runner uses this for plotting, actual epochs in train_uav_experiment_iteration
        optimizer_names=C.OPTIMIZERS,
        loss_title=loss_plot_title,
        loss_ylabel=loss_plot_ylabel,
        acc_title=iou_plot_title, # Plotting IoU as "accuracy"
        acc_ylabel=iou_plot_ylabel,
        plot_filename=f"{base_exp_name}_uav", # Corrected: plot_filename_base -> plot_filename
        csv_filename=f"{base_exp_name}_uav_metrics.csv", # Corrected: csv_filename_base -> csv_filename, and added .csv extension
        experiment_title=f"UAV Detection: {base_exp_name}",
        num_runs=C.RUNS_PER_OPTIMIZER,
        experiment_settings=experiment_settings_log,
        f1_title="F1 Score (Placeholder)", # Placeholder title
    )
    
    if resource_tracker:
        resource_info_df = resource_tracker.get_summary_df()
        if not resource_info_df.empty:
            save_resource_info(resource_info_df, results_dir_main, f"{base_exp_name}_resource_usage.csv")
            if 'plot_resource_usage' in globals():
                 plot_resource_usage(resource_info_df, visuals_dir_main, base_exp_name, "UAV_Detection")
        print("[INFO] Resource usage summary saved.")

    print(f"\\n[INFO] All experiments complete. Results in: {results_dir_main.resolve()}")
    print(f"[INFO] Visualizations in: {visuals_dir_main.resolve()}")

    print(f"[DONE] Main script finished. Check results in {results_dir_main.resolve()}")
