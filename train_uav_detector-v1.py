'''
python train_uav_detector.py 
  --data-root C:/data/processed_anti_uav 
  --batch-size 32 
  --epochs 10 
  --lr 1e-3
  --persistent_workers true
  
python train_uav_detector.py
  --data-root C:/data/processed_anti_uav 
  --batch-size 16 --epochs 50 --lr 1e-3 
  --overfit --overfit-n 32

python train_uav_detector.py 
  --data-root C:/data/processed_anti_uav 
  --model-type temporal 
  --seq-len 5 
  --batch-size 16 
  --epochs 50 
  --lr 1e-4 
  --name temporal_experiment_01
'''

#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Import models from network.py
from network import ResNet50Regressor, TemporalResNet50Regressor

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

        print(f"[INFO] Dataset '{split}': Globbing images from {self.img_dir}...")
        all_potential_image_paths = sorted(self.img_dir.glob("*.jpg"))

        if split == "train" and self.train_subset_fraction < 1.0 and self.train_subset_fraction > 0:
            original_len = len(all_potential_image_paths)
            subset_len = int(original_len * self.train_subset_fraction)
            if subset_len == 0 and original_len > 0: # Ensure at least 1 sample if original had some and fraction is tiny
                subset_len = 1
            if subset_len > 0 : # Only slice if subset_len is meaningful
                all_potential_image_paths = all_potential_image_paths[:subset_len]
                print(f"[INFO] Dataset '{split}': Applied train_subset_fraction {self.train_subset_fraction}. Using first {len(all_potential_image_paths)} of {original_len} potential paths.")
            else:
                print(f"[WARN] Dataset '{split}': train_subset_fraction {self.train_subset_fraction} resulted in 0 samples. Using full dataset for train split.")
        elif split == "train" and (self.train_subset_fraction <= 0 or self.train_subset_fraction > 1.0) and self.train_subset_fraction != 1.0:
            print(f"[WARN] Dataset '{split}': train_subset_fraction ({self.train_subset_fraction}) is out of (0, 1.0] range. Using full dataset for train split.")
        
        paths_to_check = all_potential_image_paths
        if load_n_samples is not None:
            # We will check up to load_n_samples from the sorted list
            # to find as many valid pairs as possible up to that limit.
            # The actual number of samples loaded might be less if not all have valid labels.
            paths_to_check = all_potential_image_paths[:load_n_samples]
            print(f"[INFO] Dataset '{split}': Will check up to the first {len(paths_to_check)} sorted image files for valid labels (load_n_samples={load_n_samples}).")
        
        print(f"[INFO] Dataset '{split}': Processing {len(paths_to_check)} potential image files to find valid samples...")
        
        for i, current_img_p in enumerate(paths_to_check):
            if self.is_temporal:
                # current_img_p is the *last* frame of a potential sequence.
                # Its index 'i' is within paths_to_check. Since paths_to_check is a prefix
                # of all_potential_image_paths (after subsetting), 'i' is also its index
                # in all_potential_image_paths. This is the optimization.
                current_idx_in_all_paths = i 

                if current_idx_in_all_paths < self.seq_len - 1:
                    # Not enough preceding frames in the (potentially subsetted) list
                    # for current_img_p to be the END of a sequence of seq_len.
                    continue 
                
                start_idx = current_idx_in_all_paths - self.seq_len + 1
                # Slice from all_potential_image_paths which has been correctly subsetted by train_subset_fraction and load_n_samples (via paths_to_check derivation)
                frame_sequence_paths = all_potential_image_paths[start_idx : current_idx_in_all_paths + 1]

                # Validate sequence consistency (e.g., from the same video)
                # The current frame (current_img_p) is frame_sequence_paths[-1]
                if not frame_sequence_paths: # Should not happen if logic is correct
                    continue

                expected_prefix = "_".join(frame_sequence_paths[-1].stem.split('_')[:-1]) # Video prefix from the current frame
                consistent_sequence = all(p.stem.startswith(expected_prefix) for p in frame_sequence_paths)
                
                if not consistent_sequence:
                    continue

                # Check label for the current frame (the last one in the sequence, which is current_img_p)
                lbl_p_current = self.lbl_dir / f"{frame_sequence_paths[-1].stem}.txt"
                if lbl_p_current.exists() and lbl_p_current.stat().st_size > 0:
                    try:
                        contents = lbl_p_current.read_text().split()
                        if len(contents) == 5: # class_id, cx, cy, w, h
                            bbox_current = torch.tensor(list(map(float, contents[1:])), dtype=torch.float32)
                            self.samples.append((frame_sequence_paths, bbox_current))
                    except Exception as e:
                        print(f"[WARN] Dataset '{split}': Error reading label {lbl_p_current.name} for temporal sample: {e}")
            else: # Original non-temporal logic
                lbl_p = self.lbl_dir / f"{current_img_p.stem}.txt"
                if lbl_p.exists() and lbl_p.stat().st_size > 0:
                    try:
                        contents = lbl_p.read_text().split()
                        if len(contents) == 5:
                            _, cx, cy, w, h = map(float, contents)
                            bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)
                            self.samples.append((current_img_p, bbox)) # Store single path and its bbox
                        else:
                            # print(f"[WARN] Dataset '{split}': Skipping {lbl_p} due to unexpected content length: {len(contents)}")
                            pass # Continue to next file
                    except ValueError:
                        # print(f"[WARN] Dataset '{split}': Skipping {lbl_p} due to ValueError during parsing.")
                        pass # Continue to next file
            
            if (i + 1) % 10000 == 0 and len(paths_to_check) > 20000 : # Progress for large datasets
                print(f"[INFO] Dataset '{split}': Checked {i+1}/{len(paths_to_check)} potential files. Found {len(self.samples)} valid samples so far.")

        if not self.samples:
            if load_n_samples is not None:
                 # This means that out of the 'load_n_samples' files checked, none had valid labels.
                 raise RuntimeError(f"No valid annotated frames found within the first {len(paths_to_check)} images checked in {self.img_dir} / {self.lbl_dir} for split '{split}'.")
            else:
                raise RuntimeError(f"No valid annotated frames found in {self.img_dir} / {self.lbl_dir} for split '{split}'.")
        
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
                # Apply transform to each frame individually if it's complex.
                # If transform is simple (like ToTensorV2, Normalize), it might be applicable to the stack.
                # For Albumentations, it's usually per image.
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
def bbox_iou(pred, target, eps=1e-6):
    """IoU between normalized [cx,cy,w,h]."""
    def to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        x1 = cx - w/2; y1 = cy - h/2
        x2 = cx + w/2; y2 = cy + h/2
        return torch.stack([x1,y1,x2,y2], -1)
    p = to_xyxy(pred)
    t = to_xyxy(target)
    ix1 = torch.max(p[...,0], t[...,0])
    iy1 = torch.max(p[...,1], t[...,1])
    ix2 = torch.min(p[...,2], t[...,2])
    iy2 = torch.min(p[...,3], t[...,3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    area_p = ((p[...,2]-p[...,0]).clamp(0) *
              (p[...,3]-p[...,1]).clamp(0))
    area_t = ((t[...,2]-t[...,0]).clamp(0) *
              (t[...,3]-t[...,1]).clamp(0))
    union = area_p + area_t - inter + eps
    return inter / union


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

    pbar = tqdm(loader, desc=stage, leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=scaler.is_enabled()):
            preds = model(inputs) 
            loss_l1 = criterion_l1(preds, targets)
            loss_iou_val = criterion_iou(preds, targets) 
            loss_iou = 1.0 - loss_iou_val.mean() 
            loss = loss_l1 + loss_iou # Corrected loss calculation
        
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
        # If overfit_pool_size is specified and > 0, limit the candidate pool
        # to the first 'overfit_pool_size' images from the sorted list.
        # This assumes these were the images potentially used in overfitting.
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
                print(f"[WARN] Viz: Current image {img_p_path_obj.name} not found in sorted list of {split} dir. Skipping.")
                continue

            if current_idx < seq_len - 1:
                print(f"[WARN] Viz: Not enough preceding frames for {img_p_path_obj.name} (idx {current_idx}) to form sequence of length {seq_len}. Skipping.")
                continue
            
            img_paths_sequence_objects = all_potential_image_paths_for_seq[current_idx - seq_len + 1 : current_idx + 1]
            
            expected_prefix = "_".join(img_paths_sequence_objects[-1].stem.split('_')[:-1])
            consistent = all(p.stem.startswith(expected_prefix) for p in img_paths_sequence_objects)
            if not consistent:
                print(f"[WARN] Viz: Skipping sequence for {img_p_path_obj.name} due to inconsistent video source.")
                continue

            frames_for_model = []
            for p_obj in img_paths_sequence_objects:
                img_rgb_for_model = cv2.cvtColor(cv2.imread(str(p_obj)), cv2.COLOR_BGR2RGB)
                transformed = transform_to_apply(image=img_rgb_for_model)
                frames_for_model.append(transformed["image"])
            
            input_tensor = torch.stack(frames_for_model).unsqueeze(0).to(device) # [1, T, C, H, W]

        else: # single_frame model type
            img_rgb_for_model = cv2.cvtColor(im_bgr.copy(), cv2.COLOR_BGR2RGB) # RGB for model
            transformed = transform_to_apply(image=img_rgb_for_model)
            input_tensor = transformed["image"].unsqueeze(0).to(device) # [1, C, H, W]

        # Common logic for prediction and drawing using im_bgr (loaded current frame)
        lbl_p_ground_truth = lbl_dir / f"{img_p_path_obj.stem}.txt"
        
        with torch.no_grad():
            pred_bbox_norm = model(input_tensor)[0].cpu().numpy() # [4]

        h, w = im_bgr.shape[:2]
        px, py, pw, ph = pred_bbox_norm
        px1 = int((px - pw/2) * w)
        py1 = int((py - ph/2) * h)
        px2 = int((px + pw/2) * w)
        py2 = int((py + ph/2) * h)
        cv2.rectangle(im_bgr, (px1,py1), (px2,py2), (0,255,0), 2) # Green for prediction

        if lbl_p_ground_truth.exists() and lbl_p_ground_truth.stat().st_size > 0:
            try:
                contents = lbl_p_ground_truth.read_text().split()
                if len(contents) == 5:
                    _, cx, cy, wn, hn = map(float, contents)
                    tx1 = int((cx - wn/2) * w)
                    ty1 = int((cy - hn/2) * h)
                    tx2 = int((cx + wn/2) * w)
                    ty2 = int((cy + hn/2) * h)
                    cv2.rectangle(im_bgr, (tx1,ty1), (tx2,ty2), (0,0,255), 1) # Red for ground truth
            except Exception as e:
                print(f"[WARN] Viz: Error parsing label file {lbl_p_ground_truth}: {e}")
        
        out_path_prediction_img = out_dir / img_p_path_obj.name
        cv2.imwrite(str(out_path_prediction_img), im_bgr)


if __name__=="__main__":
    parser = argparse.ArgumentParser("Train UAV bbox with Albumentations")
    parser.add_argument("--data-root", required=True)
    # Add new arguments for model type and sequence length
    parser.add_argument("--model-type", type=str, default="single_frame", 
                        choices=["single_frame", "temporal"], 
                        help="Type of model to train (single_frame or temporal).")
    parser.add_argument("--seq-len", type=int, default=5, 
                        help="Sequence length for temporal model (if model-type is temporal).")
    parser.add_argument("--train-subset-fraction", type=float, default=1.0,
                        help="Fraction of the training dataset to use (e.g., 0.25 for the first 25%%). Applied only if not overfitting and for the 'train' split. Value should be in (0, 1.0].")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--name",       default=None)
    parser.add_argument("--viz-num",    type=int, default=10)
    parser.add_argument("--val-every",  type=int, default=10,
                        help="Run validation every N epochs (e.g., every 10 epochs)")
    parser.add_argument("--overfit",    action="store_true")
    parser.add_argument("--overfit-n",  type=int, default=32)
    args = parser.parse_args()
    print(f"[INFO] Arguments parsed: {args}")

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    is_temporal_model = args.model_type == "temporal"

    # Albumentations pipelines

    train_tf = A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.5,1.0), ratio=(0.75,1.333)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    print("[INFO] Albumentations transformation pipelines defined.")

    # Datasets
    print("[INFO] Initializing datasets...")
    dataset_kwargs = {
        "root": args.data_root,
        # transform will be set based on split (train_tf or val_tf)
        "seq_len": args.seq_len if is_temporal_model else 1,
        "is_temporal": is_temporal_model
    }

    if args.overfit:
        print(f"[INFO] Overfitting enabled. Loading up to {args.overfit_n} samples.")
        # For overfit, load a small number of samples.
        # The 'load_n_samples' in FrameBBOXDataset will be used to limit this.
        # train_subset_fraction is not explicitly used here, as load_n_samples achieves the subsetting.
        # The default train_subset_fraction=1.0 will apply to the already small set defined by load_n_samples.
        
        train_ds_full = FrameBBOXDataset(split="train", transform=train_tf, load_n_samples=args.overfit_n, **dataset_kwargs)
        if not train_ds_full or len(train_ds_full) == 0:
             raise RuntimeError(f"Overfitting setup failed: No valid training samples found when trying to load up to {args.overfit_n} from 'train' split. Check data and --overfit-n.")

        num_actual_overfit_samples = len(train_ds_full)
        print(f"[INFO] Using {num_actual_overfit_samples} actual samples from 'train' split for overfitting train/val/test sets.")
        if num_actual_overfit_samples < args.overfit_n:
            print(f"[WARN] Requested {args.overfit_n} samples for overfitting, but only {num_actual_overfit_samples} valid ones were found by checking initial files in the 'train' split.")

        overfit_indices = list(range(num_actual_overfit_samples))
        train_ds = Subset(train_ds_full, overfit_indices)
        
        # For validation and test during overfit, use the same subset but with val_tf
        # Re-create a dataset instance for val/test with val_tf, then subset it.
        # This ensures correct transforms are applied if they differ.
        val_test_ds_for_overfitting = FrameBBOXDataset(split="train", transform=val_tf, load_n_samples=args.overfit_n, **dataset_kwargs)
        val_ds = Subset(val_test_ds_for_overfitting, overfit_indices) 
        test_ds = Subset(val_test_ds_for_overfitting, overfit_indices)
        
        print(f"[OVERFIT] Final dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        if len(train_ds) == 0:
            raise ValueError("Overfitting setup failed: Training dataset is empty after subsetting. Check --overfit-n and data.")

    else: # Normal training run
        train_ds = FrameBBOXDataset(split="train", transform=train_tf, 
                                    train_subset_fraction=args.train_subset_fraction, 
                                    **dataset_kwargs)
        val_ds   = FrameBBOXDataset(split="val",   transform=val_tf, 
                                    train_subset_fraction=1.0, # Val uses full data
                                    **dataset_kwargs)
        test_ds  = FrameBBOXDataset(split="test",  transform=val_tf, 
                                    train_subset_fraction=1.0, # Test uses full data
                                    **dataset_kwargs)
    
    # DataLoaders
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    print("[INFO] Creating DataLoaders...")
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)
    print("[INFO] DataLoaders created.")

    # Model, losses, optimizer, scheduler, scaler
    print(f"[INFO] Creating model: {args.model_type}")
    if args.model_type == "temporal":
        model = TemporalResNet50Regressor(seq_len=args.seq_len, overfit=args.overfit).to(device)
        print(f"[INFO] TemporalResNet50Regressor model created with seq_len={args.seq_len}.")
    else: # single_frame (default)
        model = ResNet50Regressor(overfit=args.overfit).to(device)
        print("[INFO] ResNet50Regressor model created.")

    criterion_l1 = nn.SmoothL1Loss()
    criterion_iou= bbox_iou
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0 if args.overfit else 1e-4
    )
    scheduler = None if args.overfit else optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = GradScaler(device=device.type) # Updated GradScaler

    # Results directories
    exp = args.name or time.strftime("exp_%Y%m%d_%H%M%S")
    exp_dir = Path("results")/exp
    (exp_dir/"metrics").mkdir(parents=True, exist_ok=True)
    (exp_dir/"predictions").mkdir(exist_ok=True)

    # Training loop
    best_iou = 0.0
    history = {"train":[], "val":[]}

    print("[INFO] Starting training loop...")
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, "Train",
                                  criterion_l1, criterion_iou, scheduler, args.overfit, args.model_type)
        history["train"].append({"loss": train_metrics["loss"], "iou": train_metrics["iou"]})
        print(f" Train → loss {train_metrics['loss']:.4f}, IoU {train_metrics['iou']:.4f}")

        # Run validation and visualization only every args.val_every epochs or on the last epoch
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_metrics = run_epoch(model, val_loader, None, scaler, device, "Val",
                                    criterion_l1, criterion_iou, None, args.overfit, args.model_type)
            history["val"].append({"loss": val_metrics["loss"], "iou": val_metrics["iou"]})
            print(f" Val   → loss {val_metrics['loss']:.4f}, IoU {val_metrics['iou']:.4f}")
            current_val_iou = val_metrics['iou']
            if current_val_iou > best_iou:
                best_iou = current_val_iou
                torch.save(model.state_dict(), exp_dir/"best.pth")
                print(f" ↑ saved new best (Val IoU={best_iou:.4f})")

            # Visualization on validation
            if args.viz_num > 0 and (epoch % 2 == 0 or epoch == args.epochs):
                print(f"  Visualizing {args.viz_num} validation samples...")
                # Determine output subdirectory: 'val_{epoch}' or 'val_final'
                out_subdir = f"val_{epoch}" if epoch < args.epochs else "val_final"
                visualize_samples(model, args.data_root, "val", exp_dir, args.viz_num, device, val_tf,
                                  is_overfit_visualization=args.overfit,
                                  overfit_pool_size=args.overfit_n if args.overfit else 0,
                                  model_type=args.model_type, seq_len=args.seq_len,
                                  out_subdir=out_subdir)
        #else:
        #   print(f"[INFO] Skipping validation at epoch {epoch} (validate every {args.val_every} epochs)")

    # Final test
    print("\\n[INFO] Preparing for final test run...")
    print("[INFO] Deleting train_loader, val_loader, and model to free memory...")
    del train_loader
    del val_loader
    
    if device.type == 'cuda':
        print("[INFO] Clearing CUDA cache...")
        torch.cuda.empty_cache()
    
    print("[INFO] Initializing test_loader for final test...")
    # Re-create test_loader if it was tied to deleted datasets or if its workers need to be fresh
    # Assuming test_ds is still valid:
    test_loader  = DataLoader(test_ds,   shuffle=False, **loader_kwargs)
    print("[INFO] test_loader initialized.")

    test_metrics = run_epoch(model, test_loader, None, scaler, device, "Test", 
                             criterion_l1, criterion_iou, None, args.overfit, args.model_type)
    print(f"\nTest → loss {test_metrics['loss']:.4f}, IoU {test_metrics['iou']:.4f}")
    history["test"] = {"loss": test_metrics["loss"], "iou": test_metrics["iou"]}
    (exp_dir/"metrics"/"metrics.json").write_text(
        json.dumps(history, indent=2)
    )

    # Visualizations
    # Determine the correct transform and settings for visualization
    if args.overfit:
        # When overfitting, visualize samples from the 'train' split (where overfit data came from)
        viz_transform = val_tf # Use val_tf for consistency, as train_tf might have augmentations not desired for pure viz
        viz_split = "train"
        is_overfit_viz = True
        overfit_pool_size_for_viz = args.overfit_n 
    else:
        # Standard visualization on the 'test' split with 'val_tf'
        viz_transform = val_tf
        viz_split = "test"
        is_overfit_viz = False
        overfit_pool_size_for_viz = 0

    if args.viz_num > 0:
        print(f"  Visualizing {args.viz_num} {viz_split} samples after final test...")
        visualize_samples(
            model, args.data_root,
            viz_split,
            exp_dir, args.viz_num, device, viz_transform,
            is_overfit_visualization=is_overfit_viz,
            overfit_pool_size=overfit_pool_size_for_viz,
            model_type=args.model_type, 
            seq_len=args.seq_len, out_subdir="test_final"
        )

    print(f"[DONE] results in {exp_dir.resolve()}")
