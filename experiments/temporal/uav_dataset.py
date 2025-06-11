"""
Defines the FrameBBOXDataset class for loading UAV image and bounding box data.
This dataset class is designed to work with frame-level or temporal sequence data
for bounding box regression tasks, integrating with Albumentations for transformations.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2 # For cv2.imread and cv2.cvtColor

class FrameBBOXDataset(Dataset):
    """
    A PyTorch Dataset for loading frame-level or temporal sequences of UAV images
    and their corresponding bounding box annotations.

    Handles loading images, parsing YOLO-style label files, and applying
    Albumentations transforms. Can operate in single-frame or temporal mode.
    In temporal mode, it loads a sequence of `seq_len` frames and the label
    for the last frame in the sequence.

    Attributes:
        img_dir (Path): Directory containing image files.
        lbl_dir (Path): Directory containing label files (YOLO format).
        transform (callable, optional): Albumentations transform pipeline.
        samples (list): List of tuples, where each tuple contains the
                        image path(s) and the corresponding bounding box tensor.
                        For temporal data, image paths is a list of Path objects.
        seq_len (int): Length of the image sequence for temporal data.
        is_temporal (bool): Flag indicating if the dataset is for temporal data.
        train_subset_fraction (float): Fraction of training data to use.
    """
    def __init__(self, root, split, transform=None, load_n_samples=None, seq_len=1, is_temporal=False, train_subset_fraction=1.0):
        """
        Initializes the FrameBBOXDataset.

        Args:
            root (str or Path): Root directory of the dataset (e.g., 'Anti-UAV').
            split (str): Data split to load (e.g., 'train', 'val', 'test').
            transform (callable, optional): Albumentations transform to apply.
            load_n_samples (int, optional): If specified, load only the first
                                           `load_n_samples` valid samples.
            seq_len (int): Sequence length for temporal data. Defaults to 1 (single frame).
            is_temporal (bool): True if loading temporal sequences, False otherwise.
            train_subset_fraction (float): Fraction of the training dataset to use (0.0 to 1.0).
                                           Only applies if split is 'train'. Defaults to 1.0.

        Raises:
            RuntimeError: If no valid annotated frames are found for the specified split.
        """
        self.img_dir = Path(root) / "images" / split
        self.lbl_dir = Path(root) / "labels" / split
        self.transform = transform
        self.samples = []
        self.seq_len = seq_len
        self.is_temporal = is_temporal
        self.train_subset_fraction = train_subset_fraction

        print(f"[INFO] Dataset '{split}': Globbing and pre-filtering label files from {self.lbl_dir}...")
        valid_label_files = {p.stem: p for p in self.lbl_dir.glob("*.txt") if p.stat().st_size > 0}
        if not valid_label_files:
            print(f"[WARN] Dataset '{split}': No non-empty label files found in {self.lbl_dir}. No samples will be loaded.")
        else:
            print(f"[INFO] Dataset '{split}': Found {len(valid_label_files)} non-empty label files to consider.")

        print(f"[INFO] Dataset '{split}': Globbing images from {self.img_dir}...")
        all_potential_image_paths = sorted(self.img_dir.glob("*.jpg"))

        if split == "train" and self.train_subset_fraction < 1.0 and self.train_subset_fraction > 0:
            original_len = len(all_potential_image_paths)
            subset_len = int(original_len * self.train_subset_fraction)
            if subset_len == 0 and original_len > 0:
                subset_len = 1
            if subset_len > 0 :
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
        
        if not valid_label_files:
            print(f"[INFO] Dataset '{split}': Skipping image iteration as no valid label files were pre-filtered.")
        else:
            print(f"[INFO] Dataset '{split}': Processing {len(paths_to_check)} potential image files to find valid samples with corresponding labels...")
            
            for i, current_img_p in enumerate(paths_to_check):
                if self.is_temporal:
                    current_idx_in_all_paths = i
                    if current_idx_in_all_paths < self.seq_len - 1:
                        continue 
                    try:
                        actual_current_idx = all_potential_image_paths.index(current_img_p)
                    except ValueError:
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
                else: 
                    img_stem = current_img_p.stem
                    if img_stem in valid_label_files:
                        lbl_p = valid_label_files[img_stem]
                        try:
                            contents = lbl_p.read_text().split()
                            if len(contents) == 5:
                                _, cx, cy, w, h = map(float, contents)
                                bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)
                                self.samples.append((current_img_p, bbox))
                        except ValueError:
                            pass 
                
                if (i + 1) % 10000 == 0 and len(paths_to_check) > 20000 :
                    print(f"[INFO] Dataset '{split}': Iterated {i+1}/{len(paths_to_check)} potential image files. Found {len(self.samples)} valid samples so far.")

        if not self.samples:
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
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the sample (image/sequence and bounding box) at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple:
                - img_tensor (torch.Tensor): The image tensor or frame cuboid tensor.
                  For single-frame: [C, H, W].
                  For temporal: [T, C, H, W].
                - bbox (torch.Tensor): The bounding box tensor [cx, cy, w, h].
        """
        if self.is_temporal:
            img_paths_sequence, bbox_current = self.samples[idx]
            frames = []
            for img_p in img_paths_sequence:
                img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented = self.transform(image=img)
                frames.append(augmented["image"])
            frame_cuboid = torch.stack(frames) 
            return frame_cuboid, bbox_current
        else:
            img_p, bbox = self.samples[idx]
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=img)
            img_t = augmented["image"]
            return img_t, bbox
