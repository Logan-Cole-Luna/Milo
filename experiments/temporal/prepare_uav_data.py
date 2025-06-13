#!/usr/bin/env python3
import json
import subprocess
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import random

def dump_frames_task(args):
    seq_dir, split, output_root, interval, img_size, ffmpeg_bin = args
    out_img_dir = Path(output_root) / "images" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{seq_dir.name}_%05d.jpg"
    vf = f"fps=1/{interval},scale={img_size[0]}:{img_size[1]}"
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(seq_dir / "visible.mp4"),
        "-vf", vf,
        str(out_img_dir / pattern)
    ]
    subprocess.run(cmd, check=True)

def process_sequence_task(args):
    seq_name, split, input_root, output_root, seq_len, img_size = args
    seq_dir     = Path(input_root)  / split / seq_name
    out_img_dir = Path(output_root) / "images" / split
    out_lbl_dir = Path(output_root) / "labels" / split
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # read original resolution
    cap = cv2.VideoCapture(str(seq_dir / "visible.mp4"))
    if not cap.isOpened():
        print(f"[WARN] Cannot open {seq_dir/'visible.mp4'}")
        return []
    orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # load annotations
    with open(seq_dir / "visible.json") as f:
        raw = json.load(f).get("gt_rect", [])

    sx = img_size[0] / orig_w
    sy = img_size[1] / orig_h

    frames = []
    proc = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % 1 == 0:
            im = cv2.resize(frame, tuple(img_size))
            fname = f"{seq_name}_{proc:05d}.jpg"
            cv2.imwrite(str(out_img_dir / fname), im)

            if proc < len(raw) and isinstance(raw[proc], (list,tuple)) and len(raw[proc])==4:
                x,y,w,h = raw[proc]
                x *= sx; w *= sx
                y *= sy; h *= sy
            else:
                x=y=w=h=0

            lbl = out_lbl_dir / f"{seq_name}_{proc:05d}.txt"
            if w>0 and h>0:
                cx = (x + w/2) / img_size[0]
                cy = (y + h/2) / img_size[1]
                wn = w / img_size[0]
                hn = h / img_size[1]
                cx = min(max(cx,0),1)
                cy = min(max(cy,0),1)
                wn = min(max(wn,0),1)
                hn = min(max(hn,0),1)
                lbl.write_text(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
            else:
                lbl.touch()

            frames.append(f"images/{split}/{fname}")
            proc += 1
        idx += 1
    cap.release()

    lines = []
    for i in range(max(0, len(frames)-seq_len+1)):
        lines.append(" ".join(frames[i:i+seq_len]))
    return lines

def main():
    # Define script parameters here
    input_root_val = "C:\Users\Logan\Downloads\Anti-UAV-RGBT"
    output_root_val = "C:\data\processed_uav"
    splits_val = ["train", "val", "test"]
    seq_len_val = 5
    interval_val = 1
    img_size_val = [640, 512]
    workers_val = os.cpu_count() # Default to number of CPU cores
    ffmpeg_bin_val = "ffmpeg"
    create_subsplits_val = [] # Example: ["trainval:train:val:0.8"]

    input_root  = Path(input_root_val)
    output_root = Path(output_root_val)

    for split in splits_val:
        print(f"[INFO] Split '{split}'")
        seq_dirs = [d for d in (input_root/split).iterdir() if d.is_dir()]
        print(f"  → {len(seq_dirs)} sequences found")

        print("  [STEP 1] dumping frames...")
        dump_args = [
            (sd, split, output_root, interval_val, img_size_val, ffmpeg_bin_val)
            for sd in seq_dirs
        ]
        with ProcessPoolExecutor(max_workers=workers_val) as exe:
            exe.map(dump_frames_task, dump_args)

        print("  [STEP 2] writing labels and sequences...")
        proc_args = [
            (sd.name, split, input_root_val, output_root_val,
             seq_len_val, img_size_val)
            for sd in seq_dirs
        ]
        all_lines = []
        with ProcessPoolExecutor(max_workers=workers_val) as exe:
            for lines in exe.map(process_sequence_task, proc_args):
                all_lines.extend(lines)

        seq_file = output_root/"sequences"/f"{split}.txt"
        seq_file.parent.mkdir(parents=True, exist_ok=True)
        seq_file.write_text("\\n".join(all_lines))
        print(f"  ✔ wrote {len(all_lines)} lines to {seq_file}\\n")

    # Create sub-splits if defined
    if create_subsplits_val:
        print("[INFO] Creating sub-splits...")
        for subsplit_def in create_subsplits_val:
            try:
                parts = subsplit_def.split(':')
                if len(parts) != 4:
                    raise ValueError("Incorrect number of parts")
                source_split, target1_name, target2_name, ratio1_str = parts
                ratio1 = float(ratio1_str)
                if not (0 < ratio1 < 1):
                    print(f"[WARN] Invalid ratio {ratio1} for sub-split '{subsplit_def}'. Must be between 0 (exclusive) and 1 (exclusive). Skipping.")
                    continue
            except ValueError as e:
                print(f"[WARN] Invalid format for sub-split definition '{subsplit_def}'. Error: {e}. "
                      "Expected 'source:target1:target2:ratio1'. Skipping.")
                continue

            source_seq_file = output_root / "sequences" / f"{source_split}.txt"
            if not source_seq_file.exists():
                print(f"[WARN] Source sequence file {source_seq_file} for sub-splitting not found. "
                      f"Ensure '{source_split}' was processed via --splits argument. Skipping '{subsplit_def}'.")
                continue

            print(f"  Processing sub-split: {source_split} -> {target1_name} ({ratio1*100:.1f}%), {target2_name} ({(1-ratio1)*100:.1f}%)")

            with open(source_seq_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                print(f"[WARN] Source sequence file {source_seq_file} is empty. Skipping sub-split '{subsplit_def}'.")
                continue

            random.shuffle(lines)

            split_idx = int(len(lines) * ratio1)
            lines_target1 = lines[:split_idx]
            lines_target2 = lines[split_idx:]

            target1_seq_file = output_root / "sequences" / f"{target1_name}.txt"
            # Parent directory output_root/"sequences" should exist from main split processing
            with open(target1_seq_file, 'w') as f:
                f.write("\\n".join(lines_target1))
            print(f"  ✔ Wrote {len(lines_target1)} lines to {target1_seq_file}")

            target2_seq_file = output_root / "sequences" / f"{target2_name}.txt"
            with open(target2_seq_file, 'w') as f:
                f.write("\\n".join(lines_target2))
            print(f"  ✔ Wrote {len(lines_target2)} lines to {target2_seq_file}")
        print("") # Add a newline for better formatting

    print("[INFO] All splits processed!")

if __name__=="__main__":
    main()
