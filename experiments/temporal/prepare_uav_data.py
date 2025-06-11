
#python prepare_uav_data.py 
#  --input-root "C:\Users\Logan\Downloads\Anti-UAV-RGBT" 
#  --output-root "C:\data\processed_anti_uav" 
#  --splits train val test 
#  --img-size 640 512 
#  --workers 8 
#  --ffmpeg-bin ffmpeg


#!/usr/bin/env python3
import argparse
import json
import subprocess
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2

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
    parser = argparse.ArgumentParser("Prepare Anti-UAV YOLO data")
    parser.add_argument("--input-root",  required=True,
                        help="Root with train/val/test folders")
    parser.add_argument("--output-root", required=True,
                        help="Where to write images/, labels/, sequences/")
    parser.add_argument("--splits", nargs="+", default=["train","val","test"])
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--img-size", nargs=2, type=int, default=[640,512])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    args = parser.parse_args()

    input_root  = Path(args.input_root)
    output_root = Path(args.output_root)

    for split in args.splits:
        print(f"[INFO] Split '{split}'")
        seq_dirs = [d for d in (input_root/split).iterdir() if d.is_dir()]
        print(f"  → {len(seq_dirs)} sequences found")

        print("  [STEP 1] dumping frames...")
        dump_args = [
            (sd, split, output_root, args.interval, args.img_size, args.ffmpeg_bin)
            for sd in seq_dirs
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            exe.map(dump_frames_task, dump_args)

        print("  [STEP 2] writing labels and sequences...")
        proc_args = [
            (sd.name, split, args.input_root, args.output_root,
             args.seq_len, args.img_size)
            for sd in seq_dirs
        ]
        all_lines = []
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            for lines in exe.map(process_sequence_task, proc_args):
                all_lines.extend(lines)

        seq_file = output_root/"sequences"/f"{split}.txt"
        seq_file.parent.mkdir(parents=True, exist_ok=True)
        seq_file.write_text("\n".join(all_lines))
        print(f"  ✔ wrote {len(all_lines)} lines to {seq_file}\n")

    print("[INFO] All splits processed!")

if __name__=="__main__":
    main()
