'''
python debug_labels.py 
  --output-root C:/data/processed_anti_uav 
  --split train

'''
#!/usr/bin/env python3
import argparse
import random
import cv2
from pathlib import Path

def main():
    p = argparse.ArgumentParser("Debug processed YOLO labels")
    p.add_argument("--output-root", required=True,
                   help="Root of processed dataset (images/, labels/)")
    p.add_argument("--split", default="train",
                   help="Which split to debug (train/val/test)")
    args = p.parse_args()

    out = Path(args.output_root)
    img_dir = out / "images" / args.split
    lbl_dir = out / "labels" / args.split

    images = sorted(img_dir.glob("*.jpg"))
    labels = sorted(lbl_dir.glob("*.txt"))

    print(f"{args.split}: {len(images)} images, {len(labels)} label files")

    nonempty = [lbl for lbl in labels if lbl.stat().st_size > 0]
    print(f"Non-empty label files: {len(nonempty)}")

    # print first few label contents
    print("\nFirst 5 non-empty labels:")
    for lbl in nonempty[:5]:
        print(f"  {lbl.name}: {lbl.read_text().strip()}")

    if not nonempty:
        print("No ground-truth boxes found! Your processing script may have skipped all annotations.")
        return

    # draw one random label
    lbl = random.choice(nonempty)
    img_name = lbl.stem + ".jpg"
    img = cv2.imread(str(img_dir / img_name))
    H, W = img.shape[:2]

    # parse YOLO format: "0 cx cy w h"
    _, cx, cy, w, h = map(float, lbl.read_text().split())
    x1 = int((cx - w/2) * W)
    y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W)
    y2 = int((cy + h/2) * H)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, lbl.name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    out_path = Path("debug_sample.png")
    cv2.imwrite(str(out_path), img)
    print(f"\nWrote one example with GT box → {out_path}")

if __name__=="__main__":
    main()
