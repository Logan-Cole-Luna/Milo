
#!/usr/bin/env python3
import random
import cv2
from pathlib import Path

def main():
    # === EDIT THIS TO YOUR DATA ROOT ===
    data_root = Path("C:/data/processed_anti_uav")
    split = "train"
    out_path = Path("sample_verify.png")

    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split

    # collect all image files
    imgs = list(img_dir.glob("*.jpg"))
    if not imgs:
        raise RuntimeError(f"No images found in {img_dir}")

    # pick a random one
    img_p = random.choice(imgs)
    lbl_p = lbl_dir / (img_p.stem + ".txt")

    # load image
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]

    # attempt to read GT bbox
    if lbl_p.exists() and lbl_p.stat().st_size > 0:
        cls, cx, cy, bw, bh = map(float, lbl_p.read_text().split())
        # convert normalized center/size to pixel corners
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "UAV"
    else:
        label = "No UAV"

    # put text
    cv2.putText(img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    # save
    cv2.imwrite(str(out_path), img)
    print(f"Sample: {img_p.name} → {out_path}")

if __name__ == "__main__":
    main()
