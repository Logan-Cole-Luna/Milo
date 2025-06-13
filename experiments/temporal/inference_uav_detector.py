#!/usr/bin/env python3

import argparse
from pathlib import Path
import cv2
import numpy as np
import onnxruntime
from collections import deque
import time # Added import

# Normalization parameters (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224

def preprocess(frame):
    # BGR -> RGB, resize, normalize, transpose
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    return img


def bbox_to_pixels(bbox_norm, width, height):
    # bbox_norm: [cx, cy, w, h] in normalized (0-1) coords
    cx, cy, w_norm, h_norm = bbox_norm
    x1 = int((cx - w_norm/2) * width)
    y1 = int((cy - h_norm/2) * height)
    x2 = int((cx + w_norm/2) * width)
    y2 = int((cy + h_norm/2) * height)
    return x1, y1, x2, y2


def main():
    # Configuration (hardcoded)
    # 3700000000002_113556_2
    INPUT_PATH = Path(r"C:\Users\Logan\Downloads\train_uav\train\3700000000002_113556_2")
    SINGLE_MODEL_PATH = Path(r"experiments\temporal\results\uav_detection_v3\single_frame_default_resnet18\AdamW\single_frame_default_resnet18_AdamW_model.onnx")
    TEMPORAL_MODEL_PATH = Path(r"experiments\temporal\results\uav_detection_v3\temporal_s5_default_resnet18\AdamW\temporal_s5_default_resnet18_AdamW_model.onnx")
    SEQ_LEN = 5
    OUTPUT_PATH = Path("experiments/temporal/output_video_v3.mp4")

    input_path = INPUT_PATH
    single_model_path = SINGLE_MODEL_PATH
    temporal_model_path = TEMPORAL_MODEL_PATH
    seq_len = SEQ_LEN

    # Create ONNX runtime sessions
    single_sess = None
    if single_model_path:
        print(f"Loading single-frame model from {single_model_path}")
        single_sess = onnxruntime.InferenceSession(str(single_model_path))
    temporal_sess = None
    if temporal_model_path:
        print(f"Loading temporal model from {temporal_model_path}")
        temporal_sess = onnxruntime.InferenceSession(str(temporal_model_path))

    # Initialize sequence buffer (for temporal inference) and font for annotations
    seq_buffer = deque(maxlen=seq_len)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Determine input type
    ext = input_path.suffix.lower()
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    is_video = input_path.is_file() and ext in video_exts
    is_dir = input_path.is_dir()

    if is_video:
        cap = cv2.VideoCapture(str(input_path))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        writer = None
        if OUTPUT_PATH:
            # Ensure overwriting by deleting if exists
            if OUTPUT_PATH.exists():
                OUTPUT_PATH.unlink()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))
    elif is_dir:
        # Process a folder of frames into a single MP4 video
        frame_files = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
        if not frame_files:
            print(f"No images found in directory {input_path}")
            return
        # Determine video size from first frame
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]
        # Initialize VideoWriter for MP4
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, 30.0, (width, height))
        for img_file in frame_files:
            frame = cv2.imread(str(img_file))
            if frame is None:
                continue
            # Single-frame inference and drawing
            inp = preprocess(frame)
            pred = single_sess.run(None, {'input_image': inp[None, ...]})[0][0]
            x1, y1, x2, y2 = bbox_to_pixels(pred, width, height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Temporal inference if available
            if temporal_sess:
                seq_buffer.append(inp)
                if len(seq_buffer) == seq_len:
                    pred_t = temporal_sess.run(None, {'input_sequence': np.stack(seq_buffer)[None, ...]})[0][0]
                    tx1, ty1, tx2, ty2 = bbox_to_pixels(pred_t, width, height)
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
            writer.write(frame)
        writer.release()
        print(f"Saved annotated video to {OUTPUT_PATH}")
        return
    else:
        # Single image inference
        frame = cv2.imread(str(input_path))
        if frame is None:
            print(f"Error: Could not load image {input_path}")
            return
        height, width = frame.shape[:2]

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        # else: frame is already loaded

        # Single-frame inference
        inp = preprocess(frame)
        input_array = inp[np.newaxis, ...]  # shape: 1,3,224,224
        
        start_time_single = time.perf_counter()
        pred = single_sess.run(None, {'input_image': input_array})[0][0]
        end_time_single = time.perf_counter()
        inference_time_single = (end_time_single - start_time_single) * 1000  # milliseconds

        x1, y1, x2, y2 = bbox_to_pixels(pred, width, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Single", (x1, y1 - 10), font, 0.9, (0, 255, 0), 2) # Green text

        # Temporal inference (if enabled)
        if temporal_sess:
            seq_buffer.append(inp)
            if len(seq_buffer) == seq_len:
                seq_arr = np.stack(seq_buffer, axis=0)  # seq_len,3,224,224
                seq_input = seq_arr[np.newaxis, ...]    # 1,seq_len,3,224,224
                
                start_time_temporal = time.perf_counter()
                pred_t = temporal_sess.run(None, {'input_sequence': seq_input})[0][0]
                end_time_temporal = time.perf_counter()
                inference_time_temporal = (end_time_temporal - start_time_temporal) * 1000 # milliseconds
                
                tx1, ty1, tx2, ty2 = bbox_to_pixels(pred_t, width, height)
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2) # Red box (BGR)
                cv2.putText(frame, "Temporal", (tx1, ty1 - 10), font, 0.9, (0, 0, 255), 2) # Red text (BGR)
                
                # Display temporal inference time
                cv2.putText(frame, f"Temporal: {inference_time_temporal:.2f} ms", (10, height - 20), font, 0.7, (0, 0, 255), 2) # Red text

        # Display single-frame inference time
        cv2.putText(frame, f"Single: {inference_time_single:.2f} ms", (10, height - 50), font, 0.7, (0, 255, 0), 2) # Green text


        # Output without GUI display
        if is_video:
            # write processed frame to output video
            if writer:
                writer.write(frame)
            # continue until video ends
        else:
            # save annotated image
            if OUTPUT_PATH:
                # Ensure overwriting by deleting if exists
                if OUTPUT_PATH.exists():
                    OUTPUT_PATH.unlink()
                cv2.imwrite(str(OUTPUT_PATH), frame)
                print(f"Saved output image to {OUTPUT_PATH}")
            break

    if is_video:
        cap.release()
        print(f"Saved output video to {OUTPUT_PATH}")
        if writer:
            writer.release()

if __name__ == '__main__':
    main()
