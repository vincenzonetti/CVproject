# evaluate_detections.py — CLI tool to evaluate detection performance on sampled frames

import argparse
import json
import os
from utils import evaluate_tracking,read_yolo_label
import glob

def main():
    parser = argparse.ArgumentParser(description="Evaluate detections (not tracking) on every 5th frame")
    parser.add_argument('--results', type=str, required=True, help='Path to tracking_results.json')
    parser.add_argument('--video_name', type=str, required=True, help='Name of the video file (without extension)')

    args = parser.parse_args()

    # Constants
    LABEL_DIR = 'data/frames_annotated_yolov8/train/labels'
    NUM_FRAMES = 600
    FRAME_INTERVAL = 5
    IOU_THRESHOLD = 0.5

    # Load and filter tracking results — only every 5th key
    with open(args.results, 'r') as f:
        raw_tracking = json.load(f)

    detection_results = {
        i // FRAME_INTERVAL: raw_tracking[f"{args.video_name}_{i}"]
        for i in range(0, NUM_FRAMES, FRAME_INTERVAL)
        if f"{args.video_name}_{i}" in raw_tracking
    }

    label_paths = sorted(glob.glob(f'{LABEL_DIR}/{args.video_name}*.txt'))
    yolo_labels = []
    for lbl in label_paths:
        yolo_labels.append(read_yolo_label(lbl))
    
    # Run evaluation (detection-wise, not tracking-wise)
    metrics = evaluate_tracking(
        iou_threshold=IOU_THRESHOLD,
        ground_truth_boxes=yolo_labels,
        tracking_results=detection_results
    )
    breakpoint()
    print("\nDetection Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == '__main__':
    main()