import argparse
import json
import os
from utils import read_yolo_label_mod, yolo_to_absolute
from particle_histogram_matching import calculate_histogram
import glob
import cv2
from tqdm import tqdm
import numpy as np

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main():
    parser = argparse.ArgumentParser(description="Evaluate detections (not tracking) on every 5th frame")

    parser.add_argument('-c', '--camera_name', type=str, required=True, help='camera_name')
    parser.add_argument('-i', '--image_dir', type=str, required=True, help='path to images')
    parser.add_argument('-l', '--label_dir', type=str, required=True, help='path to YOLO annotation')

    args = parser.parse_args()

    # Constants
    LABEL_DIR = args.label_dir
    IMAGE_DIR = args.image_dir
    CAM_NAME = args.camera_name
    ball_class = 0
    label_paths = sorted(glob.glob(f'{LABEL_DIR}/out{CAM_NAME}*.txt'))
    image_paths = sorted(glob.glob(f'{IMAGE_DIR}/out{CAM_NAME}*.jpg'))

    sample_frame = cv2.imread(image_paths[0], cv2.IMREAD_COLOR_BGR)
    img_h, img_w = sample_frame.shape[:2]

    histograms = []
    
    for idx, (lbl, frame_name) in enumerate(zip(label_paths, image_paths)):
        yolo_lbl = read_yolo_label_mod(lbl)

        if ball_class in yolo_lbl:
            ball_bbox = yolo_to_absolute(yolo_lbl[ball_class], img_w, img_h)
            xc, yc, bw, bh = ball_bbox

            frame = cv2.imread(frame_name, cv2.IMREAD_COLOR_BGR)

            ball_roi = frame[yc:yc+15, xc:xc+15]
            
            blue_channel, green_channel, red_channel = cv2.split(ball_roi)
            
            # Calculate histograms for each channel
            hist_r = calculate_histogram(red_channel, 'Red', 'red')
            hist_g = calculate_histogram(green_channel, 'Green', 'green')
            hist_b = calculate_histogram(blue_channel, 'Blue', 'blue')

            # Combine histograms
            reference_histogram = {
                'red': hist_r,
                'green': hist_g,
                'blue': hist_b
            }
            histograms.append(reference_histogram)
        else:
            histograms.append({})

    # Save histograms to a JSON file
    with open('histograms.json', 'w') as json_file:
        json.dump(histograms, json_file, indent=4, default=convert_ndarray_to_list)

if __name__ == '__main__':
    main()
