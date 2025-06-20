import json
import numpy as np
import cv2
import argparse
import os

IMG_WIDTH = 3840
IMG_HEIGHT = 2160

def load_calibration(calib_path):
    with open(calib_path, "r") as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def rectify_bbox(bbox, mtx, dist):
    """
    Rectify bounding box by undistorting all 4 corners and finding new bbox
    """
    # Assuming bbox format is [x_center, y_center, width, height] in normalized coords
    # Convert to pixel coordinates
    x_center, y_center, width, height = bbox
    x_center_px = x_center * IMG_WIDTH
    y_center_px = y_center * IMG_HEIGHT
    width_px = width * IMG_WIDTH
    height_px = height * IMG_HEIGHT
    
    # Calculate all 4 corners
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2
    
    # Define all 4 corners of the bounding box
    corners = np.array([
        [[x1, y1]],  # top-left
        [[x2, y1]],  # top-right
        [[x2, y2]],  # bottom-right
        [[x1, y2]]   # bottom-left
    ], dtype=np.float32)
    
    # Undistort all corners
    undistorted_corners = cv2.undistortPoints(corners, mtx, dist, P=mtx)
    
    # Extract undistorted coordinates
    undistorted_coords = undistorted_corners.reshape(-1, 2)
    
    # Find bounding box of undistorted corners
    x_min = np.min(undistorted_coords[:, 0])
    y_min = np.min(undistorted_coords[:, 1])
    x_max = np.max(undistorted_coords[:, 0])
    y_max = np.max(undistorted_coords[:, 1])
    
    # Convert back to normalized coordinates
    x_center_u = (x_min + x_max) / 2 / IMG_WIDTH
    y_center_u = (y_min + y_max) / 2 / IMG_HEIGHT
    width_u = (x_max - x_min) / IMG_WIDTH
    height_u = (y_max - y_min) / IMG_HEIGHT
    
    return [x_center_u, y_center_u, width_u, height_u]

def process_tracking(tracking_path, calib_path):
    mtx, dist = load_calibration(calib_path)

    with open(tracking_path, "r") as f:
        tracking_data = json.load(f)

    rectified_data = {}
    for frame_id, objects in tracking_data.items():
        rectified_objects = []
        for obj in objects:
            bbox_u = rectify_bbox(obj["bbox"], mtx, dist)
            rectified_objects.append({"class_id": obj["class_id"], "bbox": bbox_u})
        rectified_data[frame_id] = rectified_objects

    base_dir = os.path.dirname(tracking_path)
    base_name = os.path.basename(tracking_path)
    name_wo_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(base_dir, f"rectified_{name_wo_ext}.json")

    with open(output_path, "w") as f:
        json.dump(rectified_data, f)
    print(f"Saved rectified tracking to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rectify bounding boxes using camera calibration.")
    parser.add_argument("--results", type=str, required=True, help="Path to tracking results JSON.")
    parser.add_argument("--calib", type=str, required=True, help="Path to camera calibration JSON.")
    args = parser.parse_args()

    process_tracking(args.results, args.calib)
