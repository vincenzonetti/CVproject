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
    x, y, w, h = bbox
    pt1 = np.array([[(x * IMG_WIDTH, y * IMG_HEIGHT)]], dtype=np.float32)
    pt2 = np.array([[((x + w) * IMG_WIDTH, (y + h) * IMG_HEIGHT)]], dtype=np.float32)

    undistorted = cv2.undistortPoints(np.concatenate([pt1, pt2]), mtx, dist, P=mtx)
    pt1_u, pt2_u = undistorted[0][0], undistorted[1][0]

    x_u = float(pt1_u[0] / IMG_WIDTH)
    y_u = float(pt1_u[1] / IMG_HEIGHT)
    w_u = float((pt2_u[0] - pt1_u[0]) / IMG_WIDTH)
    h_u = float((pt2_u[1] - pt1_u[1]) / IMG_HEIGHT)

    return [x_u, y_u, w_u, h_u]

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
