import json
import numpy as np
import cv2
import argparse
import os

IMG_WIDTH = 3840
IMG_HEIGHT = 2160

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_calibration(calib_path):
    with open(calib_path, "r") as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def rectify_bbox_simple(bbox, mtx, dist):
    """
    Simple rectification - only center point (for comparison)
    """
    x_center, y_center, width, height = bbox
    
    # Undistort center point
    center_point = np.array([[[x_center * IMG_WIDTH, y_center * IMG_HEIGHT]]], dtype=np.float32)
    undistorted_center = cv2.undistortPoints(center_point, mtx, dist, P=mtx)
    
    x_center_u = undistorted_center[0][0][0] / IMG_WIDTH
    y_center_u = undistorted_center[0][0][1] / IMG_HEIGHT
    
    # Keep original width and height (approximation)
    return [x_center_u, y_center_u, width, height]

def process_tracking(tracking_path, calib_path):
    mtx, dist = load_calibration(calib_path)
    
    with open(tracking_path, "r") as f:
        tracking_data = json.load(f)
    
    rectified_data = {}
    
    for frame_id, objects in tracking_data.items():
        rectified_objects = []
        
        for obj in objects:
            bbox_u = rectify_bbox_simple(obj["bbox"], mtx, dist)
            
            # Skip objects that are completely outside the image
            if bbox_u is None:
                continue
                
            rectified_obj = {
                "class_id": obj["class_id"],
                "bbox": bbox_u
            }
            
            # Preserve other fields if they exist
            for key, value in obj.items():
                if key not in ["class_id", "bbox"]:
                    rectified_obj[key] = value
            
            rectified_objects.append(rectified_obj)
        
        rectified_data[frame_id] = rectified_objects
    
    # Save output
    base_dir = os.path.dirname(tracking_path)
    base_name = os.path.basename(tracking_path)
    name_wo_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(base_dir, f"rectified_{name_wo_ext}.json")
    with open(output_path, "w") as f:
        json.dump(rectified_data, f,indent=2,cls=NumpyEncoder)
    
    print(f"Saved rectified tracking to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rectify bounding boxes using camera calibration.")
    parser.add_argument("--results", type=str, required=True, help="Path to tracking results JSON.")
    parser.add_argument("--calib", type=str, required=True, help="Path to camera calibration JSON.")
    
    args = parser.parse_args()
    
    output_path = process_tracking(args.results, args.calib)