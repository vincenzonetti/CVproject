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

def rectify_bbox_improved(bbox, mtx, dist):
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

def rectify_bbox_clipped(bbox, mtx, dist):
    """
    Rectify bounding box and clip to image boundaries
    """
    # Get unclamped rectified bbox
    rect_bbox = rectify_bbox_improved(bbox, mtx, dist)
    x_center, y_center, width, height = rect_bbox
    
    # Calculate bbox boundaries
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Clip to image boundaries [0, 1]
    x1_clipped = max(0.0, min(1.0, x1))
    y1_clipped = max(0.0, min(1.0, y1))
    x2_clipped = max(0.0, min(1.0, x2))
    y2_clipped = max(0.0, min(1.0, y2))
    
    # Recalculate center and dimensions
    width_clipped = x2_clipped - x1_clipped
    height_clipped = y2_clipped - y1_clipped
    x_center_clipped = (x1_clipped + x2_clipped) / 2
    y_center_clipped = (y1_clipped + y2_clipped) / 2
    
    # Check if bbox is still valid after clipping
    if width_clipped <= 0 or height_clipped <= 0:
        print(f"Warning: Bbox completely outside image after rectification")
        return None
    
    return [x_center_clipped, y_center_clipped, width_clipped, height_clipped]

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

def process_tracking(tracking_path, calib_path, method='improved'):
    mtx, dist = load_calibration(calib_path)
    
    with open(tracking_path, "r") as f:
        tracking_data = json.load(f)
    
    rectified_data = {}
    
    for frame_id, objects in tracking_data.items():
        rectified_objects = []
        
        for obj in objects:
            if method == 'improved':
                bbox_u = rectify_bbox_improved(obj["bbox"], mtx, dist)
            elif method == 'clipped':
                bbox_u = rectify_bbox_clipped(obj["bbox"], mtx, dist)
            else:
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
    output_path = os.path.join(base_dir, f"rectified_{method}_{name_wo_ext}.json")
    breakpoint()
    with open(output_path, "w") as f:
        json.dump(rectified_data, f,indent=2,cls=NumpyEncoder)
    
    print(f"Saved rectified tracking to: {output_path}")
    return output_path

def visualize_rectification_difference(original_bbox, rectified_bbox, frame_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """
    Visualize the difference between original and rectified bounding boxes
    """
    img = np.zeros(frame_shape, dtype=np.uint8)
    
    def draw_bbox(img, bbox, color, label):
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width/2) * IMG_WIDTH)
        y1 = int((y_center - height/2) * IMG_HEIGHT)
        x2 = int((x_center + width/2) * IMG_WIDTH)
        y2 = int((y_center + height/2) * IMG_HEIGHT)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    draw_bbox(img, original_bbox, (0, 255, 0), "Original")  # Green
    draw_bbox(img, rectified_bbox, (0, 0, 255), "Rectified")  # Red
    
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rectify bounding boxes using camera calibration.")
    parser.add_argument("--results", type=str, required=True, help="Path to tracking results JSON.")
    parser.add_argument("--calib", type=str, required=True, help="Path to camera calibration JSON.")
    parser.add_argument("--method", type=str, choices=['simple', 'improved', 'clipped'], default='improved',
                       help="Rectification method: 'simple' (center only), 'improved' (all corners), or 'clipped' (clipped to image)")
    parser.add_argument("--visualize", action='store_true', help="Create visualization of first bbox")
    
    args = parser.parse_args()
    
    output_path = process_tracking(args.results, args.calib, args.method)
    
    if args.visualize:
        # Load original and rectified data for comparison
        with open(args.results, 'r') as f:
            original_data = json.load(f)
        with open(output_path, 'r') as f:
            rectified_data = json.load(f) 
        
        # Get first frame with objects
        for frame_id in original_data:
            if original_data[frame_id]:
                original_bbox = original_data[frame_id][0]["bbox"]
                rectified_bbox = rectified_data[frame_id][0]["bbox"]
                
                viz_img = visualize_rectification_difference(original_bbox, rectified_bbox)
                
                output_dir = os.path.dirname(output_path)
                viz_path = os.path.join(output_dir, "bbox_rectification_comparison.jpg")
                cv2.imwrite(viz_path, viz_img)
                print(f"Visualization saved to: {viz_path}")
                break