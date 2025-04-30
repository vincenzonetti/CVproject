import cv2
import numpy as np
import json
import os
import glob
import re
import yaml
from pathlib import Path
from typing import List, Dict, Union, Tuple
def load_calibration(calib_path):
    # Load the camera calibration parameters from a JSON file.
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

AbsoluteBox = List[int]  # [x1, y1, w, h]
YoloBoxCoords = List[float]  # [x_center, y_center, width, height]
# Define a type alias for a YOLO bounding box dictionary
YoloBoxDict = Dict[str, Union[int, YoloBoxCoords]]


def absolute_to_yolo(abs_box: AbsoluteBox, img_width: int, img_height: int) -> YoloBoxCoords:
        """Convert absolute box [x1, y1, w, h] to YOLO format"""
        x1, y1, w, h = abs_box
        
        x_center = (x1 + w/2) / img_width
        y_center = (y1 + h/2) / img_height
        width = w / img_width
        height = h / img_height
        
        return [x_center, y_center, width, height]


def read_yolo_label(label_path:str) -> List[YoloBoxDict] :
        """Read YOLO format label file"""
        boxes: List[YoloBoxDict] = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    class_id = int(data[0])
                    x_center, y_center, width, height = map(float, data[1:5])
                    boxes.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height],  # YOLO format
                    })
        return boxes
        
def yolo_to_absolute(yolo_box: YoloBoxCoords, img_width: int, img_height: int) -> AbsoluteBox:
        """Convert YOLO format box to absolute coordinates [x1, y1, w, h]"""
        x_center, y_center, width, height = yolo_box
        
        # Convert to absolute coordinates
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        w = int(width * img_width)
        h = int(height * img_height)
        
        return [x1, y1, w, h]


def process_frames(config_path, cam_idx, output_base_dir):
    
    calib_path = f'camera_data/cam_{cam_idx}/calib/camera_calib.json'
    
    mtx, dist = load_calibration(calib_path)
    
    data_dir = Path('../dataset')
    image_dir = data_dir / 'train' / 'images'
    label_dir = data_dir / 'train' / 'labels'
    image_paths = sorted(glob.glob(str(image_dir / f'out{cam_idx}*.jpg')))
    label_paths = sorted(glob.glob(str(label_dir / f'out{cam_idx}*.txt')))
    print(f"Found {len(image_paths)} images")
    
    height, width = cv2.imread(image_paths[0]).shape[:2]

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    

    undistorted_pts = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    map_x = undistorted_map[:, :, 0]
    map_y = undistorted_map[:, :, 1]
    
    output_image_dir = Path('rectified_frames') / 'train' / 'images'
    output_label_dir = Path('rectified_frames') / 'train' / 'labels'

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    
    for i in range(len(image_paths)):
        frame = cv2.imread(image_paths[i])
        rectified_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        detections = read_yolo_label(label_path=label_paths[i])
        remapped_detections: List[YoloBoxDict] = []
        for det in detections:
            class_id = det['class_id']
            coords = det['bbox'] 
            x1_abs_float, y1_abs_float, w_abs_float, h_abs_float = yolo_to_absolute(coords, width, height)
            corners_orig = np.array([
                [x1_abs_float, y1_abs_float],  # Top-left
                [x1_abs_float + w_abs_float, y1_abs_float],  # Top-right
                [x1_abs_float, y1_abs_float + h_abs_float],  # Bottom-left
                [x1_abs_float + w_abs_float, y1_abs_float + h_abs_float]  # Bottom-right
            ], dtype=np.float32)
            
            corners_orig_reshaped = corners_orig.reshape(-1, 1, 2)
            
            undistorted_corners = cv2.undistortPoints(corners_orig_reshaped, mtx, dist, P=mtx)
            undistorted_corners = undistorted_corners.reshape(-1, 2)
            
            # Find the min/max coordinates to form the new axis-aligned bounding box
            min_x = np.min(undistorted_corners[:, 0])
            min_y = np.min(undistorted_corners[:, 1])
            max_x = np.max(undistorted_corners[:, 0])
            max_y = np.max(undistorted_corners[:, 1])
            
            new_x1_abs_float = min_x
            new_y1_abs_float = min_y
            new_w_abs_float = max_x - min_x
            new_h_abs_float = max_y - min_y
            
            new_x1_abs_float = max(0.0, new_x1_abs_float)
            new_y1_abs_float = max(0.0, new_y1_abs_float)
            new_x2_abs_float = min(float(width), new_x1_abs_float + new_w_abs_float)
            new_y2_abs_float = min(float(height), new_y1_abs_float + new_h_abs_float)
            
            new_w_abs_float = new_x2_abs_float - new_x1_abs_float
            new_h_abs_float = new_y2_abs_float - new_y1_abs_float
            # Only keep the box if it has positive dimensions after remapping and clamping
            if new_w_abs_float > 0 and new_h_abs_float > 0:
                remapped_coords_yolo = absolute_to_yolo(
                    [new_x1_abs_float, new_y1_abs_float, new_w_abs_float, new_h_abs_float],
                    width,
                    height
                )

                # Store the remapped detection
                remapped_detections.append({
                    'class_id': class_id,
                    'bbox': remapped_coords_yolo
                })
                
        output_label_file = output_label_dir / Path(image_paths[i]).with_suffix('.txt').name
        with open(output_label_file, 'w') as f:
            for det in remapped_detections:
                class_id = det['class_id']
                bbox_yolo = det['bbox']
                # Write in YOLO format: class_id center_x center_y width height
                f.write(f"{class_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n") # Use .6f for floating point precision
        output_image_file = output_image_dir / Path(image_paths[i]).name
        cv2.imwrite(str(output_image_file), rectified_frame)
        
        #save frame
        frame_count += 1
        if frame_count % 100 == 0: # Print progress every 100 frames
             print(f"Processed frame {frame_count}/{len(image_paths)}")
        
    print("Finished processing")



if __name__ == "__main__":
    
    for i in [2,13]:
        process_frames(output_base_dir="rectified_frames", cam_idx=i ,config_path='../dataset')
