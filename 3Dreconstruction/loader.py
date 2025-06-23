"""
Data loading utilities for basketball tracking system.

This module handles loading of detection data, camera parameters,
and court corner information from JSON files.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMG_WIDTH, IMG_HEIGHT, CLASSES, CAMERA_PARAMS_KEYS


def load_detections(detection_path: str) -> List[Dict[str, Dict]]:
    """
    Load detection data from JSON file and convert to proper format.
    
    Args:
        detection_path: Path to detection JSON file
        
    Returns:
        List of dictionaries containing detections for each frame,
        with class names as keys and bbox/center information as values
    """
    with open(detection_path, 'r') as f:
        params = json.load(f)

    formatted_params = []
    for key, value in params.items():
        frame_detections = {}
        
        # Group detections by class_id to handle duplicates
        detections_by_class = {}
        for detection in value:
            class_id = detection['class_id']
            
            # If we already have a detection for this class, keep the one with higher confidence
            if class_id in detections_by_class:
                if detection.get('conf', 0) > detections_by_class[class_id].get('conf', 0):
                    detections_by_class[class_id] = detection
            else:
                detections_by_class[class_id] = detection
        
        # Process the filtered detections
        for class_id, detection in detections_by_class.items():
            class_name = CLASSES[class_id]  # Convert to class name
            bbox = detection['bbox']
            
            # Convert normalized coordinates to pixel coordinates
            # bbox format: [x_center, y_center, width, height] (normalized)
            x_center_px = bbox[0] * IMG_WIDTH
            y_center_px = bbox[1] * IMG_HEIGHT
            
            frame_detections[class_name] = {
                'bbox': bbox,
                'center': [x_center_px, y_center_px],
                'confidence': detection.get('conf', 1.0)  # Store confidence if available
            }
        
        formatted_params.append(frame_detections)

    return formatted_params

def load_camera_params(params_path: str) -> Dict[str, np.ndarray]:
    """
    Load camera calibration parameters from JSON file.
    
    Args:
        params_path: Path to camera parameters JSON file
        
    Returns:
        Dictionary containing camera matrices and distortion coefficients
    """
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Convert lists to numpy arrays
    for key in CAMERA_PARAMS_KEYS:
        if key in params:
            params[key] = np.array(params[key])
    
    return params


def load_court_corners(corners_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load court corner coordinates from JSON file.
    
    Args:
        corners_path: Path to court corners JSON file
        
    Returns:
        Tuple of (real_corners, img_corners) as numpy arrays
    """
    with open(corners_path, 'r') as f:
        corners = json.load(f)
    
    real_corners = np.array(corners['real_corners'])
    img_corners = np.array(corners['img_corners'])
    
    return real_corners, img_corners


def load_stereo_data(detection1_path: str, detection2_path: str,
                     cam1_params_path: str, cam2_params_path: str,
                     cam1_corners_path: str = None, cam2_corners_path: str = None) -> Dict[str, Any]:
    """
    Load all data required for stereo tracking.
    
    Args:
        detection1_path: Path to camera 1 detections
        detection2_path: Path to camera 2 detections
        cam1_params_path: Path to camera 1 parameters
        cam2_params_path: Path to camera 2 parameters
        cam1_corners_path: Optional path to camera 1 court corners
        cam2_corners_path: Optional path to camera 2 court corners
        
    Returns:
        Dictionary containing all loaded data
    """
    data = {
        'detections1': load_detections(detection1_path),
        'detections2': load_detections(detection2_path),
        'cam1_params': load_camera_params(cam1_params_path),
        'cam2_params': load_camera_params(cam2_params_path)
    }
    
    if cam1_corners_path and cam2_corners_path:
        data['cam1_real_corners'], data['cam1_img_corners'] = load_court_corners(cam1_corners_path)
        data['cam2_real_corners'], data['cam2_img_corners'] = load_court_corners(cam2_corners_path)
    
    return data