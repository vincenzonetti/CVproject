# In 3Dreconstruction/tracking/triangulator.py

import cv2
import numpy as np
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (MAX_REASONABLE_HEIGHT_M, HEIGHT_SCALE_FACTOR,
                   BALL_FIRST_DETECTION_HEIGHT_M, BALL_MAX_HEIGHT_M)


class Triangulator:
    """Handles 3D triangulation and coordinate transformations."""
    
    def __init__(self, P1: np.ndarray, P2: np.ndarray, 
                 cam1_params: dict, cam2_params: dict):
        """
        Initialize triangulator with projection matrices and homographies.
        
        Args:
            P1: Projection matrix for camera 1
            P2: Projection matrix for camera 2
            cam1_params: Camera 1 parameters dictionary
            cam2_params: Camera 2 parameters dictionary
            H1: Optional homography for camera 1 to court transformation
            H2: Optional homography for camera 2 to court transformation
        """
        self.P1 = P1
        self.P2 = P2
        self.cam1_params = cam1_params
        self.cam2_params = cam2_params
        self.first_ball_height = None  # Will be set during scaling
        self.ball_height_scale = None  # Will be computed after cleaning
        self.ball_height_offset = None  # Will be computed after cleaning
    
    def triangulate_point(self, pt1: list, pt2: list, obj_name: str) -> np.ndarray:
        """
        Triangulate 3D point from 2D correspondences.
        
        Args:
            pt1: 2D point in camera 1 [x, y]
            pt2: 2D point in camera 2 [x, y]
            obj_name: Name of the object (used for special handling of 'Ball')
            
        Returns:
            3D point in court coordinates [X, Y, Z] where:
            - X = width (side to side)
            - Y = height (vertical, 0 for ground)
            - Z = depth (baseline to baseline)
        """
        # Convert points to homogeneous coordinates
        pt1_homo = np.array([[pt1[0]], [pt1[1]]], dtype=np.float32)
        pt2_homo = np.array([[pt2[0]], [pt2[1]]], dtype=np.float32)
        
        # Triangulate in camera coordinate system
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pt1_homo, pt2_homo)
        
        # Convert from homogeneous to 3D coordinates
        points_3d_camera = points_4d[:3] / points_4d[3]
        if obj_name == 'Ball':
            # No court transformation, return camera coordinates
            points_3d = points_3d_camera.flatten()
            points_3d[1], points_3d[2] = points_3d[2], points_3d[1]
            
        else:
            points_3d = points_3d_camera.flatten()
            points_3d[1], points_3d[2] = points_3d[2], points_3d[1]
            for i in range(len(points_3d)):
                points_3d[i]/=1000
        
        return points_3d
    
    def compute_ball_height_scaling(self, tracking_3d_results: Dict) -> None:
        """
        Compute scaling parameters for ball heights from cleaned trajectories.
        
        Args:
            tracking_3d_results: Cleaned 3D tracking results
        """
        # Extract all ball heights from cleaned trajectories
        ball_heights = []
        frame_numbers = []
        
        for frame_key in sorted(tracking_3d_results.keys(), 
                               key=lambda x: int(x.split('_')[1])):
            if 'Ball' in tracking_3d_results[frame_key]:
                height = tracking_3d_results[frame_key]['Ball']['position'][1]
                ball_heights.append(height)
                frame_numbers.append(int(frame_key.split('_')[1]))
        
        if not ball_heights:
            print("Warning: No ball heights found in cleaned trajectories")
            return
        
        
        ball_heights = np.array(ball_heights)
        
        # Get first and maximum heights
        self.first_ball_height = ball_heights[0]
        max_ball_height = np.max(ball_heights)
        
        # Compute scaling factor
        if max_ball_height != self.first_ball_height:
            height_range_raw = max_ball_height - self.first_ball_height
            height_range_target = BALL_MAX_HEIGHT_M - BALL_FIRST_DETECTION_HEIGHT_M
            self.ball_height_scale = height_range_target / height_range_raw
        else:
            self.ball_height_scale = 1.0
        
        # Set offset to 0 (not used in this scaling method)
        self.ball_height_offset = 0.0
        
        # Verify the scaling
        print(f"\nBall height scaling computed from cleaned trajectories:")
        print(f"  Total ball detections: {len(ball_heights)}")
        print(f"  First raw height: {self.first_ball_height:.3f}m -> {BALL_FIRST_DETECTION_HEIGHT_M}m")
        print(f"  Max raw height: {max_ball_height:.3f}m -> {BALL_MAX_HEIGHT_M}m")
        print(f"  Scale factor: {self.ball_height_scale:.3f}")
        
        # Show distribution
        print(f"  Height distribution:")
        print(f"    Min: {np.min(ball_heights):.3f}m")
        print(f"    Mean: {np.mean(ball_heights):.3f}m")
        print(f"    Median: {np.median(ball_heights):.3f}m")
        print(f"    Max: {np.max(ball_heights):.3f}m")
    
    def apply_ball_height_scaling(self, tracking_3d_results: Dict) -> Dict:
        """
        Apply the computed scaling to all ball positions in the tracking results.
        
        Args:
            tracking_3d_results: Original tracking results
            
        Returns:
            Updated tracking results with scaled ball heights
        """
        if self.ball_height_scale is None or self.first_ball_height is None:
            print("Warning: Ball height scaling not computed, returning original results")
            return tracking_3d_results
        
        # Create a copy to avoid modifying the original
        scaled_results = {}
        
        for frame_key, frame_data in tracking_3d_results.items():
            scaled_results[frame_key] = {}
            
            for obj_name, obj_data in frame_data.items():
                if obj_name == 'Ball':
                    # Apply scaling to ball height
                    position = obj_data['position'].copy()
                    raw_height = position[1]
                    
                    # Apply scaling formula
                    scaled_height = BALL_FIRST_DETECTION_HEIGHT_M + \
                                  (raw_height - self.first_ball_height) * self.ball_height_scale
                    
                    # Ensure within bounds
                    scaled_height = np.clip(scaled_height, 0.0, BALL_MAX_HEIGHT_M)
                    position[1] = scaled_height
                    position[0]/=1000
                    position[2]/=1000
                    # Store updated data
                    
                    scaled_results[frame_key][obj_name] = {
                        'position': position,
                    }
                    
                else:
                    # Copy non-ball objects as-is
                    scaled_results[frame_key][obj_name] = obj_data
        
        return scaled_results

def compute_projection_matrix(cam_params: dict) -> np.ndarray:
    """
    Compute projection matrix P = K[R|t].
    
    Args:
        cam_params: Camera parameters containing intrinsics and extrinsics
        
    Returns:
        3x4 projection matrix
    """
    K = cam_params['mtx']  # Intrinsic matrix
    rvec = cam_params['rvecs']
    tvec = cam_params['tvecs']
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create [R|t] matrix
    Rt = np.hstack((R, tvec.reshape(-1, 1)))
    
    # Compute projection matrix P = K[R|t]
    P = K @ Rt
    
    return P