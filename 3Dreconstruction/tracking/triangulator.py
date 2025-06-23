"""
3D triangulation utilities for stereo tracking.

This module handles triangulation of 3D points from stereo correspondences
and coordinate transformations to the basketball court system.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAX_REASONABLE_HEIGHT_M, HEIGHT_SCALE_FACTOR,BALL_MAX_HEIGHT_M,BALL_MIN_HEIGHT_M,BALL_TARGET_AVG_HEIGHT_M,BALL_FIRST_DETECTION_HEIGHT_M


# In 3Dreconstruction/tracking/triangulator.py

class Triangulator:
    """Handles 3D triangulation and coordinate transformations."""
    
    def __init__(self, P1: np.ndarray, P2: np.ndarray, 
                 cam1_params: dict, cam2_params: dict,
                 H1: Optional[np.ndarray] = None, 
                 H2: Optional[np.ndarray] = None):
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
        self.H1 = H1
        self.H2 = H2
        self.ball_raw_heights = []  # Store raw ball heights for later scaling
        self.ball_height_scale = None  # Will be computed after first pass
        self.ball_height_offset = None  # Will be computed after first pass
    
    def triangulate_point(self, pt1: list, pt2: list, obj_name: str, 
                         collect_only: bool = False) -> np.ndarray:
        """
        Triangulate 3D point from 2D correspondences.
        
        Args:
            pt1: 2D point in camera 1 [x, y]
            pt2: 2D point in camera 2 [x, y]
            obj_name: Name of the object (used for special handling of 'Ball')
            collect_only: If True, only collect data without scaling (first pass)
            
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
        
        # Transform to court coordinates if homographies available
        if self.H1 is not None and self.H2 is not None:
            points_3d = self._transform_to_court_coords(
                pt1, pt2, points_3d_camera, obj_name, collect_only
            )
        else:
            # No court transformation, return camera coordinates
            points_3d = points_3d_camera.flatten()
        
        return points_3d
    
    def compute_ball_height_scaling(self):
        """
        Compute scaling parameters for ball heights to achieve:
        - First detected height: 1.1 meters
        - Maximum detected height: 4.5 meters
        - Smooth scaling for all other heights
        """
        if not self.ball_raw_heights:
            print("Warning: No ball heights collected for scaling")
            return
        
        raw_heights = np.array(self.ball_raw_heights)
        
        # Get first and maximum heights
        self.first_ball_height = raw_heights[0]  # First detected height
        max_raw_height = np.max(raw_heights)
        
        # Compute scaling factor
        # We want: first_height -> 1.1m and max_height -> 4.5m
        if max_raw_height != self.first_ball_height:
            # Scale factor based on the range from first to max
            height_range_raw = max_raw_height - self.first_ball_height
            height_range_target = BALL_MAX_HEIGHT_M - BALL_FIRST_DETECTION_HEIGHT_M
            self.ball_height_scale = height_range_target / height_range_raw
        else:
            # All heights are the same, no scaling needed
            self.ball_height_scale = 1.0
        
        # Verify the scaling
        print(f"\nBall height scaling computed:")
        print(f"  First raw height: {self.first_ball_height:.3f}m -> {BALL_FIRST_DETECTION_HEIGHT_M}m")
        print(f"  Max raw height: {max_raw_height:.3f}m -> {BALL_MAX_HEIGHT_M}m")
        print(f"  Scale factor: {self.ball_height_scale:.3f}")
        
        # Show some example scaled heights
        example_heights = [np.min(raw_heights), np.median(raw_heights), np.mean(raw_heights)]
        print(f"  Examples:")
        for h in example_heights:
            scaled = BALL_FIRST_DETECTION_HEIGHT_M + (h - self.first_ball_height) * self.ball_height_scale
            print(f"    {h:.3f}m -> {scaled:.3f}m")
            
    def _transform_to_court_coords(self, pt1: list, pt2: list, 
                                   points_3d_camera: np.ndarray, 
                                   obj_name: str, collect_only: bool = False) -> np.ndarray:
        """
        Transform 3D point from camera coordinates to court coordinates.
        
        Args:
            pt1: 2D point in camera 1
            pt2: 2D point in camera 2
            points_3d_camera: 3D point in camera coordinates
            obj_name: Object name for special handling
            collect_only: If True, only collect data without scaling
            
        Returns:
            3D point in court coordinates [X, Y, Z]
        """
        # Transform each camera's point to court coordinates
        pt1_court = cv2.perspectiveTransform(
            np.array([[pt1]], dtype=np.float32), self.H1
        )[0, 0]
        pt2_court = cv2.perspectiveTransform(
            np.array([[pt2]], dtype=np.float32), self.H2
        )[0, 0]
        
        # Average the court positions from both views
        court_xy = (pt1_court + pt2_court) / 2
        
        # Get height from triangulated point using both cameras
        height = self._extract_height_stereo(points_3d_camera, pt1, pt2)
        
        # Basketball convention: X (width), Y (height), Z (depth)
        if obj_name == 'Ball':
            if collect_only:
                # First pass: collect raw heights
                self.ball_raw_heights.append(abs(height))
                y_coord = abs(height)  # Use raw height for now
            else:
                # Second pass: apply scaling
                if self.ball_height_scale is not None and self.ball_height_offset is not None:
                    # Apply the computed scaling
                    raw_height = abs(height)
                    # Find the min height from collected data for normalization
                    min_height = min(self.ball_raw_heights)
                    scaled_height = (raw_height - min_height) * self.ball_height_scale + BALL_MIN_HEIGHT_M
                    y_coord = scaled_height + self.ball_height_offset
                    
                    # Ensure we stay within bounds
                    y_coord = np.clip(y_coord, BALL_MIN_HEIGHT_M, BALL_MAX_HEIGHT_M)
                else:
                    # Fallback if scaling not computed
                    y_coord = abs(height)
        else:
            y_coord = 0.0  # Players on the ground
        
        # Create final position: X, Y (height), Z (depth)
        return np.array([court_xy[0], y_coord, court_xy[1]])
    
    def _extract_height_stereo(self, points_3d_camera: np.ndarray, 
                               pt1: list, pt2: list) -> float:
        """
        Extract and scale height from camera coordinates using stereo information.
        
        Args:
            points_3d_camera: 3D point in camera coordinates
            pt1: 2D point from camera 1
            pt2: 2D point from camera 2
            
        Returns:
            Height value in meters
        """
        # Use both camera parameters for better height estimation
        R1, _ = cv2.Rodrigues(self.cam1_params['rvecs'])
        t1 = self.cam1_params['tvecs'].reshape(-1, 1)
        
        R2, _ = cv2.Rodrigues(self.cam2_params['rvecs'])
        t2 = self.cam2_params['tvecs'].reshape(-1, 1)
        
        # Transform to world coordinates using camera 1
        points_3d_world1 = R1.T @ (points_3d_camera - t1)
        
        # Also compute using camera 2 for comparison
        # First need to transform the 3D point to camera 2's coordinate system
        points_3d_cam2 = R2 @ points_3d_world1 + t2
        points_3d_world2 = R2.T @ (points_3d_cam2 - t2)
        
        # Average the height estimates from both cameras
        z_world1 = points_3d_world1[2, 0]
        z_world2 = points_3d_world2[2, 0]
        z_world = (z_world1 + z_world2) / 2
        
        # Scale if necessary
        if abs(z_world) > MAX_REASONABLE_HEIGHT_M:
            z_world *= HEIGHT_SCALE_FACTOR
        
        return z_world

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