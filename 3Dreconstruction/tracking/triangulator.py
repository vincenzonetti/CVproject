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

from config import MAX_REASONABLE_HEIGHT_M, HEIGHT_SCALE_FACTOR


class Triangulator:
    """Handles 3D triangulation and coordinate transformations."""
    
    def __init__(self, P1: np.ndarray, P2: np.ndarray, 
                 cam_params: dict, H1: Optional[np.ndarray] = None, 
                 H2: Optional[np.ndarray] = None):
        """
        Initialize triangulator with projection matrices and homographies.
        
        Args:
            P1: Projection matrix for camera 1
            P2: Projection matrix for camera 2
            cam_params: Camera parameters dictionary
            H1: Optional homography for camera 1 to court transformation
            H2: Optional homography for camera 2 to court transformation
        """
        self.P1 = P1
        self.P2 = P2
        self.cam_params = cam_params
        self.H1 = H1
        self.H2 = H2
    
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
        
        # Transform to court coordinates if homographies available
        if self.H1 is not None and self.H2 is not None:
            points_3d = self._transform_to_court_coords(
                pt1, pt2, points_3d_camera, obj_name
            )
        else:
            # No court transformation, return camera coordinates
            points_3d = points_3d_camera.flatten()
        
        return points_3d
    
    def _transform_to_court_coords(self, pt1: list, pt2: list, 
                                   points_3d_camera: np.ndarray, 
                                   obj_name: str) -> np.ndarray:
        """
        Transform 3D point from camera coordinates to court coordinates.
        
        Args:
            pt1: 2D point in camera 1
            pt2: 2D point in camera 2
            points_3d_camera: 3D point in camera coordinates
            obj_name: Object name for special handling
            
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
        
        # Get height from triangulated point
        height = self._extract_height(points_3d_camera)
        
        # Basketball convention: X (width), Y (height), Z (depth)
        # For the ball, keep the triangulated height but divide by 2
        if obj_name == 'Ball':
            y_coord = abs(height) / 3.0  # Height divided by 2 for ball
        else:
            y_coord = 0.0  # Players on the ground
        
        # Create final position: X, Y (height), Z (depth)
        return np.array([court_xy[0], y_coord, court_xy[1]])
    
    def _extract_height(self, points_3d_camera: np.ndarray) -> float:
        """
        Extract and scale height from camera coordinates.
        
        Args:
            points_3d_camera: 3D point in camera coordinates
            
        Returns:
            Height value in meters
        """
        # Get rotation matrix from camera 1
        R1, _ = cv2.Rodrigues(self.cam_params['rvecs'])
        t1 = self.cam_params['tvecs'].reshape(-1, 1)
        
        # Transform to world coordinates
        points_3d_world = R1.T @ (points_3d_camera - t1)
        
        # Extract height (Z in world coordinates)
        z_world = points_3d_world[2, 0]
        
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