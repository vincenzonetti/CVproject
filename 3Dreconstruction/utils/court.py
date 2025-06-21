"""
Court coordinate transformation utilities.

This module handles transformations between camera coordinates
and basketball court coordinates using homography.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict
from config import STANDARD_COURT_WIDTH_M


def setup_court_transformation(cam1_corners: Tuple[np.ndarray, np.ndarray],
                               cam2_corners: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Setup homography matrices for court coordinate transformation.
    
    Args:
        cam1_corners: Tuple of (real_corners, img_corners) for camera 1
        cam2_corners: Tuple of (real_corners, img_corners) for camera 2
        
    Returns:
        Tuple of (H1, H2, all_court_corners) where:
        - H1: Homography matrix for camera 1
        - H2: Homography matrix for camera 2
        - all_court_corners: Combined unique court corners in 3D
    """
    cam1_real_corners, cam1_img_corners = cam1_corners
    cam2_real_corners, cam2_img_corners = cam2_corners
    
    # Calculate homography for each camera
    # Using only x,y coordinates since z=0 for court
    cam1_court_2d = cam1_real_corners[:, :2]
    cam2_court_2d = cam2_real_corners[:, :2]
    
    H1, _ = cv2.findHomography(cam1_img_corners, cam1_court_2d)
    H2, _ = cv2.findHomography(cam2_img_corners, cam2_court_2d)
    
    # Combine all unique court corners
    all_corners = np.vstack([cam1_real_corners, cam2_real_corners])
    unique_corners = np.unique(all_corners, axis=0)
    
    # Print court information
    print_court_info(cam1_real_corners, cam2_real_corners, unique_corners)
    
    return H1, H2, unique_corners


def print_court_info(cam1_corners: np.ndarray, cam2_corners: np.ndarray, 
                     unique_corners: np.ndarray) -> None:
    """Print information about court dimensions and camera coverage."""
    max_x = np.max(unique_corners[:, 0])
    min_x = np.min(unique_corners[:, 0])
    max_y = np.max(unique_corners[:, 1])
    min_y = np.min(unique_corners[:, 1])
    court_width = max_x - min_x
    
    print(f"Camera 1 sees {len(cam1_corners)} corners")
    print(f"Camera 2 sees {len(cam2_corners)} corners")
    print(f"Total unique corners: {len(unique_corners)}")
    print(f"Court width detected: {court_width:.1f} meters")
    print(f"Court dimensions: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")


def transform_point_to_court(pt: List[float], H: np.ndarray) -> np.ndarray:
    """
    Transform a 2D image point to court coordinates using homography.
    
    Args:
        pt: 2D point in image coordinates [x, y]
        H: Homography matrix
        
    Returns:
        2D point in court coordinates [x, y]
    """
    pt_array = np.array([[pt]], dtype=np.float32)
    pt_court = cv2.perspectiveTransform(pt_array, H)[0, 0]
    return pt_court


def get_court_bounds(court_corners: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Get the bounds of the basketball court.
    
    Args:
        court_corners: Array of court corner coordinates
        
    Returns:
        Dictionary with 'x' and 'y' bounds as (min, max) tuples
    """
    return {
        'x': (np.min(court_corners[:, 0]), np.max(court_corners[:, 0])),
        'y': (np.min(court_corners[:, 1]), np.max(court_corners[:, 1]))
    }


def create_court_grid(court_corners: np.ndarray, grid_size: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid for court surface visualization.
    
    Args:
        court_corners: Array of court corner coordinates
        grid_size: Number of grid points in each dimension
        
    Returns:
        Tuple of (X, Y, Z) grid arrays for surface plotting
    """
    bounds = get_court_bounds(court_corners)
    
    x = np.linspace(bounds['x'][0], bounds['x'][1], grid_size)
    y = np.linspace(bounds['y'][0], bounds['y'][1], grid_size)
    
    # Create meshgrid
    # Note: In basketball convention, X is width, Z is depth, Y is height
    # So court surface has Y=0
    xx, zz = np.meshgrid(x, y)
    yy = np.zeros_like(xx)  # Y = 0 for court surface
    
    return xx, yy, zz