"""
Trajectory metrics calculation for basketball tracking.

This module computes various movement metrics including speed,
distance, acceleration, and coverage area for tracked objects.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DIRECTION_CHANGE_THRESHOLD_RAD, MIN_TRAJECTORY_POINTS


def extract_trajectories(tracking_3d_results: Dict) -> Dict[str, Dict]:
    """
    Extract trajectory data for each object from tracking results.
    
    Args:
        tracking_3d_results: Dictionary with frame-wise 3D positions
        
    Returns:
        Dictionary mapping object names to their trajectories
        {obj_name: {'frames': [...], 'positions': [...]}}
    """
    trajectories = {}
    
    for frame_key, frame_data in tracking_3d_results.items():
        frame_num = int(frame_key.split('_')[1])
        for obj_name, obj_data in frame_data.items():
            if obj_name not in trajectories:
                trajectories[obj_name] = {'frames': [], 'positions': []}
            trajectories[obj_name]['frames'].append(frame_num)
            trajectories[obj_name]['positions'].append(obj_data['position'])
    
    return trajectories


def calculate_trajectory_metrics(tracking_3d_results: Dict, fps: float = 25) -> Dict[str, Dict]:
    """
    Calculate comprehensive trajectory metrics for each tracked object.
    
    Args:
        tracking_3d_results: Dictionary with frame-wise 3D positions
        fps: Frames per second for speed calculations
        
    Returns:
        Dictionary of metrics for each object including:
        - total_ground_distance_m: Distance traveled on court (X-Z plane)
        - total_3d_distance_m: Total 3D distance including height
        - avg_ground_speed_ms: Average speed on court
        - max_ground_speed_ms: Maximum speed on court
        - avg_acceleration_ms2: Average acceleration
        - direction_changes: Number of significant direction changes
        - coverage_area_m2: Area covered on court
        - avg_height_m: Average height above court
        - max_height_m: Maximum height above court
        - height_variation_m: Standard deviation of height
        - total_frames: Number of frames tracked
        - time_tracked_s: Total time tracked in seconds
    """
    metrics = {}
    trajectories = extract_trajectories(tracking_3d_results)
    
    for obj_name, traj in trajectories.items():
        if len(traj['positions']) < MIN_TRAJECTORY_POINTS:
            continue
        
        positions = np.array(traj['positions'])
        frames = np.array(traj['frames'])
        
        # Calculate ground movement (X-Z plane)
        ground_metrics = _calculate_ground_metrics(positions, frames, fps)
        
        # Calculate height statistics (Y axis)
        height_metrics = _calculate_height_metrics(positions)
        
        # Calculate 3D distance
        total_3d_distance = _calculate_total_3d_distance(positions)
        
        # Combine all metrics
        metrics[obj_name] = {
            **ground_metrics,
            **height_metrics,
            'total_3d_distance_m': float(total_3d_distance),
            'total_frames': len(frames),
            'time_tracked_s': float((frames[-1] - frames[0]) / fps) if len(frames) > 1 else 0
        }
    
    return metrics


def _calculate_ground_metrics(positions: np.ndarray, frames: np.ndarray, 
                             fps: float) -> Dict[str, float]:
    """Calculate metrics for movement on the court (X-Z plane)."""
    # Extract ground positions (X and Z only)
    ground_positions = positions[:, [0, 2]]
    
    # Calculate distances between consecutive positions
    ground_distances = np.sqrt(np.sum(np.diff(ground_positions, axis=0)**2, axis=1))
    total_ground_distance = np.sum(ground_distances)
    
    # Calculate speeds
    time_diffs = np.diff(frames) / fps
    ground_speeds = ground_distances / time_diffs
    avg_ground_speed = np.mean(ground_speeds) if len(ground_speeds) > 0 else 0
    max_ground_speed = np.max(ground_speeds) if len(ground_speeds) > 0 else 0
    
    # Calculate acceleration
    accelerations = np.diff(ground_speeds) / time_diffs[:-1] if len(ground_speeds) > 1 else []
    avg_acceleration = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0
    
    # Count direction changes
    direction_changes = _count_direction_changes(ground_positions)
    
    # Calculate coverage area
    coverage_area = _calculate_coverage_area(ground_positions)
    
    return {
        'total_ground_distance_m': float(total_ground_distance),
        'avg_ground_speed_ms': float(avg_ground_speed),
        'max_ground_speed_ms': float(max_ground_speed),
        'avg_acceleration_ms2': float(avg_acceleration),
        'direction_changes': int(direction_changes),
        'coverage_area_m2': float(coverage_area)
    }


def _calculate_height_metrics(positions: np.ndarray) -> Dict[str, float]:
    """Calculate metrics related to height (Y axis)."""
    heights = positions[:, 1]
    
    return {
        'avg_height_m': float(np.mean(heights)),
        'max_height_m': float(np.max(heights)),
        'height_variation_m': float(np.std(heights))
    }


def _calculate_total_3d_distance(positions: np.ndarray) -> float:
    """Calculate total distance traveled in 3D space."""
    distances_3d = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    return np.sum(distances_3d)


def _count_direction_changes(ground_positions: np.ndarray) -> int:
    """Count significant direction changes in movement."""
    if len(ground_positions) < 3:
        return 0
    
    vectors = np.diff(ground_positions, axis=0)
    direction_changes = 0
    
    for i in range(len(vectors) - 1):
        v1 = vectors[i]
        v2 = vectors[i + 1]
        
        # Calculate angle between consecutive movement vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        if angle > DIRECTION_CHANGE_THRESHOLD_RAD:
            direction_changes += 1
    
    return direction_changes


def _calculate_coverage_area(ground_positions: np.ndarray) -> float:
    """Calculate the area covered on the court (bounding box area)."""
    if len(ground_positions) < 2:
        return 0.0
    
    min_pos = np.min(ground_positions, axis=0)
    max_pos = np.max(ground_positions, axis=0)
    
    return np.prod(max_pos - min_pos)