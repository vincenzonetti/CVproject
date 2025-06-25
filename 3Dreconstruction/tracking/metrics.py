"""
Enhanced trajectory metrics calculation with speed outlier removal.

This module computes movement metrics with robust outlier detection
and removal for speed and acceleration calculations.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DIRECTION_CHANGE_THRESHOLD_RAD, MIN_TRAJECTORY_POINTS, DEFAULT_FPS

# Configuration for outlier removal
SPEED_OUTLIER_STD_THRESHOLD = 2.0  # Standard deviations for speed outlier detection
ACCELERATION_OUTLIER_STD_THRESHOLD = 2.0  # Standard deviations for acceleration outlier detection
MIN_SPEED_SAMPLES = 5  # Minimum samples needed for outlier detection
MAX_REASONABLE_SPEED_MS = 12.0  # Maximum reasonable speed for basketball players (m/s)
MAX_REASONABLE_ACCELERATION_MS2 = 8.0  # Maximum reasonable acceleration (m/s²) - reduced from 15.0


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


def remove_speed_outliers(speeds: np.ndarray, indices: np.ndarray, 
                         obj_name: str, std_threshold: float = SPEED_OUTLIER_STD_THRESHOLD) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Remove speed outliers using statistical thresholding with physical constraints.
    
    Args:
        speeds: Array of speeds (m/s)
        indices: Corresponding indices for the speeds
        obj_name: Object name for debugging
        std_threshold: Number of standard deviations for outlier detection
        
    Returns:
        Tuple of (clean_speeds, clean_indices, outlier_stats)
    """
    # First pass: remove physically impossible speeds
    physical_mask = speeds <= MAX_REASONABLE_SPEED_MS
    
    # Second pass: statistical outlier detection on physically reasonable speeds
    if np.sum(physical_mask) >= MIN_SPEED_SAMPLES:
        reasonable_speeds = speeds[physical_mask]
        
        # Calculate statistics on reasonable speeds
        mean_speed = np.mean(reasonable_speeds)
        std_speed = np.std(reasonable_speeds)
        
        # Create statistical mask for all speeds
        lower_bound = max(0, mean_speed - std_threshold * std_speed)  # Speed can't be negative
        upper_bound = mean_speed + std_threshold * std_speed
        
        # Combine physical and statistical constraints
        statistical_mask = (speeds >= lower_bound) & (speeds <= upper_bound)
        final_mask = physical_mask & statistical_mask
    else:
        # If too few reasonable speeds, just use physical constraint
        final_mask = physical_mask
        mean_speed = np.mean(speeds[physical_mask]) if np.sum(physical_mask) > 0 else 0
        std_speed = np.std(speeds[physical_mask]) if np.sum(physical_mask) > 0 else 0
        lower_bound = 0
        upper_bound = MAX_REASONABLE_SPEED_MS
    
    # Apply mask
    clean_speeds = speeds[final_mask]
    clean_indices = indices[final_mask]
    
    # Statistics
    outliers_removed = len(speeds) - len(clean_speeds)
    outlier_stats = {
        'outliers_removed': outliers_removed,
        'total_samples': len(speeds),
        'outlier_percentage': (outliers_removed / len(speeds)) * 100 if len(speeds) > 0 else 0,
        'original_max_speed': float(np.max(speeds)) if len(speeds) > 0 else 0,
        'cleaned_max_speed': float(np.max(clean_speeds)) if len(clean_speeds) > 0 else 0,
        'mean_speed': float(mean_speed),
        'std_speed': float(std_speed),
        'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
    }
    
    if outliers_removed > 0:
        print(f"  {obj_name}: Removed {outliers_removed}/{len(speeds)} speed outliers "
              f"({outlier_stats['outlier_percentage']:.1f}%) - "
              f"Max speed: {outlier_stats['original_max_speed']:.1f} → {outlier_stats['cleaned_max_speed']:.1f} m/s")
    
    return clean_speeds, clean_indices, outlier_stats


def remove_acceleration_outliers(accelerations: np.ndarray, 
                                std_threshold: float = ACCELERATION_OUTLIER_STD_THRESHOLD) -> Tuple[np.ndarray, Dict]:
    """
    Remove acceleration outliers using statistical thresholding with physical constraints.
    
    Args:
        accelerations: Array of acceleration magnitudes (m/s²)
        std_threshold: Number of standard deviations for outlier detection
        
    Returns:
        Tuple of (clean_accelerations, outlier_stats)
    """
    if len(accelerations) < MIN_SPEED_SAMPLES:
        return accelerations, {'outliers_removed': 0, 'total_samples': len(accelerations)}
    
    # First pass: remove physically impossible accelerations
    physical_mask = accelerations <= MAX_REASONABLE_ACCELERATION_MS2
    
    # Second pass: statistical outlier detection
    if np.sum(physical_mask) >= MIN_SPEED_SAMPLES:
        reasonable_accel = accelerations[physical_mask]
        
        # Calculate statistics
        mean_accel = np.mean(reasonable_accel)
        std_accel = np.std(reasonable_accel)
        
        # Create statistical mask
        upper_bound = mean_accel + std_threshold * std_accel
        statistical_mask = accelerations <= upper_bound
        
        # Combine constraints
        final_mask = physical_mask & statistical_mask
    else:
        # If too few reasonable accelerations, just use physical constraint
        final_mask = physical_mask
    
    clean_accelerations = accelerations[final_mask]
    
    # Statistics
    outliers_removed = len(accelerations) - len(clean_accelerations)
    outlier_stats = {
        'outliers_removed': outliers_removed,
        'total_samples': len(accelerations),
        'outlier_percentage': (outliers_removed / len(accelerations)) * 100 if len(accelerations) > 0 else 0,
        'original_max_accel': float(np.max(accelerations)) if len(accelerations) > 0 else 0,
        'cleaned_max_accel': float(np.max(clean_accelerations)) if len(clean_accelerations) > 0 else 0
    }
    
    return clean_accelerations, outlier_stats


def calculate_trajectory_metrics(tracking_3d_results: Dict, fps: float = DEFAULT_FPS) -> Dict[str, Dict]:
    """
    Calculate comprehensive trajectory metrics with outlier removal for speed/acceleration.
    
    Args:
        tracking_3d_results: Dictionary with frame-wise 3D positions
        fps: Frames per second for speed calculations
        
    Returns:
        Dictionary of metrics for each object including outlier removal statistics
    """
    print("\n=== Calculating Trajectory Metrics with Outlier Removal ===")
    
    metrics = {}
    trajectories = extract_trajectories(tracking_3d_results)
    
    for obj_name, traj in trajectories.items():
        if len(traj['positions']) < MIN_TRAJECTORY_POINTS:
            continue
        
        print(f"\nProcessing {obj_name}...")
        
        positions = np.array(traj['positions'])
        frames = np.array(traj['frames'])
        
        # Calculate ground movement (X-Z plane) - positions unchanged
        ground_metrics = _calculate_ground_metrics_with_outlier_removal(
            positions, frames, fps, obj_name
        )
        
        # Calculate height statistics (Y axis) - unchanged
        height_metrics = _calculate_height_metrics(positions)
        
        # Calculate 3D distance - unchanged
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


def _calculate_ground_metrics_with_outlier_removal(positions: np.ndarray, frames: np.ndarray, 
                                                  fps: float, obj_name: str) -> Dict[str, float]:
    """Calculate movement metrics with speed/acceleration outlier removal. Uses 3D for ball, 2D for players."""
    
    # For ball, use 3D movement; for players, use ground plane (X-Z) only
    if obj_name == 'Ball':
        # Ball moves in full 3D space (X, Y, Z)
        movement_positions = positions  # All 3 dimensions
        movement_type = "3D"
    else:
        # Players move primarily on ground plane (X, Z only)
        movement_positions = positions[:, [0, 2]]  # X and Z only
        movement_type = "2D"
    
    # Calculate distances between consecutive positions
    movement_distances = np.sqrt(np.sum(np.diff(movement_positions, axis=0)**2, axis=1))
    
    # Keep ground distance for players (for compatibility with existing metrics)
    if obj_name == 'Ball':
        ground_positions = positions[:, [0, 2]]  # Still calculate ground distance for ball
        ground_distances = np.sqrt(np.sum(np.diff(ground_positions, axis=0)**2, axis=1))
        total_ground_distance = np.sum(ground_distances)
    else:
        total_ground_distance = np.sum(movement_distances)  # Same as movement distance for players
    
    # Calculate raw speeds with actual time differences
    time_diffs = np.diff(frames) / fps
    raw_movement_speeds = movement_distances / time_diffs
    
    # Remove speed outliers
    if len(raw_movement_speeds) > 0:
        clean_speeds, clean_speed_indices, speed_outlier_stats = remove_speed_outliers(
            raw_movement_speeds, np.arange(len(raw_movement_speeds)), obj_name
        )
        
        avg_ground_speed = np.mean(clean_speeds) if len(clean_speeds) > 0 else 0
        max_ground_speed = np.max(clean_speeds) if len(clean_speeds) > 0 else 0
        
        # For ball, also add 3D speed metrics
        if obj_name == 'Ball':
            speed_outlier_stats['movement_type'] = movement_type
            speed_outlier_stats['avg_3d_speed_ms'] = avg_ground_speed  # Using same cleaned speeds
            speed_outlier_stats['max_3d_speed_ms'] = max_ground_speed
    else:
        clean_speeds = np.array([])
        clean_speed_indices = np.array([])
        avg_ground_speed = 0
        max_ground_speed = 0
        speed_outlier_stats = {'outliers_removed': 0, 'total_samples': 0, 'outlier_percentage': 0}
    
    # Calculate accelerations using a simpler approach
    if len(clean_speeds) > 1:
        # Get consecutive clean speeds and their corresponding time differences
        consecutive_pairs = []
        
        for i in range(len(clean_speed_indices) - 1):
            idx1 = clean_speed_indices[i]
            idx2 = clean_speed_indices[i + 1]
            
            # Only use consecutive or near-consecutive measurements
            if idx2 - idx1 <= 3:  # Allow small gaps (max 3 frames)
                time_diff = (frames[idx2 + 1] - frames[idx1 + 1]) / fps  # +1 because speeds are for transitions
                if time_diff > 0:
                    speed_diff = clean_speeds[i + 1] - clean_speeds[i]
                    acceleration = abs(speed_diff / time_diff)
                    consecutive_pairs.append(acceleration)
        
        if len(consecutive_pairs) > 0:
            raw_accelerations = np.array(consecutive_pairs)
            
            # Remove acceleration outliers
            clean_accelerations, accel_outlier_stats = remove_acceleration_outliers(raw_accelerations)
            avg_acceleration = np.mean(clean_accelerations) if len(clean_accelerations) > 0 else 0
        else:
            avg_acceleration = 0
            accel_outlier_stats = {'outliers_removed': 0, 'total_samples': 0, 'outlier_percentage': 0}
    else:
        avg_acceleration = 0
        accel_outlier_stats = {'outliers_removed': 0, 'total_samples': 0, 'outlier_percentage': 0}
    
    # Count direction changes (use ground plane for all objects for consistency)
    if obj_name == 'Ball':
        direction_changes = _count_direction_changes(ground_positions)
    else:
        direction_changes = _count_direction_changes(movement_positions)
    
    # Calculate coverage area (use ground plane for all objects)
    if obj_name == 'Ball':
        coverage_area = _calculate_coverage_area(ground_positions)
    else:
        coverage_area = _calculate_coverage_area(movement_positions)
    
    # Build return dictionary with additional ball metrics if applicable
    result = {
        'total_ground_distance_m': float(total_ground_distance),
        'avg_ground_speed_ms': float(avg_ground_speed),
        'max_ground_speed_ms': float(max_ground_speed),
        'avg_acceleration_ms2': float(avg_acceleration),
        'direction_changes': int(direction_changes),
        'coverage_area_m2': float(coverage_area),
        # Outlier removal statistics
        'speed_outliers_removed': speed_outlier_stats['outliers_removed'],
        'speed_outlier_percentage': speed_outlier_stats['outlier_percentage'],
        'acceleration_outliers_removed': accel_outlier_stats['outliers_removed'],
        'acceleration_outlier_percentage': accel_outlier_stats['outlier_percentage'],
        'original_max_speed_ms': speed_outlier_stats.get('original_max_speed', max_ground_speed),
        'cleaned_max_speed_ms': float(max_ground_speed)
    }
    
    # Add 3D metrics for ball
    if obj_name == 'Ball':
        result.update({
            'movement_type': '3D',
            'avg_3d_speed_ms': float(avg_ground_speed),  # This is actually 3D speed for ball
            'max_3d_speed_ms': float(max_ground_speed),  # This is actually 3D speed for ball
            'total_3d_movement_distance_m': float(np.sum(movement_distances))
        })
    else:
        result['movement_type'] = '2D'
    
    return result


def _calculate_height_metrics(positions: np.ndarray) -> Dict[str, float]:
    """Calculate metrics related to height (Y axis) - unchanged."""
    heights = positions[:, 1]
    
    return {
        'avg_height_m': float(np.mean(heights)),
        'max_height_m': float(np.max(heights)),
        'height_variation_m': float(np.std(heights))
    }


def _calculate_total_3d_distance(positions: np.ndarray) -> float:
    """Calculate total distance traveled in 3D space - unchanged."""
    distances_3d = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    return np.sum(distances_3d)


def _count_direction_changes(ground_positions: np.ndarray) -> int:
    """Count significant direction changes in movement - unchanged."""
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
    """Calculate the area covered on the court (bounding box area) - unchanged."""
    if len(ground_positions) < 2:
        return 0.0
    
    min_pos = np.min(ground_positions, axis=0)
    max_pos = np.max(ground_positions, axis=0)
    
    return np.prod(max_pos - min_pos)

