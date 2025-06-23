"""
Trajectory cleaning and interpolation utilities.

This module handles removal of false positives and outliers
from tracked trajectories, and interpolation of missing points.
"""

import numpy as np
from scipy import interpolate
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_trajectories(tracking_3d_results: Dict) -> Dict:
    """
    Clean trajectories by removing outliers and interpolating missing points.
    
    Args:
        tracking_3d_results: Dictionary with frame-wise 3D positions
        
    Returns:
        Cleaned tracking results with outliers removed and interpolated
    """
    # First extract all trajectories
    trajectories = extract_trajectories_with_frames(tracking_3d_results)
    
    # Clean each trajectory
    cleaned_trajectories = {}
    for obj_name, traj_data in trajectories.items():        # Clean player trajectories
        cleaned_traj = clean_single_trajectory(
            traj_data['frames'], 
            traj_data['positions'], 
            obj_name
        )
        if cleaned_traj:
            cleaned_trajectories[obj_name] = cleaned_traj
    
    # Rebuild tracking results from cleaned trajectories
    cleaned_results = rebuild_tracking_results(cleaned_trajectories)
    
    return cleaned_results


def extract_trajectories_with_frames(tracking_3d_results: Dict) -> Dict:
    """Extract trajectories with frame numbers."""
    trajectories = {}
    
    for frame_key, frame_data in tracking_3d_results.items():
        frame_num = int(frame_key.split('_')[1])
        for obj_name, obj_data in frame_data.items():
            if obj_name not in trajectories:
                trajectories[obj_name] = {'frames': [], 'positions': []}
            trajectories[obj_name]['frames'].append(frame_num)
            trajectories[obj_name]['positions'].append(obj_data['position'])
    
    return trajectories


def clean_single_trajectory(frames: List[int], positions: List[List[float]], 
                           obj_name: str) -> Dict:
    """
    Clean a single trajectory by removing outliers.
    
    Args:
        frames: List of frame numbers
        positions: List of 3D positions
        obj_name: Name of the object for debugging
        
    Returns:
        Dictionary with cleaned frames and positions
    """
    if len(positions) < 3:
        # Not enough points to clean
        return {'frames': frames, 'positions': positions}
    
    frames = np.array(frames)
    positions = np.array(positions)
    
    # Check for teleportation (sudden jumps in displacement)
    teleport_threshold = 0.4  # meters - adjust as needed
    
    # Mark points to keep
    keep_points = np.ones(len(positions), dtype=bool)
    
    # Check each transition
    for i in range(1, len(positions)):
        # Check displacement between consecutive points
        displacement = np.linalg.norm(positions[i] - positions[i-1])
        teleport_threshold = 1000 if obj_name == 'Ball' else teleport_threshold
        if displacement > teleport_threshold:
            keep_points[i] = False
            print(f"Removing {obj_name} frame {frames[i]}: teleportation {displacement:.2f} m > {teleport_threshold:.2f} m")
    
    # Keep at least 3 points for interpolation
    
    if np.sum(keep_points) < 3:
        print(f"Warning: {obj_name} has too few valid points after cleaning")
        return {'frames': frames.tolist(), 'positions': positions.tolist()}
    
    # Extract clean points
    clean_frames = frames[keep_points]
    clean_positions = positions[keep_points]
    
    # Interpolate to fill gaps
    interpolated_data = interpolate_trajectory(
        clean_frames, clean_positions, frames[0], frames[-1]
    )
    
    return interpolated_data


def interpolate_trajectory(clean_frames: np.ndarray, clean_positions: np.ndarray,
                          start_frame: int, end_frame: int) -> Dict:
    """
    Interpolate trajectory to fill missing frames using spline interpolation.
    
    Args:
        clean_frames: Frame numbers of clean points
        clean_positions: 3D positions of clean points
        start_frame: First frame number
        end_frame: Last frame number
        
    Returns:
        Dictionary with interpolated frames and positions
    """
    # Create frame range
    all_frames = np.arange(start_frame, end_frame + 1)
    
    # Interpolate each dimension separately
    interpolated_positions = []
    
    try:
        # Use cubic spline interpolation if enough points
        if len(clean_frames) >= 4:
            # Cubic spline for each dimension
            interp_x = interpolate.CubicSpline(clean_frames, clean_positions[:, 0])
            interp_y = interpolate.CubicSpline(clean_frames, clean_positions[:, 1])
            interp_z = interpolate.CubicSpline(clean_frames, clean_positions[:, 2])
        else:
            # Linear interpolation if too few points
            interp_x = interpolate.interp1d(clean_frames, clean_positions[:, 0], 
                                          kind='linear', fill_value='extrapolate')
            interp_y = interpolate.interp1d(clean_frames, clean_positions[:, 1], 
                                          kind='linear', fill_value='extrapolate')
            interp_z = interpolate.interp1d(clean_frames, clean_positions[:, 2], 
                                          kind='linear', fill_value='extrapolate')
        
        # Interpolate for all frames
        for frame in all_frames:
            if frame < clean_frames[0] or frame > clean_frames[-1]:
                # Skip extrapolation beyond data range
                continue
            
            x = float(interp_x(frame))
            y = float(interp_y(frame))
            z = float(interp_z(frame))
            interpolated_positions.append([x, y, z])
        
        # Only keep frames within interpolation range
        valid_frames = all_frames[(all_frames >= clean_frames[0]) & 
                                 (all_frames <= clean_frames[-1])]
        
        return {
            'frames': valid_frames.tolist(),
            'positions': interpolated_positions
        }
        
    except Exception as e:
        print(f"Interpolation failed: {e}")
        # Return original clean data if interpolation fails
        return {
            'frames': clean_frames.tolist(),
            'positions': clean_positions.tolist()
        }


def rebuild_tracking_results(cleaned_trajectories: Dict) -> Dict:
    """
    Rebuild tracking results dictionary from cleaned trajectories.
    
    Args:
        cleaned_trajectories: Dictionary of cleaned trajectories
        
    Returns:
        Tracking results in original format
    """
    new_results = {}
    
    # Find all unique frames
    all_frames = set()
    for traj_data in cleaned_trajectories.values():
        all_frames.update(traj_data['frames'])
    
    # Build results for each frame
    for frame_num in sorted(all_frames):
        frame_key = f"frame_{frame_num}"
        frame_data = {}
        
        # Add data for each object present in this frame
        for obj_name, traj_data in cleaned_trajectories.items():
            if frame_num in traj_data['frames']:
                idx = traj_data['frames'].index(frame_num)
                frame_data[obj_name] = {
                    'position': traj_data['positions'][idx]
                }
        
        if frame_data:  # Only add frame if it has data
            new_results[frame_key] = frame_data
    
    return new_results