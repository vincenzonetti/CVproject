"""
Object matching utilities for stereo tracking.

This module handles matching detected objects between two camera views
based on their class identities.
"""

from typing import Dict, Tuple, List


# In 3Dreconstruction/tracking/matcher.py

def match_objects(det1: List[Dict[str, Dict]], det2: List[Dict[str, Dict]]) -> Tuple[List[Dict], Dict, Dict]:
    """
    Match objects between two camera views across all frames based on class names.
    
    Args:
        det1: List of detections from camera 1, one dict per frame
        det2: List of detections from camera 2, one dict per frame
        
    Returns:
        Tuple of:
        - matches: List of frame matches, each containing {class_name: (idx1, idx2)}
        - frame_map1: Mapping of detections by frame for camera 1
        - frame_map2: Mapping of detections by frame for camera 2
    """
    # Ensure both cameras have the same number of frames
    num_frames = min(len(det1), len(det2))
    if len(det1) != len(det2):
        print(f"Warning: Camera 1 has {len(det1)} frames, Camera 2 has {len(det2)} frames. Using first {num_frames} frames.")
    
    matches = []
    frame_map1 = {}
    frame_map2 = {}
    
    # Process each frame
    for frame_idx in range(num_frames):
        frame1_detections = det1[frame_idx]
        frame2_detections = det2[frame_idx]
        
        # Store frame detections in maps
        frame_map1[f'frame_{frame_idx}'] = frame1_detections
        frame_map2[f'frame_{frame_idx}'] = frame2_detections
        
        # Match objects in this frame
        frame_matches = {}
        
        # For each object detected in camera 1
        for obj_name in frame1_detections:
            if obj_name in frame2_detections:
                # Both cameras see this object
                frame_matches[obj_name] = (frame_idx, frame_idx)
            else:
                # Only camera 1 sees this object
                frame_matches[obj_name] = (frame_idx, None)
        
        # Check for objects only seen by camera 2
        for obj_name in frame2_detections:
            if obj_name not in frame1_detections:
                # Only camera 2 sees this object
                frame_matches[obj_name] = (None, frame_idx)
        
        matches.append(frame_matches)
    
    return matches, frame_map1, frame_map2


def compute_matching_statistics(matches: List[Dict], det1: List[Dict], det2: List[Dict]) -> Dict:
    """
    Compute statistics about the matching results.
    
    Args:
        matches: List of frame matches
        det1: List of detections from camera 1
        det2: List of detections from camera 2
        
    Returns:
        Dictionary containing matching statistics
    """
    total_det1 = sum(len(frame_det) for frame_det in det1)
    total_det2 = sum(len(frame_det) for frame_det in det2)
    
    matched_count = 0
    cam1_only_count = 0
    cam2_only_count = 0
    
    # Track unique objects
    all_objects = set()
    matched_objects = set()
    cam1_only_objects = set()
    cam2_only_objects = set()
    
    for frame_matches in matches:
        for obj_name, (idx1, idx2) in frame_matches.items():
            all_objects.add(obj_name)
            if idx1 is not None and idx2 is not None:
                matched_count += 1
                matched_objects.add(obj_name)
            elif idx1 is not None:
                cam1_only_count += 1
                cam1_only_objects.add(obj_name)
            else:
                cam2_only_count += 1
                cam2_only_objects.add(obj_name)
    
    return {
        'total_frames': len(matches),
        'total_detections_cam1': total_det1,
        'total_detections_cam2': total_det2,
        'matched_detections': matched_count,
        'cam1_only_detections': cam1_only_count,
        'cam2_only_detections': cam2_only_count,
        'unique_objects': len(all_objects),
        'matched_objects': matched_objects,
        'cam1_only_objects': cam1_only_objects,
        'cam2_only_objects': cam2_only_objects
    }


def print_matching_statistics(stats: Dict[str, Dict]) -> None:
    """
    Print formatted matching statistics.
    
    Args:
        stats: Dictionary of statistics from compute_matching_statistics
    """
    print("\n=== Object Detection and Matching Statistics ===")
    print(f"{'Object':<15} {'Cam1':<8} {'Cam2':<8} {'Matched':<8} {'Match %':<8}")
    print("-" * 50)
    
    for obj_name, counts in sorted(stats.items()):
        print(f"{obj_name:<25}: {counts}")