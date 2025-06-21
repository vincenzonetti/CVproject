"""
Object matching utilities for stereo tracking.

This module handles matching detected objects between two camera views
based on their class identities.
"""

from typing import Dict, Tuple, List


def match_objects(det1: Dict[str, Dict], det2: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Match objects between two camera views based on class names.
    
    Args:
        det1: Detections from camera 1 {class_name: {'center': [x, y], 'bbox': [...]}}
        det2: Detections from camera 2 {class_name: {'center': [x, y], 'bbox': [...]}}
        
    Returns:
        Dictionary of matches {class_name: {'pt1': [x1, y1], 'pt2': [x2, y2]}}
    """
    matches = {}
    
    for obj_name in det1:
        if obj_name in det2:
            matches[obj_name] = {
                'pt1': det1[obj_name]['center'],
                'pt2': det2[obj_name]['center']
            }
    
    return matches


def compute_matching_statistics(detections1: List[Dict], detections2: List[Dict]) -> Dict[str, Dict]:
    """
    Compute detection and matching statistics for all objects across all frames.
    
    Args:
        detections1: List of detections from camera 1 for all frames
        detections2: List of detections from camera 2 for all frames
        
    Returns:
        Dictionary with statistics for each object:
        {object_name: {'cam1': count, 'cam2': count, 'matched': count}}
    """
    stats = {}
    
    for det1, det2 in zip(detections1, detections2):
        # Count appearances in each camera
        for obj_name in det1:
            if obj_name not in stats:
                stats[obj_name] = {'cam1': 0, 'cam2': 0, 'matched': 0}
            stats[obj_name]['cam1'] += 1
            
        for obj_name in det2:
            if obj_name not in stats:
                stats[obj_name] = {'cam1': 0, 'cam2': 0, 'matched': 0}
            stats[obj_name]['cam2'] += 1
        
        # Count matches
        matches = match_objects(det1, det2)
        for obj_name in matches:
            stats[obj_name]['matched'] += 1
    
    return stats


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
        cam1_count = counts['cam1']
        cam2_count = counts['cam2']
        matched = counts['matched']
        max_possible = min(cam1_count, cam2_count)
        match_percentage = (matched / max_possible * 100) if max_possible > 0 else 0
        print(f"{obj_name:<15} {cam1_count:<8} {cam2_count:<8} {matched:<8} {match_percentage:<8.1f}%")