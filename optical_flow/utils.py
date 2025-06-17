"""
Utility functions for ball tracking system.
Contains common helper functions used across modules.
"""

import cv2
import numpy as np
import json
import os
from scipy.ndimage import gaussian_filter1d
from config import HistogramConfig


def calculate_histogram(image, smooth=True):
    """
    Calculate histogram for a single channel image.
    
    Args:
        image: Single channel image (grayscale)
        smooth: Whether to apply gaussian smoothing
        
    Returns:
        numpy.ndarray: Processed histogram
    """
    hist = cv2.calcHist([image], [0], None, [HistogramConfig.BINS], HistogramConfig.RANGE)
    
    # Remove outliers by trimming
    sorted_hist = np.sort(hist)
    trim = HistogramConfig.TRIM_OUTLIERS
    trimmed_hist = sorted_hist[trim:-trim] if trim > 0 else sorted_hist
    
    # Apply gaussian smoothing
    if smooth:
        hist_smooth = gaussian_filter1d(trimmed_hist, sigma=HistogramConfig.GAUSSIAN_SIGMA)
        return hist_smooth
    
    return trimmed_hist


def calculate_chi_square_distance(hist1, hist2):
    """
    Calculate Chi-square distance between two histograms.
    
    Args:
        hist1, hist2: Histograms to compare
        
    Returns:
        float: Chi-square distance
    """
    min_len = min(len(hist1), len(hist2))
    h1 = hist1[:min_len].flatten()
    h2 = hist2[:min_len].flatten()
    
    # Normalize histograms
    h1_norm = h1 / (np.sum(h1) + HistogramConfig.CHI_SQUARE_EPSILON)
    h2_norm = h2 / (np.sum(h2) + HistogramConfig.CHI_SQUARE_EPSILON)
    
    # Calculate Chi-square distance
    chi_square = 0.5 * np.sum((h1_norm - h2_norm) ** 2 / 
                              (h1_norm + h2_norm + HistogramConfig.CHI_SQUARE_EPSILON))
    return chi_square


def load_histograms(file_path):
    """
    Load histograms from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        dict: Loaded histograms or empty dict if failed
    """
    try:
        with open(file_path, 'r') as json_file:
            histograms = json.load(json_file)
        return histograms
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load histograms from {file_path}: {e}")
        return {}


def save_json(data, file_path):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def normalize_bbox(bbox, img_width, img_height):
    """
    Convert normalized bbox coordinates to pixel coordinates.
    
    Args:
        bbox: [x_center, y_center, width, height] normalized (0-1)
        img_width, img_height: Image dimensions
        
    Returns:
        tuple: (x_center, y_center) in pixel coordinates
    """
    x_center = bbox[0] * img_width
    y_center = bbox[1] * img_height
    return x_center, y_center


def denormalize_bbox(bbox, img_width, img_height):
    """
    Convert pixel bbox coordinates to normalized coordinates.
    
    Args:
        bbox: [x_center, y_center, width, height] in pixels
        img_width, img_height: Image dimensions
        
    Returns:
        list: Normalized bbox coordinates
    """
    return [
        bbox[0] / img_width,
        bbox[1] / img_height,
        bbox[2] / img_width,
        bbox[3] / img_height
    ]


def clip_coordinates(x, y, img_width, img_height):
    """
    Clip coordinates to image boundaries.
    
    Args:
        x, y: Coordinates to clip
        img_width, img_height: Image dimensions
        
    Returns:
        tuple: Clipped coordinates
    """
    x = np.clip(x, 0, img_width)
    y = np.clip(y, 0, img_height)
    return x, y


def get_roi_bounds(center_x, center_y, width, height, img_width, img_height):
    """
    Get region of interest bounds with boundary checking.
    
    Args:
        center_x, center_y: Center coordinates
        width, height: ROI dimensions
        img_width, img_height: Image dimensions
        
    Returns:
        tuple: (x1, y1, x2, y2) bounds
    """
    x1 = max(0, int(center_x - width/2))
    y1 = max(0, int(center_y - height/2))
    x2 = min(img_width, int(center_x + width/2))
    y2 = min(img_height, int(center_y + height/2))
    
    return x1, y1, x2, y2


def euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1, p2: Points as (x, y) tuples
        
    Returns:
        float: Euclidean distance
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def is_ball_too_bright(roi):
    """
    Check if a region is too bright to be a ball (false positive filter).
    
    Args:
        roi: Region of interest (BGR image)
        
    Returns:
        bool: True if too bright
    """
    try:
        blue_channel, green_channel, red_channel = cv2.split(roi)
        hist_r = calculate_histogram(red_channel)
        
        tot_pixel = np.sum(hist_r)
        if tot_pixel > 0:
            pixel_intens = np.arange(len(hist_r))
            weighted_sum = np.sum(pixel_intens * hist_r[:, 0])
            avg_pixel = weighted_sum / tot_pixel
            
            from config import BRIGHTNESS_THRESHOLD
            return avg_pixel > BRIGHTNESS_THRESHOLD
    except Exception:
        pass
    
    return False


def create_output_directory(video_name, model_name):
    """
    Create output directory for results.
    
    Args:
        video_name: Name of the video file
        model_name: Name of the model file
        
    Returns:
        str: Output directory path
    """
    from config import OutputConfig
    
    output_dir = os.path.join(OutputConfig.OUTPUT_DIR, f"{video_name}_{model_name}_enhanced")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir