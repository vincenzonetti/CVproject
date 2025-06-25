"""
Configuration and constants for basketball tracking system.

This module contains all global constants, configuration parameters,
and class definitions used throughout the tracking system.
"""

# Image dimensions
IMG_WIDTH = 3840
IMG_HEIGHT = 2160

# Class names mapping
CLASSES = [
    'Ball', 'Red_0', 'Red_11', 'Red_12', 'Red_16', 'Red_2', 
    'Refree_F', 'Refree_M', 'White_13', 'White_16', 'White_25', 
    'White_27', 'White_34'
]

# Default parameters
DEFAULT_FPS = 25

# Plotly colors for interactive visualization
PLOTLY_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
    '#00FFFF', '#FFA500', '#800080', '#A52A2A', '#000000',
    '#808080', '#008000', '#FFC0CB'
]

# Basketball court dimensions (standard)
STANDARD_COURT_WIDTH_M = 28.0  # meters
STANDARD_COURT_LENGTH_M = 15.0  # meters

# Tracking thresholds
DIRECTION_CHANGE_THRESHOLD_RAD = 0.785398  # 45 degrees in radians
MIN_TRAJECTORY_POINTS = 2  # Minimum points needed for trajectory

# Height constraints
MAX_REASONABLE_HEIGHT_M = 4.5  # Maximum reasonable height in meters
HEIGHT_SCALE_FACTOR = 0.001  # Convert mm to m if needed

# Axis convention for basketball court
# X = Width (side to side)
# Y = Height (vertical, 0 for ground)
# Z = Depth (baseline to baseline)
AXIS_LABELS = {
    'x': 'X - Width (meters)',
    'y': 'Y - Height (meters)', 
    'z': 'Z - Depth (meters)'
}

# Camera configuration
CAMERA_PARAMS_KEYS = ['mtx', 'dist', 'rvecs', 'tvecs']

# Output directory
OUTPUT_DIR = 'outputs/stereo'

# File naming conventions
TRACKING_2D_FILENAME = 'tracking_2d_results.json'
TRACKING_3D_FILENAME = 'tracking_3d_results.json'
METRICS_FILENAME = 'trajectory_metrics.json'
INTERACTIVE_PLOT_FILENAME = 'interactive_3d_trajectories.html'
METRICS_DASHBOARD_FILENAME = 'interactive_3d_trajectories_metrics.html'
METRICS_CSV_FILENAME = 'interactive_3d_trajectories_metrics.csv'


# In 3Dreconstruction/config.py

# Add these constants for ball height scaling
BALL_MIN_HEIGHT_M = 0.0      # Minimum ball height in meters
BALL_MAX_HEIGHT_M = 4.5      # Maximum ball height in meters  
BALL_TARGET_AVG_HEIGHT_M = 1.3  # Target average height for ball in meters
BALL_FIRST_DETECTION_HEIGHT_M = 1.1