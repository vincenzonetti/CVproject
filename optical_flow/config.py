"""
Configuration file for ball tracking system.
Contains all constants and configuration parameters.
"""
import cv2
# Video processing constants
DEFAULT_FPS = 30
VIDEO_CODEC = 'mp4v'

# Ball detection constants
BALL_CLASS_ID = 0
DEFAULT_BALL_SIZE = (30, 30)
BRIGHTNESS_THRESHOLD = 142  # Pixel intensity threshold for false positive filtering

# Kalman Filter parameters
class KalmanConfig:
    INITIAL_COVARIANCE = 100
    PROCESS_NOISE_SCALE = 1.0
    MEASUREMENT_NOISE_SCALE = 10.0
    GRAVITY = 0.5
    MAX_VELOCITY = 50  # pixels per frame
    MAX_ACCELERATION = 10
    ACCELERATION_DAMPING = 0.8

# Optical Flow parameters
class OpticalFlowConfig:
    WINDOW_SIZE = (21, 21)
    MAX_LEVEL = 3
    CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    ROI_SIZE = 30
    MAX_CORNERS = 20
    QUALITY_LEVEL = 0.01
    MIN_DISTANCE = 5
    VELOCITY_DAMPING = 0.98
    MIN_TRACKING_POINTS = 5

# Tracking parameters
class TrackingConfig:
    MAX_HYPOTHESES = 5
    HYPOTHESIS_MAX_AGE = 30
    DETECTION_DISTANCE_THRESHOLD = 50
    CONFIDENCE_INCREMENT = 0.1
    CONFIDENCE_DECREMENT = 0.05
    MIN_HYPOTHESIS_CONFIDENCE = 0.1
    
    # Enhanced tracker parameters
    INITIAL_CONFIDENCE_THRESHOLD = 0.3
    MIN_ADAPTIVE_THRESHOLD = 0.1
    ADAPTIVE_THRESHOLD_DECAY = 0.02
    HISTOGRAM_DISTANCE_THRESHOLD = 1.5
    HISTOGRAM_DISTANCE_THRESHOLD_RELAXED = 2.5
    
    # Search parameters
    BASE_SEARCH_RADIUS = 100
    SEARCH_RADIUS_MULTIPLIER = 2
    SEARCH_STEP_DIVISOR = 3
    
    # Scales for multi-scale search
    NORMAL_SCALES = [0.8, 1.0, 1.2]
    EXTENDED_SCALES = [0.6, 0.8, 1.0, 1.2, 1.4]

# Histogram parameters
class HistogramConfig:
    BINS = 256
    RANGE = [0, 256]
    TRIM_OUTLIERS = 10  # Remove top and bottom N values
    GAUSSIAN_SIGMA = 2.0
    CHI_SQUARE_EPSILON = 1e-10

# Output parameters
class OutputConfig:
    OUTPUT_DIR = "outputs"
    RESULTS_FILENAME = "enhanced_tracking_results.json"
    VIDEO_FILENAME = "enhanced_tracked_video.mp4"
    HISTOGRAM_FILENAME = "out13_histograms.json"
    
    # Visualization colors (BGR format for OpenCV)
    COLORS = {
        'yolo': (0, 255, 0),        # Green
        'kalman': (255, 0, 0),      # Blue
        'optical_flow': (0, 255, 255),  # Yellow
        'histogram': (255, 255, 0), # Cyan
        'interpolation': (255, 0, 255),  # Magenta
        'default': (128, 128, 128)  # Gray
    }
    
    # Display parameters
    DISPLAY_WIDTH = 1080
    DISPLAY_HEIGHT = 720
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    BOX_THICKNESS = 2