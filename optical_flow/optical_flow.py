"""
Optical flow tracker implementation.
Uses Lucas-Kanade optical flow with prediction for out-of-view objects.
"""

import cv2
import numpy as np
from config import OpticalFlowConfig


class OpticalFlowTracker:
    """Optical flow tracker with prediction for out-of-view objects."""
    
    def __init__(self):
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=OpticalFlowConfig.WINDOW_SIZE,
            maxLevel=OpticalFlowConfig.MAX_LEVEL,
            criteria=OpticalFlowConfig.CRITERIA
        )
        
        self.prev_gray = None
        self.tracking_points = None
        self.last_known_velocity = np.array([0.0, 0.0])
        
    def initialize(self, frame, center_x, center_y, roi_size=OpticalFlowConfig.ROI_SIZE):
        """
        Initialize optical flow tracking around a detection.
        
        Args:
            frame: Input frame (BGR)
            center_x, center_y: Center coordinates for initialization
            roi_size: Size of region of interest for feature detection
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Define ROI around the detection
        x1 = max(0, int(center_x - roi_size/2))
        y1 = max(0, int(center_y - roi_size/2))
        x2 = min(frame.shape[1], int(center_x + roi_size/2))
        y2 = min(frame.shape[0], int(center_y + roi_size/2))
        
        roi = gray[y1:y2, x1:x2]
        corners = cv2.goodFeaturesToTrack(
            roi, 
            maxCorners=OpticalFlowConfig.MAX_CORNERS,
            qualityLevel=OpticalFlowConfig.QUALITY_LEVEL,
            minDistance=OpticalFlowConfig.MIN_DISTANCE
        )
        
        if corners is not None and len(corners) > 0:
            # Adjust coordinates to global frame
            corners[:, :, 0] += x1
            corners[:, :, 1] += y1
            self.tracking_points = corners
        else:
            # Fallback to center point if no features found
            self.tracking_points = np.array([[[center_x, center_y]]], dtype=np.float32)
            
        self.prev_gray = gray
        self.last_known_velocity = np.array([0.0, 0.0])
        
    def track(self, frame):
        """
        Track points using optical flow.
        If points are lost (e.g., ball out of frame), predict the new position
        based on the last known velocity.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            tuple: Tracked position (x, y) or None if tracking failed
        """
        if self.prev_gray is None or self.tracking_points is None or len(self.tracking_points) == 0:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.tracking_points, None, **self.lk_params
        )
        
        if next_points is not None:
            good_new_points = next_points[status == 1]
            good_old_points = self.tracking_points[status == 1]
        else:
            good_new_points = np.array([])
            good_old_points = np.array([])

        # Check for sufficient number of tracked points
        if len(good_new_points) > OpticalFlowConfig.MIN_TRACKING_POINTS:
            # Successful track
            self._update_successful_track(good_new_points, good_old_points, gray)
            
            # Calculate centroid of tracked points
            center_x = np.mean(good_new_points[:, 0])
            center_y = np.mean(good_new_points[:, 1])
            
            return (center_x, center_y)
        else:
            # Track lost - predict using last known velocity
            return self._predict_lost_track(gray)
            
    def _update_successful_track(self, good_new_points, good_old_points, gray):
        """Update tracker state after successful tracking."""
        # Calculate and store the current velocity
        self.last_known_velocity = np.mean(good_new_points - good_old_points, axis=0)
        
        # Update tracking points for the next frame
        self.tracking_points = good_new_points.reshape(-1, 1, 2)
        self.prev_gray = gray
        
    def _predict_lost_track(self, gray):
        """Predict position when tracking is lost."""
        if self.tracking_points is None:
            return None
            
        # Get the centroid of the last known point cloud
        last_centroid = np.mean(self.tracking_points, axis=0).flatten()
        
        # Apply damping to simulate friction/air resistance
        self.last_known_velocity *= OpticalFlowConfig.VELOCITY_DAMPING
        
        # Predict the new position
        predicted_position = last_centroid + self.last_known_velocity
        
        # Update the tracking points to the predicted location for the next frame
        self.tracking_points = (self.tracking_points.reshape(-1, 2) + 
                               self.last_known_velocity).reshape(-1, 1, 2)
        
        self.prev_gray = gray
        
        # Return the predicted position (may be outside the frame)
        return (predicted_position[0], predicted_position[1])
        
    def reset(self):
        """Reset the tracker to uninitialized state."""
        self.prev_gray = None
        self.tracking_points = None
        self.last_known_velocity = np.array([0.0, 0.0])
        
    def is_initialized(self):
        """Check if tracker is initialized."""
        return self.prev_gray is not None and self.tracking_points is not None
        
    def get_velocity(self):
        """Get the last known velocity."""
        return self.last_known_velocity.copy()