import numpy as np
import cv2

#init (first_frame, tuple(abs_det))
#call ()
#update (frame)

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox, class_id):
        """
        Initialize a tracker using initial bounding box
        bbox is [x, y, w, h] in absolute coordinates
        """
        # Define constant velocity model
        self.kf = cv2.KalmanFilter(9, 4)
        # Measurement matrix relates state to measurement [x, y, w, h]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0]
        ], np.float32)

        # State transition matrix with acceleration components
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0, 0.5], # x with velocity and acceleration
            [0, 1, 0, 0, 0, 1, 0, 0, 0.5], # y with velocity and acceleration
            [0, 0, 1, 0, 0, 0, 1, 0, 0],   # width with velocity
            [0, 0, 0, 1, 0, 0, 0, 1, 0],   # height with velocity
            [0, 0, 0, 0, 1, 0, 0, 0, 1],   # vx with acceleration
            [0, 0, 0, 0, 0, 1, 0, 0, 1],   # vy with acceleration
            [0, 0, 0, 0, 0, 0, 1, 0, 0],   # vw
            [0, 0, 0, 0, 0, 0, 0, 1, 0],   # vh
            [0, 0, 0, 0, 0, 0, 0, 0, 1]    # acceleration component
        ], np.float32)
        
        
        self.kf.processNoiseCov = np.eye(9, dtype=np.float32) * 0.01
        self.kf.processNoiseCov[4:,4:] *= 10  # Higher noise for velocity
        
        # Measurement noise - tune this based on detection quality
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 2
    
        
        # Convert [x, y, w, h] to [x, y, s, r] (center x, center y, area, aspect ratio)
        x, y, w, h = bbox
        center_x = x + w/2
        center_y = y + h/2
        self.kf.statePost = np.array([center_x, center_y, w, h, 0, 0, 0, 0, 0], np.float32)
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.class_id = class_id
        self.time_since_update = 0
        
        
        
        self.maxAge = 25
        
    def update(self, bbox):
        """
        Updates the state vector with observed bbox
        """
        self.time_since_update = 0
        
        # Convert [x, y, w, h] to [x, y, s, r]
        x, y, w, h = bbox
        center_x = x + w/2
        center_y = y + h/2
        
        prev_x,prev_y = self.kf.statePost[0],self.kf.statePost[1]
        dx = abs(center_x - prev_x)
        dy = abs(center_y - prev_y)
        speed = np.sqrt(dx*dx + dy*dy)
            # Adjust process noise based on speed
        if speed > 30:  # Fast movement
             self.kf.processNoiseCov[4:6,4:6] *= 4  # Increase velocity component noise
        else:
             self.kf.processNoiseCov[4:6,4:6] = np.eye(2, dtype=np.float32) * 0.25
        
        measurement = np.array([center_x, center_y, w, h], np.float32)
        self.kf.correct(measurement)
        
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        self.kf.predict()
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        
        
        # Get predicted state
        state = self.kf.statePost
        center_x, center_y, w, h = state[0:4]
        
        # Convert back to [x, y, w, h] format
        x = center_x - w/2
        y = center_y - h/2
        
        return [int(x), int(y), int(w), int(h)]
        
    def get_state(self):
        """
        Returns the current bounding box estimate
        """
        state = self.kf.statePost
        center_x, center_y, w, h = state[0:4]
        
        # Convert back to [x, y, w, h] format
        x = center_x - w/2
        y = center_y - h/2
        
        return [int(x), int(y), int(w), int(h)]