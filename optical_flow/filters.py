"""
Kalman filter implementation for ball tracking.
Enhanced Kalman filter with physics constraints for improved tracking accuracy.
"""

import numpy as np
from config import KalmanConfig


class KalmanFilter:
    """Enhanced Kalman filter for ball tracking with physics constraints"""
    
    def __init__(self):
        # State: [x, y, vx, vy, ax, ay] - position, velocity, acceleration
        self.state = np.zeros(6)
        self.P = np.eye(6) * KalmanConfig.INITIAL_COVARIANCE  # Covariance matrix
        self.Q = np.eye(6) * KalmanConfig.PROCESS_NOISE_SCALE  # Process noise
        self.R = np.eye(2) * KalmanConfig.MEASUREMENT_NOISE_SCALE  # Measurement noise
        
        # Transition matrix (constant acceleration model)
        self.F = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, KalmanConfig.ACCELERATION_DAMPING, 0],  # Damping on acceleration
            [0, 0, 0, 0, 0, KalmanConfig.ACCELERATION_DAMPING]
        ])
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        self.initialized = False
        
    def initialize(self, x, y):
        """
        Initialize filter with first detection.
        
        Args:
            x, y: Initial position coordinates
        """
        self.state = np.array([x, y, 0, 0, 0, KalmanConfig.GRAVITY])
        self.P = np.eye(6) * KalmanConfig.INITIAL_COVARIANCE
        self.initialized = True
        
    def predict(self):
        """
        Predict next state.
        
        Returns:
            tuple: Predicted position (x, y) or None if not initialized
        """
        if not self.initialized:
            return None
            
        # Apply physics (gravity)
        self.state[5] = KalmanConfig.GRAVITY
        
        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Apply velocity constraints
        self.state[2] = np.clip(self.state[2], -KalmanConfig.MAX_VELOCITY, KalmanConfig.MAX_VELOCITY)
        self.state[3] = np.clip(self.state[3], -KalmanConfig.MAX_VELOCITY, KalmanConfig.MAX_VELOCITY)
        
        # Apply acceleration constraints
        self.state[4] = np.clip(self.state[4], -KalmanConfig.MAX_ACCELERATION, KalmanConfig.MAX_ACCELERATION)
        self.state[5] = np.clip(self.state[5], -KalmanConfig.MAX_ACCELERATION, 
                               KalmanConfig.MAX_ACCELERATION + KalmanConfig.GRAVITY)
        
        return self.get_position()
        
    def update(self, measurement):
        """
        Update with measurement.
        
        Args:
            measurement: Observed position [x, y]
        """
        if not self.initialized:
            return
            
        z = np.array([measurement[0], measurement[1]])
        
        # Innovation
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
    def get_position(self):
        """
        Get current position estimate.
        
        Returns:
            tuple: Current position (x, y) or None if not initialized
        """
        if not self.initialized:
            return None
        return (self.state[0], self.state[1])
        
    def get_velocity(self):
        """
        Get current velocity estimate.
        
        Returns:
            tuple: Current velocity (vx, vy) or None if not initialized
        """
        if not self.initialized:
            return None
        return (self.state[2], self.state[3])
        
    def get_acceleration(self):
        """
        Get current acceleration estimate.
        
        Returns:
            tuple: Current acceleration (ax, ay) or None if not initialized
        """
        if not self.initialized:
            return None
        return (self.state[4], self.state[5])
        
    def reset(self):
        """Reset the filter to uninitialized state."""
        self.state = np.zeros(6)
        self.P = np.eye(6) * KalmanConfig.INITIAL_COVARIANCE
        self.initialized = False