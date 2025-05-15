from filterpy.kalman import KalmanFilter
import numpy as np

from filterpy.kalman import KalmanFilter
import numpy as np

def init_ball_kf(dt=1.0):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]])

    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    kf.P *= 10.0       # Initial uncertainty
    kf.R = np.eye(2) * 5  # YOLO noise
    kf.Q = np.eye(4) * 0.01  # Process noise

    return kf

