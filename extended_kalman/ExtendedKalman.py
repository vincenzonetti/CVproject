import numpy as np
import cv2

class ExtendedKalman:
    """
    This class represents the internal state of individual tracked objects observed as bbox
    using an Extended Kalman Filter.
    The state is [cx, cy, w, h, vx, vy, vw, vh, ax, ay], where:
    cx, cy: center coordinates of the bounding box
    w, h: width and height of the bounding box
    vx, vy: velocities of cx and cy
    vw, vh: velocities of w and h (representing rate of change of size)
    ax, ay: accelerations of cx and cy
    """
    count = 0

    def __init__(self, bbox, class_id, dt=1.0/25.0): # Assuming a default FPS, dt should be updated dynamically
        """
        Initialize a tracker using initial bounding box.
        bbox is [x, y, w, h] in absolute coordinates (top-left x, top-left y, width, height)
        class_id is the identifier for the class of the object (e.g., player, ball)
        dt is the initial time step, typically 1/FPS.
        """
        self.id = ExtendedKalman.count
        ExtendedKalman.count += 1
        self.class_id = class_id
        self.time_since_update = 0
        self.hits = 0 # Number of consecutive updates (hits)
        self.hit_streak = 0 # Current hit streak
        self.age = 0
        self.max_age = 25  # Maximum number of frames to keep a track without updates

        # State vector: [cx, cy, w, h, vx, vy, vw, vh, ax, ay] - 10 states
        # Measurement vector: [cx, cy, area, aspect_ratio] - 4 measurements
        self.kf = cv2.KalmanFilter(10, 4)

        # State Transition Matrix (A) - To be updated with dt in predict step
        # Placeholder, will be dynamically set in predict based on dt
        self.kf.transitionMatrix = np.eye(10, dtype=np.float32)
        # We will set dt dependent terms in the predict method. Example for cx:
        # x_k = x_{k-1} + vx_{k-1}*dt + 0.5*ax_{k-1}*dt^2
        # vx_k = vx_{k-1} + ax_{k-1}*dt
        # ax_k = ax_{k-1} (acceleration is assumed constant between steps, driven by process noise)

        # Measurement Matrix (H_k) - This will be the Jacobian, set in update
        # For EKF, self.kf.measurementMatrix is set to the Jacobian H_k during the update step.
        # It's initialized here but effectively overridden.
        self.kf.measurementMatrix = np.zeros((4, 10), np.float32)


        # Process Noise Covariance (Q)
        # Represents the uncertainty in the motion model.
        # Tuned empirically. Higher values for more erratic movements.
        q_pos_acc = 0.5 # Process noise for position due to acceleration (m/s^2)^2 * (s^2)^2 for (0.5*a*dt^2) term
        q_vel_acc = 1.0 # Process noise for velocity due to acceleration (m/s^2)^2 * (s)^2 for (a*dt) term
        q_acc = 2.0     # Process noise for acceleration itself (m/s^2)^2
        q_wh_vel = 0.1 # Process noise for w/h velocity (pixels/s)^2
        q_wh = 0.05    # Process noise for w/h (pixels)^2 (if modeling w/h directly, not just velocity)

        # Assuming dt=1 for initial diagonal values, will scale with dt in a more advanced Q formulation if needed
        # For simplicity, we define general magnitudes here.
        # Variances for cx, cy, w, h, vx, vy, vw, vh, ax, ay
        # These values represent the *variance* of the noise added to each state variable per time step.
        # It's often better to define Q based on physical limits or expectations of random accelerations.
        # A common way is Q = G * G.T * sigma_a^2 where G is noise gain matrix.
        # For now, direct diagonal assignment:
        process_noise_vars = np.array([
            0.1, 0.1, # cx, cy (affected by vx, ax)
            0.5, 0.5, # w, h (affected by vw, vh - perspective, scale changes)
            1.0, 1.0, # vx, vy (affected by ax, ay)
            2.0, 2.0, # vw, vh (velocity of w,h - more uncertain)
            10.0, 10.0 # ax, ay (acceleration can change rapidly for players)
        ], dtype=np.float32) * (dt**2) # Rough scaling with dt^2 for positions, dt for velocities.
                                         # More precise Q would involve integrating noise over dt.
                                         # For simplicity, let's use fixed values and tune.
        self.kf.processNoiseCov = np.diag([
            0.2, 0.2,  # cx, cy variance (pixels^2)
            0.5, 0.5,  # w, h variance (pixels^2)
            1.0, 1.0,  # vx, vy variance ((pixels/s)^2)
            1.0, 1.0,  # vw, vh variance ((pixels/s)^2) - if player size change is somewhat smooth
            5.0, 5.0   # ax, ay variance ((pixels/s^2)^2) - players can accelerate/decelerate quickly
        ]).astype(np.float32)
        # Example: Basketball players can change velocity quickly, so ax, ay process noise should be significant.
        # Size changes (w,h) are more due to perspective or zoom, so vw,vh might be less dynamic than player motion.

        # Measurement Noise Covariance (R)
        # Represents the uncertainty of the detector.
        # [cx, cy, area, ratio]
        # Values depend on the accuracy of your object detector.
        # R_pos: variance of center position detection (pixels^2)
        # R_area: variance of area detection (pixels^4)
        # R_ratio: variance of aspect ratio detection (unitless^2)
        R_pos_stddev = 2.0  # Std dev for x, y in pixels
        R_area_stddev = 20.0 # Std dev for area in pixels^2
        R_ratio_stddev = 0.1 # Std dev for aspect ratio
        self.kf.measurementNoiseCov = np.diag([
            R_pos_stddev**2,
            R_pos_stddev**2,
            R_area_stddev**2,
            R_ratio_stddev**2
        ]).astype(np.float32)

        # Initial Error Covariance (P_post)
        # Uncertainty of the initial state.
        self.kf.errorCovPost = np.eye(10, dtype=np.float32)
        self.kf.errorCovPost[0:2, 0:2] *= (R_pos_stddev*2)**2  # cx, cy initial uncertainty (a bit more than measurement noise)
        self.kf.errorCovPost[2:4, 2:4] *= (5.0)**2    # w, h initial uncertainty (pixels^2)
        self.kf.errorCovPost[4:8, 4:8] *= (100.0)**2 # vx, vy, vw, vh initial uncertainty (high, as velocity is unknown)
        self.kf.errorCovPost[8:10, 8:10] *= (50.0)**2 # ax, ay initial uncertainty (high, as acceleration is unknown)


        # Initial State (x_post)
        # Convert [x_tl, y_tl, w, h] to [cx, cy, w, h, 0, 0, 0, 0, 0, 0]
        x_tl, y_tl, w_init, h_init = bbox
        cx_init = x_tl + w_init / 2.0
        cy_init = y_tl + h_init / 2.0
        self.kf.statePost = np.array([cx_init, cy_init, w_init, h_init,
                                      0, 0, 0, 0, 0, 0], dtype=np.float32)
        # Initialize statePre as well, as predict() might be called before first update in some loops
        self.kf.statePre = self.kf.statePost.copy()


    def _h(self, state_vector):
        """
        Non-linear measurement function h(x).
        Maps state [cx, cy, w, h, ...] to measurement [cx, cy, area, ratio].
        """
        cx, cy, w, h = state_vector[0:4]
        w_safe = max(w, 1e-6) # Prevent division by zero or log(0) issues
        h_safe = max(h, 1e-6)
        area = w_safe * h_safe
        ratio = w_safe / h_safe if h_safe > 1e-6 else 0.0 # Avoid division by zero for ratio
        return np.array([cx, cy, area, ratio], dtype=np.float32)

    def _jacobian_H(self, state_vector):
        """
        Computes the Jacobian of the measurement function h(x) with respect to the state x.
        H = dh/dx.
        State x is [cx, cy, w, h, vx, vy, vw, vh, ax, ay].
        Measurement z is [cx, cy, area, ratio].
        """
        w = state_vector[2]
        h = state_vector[3]

        w_s = max(w, 1e-6) # Small epsilon to avoid division by zero
        h_s = max(h, 1e-6)

        H = np.zeros((4, 10), dtype=np.float32)
        H[0, 0] = 1.0  # d(cx)/d(cx)
        H[1, 1] = 1.0  # d(cy)/d(cy)
        H[2, 2] = h_s  # d(area)/d(w) = h
        H[2, 3] = w_s  # d(area)/d(h) = w
        H[3, 2] = 1.0 / h_s if h_s > 1e-6 else 0.0 # d(ratio)/d(w) = 1/h
        H[3, 3] = -w_s / (h_s**2) if h_s > 1e-6 else 0.0 # d(ratio)/d(h) = -w/h^2
        return H

    def update(self, bbox_detected):
        """
        Updates the state vector with observed bbox.
        bbox_detected is [x_tl, y_tl, w, h]
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak +=1

        # Convert detected [x_tl, y_tl, w, h] to measurement [cx, cy, area, ratio]
        x_tl, y_tl, w, h = bbox_detected
        # Ensure w and h from detection are positive
        w = max(w, 1e-6)
        h = max(h, 1e-6)

        cx = x_tl + w / 2.0
        cy = y_tl + h / 2.0
        area = w * h
        ratio = w / h if h > 1e-6 else 0.0
        z_actual = np.array([cx, cy, area, ratio], dtype=np.float32)

        # Get the a priori state estimate (result of the last predict step)
        state_pred_arr = self.kf.statePre.flatten()

        # Calculate Jacobian H_k using the a priori state estimate
        H_k = self._jacobian_H(state_pred_arr)
        self.kf.measurementMatrix = H_k # Set it for OpenCV's EKF math

        # Calculate predicted measurement h(x_priori)
        z_pred_arr = self._h(state_pred_arr)

        # EKF correction for OpenCV:
        # OpenCV's correct method for a linear KF computes: K * (z_actual - H * x_priori)
        # For EKF, the innovation is (z_actual - h(x_priori)).
        # To use OpenCV's linear KF framework for EKF, we provide an "adjusted" measurement:
        # z_adjusted = z_actual - h(x_priori) + H_k * x_priori
        # So, K * (z_adjusted - H_k * x_priori) = K * (z_actual - h(x_priori))
        measurement_for_cv_ekf = z_actual - z_pred_arr + (H_k @ state_pred_arr.reshape(-1,1)).flatten()

        self.kf.correct(measurement_for_cv_ekf) # Updates statePost and errorCovPost

    def predict(self, dt=1.0/25.0): # dt should be actual time elapsed
        """
        Advances the state vector and returns the predicted bounding box estimate.
        dt: time step (in seconds) since the last prediction.
        Returns: [x_tl, y_tl, w, h] based on the a priori state estimate (statePre).
        """
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0 # Reset hit streak if no update in the last cycle
        self.time_since_update += 1

        # Update Transition Matrix (A) based on dt
        # x_k = x_{k-1} + vx_{k-1}*dt + 0.5*ax_{k-1}*dt^2
        # vx_k = vx_{k-1} + ax_{k-1}*dt
        # ... and similarly for y, w, h (if w, h also have acceleration terms, which they don't here)
        # For w, h: w_k = w_{k-1} + vw_{k-1}*dt
        self.kf.transitionMatrix = np.array([
            [1,0,0,0, dt, 0, 0, 0, 0.5*dt*dt, 0],  # cx
            [0,1,0,0, 0, dt, 0, 0, 0, 0.5*dt*dt],  # cy
            [0,0,1,0, 0, 0, dt, 0, 0, 0],          # w
            [0,0,0,1, 0, 0, 0, dt, 0, 0],          # h
            [0,0,0,0, 1, 0, 0, 0, dt, 0],          # vx
            [0,0,0,0, 0, 1, 0, 0, 0, dt],          # vy
            [0,0,0,0, 0, 0, 1, 0, 0, 0],          # vw
            [0,0,0,0, 0, 0, 0, 1, 0, 0],          # vh
            [0,0,0,0, 0, 0, 0, 0, 1, 0],          # ax
            [0,0,0,0, 0, 0, 0, 0, 0, 1]           # ay
        ], np.float32)


        self.kf.predict() # Updates self.kf.statePre and self.kf.errorCovPre

        state_predicted = self.kf.statePre.flatten()
        cx, cy, w, h = state_predicted[0:4]

        # Ensure predicted w and h are positive and realistic
        w = max(w, 1.0) # Minimum width of 1 pixel
        h = max(h, 1.0) # Minimum height of 1 pixel

        # Convert predicted [cx, cy, w, h] back to [x_tl, y_tl, w, h]
        x_tl = cx - w / 2.0
        y_tl = cy - h / 2.0

        return [x_tl, y_tl, w, h]

    def get_state(self):
        """
        Returns the current bounding box estimate based on the a posteriori state (statePost).
        Format: [x_tl, y_tl, w, h]
        """
        state_current = self.kf.statePost.flatten()
        cx, cy, w, h = state_current[0:4]

        # Ensure w and h are positive
        w = max(w, 1.0)
        h = max(h, 1.0)

        x_tl = cx - w / 2.0
        y_tl = cy - h / 2.0
        return [x_tl, y_tl, w, h]