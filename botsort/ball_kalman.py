import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolo11s_ft.pt") # Or your custom model path

video_path = '../data/videos/out2.mp4'
output_path = "../data/output/out2_yolo_kalman_ball.mp4"

cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Kalman Filter Initialization
# State: [x_center, y_center, x_velocity, y_velocity] - 4 state variables
# Measurement: [x_center_detected, y_center_detected] - 2 measurement variables
kalman = cv2.KalmanFilter(4, 2)

# State Transition Matrix (F) - Constant Velocity Model
# x_k = F * x_{k-1}
# x' = x + dt*vx
# y' = y + dt*vy
# vx' = vx
# vy' = vy
dt = 1.0 / fps # Time interval between frames
kalman.transitionMatrix = np.array([[1, 0, dt, 0],
                                     [0, 1, 0, dt],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)

# Measurement Matrix (H) - We only measure position
# z_k = H * x_k
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], dtype=np.float32)

# Process Noise Covariance (Q) - Uncertainty in the model
# Higher values indicate that the model's prediction is less certain (e.g., ball can change direction abruptly)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32) * 0.05 # Adjust this value

kalman.processNoiseCov[2, 2] *= 64
kalman.processNoiseCov[3, 3] *= 64
kalman.processNoiseCov *= 0.05  # Scale factor remains

# Measurement Noise Covariance (R) - Uncertainty in the measurement (YOLO detection)
# Higher values indicate that the YOLO detections are less trusted
kalman.measurementNoiseCov = np.array([[1, 0],
                                        [0, 1]], dtype=np.float32) * 0.3 # Adjust this value

# Initial State Covariance (P)
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1 # Initial uncertainty

# Initial State (x_0) - To be initialized when the ball is first detected
initial_state_set = False
ball_kalman_pos = None # To store the (x,y) from Kalman filter

# Assume a constant ball size for drawing (as per your assumption)
BALL_RADIUS = 40 # Example radius, adjust as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    yolo_results = model.track(frame, persist=True, line_width=1, tracker="botsort.yaml", verbose=False)
    annotated_frame = yolo_results[0].plot()

    ball_detected_this_frame = False
    ball_center_measurement = None

    if yolo_results[0].boxes is not None and len(yolo_results[0].boxes.xyxy) > 0:
        for i, cls_id in enumerate(yolo_results[0].boxes.cls):
            if int(cls_id) == 0: # Ball class ID
                ball_detected_this_frame = True
                box = yolo_results[0].boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                ball_center_x = (x1 + x2) / 2
                ball_center_y = (y1 + y2) / 2
                ball_center_measurement = np.array([[ball_center_x], [ball_center_y]], dtype=np.float32)
                break

    # 1. Kalman Filter Prediction
    if initial_state_set:
        predicted_state = kalman.predict() # This updates kalman.statePre

        # Extract predicted components for boundary check
        px, py, pvx, pvy = predicted_state[0,0], predicted_state[1,0], predicted_state[2,0], predicted_state[3,0]
        
        bounce_applied = False

        # Boundary checks and velocity inversion ("bounce" logic)
        # Horizontal bounce
        if px < 0:
            px = 0 # Clamp to boundary
            pvx = -pvx * 0.8 # Invert velocity, optionally add damping factor (e.g., 0.8 for loss of energy)
            bounce_applied = True
        elif px >= width:
            px = float(width - 1) # Clamp to boundary
            pvx = -pvx * 0.8
            bounce_applied = True

        # Vertical bounce (especially if ball goes above the frame and is expected to come down)
        if py < 0: # Top boundary
            py = 0 # Clamp to boundary
            pvy = -pvy * 0.8 # Invert velocity, optionally add damping factor
            bounce_applied = True
        elif py >= height: # Bottom boundary (less common for a "return" unless it hits the floor)
            py = float(height - 1) # Clamp to boundary
            # Decide if bottom bounce makes sense for your scenario.
            # If it's a high shot going out and not expected to hit the floor in view,
            # you might not want to invert velocity here, or handle it differently.
            # For now, we'll apply a bounce.
            pvy = -pvy * 0.8
            bounce_applied = True
        
        if bounce_applied:
            # If a bounce occurred, update the Kalman filter's predicted state (statePre)
            # before correction or use as the final estimate.
            kalman.statePre[0,0] = px
            kalman.statePre[1,0] = py
            kalman.statePre[2,0] = pvx
            kalman.statePre[3,0] = pvy
        
        # The ball_kalman_pos will be based on this (potentially modified) predicted state
        ball_kalman_pos = (int(kalman.statePre[0,0]), int(kalman.statePre[1,0]))


    # 2. Kalman Filter Update (Correction)
    if ball_detected_this_frame:
        if not initial_state_set:
            kalman.statePost = np.array([ball_center_measurement[0,0],
                                         ball_center_measurement[1,0],
                                         0, 0], dtype=np.float32).reshape(4, 1)
            # Reset error covariance for a new track or after a long gap
            kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1
            initial_state_set = True
            ball_kalman_pos = (int(ball_center_measurement[0,0]), int(ball_center_measurement[1,0]))
            print(f"Kalman filter initialized for ball at: {ball_kalman_pos}")
        else:
            # Correct the Kalman filter state with the new measurement
            # The .correct() method uses kalman.statePre (which we might have modified with bounce logic)
            corrected_state = kalman.correct(ball_center_measurement)
            ball_kalman_pos = (int(corrected_state[0,0]), int(corrected_state[1,0]))
    elif initial_state_set:
        # If not detected, ball_kalman_pos remains the (potentially bounced) predicted position
        # from kalman.statePre which was set earlier.
        pass


    # 3. Visualization
    if ball_kalman_pos:
        cv2.circle(annotated_frame, ball_kalman_pos, BALL_RADIUS, (0, 255, 0), 2) # Green: Kalman
        cv2.putText(annotated_frame, "Ball (K)", (ball_kalman_pos[0] + 15, ball_kalman_pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if ball_detected_this_frame and ball_center_measurement is not None:
         raw_detection_pos = (int(ball_center_measurement[0,0]), int(ball_center_measurement[1,0]))
         cv2.circle(annotated_frame, raw_detection_pos, BALL_RADIUS, (255, 0, 0), 1) # Blue: YOLO

    out.write(annotated_frame)
    # cv2.imshow("Frame", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
# cv2.destroyAllWindows()

print(f"Processed video with bounce logic saved to: {output_path}")
