import argparse
import os
import json
import cv2
from ultralytics import YOLO
import numpy as np
import typing
from tqdm import tqdm
import time
import numpy as np
import cv2
import time

class KalmanTracker():
    def __init__(self, bbox, cls_id, img_w, img_h, fps=25):
        """
        Initializes the tracker with physical constraints for basketball tracking.
        
        Args:
            bbox (tuple): Initial bounding box (x_norm, y_norm, w_norm, h_norm).
            cls_id (int): Class ID of the object (0 for the ball).
            img_w (int): Width of the image frame.
            img_h (int): Height of the image frame.
            court_bbox (tuple): Bounding box of the court (x_min, y_min, x_max, y_max) in pixels.
                                This defines the valid area for the ball.
            fps (int): Frames per second of the video source.
        """
        self.kf = cv2.KalmanFilter(6, 2) # State: [x, y, vx, vy, ax, ay]
        
        # 1. State Initialization
        x, y, w, h = bbox
        center_x = (x + w / 2) * img_w
        center_y = (y + h / 2) * img_h
        
        self.kf.statePost = np.array([center_x, center_y, 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePre = self.kf.statePost.copy()
        
        # 2. State-to-Measurement Mapping (H)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        
        # 3. Transition Matrix (F) - will be updated dynamically
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        
        # 4. Process Noise Covariance (Q)
        # Noise is higher for acceleration, as it's the most unpredictable element.
        # Acceleration in x (player action) is less predictable than in y (gravity).
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov[0:2, 0:2] *= 0.1  # Low noise on position
        self.kf.processNoiseCov[2:4, 2:4] *= 1.0  # Medium noise on velocity
        self.kf.processNoiseCov[4, 4] *= 10.0      # High noise on x-acceleration
        self.kf.processNoiseCov[5, 5] *= 5.0       # Medium-high noise on y-acceleration
        
        # 5. Measurement Noise Covariance (R)
        # Represents the inaccuracy of the YOLO detector.
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
        
        # 6. Error Covariance (P)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 100
        
        # Tracker metadata
        self.last_update_frame = 0
        self.id = cls_id
        self.fps = fps

        # --- Basketball-Specific Physics Parameters ---
        self.is_ball = (cls_id == 0)
        if self.is_ball:
            self.court_x_min, self.court_y_min, self.court_x_max, self.court_y_max = [0.1*img_w,0.1*img_h,0.8*img_w, 0.7*img_h]
            self.ceiling_y = img_h * 0.7  # Max ball height constraint
            self.gravity_accel = 9.8 * 40  # Approx. g in pixels/s^2. TUNE THIS VALUE.
            self.bounce_damping = 0.75     # Energy retention after a bounce.
        
    def _update_transition_matrix(self, dt):
        """Updates the transition matrix based on the time delta 'dt'."""
        F = np.eye(6, dtype=np.float32)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 4] = dt
        F[3, 5] = dt
        F[0, 4] = 0.5 * dt**2
        F[1, 5] = 0.5 * dt**2
        self.kf.transitionMatrix = F
        
    def _apply_physics_constraints(self, predicted_state):
        """
        Applies physical laws of motion to the predicted state of the basketball.
        This is the core improvement.
        """
        if not self.is_ball:
            return predicted_state
            
        x, y, vx, vy, ax, ay = predicted_state.flatten()
        
        # 1. Apply Gravity: Enforce a constant downward acceleration.
        # This overwrites the filter's estimation of 'ay', making it a known input.
        ay = self.gravity_accel
        
        # 2. Ground Bounce Logic
        if y >= self.court_y_max and vy > 0:
            y = self.court_y_max  # Clamp position to the ground
            vy = -vy * self.bounce_damping  # Reverse velocity and apply damping
            vx *= 0.95  # Apply friction to horizontal velocity
            
        # 3. Court Boundary Clamping
        # Prevents the ball from leaving the defined court area.
        x = np.clip(x, self.court_x_min, self.court_x_max)
        y = np.clip(y, self.ceiling_y, self.court_y_max) # Enforces ceiling and floor
        
        # If ball hits horizontal boundaries, reverse x-velocity
        if (x <= self.court_x_min and vx < 0) or (x >= self.court_x_max and vx > 0):
            vx = -vx * self.bounce_damping
            
        return np.array([x, y, vx, vy, ax, ay], dtype=np.float32)
        
    def predict(self, frame_idx):
        """Predicts the next state of the object."""
        dt = (frame_idx - self.last_update_frame) / self.fps
        if dt < 0: dt = 0 # Ensure non-negative time delta

        self._update_transition_matrix(dt)
        
        # Get standard Kalman prediction
        prediction = self.kf.predict()
        
        # --- Apply Physical Constraints for the Ball ---
        if self.is_ball:
            constrained_state = self._apply_physics_constraints(prediction)
            self.kf.statePre = constrained_state # Overwrite the prediction with the constrained one
            
        return self.kf.statePre[:2] # Return predicted [x, y]
        
    def update(self, bbox, frame_idx):
        """Updates the filter with a new measurement."""
        x, y, w, h = bbox
        
        center_x = (x + w / 2)
        center_y = (y + h / 2)
        measurement = np.array([center_x, center_y], dtype=np.float32)
        
        self.kf.correct(measurement)
        self.last_update_frame = frame_idx

def run_tracker(model_path: str, video_path: str):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Extract video name for output and frame keys
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_classid")
    os.makedirs(output_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, "kalman_tracking_results.json")
    video_output_path = os.path.join(output_dir, "kalman_tracked_video.mp4")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    tracking_results = {}
    frame_idx = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trackers:dict[int,KalmanTracker] = {}
    boxes_dict:dict[int,(float,float)] = {}
    pbar = tqdm(total=length)
    
    while cap.isOpened():
        pbar.update(1)
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame,verbose=False)[0]
        detections = []

        if results.boxes is not None:
            boxes = results.boxes
            best_by_class = {}

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x_center, y_center, w, h = boxes.xywh[i].tolist()
                img_h, img_w = frame.shape[:2]
                
                if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                    
                    best_by_class[cls_id] = {
                        'class_id': cls_id,
                        'track_id': cls_id,  # Same as class ID
                        'bbox': [
                            x_center / img_w,
                            y_center / img_h,
                            w / img_w,
                            h / img_h
                        ],
                        'conf': conf
                    }

            for k,det in best_by_class.items():
                if k not in trackers:
                    trackers[k] = KalmanTracker(det['bbox'], cls_id,width,height)
                else:
                    trackers[k].predict(frame_idx)
                    pixel_bbox = np.array([
                        det['bbox'][0] * img_w,
                        det['bbox'][1] * img_h, 
                        det['bbox'][2] * img_w,
                        det['bbox'][3] * img_h
                    ], dtype=np.float32)
                    trackers[k].update(bbox=pixel_bbox,frame_idx=frame_idx)
                boxes_dict[k] = (det['bbox'][2] * img_w,det['bbox'][3] * img_h)
                
                detections.append(det)

                # Draw box on frame
                xc, yc, bw, bh = det['bbox']
                x1 = int((xc - bw / 2) * img_w)
                y1 = int((yc - bh / 2) * img_h)
                x2 = int((xc + bw / 2) * img_w)
                y2 = int((yc + bh / 2) * img_h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {det['track_id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            undetected_boxes = set(trackers.keys()).difference(set(best_by_class.keys()))
            
            for k in undetected_boxes:
                xc,yc = trackers[k].predict(frame_idx)
                bw,bh = boxes_dict[k]
                
                x1 = int((xc - bw / 2))
                y1 = int((yc - bh / 2))
                x2 = int((xc + bw / 2))
                y2 = int((yc + bh / 2))
                
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID {det['track_id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                

        frame_key = f"{video_name}_{frame_idx}"
        tracking_results[frame_key] = detections
        frame_idx += 1

        out_video.write(frame)

    cap.release()
    out_video.release()
    pbar.close()
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f)

    print(f"Tracking complete. Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO class-ID-based tracking on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    args = parser.parse_args()

    run_tracker(args.model, args.video)