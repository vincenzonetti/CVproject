import argparse
import os
import json
import cv2
from ultralytics import YOLO
import numpy as np
import typing
from tqdm import tqdm
import time
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis, entropy
from collections import deque


class KalmanFilter:
    """Enhanced Kalman filter for ball tracking with physics constraints"""
    def __init__(self):
        # State: [x, y, vx, vy, ax, ay] - position, velocity, acceleration
        self.state = np.zeros(6)
        self.P = np.eye(6) * 100  # Covariance matrix
        self.Q = np.eye(6)  # Process noise
        self.R = np.eye(2) * 10  # Measurement noise
        
        # Transition matrix (constant acceleration model)
        self.F = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0.8, 0],  # Damping on acceleration
            [0, 0, 0, 0, 0, 0.8]
        ])
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Physics constraints
        self.gravity = 0.5
        self.max_velocity = 50  # pixels per frame
        self.max_acceleration = 10
        self.initialized = False
        
    def initialize(self, x, y):
        """Initialize filter with first detection"""
        self.state = np.array([x, y, 0, 0, 0, self.gravity])
        self.P = np.eye(6) * 100
        self.initialized = True
        
    def predict(self):
        """Predict next state"""
        if not self.initialized:
            return None
            
        # Apply physics (gravity)
        self.state[5] = self.gravity
        
        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Apply velocity constraints
        self.state[2] = np.clip(self.state[2], -self.max_velocity, self.max_velocity)
        self.state[3] = np.clip(self.state[3], -self.max_velocity, self.max_velocity)
        
        # Apply acceleration constraints
        self.state[4] = np.clip(self.state[4], -self.max_acceleration, self.max_acceleration)
        self.state[5] = np.clip(self.state[5], -self.max_acceleration, self.max_acceleration + self.gravity)
        
        return self.get_position()
        
    def update(self, measurement):
        """Update with measurement"""
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
        """Get current position estimate"""
        if not self.initialized:
            return None
        return (self.state[0], self.state[1])
        
    def get_velocity(self):
        """Get current velocity estimate"""
        if not self.initialized:
            return None
        return (self.state[2], self.state[3])


class OpticalFlowTracker:
    """Optical flow tracker with prediction for out-of-view objects."""
    def __init__(self):
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.prev_gray = None
        self.tracking_points = None
        # **NEW**: Store the last known velocity to predict position when tracking is lost.
        self.last_known_velocity = np.array([0.0, 0.0])
        
    def initialize(self, frame, center_x, center_y, roi_size=30):
        """Initialize optical flow tracking around a detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        x1 = max(0, int(center_x - roi_size/2))
        y1 = max(0, int(center_y - roi_size/2))
        x2 = min(frame.shape[1], int(center_x + roi_size/2))
        y2 = min(frame.shape[0], int(center_y + roi_size/2))
        
        roi = gray[y1:y2, x1:x2]
        corners = cv2.goodFeaturesToTrack(roi, maxCorners=20, qualityLevel=0.01, minDistance=5)
        
        if corners is not None and len(corners) > 0:
            corners[:, :, 0] += x1
            corners[:, :, 1] += y1
            self.tracking_points = corners
        else:
            self.tracking_points = np.array([[[center_x, center_y]]], dtype=np.float32)
            
        self.prev_gray = gray
        # **NEW**: Reset velocity on re-initialization.
        self.last_known_velocity = np.array([0.0, 0.0])
        
    def track(self, frame):
        """
        Track points using optical flow.
        If points are lost (e.g., ball out of frame), predict the new position
        based on the last known velocity.
        """
        if self.prev_gray is None or self.tracking_points is None or len(self.tracking_points) == 0:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.tracking_points, None, **self.lk_params
        )
        
        good_new_points = next_points[status == 1] if next_points is not None else np.array([])
        good_old_points = self.tracking_points[status == 1]

        # **MODIFIED**: Check for a sufficient number of tracked points.
        if len(good_new_points) > 5:
            # --- Successful Track ---
            # **NEW**: Calculate and store the current velocity.
            self.last_known_velocity = np.mean(good_new_points - good_old_points, axis=0)
            
            # Calculate centroid of tracked points
            center_x = np.mean(good_new_points[:, 0])
            center_y = np.mean(good_new_points[:, 1])
            
            # Update tracking points for the next frame
            self.tracking_points = good_new_points.reshape(-1, 1, 2)
            
            self.prev_gray = gray
            return (center_x, center_y)
        else:
            # --- Track Lost / Ball Out of View ---
            # **NEW**: Predict the position using the last known velocity.
            
            # Get the centroid of the last known point cloud
            last_centroid = np.mean(self.tracking_points, axis=0).flatten()
            
            # Apply damping to simulate friction/air resistance
            self.last_known_velocity *= 0.98 
            
            # Predict the new position
            predicted_position = last_centroid + self.last_known_velocity
            
            # Update the tracking points to the predicted location for the next frame
            if self.tracking_points is not None:
                self.tracking_points = (self.tracking_points.reshape(-1, 2) + self.last_known_velocity).reshape(-1, 1, 2)
            
            self.prev_gray = gray
            
            # Return the predicted position, which may be outside the frame
            return (predicted_position[0], predicted_position[1])


class MultiHypothesisTracker:
    """Multi-hypothesis tracker for handling multiple possible tracks"""
    def __init__(self, max_hypotheses=5):
        self.hypotheses = []
        self.max_hypotheses = max_hypotheses
        self.next_id = 0
        
    def add_hypothesis(self, detection, confidence):
        """Add new tracking hypothesis"""
        hypothesis = {
            'id': self.next_id,
            'positions': deque([detection], maxlen=10),
            'confidence': confidence,
            'age': 0,
            'last_update': 0
        }
        self.hypotheses.append(hypothesis)
        self.next_id += 1
        
    def update_hypotheses(self, frame_idx, detections=None):
        """Update all hypotheses"""
        for hyp in self.hypotheses:
            hyp['age'] += 1
            
            if detections:
                # Find best matching detection for this hypothesis
                best_match = None
                best_distance = float('inf')
                
                last_pos = hyp['positions'][-1]
                for det in detections:
                    distance = np.sqrt((det[0] - last_pos[0])**2 + (det[1] - last_pos[1])**2)
                    if distance < best_distance and distance < 50:  # threshold
                        best_distance = distance
                        best_match = det
                
                if best_match:
                    hyp['positions'].append(best_match)
                    hyp['confidence'] = min(1.0, hyp['confidence'] + 0.1)
                    hyp['last_update'] = frame_idx
                else:
                    hyp['confidence'] = max(0.0, hyp['confidence'] - 0.05)
                    
        # Remove low confidence hypotheses
        self.hypotheses = [h for h in self.hypotheses if h['confidence'] > 0.1 and h['age'] < 30]
        
        # Limit number of hypotheses
        if len(self.hypotheses) > self.max_hypotheses:
            self.hypotheses.sort(key=lambda x: x['confidence'], reverse=True)
            self.hypotheses = self.hypotheses[:self.max_hypotheses]
            
    def get_best_hypothesis(self):
        """Get the best hypothesis"""
        if not self.hypotheses:
            return None
            
        best = max(self.hypotheses, key=lambda x: x['confidence'])
        if best['confidence'] > 0.3:
            return best['positions'][-1]
        return None


class EnhancedBallTracker:
    """Enhanced ball tracker combining multiple techniques"""
    def __init__(self, img_width, img_height):
        self.kalman = KalmanFilter()
        self.optical_flow = OpticalFlowTracker()
        self.multi_hypothesis = MultiHypothesisTracker()
        
        self.img_width = img_width
        self.img_height = img_height
        
        # Detection history for interpolation
        self.detection_history = deque(maxlen=10)
        self.frames_without_detection = 0
        self.confidence_threshold = 0.3
        self.adaptive_threshold = 0.3
        
        # Reference histogram for appearance matching
        self.reference_histogram = None
        self.last_detection_pos = None
        self.ball_size = (30, 30)
        
    def update_reference_histogram(self, frame, x_center, y_center):
        """Update reference histogram from detected ball region"""
        width, height = self.ball_size
        x1 = int(max(0, x_center - width/2))
        y1 = int(max(0, y_center - height/2))
        x2 = int(min(frame.shape[1], x_center + width/2))
        y2 = int(min(frame.shape[0], y_center + height/2))
        
        ball_roi = frame[y1:y2, x1:x2]
        if ball_roi.size > 0:
            try:
                blue_channel, green_channel, red_channel = cv2.split(ball_roi)
                
                hist_r = calculate_histogram(red_channel, 'Red', 'red')
                hist_g = calculate_histogram(green_channel, 'Green', 'green')
                hist_b = calculate_histogram(blue_channel, 'Blue', 'blue')
                
                self.reference_histogram = {
                    'red': hist_r,
                    'green': hist_g,
                    'blue': hist_b
                }
                self.last_detection_pos = (x_center, y_center)
            except:
                pass
                
    def find_ball_by_histogram(self, frame, search_center=None, search_radius=100):
        """Enhanced histogram matching with adaptive search"""
        if self.reference_histogram is None:
            return None
            
        if search_center is None:
            if self.last_detection_pos is None:
                return None
            search_center = self.last_detection_pos
            
        # Adaptive search radius based on tracking confidence
        if self.frames_without_detection > 5:
            search_radius = min(200, search_radius * 2)
            
        center_x, center_y = search_center
        
        # Define search region
        x1 = max(0, int(center_x - search_radius))
        y1 = max(0, int(center_y - search_radius))
        x2 = min(frame.shape[1], int(center_x + search_radius))
        y2 = min(frame.shape[0], int(center_y + search_radius))
        
        search_region = frame[y1:y2, x1:x2]
        if search_region.size == 0:
            return None
            
        best_match = None
        best_score = float('inf')
        
        # Multi-scale search
        scales = [0.8, 1.0, 1.2] if self.frames_without_detection < 3 else [0.6, 0.8, 1.0, 1.2, 1.4]
        
        for scale in scales:
            w = int(self.ball_size[0] * scale)
            h = int(self.ball_size[1] * scale)
            
            if w >= search_region.shape[1] or h >= search_region.shape[0]:
                continue
                
            step = max(3, min(w//3, h//3))
            
            for y in range(0, search_region.shape[0] - h, step):
                for x in range(0, search_region.shape[1] - w, step):
                    roi = search_region[y:y+h, x:x+w]
                    
                    try:
                        blue_channel, green_channel, red_channel = cv2.split(roi)
                        hist_r = calculate_histogram(red_channel, 'Red', 'red')
                        
                        # Quick brightness check
                        tot_pixel = np.sum(hist_r)
                        if tot_pixel > 0:
                            pixel_intens = np.arange(len(hist_r))
                            weighted_sum = np.sum(pixel_intens * hist_r[:, 0])
                            avg_pixel = weighted_sum / tot_pixel
                            if avg_pixel > 142:  # Too bright
                                continue
                                
                        # Calculate histogram distance
                        score = self._calculate_histogram_distance(
                            self.reference_histogram['red'], hist_r
                        )
                        
                        if score < best_score:
                            best_score = score
                            global_x = x1 + x + w//2
                            global_y = y1 + y + h//2
                            best_match = {
                                'x': global_x,
                                'y': global_y,
                                'width': w,
                                'height': h,
                                'score': score,
                                'confidence': max(0.1, min(0.9, 1.0 / (1.0 + score)))
                            }
                    except:
                        continue
                        
        # Return match if score is acceptable
        threshold = 1.5 if self.frames_without_detection < 5 else 2.5
        if best_match and best_score < threshold:
            return best_match
            
        return None
        
    def _calculate_histogram_distance(self, hist1, hist2):
        """Calculate Chi-square distance between histograms"""
        min_len = min(len(hist1), len(hist2))
        h1 = hist1[:min_len].flatten()
        h2 = hist2[:min_len].flatten()
        
        h1_norm = h1 / (np.sum(h1) + 1e-10)
        h2_norm = h2 / (np.sum(h2) + 1e-10)
        
        chi_square = 0.5 * np.sum((h1_norm - h2_norm) ** 2 / (h1_norm + h2_norm + 1e-10))
        return chi_square
        
    def interpolate_missing_detection(self):
        """Interpolate ball position using physics model"""
        if len(self.detection_history) < 2:
            return None
            
        # Use last two detections to estimate trajectory
        p2 = self.detection_history[-1]
        p1 = self.detection_history[-2]
        
        # Calculate velocity and predict next position
        dt = 1  # frame difference
        vx = (p2[0] - p1[0]) / dt
        vy = (p2[1] - p1[1]) / dt
        
        # Apply physics (gravity and air resistance)
        predicted_x = p2[0] + vx * dt
        predicted_y = p2[1] + vy * dt + 0.5 * self.frames_without_detection
        
        # Bounds checking
        predicted_x = np.clip(predicted_x, 0, self.img_width)
        predicted_y = np.clip(predicted_y, 0, self.img_height)
        
        return (predicted_x, predicted_y)
        
    def process_frame(self, frame, yolo_detections):
        """Process frame with multi-modal tracking"""
        detection_result = None
        detection_source = None
        
        # Check for YOLO detection of ball (class 0)
        yolo_ball = None
        for det in yolo_detections:
            if det.get('class_id') == 0:
                yolo_ball = det
                break
                
        if yolo_ball:
            # YOLO detected ball
            bbox = yolo_ball['bbox']
            img_h, img_w = frame.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            if len(bbox) == 4:  # [x_center, y_center, width, height] normalized
                x_center = bbox[0] * img_w
                y_center = bbox[1] * img_h
            else:  # [x_center, y_center] pixel coordinates
                x_center, y_center = bbox[0], bbox[1]
                
            detection_result = (x_center, y_center)
            detection_source = 'yolo'
            self.frames_without_detection = 0
            
            # Update all trackers
            if not self.kalman.initialized:
                self.kalman.initialize(x_center, y_center)
                self.optical_flow.initialize(frame, x_center, y_center)
            else:
                self.kalman.update([x_center, y_center])
                
            # Update appearance model
            self.update_reference_histogram(frame, x_center, y_center)
            self.detection_history.append((x_center, y_center))
            
            # Reset adaptive threshold
            self.adaptive_threshold = 0.3
            
        else:
            # No YOLO detection - use alternative methods
            self.frames_without_detection += 1
            
            # Lower confidence threshold adaptively
            self.adaptive_threshold = max(0.1, self.adaptive_threshold - 0.02)
            
            # Try multiple tracking methods in order of reliability
            candidates = []
            
            # 1. Kalman filter prediction
            if self.kalman.initialized:
                kalman_pred = self.kalman.predict()
                if kalman_pred:
                    candidates.append({
                        'position': kalman_pred,
                        'confidence': max(0.3, 0.8 - self.frames_without_detection * 0.05),
                        'source': 'kalman'
                    })
                    
            # 2. Optical flow tracking
            optical_result = self.optical_flow.track(frame)
            if optical_result:
                candidates.append({
                    'position': optical_result,
                    'confidence': max(0.2, 0.7 - self.frames_without_detection * 0.03),
                    'source': 'optical_flow'
                })
                
            # 3. Histogram matching
            search_pos = None
            if candidates:
                # Search around best prediction
                search_pos = candidates[0]['position']
            elif self.last_detection_pos:
                search_pos = self.last_detection_pos
                
            if search_pos:
                hist_result = self.find_ball_by_histogram(frame, search_pos)
                if hist_result:
                    candidates.append({
                        'position': (hist_result['x'], hist_result['y']),
                        'confidence': hist_result['confidence'],
                        'source': 'histogram'
                    })
                    
            # 4. Physics-based interpolation
            if not candidates and len(self.detection_history) >= 2:
                interp_result = self.interpolate_missing_detection()
                if interp_result:
                    candidates.append({
                        'position': interp_result,
                        'confidence': max(0.1, 0.5 - self.frames_without_detection * 0.05),
                        'source': 'interpolation'
                    })
                    
            # Select best candidate
            if candidates:
                # Weight by confidence and recency
                best_candidate = max(candidates, key=lambda x: x['confidence'])
                
                if best_candidate['confidence'] > self.adaptive_threshold:
                    detection_result = best_candidate['position']
                    detection_source = best_candidate['source']
                    
                    # Update Kalman filter with estimated position
                    if detection_source != 'kalman':
                        self.kalman.update(detection_result)
                        
        return detection_result, detection_source


def calculate_histogram(image, channel_name, color):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    sorted_hist = np.sort(hist)
    trimmed_hist = sorted_hist[10:-10]
    hist_smooth = gaussian_filter1d(trimmed_hist, sigma=2)
    return hist_smooth


def load_histograms(file_path):
    try:
        with open(file_path, 'r') as json_file:
            histograms = json.load(json_file)
        return histograms
    except:
        return {}


def run_tracker(model_path: str, video_path: str):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Try to load histograms, continue without if not available
    histograms = load_histograms('out13_histograms.json')
    
    # Extract video name for output and frame keys
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_enhanced")
    os.makedirs(output_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, "enhanced_tracking_results.json")
    video_output_path = os.path.join(output_dir, "enhanced_tracked_video.mp4")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    # Initialize enhanced tracker
    ball_tracker = EnhancedBallTracker(width, height)
    
    tracking_results = {}
    frame_idx = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length)
    
    while cap.isOpened():
        pbar.update(1)
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, verbose=False)[0]
        detections = []

        # Process YOLO results
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
                        'track_id': cls_id,
                        'bbox': [
                            x_center / img_w,
                            y_center / img_h,
                            w / img_w,
                            h / img_h
                        ],
                        'conf': conf
                    }

            # Filter ball detections (class 0) with color validation
            for k in sorted(best_by_class.keys()):
                det = best_by_class[k]
                
                if k == 0:  # Ball detection
                    xc, yc, bw, bh = det['bbox']
                    x1 = int((xc - bw / 2) * img_w)
                    y1 = int((yc - bh / 2) * img_h)
                    x2 = int((xc + bw / 2) * img_w)
                    y2 = int((yc + bh / 2) * img_h)
                    
                    # Color filtering for false positive reduction
                    ball_roi = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    if ball_roi.size > 0:
                        try:
                            blue_channel, green_channel, red_channel = cv2.split(ball_roi)
                            hist_r = calculate_histogram(red_channel, 'Red', 'red')
                            
                            tot_pixel = np.sum(hist_r)
                            if tot_pixel > 0:
                                pixel_intens = np.arange(len(hist_r))
                                weighted_sum = np.sum(pixel_intens * hist_r[:, 0])
                                avg_pixel = weighted_sum / tot_pixel
                                
                                if avg_pixel > 142:
                                    continue  # Skip false positive
                        except:
                            pass
                
                detections.append(det)

        # Process frame with enhanced tracker
        ball_position, detection_source = ball_tracker.process_frame(frame, detections)
        
        # Update histograms from external source if available
        if frame_idx % 5 == 0 and histograms and frame_idx//5 in histograms:
            hist_data = histograms[frame_idx//5]
            if 'red' in hist_data:
                ball_tracker.reference_histogram = {
                    'red': hist_data['red'],
                    'green': hist_data['green'],
                    'blue': hist_data['blue']
                }

        # Draw all detections
        for det in detections:
            xc, yc, bw, bh = det['bbox']
            x1 = int((xc - bw / 2) * img_w)
            y1 = int((yc - bh / 2) * img_h)
            x2 = int((xc + bw / 2) * img_w)
            y2 = int((yc + bh / 2) * img_h)
            
            color = (0, 255, 0) if det['class_id'] == 0 else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"YOLO ID {det['track_id']} ({det['conf']:.2f})", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw enhanced tracking result
        if ball_position and detection_source:
            x, y = ball_position
            w, h = ball_tracker.ball_size
            
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            # Color coding by source
            colors = {
                'yolo': (0, 255, 0),
                'kalman': (255, 0, 0),
                'optical_flow': (0, 255, 255),
                'histogram': (255, 255, 0),
                'interpolation': (255, 0, 255)
            }
            
            color = colors.get(detection_source, (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{detection_source.upper()}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
            # Add ball detection to results
            if detection_source != 'yolo':  # Avoid duplicate
                ball_detection = {
                    'class_id': 0,
                    'track_id': 0,
                    'bbox': [x/img_w, y/img_h, w/img_w, h/img_h],
                    'conf': 0.8 if detection_source in ['kalman', 'optical_flow'] else 0.6,
                    'source': detection_source
                }
                detections.append(ball_detection)

        frame_key = f"{video_name}_{frame_idx}"
        tracking_results[frame_key] = detections
        frame_idx += 1

        out_video.write(frame)

    cap.release()
    out_video.release()
    pbar.close()
    
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f)

    print(f"Enhanced tracking complete. Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Enhanced Ball Tracking")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    args = parser.parse_args()

    run_tracker(args.model, args.video)