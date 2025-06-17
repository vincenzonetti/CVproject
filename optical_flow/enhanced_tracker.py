"""
Enhanced ball tracker implementation.
Combines multiple tracking techniques for robust ball tracking.
"""

import cv2
import numpy as np
from collections import deque

from filters import KalmanFilter
from optical_flow import OpticalFlowTracker
from multi_hypothesis import MultiHypothesisTracker
from utils import (
    calculate_histogram, calculate_chi_square_distance,
    get_roi_bounds, clip_coordinates, is_ball_too_bright
)
from config import DEFAULT_BALL_SIZE, TrackingConfig, HistogramConfig
from scipy.ndimage import gaussian_filter1d
import json

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
