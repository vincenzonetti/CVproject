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
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis, entropy


class ParticleFilter:
    def __init__(self, num_particles=500, img_width=640, img_height=480):
        self.num_particles = num_particles
        self.img_width = img_width
        self.img_height = img_height
        self.particles = None
        self.weights = None
        self.initialized = False
        self.last_detection_frame = -1
        self.frames_without_detection = 0
        self.max_frames_without_detection = 30  # Track lost after 30 frames
        self.gravity = 0.3
        # Motion model parameters
        self.process_noise_pos = 5.0  # Position noise
        self.process_noise_vel = 2.0  # Velocity noise
        
        # Histogram tracking for ball appearance
        self.reference_histogram = None
        self.last_ball_position = None
        self.ball_size = (30,30)
        
    def initialize(self, x_center, y_center):
        """Initialize particles around the first detection"""
        self.particles = np.zeros((self.num_particles, 4)) 
        
        # Initialize positions with some spread around detection
        self.particles[:, 0] = np.random.normal(x_center, 10, self.num_particles)  # x
        self.particles[:, 1] = np.random.normal(y_center, 10, self.num_particles)  # y
        self.particles[:, 2] = np.random.normal(0, 2, self.num_particles)  # vx
        self.particles[:, 3] = np.random.normal(0, 2, self.num_particles)  # vy
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = True
        self.frames_without_detection = 0
        
        # Store last known position and size
        self.last_ball_position = (x_center, y_center)
        
        
    def update_reference_histogram(self, frame, x_center, y_center):
        """Update the reference histogram from the detected ball region"""
        width,height = self.ball_size
        # Extract ball region
        x1 = int(max(0, x_center - width/2))
        y1 = int(max(0, y_center - height/2))
        x2 = int(min(frame.shape[1], x_center + width/2))
        y2 = int(min(frame.shape[0], y_center + height/2))
        
        ball_roi = frame[y1:y2, x1:x2]
        if ball_roi.size > 0:
            # Calculate histogram for all three channels
            blue_channel, green_channel, red_channel = cv2.split(ball_roi)
            
            # Calculate histograms for each channel
            hist_r = calculate_histogram(red_channel, 'Red', 'red')
            hist_g = calculate_histogram(green_channel, 'Green', 'green')
            hist_b = calculate_histogram(blue_channel, 'Blue', 'blue')
            
            # Combine histograms
            self.reference_histogram = {
                'red': hist_r,
                'green': hist_g,
                'blue': hist_b
            }
            
            # Update last known position and size
            self.last_ball_position = (x_center, y_center)
    
    def find_ball_by_histogram(self, frame):
        """Find ball using histogram matching in search window"""
        if self.reference_histogram is None or self.last_ball_position is None:
            return None
            
        last_x, last_y = self.last_ball_position
        w, h = self.ball_size
        
        # Define search window (150px left, 300px right, 300px up/down)
        search_left = int(max(0, last_x - 150))
        search_right = int(min(frame.shape[1], last_x + 300))
        search_top = int(max(0, last_y - 300))
        search_bottom = int(min(frame.shape[0], last_y + 300))
        
        search_region = frame[search_top:search_bottom, search_left:search_right]
        
        if search_region.size == 0:
            return None
        
        # Generate candidate windows
        candidates = self._generate_candidate_windows(
            search_region, search_left, search_top, w, h
        )
        
        best_match = None
        best_score = float('inf')
        
        for candidate in candidates:
            x, y, w, h, roi = candidate
            
            # Skip if ROI is too small
            if roi.size < 100:  # Minimum size threshold
                continue
                
            # Calculate histogram for candidate
            try:
                blue_channel, green_channel, red_channel = cv2.split(roi)
                
                hist_r = calculate_histogram(red_channel, 'Red', 'red')
                hist_g = calculate_histogram(green_channel, 'Green', 'green')
                hist_b = calculate_histogram(blue_channel, 'Blue', 'blue')
                
                # Calculate histogram distance (Chi-square distance)
                score_r = self._calculate_histogram_distance(self.reference_histogram['red'], hist_r)
                score_g = self._calculate_histogram_distance(self.reference_histogram['green'], hist_g)
                score_b = self._calculate_histogram_distance(self.reference_histogram['blue'], hist_b)
                
                # Combined score (weighted average)
                combined_score = 0.8 * score_r + 0.1 * score_g + 0.1 * score_b
                
                # Additional validation: check if red channel average is reasonable for ball
                tot_pixel = np.sum(hist_r)
                if tot_pixel > 0:
                    pixel_intens = np.arange(len(hist_r))
                    weighted_sum = np.sum(pixel_intens * hist_r[:, 0])
                    avg_pixel = weighted_sum / tot_pixel
                    
                    # Skip if too bright (likely false positive)
                    if avg_pixel > 142:
                        continue
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_match = {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'score': combined_score,
                        'confidence': max(0.1, min(0.8, 1.0 / (1.0 + combined_score)))
                    }
                    
            except Exception as e:
                continue  # Skip problematic candidates
        
        # Only return match if score is reasonable
        if best_match and best_score < 2.0:  # Threshold for acceptable match
            return best_match
            
        return None
    
    def _generate_candidate_windows(self, search_region, offset_x, offset_y, ref_w, ref_h):
        """Generate candidate windows for histogram matching"""
        candidates = []
        h, w = search_region.shape[:2]
        
        # Scale factors for window sizes
        scale_factors = [1]
        
        for scale in scale_factors:
            cand_w = int(ref_w * scale)
            cand_h = int(ref_h * scale)
            
            # Skip if candidate size is too large for search region
            if cand_w >= w or cand_h >= h:
                continue
            
            # Slide window with step size
            step_x = max(5, cand_w // 4)
            step_y = max(5, cand_h // 4)
            
            for y in range(0, h - cand_h, step_y):
                for x in range(0, w - cand_w, step_x):
                    roi = search_region[y:y+cand_h, x:x+cand_w]
                    
                    # Convert local coordinates to global coordinates
                    global_x = offset_x + x + cand_w // 2
                    global_y = offset_y + y + cand_h // 2
                    
                    candidates.append((global_x, global_y, cand_w, cand_h, roi))
        
        return candidates
    
    def _calculate_histogram_distance(self, hist1, hist2):
        """Calculate Chi-square distance between histograms"""
        # Ensure both histograms have the same shape
        min_len = min(len(hist1), len(hist2))
        h1 = hist1[:min_len].flatten()
        h2 = hist2[:min_len].flatten()
        
        # Normalize histograms
        h1_norm = h1 / (np.sum(h1) + 1e-10)
        h2_norm = h2 / (np.sum(h2) + 1e-10)
        
        # Chi-square distance
        chi_square = 0.5 * np.sum((h1_norm - h2_norm) ** 2 / (h1_norm + h2_norm + 1e-10))
        
        return chi_square
        
    def predict(self):
        """Predict particle states for next frame"""
        if not self.initialized:
            return
            
        # Add process noise
        noise = np.random.normal(0, 1, (self.num_particles, 6))
        
        # Update positions based on velocity
        self.particles[:, 0] += self.particles[:, 2] + noise[:, 0] * self.process_noise_pos
        self.particles[:, 1] += self.particles[:, 3] + noise[:, 1] * self.process_noise_pos + self.gravity
        
        # Update velocities with noise
        self.particles[:, 2] += noise[:, 2] * self.process_noise_vel
        self.particles[:, 3] += noise[:, 3] * self.process_noise_vel
        

        # Keep particles within image bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.img_width)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.img_height)

        
    def update_weights(self, observation_x, observation_y):
        """Update particle weights based on observation"""
        if not self.initialized:
            return
            
        # Calculate likelihood for each particle
        for i in range(self.num_particles):
            # Distance from particle to observation
            dx = self.particles[i, 0] - observation_x
            dy = self.particles[i, 1] - observation_y

            
            # Gaussian likelihood
            distance = np.sqrt(dx**2 + dy**2)

            
            # Weight based on position and size similarity
            self.weights[i] = np.exp(-0.5 * (distance**2 / 100))
        
        # Normalize weights
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)
        
    def resample(self):
        """Resample particles based on weights"""
        if not self.initialized:
            return
            
        # Systematic resampling
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        cumulative_sum = np.cumsum(self.weights)
        
        new_particles = np.zeros_like(self.particles)
        i, j = 0, 0
        
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1
                
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def get_estimate(self):
        """Get current best estimate (weighted average)"""
        if not self.initialized:
            return None
            
        estimate = np.average(self.particles, weights=self.weights, axis=0)
        return {
            'x': estimate[0],
            'y': estimate[1],
            'confidence': np.max(self.weights) * self.num_particles  # Rough confidence measure
        }
    
    def is_tracking_lost(self):
        """Check if tracking should be considered lost"""
        return self.frames_without_detection > self.max_frames_without_detection


def calculate_histogram(image, channel_name, color):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    sorted_hist = np.sort(hist)
    trimmed_hist = sorted_hist[10:-10]
    hist_smooth = gaussian_filter1d(trimmed_hist, sigma=2)
    return hist_smooth


def load_histograms(file_path):
    with open(file_path, 'r') as json_file:
        histograms = json.load(json_file)
    return histograms

def run_tracker(model_path: str, video_path: str):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    histograms = load_histograms('out13_histograms.json')
    
    # Extract video name for output and frame keys
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_classid")
    os.makedirs(output_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, "particleHist_tracking_results.json")
    video_output_path = os.path.join(output_dir, "particleHist_tracked_video.mp4")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    # Initialize particle filter
    particle_filter = ParticleFilter(num_particles=500, img_width=width, img_height=height)
    
    tracking_results = {}
    frame_idx = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    boxes_dict: dict[int, (float, float)] = {}
    pbar = tqdm(total=length)
    
    while cap.isOpened():
        pbar.update(1)
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []
        ball_detected = False

        # Predict particle filter state
        particle_filter.predict()

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

            for k in sorted(best_by_class.keys()):
                det = best_by_class[k]
                boxes_dict[k] = (det['bbox'][2] * img_w, det['bbox'][3] * img_h)
                
                # Handle ball detection (class 0)
                if k == 0:
                    xc, yc, bw, bh = det['bbox']
                    x1 = int((xc - bw / 2) * img_w)
                    y1 = int((yc - bh / 2) * img_h)
                    x2 = int((xc + bw / 2) * img_w)
                    y2 = int((yc + bh / 2) * img_h)
                    
                    # Color filtering for false positive reduction
                    ball_roi = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    if ball_roi.size > 0:
                        blue_channel, green_channel, red_channel = cv2.split(ball_roi)
                        hist_r = calculate_histogram(red_channel, 'Red', 'red')
                        
                        tot_pixel = np.sum(hist_r)
                        if tot_pixel > 0:
                            pixel_intens = np.arange(len(hist_r))
                            weighted_sum = np.sum(pixel_intens * hist_r[:, 0])
                            avg_pixel = weighted_sum / tot_pixel
                            
                            if avg_pixel > 142:
                                continue  # Skip false positive
                    
                    # Valid ball detection
                    ball_detected = True
                    ball_x = xc * img_w
                    ball_y = yc * img_h

                    
                    # Initialize or update particle filter
                    if not particle_filter.initialized:
                        particle_filter.initialize(ball_x, ball_y)
                    else:
                        particle_filter.update_weights(ball_x, ball_y)
                        particle_filter.resample()
                    
                    # Update reference histogram for tracking
                    particle_filter.update_reference_histogram(frame, ball_x, ball_y)
                    
                    particle_filter.frames_without_detection = 0
                    particle_filter.last_detection_frame = frame_idx
                
                detections.append(det)

                # Draw YOLO detections
                xc, yc, bw, bh = det['bbox']
                x1 = int((xc - bw / 2) * img_w)
                y1 = int((yc - bh / 2) * img_h)
                x2 = int((xc + bw / 2) * img_w)
                y2 = int((yc + bh / 2) * img_h)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"YOLO ID {det['track_id']}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Handle particle filter tracking when ball not detected by YOLO
        if not ball_detected and particle_filter.initialized:
            particle_filter.frames_without_detection += 1
            
            if not particle_filter.is_tracking_lost():
                detection_source = None
                estimate = None
                
                # Try histogram-based detection first (more reliable)
                histogram_match = particle_filter.find_ball_by_histogram(frame)
                if histogram_match:
                    # Update particle filter with histogram match
                    particle_filter.update_weights(histogram_match['x'], histogram_match['y'])
                    particle_filter.resample()
                    
                    # Reset frames without detection counter
                    particle_filter.frames_without_detection = 0
                    
                    estimate = histogram_match
                    detection_source = 'histogram_matching'
                    
                    # Update reference histogram with new detection
                    particle_filter.update_reference_histogram(frame, histogram_match['x'], histogram_match['y'])
                else:
                    # Fall back to particle filter estimate
                    estimate = particle_filter.get_estimate()
                    detection_source = 'particle_filter'
                
                if estimate:
                    # Add detection to results
                    pf_x = estimate['x'] / img_w
                    pf_y = estimate['y'] / img_h
                    pf_w,pf_h = particle_filter.ball_size
                    pf_detection = {
                        'class_id': 0,
                        'track_id': 0,
                        'bbox': [pf_x, pf_y],
                        'conf': min(0.9, estimate['confidence']),
                        'source': detection_source
                    }
                    detections.append(pf_detection)
                    
                    # Draw detection
                    x1 = int((pf_x - pf_w / 2) * img_w)
                    y1 = int((pf_y - pf_h / 2) * img_h)
                    x2 = int((pf_x + pf_w / 2) * img_w)
                    y2 = int((pf_y + pf_h / 2) * img_h)
                    
                    # Different colors for different sources
                    color = (255, 0, 0) if detection_source == 'particle_filter' else (0, 255, 255)  # Blue for PF, Yellow for histogram
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{detection_source.upper()} {estimate['confidence']:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw some particles for visualization (only for particle filter)
                    if detection_source == 'particle_filter':
                        for i in range(0, particle_filter.num_particles, 10):
                            px = int(particle_filter.particles[i, 0])
                            py = int(particle_filter.particles[i, 1])
                            cv2.circle(frame, (px, py), 1, (255, 255, 0), -1)

        frame_key = f"{video_name}_{frame_idx}"
        tracking_results[frame_key] = detections
        if frame_idx % 5 == 0:
            particle_filter.reference_histogram={
                'red': histograms[frame_idx//5]['red'],
                'green': histograms[frame_idx//5]['green'],
                'blue': histograms[frame_idx//5]['blue']
            }
            
        frame_idx += 1

        out_video.write(frame)

    cap.release()
    out_video.release()
    pbar.close()
    
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f)

    print(f"Enhanced particle filter tracking complete. Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO + Enhanced Particle Filter tracking on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    args = parser.parse_args()

    run_tracker(args.model, args.video)