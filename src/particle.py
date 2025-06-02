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
    def __init__(self, num_particles=100, img_width=640, img_height=480):
        self.num_particles = num_particles
        self.img_width = img_width
        self.img_height = img_height
        self.particles = None
        self.weights = None
        self.initialized = False
        self.last_detection_frame = -1
        self.frames_without_detection = 0
        self.max_frames_without_detection = 30
        
        # Motion model parameters
        self.gravity = 0.8  # Gravity acceleration (pixels/frame^2)
        self.bounce_damping = 0.7  # Velocity reduction on bounce
        self.air_resistance = 0.98  # Air resistance factor
        self.process_noise_pos = 3.0
        self.process_noise_vel = 1.5
        self.bounce_threshold = 0.5  # 50% of image height
        
        # Color model for ball (updated from detections)
        self.expected_color_hist = None
        self.color_weight = 0.3  # Weight of color in likelihood
        
    def initialize(self, x_center, y_center, width, height, frame_roi=None):
        """Initialize particles around the first detection"""
        self.particles = np.zeros((self.num_particles, 6))  # [x, y, vx, vy, w, h]
        
        # Initialize positions with spread around detection
        self.particles[:, 0] = np.random.normal(x_center, 8, self.num_particles)  # x
        self.particles[:, 1] = np.random.normal(y_center, 8, self.num_particles)  # y
        self.particles[:, 2] = np.random.normal(0, 5, self.num_particles)  # vx (higher variance)
        self.particles[:, 3] = np.random.normal(0, 5, self.num_particles)  # vy (higher variance)
        self.particles[:, 4] = np.random.normal(width, 3, self.num_particles)  # w
        self.particles[:, 5] = np.random.normal(height, 3, self.num_particles)  # h
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = True
        self.frames_without_detection = 0
        
        # Update color model if ROI provided
        if frame_roi is not None:
            self.update_color_model(frame_roi)
        
    def update_color_model(self, roi):
        """Update expected color histogram from ball ROI"""
        if roi.size > 0:
            # Convert to different color spaces for better discrimination
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for H and S channels
            h_hist = cv2.calcHist([hsv_roi], [0], None, [32], [0, 180])  # Hue
            s_hist = cv2.calcHist([hsv_roi], [1], None, [32], [0, 256])  # Saturation
            
            # Normalize histograms
            h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-10)
            s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-10)
            
            self.expected_color_hist = {'hue': h_hist, 'saturation': s_hist}
    
    def predict(self):
        """Predict particle states with basketball physics"""
        if not self.initialized:
            return
            
        # Add process noise
        noise = np.random.normal(0, 1, (self.num_particles, 6))
        
        for i in range(self.num_particles):
            # Current state
            x, y, vx, vy, w, h = self.particles[i]
            
            # Basketball physics motion model
            
            # 1. Apply gravity (always downward)
            vy += self.gravity
            
            # 2. Air resistance (reduces velocity)
            vx *= self.air_resistance
            vy *= self.air_resistance
            
            # 3. Parabolic motion - if ball is going up, it should have higher initial velocity
            # This is implicitly handled by the velocity prediction and gravity
            
            # 4. Update position
            new_x = x + vx
            new_y = y + vy
            
            # 5. Boundary handling and bouncing
            # Horizontal boundaries
            if new_x <= w/2 or new_x >= self.img_width - w/2:
                vx = -vx * self.bounce_damping
                new_x = np.clip(new_x, w/2, self.img_width - w/2)
            
            # Vertical boundaries with bounce logic
            if new_y <= h/2:
                # Hit top boundary
                vy = abs(vy) * self.bounce_damping
                new_y = h/2
            elif new_y >= self.img_height - h/2:
                # Hit bottom boundary
                if y < self.img_height * self.bounce_threshold:
                    # Ball was above 50% of image, likely to bounce
                    vy = -abs(vy) * self.bounce_damping
                    new_y = self.img_height - h/2
                else:
                    # Ball was in lower part, less likely to bounce (might roll)
                    if abs(vy) > 3:  # High velocity bounce
                        vy = -abs(vy) * self.bounce_damping
                        new_y = self.img_height - h/2
                    else:  # Low velocity, might stop bouncing
                        vy = 0
                        new_y = self.img_height - h/2
            
            # 6. Add process noise
            new_x += noise[i, 0] * self.process_noise_pos
            new_y += noise[i, 1] * self.process_noise_pos
            vx += noise[i, 2] * self.process_noise_vel
            vy += noise[i, 3] * self.process_noise_vel
            
            # 7. Update particle
            self.particles[i, 0] = np.clip(new_x, 0, self.img_width)
            self.particles[i, 1] = np.clip(new_y, 0, self.img_height)
            self.particles[i, 2] = vx
            self.particles[i, 3] = vy
            self.particles[i, 4] = max(5, w + noise[i, 4] * 1.0)  # Min width 5
            self.particles[i, 5] = max(5, h + noise[i, 5] * 1.0)  # Min height 5
        
    def calculate_color_likelihood(self, frame, x, y, w, h):
        """Calculate likelihood based on color histogram"""
        if self.expected_color_hist is None:
            return 1.0
        
        # Extract ROI
        x1 = int(max(0, x - w/2))
        y1 = int(max(0, y - h/2))
        x2 = int(min(self.img_width, x + w/2))
        y2 = int(min(self.img_height, y + h/2))
        
        if x2 <= x1 or y2 <= y1:
            return 0.1
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.1
            
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms
            h_hist = cv2.calcHist([hsv_roi], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv_roi], [1], None, [32], [0, 256])
            
            # Normalize
            h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-10)
            s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-10)
            
            # Calculate histogram correlation
            h_corr = cv2.compareHist(h_hist.astype(np.float32), 
                                   self.expected_color_hist['hue'].astype(np.float32), 
                                   cv2.HISTCMP_CORREL)
            s_corr = cv2.compareHist(s_hist.astype(np.float32), 
                                   self.expected_color_hist['saturation'].astype(np.float32), 
                                   cv2.HISTCMP_CORREL)
            
            # Combine correlations
            color_similarity = (h_corr + s_corr) / 2
            return max(0.1, color_similarity)  # Minimum likelihood
            
        except:
            return 0.1
    
    def update_weights(self, frame, observation_x, observation_y, observation_w, observation_h):
        """Update particle weights based on observation"""
        if not self.initialized:
            return
            
        for i in range(self.num_particles):
            # Position likelihood
            dx = self.particles[i, 0] - observation_x
            dy = self.particles[i, 1] - observation_y
            dw = self.particles[i, 4] - observation_w
            dh = self.particles[i, 5] - observation_h
            
            distance = np.sqrt(dx**2 + dy**2)
            size_diff = np.sqrt(dw**2 + dh**2)
            
            # Position and size likelihood
            pos_likelihood = np.exp(-0.5 * (distance**2 / 50))
            size_likelihood = np.exp(-0.5 * (size_diff**2 / 25))
            
            # Color likelihood
            color_likelihood = self.calculate_color_likelihood(
                frame, self.particles[i, 0], self.particles[i, 1],
                self.particles[i, 4], self.particles[i, 5]
            )
            
            # Combine likelihoods
            self.weights[i] = (pos_likelihood * size_likelihood * 
                             (1 - self.color_weight) + color_likelihood * self.color_weight)
        
        # Normalize weights
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)
        
    def resample_around_detection(self, det_x, det_y, det_w, det_h, frame_roi=None):
        """Resample particles around YOLO detection (high confidence)"""
        # Update color model if ROI provided
        if frame_roi is not None:
            self.update_color_model(frame_roi)
        
        # Resample majority of particles around detection
        num_around_detection = int(0.8 * self.num_particles)  # 80% around detection
        num_keep_existing = self.num_particles - num_around_detection
        
        new_particles = np.zeros_like(self.particles)
        
        # Keep some existing particles (diversity)
        if num_keep_existing > 0:
            indices = np.random.choice(self.num_particles, num_keep_existing, 
                                     p=self.weights, replace=True)
            new_particles[:num_keep_existing] = self.particles[indices]
        
        # Generate new particles around detection
        for i in range(num_keep_existing, self.num_particles):
            new_particles[i, 0] = np.random.normal(det_x, 5)  # x
            new_particles[i, 1] = np.random.normal(det_y, 5)  # y
            new_particles[i, 2] = np.random.normal(0, 8)      # vx (varied for different trajectories)
            new_particles[i, 3] = np.random.normal(0, 8)      # vy
            new_particles[i, 4] = np.random.normal(det_w, 2)  # w
            new_particles[i, 5] = np.random.normal(det_h, 2)  # h
        
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def resample(self):
        """Standard systematic resampling"""
        if not self.initialized:
            return
            
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
        
        # Calculate confidence based on weight concentration
        effective_sample_size = 1.0 / np.sum(self.weights**2)
        confidence = effective_sample_size / self.num_particles
        
        return {
            'x': estimate[0],
            'y': estimate[1],
            'vx': estimate[2],
            'vy': estimate[3],
            'width': estimate[4],
            'height': estimate[5],
            'confidence': confidence
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


def run_tracker(model_path: str, video_path: str):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Extract video name for output and frame keys
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_classid")
    os.makedirs(output_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, "particle_tracking_results.json")
    video_output_path = os.path.join(output_dir, "particle_tracked_video.mp4")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    # Initialize particle filter
    particle_filter = ParticleFilter(num_particles=100, img_width=width, img_height=height)
    
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
                    valid_detection = True
                    
                    if ball_roi.size > 0:
                        blue_channel, green_channel, red_channel = cv2.split(ball_roi)
                        hist_r = calculate_histogram(red_channel, 'Red', 'red')
                        
                        tot_pixel = np.sum(hist_r)
                        if tot_pixel > 0:
                            pixel_intens = np.arange(len(hist_r))
                            weighted_sum = np.sum(pixel_intens * hist_r[:, 0])
                            avg_pixel = weighted_sum / tot_pixel
                            
                            if avg_pixel > 142:
                                valid_detection = False  # Skip false positive
                    
                    if valid_detection:
                        # Valid ball detection
                        ball_detected = True
                        ball_x = xc * img_w
                        ball_y = yc * img_h
                        ball_w = bw * img_w
                        ball_h = bh * img_h
                        
                        # Initialize or update particle filter with strong resampling
                        if not particle_filter.initialized:
                            particle_filter.initialize(ball_x, ball_y, ball_w, ball_h, ball_roi)
                        else:
                            # Strong resampling around YOLO detection (high confidence)
                            particle_filter.resample_around_detection(ball_x, ball_y, ball_w, ball_h, ball_roi)
                        
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
                cv2.putText(frame, f"YOLO ID {det['track_id']:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Handle particle filter tracking when ball not detected by YOLO
        if not ball_detected and particle_filter.initialized:
            particle_filter.frames_without_detection += 1
            
            if not particle_filter.is_tracking_lost():
                # Get particle filter estimate
                estimate = particle_filter.get_estimate()
                if estimate:
                    # Add particle filter detection to results
                    pf_x = estimate['x'] / img_w
                    pf_y = estimate['y'] / img_h
                    pf_w = estimate['width'] / img_w
                    pf_h = estimate['height'] / img_h
                    
                    pf_detection = {
                        'class_id': 0,
                        'track_id': 0,
                        'bbox': [pf_x, pf_y, pf_w, pf_h],
                        'conf': min(0.9, estimate['confidence']),
                        'source': 'particle_filter',
                        'velocity': [estimate['vx'], estimate['vy']]
                    }
                    detections.append(pf_detection)
                    
                    # Draw particle filter prediction
                    x1 = int((pf_x - pf_w / 2) * img_w)
                    y1 = int((pf_y - pf_h / 2) * img_h)
                    x2 = int((pf_x + pf_w / 2) * img_w)
                    y2 = int((pf_y + pf_h / 2) * img_h)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for particle filter
                    cv2.putText(frame, f"PF Ball {estimate['confidence']:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # Draw velocity vector
                    center_x, center_y = int(estimate['x']), int(estimate['y'])
                    vel_end_x = int(center_x + estimate['vx'] * 5)  # Scale for visibility
                    vel_end_y = int(center_y + estimate['vy'] * 5)
                    cv2.arrowedLine(frame, (center_x, center_y), (vel_end_x, vel_end_y), 
                                   (0, 255, 255), 2)  # Yellow arrow for velocity
                    
                    # Draw subset of particles for visualization
                    for i in range(0, particle_filter.num_particles, 5):
                        px = int(particle_filter.particles[i, 0])
                        py = int(particle_filter.particles[i, 1])
                        if 0 <= px < img_w and 0 <= py < img_h:
                            cv2.circle(frame, (px, py), 1, (255, 255, 0), -1)

        frame_key = f"{video_name}_{frame_idx}"
        tracking_results[frame_key] = detections
        frame_idx += 1

        out_video.write(frame)

    cap.release()
    out_video.release()
    pbar.close()
    
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f)

    print(f"Particle filter tracking complete. Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO + Particle Filter tracking on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    args = parser.parse_args()

    run_tracker(args.model, args.video)