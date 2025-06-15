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



class ParticleFilter:
    def __init__(self, num_particles=100, img_width=640, img_height=480, homography_matrix=None):
        self.num_particles = num_particles
        self.img_width = img_width
        self.img_height = img_height
        self.particles = None
        self.weights = None
        self.initialized = False
        self.last_detection_frame = -1
        self.frames_without_detection = 0
        self.max_frames_without_detection = 30
        
        self.H = homography_matrix
        self.ground_y_world = None
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
            ground_point_image = np.array([self.img_width / 2.0, self.img_height - 1.0])
            world_ground_point = self._transform_point(ground_point_image, self.H_inv)
            if world_ground_point is not None:
                self.ground_y_world = world_ground_point[1]

        self.particle_width = 30
        self.particle_height = 30

        self.gravity = 0.8
        self.bounce_damping = 0.7
        self.air_resistance = 0.98
        self.process_noise_pos = 1.0
        self.process_noise_vel = 0.5
        
        self.expected_color_hist = None
        self.color_weight = 0.3
        
    def _transform_point(self, point, M):
        px, py = point
        p_homogeneous = np.array([px, py, 1.0])
        p_transformed = M @ p_homogeneous
        w = p_transformed[2]
        if abs(w) < 1e-6: return None
        return np.array([p_transformed[0] / w, p_transformed[1] / w])

    def initialize(self, x_center, y_center, frame_roi=None):
        self.particles = np.zeros((self.num_particles, 4))
        self.particles[:, 0] = np.random.normal(x_center, 8, self.num_particles)
        self.particles[:, 1] = np.random.normal(y_center, 8, self.num_particles)
        self.particles[:, 2] = np.random.normal(0, 5, self.num_particles)
        self.particles[:, 3] = np.random.normal(0, 5, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = True
        self.frames_without_detection = 0
        if frame_roi is not None: self.update_color_model(frame_roi)
        
    def update_color_model(self, roi):
        if roi.size > 0:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv_roi], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv_roi], [1], None, [32], [0, 256])
            h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-10)
            s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-10)
            self.expected_color_hist = {'hue': h_hist, 'saturation': s_hist}
    
    def predict(self):
        if not self.initialized: return
        noise = np.random.normal(0, 1, (self.num_particles, 4))
        for i in range(self.num_particles):
            x, y, vx, vy = self.particles[i]
            if self.H is not None and self.H_inv is not None and self.ground_y_world is not None:
                pos_world = self._transform_point((x, y), self.H_inv)
                vel_end_world = self._transform_point((x + vx, y + vy), self.H_inv)
                if pos_world is None or vel_end_world is None: continue
                vx_world, vy_world = vel_end_world - pos_world
                vy_world += self.gravity
                vx_world *= self.air_resistance
                vy_world *= self.air_resistance
                new_pos_world = pos_world + np.array([vx_world, vy_world])
                if new_pos_world[1] >= self.ground_y_world and vy_world > 0:
                    new_pos_world[1] = self.ground_y_world
                    vy_world = -vy_world * self.bounce_damping
                new_pos_img = self._transform_point(new_pos_world, self.H)
                new_vel_end_world = new_pos_world + np.array([vx_world, vy_world])
                new_vel_end_img = self._transform_point(new_vel_end_world, self.H)
                if new_pos_img is None or new_vel_end_img is None: continue
                new_x, new_y = new_pos_img
                vx, vy = new_vel_end_img - new_pos_img
            else:
                vy += self.gravity
                vx *= self.air_resistance
                vy *= self.air_resistance
                new_x, new_y = x + vx, y + vy
                if new_y >= self.img_height - self.particle_height/2:
                     vy = -abs(vy) * self.bounce_damping
                     new_y = self.img_height - self.particle_height/2
            w, h = self.particle_width, self.particle_height
            if new_x <= w/2 or new_x >= self.img_width - w/2:
                vx = -vx * self.bounce_damping
                new_x = np.clip(new_x, w/2, self.img_width - w/2)
            if new_y <= h/2:
                vy = abs(vy) * self.bounce_damping
                new_y = h/2
            new_x += noise[i, 0] * self.process_noise_pos
            new_y += noise[i, 1] * self.process_noise_pos
            vx += noise[i, 2] * self.process_noise_vel
            vy += noise[i, 3] * self.process_noise_vel
            self.particles[i, 0] = np.clip(new_x, 0, self.img_width)
            self.particles[i, 1] = np.clip(new_y, 0, self.img_height)
            self.particles[i, 2] = vx
            self.particles[i, 3] = vy

    def calculate_color_likelihood(self, frame, x, y):
        if self.expected_color_hist is None: return 1.0
        w, h = self.particle_width, self.particle_height
        x1, y1 = int(max(0, x - w/2)), int(max(0, y - h/2))
        x2, y2 = int(min(self.img_width, x + w/2)), int(min(self.img_height, y + h/2))
        if x2 <= x1 or y2 <= y1: return 0.1
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return 0.1
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv_roi], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv_roi], [1], None, [32], [0, 256])
            h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-10)
            s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-10)
            h_corr = cv2.compareHist(h_hist.astype(np.float32), self.expected_color_hist['hue'].astype(np.float32), cv2.HISTCMP_CORREL)
            s_corr = cv2.compareHist(s_hist.astype(np.float32), self.expected_color_hist['saturation'].astype(np.float32), cv2.HISTCMP_CORREL)
            return max(0.1, (h_corr + s_corr) / 2)
        except: return 0.1
    
    def update_weights(self, frame, observation_x, observation_y):
        if not self.initialized: return
        for i in range(self.num_particles):
            dx, dy = self.particles[i, 0] - observation_x, self.particles[i, 1] - observation_y
            distance = np.sqrt(dx**2 + dy**2)
            pos_likelihood = np.exp(-0.5 * (distance**2 / 50))
            color_likelihood = self.calculate_color_likelihood(frame, self.particles[i, 0], self.particles[i, 1])
            self.weights[i] = (pos_likelihood * (1 - self.color_weight) + color_likelihood * self.color_weight)
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)
        
    def resample_around_detection(self, det_x, det_y, frame_roi=None):
        if frame_roi is not None: self.update_color_model(frame_roi)
        num_around = int(0.8 * self.num_particles)
        num_keep = self.num_particles - num_around
        new_particles = np.zeros_like(self.particles)
        if num_keep > 0:
            indices = np.random.choice(self.num_particles, num_keep, p=self.weights, replace=True)
            new_particles[:num_keep] = self.particles[indices]
        for i in range(num_keep, self.num_particles):
            new_particles[i, 0] = np.random.normal(det_x, 5)
            new_particles[i, 1] = np.random.normal(det_y, 5)
            new_particles[i, 2] = np.random.normal(0, 8)
            new_particles[i, 3] = np.random.normal(0, 8)
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def resample(self):
        if not self.initialized: return
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        cumulative_sum = np.cumsum(self.weights)
        new_particles, i, j = np.zeros_like(self.particles), 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else: j += 1
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def get_estimate(self):
        if not self.initialized: return None
        estimate = np.average(self.particles, weights=self.weights, axis=0)
        confidence = (1.0 / np.sum(self.weights**2)) / self.num_particles
        return {'x': estimate[0], 'y': estimate[1], 'vx': estimate[2], 'vy': estimate[3], 'width': self.particle_width, 'height': self.particle_height, 'confidence': confidence}
    
    def is_tracking_lost(self):
        return self.frames_without_detection > self.max_frames_without_detection



def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    sorted_hist = np.sort(hist, axis=0)
    trimmed_hist = sorted_hist[10:-10]
    return gaussian_filter1d(trimmed_hist, sigma=2)


def run_tracker(model_path: str, video_path: str):
    # The homography matrix you provided
    H = np.array([
        [-7.38116462e+00, -1.08710040e+01,  8.47854924e+03],
        [ 4.26095662e-01, -2.65510860e+01,  1.28280893e+04],
        [ 1.37685820e-04, -7.32618644e-03,  1.00000000e+00]
    ])
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_classid_homography")
    os.makedirs(output_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, "particle_tracking_results.json")
    video_output_path = os.path.join(output_dir, "particle_tracked_video.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    # Initialize particle filter with the homography matrix
    particle_filter = ParticleFilter(num_particles=100, img_width=width, img_height=height, homography_matrix=H)
    
    tracking_results = {}
    frame_idx = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length)
    
    while cap.isOpened():
        pbar.update(1)
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []
        ball_detected = False

        particle_filter.predict()

        if results.boxes is not None:
            boxes = results.boxes
            best_by_class = {}

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x_center, y_center, w, h = boxes.xywh[i].tolist()
                
                if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                    best_by_class[cls_id] = {'conf': conf, 'xywh': (x_center, y_center, w, h), 'cls_id': cls_id}

            for k in sorted(best_by_class.keys()):
                det_data = best_by_class[k]
                xc, yc, bw, bh = det_data['xywh']
                img_h, img_w = frame.shape[:2]
                
                det = {
                    'class_id': det_data['cls_id'],
                    'track_id': det_data['cls_id'],
                    'bbox': [xc / img_w, yc / img_h, bw / img_w, bh / img_h],
                    'conf': det_data['conf']
                }
                
                if k == 0:
                    ball_x, ball_y = xc, yc
                    x1, y1 = int(ball_x - bw/2), int(ball_y - bh/2)
                    x2, y2 = int(ball_x + bw/2), int(ball_y + bh/2)
                    
                    ball_roi = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    valid_detection = True
                    
                    if ball_roi.size > 0:
                        _, _, red_channel = cv2.split(ball_roi)
                        hist_r = calculate_histogram(red_channel)
                        if np.sum(hist_r) > 0:
                            avg_pixel = np.sum(np.arange(len(hist_r)) * hist_r[:, 0]) / np.sum(hist_r)
                            if avg_pixel > 142: valid_detection = False
                    
                    if valid_detection:
                        ball_detected = True
                        if not particle_filter.initialized:
                            # Initialize using detection center, ignore w/h
                            particle_filter.initialize(ball_x, ball_y, ball_roi)
                        else:
                            # Resample using detection center, ignore w/h
                            particle_filter.resample_around_detection(ball_x, ball_y, ball_roi)
                        
                        particle_filter.frames_without_detection = 0
                        particle_filter.last_detection_frame = frame_idx
                
                detections.append(det)

                # Draw YOLO detections
                x1, y1 = int((det['bbox'][0] - det['bbox'][2]/2) * img_w), int((det['bbox'][1] - det['bbox'][3]/2) * img_h)
                x2, y2 = int((det['bbox'][0] + det['bbox'][2]/2) * img_w), int((det['bbox'][1] + det['bbox'][3]/2) * img_h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"YOLO ID {det['track_id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if not ball_detected and particle_filter.initialized and not particle_filter.is_tracking_lost():
            estimate = particle_filter.get_estimate()
            if estimate:
                img_h, img_w = frame.shape[:2]
                pf_x, pf_y = estimate['x'], estimate['y']
                pf_w, pf_h = estimate['width'], estimate['height']
                
                detections.append({
                    'class_id': 0, 'track_id': 0,
                    'bbox': [pf_x / img_w, pf_y / img_h, pf_w / img_w, pf_h / img_h],
                    'conf': min(0.9, estimate['confidence']), 'source': 'particle_filter',
                    'velocity': [estimate['vx'], estimate['vy']]
                })
                
                x1, y1 = int(pf_x - pf_w/2), int(pf_y - pf_h/2)
                x2, y2 = int(pf_x + pf_w/2), int(pf_y + pf_h/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"PF Ball {estimate['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                center_x, center_y = int(pf_x), int(pf_y)
                vel_end_x, vel_end_y = int(center_x + estimate['vx'] * 5), int(center_y + estimate['vy'] * 5)
                cv2.arrowedLine(frame, (center_x, center_y), (vel_end_x, vel_end_y), (0, 255, 255), 2)
                
                for i in range(0, particle_filter.num_particles, 5):
                    px, py = int(particle_filter.particles[i, 0]), int(particle_filter.particles[i, 1])
                    if 0 <= px < img_w and 0 <= py < img_h:
                        cv2.circle(frame, (px, py), 1, (255, 255, 0), -1)

        tracking_results[f"{video_name}_{frame_idx}"] = detections
        frame_idx += 1
        out_video.write(frame)

    cap.release()
    out_video.release()
    pbar.close()
    
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f, indent=4)

    print(f"Particle filter tracking complete. Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO + Particle Filter tracking on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    args = parser.parse_args()

    run_tracker(args.model, args.video)