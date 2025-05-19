import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from scipy import interpolate
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import math
class BasketballTracker:
    def __init__(self, gravity=9.81, pixels_per_meter=50, bounce_damping=0.85):
        self.gravity = gravity  # m/s^2
        self.pixels_per_meter = pixels_per_meter  # Approximate conversion
        self.bounce_damping = bounce_damping  # Energy loss on bounce
        self.min_bounce_threshold = 5  # Minimum vertical velocity change to detect bounce
        
    def fit_parabolic_segments(self, positions, frame_indices, fps):
        """
        Fit parabolic trajectories between detected positions, accounting for bounces
        """
        if len(positions) < 2:
            return positions
            
        # Convert to numpy arrays
        positions = np.array(positions)
        frame_indices = np.array(frame_indices)
        
        # Time in seconds for each frame
        times = frame_indices / fps
        
        # Detect potential bounce points by looking for direction changes
        bounce_indices = self.detect_bounces(positions, times)
        
        # Add start and end indices
        segment_boundaries = [0] + bounce_indices + [len(positions) - 1]
        
        # Fit parabolic trajectory for each segment
        interpolated_positions = []
        interpolated_frames = []
        
        for i in range(len(segment_boundaries) - 1):
            start_idx = segment_boundaries[i]
            end_idx = segment_boundaries[i + 1]
            
            if end_idx - start_idx < 2:
                continue
                
            segment_positions = positions[start_idx:end_idx + 1]
            segment_times = times[start_idx:end_idx + 1]
            segment_frames = frame_indices[start_idx:end_idx + 1]
            
            # Fit parabolic trajectory
            fitted_trajectory = self.fit_parabola(segment_positions, segment_times)
            
            # Interpolate for all frames in segment
            all_frames = np.arange(segment_frames[0], segment_frames[-1] + 1)
            all_times = all_frames / fps
            
            interpolated_x = fitted_trajectory['x_func'](all_times)
            interpolated_y = fitted_trajectory['y_func'](all_times)
            
            for frame, x, y in zip(all_frames, interpolated_x, interpolated_y):
                interpolated_positions.append([x, y])
                interpolated_frames.append(frame)
        
        return np.array(interpolated_positions), np.array(interpolated_frames)
    
    def detect_bounces(self, positions, times):
        """
        Detect bounce points by analyzing vertical velocity changes
        """
        if len(positions) < 3:
            return []
            
        # Calculate vertical velocities
        y_positions = positions[:, 1]
        velocities = np.gradient(y_positions, times)
        
        # Smooth velocities to reduce noise
        if len(velocities) > 5:
            velocities = savgol_filter(velocities, 5, 2)
        
        bounce_indices = []
        
        for i in range(1, len(velocities) - 1):
            # Look for sign changes in velocity (moving down to up)
            if velocities[i-1] > 0 and velocities[i+1] < 0:  # Peak (ball at highest point)
                continue
            elif velocities[i-1] < 0 and velocities[i+1] > 0:  # Bounce
                velocity_change = abs(velocities[i+1] - velocities[i-1])
                if velocity_change > self.min_bounce_threshold:
                    bounce_indices.append(i)
        
        return bounce_indices
    
    def fit_parabola(self, positions, times):
        """
        Fit a parabolic trajectory to a set of positions
        """
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]
        
        # Fit x as linear function of time (constant horizontal velocity)
        x_coeffs = np.polyfit(times, x_positions, 1)
        x_func = np.poly1d(x_coeffs)
        
        # Fit y as quadratic function of time (constant acceleration)
        y_coeffs = np.polyfit(times, y_positions, 2)
        y_func = np.poly1d(y_coeffs)
        
        return {
            'x_func': x_func,
            'y_func': y_func,
            'x_coeffs': x_coeffs,
            'y_coeffs': y_coeffs
        }
    
    def extrapolate_position(self, last_positions, last_frames, current_frame, fps, frame_width, frame_height):
        """
        Extrapolate ball position when it goes out of frame
        """
        if len(last_positions) < 2:
            return last_positions[-1]
            
        # Get last few positions for trajectory estimation
        recent_positions = np.array(last_positions[-5:])
        recent_frames = np.array(last_frames[-5:])
        recent_times = recent_frames / fps
        
        # Fit trajectory
        trajectory = self.fit_parabola(recent_positions, recent_times)
        
        # Extrapolate to current frame
        current_time = current_frame / fps
        extrapolated_x = trajectory['x_func'](current_time)
        extrapolated_y = trajectory['y_func'](current_time)
        
        # Apply bounds (keep ball visible near frame edges)
        margin = 50  # pixels
        extrapolated_x = np.clip(extrapolated_x, -margin, frame_width + margin)
        extrapolated_y = np.clip(extrapolated_y, -margin, frame_height + margin)
        
        return np.array([extrapolated_x, extrapolated_y])
    
    def smooth_trajectory(self, positions, window_size=5):
        """
        Apply smoothing to the final trajectory
        """
        if len(positions) < window_size:
            return positions
            
        positions = np.array(positions)
        smoothed = np.copy(positions)
        
        # Apply Savitzky-Golay filter for smoothing
        if len(positions) > window_size:
            smoothed[:, 0] = savgol_filter(positions[:, 0], window_size, 2)
            smoothed[:, 1] = savgol_filter(positions[:, 1], window_size, 2)
            
        return smoothed

# Main tracking code
video_name = 'out2'
video_path = f'../data/videos/{video_name}.mp4'
output_path = f"../data/output/{video_name}_physics_tracking.mp4"
storage_dir = '../data/processed_frames_all_detections'

# Initialize tracker
tracker = BasketballTracker(gravity=9.81, pixels_per_meter=50)

# Load detection data
detections_filepath = os.path.join(storage_dir, f"{video_name}_all_detections.json")
with open(detections_filepath, 'r') as f:
    all_detections_data = json.load(f)

# Get video properties
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process detections to extract ball positions
ball_positions = []
ball_frames = []
BALL_CLASS_ID = 0
ball_speeds = []

print("Extracting ball positions from detections...")
for frame_idx in range(total_frames):
    detections = all_detections_data.get(str(frame_idx), [])
    
    for detection in detections:
        if detection['class_id'] == BALL_CLASS_ID:
            
            box = detection['box']
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            if len(ball_positions) == 0 or math.sqrt((cx-ball_positions[-1][0])**2 + (cy-ball_positions[-1][1])**2)/(frame_idx-ball_frames[-1]) < 50:
                if len(ball_positions) > 0:
                   ball_speeds.append(math.sqrt((cx-ball_positions[-1][0])**2 + (cy-ball_positions[-1][1])**2)/(frame_idx-ball_frames[-1]))
                ball_positions.append([cx, cy])
                ball_frames.append(frame_idx)
                
            break
ball_speeds.sort()
ball_speeds = np.array(ball_speeds)
breakpoint()
# Apply physics-based interpolation
print("Applying physics-based tracking...")
if len(ball_positions) > 1:
    interpolated_positions, interpolated_frames = tracker.fit_parabolic_segments(
        ball_positions, ball_frames, fps
    )
    
    # Handle frames before first detection and after last detection
    all_ball_positions = []
    
    for frame_idx in range(total_frames):
        if frame_idx < interpolated_frames[0]:
            # Before first detection - use first position
            all_ball_positions.append(interpolated_positions[0])
        elif frame_idx > interpolated_frames[-1]:
            # After last detection - extrapolate
            position = tracker.extrapolate_position(
                ball_positions, ball_frames, frame_idx, fps, width, height
            )
            all_ball_positions.append(position)
        else:
            # Find interpolated position
            idx = np.where(interpolated_frames == frame_idx)[0]
            if len(idx) > 0:
                all_ball_positions.append(interpolated_positions[idx[0]])
            else:
                # Frame not in interpolated set - extrapolate
                position = tracker.extrapolate_position(
                    ball_positions, ball_frames, frame_idx, fps, width, height
                )
                all_ball_positions.append(position)
    
    # Smooth the final trajectory
    all_ball_positions = tracker.smooth_trajectory(np.array(all_ball_positions))
    
else:
    # Fallback to simple interpolation if not enough detections
    all_ball_positions = np.zeros((total_frames, 2))
    if len(ball_positions) > 0:
        all_ball_positions[:] = ball_positions[0]

# Draw on frames
print("Creating output video...")
cap.release()
cap = cv2.VideoCapture(video_path)

for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw all detections
    detections = all_detections_data.get(str(frame_idx), [])
    for detection in detections:
        box = detection['box']
        confidence = detection['confidence']
        class_id = detection['class_id']
        track_id = detection['track_id']
        
        x1, y1, x2, y2 = map(int, box)
        
        # Different colors for different classes
        if class_id == BALL_CLASS_ID:
            color = (0, 255, 0)  # Green for ball
        else:
            color = (255, 0, 0)  # Blue for players
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"ID:{class_id}"
        if track_id != -1:
            label += f" T:{track_id}"
        label += f" {confidence:.2f}"
        
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # Draw interpolated ball position
    ball_pos = all_ball_positions[frame_idx].astype(int)
    cv2.circle(frame, tuple(ball_pos), 30, (0, 255, 255), 3)  # Yellow circle for interpolated
    
    # Draw trajectory trail
    trail_length = 20
    start_idx = max(0, frame_idx - trail_length)
    for i in range(start_idx, frame_idx):
        alpha = (i - start_idx) / trail_length
        pos1 = all_ball_positions[i].astype(int)
        pos2 = all_ball_positions[i + 1].astype(int)
        cv2.line(frame, tuple(pos1), tuple(pos2), 
                (0, int(255 * alpha), int(255 * alpha)), 2)
    
    out.write(frame)

cap.release()
out.release()

print(f"Physics-based tracking video saved to: {output_path}")