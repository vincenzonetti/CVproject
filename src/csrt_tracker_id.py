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


class CSRTBallTracker:
    def __init__(self, img_width=640, img_height=480):
        self.img_width = img_width
        self.img_height = img_height
        self.tracker = None
        self.initialized = False
        self.last_detection_frame = -1
        self.frames_without_detection = 0
        self.max_frames_without_detection = 30  # Track lost after 30 frames
        self.last_ball_position = None
        self.last_bbox = None
        
    def initialize(self, frame, bbox):
        """Initialize CSRT tracker with bounding box from YOLO
        bbox format: (x, y, width, height) in pixels
        """
        # Create new CSRT tracker
        self.tracker = cv2.TrackerCSRT_create()
        
        # Expand bbox if it's too small (minimum 20x20 pixels)
        x, y, w, h = bbox
        min_size = 20
        
        if w < min_size or h < min_size:
            # Calculate expansion needed
            expand_w = max(0, (min_size - w) // 2)
            expand_h = max(0, (min_size - h) // 2)
            
            # Expand bbox while keeping it within frame bounds
            new_x = max(0, x - expand_w)
            new_y = max(0, y - expand_h)
            new_w = min(self.img_width - new_x, w + 2 * expand_w)
            new_h = min(self.img_height - new_y, h + 2 * expand_h)
            
            expanded_bbox = (new_x, new_y, new_w, new_h)
            print(f"Expanding small bbox from {bbox} to {expanded_bbox}")
            bbox = expanded_bbox
        
        # Validate bbox dimensions and position
        x, y, w, h = bbox
        if (x < 0 or y < 0 or x + w > self.img_width or y + h > self.img_height or w <= 0 or h <= 0):
            print(f"Invalid bbox for CSRT initialization: {bbox} (frame size: {self.img_width}x{self.img_height})")
            self.initialized = False
            return False
        
        # Initialize tracker with the frame and bounding box
        success = self.tracker.init(frame, bbox)
        
        if success:
            self.initialized = True
            self.frames_without_detection = 0
            self.last_bbox = bbox
            # Store center position for search window
            self.last_ball_position = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            print(f"CSRT tracker initialized successfully with bbox: {bbox}")
        else:
            print(f"Failed to initialize CSRT tracker with bbox: {bbox}")
            print(f"Frame shape: {frame.shape}, bbox area: {w*h}")
            self.initialized = False
        
        return success
    
    def update(self, frame):
        """Update CSRT tracker and return detection within search window"""
        if not self.initialized or self.tracker is None:
            return None
        
        # Create search window based on last known position
        if self.last_ball_position is None:
            return None
            
        last_x, last_y = self.last_ball_position
        
        # Define search window (150px left, 300px right, 300px up/down)
        search_left = int(max(0, last_x - 150))
        search_right = int(min(self.img_width, last_x + 300))
        search_top = int(max(0, last_y - 300))
        search_bottom = int(min(self.img_height, last_y + 300))
        
        # Extract search region
        search_region = frame[search_top:search_bottom, search_left:search_right]
        
        if search_region.size == 0:
            return None
        
        # Update tracker on the search region
        success, bbox = self.tracker.update(search_region)
        
        if success:
            # Convert bbox from search region coordinates to full frame coordinates
            x, y, w, h = bbox
            global_x = search_left + x
            global_y = search_top + y
            global_bbox = (global_x, global_y, w, h)
            
            # Check if the tracked object is still within the search window
            center_x = global_x + w/2
            center_y = global_y + h/2
            
            if (search_left <= center_x <= search_right and 
                search_top <= center_y <= search_bottom):
                
                # Update last known position and bbox
                self.last_ball_position = (center_x, center_y)
                self.last_bbox = global_bbox
                self.frames_without_detection = 0
                
                return {
                    'x': center_x,
                    'y': center_y,
                    'width': w,
                    'height': h,
                    'bbox': global_bbox,
                    'confidence': 0.7  # Fixed confidence for CSRT
                }
            else:
                # Tracked object moved outside search window
                self.frames_without_detection += 1
                return None
        else:
            # Tracking failed
            self.frames_without_detection += 1
            return None
    
    def reinitialize(self, frame, bbox):
        """Reinitialize tracker with new detection from YOLO"""
        # Reset tracker
        self.tracker = None
        self.initialized = False
        
        # Initialize with new bbox
        return self.initialize(frame, bbox)
    
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

    json_output_path = os.path.join(output_dir, "csrt_tracking_results.json")
    video_output_path = os.path.join(output_dir, "csrt_tracked_video.mp4")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    # Initialize CSRT tracker
    csrt_tracker = CSRTBallTracker(img_width=width, img_height=height)
    
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
                    ball_w = bw * img_w
                    ball_h = bh * img_h
                    
                    # Convert to bbox format (x, y, width, height) for CSRT
                    csrt_bbox = (x1, y1, x2 - x1, y2 - y1)
                    
                    # Initialize or reinitialize CSRT tracker
                    if not csrt_tracker.initialized:
                        csrt_tracker.initialize(frame, csrt_bbox)
                    else:
                        # Reinitialize with new YOLO detection
                        csrt_tracker.reinitialize(frame, csrt_bbox)
                    
                    csrt_tracker.frames_without_detection = 0
                    csrt_tracker.last_detection_frame = frame_idx
                
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

        # Handle CSRT tracking when ball not detected by YOLO
        if not ball_detected and csrt_tracker.initialized:
            csrt_tracker.frames_without_detection += 1
            
            if not csrt_tracker.is_tracking_lost():
                # Try CSRT tracking
                csrt_result = csrt_tracker.update(frame)
                
                if csrt_result:
                    # Add CSRT detection to results
                    csrt_x = csrt_result['x'] / img_w
                    csrt_y = csrt_result['y'] / img_w
                    csrt_w = csrt_result['width'] / img_w
                    csrt_h = csrt_result['height'] / img_h
                    
                    csrt_detection = {
                        'class_id': 0,
                        'track_id': 0,
                        'bbox': [csrt_x, csrt_y, csrt_w, csrt_h],
                        'conf': csrt_result['confidence'],
                        'source': 'csrt_tracker'
                    }
                    detections.append(csrt_detection)
                    
                    # Draw CSRT detection
                    x, y, w, h = csrt_result['bbox']
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    
                    # Blue color for CSRT tracking
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"CSRT {csrt_result['confidence']:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # Draw center point
                    center_x = int(csrt_result['x'])
                    center_y = int(csrt_result['y'])
                    cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
                else:
                    # CSRT tracking failed, stop tracking
                    print(f"CSRT tracking failed at frame {frame_idx}")
            else:
                # Tracking lost for too long, reset tracker
                csrt_tracker.initialized = False
                csrt_tracker.tracker = None
                print(f"CSRT tracking lost at frame {frame_idx}")

        frame_key = f"{video_name}_{frame_idx}"
        tracking_results[frame_key] = detections
        frame_idx += 1

        out_video.write(frame)

    cap.release()
    out_video.release()
    pbar.close()
    
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f)

    print(f"CSRT tracking complete. Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO + CSRT tracking on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    args = parser.parse_args()

    run_tracker(args.model, args.video)