import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from collections import deque
from tqdm import tqdm

@dataclass
class BallTemplate:
    """Stores ball template information for matching"""
    image: np.ndarray
    histogram: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frame_idx: int

class BallFallbackDetector:
    """Fallback detection system for basketball when YOLO fails"""
    
    def __init__(self, max_templates: int = 5, search_radius: int = 150):
        self.max_templates = max_templates
        self.search_radius = search_radius
        self.templates = deque(maxlen=max_templates)
        self.last_known_position = None
        self.position_history = deque(maxlen=10)
        
        # Initialize feature detector (ORB is fast and efficient)
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        
        # Initialize template matcher
        self.template_threshold = 0.6
        self.hist_threshold = 0.7
        self.feature_match_threshold = 15  # Minimum good matches
        
        # Ball appearance constraints
        self.min_ball_size = 8
        self.max_ball_size = 50
        self.ball_color_ranges = [
            # Orange basketball color ranges in HSV
            (np.array([5, 100, 100]), np.array([15, 255, 255])),   # Orange
            (np.array([10, 50, 50]), np.array([25, 255, 255]))     # Light orange/brown
        ]
    
    def add_template(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                    confidence: float, frame_idx: int) -> bool:
        """Add a new ball template from successful YOLO detection"""
        x1, y1, x2, y2 = bbox
        
        # Validate bbox
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
            return False
            
        # Extract ball region with padding
        padding = 5
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(frame.shape[1], x2 + padding)
        y2_pad = min(frame.shape[0], y2 + padding)
        
        ball_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if ball_roi.size == 0:
            return False
        
        # Calculate histogram
        hsv_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [50, 60, 60], 
                           [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Extract features
        gray_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray_roi, None)
        
        if descriptors is None:
            descriptors = np.array([])
        
        # Create template
        template = BallTemplate(
            image=ball_roi.copy(),
            histogram=hist,
            keypoints=keypoints,
            descriptors=descriptors,
            bbox=(x1, y1, x2, y2),
            confidence=confidence,
            frame_idx=frame_idx
        )
        
        self.templates.append(template)
        self.last_known_position = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.position_history.append(self.last_known_position)
        
        return True
    
    def get_search_region(self, frame_shape: Tuple[int, int], min_size: Tuple[int, int] = None) -> Tuple[int, int, int, int]:
        """Get search region based on last known position and constraints"""
        height, width = frame_shape[:2]
        
        # Set minimum search region size
        min_width = min_size[0] if min_size else 100
        min_height = min_size[1] if min_size else 100
        
        if self.last_known_position is None:
            # Search in lower 70% of frame if no previous position
            return 0, int(height * 0.3), width, height
        
        cx, cy = self.last_known_position
        
        # Expand search area around last position
        x1 = max(0, cx - self.search_radius)
        y1 = max(0, cy - self.search_radius)
        x2 = min(width, cx + self.search_radius)
        y2 = min(height, cy + self.search_radius)
        
        # Ensure we don't search above 70% height constraint
        y1 = max(y1, int(height * 0.3))
        
        # Ensure minimum search region size
        current_width = x2 - x1
        current_height = y2 - y1
        
        if current_width < min_width:
            # Expand width symmetrically
            expand_x = (min_width - current_width) // 2
            x1 = max(0, x1 - expand_x)
            x2 = min(width, x2 + expand_x)
            # If still not enough, expand the other side
            if x2 - x1 < min_width:
                if x1 == 0:
                    x2 = min(width, x1 + min_width)
                else:
                    x1 = max(0, x2 - min_width)
        
        if current_height < min_height:
            # Expand height, but respect the 70% constraint
            expand_y = (min_height - current_height) // 2
            y1_new = max(int(height * 0.3), y1 - expand_y)
            y2_new = min(height, y2 + expand_y)
            
            # If still not enough height, expand downward only
            if y2_new - y1_new < min_height:
                y2_new = min(height, y1_new + min_height)
            
            y1, y2 = y1_new, y2_new
        
        return x1, y1, x2, y2
    
    def template_matching(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """Perform template matching with best template"""
        if not self.templates:
            return None
        
        best_template = max(self.templates, key=lambda t: t.confidence)
        template = best_template.image
        
        if template.size == 0:
            return None
        
        # Get search region
        x1, y1, x2, y2 = self.get_search_region(frame.shape)
        search_region = frame[y1:y2, x1:x2]
        
        if search_region.size == 0:
            return None
        
        # Perform template matching
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < self.template_threshold:
            return None
        
        # Convert back to full frame coordinates
        match_x = max_loc[0] + x1
        match_y = max_loc[1] + y1
        
        # Return bounding box
        th, tw = template.shape[:2]
        return (match_x, match_y, match_x + tw, match_y + th, max_val)
    
    def histogram_matching(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """Find ball using histogram comparison"""
        if not self.templates:
            return None
        
        # Find the largest template to ensure minimum search size
        max_template_size = (0, 0)
        for template in self.templates:
            th, tw = template.image.shape[:2]
            max_template_size = (max(max_template_size[0], tw), max(max_template_size[1], th))
        
        # Get search region with minimum size
        min_search_size = (max_template_size[0] + 50, max_template_size[1] + 50)
        x1, y1, x2, y2 = self.get_search_region(frame.shape, min_search_size)
        search_region = frame[y1:y2, x1:x2]
        
        if search_region.size == 0:
            return None
        
        hsv_search = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        best_match = None
        best_score = 0
        
        # Use sliding window approach
        for template in self.templates:
            if template.histogram is None:
                continue
                
            th, tw = template.image.shape[:2]
            
            # Check if search region is large enough
            if hsv_search.shape[0] < th or hsv_search.shape[1] < tw:
                continue
            
            # Slide window across search region
            for y in range(0, hsv_search.shape[0] - th + 1, 5):  # Step by 5 for efficiency
                for x in range(0, hsv_search.shape[1] - tw + 1, 5):
                    window = hsv_search[y:y+th, x:x+tw]
                    
                    if window.size == 0:
                        continue
                    
                    # Calculate histogram
                    hist = cv2.calcHist([window], [0, 1, 2], None, [50, 60, 60], 
                                      [0, 180, 0, 256, 0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    
                    # Compare histograms
                    score = cv2.compareHist(template.histogram, hist, cv2.HISTCMP_CORREL)
                    
                    if score > best_score and score > self.hist_threshold:
                        best_score = score
                        # Convert to full frame coordinates
                        full_x1 = x + x1
                        full_y1 = y + y1
                        best_match = (full_x1, full_y1, full_x1 + tw, full_y1 + th, score)
        
        return best_match
    
    def feature_matching(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """Find ball using feature matching (ORB)"""
        if not self.templates:
            return None
        
        # Get search region
        x1, y1, x2, y2 = self.get_search_region(frame.shape)
        search_region = frame[y1:y2, x1:x2]
        
        if search_region.size == 0:
            return None
        
        gray_search = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.orb.detectAndCompute(gray_search, None)
        
        if des2 is None:
            return None
        
        # FLANN matcher for feature matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        best_match = None
        best_score = 0
        
        for template in self.templates:
            if template.descriptors is None or len(template.descriptors) == 0:
                continue
            
            try:
                matches = flann.knnMatch(template.descriptors, des2, k=2)
                
                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) > self.feature_match_threshold:
                    # Find centroid of matches
                    points = [kp2[m.trainIdx].pt for m in good_matches]
                    if points:
                        cx = int(np.mean([p[0] for p in points]))
                        cy = int(np.mean([p[1] for p in points]))
                        
                        # Estimate bounding box size from template
                        th, tw = template.image.shape[:2]
                        
                        # Convert to full frame coordinates
                        full_cx = cx + x1
                        full_cy = cy + y1
                        
                        score = len(good_matches) / max(len(template.descriptors), 1)
                        
                        if score > best_score:
                            best_score = score
                            best_match = (full_cx - tw//2, full_cy - th//2, 
                                        full_cx + tw//2, full_cy + th//2, score)
                            
            except Exception as e:
                continue
        
        return best_match
    
    def color_based_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect ball candidates using color filtering"""
        # Get search region
        x1, y1, x2, y2 = self.get_search_region(frame.shape)
        search_region = frame[y1:y2, x1:x2]
        
        if search_region.size == 0:
            return []
        
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Create mask for basketball colors
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.ball_color_ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_ball_size * self.min_ball_size:
                continue
            
            # Get bounding rectangle
            rect_x, rect_y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (ball should be roughly circular)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.7 <= aspect_ratio <= 1.3):
                continue
            
            # Check size constraints
            if not (self.min_ball_size <= w <= self.max_ball_size and 
                   self.min_ball_size <= h <= self.max_ball_size):
                continue
            
            # Convert to full frame coordinates
            full_x1 = rect_x + x1
            full_y1 = rect_y + y1
            full_x2 = full_x1 + w
            full_y2 = full_y1 + h
            
            # Calculate confidence based on area and circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                confidence = min(circularity, 1.0)
            else:
                confidence = 0.5
            
            candidates.append((full_x1, full_y1, full_x2, full_y2, confidence))
        
        return candidates
    
    def detect_ball_fallback(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Main fallback detection method that combines all techniques"""
        results = []
        
        # Try template matching
        template_result = self.template_matching(frame)
        if template_result:
            x1, y1, x2, y2, conf = template_result
            results.append({
                'method': 'template',
                'bbox': [x1, y1, x2, y2],
                'confidence': conf * 0.8,  # Slightly reduce confidence
                'score': conf
            })
        
        # Try histogram matching
        hist_result = self.histogram_matching(frame)
        if hist_result:
            x1, y1, x2, y2, conf = hist_result
            results.append({
                'method': 'histogram',
                'bbox': [x1, y1, x2, y2],
                'confidence': conf * 0.7,  # Reduce confidence more
                'score': conf
            })
        
        # Try feature matching
        feature_result = self.feature_matching(frame)
        if feature_result:
            x1, y1, x2, y2, conf = feature_result
            results.append({
                'method': 'features',
                'bbox': [x1, y1, x2, y2],
                'confidence': min(conf * 0.6, 0.9),  # Cap confidence
                'score': conf
            })
        
        # Try color-based detection
        color_results = self.color_based_detection(frame)
        for x1, y1, x2, y2, conf in color_results:
            results.append({
                'method': 'color',
                'bbox': [x1, y1, x2, y2],
                'confidence': conf * 0.5,  # Lowest confidence
                'score': conf
            })
        
        if not results:
            return None
        
        # Select best result based on confidence and consistency with previous position
        best_result = None
        best_score = 0
        
        for result in results:
            score = result['confidence']
            
            # Boost score if consistent with previous position
            if self.last_known_position:
                x1, y1, x2, y2 = result['bbox']
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                last_cx, last_cy = self.last_known_position
                
                distance = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                if distance < self.search_radius:
                    # Closer to last position gets bonus
                    proximity_bonus = 1.0 - (distance / self.search_radius) * 0.3
                    score *= proximity_bonus
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result


def run_tracker_with_fallback(model_path: str, video_path: str, template_path: str = None):
    """Enhanced tracker with fallback detection"""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # Initialize fallback detector
    fallback_detector = BallFallbackDetector()
    
    # Load template if provided
    if template_path and os.path.exists(template_path):
        template_img = cv2.imread(template_path)
        if template_img is not None:
            h, w = template_img.shape[:2]
            # Add as initial template
            fallback_detector.add_template(template_img, (0, 0, w, h), 1.0, 0)
            print(f"Loaded initial template from {template_path}")
    
    # Extract video name for output
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_fallback")
    os.makedirs(output_dir, exist_ok=True)
    
    json_output_path = os.path.join(output_dir, "ball_fallback_tracking_results.json")
    video_output_path = os.path.join(output_dir, "ball_fallback_tracked_video.mp4")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    tracking_results = {}
    frame_idx = 0
    yolo_detections = 0
    fallback_detections = 0
    
    pbar = tqdm(total= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    while cap.isOpened():
        pbar.update(1)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame,verbose=False)[0]
        detections = []
        ball_detected_by_yolo = False
        
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
                        'conf': conf,
                        'detection_method': 'yolo'
                    }
            
            # Check if ball (class_id 0) was detected
            if 0 in best_by_class:
                ball_detected_by_yolo = True
                yolo_detections += 1
                
                # Add ball template for fallback detection
                ball_det = best_by_class[0]
                xc, yc, bw, bh = ball_det['bbox']
                x1 = int((xc - bw / 2) * img_w)
                y1 = int((yc - bh / 2) * img_h)
                x2 = int((xc + bw / 2) * img_w)
                y2 = int((yc + bh / 2) * img_h)
                
                fallback_detector.add_template(frame, (x1, y1, x2, y2), 
                                             ball_det['conf'], frame_idx)
            
            # Add all detections
            for det in best_by_class.values():
                detections.append(det)
        
        # If ball not detected by YOLO, try fallback detection
        if not ball_detected_by_yolo:
            fallback_result = fallback_detector.detect_ball_fallback(frame)
            
            if fallback_result:
                fallback_detections += 1
                x1, y1, x2, y2 = fallback_result['bbox']
                
                # Convert to normalized coordinates
                ball_detection = {
                    'class_id': 0,
                    'track_id': 0,
                    'bbox': [
                        (x1 + x2) / 2 / width,   # x_center
                        (y1 + y2) / 2 / height,  # y_center
                        (x2 - x1) / width,       # width
                        (y2 - y1) / height       # height
                    ],
                    'conf': fallback_result['confidence'],
                    'detection_method': f"fallback_{fallback_result['method']}"
                }
                detections.append(ball_detection)
                
                # Update fallback detector with successful detection
                fallback_detector.add_template(frame, (x1, y1, x2, y2), 
                                             fallback_result['confidence'], frame_idx)
        
        # Draw detections on frame
        for det in detections:
            xc, yc, bw, bh = det['bbox']
            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)
            
            # Color coding: Green for YOLO, Red for fallback
            color = (0, 255, 0) if det['detection_method'] == 'yolo' else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label with method and confidence
            method = det['detection_method']
            conf = det['conf']
            label = f"ID {det['track_id']} ({method}: {conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        # Store results
        frame_key = f"{video_name}_{frame_idx}"
        tracking_results[frame_key] = detections
        
        frame_idx += 1
        out_video.write(frame)
        
        # Print progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames - YOLO: {yolo_detections}, "
                  f"Fallback: {fallback_detections}")
    
    cap.release()
    out_video.release()
    pbar.close()
    # Save results
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f, indent=2)
    
    print(f"\nTracking complete!")
    print(f"Total frames: {frame_idx}")
    print(f"YOLO detections: {yolo_detections}")
    print(f"Fallback detections: {fallback_detections}")
    print(f"Ball detection rate: {((yolo_detections + fallback_detections) / frame_idx * 100):.1f}%")
    print(f"Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO tracking with fallback detection")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--template", type=str, help="Path to ball template image (optional)")
    
    args = parser.parse_args()
    run_tracker_with_fallback(args.model, args.video, args.template)