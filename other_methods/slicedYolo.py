import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions (normalized coordinates)"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def find_best_ball_with_temporal_consistency(detections, prev_ball_pos, max_distance_threshold=0.1):
    """
    Find the best ball detection considering temporal consistency
    
    Args:
        detections: List of ball detections from SAHI
        prev_ball_pos: Previous ball position (normalized x, y) or None
        max_distance_threshold: Maximum allowed distance for temporal consistency
    
    Returns:
        Best ball detection or None
    """
    if not detections:
        return None
    
    if prev_ball_pos is None:
        # No previous position, return highest confidence detection
        return max(detections, key=lambda x: x['conf'])
    
    # Filter detections based on distance from previous position
    valid_detections = []
    for det in detections:
        ball_pos = (det['bbox'][0], det['bbox'][1])  # x_center, y_center
        distance = calculate_distance(ball_pos, prev_ball_pos)
        if distance <= max_distance_threshold:
            valid_detections.append((det, distance))
    
    if valid_detections:
        # Return detection with best balance of confidence and proximity
        # Weight: 70% proximity, 30% confidence
        best_det = min(valid_detections, 
                      key=lambda x: (x[1] * 0.7) - (x[0]['conf'] * 0.3))
        return best_det[0]
    else:
        # If no detection is close enough, return highest confidence
        # (ball might have moved quickly)
        return max(detections, key=lambda x: x['conf'])

def run_tracker(model_path: str, video_path: str):
    # Load standard YOLO model
    model = YOLO(model_path)
    
    # Initialize SAHI model for ball detection
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.3,  # Lower threshold for ball detection
        device="cuda:0" if model.device.type == 'cuda' else "cpu"
    )
    
    cap = cv2.VideoCapture(video_path)
    
    # Extract video name for output and frame keys
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_classid_sahi")
    os.makedirs(output_dir, exist_ok=True)
    
    json_output_path = os.path.join(output_dir, "tracking_results.json")
    video_output_path = os.path.join(output_dir, "tracked_video.mp4")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    tracking_results = {}
    frame_idx = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length)
    
    # For temporal consistency
    prev_ball_pos = None
    sahi_used_count = 0
    
    print(f"Processing video: {width}x{height} at {fps} FPS")
    print("Using SAHI fallback for ball detection when standard YOLO fails")
    
    while cap.isOpened():
        pbar.update(1)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Standard YOLO detection
        results = model(frame, verbose=False)[0]
        detections = []
        ball_detected_standard = False
        
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
            
            # Check if ball (class 0) was detected
            if 0 in best_by_class:
                ball_detected_standard = True
                prev_ball_pos = (best_by_class[0]['bbox'][0], best_by_class[0]['bbox'][1])
            
            for det in best_by_class.values():
                detections.append(det)
        
        # Use SAHI for ball detection if standard YOLO didn't detect it
        if not ball_detected_standard:
            try:
                # Run SAHI with slicing
                slice_prediction = get_sliced_prediction(
                    frame,
                    sahi_model,
                    slice_height=512,
                    slice_width=512,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    perform_standard_pred=False  # Only sliced prediction
                )
                
                # Extract ball detections (class 0) from SAHI results
                sahi_ball_detections = []
                for pred in slice_prediction.object_prediction_list:
                    if pred.category.id == 0:  # Ball class
                        bbox = pred.bbox
                        x_center = (bbox.minx + bbox.maxx) / 2 / width
                        y_center = (bbox.miny + bbox.maxy) / 2 / height
                        w = (bbox.maxx - bbox.minx) / width
                        h = (bbox.maxy - bbox.miny) / height
                        
                        sahi_ball_detections.append({
                            'class_id': 0,
                            'track_id': 0,
                            'bbox': [x_center, y_center, w, h],
                            'conf': pred.score.value
                        })
                
                # Find best ball using temporal consistency
                if sahi_ball_detections:
                    best_ball = find_best_ball_with_temporal_consistency(
                        sahi_ball_detections, 
                        prev_ball_pos,
                        max_distance_threshold=0.15  # Slightly higher threshold for SAHI
                    )
                    
                    if best_ball:
                        # Remove any existing ball detection and add the SAHI one
                        detections = [det for det in detections if det['class_id'] != 0]
                        detections.append(best_ball)
                        prev_ball_pos = (best_ball['bbox'][0], best_ball['bbox'][1])
                        sahi_used_count += 1
                        
            except Exception as e:
                print(f"SAHI detection failed for frame {frame_idx}: {e}")
        
        # Draw detections on frame
        img_h, img_w = frame.shape[:2]
        for det in detections:
            xc, yc, bw, bh = det['bbox']
            x1 = int((xc - bw / 2) * img_w)
            y1 = int((yc - bh / 2) * img_h)
            x2 = int((xc + bw / 2) * img_w)
            y2 = int((yc + bh / 2) * img_h)
            
            # Different colors for different classes
            color = (0, 255, 0) if det['class_id'] != 0 else (0, 0, 255)  # Red for ball
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"Ball ID {det['track_id']}" if det['class_id'] == 0 else f"Player ID {det['track_id']}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        frame_key = f"{video_name}_{frame_idx}"
        tracking_results[frame_key] = detections
        frame_idx += 1
        out_video.write(frame)
    
    cap.release()
    pbar.close()
    out_video.release()
    
    # Save results
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f)
    
    print(f"\nTracking complete!")
    print(f"SAHI was used for ball detection in {sahi_used_count} frames ({sahi_used_count/frame_idx*100:.1f}%)")
    print(f"Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO tracking with SAHI fallback for ball detection.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    
    args = parser.parse_args()
    
    print("Enhanced YOLO Tracker with SAHI")
    print("=" * 40)
    print("Installation requirements:")
    print("pip install sahi")
    print("pip install ultralytics")
    print("=" * 40)
    
    run_tracker(args.model, args.video)