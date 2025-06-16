import argparse
import os
import json
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from typing import List, Optional, Dict, Any

def get_class_color(class_id: int) -> tuple:
    """Generate consistent color for each class ID"""
    # Create a color map using HSV space for better color distribution
    colors = [
        (0, 255, 255),    # Red for class 0 (ball)
        (120, 255, 255),  # Green for class 1
        (240, 255, 255),  # Blue for class 2
        (60, 255, 255),   # Yellow for class 3
        (300, 255, 255),  # Magenta for class 4
        (180, 255, 255),  # Cyan for class 5
        (30, 255, 255),   # Orange for class 6
        (270, 255, 255),  # Purple for class 7
    ]
    
    if class_id < len(colors):
        hsv_color = np.uint8([[colors[class_id]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr_color))
    else:
        # Generate color for classes beyond predefined ones
        hue = (class_id * 137) % 360  # Use golden angle for good distribution
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr_color))

def setup_device():
    """Setup and return the appropriate device (CUDA if available)"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")
    return device

def load_models(model_paths: List[str], device: str) -> List[YOLO]:
    """Load YOLO models and move them to the specified device"""
    models = []
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}: {model_path}")
        model = YOLO(model_path)
        # Move model to device
        model.to(device)
        models.append(model)
    return models

def run_inference(models: List[YOLO], frame: np.ndarray, device: str) -> List[Dict[str, Any]]:
    """
    Run inference with multiple models.
    If primary model doesn't detect ball (class 0), run ball-specific model.
    """
    primary_model = models[0]
    
    # Run primary model
    results = primary_model(frame, verbose=False, device=device)[0]
    
    detections = []
    ball_detected = False
    
    if results.boxes is not None:
        boxes = results.boxes
        best_by_class = {}
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x_center, y_center, w, h = boxes.xywh[i].tolist()
            
            if cls_id == 0:  # Ball class
                ball_detected = True
            
            if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                best_by_class[cls_id] = {
                    'class_id': cls_id,
                    'track_id': cls_id,
                    'bbox': [x_center, y_center, w, h],
                    'conf': conf,
                    'model_used': 'primary'
                }
        
        detections.extend(best_by_class.values())
    
    # If no ball detected and we have a ball-specific model, run it
    if not ball_detected and len(models) > 1:
        ball_model = models[1]
        ball_results = ball_model(frame, verbose=False, device=device)[0]
        
        if ball_results.boxes is not None:
            ball_boxes = ball_results.boxes
            best_ball_conf = 0
            best_ball_detection = None
            
            for i in range(len(ball_boxes)):
                cls_id = int(ball_boxes.cls[i].item())
                conf = float(ball_boxes.conf[i].item())
                
                # Only consider ball detections (class 0)
                if cls_id == 0 and conf > best_ball_conf:
                    x_center, y_center, w, h = ball_boxes.xywh[i].tolist()
                    best_ball_detection = {
                        'class_id': cls_id,
                        'track_id': cls_id,
                        'bbox': [x_center, y_center, w, h],
                        'conf': conf,
                        'model_used': 'ball_specific'
                    }
                    best_ball_conf = conf
            
            if best_ball_detection:
                detections.append(best_ball_detection)
                print(f"Ball detected by ball-specific model (conf: {best_ball_conf:.3f})")
    
    return detections

def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and labels on frame with class-specific colors"""
    img_h, img_w = frame.shape[:2]
    
    for det in detections:
        cls_id = det['class_id']
        track_id = det['track_id']
        conf = det['conf']
        model_used = det.get('model_used', 'primary')
        
        # Get normalized bbox coordinates
        x_center, y_center, w, h = det['bbox']
        
        # Convert to pixel coordinates
        x1 = int((x_center - w / 2) * img_w / img_w * img_w)  # Normalize by image width
        y1 = int((y_center - h / 2) * img_h / img_h * img_h)  # Normalize by image height
        x2 = int((x_center + w / 2) * img_w / img_w * img_w)
        y2 = int((y_center + h / 2) * img_h / img_h * img_h)
        
        # Get color for this class
        color = get_class_color(cls_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        label = f"ID {track_id} ({conf:.2f})"
        if model_used == 'ball_specific':
            label += " [B]"  # Indicate ball-specific model was used
        
        # Calculate text size and background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw text background
        cv2.rectangle(
            frame, 
            (x1, y1 - text_height - baseline - 5), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255),  # White text
            1
        )
    
    return frame

def normalize_bbox_for_json(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """Normalize bbox coordinates for JSON output"""
    x_center, y_center, w, h = bbox
    return [
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h
    ]

def run_tracker(model_paths: List[str], video_path: str):
    """Main tracking function with multiple models"""
    # Setup device
    device = setup_device()
    
    # Load models
    models = load_models(model_paths, device)
    
    # Setup video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Extract video name for output
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_names = "_".join([os.path.splitext(os.path.basename(p))[0] for p in model_paths])
    output_dir = os.path.join("outputs", f"{video_name}_{model_names}_multimodel")
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
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {length}")
    print(f"Output directory: {output_dir}")
    
    pbar = tqdm(total=length, desc="Processing frames")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference with multiple models
            detections = run_inference(models, frame, device)
            
            # Normalize bbox coordinates for JSON
            json_detections = []
            for det in detections:
                json_det = det.copy()
                json_det['bbox'] = normalize_bbox_for_json(det['bbox'], width, height)
                json_detections.append(json_det)
            
            # Draw detections on frame
            annotated_frame = draw_detections(frame, detections)
            
            # Save results
            frame_key = f"{video_name}_{frame_idx}"
            tracking_results[frame_key] = json_detections
            
            # Write frame to output video
            out_video.write(annotated_frame)
            
            frame_idx += 1
            pbar.update(1)
    
    finally:
        cap.release()
        out_video.release()
        pbar.close()
    
    # Save tracking results to JSON
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f, indent=2)
    
    print(f"\nTracking complete!")
    print(f"Results saved to: {json_output_path}")
    print(f"Annotated video saved to: {video_output_path}")
    print(f"Device used: {device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-model YOLO tracking with GPU support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model (original behavior)
  python script.py --models model1.pt --video video.mp4
  
  # Multiple models (primary + ball-specific)
  python script.py --models model1.pt model2.pt --video video.mp4
        """
    )
    
    parser.add_argument(
        "--models", 
        type=str, 
        nargs='+', 
        required=True, 
        help="Path(s) to YOLO model(s). First model is primary, second is ball-specific backup."
    )
    parser.add_argument(
        "--video", 
        type=str, 
        required=True, 
        help="Path to the input video"
    )
    
    args = parser.parse_args()
    
    # Validate model files
    for model_path in args.models:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Validate video file
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    run_tracker(args.models, args.video)