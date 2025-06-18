import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- Configuration for ROI Search ---
# The size (in pixels) of the search window around the last known ball position.
# A larger value is more robust to fast movements but is computationally slower.
ROI_SEARCH_AREA_SIZE = 300

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions (normalized coordinates)"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def find_best_ball_with_temporal_consistency(detections, prev_ball_pos, max_distance_threshold=0.1):
    """
    Find the best ball detection considering temporal consistency.
    
    Args:
        detections: List of ball detections.
        prev_ball_pos: Previous ball position (normalized x, y) or None.
        max_distance_threshold: Maximum allowed normalized distance for a detection to be considered valid.
    
    Returns:
        The most likely ball detection or None.
    """
    if not detections:
        return None
    
    # If no previous position is known, return the detection with the highest confidence.
    if prev_ball_pos is None:
        return max(detections, key=lambda x: x['conf'])
    
    # Filter detections by proximity to the previous position.
    valid_detections = []
    for det in detections:
        ball_pos = (det['bbox'][0], det['bbox'][1])  # Normalized (x_center, y_center)
        distance = calculate_distance(ball_pos, prev_ball_pos)
        if distance <= max_distance_threshold:
            valid_detections.append((det, distance))
    
    if valid_detections:
        # Prioritize proximity, then confidence.
        # Weighting: 70% for proximity, 30% for confidence.
        best_det = min(valid_detections, 
                      key=lambda x: (x[1] * 0.7) - (x[0]['conf'] * 0.3))
        return best_det[0]
    else:
        # If no detections are close, it may indicate rapid movement.
        # In this case, fall back to the highest confidence detection.
        return max(detections, key=lambda x: x['conf'])

def run_tracker(model_path: str, video_path: str):
    """
    Runs the main tracking loop with standard YOLO and a targeted SAHI fallback for ball detection.
    """
    model = YOLO(model_path)
    
    # Initialize SAHI model for high-detail ball detection.
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda:0" if model.device.type == 'cuda' else "cpu"
    )
    
    cap = cv2.VideoCapture(video_path)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_classid_sahi_roi")
    os.makedirs(output_dir, exist_ok=True)
    
    json_output_path = os.path.join(output_dir, "tracking_results.json")
    video_output_path = os.path.join(output_dir, "tracked_video.mp4")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    tracking_results = {}
    frame_idx = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length, desc="Processing Video")
    
    prev_ball_pos = None
    sahi_used_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- Standard YOLO Detection ---
        results = model(frame, verbose=False)[0]
        detections = []
        ball_detected_standard = False
        
        if results.boxes is not None:
            best_by_class = {}
            for box in results.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x_center, y_center, w, h = box.xywhn.tolist()[0]
                
                if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                    best_by_class[cls_id] = {
                        'class_id': cls_id, 'track_id': cls_id,
                        'bbox': [x_center, y_center, w, h], 'conf': conf
                    }
            
            if 0 in best_by_class: # Class 0 is the ball
                ball_detected_standard = True
                prev_ball_pos = (best_by_class[0]['bbox'][0], best_by_class[0]['bbox'][1])
            
            detections.extend(best_by_class.values())

        # --- SAHI Fallback Logic ---
        if not ball_detected_standard:
            sahi_ball_detections = []
            roi_coords = None

            try:
                # If a previous position exists, create a focused ROI.
                if prev_ball_pos:
                    center_x_px = int(prev_ball_pos[0] * width)
                    center_y_px = int(prev_ball_pos[1] * height)
                    half_size = ROI_SEARCH_AREA_SIZE // 2

                    # Define ROI boundaries, ensuring they are within the frame.
                    x1 = max(0, center_x_px - half_size)
                    y1 = max(0, center_y_px - half_size)
                    x2 = min(width, center_x_px + half_size)
                    y2 = min(height, center_y_px + half_size)
                    
                    # Crop the frame to the ROI if the area is valid.
                    if x1 < x2 and y1 < y2:
                        roi_frame = frame[y1:y2, x1:x2]
                        roi_coords = (x1, y1)
                        prediction_source = roi_frame
                    else:
                        prediction_source = frame # Fallback to full frame if ROI is invalid
                else:
                    # If no previous position, scan the whole frame.
                    prediction_source = frame
                
                # Perform sliced prediction on the selected area (ROI or full frame).
                slice_prediction = get_sliced_prediction(
                    prediction_source, sahi_model,
                    slice_height=256, slice_width=256, # Smaller slices for detailed search
                    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                    perform_standard_pred=False
                )
                
                # Process SAHI results.
                source_h, source_w = prediction_source.shape[:2]
                for pred in slice_prediction.object_prediction_list:
                    if pred.category.id == 0: # Ball class
                        bbox = pred.bbox
                        # Calculate center and dimensions relative to the source of prediction
                        x_center_rel = (bbox.minx + bbox.maxx) / (2 * source_w)
                        y_center_rel = (bbox.miny + bbox.maxy) / (2 * source_h)
                        w_rel = (bbox.maxx - bbox.minx) / source_w
                        h_rel = (bbox.maxy - bbox.miny) / source_h

                        # Translate coordinates if prediction was on an ROI.
                        if roi_coords:
                            x_center_abs = roi_coords[0] + x_center_rel * source_w
                            y_center_abs = roi_coords[1] + y_center_rel * source_h
                            w_abs = w_rel * source_w
                            h_abs = h_rel * source_h
                        else:
                            x_center_abs = x_center_rel * width
                            y_center_abs = y_center_rel * height
                            w_abs = w_rel * width
                            h_abs = h_rel * height
                        
                        # Normalize coordinates to the full frame size.
                        sahi_ball_detections.append({
                            'class_id': 0, 'track_id': 0,
                            'bbox': [x_center_abs / width, y_center_abs / height, w_abs / width, h_abs / height],
                            'conf': pred.score.value
                        })

                if sahi_ball_detections:
                    best_ball = find_best_ball_with_temporal_consistency(
                        sahi_ball_detections, prev_ball_pos, max_distance_threshold=0.2 # Higher threshold for SAHI
                    )
                    if best_ball:
                        detections = [det for det in detections if det['class_id'] != 0]
                        detections.append(best_ball)
                        prev_ball_pos = (best_ball['bbox'][0], best_ball['bbox'][1])
                        sahi_used_count += 1
                        
            except Exception as e:
                print(f"SAHI detection failed for frame {frame_idx}: {e}")
        
        # --- Visualization and Output ---
        for det in detections:
            xc, yc, bw, bh = det['bbox']
            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)
            
            color = (0, 0, 255) if det['class_id'] == 0 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{model.names[det['class_id']]} ({det['conf']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        tracking_results[f"{video_name}_{frame_idx}"] = detections
        out_video.write(frame)
        pbar.update(1)
        frame_idx += 1
    
    cap.release()
    out_video.release()
    pbar.close()
    
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f, indent=4)
    
    print("\n--- Tracking Complete ---")
    if frame_idx > 0:
        sahi_percentage = (sahi_used_count / frame_idx) * 100
        print(f"Targeted SAHI was used for ball detection in {sahi_used_count} frames ({sahi_percentage:.1f}%).")
    print(f"Results saved to: {json_output_path}")
    print(f"Annotated video saved to: {video_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO tracking with a focused SAHI fallback for ball detection.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLOv8 model file (.pt).")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    args = parser.parse_args()
    
    run_tracker(args.model, args.video)