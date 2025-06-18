import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from ultralytics.engine.results import Results


colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (165, 42, 42),  # Brown
    (0, 0, 0),      # Black
    (255, 255, 255),# White
    (128, 128, 128),# Gray
    (0, 128, 0)     # Dark Green
]


def run_tracker(model_path_players: str,model_path_ball: str, video_path: str):
    model_player = YOLO(model_path_players)
    model_ball = YOLO(model_path_ball)
    cap = cv2.VideoCapture(video_path)

    # Extract video name for output and frame keys
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path_ball))[0]

    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_classid")
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

    while cap.isOpened():
        pbar.update(1)
        
        ret, frame = cap.read()
        if not ret:
            break

        results = model_player(frame,verbose=False)[0]
        results_ball = model_ball(frame,verbose=False)[0]
        detections = []
        best_by_class = {}
        if results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id == 0:
                    continue
                conf = float(boxes.conf[i].item())
                x_center, y_center, w, h = boxes.xywh[i].tolist()
                img_h, img_w = frame.shape[:2]

                if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                    best_by_class[cls_id] = {
                        'class_id': cls_id,
                        'track_id': cls_id,  # Same as class ID
                        'bbox': [
                            x_center / img_w,
                            y_center / img_h,
                            w / img_w,
                            h / img_h
                        ],
                        'conf': conf
                    }

        if results_ball.boxes is not None:
            boxes = results_ball.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
             
                conf = float(boxes.conf[i].item())
                x_center, y_center, w, h = boxes.xywh[i].tolist()
                img_h, img_w = frame.shape[:2]

                if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                    best_by_class[cls_id] = {
                        'class_id': cls_id,
                        'track_id': cls_id,  # Same as class ID
                        'bbox': [
                            x_center / img_w,
                            y_center / img_h,
                            w / img_w,
                            h / img_h
                        ],
                        'conf': conf
                    }
        

        for id,det in best_by_class.items():
            
            detections.append(det)
            # Draw box on frame
            xc, yc, bw, bh = det['bbox']
            x1 = int((xc - bw / 2) * img_w)
            y1 = int((yc - bh / 2) * img_h)
            x2 = int((xc + bw / 2) * img_w)
            y2 = int((yc + bh / 2) * img_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[id], 2)
            cv2.putText(frame, f"ID {det['track_id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[id], 1)
        
        frame_key = f"{video_name}_{frame_idx}"
        tracking_results[frame_key] = detections
        frame_idx += 1

        out_video.write(frame)

    cap.release()
    pbar.close()
    out_video.release()

    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f)

    print(f"Tracking complete. Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO class-ID-based tracking on a video.")
    parser.add_argument("--modelP", type=str, required=True, help="Path to the YOLO model for players")
    parser.add_argument("--modelB", type=str, required=True, help="Path to the YOLO model for the ball")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    parser.add_argument("--trackerP", type=str, required=True, help="Path to the Tracker(yaml) config for the player")
    parser.add_argument("--trackerB", type=str, required=True, help="Path to the Tracker(yaml) config for the ball")
    args = parser.parse_args()

    run_tracker(args.modelP,args.modelB, args.video)
