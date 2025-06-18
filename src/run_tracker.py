import argparse
import os
import json
import cv2
from ultralytics import YOLO


def run_tracker(model_path: str, video_path: str, tracker_path: str):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Extract names for output folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    tracker_name = os.path.splitext(os.path.basename(tracker_path))[0]

    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_{tracker_name}")
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker=tracker_path)
        detections = []

        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                x_center, y_center, w, h = boxes.xywh[i].tolist()
                img_h, img_w = frame.shape[:2]

                # Append detection
                detections.append({
                    'class_id': cls_id,
                    'bbox': [
                        x_center / img_w,
                        y_center / img_h,
                        w / img_w,
                        h / img_h
                    ]
                })

                # Draw box on frame
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                track_id = int(boxes.id[i].item() if boxes.id[i] is not None else -1)
                # Generate a color based on track_id
                color = (
                    (37 * track_id) % 256,
                    (17 * track_id) % 256,
                    (29 * track_id) % 256
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        tracking_results[frame_idx] = detections
        frame_idx += 1

        out_video.write(frame)

    cap.release()
    out_video.release()

    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f)

    print(f"Tracking complete. Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO tracking on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Path to tracker config (e.g. botsort.yaml)")
    args = parser.parse_args()

    run_tracker(args.model, args.video, args.tracker)
