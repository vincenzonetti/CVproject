import os
import cv2
import numpy as np
import yaml
from pathlib import Path
import glob
from tqdm import tqdm
import math
from utils import *
from KalmanTracker import KalmanBoxTracker

class PlayerTracker:
    def __init__(self, config_path):
        """Initialize the player tracker with configuration file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_dir = Path(os.path.dirname(config_path))
        self.image_dir = self.data_dir / 'train' / 'images'
        self.label_dir = self.data_dir / 'train' / 'labels'

        # Load all images and sort them
        self.image_paths = sorted(glob.glob(str(self.image_dir / '*.jpg')))
        print(f"Found {len(self.image_paths)} images")

        # Tracking parameters
        self.annotation_fps = 5  # Annotations are provided at 5 fps
        self.native_fps = 25  # Need to track at 25 fps
        self.frame_step = self.native_fps // self.annotation_fps

        # Initialize tracker
        #self.tracker = cv2.legacy.TrackerCSRT_create
        self.active_trackers = {}  # Dictionary to store active trackers
        self.tracking_results = {} # Store tracking results for all frames

    def update(self, frame_idx,img_width, img_height,trackers):
        """Update tracker with detections in the image"""
        label_path = self.get_label_path(frame_idx)
        detections = read_yolo_label(label_path)

        activeTrackers = []
        current_ids = [v['class_id'] for v in trackers]

        for i,det in enumerate(detections):
            if det['class_id'] not in current_ids:
                abs_det =yolo_to_absolute(det['bbox'],img_width=img_width,img_height=img_height)
                tracker:KalmanBoxTracker = KalmanBoxTracker(bbox=abs_det,class_id = det['class_id'])
                activeTrackers.append({
                    'tracker': tracker,
                    'class_id': det['class_id']
                })

        predicted_boxes = []
        for v in trackers:
            tracker:KalmanBoxTracker = v['tracker']
            y_hat = tracker.predict()
            for det in detections:
                if det['class_id'] == v['class_id']:
                    predicted_boxes.append({
                        'bbox': absolute_to_yolo(y_hat,img_width=img_width,img_height=img_height),
                        'class_id': v['class_id']
                    })
                    y = yolo_to_absolute(det['bbox'],img_width,img_height)
                    if tracker.time_since_update > (self.native_fps // self.annotation_fps):
                        tracker.update(y)
                    activeTrackers.append({
                        'tracker': tracker,
                        'class_id': det['class_id']
                    })
                    break

        return predicted_boxes,activeTrackers


    def process_frame_sequence(self, start_idx, end_idx, current_tracks=None):
        """Process a sequence of frames between two annotation points"""
        if current_tracks is None:
            current_tracks = {}

        frame_path = self.image_paths[start_idx]
        height, width = cv2.imread(frame_path).shape[:2]
        label_path = self.get_label_path(start_idx)

        # Read detections from the first frame
        detections = read_yolo_label(label_path)

        # Convert YOLO format to absolute coordinates
        abs_detections = [yolo_to_absolute(det['bbox'], width, height) for det in detections]

        # Store detections for the first frame
        self.tracking_results[start_idx] = detections

        # Initialize trackers for each detection
        trackers = []
        for i, (det, abs_det) in enumerate(zip(detections, abs_detections)):
            # Create a unique track ID if it's a new track
            #track_id = f"track_{start_idx}_{i}"
            tracker = KalmanBoxTracker(bbox=abs_det,class_id = det['class_id'])
            trackers.append({
                'tracker': tracker,
                'class_id': det['class_id']
            })


        # Track objects through subsequent frames
        for frame_idx in range(start_idx + 1, min(end_idx + 1, len(self.image_paths))):
            frame_path = self.image_paths[frame_idx]
            height, width = cv2.imread(frame_path).shape[:2]
            predicted_boxes,trackers = self.update(frame_idx=frame_idx,img_width=width,img_height=height,trackers=trackers)
            self.tracking_results[frame_idx] = predicted_boxes


        return trackers


    def get_label_path(self, frame_idx):
        frame_path = self.image_paths[frame_idx]
        # Get corresponding label file
        label_name = os.path.basename(frame_path).replace('.jpg', '.txt')
        label_path = self.label_dir / label_name
        return label_path

    def run_tracking(self):
        """Run tracking on the entire dataset"""
        # Find all frames with annotations
        annotated_frames = []
        for i, img_path in enumerate(self.image_paths):
            label_name = os.path.basename(img_path).replace('.jpg', '.txt')
            label_path = self.label_dir / label_name
            if os.path.exists(label_path):
                annotated_frames.append(i)
        print(f"Found {len(annotated_frames)} annotated frames")

        # Process sequences between annotated frames
        active_trackers = None
        active_trackers = self.process_frame_sequence(0, len(annotated_frames), active_trackers)


    def evaluate_tracking(self, iou_threshold=0.5):
        """Evaluate tracking performance on annotated frames"""
        metrics = {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'iou': 0.0
        }

        annotated_frames = []
        for i, img_path in enumerate(self.image_paths):
            label_name = os.path.basename(img_path).replace('.jpg', '.txt')
            label_path = self.label_dir / label_name
            if os.path.exists(label_path) and i in self.tracking_results:
                annotated_frames.append(i)

        if not annotated_frames:
            print("No frames to evaluate!")
            return metrics

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_iou = 0.0
        total_detections = 0

        for frame_idx in annotated_frames:
            img_path = self.image_paths[frame_idx]
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            # Get ground truth
            label_name = os.path.basename(img_path).replace('.jpg', '.txt')
            label_path = self.label_dir / label_name
            gt_boxes = read_yolo_label(label_path)
            # Convert to absolute coordinates
            gt_abs_boxes = [yolo_to_absolute(box['bbox'], width, height) for box in gt_boxes]


            # Get tracked boxes
            tracked_boxes = self.tracking_results.get(frame_idx, [])
            tracked_abs_boxes = [yolo_to_absolute(box['bbox'], width, height) for box in tracked_boxes]

            # Calculate IoU between each GT box and tracked box
            matches = []
            for i, gt_box in enumerate(gt_abs_boxes):
                best_iou = 0
                best_match = -1
                for j, tracked_box in enumerate(tracked_abs_boxes):
                    iou = calculate_iou(gt_box, tracked_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j

                if best_iou >= iou_threshold:
                    matches.append((i, best_match, best_iou))
                    total_iou += best_iou
                    total_detections += 1


            # Calculate TP, FP, FN
            tp = len(matches)
            fp = len(tracked_abs_boxes) - tp
            fn = len(gt_abs_boxes) - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn


        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = total_iou / total_detections if total_detections > 0 else 0


        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_iou': avg_iou,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }

        return metrics

    def visualize_tracking(self, output_dir='tracking_results'):
        """Visualize tracking results"""
        os.makedirs(output_dir, exist_ok=True)

        # Define colors for visualization
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8).tolist()

        for frame_idx, img_path in tqdm(enumerate(self.image_paths), total=len(self.image_paths)):
            if frame_idx not in self.tracking_results:
                continue

            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            for detection in self.tracking_results[frame_idx]:
                # Extract tracking info
                bbox = detection['bbox']
                class_id = detection['class_id']

                # Convert to absolute coordinates
                abs_bbox = yolo_to_absolute(bbox, width, height)
                x1, y1, w, h = abs_bbox


                # Get color based on track ID
                color_idx = hash(class_id) % len(colors)
                color = colors[color_idx]


                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)

                # Draw label
                label = f"C:{class_id}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            # Save visualization
            output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(output_path, img)

        print(f"Saved visualization to {output_dir}/")


# Main execution
if __name__ == "__main__":
    # Define the path to the configuration file
    config_path = "../out2/data.yaml"

    # Create the tracker
    tracker = PlayerTracker(config_path)

    # Run tracking
    tracker.run_tracking()

    # Evaluate tracking performance
    metrics = tracker.evaluate_tracking()
    print("Tracking Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Visualize tracking results
    tracker.visualize_tracking()