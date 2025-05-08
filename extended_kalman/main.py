import os
import cv2
import yaml
from pathlib import Path
import glob
from utils import *
from ExtendedKalman import ExtendedKalman

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
        self.label_paths = sorted(glob.glob(str(self.label_dir / '*.txt')))
        print(f"Found {len(self.image_paths)} images")
        self.image_nums = len(self.image_paths)
        
        #assume all images have same width and height
        self.height, self.width = cv2.imread(self.image_paths[0]).shape[:2]
        
        # Tracking parameters
        self.annotation_fps = 5  # Annotations are provided at 5 fps
        self.native_fps = 25  # Need to track at 25 fps
        self.frame_step = self.native_fps // self.annotation_fps
        self.tracking_results = {} # Store tracking results for all frames

    def update(self, frame_idx,trackers):
        """Update tracker with detections in the image"""
        label_path = self.label_paths[frame_idx]
        detections = read_yolo_label(label_path)
        activeTrackers = []
        current_ids = [v['class_id'] for v in trackers]

        for i,det in enumerate(detections):
            if det['class_id'] not in current_ids:
                abs_det =yolo_to_absolute(det['bbox'],img_width=self.width,img_height=self.height)
                
                tracker:ExtendedKalman = ExtendedKalman(bbox=abs_det,class_id = det['class_id'])
                activeTrackers.append({
                    'tracker': tracker,
                    'class_id': det['class_id']
                })

        predicted_boxes = []
        for v in trackers:
            tracker:ExtendedKalman = v['tracker']
            y_hat = tracker.predict()
            for det in detections:
                if det['class_id'] == v['class_id']:
                    predicted_boxes.append({
                        'bbox': absolute_to_yolo(y_hat,img_width=self.width,img_height=self.height),
                        'class_id': v['class_id']
                    })
                    y = yolo_to_absolute(det['bbox'],self.width,self.height)
                    if tracker.time_since_update > (self.native_fps // self.annotation_fps):
                        tracker.update(y)
                    activeTrackers.append({
                        'tracker': tracker,
                        'class_id': det['class_id']
                    })
                    break

        return predicted_boxes,activeTrackers

    def process_frames(self):
        """Process a sequence of frames between two annotation points"""
        start_idx = 0
        
        label_path = self.label_paths[start_idx]
        # Read detections from the first frame
        detections = read_yolo_label(label_path)
        # Convert YOLO format to absolute coordinates
        abs_detections = [yolo_to_absolute(det['bbox'], self.width, self.height) for det in detections]
        # Store detections for the first frame
        self.tracking_results[start_idx] = detections
        # Initialize trackers for each detection
        trackers = []
        for _, (det, abs_det) in enumerate(zip(detections, abs_detections)):
            tracker = ExtendedKalman(bbox=abs_det,class_id = det['class_id'])
            trackers.append({
                'tracker': tracker,
                'class_id': det['class_id']
            })
        # Track objects through subsequent frames
        for frame_idx in range(len(self.image_paths)):
            predicted_boxes,trackers = self.update(frame_idx=frame_idx,trackers=trackers)
            self.tracking_results[frame_idx] = predicted_boxes
        return trackers


# Main execution
if __name__ == "__main__":
    # Define the path to the configuration file
    config_path = "../dataset/data.yaml"

    # Create the tracker
    tracker = PlayerTracker(config_path)

    # Run tracking
    tracker.process_frames()

    # Evaluate tracking performance
    metrics = evaluate_tracking(image_paths=tracker.image_paths,label_paths=tracker.label_paths,tracking_results=tracker.tracking_results)
    print("Tracking Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Visualize tracking results
    visualize_tracking(image_paths=tracker.image_paths, tracking_results=tracker.tracking_results)