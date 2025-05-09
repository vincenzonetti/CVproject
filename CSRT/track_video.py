import os
import cv2
import yaml
from pathlib import Path
import glob
from utils import *
import json
class PlayerTracker:
    def __init__(self, config_path):
        """Initialize the player tracker with configuration file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_dir = Path(os.path.dirname(config_path))
        self.image_dir = self.data_dir / 'train' / 'images'
        self.label_dir = self.data_dir / 'train' / 'labels'

        # Load all images and sort them
        self.image_paths = sorted(glob.glob(str(self.image_dir / 'out2*.jpg')))
        self.label_paths = sorted(glob.glob(str(self.label_dir / 'out2*.txt')))
        print(f"Found {len(self.image_paths)} images")
        self.image_nums = len(self.image_paths)
        
        # Assume all images have same width and height
        self.height, self.width = cv2.imread(self.image_paths[0]).shape[:2]
        
        # Tracking parameters
        self.annotation_fps = 5  # Annotations are provided at 5 fps
        self.native_fps = 25  # Need to track at 25 fps
        self.frame_step = self.native_fps // self.annotation_fps
        self.tracking_results = {} # Store tracking results for all frames
        self.tracker = cv2.legacy.TrackerCSRT_create
        
        # IOU threshold for removing trackers
        self.iou_threshold = 0.1
        
        # Store IOU scores
        self.iou_scores = {}

    def process_video(self, filename):
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8).tolist()
        """Process a sequence of frames between two annotation points"""
        video_path = os.path.join('../dataset/video', filename)
        cap = cv2.VideoCapture(video_path)
        
        # Get original video dimensions for rescaling at the end
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join('../dataset/output', f'tracked_{filename}')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
        
        trackers = []
        total_boxes = []
        frame_idx = 0
        ious = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
                
            predicted_boxes = []
            activeTrackers = []
            
            frame = cv2.resize(frame, (self.width, self.height))
            display_frame = frame.copy()  # Create a copy for visualization

            if frame_idx == 0:
                label_path = self.label_paths[frame_idx]
                # Read detections from the first frame
                detections = read_yolo_label(label_path)
                # Convert YOLO format to absolute coordinates
                abs_detections = [yolo_to_absolute(det['bbox'], self.width, self.height) for det in detections]
                
                # Store detections for the first frame
                self.tracking_results[frame_idx] = detections
                # Initialize trackers for each detection
                for _, (det, abs_det) in enumerate(zip(detections, abs_detections)):
                    tracker = self.tracker()
                    tracker.init(image=frame, boundingBox=tuple(abs_det))
                    trackers.append({
                        'tracker': tracker,
                        'class_id': det['class_id'],
                        'last_bbox': abs_det  # Store the last known bounding box
                    })
            
            if frame_idx > 0 and frame_idx % 5 == 0:
                current_ids = {v['class_id']: v['last_bbox'] for v in trackers}
                new_trackers = []  # Create a new list for updated trackers

                label_path = self.label_paths[frame_idx // 5]
                detections = read_yolo_label(label_path)

                detected_classes = set()  # Keep track of detected class IDs in the current frame

                for i, det in enumerate(detections):
                    detected_classes.add(det['class_id'])
                    abs_det = yolo_to_absolute(det['bbox'], img_width=self.width, img_height=self.height)

                    found_existing = False
                    for tracker_info in list(trackers):  # Iterate over a copy for safe removal
                        if tracker_info['class_id'] == det['class_id']:
                            found_existing = True
                            if calculate_iou(abs_det, tracker_info['last_bbox']) < 0:
                                # Remove the old tracker
                                trackers.remove(tracker_info)
                                # Re-initialize a new tracker
                                tracker = self.tracker()
                                tracker.init(image=frame, boundingBox=tuple(abs_det))
                                new_trackers.append({
                                    'tracker': tracker,
                                    'class_id': det['class_id'],
                                    'last_bbox': abs_det
                                })
                            else:
                                # Keep the existing tracker
                                new_trackers.append(tracker_info)
                                ious.append(calculate_iou(abs_det, tracker_info['last_bbox']))
                                tracker_info['last_bbox'] = abs_det # Update last_bbox
                            break  # Move to the next detection
                        
                    if not found_existing:
                        # Initialize a new tracker for a newly detected object
                        tracker = self.tracker()
                        tracker.init(image=frame, boundingBox=tuple(abs_det))
                        new_trackers.append({
                            'tracker': tracker,
                            'class_id': det['class_id'],
                            'last_bbox': abs_det
                        })

                # Remove trackers for objects that were not detected in the current frame
                trackers = [t for t in new_trackers if t['class_id'] in detected_classes]
                activeTrackers = list(trackers) # Update activeTrackers based on the updated trackers
                        
                    
                        
            for v in list(trackers):
                tracker = v['tracker']
                success, bbox = tracker.update(frame)
                
                if success:
                    # Convert to proper format for IOU calculation
                    current_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    last_bbox = v['last_bbox']
                    
                    # Calculate IOU between current and last bbox
                    iou = calculate_iou(current_bbox, last_bbox)
                    
                    # Store IOU score
                    if frame_idx not in self.iou_scores:
                        self.iou_scores[frame_idx] = {}
                    self.iou_scores[frame_idx][v['class_id']] = iou
                    
                    # Only keep trackers with sufficient IOU
                    if iou > self.iou_threshold:
                        predicted_boxes.append({
                            'bbox': current_bbox,
                            'class_id': v['class_id']
                        })
                        
                        # Update display frame with bounding box and label
                        color_idx = hash(v['class_id']) % len(colors)
                        x1, y1, w, h = current_bbox
                        cv2.rectangle(display_frame, (x1, y1), (x1 + w, y1 + h), colors[color_idx], 2)
                        
                        # Add label text on top of the box
                        label_text = f"{v['class_id']} (IOU: {iou:.2f})"
                        cv2.putText(display_frame, label_text, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[color_idx], 2)
                        
                        # Update last bbox and keep tracker active
                        v['last_bbox'] = current_bbox
                        activeTrackers.append(v)
                    else:
                        print(f"Tracker for {v['class_id']} removed due to low IOU: {iou}")
                else:
                    print(f"Tracker for {v['class_id']} failed")
            
            trackers = activeTrackers
            
            # Display and write output frame
            cv2.imshow(filename, display_frame)
            out.write(display_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Adjust waitKey for desired FPS, press 'q' to quit
                break
            
            if frame_idx % 5 == 0:
                total_boxes.append(predicted_boxes)
            frame_idx += 1   
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save IOU scores to file
        #iou_output_path = os.path.join('../dataset/output', f'iou_scores_{filename.split(".")[0]}.json')
        #with open(iou_output_path, 'w') as f:
        #    json.dump(self.iou_scores, f, indent=4)
        #    
        #print(f"Output video saved to {output_path}")
        #print(f"IOU scores saved to {iou_output_path}")
        
        return total_boxes,ious
# Main execution
if __name__ == "__main__":
    # Define the path to the configuration file
    config_path = "../dataset/data.yaml"

    # Create the tracker
    tracker = PlayerTracker(config_path)
    # Run tracking
    tb,ious = tracker.process_video('out2.mp4')
    
    print(sum(ious)/len(ious))