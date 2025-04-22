import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Union, Tuple

AbsoluteBox = List[int]  # [x1, y1, w, h]
YoloBoxCoords = List[float]  # [x_center, y_center, width, height]
# Define a type alias for a YOLO bounding box dictionary
YoloBoxDict = Dict[str, Union[int, YoloBoxCoords]]



def read_yolo_label(label_path:str) -> List[YoloBoxDict] :
        """Read YOLO format label file"""
        boxes: List[YoloBoxDict] = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    class_id = int(data[0])
                    x_center, y_center, width, height = map(float, data[1:5])
                    boxes.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height],  # YOLO format
                    })
        return boxes
        
def yolo_to_absolute(yolo_box: YoloBoxCoords, img_width: int, img_height: int) -> AbsoluteBox:
        """Convert YOLO format box to absolute coordinates [x1, y1, w, h]"""
        x_center, y_center, width, height = yolo_box
        
        # Convert to absolute coordinates
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        w = int(width * img_width)
        h = int(height * img_height)
        
        return [x1, y1, w, h]
    
def calculate_iou(box1: AbsoluteBox, box2 : AbsoluteBox) -> float:
        """Calculate IoU between two boxes in [x1, y1, w, h] format"""
        # Convert to [x1, y1, x2, y2] format
        box1_x2 = box1[0] + box1[2]
        box1_y2 = box1[1] + box1[3]
        box2_x2 = box2[0] + box2[2]
        box2_y2 = box2[1] + box2[3]
        
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou

def absolute_to_yolo(abs_box: AbsoluteBox, img_width: int, img_height: int) -> YoloBoxCoords:
        """Convert absolute box [x1, y1, w, h] to YOLO format"""
        x1, y1, w, h = abs_box
        
        x_center = (x1 + w/2) / img_width
        y_center = (y1 + h/2) / img_height
        width = w / img_width
        height = h / img_height
        
        return [x_center, y_center, width, height]

def associate_detections_to_trackers(detections: List[AbsoluteBox],trackers: List[AbsoluteBox],iou_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Associates detected bounding boxes with existing tracker bounding boxes
            using the Hungarian algorithm based on Intersection over Union (IoU).
        
            This function aims to match detections to the most likely corresponding
            trackers from the previous frame or time step.
        
            Args:
                detections: A list of detected bounding boxes for the current frame,
                            each in absolute format [x1, y1, w, h].
                trackers: A list of active tracker bounding boxes from the previous state,
                          each in absolute format [x1, y1, w, h].
                iou_threshold: The minimum IoU score required to consider a detection
                               and a tracker as a match.
        
            Returns:
                A tuple containing three NumPy arrays:
                - matches: An array of shape (N, 2) where each row contains the indices
                           [detection_index, tracker_index] of a successful match.
                           N is the number of successful matches.
                - unmatched_detections: An array containing the indices of detections
                                        that could not be matched to any tracker.
                - unmatched_trackers: An array containing the indices of trackers
                                      that could not be matched to any detection.
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0), dtype=int), np.arange(len(trackers))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = calculate_iou(det, trk)
        
        # Hungarian algorithm to find optimal assignment
        detection_indices, tracker_indices = linear_sum_assignment(-iou_matrix)
        
        # Filter matches with low IoU
        matches = []
        unmatched_detections = []
        unmatched_trackers = list(range(len(trackers)))
        
        for d, t in zip(detection_indices, tracker_indices):
            if iou_matrix[d, t] >= iou_threshold:
                matches.append([d, t])
                if t in unmatched_trackers:
                    unmatched_trackers.remove(t)
            else:
                unmatched_detections.append(d)
        
        for d in range(len(detections)):
            if d not in detection_indices:
                unmatched_detections.append(d)
        
        return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)