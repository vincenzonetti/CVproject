import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Union, Tuple
import cv2
from tqdm import tqdm
import glob
from dataclasses import dataclass
AbsoluteBox = List[int]  # [x1, y1, w, h]
YoloBoxCoords = List[float]  # [x_center, y_center, width, height]
# Define a type alias for a YOLO bounding box dictionary
YoloBoxDict = Dict[str, Union[int, YoloBoxCoords]]



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

def evaluate_tracking(iou_threshold=0.5, ground_truth_boxes = None, tracking_results = None):
        
        """Evaluate tracking performance on annotated frames"""
        assert ground_truth_boxes is not None and tracking_results is not None
        metrics = {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'iou': 0.0
        }

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_iou = 0.0
        total_detections = 0

        for frame_idx in range(len(ground_truth_boxes)):
            
            # Get ground truth
            gt_boxes = ground_truth_boxes[frame_idx]
            
            # Get tracked boxes
            tracked_boxes = tracking_results.get(frame_idx, [])
            
            # Calculate IoU between each GT box and tracked box
            matches = []
            for i, gt_box in enumerate(gt_boxes):
                best_iou = 0
                for j, tracked_box in enumerate(tracked_boxes):
                    
                    if gt_box['class_id'] == tracked_box['class_id']:
                        best_iou = calculate_iou(gt_box['bbox'], tracked_box['bbox'])
                        
                if best_iou >= iou_threshold:
                    matches.append((gt_box['class_id'],best_iou))
                   
                total_iou += best_iou
                total_detections += 1


            # Calculate TP, FP, FN
            tp = len(matches)
            fp = len(tracked_boxes) - tp
            fn = len(gt_boxes) - tp

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
    
    
def visualize_tracking(output_dir='tracking_results',image_paths = None, tracking_results = None,):
        """Visualize tracking results"""
        assert image_paths is not None and tracking_results is not None
        os.makedirs(output_dir, exist_ok=True)

        # Define colors for visualization
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8).tolist()

        for frame_idx, img_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            if frame_idx not in tracking_results:
                continue

            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            for detection in tracking_results[frame_idx]:
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


def histogram_uniformity(hist):
    # Calculate the variance of the histogram as a measure of uniformity
    return np.var(hist)

def get_ball_templates(label_dir, image_dir, video_name):
    label_files = sorted(glob.glob(f'{label_dir}/{video_name}*.txt'))
    image_files = sorted(glob.glob(f'{image_dir}/{video_name}*.jpg'))
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)

    best_template = None

    for i, file in enumerate(label_files):
        lbls = read_yolo_label(file)
        lbls = {item['class_id']: item['bbox'] for item in lbls}

        if 0 in lbls:
            x_center, y_center, width, height = lbls[0]
            frame = cv2.imread(image_files[i], cv2.IMREAD_COLOR)
            img_h, img_w = frame.shape[:2]

            x = int((x_center - width / 2) * img_w)
            y = int((y_center - height / 2) * img_h)
            w = int(width * img_w)
            h = int(height * img_h)

            ball_roi = frame[y:y+h, x:x+w]

            hsv_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            gray_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray_roi, None)

           
            current_uniformity = histogram_uniformity(hist)

            if best_template is None or current_uniformity < histogram_uniformity(best_template.histogram):
                #cv2.imshow('Window', ball_roi)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                best_template = BallTemplate(
                    image=ball_roi.copy(),
                    histogram=hist,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    bbox=(x_center*img_w, y_center*img_h, w, h),
                    confidence=1.0,
                    frame_idx=i
                )

    
        return best_template
            
                 
def read_yolo_label_mod(label_path:str) -> Dict[int,YoloBoxCoords] :
        """Read YOLO format label file"""
        boxes = {}
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    class_id = int(data[0])
                    x_center, y_center, width, height = map(float, data[1:5])
                    boxes[class_id]=[x_center, y_center, width, height]
        return boxes