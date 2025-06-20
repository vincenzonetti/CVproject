import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

class StereoTracker:
    def __init__(self, model_players_path, model_ball_path,video1_path, video2_path, cam1_params_path, cam2_params_path):
        self.model_players = YOLO(model_players_path)
        self.model_ball = YOLO(model_ball_path)
        self.video1_path = video1_path
        self.video2_path = video2_path
        
        # Load camera parameters
        self.cam1_params = self.load_camera_params(cam1_params_path)
        self.cam2_params = self.load_camera_params(cam2_params_path)
        
        # Compute projection matrices
        self.P1 = self.compute_projection_matrix(self.cam1_params)
        self.P2 = self.compute_projection_matrix(self.cam2_params)
        
        # Initialize video captures
        self.cap1 = cv2.VideoCapture(video1_path)
        self.cap2 = cv2.VideoCapture(video2_path)
        
        # Verify both videos have same frame count and FPS
        self.verify_video_sync()
        
    def load_camera_params(self, params_path):
        """Load camera parameters from JSON file"""
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Convert lists to numpy arrays
        params['mtx'] = np.array(params['mtx'])
        params['dist'] = np.array(params['dist'])
        params['rvecs'] = np.array(params['rvecs'])
        params['tvecs'] = np.array(params['tvecs'])
        
        return params
    
    def compute_projection_matrix(self, cam_params):
        """Compute projection matrix P = K[R|t], maps 3D points into 2D one"""
        K = cam_params['mtx']  # Intrinsic matrix
        rvec = cam_params['rvecs']
        tvec = cam_params['tvecs']
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create [R|t] matrix
        Rt = np.hstack((R, tvec.reshape(-1, 1)))
        
        # Compute projection matrix P = K[R|t]
        P = K @ Rt
        
        return P
    
    def verify_video_sync(self):
        """Verify that both videos have the same properties"""
        fps1 = self.cap1.get(cv2.CAP_PROP_FPS)
        fps2 = self.cap2.get(cv2.CAP_PROP_FPS)
        
        frame_count1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if abs(fps1 - fps2) > 0.1:
            print(f"Warning: FPS mismatch - Video1: {fps1}, Video2: {fps2}")
        
        if abs(frame_count1 - frame_count2) > 1:
            print(f"Warning: Frame count mismatch - Video1: {frame_count1}, Video2: {frame_count2}")
        
        self.fps = fps1
        self.frame_count = min(frame_count1, frame_count2)
    
    def detect_objects(self, frame):
        """Detect objects in frame and return best detection per class"""
        results = self.model_players(frame,verbose = False)[0]
        results_ball = self.model_ball(frame,verbose=False)[0]
        detections = {}
        
        if results.boxes is not None:
            boxes = results.boxes
            best_by_class = {}
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())+1
                conf = float(boxes.conf[i].item())
                x_center, y_center, w, h = boxes.xywh[i].tolist()
                
                if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                    best_by_class[cls_id] = {
                        'class_id': cls_id,
                        'center': (x_center, y_center),
                        'bbox': (x_center, y_center, w, h),
                        'conf': conf
                    }
            
        if results_ball.boxes is not None:
            boxes = results_ball.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x_center, y_center, w, h = boxes.xywh[i].tolist()
                if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                    best_by_class[cls_id] = {
                        'class_id': cls_id,
                        'center': (x_center, y_center),
                        'bbox': (x_center, y_center, w, h),
                        'conf': conf
                    }
        # Convert to object names (customize based on your YOLO classes)
        classes =['Ball', 'Red_0', 'Red_11', 'Red_12', 'Red_16', 'Red_2', 'Refree_F', 'Refree_M', 'White_13', 'White_16', 'White_25', 'White_27', 'White_34']
        class_names = {id:name for id,name in enumerate(classes)}
        
        for cls_id, det in best_by_class.items():
            obj_name = class_names.get(cls_id, f'object_{cls_id}')
            detections[obj_name] = det
        
        return detections
    
    def match_objects(self, det1, det2):
        """Match objects between two views based on class similarity"""
        matches = {}
        
        for obj_name in det1:
            if obj_name in det2:
                matches[obj_name] = {
                    'pt1': det1[obj_name]['center'],
                    'pt2': det2[obj_name]['center'],
                    'conf1': det1[obj_name]['conf'],
                    'conf2': det2[obj_name]['conf']
                }
        
        return matches
    
    def triangulate_point(self, pt1, pt2):
        """Triangulate 3D point from 2D correspondences"""
        # Convert points to homogeneous coordinates for triangulation
        pt1_homo = np.array([[pt1[0]], [pt1[1]]], dtype=np.float32)
        pt2_homo = np.array([[pt2[0]], [pt2[1]]], dtype=np.float32)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pt1_homo, pt2_homo)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.flatten()
    
    def draw_detections(self, frame, detections, frame_id):
        """Draw bounding boxes and labels on frame"""
        img_h, img_w = frame.shape[:2]
        
        for obj_name, det in detections.items():
            xc, yc, w, h = det['bbox']
            x1 = int((xc - w / 2))
            y1 = int((yc - h / 2))
            x2 = int((xc + w / 2))
            y2 = int((yc + h / 2))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{obj_name}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def run_tracking(self, output_3d=None):
        """Main tracking loop with triangulation"""
        # Setup output directories
        video1_name = os.path.splitext(os.path.basename(self.video1_path))[0]
        video2_name = os.path.splitext(os.path.basename(self.video2_path))[0]
        
        
        output_dir = os.path.join("outputs", f"stereo_{video1_name}_{video2_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Output paths
        json_2d_path = os.path.join(output_dir, "tracking_2d_results.json")
        json_3d_path = os.path.join(output_dir, "tracking_3d_results.json")
        video1_output_path = os.path.join(output_dir, "tracked_video1.mp4")
        video2_output_path = os.path.join(output_dir, "tracked_video2.mp4")
        
        # Get video properties
        width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video1 = cv2.VideoWriter(video1_output_path, fourcc, self.fps, (width1, height1))
        out_video2 = cv2.VideoWriter(video2_output_path, fourcc, self.fps, (width2, height2))
        
        # Results storage
        tracking_2d_results = {}
        tracking_3d_results = {}
        
        frame_idx = 0
        length = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=length)
        while self.cap1.isOpened() and self.cap2.isOpened():
            pbar.update(1)
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Detect objects in both frames
            detections1 = self.detect_objects(frame1)
            detections2 = self.detect_objects(frame2)
            
            # Match objects between views
            matches = self.match_objects(detections1, detections2)
            # Triangulate 3D positions
            frame_3d = {}
            for obj_name, match in matches.items():
                try:
                    pos_3d = self.triangulate_point(match['pt1'], match['pt2'])
                    frame_3d[obj_name] = {
                        'position': pos_3d.tolist(),
                        'confidence': (match['conf1'] + match['conf2']) / 2
                    }
                except Exception as e:
                    print(f"Triangulation failed for {obj_name} at frame {frame_idx}: {e}")
            
            # Store results
            frame_key = f"frame_{frame_idx}"
            tracking_2d_results[frame_key] = {
                'view1': detections1,
                'view2': detections2,
                'matches': matches
            }
            tracking_3d_results[frame_key] = frame_3d
            
            # Draw detections on frames
            frame1_annotated = self.draw_detections(frame1.copy(), detections1, frame_idx)
            frame2_annotated = self.draw_detections(frame2.copy(), detections2, frame_idx)
            
            # Write annotated frames
            out_video1.write(frame1_annotated)
            out_video2.write(frame2_annotated)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")
        
        pbar.close()
        # Cleanup
        self.cap1.release()
        self.cap2.release()
        out_video1.release()
        out_video2.release()
        
        # Save results
        with open(json_2d_path, "w") as f:
            json.dump(tracking_2d_results, f, indent=2)
        
        with open(json_3d_path, "w") as f:
            json.dump(tracking_3d_results, f, indent=2)
        
        print(f"Tracking complete!")
        print(f"2D results saved to {json_2d_path}")
        print(f"3D results saved to {json_3d_path}")
        print(f"Annotated videos saved to {video1_output_path} and {video2_output_path}")
        
        # Optionally save 3D data in specific format
        if output_3d:
            self.save_3d_trajectories(tracking_3d_results, output_3d)
        
        return tracking_3d_results
    
    def save_3d_trajectories(self, tracking_3d_results, output_path):
        """Save 3D trajectories in CSV format"""
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'object', 'x', 'y', 'z', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_key, frame_data in tracking_3d_results.items():
                frame_num = int(frame_key.split('_')[1])
                for obj_name, obj_data in frame_data.items():
                    pos = obj_data['position']
                    writer.writerow({
                        'frame': frame_num,
                        'object': obj_name,
                        'x': pos[0],
                        'y': pos[1],
                        'z': pos[2],
                        'confidence': obj_data['confidence']
                    })
        
        print(f"3D trajectories saved to {output_path}")
    
    def visualize_3d_trajectories(self, tracking_3d_results, save_path=None):
        """Visualize 3D trajectories using matplotlib"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract trajectories for each object
        trajectories = {}
        for frame_key, frame_data in tracking_3d_results.items():
            frame_num = int(frame_key.split('_')[1])
            for obj_name, obj_data in frame_data.items():
                if obj_name not in trajectories:
                    trajectories[obj_name] = {'frames': [], 'positions': []}
                trajectories[obj_name]['frames'].append(frame_num)
                trajectories[obj_name]['positions'].append(obj_data['position'])
        
        # Plot trajectories
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (obj_name, traj) in enumerate(trajectories.items()):
            positions = np.array(traj['positions'])
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   color=colors[i % len(colors)], label=obj_name, linewidth=2)
            
            # Mark start and end points
            if len(positions) > 0:
                ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                          color=colors[i % len(colors)], s=100, marker='o')
                ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                          color=colors[i % len(colors)], s=100, marker='s')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Object Trajectories')
        ax.legend()
        
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D visualization saved to {save_path}_{current_timestamp}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run stereo YOLO tracking with 3D triangulation")
    
    # Video inputs
    parser.add_argument("--video1", type=str, required=True, help="Path to left camera video")
    parser.add_argument("--video2", type=str, required=True, help="Path to right camera video")
    
    # Model
    parser.add_argument("--modelP", type=str, required=True, help="Path to YOLO model for players")
    parser.add_argument("--modelB", type=str, required=True, help="Path to YOLO model for the ball")
    
    # Camera parameters
    parser.add_argument("--camparams1", type=str, required=True, help="Path to camera 1 parameters JSON")
    parser.add_argument("--camparams2", type=str, required=True, help="Path to camera 2 parameters JSON")
    
    # Optional outputs
    parser.add_argument("--output_3d", type=str, help="Path to save 3D trajectories CSV")
    parser.add_argument("--visualize", action="store_true", help="Generate 3D trajectory visualization")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = StereoTracker(
        args.modelP,args.modelB, args.video1, args.video2, args.camparams1, args.camparams2
    )
    
    # Run tracking
    tracking_3d_results = tracker.run_tracking(args.output_3d)
    
    # Optional visualization
    if args.visualize:
        output_dir = os.path.join("outputs", f"stereo_{os.path.splitext(os.path.basename(args.video1))[0]}_{os.path.splitext(os.path.basename(args.video2))[0]}")
        viz_path = os.path.join(output_dir, "3d_trajectories.png")
        tracker.visualize_3d_trajectories(tracking_3d_results, viz_path)

if __name__ == "__main__":
    main()