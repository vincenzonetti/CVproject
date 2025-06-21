import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from scipy.spatial.distance import euclidean

class StereoTracker:
    def __init__(self, model_players_path, model_ball_path, video1_path, video2_path, 
                 cam1_params_path, cam2_params_path, corners1_path, corners2_path):
        self.model_players = YOLO(model_players_path)
        self.model_ball = YOLO(model_ball_path)
        self.video1_path = video1_path
        self.video2_path = video2_path
        
        # Load camera parameters
        self.cam1_params = self.load_camera_params(cam1_params_path)
        self.cam2_params = self.load_camera_params(cam2_params_path)
        
        # Load corner correspondences
        self.corners1 = self.load_corners(corners1_path)
        self.corners2 = self.load_corners(corners2_path)
        
        # Compute projection matrices
        self.P1 = self.compute_projection_matrix(self.cam1_params)
        self.P2 = self.compute_projection_matrix(self.cam2_params)
        
        # Compute homography matrices for coordinate transformation
        self.H1 = self.compute_homography(self.corners1)
        self.H2 = self.compute_homography(self.corners2)
        
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
    
    def load_corners(self, corners_path):
        """Load corner correspondences from JSON file"""
        with open(corners_path, 'r') as f:
            corners = json.load(f)
        
        corners['real_corners'] = np.array(corners['real_corners'], dtype=np.float32)
        corners['img_corners'] = np.array(corners['img_corners'], dtype=np.float32)
        
        return corners
    
    def compute_homography(self, corners):
        """Compute homography matrix from court corners to image coordinates"""
        # Use only the first 4 corners for homography (need exactly 4 points)
        real_pts = corners['real_corners'][:4]
        img_pts = corners['img_corners'][:4]
        
        # Compute homography matrix
        H, _ = cv2.findHomography(real_pts[:, :2], img_pts, cv2.RANSAC)
        return H
    
    def compute_projection_matrix(self, cam_params):
        """Compute projection matrix P = K[R|t]"""
        K = cam_params['mtx']
        rvec = cam_params['rvecs']
        tvec = cam_params['tvecs']
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create [R|t] matrix
        Rt = np.hstack((R, tvec.reshape(-1, 1)))
        
        # Compute projection matrix P = K[R|t]
        P = K @ Rt
        
        return P
    
    def transform_to_court_coordinates(self, points_3d):
        """Transform 3D points from camera coordinates to court coordinates"""
        # This is a simplified transformation - you might need to adjust based on your setup
        # Using the camera parameters and known court points for transformation
        
        # For now, we'll use a basic transformation based on the camera poses
        # You may need to refine this based on your specific camera setup
        transformed_points = []
        
        for point in points_3d:
            # Apply transformation matrix (this is simplified - adjust as needed)
            # The actual transformation depends on your camera calibration setup
            transformed_point = point  # Placeholder - implement actual transformation
            transformed_points.append(transformed_point)
        
        return np.array(transformed_points)
    
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
        results = self.model_players(frame, verbose=False)[0]
        results_ball = self.model_ball(frame, verbose=False)[0]
        detections = {}
        best_by_class = {}
        
        if results.boxes is not None:
            boxes = results.boxes
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item()) + 1
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
                img_h, img_w = frame.shape[:2]

                if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                    best_by_class[cls_id] = {
                        'class_id': cls_id,
                        'track_id': cls_id,
                        'bbox': [
                            x_center / img_w,
                            y_center / img_h,
                            w / img_w,
                            h / img_h
                        ],
                        'conf': conf,
                        'center': (x_center, y_center)
                    }
        
        # Convert to object names
        classes = ['Ball', 'Red_0', 'Red_11', 'Red_12', 'Red_16', 'Red_2', 'Refree_F', 
                  'Refree_M', 'White_13', 'White_16', 'White_25', 'White_27', 'White_34']
        class_names = {id: name for id, name in enumerate(classes)}
        
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
        pt1_homo = np.array([[pt1[0]], [pt1[1]]], dtype=np.float32)
        pt2_homo = np.array([[pt2[0]], [pt2[1]]], dtype=np.float32)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pt1_homo, pt2_homo)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.flatten()
    
    def calculate_trajectory_metrics(self, tracking_3d_results):
        """Calculate comprehensive trajectory metrics for each object"""
        metrics = {}
        
        # Extract trajectories
        trajectories = {}
        for frame_key, frame_data in tracking_3d_results.items():
            frame_num = int(frame_key.split('_')[1])
            for obj_name, obj_data in frame_data.items():
                if obj_name not in trajectories:
                    trajectories[obj_name] = {
                        'frames': [],
                        'positions': [],
                        'confidences': []
                    }
                trajectories[obj_name]['frames'].append(frame_num)
                trajectories[obj_name]['positions'].append(obj_data['position'])
                trajectories[obj_name]['confidences'].append(obj_data['confidence'])
        
        # Calculate metrics for each object
        for obj_name, traj in trajectories.items():
            if len(traj['positions']) < 2:
                continue
                
            positions = np.array(traj['positions'])
            frames = np.array(traj['frames'])
            
            # Basic metrics
            total_distance = 0
            velocities = []
            accelerations = []
            
            for i in range(1, len(positions)):
                # Distance between consecutive points
                dist = euclidean(positions[i-1], positions[i])
                total_distance += dist
                
                # Velocity (distance/time)
                time_diff = (frames[i] - frames[i-1]) / self.fps
                if time_diff > 0:
                    velocity = dist / time_diff
                    velocities.append(velocity)
            
            # Acceleration
            for i in range(1, len(velocities)):
                time_diff = (frames[i+1] - frames[i]) / self.fps
                if time_diff > 0:
                    acceleration = (velocities[i] - velocities[i-1]) / time_diff
                    accelerations.append(acceleration)
            
            # Court positioning
            x_positions = positions[:, 0]
            y_positions = positions[:, 1]
            z_positions = positions[:, 2]
            
            metrics[obj_name] = {
                'total_distance_traveled': float(total_distance),
                'avg_velocity': float(np.mean(velocities)) if velocities else 0,
                'max_velocity': float(np.max(velocities)) if velocities else 0,
                'avg_acceleration': float(np.mean(np.abs(accelerations))) if accelerations else 0,
                'max_acceleration': float(np.max(np.abs(accelerations))) if accelerations else 0,
                'position_stats': {
                    'x_range': [float(np.min(x_positions)), float(np.max(x_positions))],
                    'y_range': [float(np.min(y_positions)), float(np.max(y_positions))],
                    'z_range': [float(np.min(z_positions)), float(np.max(z_positions))],
                    'centroid': [float(np.mean(x_positions)), float(np.mean(y_positions)), float(np.mean(z_positions))]
                },
                'detection_stats': {
                    'total_detections': len(traj['positions']),
                    'avg_confidence': float(np.mean(traj['confidences'])),
                    'min_confidence': float(np.min(traj['confidences'])),
                    'max_confidence': float(np.max(traj['confidences']))
                },
                'temporal_stats': {
                    'first_frame': int(np.min(frames)),
                    'last_frame': int(np.max(frames)),
                    'duration_frames': int(np.max(frames) - np.min(frames)),
                    'duration_seconds': float((np.max(frames) - np.min(frames)) / self.fps)
                }
            }
        
        return metrics
    
    def draw_detections(self, frame, detections, frame_id):
        """Draw bounding boxes and labels on frame"""
        for obj_name, det in detections.items():
            if 'bbox' in det and len(det['bbox']) == 4:
                xc, yc, w, h = det['bbox']
                x1 = int((xc - w / 2))
                y1 = int((yc - h / 2))
                x2 = int((xc + w / 2))
                y2 = int((yc + h / 2))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj_name}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def create_interactive_visualization(self, tracking_3d_results, output_dir):
        """Create interactive 3D visualization with toggleable classes"""
        # Extract trajectories
        trajectories = {}
        for frame_key, frame_data in tracking_3d_results.items():
            frame_num = int(frame_key.split('_')[1])
            for obj_name, obj_data in frame_data.items():
                if obj_name not in trajectories:
                    trajectories[obj_name] = {'frames': [], 'positions': []}
                trajectories[obj_name]['frames'].append(frame_num)
                trajectories[obj_name]['positions'].append(obj_data['position'])
        
        # Create main interactive plot
        fig = go.Figure()
        
        # Color scheme for different object types
        color_map = {
            'Ball': 'orange',
            'Red_0': 'red', 'Red_11': 'darkred', 'Red_12': 'crimson', 
            'Red_16': 'indianred', 'Red_2': 'lightcoral',
            'Refree_F': 'black', 'Refree_M': 'gray',
            'White_13': 'blue', 'White_16': 'navy', 'White_25': 'royalblue',
            'White_27': 'lightblue', 'White_34': 'steelblue'
        }
        
        for obj_name, traj in trajectories.items():
            if len(traj['positions']) > 1:
                positions = np.array(traj['positions'])
                
                # Add trajectory line
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='lines+markers',
                    name=obj_name,
                    line=dict(
                        color=color_map.get(obj_name, 'purple'),
                        width=3
                    ),
                    marker=dict(
                        size=3,
                        color=color_map.get(obj_name, 'purple')
                    ),
                    hovertemplate=f'<b>{obj_name}</b><br>' +
                                 'X: %{x:.2f}<br>' +
                                 'Y: %{y:.2f}<br>' +
                                 'Z: %{z:.2f}<br>' +
                                 '<extra></extra>'
                ))
                
                # Add start and end markers
                fig.add_trace(go.Scatter3d(
                    x=[positions[0, 0]],
                    y=[positions[0, 1]],
                    z=[positions[0, 2]],
                    mode='markers',
                    name=f'{obj_name}_start',
                    marker=dict(
                        size=8,
                        color=color_map.get(obj_name, 'purple'),
                        symbol='circle',
                        line=dict(width=2, color='black')
                    ),
                    showlegend=False,
                    hovertemplate=f'<b>{obj_name} Start</b><br>' +
                                 'X: %{x:.2f}<br>' +
                                 'Y: %{y:.2f}<br>' +
                                 'Z: %{z:.2f}<br>' +
                                 '<extra></extra>'
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=[positions[-1, 0]],
                    y=[positions[-1, 1]],
                    z=[positions[-1, 2]],
                    mode='markers',
                    name=f'{obj_name}_end',
                    marker=dict(
                        size=8,
                        color=color_map.get(obj_name, 'purple'),
                        symbol='square',
                        line=dict(width=2, color='black')
                    ),
                    showlegend=False,
                    hovertemplate=f'<b>{obj_name} End</b><br>' +
                                 'X: %{x:.2f}<br>' +
                                 'Y: %{y:.2f}<br>' +
                                 'Z: %{z:.2f}<br>' +
                                 '<extra></extra>'
                ))
        
        # Add court reference (using corner points)
        court_corners = self.corners1['real_corners']
        fig.add_trace(go.Scatter3d(
            x=court_corners[:, 0],
            y=court_corners[:, 1],
            z=court_corners[:, 2],
            mode='markers',
            name='Court_Corners',
            marker=dict(
                size=6,
                color='green',
                symbol='diamond'
            ),
            hovertemplate='<b>Court Corner</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='Interactive 3D Object Trajectories',
            scene=dict(
                xaxis_title='X (Court Coordinates)',
                yaxis_title='Y (Court Coordinates)',
                zaxis_title='Z (Court Coordinates)',
                aspectmode='cube'
            ),
            width=1200,
            height=800,
            hovermode='closest'
        )
        
        # Save interactive plot
        interactive_path = os.path.join(output_dir, "interactive_3d_trajectories.html")
        fig.write_html(interactive_path)
        print(f"Interactive 3D visualization saved to {interactive_path}")
        
        # Create individual class visualizations
        self.create_individual_class_plots(trajectories, color_map, output_dir)
        
        return fig
    
    def create_individual_class_plots(self, trajectories, color_map, output_dir):
        """Create separate plots for each class"""
        individual_dir = os.path.join(output_dir, "individual_trajectories")
        os.makedirs(individual_dir, exist_ok=True)
        
        for obj_name, traj in trajectories.items():
            if len(traj['positions']) > 1:
                fig = go.Figure()
                positions = np.array(traj['positions'])
                
                # Add trajectory
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='lines+markers',
                    name=obj_name,
                    line=dict(
                        color=color_map.get(obj_name, 'purple'),
                        width=4
                    ),
                    marker=dict(
                        size=4,
                        color=color_map.get(obj_name, 'purple')
                    )
                ))
                
                # Add court reference
                court_corners = self.corners1['real_corners']
                fig.add_trace(go.Scatter3d(
                    x=court_corners[:, 0],
                    y=court_corners[:, 1],
                    z=court_corners[:, 2],
                    mode='markers',
                    name='Court_Corners',
                    marker=dict(
                        size=4,
                        color='green',
                        symbol='diamond',
                        opacity=0.5
                    )
                ))
                
                fig.update_layout(
                    title=f'{obj_name} Trajectory',
                    scene=dict(
                        xaxis_title='X (Court Coordinates)',
                        yaxis_title='Y (Court Coordinates)',
                        zaxis_title='Z (Court Coordinates)',
                        aspectmode='cube'
                    ),
                    width=800,
                    height=600
                )
                
                # Save individual plot
                individual_path = os.path.join(individual_dir, f"{obj_name}_trajectory.html")
                fig.write_html(individual_path)
        
        print(f"Individual trajectory plots saved to {individual_dir}")
    
    def run_tracking(self, output_3d=None):
        """Main tracking loop with triangulation and enhanced outputs"""
        # Setup output directories
        video1_name = os.path.splitext(os.path.basename(self.video1_path))[0]
        video2_name = os.path.splitext(os.path.basename(self.video2_path))[0]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", f"stereo_{video1_name}_{video2_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Output paths
        json_2d_path = os.path.join(output_dir, "tracking_2d_results.json")
        json_3d_path = os.path.join(output_dir, "tracking_3d_results.json")
        metrics_path = os.path.join(output_dir, "trajectory_metrics.json")
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
        pbar = tqdm(total=length, desc="Processing frames")
        
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
        
        pbar.close()
        
        # Cleanup
        self.cap1.release()
        self.cap2.release()
        out_video1.release()
        out_video2.release()
        
        # Calculate trajectory metrics
        print("Calculating trajectory metrics...")
        trajectory_metrics = self.calculate_trajectory_metrics(tracking_3d_results)
        
        # Save results
        with open(json_2d_path, "w") as f:
            json.dump(tracking_2d_results, f, indent=2)
        
        with open(json_3d_path, "w") as f:
            json.dump(tracking_3d_results, f, indent=2)
        
        with open(metrics_path, "w") as f:
            json.dump(trajectory_metrics, f, indent=2)
        
        print(f"Tracking complete!")
        print(f"2D results saved to {json_2d_path}")
        print(f"3D results saved to {json_3d_path}")
        print(f"Trajectory metrics saved to {metrics_path}")
        print(f"Annotated videos saved to {video1_output_path} and {video2_output_path}")
        
        # Create interactive visualizations
        self.create_interactive_visualization(tracking_3d_results, output_dir)
        
        # Optionally save 3D data in CSV format
        if output_3d:
            self.save_3d_trajectories(tracking_3d_results, output_3d)
        
        return tracking_3d_results, trajectory_metrics
    
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

def main():
    parser = argparse.ArgumentParser(description="Run stereo YOLO tracking with 3D triangulation")
    
    # Video inputs
    parser.add_argument("--video1", type=str, required=True, help="Path to left camera video")
    parser.add_argument("--video2", type=str, required=True, help="Path to right camera video")
    
    # Model paths
    parser.add_argument("--modelP", type=str, required=True, help="Path to YOLO model for players")
    parser.add_argument("--modelB", type=str, required=True, help="Path to YOLO model for the ball")
    
    # Camera parameters
    parser.add_argument("--camparams1", type=str, required=True, help="Path to camera 1 parameters JSON")
    parser.add_argument("--camparams2", type=str, required=True, help="Path to camera 2 parameters JSON")
    
    # Corner correspondences
    parser.add_argument("--corners1", type=str, required=True, help="Path to camera 1 corner correspondences JSON")
    parser.add_argument("--corners2", type=str, required=True, help="Path to camera 2 corner correspondences JSON")
    
    # Optional outputs
    parser.add_argument("--output_3d", type=str, help="Path to save 3D trajectories CSV")
    parser.add_argument("--visualize", action="store_true", help="Generate 3D trajectory visualization")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = StereoTracker(
        args.modelP, args.modelB, args.video1, args.video2, 
        args.camparams1, args.camparams2, args.corners1, args.corners2
    )
    
    # Run tracking
    tracking_3d_results, trajectory_metrics = tracker.run_tracking(args.output_3d)
    
    # Print summary metrics
    print("\n" + "="*50)
    print("TRAJECTORY METRICS SUMMARY")
    print("="*50)
    
    for obj_name, metrics in trajectory_metrics.items():
        print(f"\n{obj_name}:")
        print(f"  Total Distance: {metrics['total_distance_traveled']:.2f} units")
        print(f"  Average Velocity: {metrics['avg_velocity']:.2f} units/sec")
        print(f"  Max Velocity: {metrics['max_velocity']:.2f} units/sec")
        print(f"  Duration: {metrics['temporal_stats']['duration_seconds']:.2f} seconds")
        print(f"  Detection Count: {metrics['detection_stats']['total_detections']}")
        print(f"  Average Confidence: {metrics['detection_stats']['avg_confidence']:.3f}")

if __name__ == "__main__":
    main()