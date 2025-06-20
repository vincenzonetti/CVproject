import os
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.spatial import distance
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


CLASSES = ['Ball', 'Red_0', 'Red_11', 'Red_12', 'Red_16', 'Red_2', 'Refree_F', 'Refree_M', 
           'White_13', 'White_16', 'White_25', 'White_27', 'White_34']

IMG_WIDTH = 3840
IMG_HEIGHT = 2160

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

# Plotly colors for better visualization
plotly_colors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
    '#00FFFF', '#FFA500', '#800080', '#A52A2A', '#000000',
    '#808080', '#008000', '#FFC0CB'
]

class StereoTracker:
    def __init__(self, detection1_path, detection2_path, cam1_params_path, cam2_params_path, 
                 cam1_corners_path=None, cam2_corners_path=None, fps=25):
        
        self.det1 = self.load_detections(detection1_path)
        self.det2 = self.load_detections(detection2_path)
        # Load camera parameters
        self.cam1_params = self.load_camera_params(cam1_params_path)
        self.cam2_params = self.load_camera_params(cam2_params_path)
        
        # Compute projection matrices
        self.P1 = self.compute_projection_matrix(self.cam1_params)
        self.P2 = self.compute_projection_matrix(self.cam2_params)
        
        # Load court corners if provided
        self.court_corners_3d = None
        if cam1_corners_path and cam2_corners_path:
            self.load_court_corners(cam1_corners_path, cam2_corners_path)
        
        self.fps = fps  # Frames per second for velocity calculations
        

    def load_court_corners(self, cam1_corners_path, cam2_corners_path):
        """Load court corners for scale reference and coordinate transformation"""
        with open(cam1_corners_path, 'r') as f:
            cam1_corners = json.load(f)
        with open(cam2_corners_path, 'r') as f:
            cam2_corners = json.load(f)
        
        # Extract real world coordinates and image coordinates for each camera
        self.cam1_real_corners = np.array(cam1_corners['real_corners'])
        self.cam1_img_corners = np.array(cam1_corners['img_corners'])
        
        self.cam2_real_corners = np.array(cam2_corners['real_corners'])
        self.cam2_img_corners = np.array(cam2_corners['img_corners'])
        
        # Calculate homography for coordinate transformation
        # Each camera has its own set of corners and homography
        # Using only x,y coordinates since z=0 for court
        cam1_court_2d = self.cam1_real_corners[:, :2]
        cam2_court_2d = self.cam2_real_corners[:, :2]
        
        # Calculate homography from image to court coordinates for both cameras
        self.H1, _ = cv2.findHomography(self.cam1_img_corners, cam1_court_2d)
        self.H2, _ = cv2.findHomography(self.cam2_img_corners, cam2_court_2d)
        
        # Combine all unique court corners for visualization
        # Since cameras see different corners, we need to merge them
        all_corners = np.vstack([self.cam1_real_corners, self.cam2_real_corners])
        # Remove duplicates (corners seen by both cameras)
        unique_corners = np.unique(all_corners, axis=0)
        self.court_corners_3d = unique_corners
        
        # Basketball court is typically 28m x 15m
        max_x = np.max(unique_corners[:, 0])
        min_x = np.min(unique_corners[:, 0])
        court_width = max_x - min_x
        
        print(f"Camera 1 sees {len(self.cam1_real_corners)} corners")
        print(f"Camera 2 sees {len(self.cam2_real_corners)} corners")
        print(f"Total unique corners: {len(unique_corners)}")
        print(f"Court width detected: {court_width} meters")
        print(f"Court dimensions: X=[{min_x:.1f}, {max_x:.1f}], Y=[{np.min(unique_corners[:, 1]):.1f}, {np.max(unique_corners[:, 1]):.1f}]")
        
        # No scaling needed as real_corners are already in meters
        self.scale_to_meters = 1.0

    def load_detections(self, detection_path):
        """Load camera parameters from JSON file and format the keys."""
        with open(detection_path, 'r') as f:
            params = json.load(f)

        # Create a new dictionary with formatted keys
        formatted_params = []
        for key, value in params.items():
            # Extract the numerical part from the key and convert it to an integer
            formatted_params.append({})
            # Extract class_id and bbox
            for x in value:
                class_id = x['class_id']
                class_name = CLASSES[class_id]  # Convert to class name
                bbox = x['bbox']    
                
                # Convert normalized coordinates to pixel coordinates
                # bbox format: [x_center, y_center, width, height] (normalized)
                x_center_px = bbox[0] * IMG_WIDTH
                y_center_px = bbox[1] * IMG_HEIGHT
                
                # Add the class_name and bbox to the nested dictionary
                formatted_params[-1][class_name] = {
                    'bbox': bbox,
                    'center': [x_center_px, y_center_px]  # Now in pixel coordinates
                }

        return formatted_params

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
    
    def match_objects(self, det1, det2):
        """Match objects between two views based on class similarity"""
        matches = {}
        
        for obj_name in det1:
            if obj_name in det2:
                matches[obj_name] = {
                    'pt1': det1[obj_name]['center'],
                    'pt2': det2[obj_name]['center']
                }
        
        return matches
    
    def triangulate_point(self, pt1, pt2, obj_name):
        """Triangulate 3D point from 2D correspondences and transform to court coordinates"""
        # Convert points to homogeneous coordinates for triangulation
        pt1_homo = np.array([[pt1[0]], [pt1[1]]], dtype=np.float32)
        pt2_homo = np.array([[pt2[0]], [pt2[1]]], dtype=np.float32)
        
        # Triangulate in camera coordinate system
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pt1_homo, pt2_homo)
        
        # Convert from homogeneous to 3D coordinates
        points_3d_camera = points_4d[:3] / points_4d[3]
        
        # If we have court corners, transform to court coordinate system
        if hasattr(self, 'H1') and hasattr(self, 'H2'):
            # Method 1: Transform each camera's point to court coordinates and average
            pt1_court = cv2.perspectiveTransform(np.array([[pt1]], dtype=np.float32), self.H1)[0, 0]
            pt2_court = cv2.perspectiveTransform(np.array([[pt2]], dtype=np.float32), self.H2)[0, 0]
            
            # Average the court positions from both views for x,y
            court_xy = (pt1_court + pt2_court) / 2
            
            # For height (z), we need to transform the triangulated z from camera coordinates
            # to world coordinates using the camera extrinsics
            
            # Get rotation matrix from camera 1 (as reference)
            R1, _ = cv2.Rodrigues(self.cam1_params['rvecs'])
            t1 = self.cam1_params['tvecs'].reshape(-1, 1)
            
            # Transform the 3D point from camera coordinates to world coordinates
            # P_world = R^T * (P_camera - t)
            points_3d_world = R1.T @ (points_3d_camera - t1)
            
            # Use the world z coordinate (height above court)
            z_world = points_3d_world[2, 0]
            
            # Scale the height to reasonable values
            if abs(z_world) > 10:  # If values are unreasonably large
                z_world = z_world * 0.001  # Convert from mm to m if needed
            
            # Basketball convention: X (width), Y (height), Z (depth)
            # Original: court_xy[0] (x), court_xy[1] (y), z_world
            # New mapping: X stays X, Y becomes height, Z becomes depth
            
            # For the ball, keep the triangulated height
            if obj_name == 'Ball':
                y_coord = abs(z_world)  # Height
            else:
                y_coord = 0.0  # Players on the ground
            
            # Create final position with basketball court convention
            points_3d = np.array([court_xy[0], y_coord, court_xy[1]])  # X, Y (height), Z (depth)
        else:
            # No court transformation available, use camera coordinates
            points_3d = points_3d_camera.flatten()
            
            # Apply scale to meters if available
            if hasattr(self, 'scale_to_meters'):
                points_3d *= self.scale_to_meters
        
        return points_3d
    
    def calculate_trajectory_metrics(self, tracking_3d_results):
        """Calculate comprehensive trajectory metrics for each player"""
        metrics = {}
        
        # Extract trajectories for each object
        trajectories = {}
        for frame_key, frame_data in tracking_3d_results.items():
            frame_num = int(frame_key.split('_')[1])
            for obj_name, obj_data in frame_data.items():
                if obj_name not in trajectories:
                    trajectories[obj_name] = {'frames': [], 'positions': []}
                trajectories[obj_name]['frames'].append(frame_num)
                trajectories[obj_name]['positions'].append(obj_data['position'])
        
        # Calculate metrics for each player
        for obj_name, traj in trajectories.items():
            if len(traj['positions']) < 2:
                continue
                
            positions = np.array(traj['positions'])
            frames = np.array(traj['frames'])
            
            # Total distance traveled (considering X and Z for court movement)
            # Y is height, so we calculate ground distance separately
            ground_positions = positions[:, [0, 2]]  # X and Z only
            ground_distances = np.sqrt(np.sum(np.diff(ground_positions, axis=0)**2, axis=1))
            total_ground_distance = np.sum(ground_distances)
            
            # 3D distance (including height changes for ball)
            distances_3d = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            total_distance_3d = np.sum(distances_3d)
            
            # Average speed (m/s) - ground speed for players
            time_diffs = np.diff(frames) / self.fps  # Convert frames to seconds
            ground_speeds = ground_distances / time_diffs
            avg_ground_speed = np.mean(ground_speeds) if len(ground_speeds) > 0 else 0
            max_ground_speed = np.max(ground_speeds) if len(ground_speeds) > 0 else 0
            
            # Acceleration
            accelerations = np.diff(ground_speeds) / time_diffs[:-1] if len(ground_speeds) > 1 else []
            avg_acceleration = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0
            
            # Direction changes (using angle between consecutive movement vectors on court)
            if len(ground_positions) > 2:
                vectors = np.diff(ground_positions, axis=0)
                direction_changes = 0
                for i in range(len(vectors) - 1):
                    v1 = vectors[i]
                    v2 = vectors[i + 1]
                    # Calculate angle between vectors
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    if angle > np.pi/4:  # 45 degrees threshold
                        direction_changes += 1
            else:
                direction_changes = 0
            
            # Coverage area (bounding box on court - X-Z plane)
            min_pos = np.min(ground_positions, axis=0)
            max_pos = np.max(ground_positions, axis=0)
            coverage_area = np.prod(max_pos - min_pos)
            
            # Height statistics (Y axis)
            heights = positions[:, 1]
            avg_height = np.mean(heights)
            max_height = np.max(heights)
            height_variation = np.std(heights)
            
            # Store metrics
            metrics[obj_name] = {
                'total_ground_distance_m': float(total_ground_distance),
                'total_3d_distance_m': float(total_distance_3d),
                'avg_ground_speed_ms': float(avg_ground_speed),
                'max_ground_speed_ms': float(max_ground_speed),
                'avg_acceleration_ms2': float(avg_acceleration),
                'direction_changes': int(direction_changes),
                'coverage_area_m2': float(coverage_area),
                'avg_height_m': float(avg_height),
                'max_height_m': float(max_height),
                'height_variation_m': float(height_variation),
                'total_frames': len(frames),
                'time_tracked_s': float((frames[-1] - frames[0]) / self.fps)
            }
        
        return metrics
    
    def create_interactive_3d_plot(self, tracking_3d_results, metrics, save_path):
        """Create an interactive 3D plot using Plotly with player filtering"""
        
        # Extract trajectories for each object
        trajectories = {}
        for frame_key, frame_data in tracking_3d_results.items():
            frame_num = int(frame_key.split('_')[1])
            for obj_name, obj_data in frame_data.items():
                if obj_name not in trajectories:
                    trajectories[obj_name] = {'frames': [], 'positions': []}
                trajectories[obj_name]['frames'].append(frame_num)
                trajectories[obj_name]['positions'].append(obj_data['position'])
        
        # Debug: Print trajectory information
        print("\n=== Trajectory Debug Information ===")
        for obj_name, traj in trajectories.items():
            print(f"{obj_name}: {len(traj['frames'])} frames")
            if len(traj['positions']) > 0:
                pos = np.array(traj['positions'])
                print(f"  Position range: X[{pos[:,0].min():.2f}, {pos[:,0].max():.2f}], "
                      f"Y[{pos[:,1].min():.2f}, {pos[:,1].max():.2f}], "
                      f"Z[{pos[:,2].min():.2f}, {pos[:,2].max():.2f}]")
        
        # Create figure
        fig = go.Figure()
        
        # Add court outline if available
        if self.court_corners_3d is not None:
            court_points = self.court_corners_3d
            
            # Convert court corners to basketball convention (X, Y=0, Z)
            court_x = court_points[:, 0]  # X stays X
            court_y = np.zeros_like(court_points[:, 0])  # Y = 0 (ground level)
            court_z = court_points[:, 1]  # Original Y becomes Z (depth)
            
            # Create court outline by connecting corners
            fig.add_trace(go.Scatter3d(
                x=court_x, y=court_y, z=court_z,
                mode='lines+markers',
                name='Court Boundary',
                line=dict(color='gray', width=2),
                marker=dict(size=6, color='black'),
                showlegend=True,
                visible=True  # Court always visible
            ))
            
            # Add court surface (as a mesh at y=0)
            court_x_range = [np.min(court_points[:, 0]), np.max(court_points[:, 0])]
            court_z_range = [np.min(court_points[:, 1]), np.max(court_points[:, 1])]
            
            # Create a grid for the court surface
            xx, zz = np.meshgrid(np.linspace(court_x_range[0], court_x_range[1], 10),
                                 np.linspace(court_z_range[0], court_z_range[1], 10))
            yy = np.zeros_like(xx)  # Y = 0 for court surface
            
            fig.add_trace(go.Surface(
                x=xx, y=yy, z=zz,
                colorscale='oranges',
                opacity=0.3,
                showscale=False,
                name='Court Surface',
                showlegend=False,
                visible=True  # Court always visible
            ))
        
        # Keep track of which traces are court-related (always visible)
        num_court_traces = len(fig.data)
        
        # Add player trajectories with grouped start/end markers
        player_traces_added = []
        for i, (obj_name, traj) in enumerate(trajectories.items()):
            if len(traj['positions']) < 2:
                print(f"Skipping {obj_name}: only {len(traj['positions'])} position(s)")
                continue
                
            positions = np.array(traj['positions'])
            frames = traj['frames']
            
            # Create hover text with metrics
            hover_text = []
            for j, frame in enumerate(frames):
                text = f"Player: {obj_name}<br>"
                text += f"Frame: {frame}<br>"
                text += f"Position: ({positions[j, 0]:.2f}, {positions[j, 1]:.2f}, {positions[j, 2]:.2f}) m<br>"
                if obj_name in metrics:
                    
                    text += f"Avg Speed: {metrics[obj_name]['avg_ground_speed_ms']:.2f} m/s<br>"
                    text += f"Total Distance: {metrics[obj_name]['total_3d_distance_m']:.2f} m"
                hover_text.append(text)
            
            # Add trajectory
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],  # X (width)
                y=positions[:, 1],  # Y (height)
                z=positions[:, 2],  # Z (depth)
                mode='lines+markers',
                name=f'{obj_name}',
                line=dict(color=plotly_colors[i % len(plotly_colors)], width=4),
                marker=dict(size=3),
                text=hover_text,
                hoverinfo='text',
                legendgroup=f'player{obj_name}',
                visible=False  # Start with all players hidden
            ))
            
            # Add start marker (grouped with main trace)
            fig.add_trace(go.Scatter3d(
                x=[positions[0, 0]],
                y=[positions[0, 1]],
                z=[positions[0, 2]],
                mode='markers',
                name=f'Start {obj_name}',
                marker=dict(size=10, color=plotly_colors[i % len(plotly_colors)], symbol='circle'),
                showlegend=False,
                legendgroup=f'player{obj_name}',
                visible=False  # Start hidden
            ))
            
            # Add end marker (grouped with main trace)
            fig.add_trace(go.Scatter3d(
                x=[positions[-1, 0]],
                y=[positions[-1, 1]],
                z=[positions[-1, 2]],
                mode='markers',
                name=f'End {obj_name}',
                marker=dict(size=10, color=plotly_colors[i % len(plotly_colors)], symbol='square'),
                showlegend=False,
                legendgroup=f'player{obj_name}',
                visible=False  # Start hidden
            ))
            
            player_traces_added.append(obj_name)
        
        print(f"\nAdded traces for {len(player_traces_added)} players: {player_traces_added}")
        
        # Update layout with basketball court axis convention
        fig.update_layout(
            title={
                'text': 'Basketball Player 3D Trajectories<br><sub>Click on player names to show/hide trajectories</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis=dict(
                    title='X - Width (meters)',
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    title='Y - Height (meters)',
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    range=[0, 4]  # Limit height range
                ),
                zaxis=dict(
                    title='Z - Depth (meters)',
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                aspectmode='manual',
                aspectratio=dict(x=2, y=0.5, z=1),  # Adjust for basketball court proportions
                camera=dict(
                    eye=dict(x=0, y=1.5, z=-2.5),  # Side view
                    center=dict(x=0, y=0, z=0)
                )
            ),
            showlegend=True,
            legend=dict(
                itemsizing='constant',
                x=1.02,
                y=0.5,
                yanchor='middle',
                title=dict(text='Players (click to show/hide)')
            ),
            width=1200,
            height=800
        )
        
        # Add buttons to show/hide all players
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * num_court_traces + [True] * (len(fig.data) - num_court_traces)}],
                            label="Show All Players",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True] * num_court_traces + [False] * (len(fig.data) - num_court_traces)}],
                            label="Hide All Players",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        
        # Save the interactive plot
        fig.write_html(save_path)
        print(f"Interactive 3D plot saved to {save_path}")
        
        # Also create a metrics dashboard
        self.create_metrics_dashboard(metrics, trajectories, save_path.replace('.html', '_metrics.html'))
    
    def create_metrics_dashboard(self, metrics, trajectories, save_path):
        """Create a comprehensive metrics dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Speed Distribution', 'Distance Traveled', 
                          'Coverage Area', 'Movement Intensity'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Extract player names and metrics
        players = list(metrics.keys())
        avg_speeds = [metrics[p]['avg_ground_speed_ms'] for p in players]
        max_speeds = [metrics[p]['max_ground_speed_ms'] for p in players]
        distances = [metrics[p]['total_ground_distance_m'] for p in players]
        areas = [metrics[p]['coverage_area_m2'] for p in players]
        accelerations = [metrics[p]['avg_acceleration_ms2'] for p in players]
        direction_changes = [metrics[p]['direction_changes'] for p in players]
        
        # Speed distribution
        fig.add_trace(
            go.Bar(name='Avg Speed', x=players, y=avg_speeds, marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Max Speed', x=players, y=max_speeds, marker_color='darkblue'),
            row=1, col=1
        )
        
        # Distance traveled
        fig.add_trace(
            go.Bar(x=players, y=distances, marker_color='green', showlegend=False),
            row=1, col=2
        )
        
        # Coverage area
        fig.add_trace(
            go.Bar(x=players, y=areas, marker_color='orange', showlegend=False),
            row=2, col=1
        )
        
        # Movement intensity (acceleration vs direction changes)
        fig.add_trace(
            go.Scatter(
                x=accelerations, 
                y=direction_changes,
                mode='markers+text',
                text=players,
                textposition="top center",
                marker=dict(size=10, color='red'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Player", row=1, col=1)
        fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
        
        fig.update_xaxes(title_text="Player", row=1, col=2)
        fig.update_yaxes(title_text="Distance (m)", row=1, col=2)
        
        fig.update_xaxes(title_text="Player", row=2, col=1)
        fig.update_yaxes(title_text="Area (m²)", row=2, col=1)
        
        fig.update_xaxes(title_text="Avg Acceleration (m/s²)", row=2, col=2)
        fig.update_yaxes(title_text="Direction Changes", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Player Movement Metrics Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html(save_path)
        print(f"Metrics dashboard saved to {save_path}")
        
        # Save detailed metrics to CSV
        metrics_df = pd.DataFrame(metrics).T
        csv_path = save_path.replace('.html', '.csv')
        metrics_df.to_csv(csv_path)
        print(f"Detailed metrics saved to {csv_path}")
    
    def run_tracking(self, output_3d=None):
        """Main tracking loop with triangulation"""
        # Setup output directories
        output_dir = os.path.join("outputs", f"stereo")
        os.makedirs(output_dir, exist_ok=True)
        
        # Output paths
        json_2d_path = os.path.join(output_dir, "tracking_2d_results.json")
        json_3d_path = os.path.join(output_dir, "tracking_3d_results.json")
        
        # Results storage
        tracking_2d_results = {}
        tracking_3d_results = {}
        
        # Track which objects appear and how often
        object_appearance_count = {}
        matched_count = {}
        
        frame_idx = 0
        
        for detection1, detection2 in tqdm(zip(self.det1, self.det2)):
            # Track objects in each camera
            for obj in detection1:
                if obj not in object_appearance_count:
                    object_appearance_count[obj] = {'cam1': 0, 'cam2': 0, 'matched': 0}
                object_appearance_count[obj]['cam1'] += 1
                
            for obj in detection2:
                if obj not in object_appearance_count:
                    object_appearance_count[obj] = {'cam1': 0, 'cam2': 0, 'matched': 0}
                object_appearance_count[obj]['cam2'] += 1
            
            # Match objects between views
            matches = self.match_objects(detection1, detection2)
            
            # Track matched objects
            for obj_name in matches:
                if obj_name not in matched_count:
                    matched_count[obj_name] = 0
                matched_count[obj_name] += 1
                object_appearance_count[obj_name]['matched'] += 1
            
            # Triangulate 3D positions
            frame_3d = {}
            for obj_name, match in matches.items():
                try:
                    pos_3d = self.triangulate_point(match['pt1'], match['pt2'], obj_name)
                    frame_3d[obj_name] = {
                        'position': pos_3d.tolist(),
                    }
                except Exception as e:
                    print(f"Triangulation failed for {obj_name} at frame {frame_idx}: {e}")

            # Store results
            frame_key = f"frame_{frame_idx}"
            tracking_2d_results[frame_key] = {
                'view1': detection1,
                'view2': detection2,
                'matches': matches
            }
            tracking_3d_results[frame_key] = frame_3d
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")
        
        # Print detection and matching statistics
        print("\n=== Object Detection and Matching Statistics ===")
        print(f"{'Object':<15} {'Cam1':<8} {'Cam2':<8} {'Matched':<8} {'Match %':<8}")
        print("-" * 50)
        for obj_name, counts in sorted(object_appearance_count.items()):
            cam1_count = counts['cam1']
            cam2_count = counts['cam2']
            matched = counts['matched']
            max_possible = min(cam1_count, cam2_count)
            match_percentage = (matched / max_possible * 100) if max_possible > 0 else 0
            print(f"{obj_name:<15} {cam1_count:<8} {cam2_count:<8} {matched:<8} {match_percentage:<8.1f}%")
        
        # Save results
        with open(json_2d_path, "w") as f:
            json.dump(tracking_2d_results, f, indent=2)
        
        with open(json_3d_path, "w") as f:
            json.dump(tracking_3d_results, f, indent=2)
        
        print(f"\nTracking complete!")
        print(f"2D results saved to {json_2d_path}")
        print(f"3D results saved to {json_3d_path}")
        
        # Calculate trajectory metrics
        metrics = self.calculate_trajectory_metrics(tracking_3d_results)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "trajectory_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Trajectory metrics saved to {metrics_path}")
        
        # Create interactive visualization
        interactive_path = os.path.join(output_dir, "interactive_3d_trajectories.html")
        self.create_interactive_3d_plot(tracking_3d_results, metrics, interactive_path)
        
        # Optionally save 3D data in specific format
        if output_3d:
            self.save_3d_trajectories(tracking_3d_results, output_3d)
        
        return tracking_3d_results, metrics
    
    def save_3d_trajectories(self, tracking_3d_results, output_path):
        """Save 3D trajectories in CSV format"""
        import csv
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'object', 'x', 'y', 'z']
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
                        'z': pos[2]
                    })
        
        print(f"3D trajectories saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run stereo YOLO tracking with 3D triangulation and interactive visualization")
    
    # Video inputs
    parser.add_argument("--video1", type=str, required=True, help="Path to left camera detection")
    parser.add_argument("--video2", type=str, required=True, help="Path to right camera detection")
    
    # Camera parameters
    parser.add_argument("--camparams1", type=str, required=True, help="Path to camera 1 parameters JSON")
    parser.add_argument("--camparams2", type=str, required=True, help="Path to camera 2 parameters JSON")
    
    # Court corners
    parser.add_argument("--corners1", type=str, help="Path to camera 1 court corners JSON")
    parser.add_argument("--corners2", type=str, help="Path to camera 2 court corners JSON")
    
    # Optional parameters
    parser.add_argument("--fps", type=int, default=25, help="Video frame rate (default: 25)")
    parser.add_argument("--output_3d", type=str, help="Path to save 3D trajectories CSV")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = StereoTracker(
        args.video1, args.video2, args.camparams1, args.camparams2,
        args.corners1, args.corners2, args.fps
    )
    
    # Run tracking
    tracking_3d_results, metrics = tracker.run_tracking(args.output_3d)
    
    print("\n=== Summary of Player Metrics ===")
    for player, player_metrics in metrics.items():
        print(f"\n{player}:")
        print(f"  - Total ground distance: {player_metrics['total_ground_distance_m']:.2f} m")
        print(f"  - Average ground speed: {player_metrics['avg_ground_speed_ms']:.2f} m/s")
        print(f"  - Max ground speed: {player_metrics['max_ground_speed_ms']:.2f} m/s")
        print(f"  - Coverage area: {player_metrics['coverage_area_m2']:.2f} m²")
        print(f"  - Average height: {player_metrics['avg_height_m']:.2f} m")
        print(f"  - Max height: {player_metrics['max_height_m']:.2f} m")
        print(f"  - Direction changes: {player_metrics['direction_changes']}")


if __name__ == "__main__":
    main()