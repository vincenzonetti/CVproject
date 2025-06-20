import argparse
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


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

class StereoTracker:
    def __init__(self,detection1_path,detection2_path, cam1_params_path, cam2_params_path):
        
        
        self.det1=self.load_detections(detection1_path)
        self.det2=self.load_detections(detection2_path)
        # Load camera parameters
        self.cam1_params = self.load_camera_params(cam1_params_path)
        self.cam2_params = self.load_camera_params(cam2_params_path)
        
        # Compute projection matrices
        self.P1 = self.compute_projection_matrix(self.cam1_params)
        self.P2 = self.compute_projection_matrix(self.cam2_params)
        

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
                bbox = x['bbox']    
                
                # Add the class_id and bbox to the nested dictionary
                formatted_params[-1][class_id] = {
                    'bbox': bbox,
                    'center': [bbox[0], bbox[1]]  # x_center, y_center already in bbox
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
        
        frame_idx = 0
        
        for detection1,detection2 in tqdm(zip(self.det1,self.det2)):
            # Match objects between views
            matches = self.match_objects(detection1, detection2)
            
            # Triangulate 3D positions
            frame_3d = {}
            for obj_name, match in matches.items():
                try:
                    pos_3d = self.triangulate_point(match['pt1'], match['pt2'])
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
        
        
        # Save results
        with open(json_2d_path, "w") as f:
            json.dump(tracking_2d_results, f, indent=2)
        
        with open(json_3d_path, "w") as f:
            json.dump(tracking_3d_results, f, indent=2)
        
        print(f"Tracking complete!")
        print(f"2D results saved to {json_2d_path}")
        print(f"3D results saved to {json_3d_path}")

        
        # Optionally save 3D data in specific format
        if output_3d:
            self.save_3d_trajectories(tracking_3d_results, output_3d)
        
        return tracking_3d_results
    
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
    parser.add_argument("--video1", type=str, required=True, help="Path to left camera detection")
    parser.add_argument("--video2", type=str, required=True, help="Path to right camera detection")
    
    # Camera parameters
    parser.add_argument("--camparams1", type=str, required=True, help="Path to camera 1 parameters JSON")
    parser.add_argument("--camparams2", type=str, required=True, help="Path to camera 2 parameters JSON")
    
    # Optional outputs
    parser.add_argument("--output_3d", type=str, help="Path to save 3D trajectories CSV")
    parser.add_argument("--visualize", action="store_true", help="Generate 3D trajectory visualization")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = StereoTracker(
         args.video1, args.video2, args.camparams1, args.camparams2
    )
    
    # Run tracking
    tracking_3d_results = tracker.run_tracking(args.output_3d)
    
    # Optional visualization
    if args.visualize:
        output_dir = os.path.join("outputs", f"stereo")
        viz_path = os.path.join(output_dir, "3d_trajectories.png")
        tracker.visualize_3d_trajectories(tracking_3d_results, viz_path)

if __name__ == "__main__":
    main()