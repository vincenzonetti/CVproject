"""
Main entry point for basketball stereo tracking system.

This module orchestrates the entire tracking pipeline including
data loading, matching, triangulation, metrics calculation,
and visualization generation.
"""

import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple, Optional

# Import all modules
from config import (
    OUTPUT_DIR, TRACKING_2D_FILENAME, TRACKING_3D_FILENAME,
    METRICS_FILENAME, INTERACTIVE_PLOT_FILENAME, METRICS_DASHBOARD_FILENAME,
    DEFAULT_FPS
)
from loader import load_stereo_data
from tracking.matcher import match_objects, compute_matching_statistics, print_matching_statistics
from tracking.triangulator import Triangulator, compute_projection_matrix
from tracking.metrics import calculate_trajectory_metrics
from tracking.cleaner import clean_trajectories
from utils.court import setup_court_transformation
from visualization.plot_3d import create_interactive_3d_plot
from visualization.dashboard import create_metrics_dashboard


# In 3Dreconstruction/main.py

class StereoTracker:
    """Main class for stereo basketball tracking."""
    
    def __init__(self, data: Dict, fps: int = DEFAULT_FPS):
        """
        Initialize the stereo tracker.
        
        Args:
            data: Dictionary containing all loaded data
            fps: Frames per second for video
        """
        self.det1 = data['detections1']
        self.det2 = data['detections2']
        self.cam1_params = data['cam1_params']
        self.cam2_params = data['cam2_params']
        self.fps = fps
        
        # Compute projection matrices
        self.P1 = compute_projection_matrix(self.cam1_params)
        self.P2 = compute_projection_matrix(self.cam2_params)
        
        # Setup court transformation if corners available
        self.court_corners_3d = None
        self.H1 = None
        self.H2 = None
        
        if 'cam1_real_corners' in data and 'cam2_real_corners' in data:
            cam1_corners = (data['cam1_real_corners'], data['cam1_img_corners'])
            cam2_corners = (data['cam2_real_corners'], data['cam2_img_corners'])
            self.H1, self.H2, self.court_corners_3d = setup_court_transformation(
                cam1_corners, cam2_corners
            )
        
        # Initialize triangulator with both camera parameters
        self.triangulator = Triangulator(
            self.P1, self.P2, self.cam1_params, self.cam2_params, self.H1, self.H2
        )
    
    def run_tracking(self, output_3d: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Run the complete tracking pipeline with post-cleaning ball height scaling.
        
        Args:
            output_3d: Optional path to save 3D trajectories
            
        Returns:
            Tuple of (tracking_3d_results, metrics)
        """
        print("Starting stereo tracking pipeline...")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Match objects between cameras across all frames
        matches, frame_map1, frame_map2 = match_objects(self.det1, self.det2)
        
        # Compute and print matching statistics
        stats = compute_matching_statistics(matches, self.det1, self.det2)
        print_matching_statistics(stats)
        
        # Save 2D tracking results
        tracking_2d = {
            'camera1': frame_map1,
            'camera2': frame_map2,
            'matches': matches
        }
        with open(os.path.join(OUTPUT_DIR, TRACKING_2D_FILENAME), 'w') as f:
            json.dump(tracking_2d, f, indent=2)
        
        # SINGLE PASS: Triangulate with raw heights
        print("\nTriangulating 3D positions...")
        tracking_3d_raw = {}
        
        for frame_idx, frame_matches in enumerate(tqdm(matches, desc="Triangulating")):
            tracking_3d_raw[f'frame_{frame_idx}'] = {}
            
            for obj_name, (idx1, idx2) in frame_matches.items():
                # Skip if object not seen by both cameras
                if idx1 is None or idx2 is None:
                    continue
                
                # Get detections for this frame
                det1_frame = self.det1[frame_idx]
                det2_frame = self.det2[frame_idx]
                
                # Skip if object not in both frames
                if obj_name not in det1_frame or obj_name not in det2_frame:
                    continue
                
                # Get 2D points
                pt1 = det1_frame[obj_name]['center']
                pt2 = det2_frame[obj_name]['center']
                
                # Triangulate (no scaling applied yet)
                points_3d = self.triangulator.triangulate_point(pt1, pt2, obj_name)
                
                #if obj_name == 'Ball':
                #    breakpoint()
                
                tracking_3d_raw[f'frame_{frame_idx}'][obj_name] = {
                    'position': points_3d.tolist(),
                    'camera1_2d': pt1,
                    'camera2_2d': pt2
                }
        
        # Clean trajectories BEFORE scaling
        print("\nCleaning trajectories...")
        tracking_3d_cleaned = clean_trajectories(tracking_3d_raw)
        
        # Compute ball height scaling from cleaned data
        print("\nComputing ball height scaling from cleaned trajectories...")
        self.triangulator.compute_ball_height_scaling(tracking_3d_cleaned)
        
        # Apply scaling to get final results
        print("\nApplying ball height scaling...")
        tracking_3d_results = self.triangulator.apply_ball_height_scaling(tracking_3d_cleaned)
        
        # Save 3D tracking results
        with open(os.path.join(OUTPUT_DIR, TRACKING_3D_FILENAME), 'w') as f:
            json.dump(tracking_3d_results, f, indent=2)
        
        # Calculate metrics
        print("\nCalculating trajectory metrics...")
        metrics = calculate_trajectory_metrics(tracking_3d_results, self.fps)
        
        # Save metrics
        with open(os.path.join(OUTPUT_DIR, METRICS_FILENAME), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        print("\nCreating interactive 3D visualization...")
        create_interactive_3d_plot(
            tracking_3d_results, metrics, self.court_corners_3d,
            os.path.join(OUTPUT_DIR, INTERACTIVE_PLOT_FILENAME)
        )
        
        # Create metrics dashboard
        print("Creating metrics dashboard...")
        create_metrics_dashboard(
            metrics,
            os.path.join(OUTPUT_DIR, METRICS_DASHBOARD_FILENAME)
        )
        
        # Save to CSV if requested
        if output_3d:
            self._save_3d_trajectories_csv(tracking_3d_results, output_3d)
        
        return tracking_3d_results, metrics
    
    def _save_results(self, tracking_2d_results: Dict, tracking_3d_results: Dict) -> None:
        """Save tracking results to JSON files."""
        # Save 2D results
        json_2d_path = os.path.join(OUTPUT_DIR, TRACKING_2D_FILENAME)
        with open(json_2d_path, "w") as f:
            json.dump(tracking_2d_results, f, indent=2)
        print(f"\n2D results saved to {json_2d_path}")
        
        # Save 3D results
        json_3d_path = os.path.join(OUTPUT_DIR, TRACKING_3D_FILENAME)
        with open(json_3d_path, "w") as f:
            json.dump(tracking_3d_results, f, indent=2)
        print(f"3D results saved to {json_3d_path}")
    
    def _create_visualizations(self, tracking_3d_results: Dict, metrics: Dict) -> None:
        """Create interactive visualizations."""
        print("\nCreating visualizations...")
        
        # Create 3D interactive plot
        plot_path = os.path.join(OUTPUT_DIR, INTERACTIVE_PLOT_FILENAME)
        create_interactive_3d_plot(
            tracking_3d_results, metrics, self.court_corners_3d, plot_path
        )
        
        # Create metrics dashboard
        dashboard_path = os.path.join(OUTPUT_DIR, METRICS_DASHBOARD_FILENAME)
        from tracking.metrics import extract_trajectories
        trajectories = extract_trajectories(tracking_3d_results)
        create_metrics_dashboard(metrics, trajectories, dashboard_path)
    
    def _save_3d_trajectories_csv(self, tracking_3d_results: Dict, output_path: str) -> None:
        """Save 3D trajectories in CSV format."""
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
    
    def _print_summary(self, metrics: Dict) -> None:
        """Print summary of player metrics."""
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


def main():
    """Main entry point for the basketball tracking system."""
    parser = argparse.ArgumentParser(
        description="Basketball stereo tracking with 3D triangulation and interactive visualization"
    )
    
    # Required arguments
    parser.add_argument("--video1", type=str, required=True, 
                       help="Path to camera 1 detection JSON")
    parser.add_argument("--video2", type=str, required=True, 
                       help="Path to camera 2 detection JSON")
    parser.add_argument("--camparams1", type=str, required=True, 
                       help="Path to camera 1 parameters JSON")
    parser.add_argument("--camparams2", type=str, required=True, 
                       help="Path to camera 2 parameters JSON")
    
    # Optional arguments
    parser.add_argument("--corners1", type=str, 
                       help="Path to camera 1 court corners JSON")
    parser.add_argument("--corners2", type=str, 
                       help="Path to camera 2 court corners JSON")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, 
                       help=f"Video frame rate (default: {DEFAULT_FPS})")
    parser.add_argument("--output_3d", type=str, 
                       help="Path to save 3D trajectories CSV")
    
    args = parser.parse_args()
    
    # Load all data
    print("Loading data...")
    data = load_stereo_data(
        args.video1, args.video2, 
        args.camparams1, args.camparams2,
        args.corners1, args.corners2
    )
    
    # Initialize and run tracker
    tracker = StereoTracker(data, args.fps)
    tracking_3d_results, metrics = tracker.run_tracking(args.output_3d)
    
    print("\nTracking complete!")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()