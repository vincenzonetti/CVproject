import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Updated camera parameters from your calibration
K = np.array([
    [2078.0703718989125, 0.0, 1946.604176181301],
    [0.0, 2073.6024074372485, 1201.5256035243012],
    [0.0, 0.0, 1.0]
])

# Distortion coefficients
dist_coeffs = np.array([-0.3220696124706172, 0.10639169555862117, -0.002042411604460586, -0.0018164567522046646, -0.01580880868147406])

# Convert rotation vector to rotation matrix
rvec = np.array([0.016147218644618988, 2.546968460083008, -1.7685179710388184])
R, _ = cv2.Rodrigues(rvec)

# Translation vector
T = np.array([[-938.4185180664062], [-210.0382537841797], [19418.685546875]])

# Basketball court boundaries for validation (in meters)
COURT_BOUNDS = {
    'x_min': -15.0,  # A bit beyond actual court for safety
    'x_max': 15.0,
    'y_min': -8.0,
    'y_max': 8.0
}

def undistort_point(u, v, K, dist_coeffs):
    """
    Undistort a single image point using camera calibration
    """
    # Convert to numpy array format expected by OpenCV
    points = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(points, K, dist_coeffs, P=K)
    return undistorted[0][0]

def image_point_to_floor_coords(u, v, K, dist_coeffs, R, T):
    """
    Back-project a 2D image point onto the floor plane (Z=0) in world coords
    Now includes lens distortion correction
    """
    try:
        # First undistort the point
        u_undist, v_undist = undistort_point(u, v, K, dist_coeffs)
        
        # Build camera ray in camera frame using undistorted coordinates
        pixel = np.array([u_undist, v_undist, 1.0]).reshape((3, 1))
        ray_cam = np.linalg.inv(K) @ pixel
        
        # Transform ray to world frame
        ray_world = R @ ray_cam
        
        # Compute camera center in world
        cam_center_world = -R.T @ T
        
        # Check if ray is pointing towards the floor
        if ray_world[2, 0] >= -1e-6:  # Small epsilon for numerical stability
            return None, False
        
        # Ray-plane intersection: plane Z=0
        lam = -cam_center_world[2, 0] / ray_world[2, 0]
        
        # Check if intersection is in front of camera
        if lam <= 0:
            return None, False
        
        intersection = cam_center_world + lam * ray_world
        world_coords = intersection.flatten()
        
        # Validate that the point is within reasonable court bounds
        if (COURT_BOUNDS['x_min'] <= world_coords[0] <= COURT_BOUNDS['x_max'] and 
            COURT_BOUNDS['y_min'] <= world_coords[1] <= COURT_BOUNDS['y_max']):
            return world_coords, True
        else:
            return world_coords, False  # Outside court bounds
            
    except:
        return None, False

def estimate_ball_radius_in_world(bbox, img_w, img_h, K, dist_coeffs, R, T):
    """
    Estimate the ball's radius in world coordinates based on its bounding box size
    Basketball diameter is ~24cm, so radius ~12cm = 0.12m
    """
    xc, yc, bw, bh = bbox
    
    # Use the smaller dimension to estimate radius
    ball_pixel_radius = min(bw * img_w, bh * img_h) / 2
    
    # Project center point to get depth
    center_world, is_valid = image_point_to_floor_coords(
        int(xc * img_w), int(yc * img_h), K, dist_coeffs, R, T
    )
    
    if not is_valid or center_world is None:
        return 0.12  # Default basketball radius in meters
    
    # Estimate world radius based on pixel size and depth
    # This is a rough approximation
    camera_center = -R.T @ T
    depth = np.linalg.norm(center_world - camera_center.flatten())
    
    # Approximate conversion from pixels to world units at this depth
    world_radius = (ball_pixel_radius * depth) / K[0, 0]  # Using focal length
    
    # Clamp to reasonable basketball size (8cm to 16cm radius)
    return np.clip(world_radius, 0.08, 0.16)

def is_ball_touching_floor(bbox, img_w, img_h, K, dist_coeffs, R, T):
    """
    Determine if ball is touching the floor with basketball-specific logic
    """
    xc, yc, bw, bh = bbox
    
    # Estimate ball radius in world coordinates
    ball_radius = estimate_ball_radius_in_world(bbox, img_w, img_h, K, dist_coeffs, R, T)
    
    # Check multiple points around the bottom of the ball
    test_points = []
    
    # Bottom center
    bottom_y = (yc + bh / 2) * img_h
    test_points.append((xc * img_w, bottom_y))
    
    # Bottom corners with slight inward offset
    offset = bw * 0.3  # 30% inward from edges
    left_x = (xc - bw / 2 + offset) * img_w
    right_x = (xc + bw / 2 - offset) * img_w
    test_points.append((left_x, bottom_y))
    test_points.append((right_x, bottom_y))
    
    # Check points slightly above bottom for better 3D estimation
    mid_bottom_y = (yc + bh * 0.3) * img_h
    test_points.append((xc * img_w, mid_bottom_y))
    
    valid_projections = []
    
    for u, v in test_points:
        world_coords, is_valid = image_point_to_floor_coords(
            int(u), int(v), K, dist_coeffs, R, T
        )
        if is_valid and world_coords is not None:
            valid_projections.append(world_coords)
    
    if not valid_projections:
        return False, None, ball_radius
    
    # Find the lowest valid projection (closest to floor)
    lowest_z = min(proj[2] for proj in valid_projections)
    best_projection = next(proj for proj in valid_projections if proj[2] == lowest_z)
    
    # Ball touches floor if its bottom is within ball_radius of the floor
    # Add some tolerance for detection uncertainty
    tolerance = ball_radius + 0.05  # 5cm additional tolerance
    
    is_touching = abs(best_projection[2]) <= tolerance
    
    return is_touching, best_projection, ball_radius

def draw_floor_contact_indicator(frame, is_touching, world_coords=None, ball_radius=None):
    """
    Draw floor contact indicator at fixed position (bottom right)
    """
    h, w = frame.shape[:2]
    
    # Position for the indicator (bottom right corner)
    indicator_x = w - 300
    indicator_y = h - 100
    
    # Background rectangle
    cv2.rectangle(frame, (indicator_x - 15, indicator_y - 60), 
                  (w - 10, indicator_y + 40), (0, 0, 0), -1)
    cv2.rectangle(frame, (indicator_x - 15, indicator_y - 60), 
                  (w - 10, indicator_y + 40), (255, 255, 255), 2)
    
    if is_touching:
        # Red indicator for ball on floor
        cv2.circle(frame, (indicator_x, indicator_y - 30), 20, (0, 0, 255), -1)
        cv2.putText(frame, "BALL ON FLOOR", (indicator_x + 30, indicator_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show world coordinates and ball info
        if world_coords is not None:
            coord_text = f"Pos: ({world_coords[0]:.1f}, {world_coords[1]:.1f}, {world_coords[2]:.2f})"
            cv2.putText(frame, coord_text, (indicator_x - 10, indicator_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        if ball_radius is not None:
            radius_text = f"Radius: {ball_radius*100:.1f}cm"
            cv2.putText(frame, radius_text, (indicator_x - 10, indicator_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        # Green indicator for ball in air
        cv2.circle(frame, (indicator_x, indicator_y - 30), 20, (0, 255, 0), -1)
        cv2.putText(frame, "BALL IN AIR", (indicator_x + 30, indicator_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show height above floor if available
        if world_coords is not None:
            height_text = f"Height: {world_coords[2]:.2f}m"
            cv2.putText(frame, height_text, (indicator_x - 10, indicator_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_court_overlay(frame, K, dist_coeffs, R, T):
    """
    Draw basketball court lines overlay for validation
    """
    # Key court points to project
    court_lines = [
        # Center line
        [[0, -7.5, 0], [0, 7.5, 0]],
        # Sidelines
        [[-14, -7.5, 0], [-14, 7.5, 0]],
        [[14, -7.5, 0], [14, 7.5, 0]],
        # Baselines
        [[-14, -7.5, 0], [14, -7.5, 0]],
        [[-14, 7.5, 0], [14, 7.5, 0]],
    ]
    
    for line in court_lines:
        points_2d = []
        for point_3d in line:
            # Project 3D point to 2D
            point_3d = np.array(point_3d, dtype=np.float32).reshape(1, 1, 3)
            point_2d, _ = cv2.projectPoints(point_3d, rvec, T, K, dist_coeffs)
            x, y = point_2d[0][0]
            
            # Check if point is within image bounds
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                points_2d.append((int(x), int(y)))
        
        # Draw line if both points are valid
        if len(points_2d) == 2:
            cv2.line(frame, points_2d[0], points_2d[1], (255, 255, 0), 1)

def run_tracker(model_path: str, video_path: str, show_court_overlay: bool = False):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Prepare output
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join("outputs", f"{video_name}_{model_name}_basketball")
    os.makedirs(output_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, "tracking_results_floor.json")
    video_output_path = os.path.join(output_dir, "tracked_video_floor.mp4")

    # Video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    tracking_results = {}
    frame_idx = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length)

    # Statistics
    ball_on_floor_frames = 0
    total_ball_detections = 0

    while cap.isOpened():
        pbar.update(1)
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: draw court overlay for debugging
        if show_court_overlay:
            draw_court_overlay(frame, K, dist_coeffs, R, T)

        results = model(frame, verbose=False)[0]
        detections = []
        img_h, img_w = frame.shape[:2]
        
        # Track if any ball is touching the floor
        any_ball_touching_floor = False
        floor_contact_coords = None
        ball_radius_estimate = None

        if results.boxes is not None:
            boxes = results.boxes
            best_by_class = {}

            # Keep highest-confidence detection per class
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x_center, y_center, w, h = boxes.xywh[i].tolist()

                if cls_id not in best_by_class or conf > best_by_class[cls_id]['conf']:
                    best_by_class[cls_id] = {
                        'class_id': cls_id,
                        'track_id': cls_id,
                        'bbox': [x_center / img_w, y_center / img_h, w / img_w, h / img_h],
                        'conf': conf
                    }

            # Process each selected detection
            for det in best_by_class.values():
                xc, yc, bw, bh = det['bbox']
                x1 = int((xc - bw / 2) * img_w)
                y1 = int((yc - bh / 2) * img_h)
                x2 = int((xc + bw / 2) * img_w)
                y2 = int((yc + bh / 2) * img_h)

                # Default: not touching floor
                det['touch_floor'] = False
                det['world_coords'] = None
                det['ball_radius'] = None

                # Check for ball class (typically 0 for ball, but verify with your model)
                if det['class_id'] == 0:  # Adjust this if needed
                    total_ball_detections += 1
                    
                    is_touching, world_coords, ball_radius = is_ball_touching_floor(
                        det['bbox'], img_w, img_h, K, dist_coeffs, R, T
                    )
                    
                    det['touch_floor'] = is_touching
                    det['world_coords'] = world_coords.tolist() if world_coords is not None else None
                    det['ball_radius'] = ball_radius
                    
                    if is_touching:
                        any_ball_touching_floor = True
                        floor_contact_coords = world_coords
                        ball_radius_estimate = ball_radius
                        ball_on_floor_frames += 1
                        
                        # Draw contact indication on the ball
                        u = int(xc * img_w)
                        v = int((yc + bh / 2) * img_h)
                        cv2.circle(frame, (u, v), 12, (0, 0, 255), 4)

                # Draw bounding box and ID
                color = (0, 0, 255) if det.get('touch_floor', False) else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add class label
                class_names = {0: 'Ball', 1: 'Player'}  # Adjust based on your model
                class_name = class_names.get(det['class_id'], f'Class_{det["class_id"]}')
                
                cv2.putText(frame, f"{class_name} {det['track_id']} ({det['conf']:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detections.append(det)

        # Draw floor contact indicator at fixed position
        draw_floor_contact_indicator(frame, any_ball_touching_floor, 
                                   floor_contact_coords, ball_radius_estimate)

        # Save frame detections
        frame_key = f"{video_name}_{frame_idx}"
        tracking_results[frame_key] = {
            'detections': detections,
            'ball_touching_floor': any_ball_touching_floor,
            'floor_contact_coords': floor_contact_coords.tolist() if floor_contact_coords is not None else None,
            'ball_radius': ball_radius_estimate
        }
        frame_idx += 1

        out_video.write(frame)

    cap.release()
    pbar.close()
    out_video.release()

    # Write results
    with open(json_output_path, "w") as f:
        json.dump(tracking_results, f, indent=2)

    print(f"Tracking complete. Results saved to {json_output_path}")
    print(f"Annotated video saved to {video_output_path}")
    
    # Print statistics
    total_frames = len(tracking_results)
    if total_ball_detections > 0:
        print(f"\nStatistics:")
        print(f"Total frames: {total_frames}")
        print(f"Frames with ball detections: {total_ball_detections}")
        print(f"Frames with ball on floor: {ball_on_floor_frames}")
        print(f"Ball on floor percentage: {(ball_on_floor_frames/total_ball_detections)*100:.1f}%")
    else:
        print("No ball detections found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO tracking and detect ball-floor contact for basketball.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    parser.add_argument("--show-court", action="store_true", help="Show court line overlay for debugging")
    args = parser.parse_args()

    run_tracker(args.model, args.video, args.show_court)