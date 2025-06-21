# Basketball Stereo Tracking System

A comprehensive system for tracking basketball players and the ball in 3D using stereo camera views.

## Overview

This system performs:
- Object detection matching between two camera views
- 3D triangulation of player and ball positions
- Court coordinate transformation using homography
- Movement metrics calculation (speed, distance, acceleration, etc.)
- Interactive 3D visualization with player filtering
- Comprehensive metrics dashboard

## Project Structure

```
basketball_tracking/
│
├── config/
│   └── constants.py          # Configuration and global constants
│
├── data/
│   ├── __init__.py
│   ├── loader.py            # Data loading utilities
│   └── models.py            # Data structures (if needed)
│
├── tracking/
│   ├── __init__.py
│   ├── matcher.py           # Object matching between views
│   ├── triangulator.py      # 3D triangulation logic
│   └── metrics.py           # Trajectory metrics calculation
│
├── visualization/
│   ├── __init__.py
│   ├── plot_3d.py          # 3D interactive plotting
│   └── dashboard.py        # Metrics dashboard creation
│
├── utils/
│   ├── __init__.py
│   ├── camera.py           # Camera utilities
│   └── court.py            # Court coordinate transformations
│
├── main.py                 # Main entry point
└── README.md              # This file
```

## Installation

```bash
# Install required packages
pip install numpy opencv-python matplotlib plotly pandas scipy tqdm
```

## Usage

### Basic Usage

```bash
python main.py \
    --video1 detections_cam1.json \
    --video2 detections_cam2.json \
    --camparams1 cam1_params.json \
    --camparams2 cam2_params.json
```

### Full Usage with Court Corners

```bash
python main.py \
    --video1 detections_cam1.json \
    --video2 detections_cam2.json \
    --camparams1 cam1_params.json \
    --camparams2 cam2_params.json \
    --corners1 cam1_corners.json \
    --corners2 cam2_corners.json \
    --fps 25 \
    --output_3d trajectories_3d.csv
```

### Command Line Arguments

- `--video1`: Path to camera 1 detection JSON (required)
- `--video2`: Path to camera 2 detection JSON (required)
- `--camparams1`: Path to camera 1 parameters JSON (required)
- `--camparams2`: Path to camera 2 parameters JSON (required)
- `--corners1`: Path to camera 1 court corners JSON (optional)
- `--corners2`: Path to camera 2 court corners JSON (optional)
- `--fps`: Video frame rate (default: 25)
- `--output_3d`: Path to save 3D trajectories CSV (optional)

## Input File Formats

### Detection JSON Format
```json
{
  "frame_0": [
    {
      "class_id": 0,
      "bbox": [0.5, 0.3, 0.1, 0.2]
    }
  ]
}
```

### Camera Parameters JSON Format
```json
{
  "mtx": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist": [k1, k2, p1, p2, k3],
  "rvecs": [rx, ry, rz],
  "tvecs": [tx, ty, tz]
}
```

### Court Corners JSON Format
```json
{
  "real_corners": [[x1, y1, 0], [x2, y2, 0], ...],
  "img_corners": [[u1, v1], [u2, v2], ...]
}
```

## Output Files

The system generates the following outputs in `outputs/stereo/`:

1. **tracking_2d_results.json** - 2D tracking data for both cameras
2. **tracking_3d_results.json** - 3D positions for all tracked objects
3. **trajectory_metrics.json** - Calculated movement metrics
4. **interactive_3d_trajectories.html** - Interactive 3D visualization
5. **interactive_3d_trajectories_metrics.html** - Metrics dashboard
6. **interactive_3d_trajectories_metrics.csv** - Metrics in CSV format

## Coordinate System

The system uses basketball court conventions:
- **X-axis**: Width of court (side to side)
- **Y-axis**: Height (vertical, 0 for ground)
- **Z-axis**: Depth of court (baseline to baseline)

## Key Features

### Object Classes
The system tracks 13 different classes:
- Ball
- Red team players (Red_0, Red_11, Red_12, Red_16, Red_2)
- White team players (White_13, White_16, White_25, White_27, White_34)
- Referees (Refree_F, Refree_M)

### Metrics Calculated
- Total ground distance traveled
- Average and maximum ground speed
- Average acceleration
- Number of direction changes
- Court coverage area
- Height statistics (for ball tracking)

### Interactive Visualization
- Click player names to show/hide trajectories
- Hover over trajectories for detailed information
- Court boundary and surface visualization
- Start/end position markers

## Troubleshooting

### Missing Players in Visualization
Check the detection statistics output to see:
- If the player was detected in both cameras
- If detections were successfully matched
- If there are enough frames (minimum 2) for trajectory

### Scale Issues
Ensure court corners are provided and in meters. The system assumes:
- Court corners have Z=0 (ground level)
- Real corner coordinates are in meters
- Standard basketball court is ~28m x 15m

## Module Descriptions

### `config/constants.py`
Global configuration including image dimensions, class names, colors, and file paths.

### `data/loader.py`
Handles loading of JSON files and data preprocessing.

### `tracking/matcher.py`
Matches detected objects between camera views based on class identity.

### `tracking/triangulator.py`
Performs 3D triangulation and coordinate transformations.

### `tracking/metrics.py`
Calculates comprehensive movement metrics for each tracked object.

### `visualization/plot_3d.py`
Creates interactive 3D trajectory visualizations using Plotly.

### `visualization/dashboard.py`
Generates metrics dashboard with statistical comparisons.

### `utils/court.py`
Handles court coordinate transformations and homography calculations.

### `main.py`
Orchestrates the entire tracking pipeline.

## License

[Your license here]

## Contact

[Your contact information]