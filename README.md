# Basketball Tracking System

A comprehensive basketball tracking system that performs 2D object detection and tracking, followed by 3D reconstruction using stereo camera views. The system tracks players and the ball in real-time, calculates movement metrics, and provides interactive 3D visualizations.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [2D Tracking System](#2d-tracking-system)
- [3D Reconstruction Pipeline](#3d-reconstruction-pipeline)
- [Alternative Tracking Approaches](#alternative-tracking-approaches)
- [Output Files](#output-files)

## Installation

### Required Packages

Install the following packages to create the environment for using all scripts:

```bash
pip install opencv-python
pip install ultralytics
pip install numpy
pip install scipy
pip install plotly
pip install pandas
pip install tqdm
pip install matplotlib
pip install scikit-learn
```

For YOLO models and tracking:
```bash
pip install torch torchvision  # PyTorch for YOLO models
```
If you encounter some troubles please try installing the full environment using
```bash
conda env create -f environment.yml #full conda env
```


### Project Structure

```
basketball_tracking/
├── 2D_tracking/                 # 2D detection and tracking scripts
│   ├── twoModelsTrack.py       # Two-model tracking with YOLO trackers
│   ├── twoModelsYolo.py        # Two-model detection without tracking
│   ├── evaluate_tracker.py     # Evaluation utilities
│   ├── utils.py                # Helper functions
│   ├── best_ball.pt           # Ball detection model
│   ├── best_players.pt        # Player detection model
│   └── botsort.yaml           # Tracker configuration
├── 3Dreconstruction/           # 3D reconstruction pipeline
│   ├── main.py                # Main entry point
│   ├── config.py              # Configuration and constants
│   ├── loader.py              # Data loading utilities
│   ├── rectify_results.py     # Bounding box rectification
│   ├── tracking/              # Tracking modules
│   ├── visualization/         # 3D visualization modules
│   ├── utils/                 # Utility modules
│   ├── camparams/             # Camera calibration files
│   └── rectified/             # Rectified detection results
└── other_files/               # Alternative tracking approaches
```

## Usage Examples

### 1. 3D Reconstruction (Main Pipeline)

```bash
cd 3Dreconstruction

python main.py \
    --video1 rectified/out2.json \
    --video2 rectified/out13.json \
    --camparams1 camparams/out2_camera_calib.json \
    --camparams2 camparams/out13_camera_calib.json \
    --corners1 camparams/out2_img_points.json \
    --corners2 camparams/out13_img_points.json \
    --output_3d outputs/results
```

**Note**: The detection files in the `rectified/` folder have been processed using `rectify_results.py`. This rectification step corrects camera distortion by adjusting only the center points of YOLO detections, not the entire bounding box parameters.

### 2. 2D Tracking with YOLO Trackers

```bash
cd 2D_tracking

python twoModelsTrack.py \
    --modelB best_ball.pt \
    --modelP best_players.pt \
    --video ../../data/videos/out2.mp4 \
    --trackerP botsort.yaml \
    --trackerB botsort.yaml
```

### 3. 2D Detection without Tracking

```bash
cd 2D_tracking

python twoModelsYolo.py \
    --modelB best_ball.pt \
    --modelP best_players.pt \
    --video ../../data/videos/out2.mp4
```

## 2D Tracking System

### Two-Model Approach

The 2D tracking system uses two specialized YOLO models:

- **`best_players.pt`**: Optimized for detecting basketball players
- **`best_ball.pt`**: Fine-tuned specifically for basketball detection

### Key Features

1. **Best Class Selection**: When multiple detections of the same class are present in a frame, only the detection with the highest confidence score is retained. This prevents duplicate tracking of the same object.

2. **Class-ID Mapping**: Each detected object is assigned a class ID that corresponds to specific player jerseys or the ball, enabling consistent tracking across frames.

3. **Dual Processing**: 
   - `twoModelsTrack.py`: Integrates YOLO detection with tracking algorithms (BoTSORT)
   - `twoModelsYolo.py`: Performs pure detection without temporal tracking

### Detection Process

```python
# Pseudo-code for best class selection
best_by_class = {}
for detection in frame_detections:
    class_id = detection.class_id
    confidence = detection.confidence
    
    if class_id not in best_by_class or confidence > best_by_class[class_id].confidence:
        best_by_class[class_id] = detection
```

## 3D Reconstruction Pipeline

### Overview

The 3D reconstruction system transforms 2D detections from two camera views into 3D world coordinates using stereo vision principles.

### Pipeline Steps

1. **Data Loading** (`loader.py`)
   - Loads detection results from both cameras
   - Loads camera calibration parameters
   - Loads court corner coordinates for spatial reference

2. **Object Matching** (`tracking/matcher.py`)
   - Matches detected objects between camera views based on class names
   - Handles cases where objects are visible in only one camera

3. **3D Triangulation** (`tracking/triangulator.py`)
   - Computes 3D positions using stereo triangulation
   - Applies coordinate transformations to basketball court space
   - Handles special scaling for ball height estimation

4. **Trajectory Cleaning** (`tracking/cleaner.py`)
   - Removes outliers and false positives
   - Interpolates missing trajectory points using spline fitting
   - Validates trajectory continuity

5. **Metrics Calculation** (`tracking/metrics.py`)
   - Computes movement statistics (speed, acceleration, distance)
   - Removes statistical outliers from speed/acceleration data
   - Calculates court coverage areas and direction changes

6. **Visualization** (`visualization/`)
   - Creates interactive 3D plots with player trajectory filtering
   - Generates comprehensive metrics dashboards
   - Exports results in multiple formats (HTML, CSV, JSON)

### File Interconnections

```
main.py → loader.py → tracking/matcher.py → tracking/triangulator.py 
                                          ↓
visualization/plot_3d.py ← tracking/metrics.py ← tracking/cleaner.py
```

### Coordinate System

- **X-axis**: Court width (side to side)
- **Y-axis**: Height (vertical, 0 for ground level)
- **Z-axis**: Court depth (baseline to baseline)

## Alternative Tracking Approaches

The `other_files/` folder contains experimental tracking methods that were developed to address ball tracking challenges. While our player detection model achieved good accuracy, ball tracking remained problematic, leading us to explore these alternative approaches:

### 1. `ball_hist_match.py` - Multi-Modal Fallback Detection
- Template matching with successful YOLO detections
- HSV histogram comparison for ball appearance
- ORB feature matching with FLANN
- Color-based detection (orange basketball filtering)
- Combined confidence scoring from all methods

### 2. `kalman_tracker.py` - Physics-Enhanced Kalman Filter
- 6-state motion model [x, y, vx, vy, ax, ay]
- Gravity integration (9.8 * 40 pixels/s²)
- Bounce detection with energy damping
- Court boundary constraints

### 3. `particle_histogram_matching.py` - Particle Filter + Appearance
- 500-1500 particles for probabilistic tracking
- Reference histogram from successful detections
- Multi-scale sliding window search
- Gravity-influenced particle motion

### 4. `optical_flow.py` - Enhanced Multi-Modal System
- Kalman filter + Lucas-Kanade optical flow
- Multi-hypothesis tracking
- Physics-based interpolation during gaps
- Velocity prediction for out-of-frame objects

### 5. `csrt_tracker_id.py` - OpenCV CSRT Integration
- CSRT tracker re-initialized with YOLO detections
- Search window constraints (150px left, 300px right)
- 30-frame maximum tracking without detection

### 6. `slicedYolo.py` - SAHI Small Object Detection
- 512x512 image slicing with 20% overlap
- Fallback when standard YOLO fails
- Temporal consistency with distance thresholding

### 7. `run_tracker.py` - Standard YOLO Baseline

### 8. `extract_histograms_from_annotations.py`
- Extracts histogram distribution of the ball from annotated frames

**Note**: Despite extensive experimentation with these approaches, none achieved satisfactory ball tracking results. The final solution adopted the two-model approach in the `2D_tracking/` folder by fine tuning another YOLO model on our augmented dataset of basketballs. The YOLO version used is YOLOv11 nano.

## Output Files

The system generates comprehensive outputs in `outputs/stereo/`:

- **`tracking_2d_results.json`**: 2D tracking data for both cameras
- **`tracking_3d_results.json`**: 3D positions for all tracked objects
- **`trajectory_metrics.json`**: Movement metrics and statistics
- **`interactive_3d_trajectories.html`**: Interactive 3D visualization
- **`interactive_3d_trajectories_metrics.html`**: Metrics dashboard
- **`interactive_3d_trajectories_metrics.csv`**: Metrics in CSV format