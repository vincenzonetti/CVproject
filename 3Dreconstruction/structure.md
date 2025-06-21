# Basketball Tracking Project Structure

```
basketball_tracking/
│
├── config/
│   └── constants.py          # All constants and configuration
│
├── data/
│   ├── __init__.py
│   ├── loader.py            # Data loading functions
│   └── models.py            # Data structures/classes
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
│   ├── camera.py           # Camera parameter handling
│   └── court.py            # Court coordinate transformations
│
├── main.py                 # Main entry point
└── README.md              # Project documentation
```

## Module Descriptions:

### `config/constants.py`
- Image dimensions (IMG_WIDTH, IMG_HEIGHT)
- Class names mapping
- Color definitions
- Default parameters (fps, etc.)

### `data/loader.py`
- `load_detections()` - Load and format detection JSON files
- `load_camera_params()` - Load camera calibration parameters
- `load_court_corners()` - Load court corner coordinates

### `data/models.py`
- Data structures for detections, matches, trajectories
- Type hints for better code clarity

### `tracking/matcher.py`
- `match_objects()` - Match objects between camera views
- Matching statistics and debugging

### `tracking/triangulator.py`
- `triangulate_point()` - 3D point triangulation
- Coordinate transformation logic
- Height handling for different object types

### `tracking/metrics.py`
- `calculate_trajectory_metrics()` - Compute all movement metrics
- Speed, acceleration, distance calculations
- Direction change detection

### `visualization/plot_3d.py`
- `create_interactive_3d_plot()` - Generate Plotly 3D visualization
- Player filtering and grouping logic
- Court visualization

### `visualization/dashboard.py`
- `create_metrics_dashboard()` - Generate metrics dashboard
- Statistical plots and comparisons

### `utils/camera.py`
- `compute_projection_matrix()` - Calculate camera projection matrices
- Camera calibration utilities

### `utils/court.py`
- `setup_court_transformation()` - Setup homography matrices
- `transform_to_court_coords()` - Transform points to court coordinates
- Court dimension calculations

### `main.py`
- `StereoTracker` class that orchestrates everything
- Command-line argument parsing
- Main tracking loop