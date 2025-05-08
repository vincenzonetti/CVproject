
# CVPROJECT

## File Structure

CVPROJECT/
├── dataset/ # Local folder (not shown on GitHub)
│ ├── train/ # Contains training frames
│ ├── video/ # Contains video data
│ └── data.yaml # Configuration file for dataset
│
└── kalman_tracker/ # Kalman Tracker module
├── main.py # Entry point to execute the tracker
├── utils.py # Utility functions for the tracker
└── KalmanTracker.py # Kalman Filter tracking implementation

## Directory Details

### `dataset/`
- **train/**: Directory containing image frames used for training or processing.
- **video/**: Directory containing raw video files for object tracking.
- **data.yaml**: YAML configuration file defining dataset paths and metadata.
- **Note**: This folder is excluded from GitHub as it contains local data.

### `kalman_tracker/`
- **main.py**: Run this script to execute the Kalman tracker on the specified input data.
- **utils.py**: Contains helper functions for preprocessing and visualization.
- **KalmanTracker.py**: Implements the Kalman Filter-based tracking logic.
