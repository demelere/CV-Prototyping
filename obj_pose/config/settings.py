"""
Configuration settings for the 3D pose estimation pipeline.
All parameters can be adjusted here without modifying the core application code.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# ENVIRONMENT VARIABLE CONFIGURATION
# ============================================================================

# Environment variable configuration
# These can be set in a .env file or as environment variables
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', '')
ROBOFLOW_WORKSPACE = os.getenv('ROBOFLOW_WORKSPACE', '')
ROBOFLOW_PROJECT = os.getenv('ROBOFLOW_PROJECT', '')
ROBOFLOW_VERSION = os.getenv('ROBOFLOW_VERSION', '')

# Default values if environment variables are not set
DEFAULT_API_KEY = ROBOFLOW_API_KEY if ROBOFLOW_API_KEY else "your_api_key_here"
DEFAULT_WORKSPACE = ROBOFLOW_WORKSPACE if ROBOFLOW_WORKSPACE else "your_workspace_name"
DEFAULT_PROJECT = ROBOFLOW_PROJECT if ROBOFLOW_PROJECT else "your_project_name"
DEFAULT_VERSION = ROBOFLOW_VERSION if ROBOFLOW_VERSION else "1"

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================

CONFIDENCE_THRESHOLD = 25  # Only show keypoints with this confidence % or higher

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

TARGET_FPS = 15            # Target FPS for processing (lower = faster processing, better motion tracking)
SKIP_FRAMES = True         # Whether to skip frames to match target FPS

# ============================================================================
# VISUAL DISPLAY PARAMETERS
# ============================================================================

KEYPOINT_RADIUS = 8        # Radius of keypoint circles (increased for visibility)
KEYPOINT_THICKNESS = 3     # Thickness of keypoint circles (increased for visibility)
CONNECTION_THICKNESS = 2   # Thickness of connection lines between keypoints
KEYPOINT_COLOR = (0, 255, 0)  # Green keypoints (BGR)
CONNECTION_COLOR = (255, 0, 0)  # Blue connections (BGR)

# ============================================================================
# LABEL DISPLAY OPTIONS
# ============================================================================

SHOW_CONFIDENCE = True             # Show confidence score in label
SHOW_KEYPOINT_NAME = True          # Show keypoint name in label
CONFIDENCE_DECIMAL_PLACES = 2      # Number of decimal places for confidence
LABEL_FORMAT = "keypoint_confidence"  # Options: "keypoint_confidence", "keypoint_only", "confidence_only", "percentage"
LABEL_FONT_SCALE = 1.2             # Font scale for labels (larger)
LABEL_FONT_THICKNESS = 3           # Font thickness for labels (bolder)

# ============================================================================
# 3D VISUALIZATION PARAMETERS
# ============================================================================

SHOW_DEPTH_MAP = True              # Show depth map visualization
SHOW_3D_COORDINATES = True         # Show 3D coordinates in labels
DEPTH_SCALE_FACTOR = 1000.0        # Scale factor for depth values (adjust based on your depth model)
DEPTH_COLOR_MAP = "viridis"        # Color map for depth visualization

# ============================================================================
# CAMERA PARAMETERS
# ============================================================================

# Camera Parameters (you'll need to calibrate these for your camera)
CAMERA_FX = 1512.0  # Focal length in pixels (X direction) - iPhone 15 52mm telephoto
CAMERA_FY = 1512.0  # Focal length in pixels (Y direction) - iPhone 15 52mm telephoto
CAMERA_CX = 1080.0  # Principal point X coordinate (center of 2160px width)
CAMERA_CY = 607.0   # Principal point Y coordinate (center of 1214px height)

# ============================================================================
# DEPTH SCALING PARAMETERS
# ============================================================================

DEPTH_SCALE_METERS = 0.11  # Convert depth values to meters (11cm per unit) - adjusted for electrode length
DEPTH_OFFSET = 0.0          # Depth offset in meters

# ============================================================================
# KEYPOINT CONNECTION CONFIGURATION
# ============================================================================

# Define connections between keypoints (example for pose estimation)
# Format: [(keypoint1_index, keypoint2_index), ...]
KEYPOINT_CONNECTIONS = [
    # Example connections for pose estimation (adjust based on your model)
    # (0, 1), (1, 2), (2, 3),  # Head to neck to shoulders
    # (3, 4), (4, 5), (5, 6),  # Arms
    # (7, 8), (8, 9), (9, 10), # Legs
]

# ============================================================================
# DEPTH MODEL CONFIGURATION
# ============================================================================

DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"  # Smaller, faster model
# Alternative models:
# "depth-anything/Depth-Anything-V2-Base-hf"  # Medium size
# "depth-anything/Depth-Anything-V2-Large-hf"  # Large, most accurate

# ============================================================================
# WORKPIECE DETECTION AND TRACKING PARAMETERS
# ============================================================================

# Workpiece Surface Detection
WORKPIECE_ROI_WIDTH = 0.6    # Region of interest width (fraction of frame width)
WORKPIECE_ROI_HEIGHT = 0.6   # Region of interest height (fraction of frame height)
WORKPIECE_MIN_CLUSTER_SIZE = 100  # Minimum points for valid surface cluster
WORKPIECE_DEPTH_TOLERANCE = 0.02  # Depth tolerance for plane fitting (meters)
WORKPIECE_RANSAC_THRESHOLD = 0.01  # RANSAC threshold for plane fitting (meters)
WORKPIECE_ELECTRODE_EXCLUSION_RADIUS = 0.125  # Electrode exclusion radius (fraction of frame size)

# Travel Direction Tracking
TRAVEL_HISTORY_LENGTH = 10   # Number of frames to track for velocity calculation
TRAVEL_SMOOTHING_FACTOR = 0.7  # Exponential smoothing factor (0-1)
TRAVEL_MIN_VELOCITY = 0.001  # Minimum velocity to consider movement (m/s)
TRAVEL_MAX_VELOCITY = 0.1    # Maximum expected velocity (m/s)

# Speed Calculation
SPEED_UPDATE_INTERVAL = 5    # Update speed display every N frames
SPEED_AVERAGE_WINDOW = 10    # Number of frames for rolling average
SPEED_DISPLAY_UNITS = "mm/s"  # Units for speed display

# Visual Display for Tracking
ARROW_LENGTH_SCALE = 50.0    # Scale factor for velocity arrow length
ARROW_THICKNESS = 3          # Thickness of velocity arrow
ARROW_COLOR = (0, 255, 255)  # Cyan color for velocity arrow (BGR)
COORDINATE_AXIS_LENGTH = 100  # Length of coordinate axis arrows (pixels)
AXIS_THICKNESS = 2           # Thickness of coordinate axes
X_AXIS_COLOR = (0, 0, 255)   # Red for X-axis (BGR)
Y_AXIS_COLOR = (0, 255, 0)   # Green for Y-axis (BGR)
Z_AXIS_COLOR = (255, 0, 0)   # Blue for Z-axis (BGR)

# ============================================================================
# COORDINATE SYSTEM REFERENCE
# ============================================================================

"""
Reference measurements for calibration:
- Typical electrode length: 10-15cm (0.1-0.15m)
- Typical filler rod length: 10-15cm (0.1-0.15m)  
- Camera to workpiece: 20-50cm (0.2-0.5m)
- Electrode tip to workpiece: 2-5mm (0.002-0.005m)

If coordinates seem too small, increase DEPTH_SCALE_METERS
If coordinates seem too large, decrease DEPTH_SCALE_METERS

Coordinate System:
- Origin (0,0,0): Camera optical center
- X: Left/right from camera center (positive = right)
- Y: Up/down from camera center (positive = down)  
- Z: Distance from camera (positive = away from camera)
- Units: Millimeters (mm) for display
"""
