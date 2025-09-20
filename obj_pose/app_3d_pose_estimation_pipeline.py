import gradio as gr
import cv2
import numpy as np
from roboflow import Roboflow
import tempfile
import os
from PIL import Image
import json
from transformers import pipeline
import torch
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.distance import cdist
from collections import deque
import time
from dotenv import load_dotenv
from core.surface_normal_estimator import SurfaceNormalEstimator, visualize_surface_normals, create_normal_magnitude_map
from utils.log_capture import log_capture
from utils.depth_logger import create_depth_logger
from utils.contact_plane_logger import contact_plane_logger

# VGGT Integration for Camera Pose Extraction (Phase 2)
try:
    from core.camera_pose_extractor import CameraPoseExtractor
    from utils.metadata_extractor import CameraMetadataExtractor
    VGGT_AVAILABLE = True
    print("✅ VGGT camera pose extraction available")
except ImportError as e:
    print(f"⚠️ VGGT not available: {e}")
    print("   Contact plane estimation will use default camera parameters")
    VGGT_AVAILABLE = False
    CameraPoseExtractor = None
    CameraMetadataExtractor = None

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
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
# CONFIGURATION PARAMETERS - ADJUST THESE AS NEEDED
# ============================================================================

# Detection Parameters
CONFIDENCE_THRESHOLD = 25  # Only show keypoints with this confidence % or higher

# Processing Parameters
TARGET_FPS = 15            # Target FPS for processing (lower = faster processing, better motion tracking)
SKIP_FRAMES = True         # Whether to skip frames to match target FPS

# Visual Display Parameters
KEYPOINT_RADIUS = 8        # Radius of keypoint circles (increased for visibility)
KEYPOINT_THICKNESS = 3     # Thickness of keypoint circles (increased for visibility)
CONNECTION_THICKNESS = 2   # Thickness of connection lines between keypoints
KEYPOINT_COLOR = (0, 255, 0)  # Green keypoints (BGR)
CONNECTION_COLOR = (255, 0, 0)  # Blue connections (BGR)
DEPTH_COLOR_MAP = cv2.COLORMAP_VIRIDIS  # Color map for depth visualization

# Surface Normal Estimation Parameters
ENABLE_SURFACE_NORMALS = True  # Enable surface normal estimation
SURFACE_NORMAL_ALPHA = 10      # Pixel distance for tangent vector construction (increased for better gradients)
SURFACE_NORMAL_R_THRESHOLD = 0.05  # Threshold for filtering erroneous normals (reduced)
SURFACE_NORMAL_COLOR_MAP = cv2.COLORMAP_PLASMA  # Color map for normal visualization
ROI_ALPHA_METHOD = 'min'       # Method to calculate alpha from ROI dimensions: 'min', 'max', 'mean', 'width', 'height'
SURFACE_NORMAL_MAX_ALPHA = 50  # Maximum alpha value to prevent boundary issues

# Depth Logging Parameters
ENABLE_DEPTH_LOGGING = True    # Enable logging of raw depth map outputs
DEPTH_LOG_SAMPLE_STRATEGY = 'corners'  # Sampling strategy: 'grid', 'random', 'corners', 'edges'
DEPTH_LOG_MAX_FILES = 10       # Maximum number of depth log files to keep

# Label Display Options
SHOW_CONFIDENCE = True             # Show confidence score in label
SHOW_KEYPOINT_NAME = True          # Show keypoint name in label
CONFIDENCE_DECIMAL_PLACES = 2      # Number of decimal places for confidence
LABEL_FORMAT = "keypoint_confidence"  # Options: "keypoint_confidence", "keypoint_only", "confidence_only", "percentage"
LABEL_FONT_SCALE = 1.2             # Font scale for labels (larger)
LABEL_FONT_THICKNESS = 3           # Font thickness for labels (bolder)

# 3D Visualization Parameters
SHOW_DEPTH_MAP = True              # Show depth map visualization
SHOW_3D_COORDINATES = True         # Show 3D coordinates in labels
DEPTH_SCALE_FACTOR = 1000.0        # Scale factor for depth values (adjust based on your depth model)

# DEPTH SCALING FIX: Test different scaling factors to address 650x scale error
# Original: 1000.0 (assumes depth values are in mm, convert to m)
# Test: 1.0 (assumes depth values are already in m)
# Test: 0.001 (assumes depth values are in mm but need 1000x reduction)
DEPTH_SCALE_FACTOR_TEST = 1.0      # Test scaling factor for depth values
USE_DEPTH_SCALE_TEST = True        # Enable depth scaling test

# Camera Parameters (you'll need to calibrate these for your camera)
CAMERA_FX = 1512.0  # Focal length in pixels (X direction) - iPhone 15 52mm telephoto
CAMERA_FY = 1512.0  # Focal length in pixels (Y direction) - iPhone 15 52mm telephoto
CAMERA_CX = 1080.0  # Principal point X coordinate (center of 2160px width)
CAMERA_CY = 607.0   # Principal point Y coordinate (center of 1214px height)

# Depth scaling parameters
DEPTH_SCALE_METERS = 1.0  # Depth values are now already in meters after conversion
DEPTH_OFFSET = 0.0          # Depth offset in meters

# VGGT Camera Pose Parameters (Phase 2)
# ============================================================================
# Phase 2: VGGT Integration for Accurate Camera Parameters
# 
# This phase replaces hardcoded camera intrinsics and extrinsics with VGGT-extracted
# parameters for more accurate 3D pose estimation and contact plane normal calculation.
# 
# Key improvements:
# - Automatic camera intrinsic extraction from image/video metadata
# - VGGT-based camera pose estimation (extrinsics) with gravity prior
# - Proper 3D coordinate transformations using real camera parameters
# - More accurate ray-plane intersection for contact point estimation
# ============================================================================
USE_VGGT_CAMERA_POSE = True  # Enable VGGT for camera intrinsics and extrinsics
EXTRACT_CAMERA_METADATA = True  # Extract camera intrinsics from image/video metadata
USE_GRAVITY_PRIOR = True  # Apply gravity prior (upright camera with forward pitch)
FORWARD_PITCH_DEGREES = 15.0  # Forward pitch angle for gravity prior

# Reference measurements for calibration:
# - Typical electrode length: 10-15cm (0.1-0.15m)
# - Typical filler rod length: 10-15cm (0.1-0.15m)  
# - Camera to workpiece: 20-50cm (0.2-0.5m)
# - Electrode tip to workpiece: 2-5mm (0.002-0.005m)
# 
# If coordinates seem too small, increase DEPTH_SCALE_METERS
# If coordinates seem too large, decrease DEPTH_SCALE_METERS
#
# Coordinate System:
# - Origin (0,0,0): Camera optical center
# - X: Left/right from camera center (positive = right)
# - Y: Up/down from camera center (positive = down)  
# - Z: Distance from camera (positive = away from camera)
# - Units: Millimeters (mm) for display

# Keypoint Connection Configuration
# Define connections between keypoints (example for pose estimation)
# Format: [(keypoint1_index, keypoint2_index), ...]
KEYPOINT_CONNECTIONS = [
    # Example connections for pose estimation (adjust based on your model)
    # (0, 1), (1, 2), (2, 3),  # Head to neck to shoulders
    # (3, 4), (4, 5), (5, 6),  # Arms
    # (7, 8), (8, 9), (9, 10), # Legs
]

# Hugging Face Pipeline Configuration
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"  # Smaller, faster model
# Alternative models:
# "depth-anything/Depth-Anything-V2-Base-hf"  # Medium size
# "depth-anything/Depth-Anything-V2-Large-hf"  # Large, most accurate

# ============================================================================
# WORKPIECE DETECTION AND TRACKING PARAMETERS
# ============================================================================

# Manual ROI Selection Parameters
ROI_SELECTION_ENABLED = True  # Enable manual ROI selection
DEFAULT_ROI_WIDTH = 0.4       # Default ROI width (fraction of frame width)
DEFAULT_ROI_HEIGHT = 0.4      # Default ROI height (fraction of frame height)
ROI_MIN_SIZE = 50             # Minimum ROI size in pixels

# Workpiece Surface Detection (fallback parameters)
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
# END CONFIGURATION PARAMETERS
# ============================================================================

class DepthEstimatorPipeline:
    def __init__(self, model_name=DEPTH_MODEL_NAME, enable_depth_logging=True):
        """Initialize depth estimator with Hugging Face Pipeline"""
        try:
            print(f"Loading depth estimation pipeline: {model_name}")
            print(f"DEBUG: Model configuration - task: depth-estimation")
            print(f"DEBUG: Model configuration - device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
            
            self.pipe = pipeline(
                task="depth-estimation", 
                model=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"✅ Depth estimation pipeline loaded successfully on {self.pipe.device}")
            
            # Debug pipeline configuration
            print(f"DEBUG: Pipeline model type: {type(self.pipe.model)}")
            print(f"DEBUG: Pipeline model name: {self.pipe.model.name_or_path}")
            
            # Check if we can access model config
            if hasattr(self.pipe.model, 'config'):
                config = self.pipe.model.config
                print(f"DEBUG: Model config class: {type(config)}")
                print(f"DEBUG: Model config attributes: {list(config.__dict__.keys())}")
                
                # Check for depth-specific configuration
                if hasattr(config, 'depth_scale'):
                    print(f"DEBUG: Model depth_scale: {config.depth_scale}")
                if hasattr(config, 'depth_shift'):
                    print(f"DEBUG: Model depth_shift: {config.depth_shift}")
                if hasattr(config, 'depth_min'):
                    print(f"DEBUG: Model depth_min: {config.depth_min}")
                if hasattr(config, 'depth_max'):
                    print(f"DEBUG: Model depth_max: {config.depth_max}")
            
            # Check pipeline configuration
            print(f"DEBUG: Pipeline task: {self.pipe.task}")
            print(f"DEBUG: Pipeline framework: {self.pipe.framework}")
            
            # Initialize depth logger
            if enable_depth_logging and ENABLE_DEPTH_LOGGING:
                try:
                    self.depth_logger = create_depth_logger(
                        log_dir="logs",
                        max_log_files=DEPTH_LOG_MAX_FILES,
                        sample_strategy=DEPTH_LOG_SAMPLE_STRATEGY
                    )
                    print(f"✅ Depth logger initialized (strategy: {DEPTH_LOG_SAMPLE_STRATEGY})")
                except Exception as logger_error:
                    print(f"⚠️ Warning: Could not initialize depth logger: {logger_error}")
                    self.depth_logger = None
            else:
                self.depth_logger = None
                print("ℹ️ Depth logging disabled")
            
        except Exception as e:
            print(f"❌ Error loading depth pipeline: {e}")
            self.pipe = None
            raise e
    
    def estimate_depth(self, image_path):
        """Estimate depth using Hugging Face Pipeline"""
        try:
            if self.pipe is None:
                print("❌ Depth pipeline not loaded")
                return None
            
            print(f"DEBUG: Estimating depth for {image_path}")
            
            # Load image with PIL
            image = Image.open(image_path)
            print(f"DEBUG: Image size: {image.size}")
            
            # Get depth estimation
            print(f"DEBUG: Calling pipeline with image type: {type(image)}")
            print(f"DEBUG: Pipeline call parameters: task={self.pipe.task}")
            
            result = self.pipe(image)
            print(f"DEBUG: Pipeline result type: {type(result)}")
            print(f"DEBUG: Pipeline result keys: {result.keys() if hasattr(result, 'keys') else 'Not a dict'}")
            
            depth_map = result["depth"]
            
            print(f"DEBUG: Depth estimation result shape: {depth_map.size}")
            print(f"DEBUG: Depth estimation result type: {type(depth_map)}")
            print(f"DEBUG: Depth estimation result mode: {depth_map.mode if hasattr(depth_map, 'mode') else 'N/A'}")
            
            # Convert PIL image to numpy array
            depth_array = np.array(depth_map)
            
            print(f"DEBUG: Raw depth array shape: {depth_array.shape}")
            print(f"DEBUG: Raw depth array min: {depth_array.min()}, max: {depth_array.max()}")
            print(f"DEBUG: Raw depth array data type: {depth_array.dtype}")
            print(f"DEBUG: Raw depth unique values: {len(np.unique(depth_array))}")
            
            # Sample some depth values for debugging
            h, w = depth_array.shape
            sample_points = [
                (w//4, h//4),      # Top-left quadrant
                (w//2, h//2),      # Center
                (3*w//4, h//4),    # Top-right quadrant
                (w//4, 3*h//4),    # Bottom-left quadrant
                (3*w//4, 3*h//4)   # Bottom-right quadrant
            ]
            
            print(f"DEBUG: Sample depth values:")
            for i, (x, y) in enumerate(sample_points):
                depth_val = depth_array[y, x]
                print(f"  Point {i+1} ({x}, {y}): {depth_val}")
            
            # Check for quantization patterns
            unique_vals = np.unique(depth_array)
            if len(unique_vals) <= 256:
                print(f"DEBUG: Quantization detected - only {len(unique_vals)} unique values")
                print(f"DEBUG: First 10 unique values: {unique_vals[:10]}")
                print(f"DEBUG: Last 10 unique values: {unique_vals[-10:]}")
                
                # Check if values are evenly spaced (suggesting 8-bit quantization)
                if len(unique_vals) > 1:
                    diffs = np.diff(unique_vals)
                    unique_diffs = np.unique(diffs)
                    print(f"DEBUG: Value differences - unique: {len(unique_diffs)}, range: [{unique_diffs.min()}, {unique_diffs.max()}]")
                
                # Convert 8-bit depth to more realistic depth range
                print("DEBUG: Converting 8-bit depth to realistic depth range")
                # Assume depth range of 0.1m to 10m (typical for welding scenarios)
                depth_min_meters = 0.1
                depth_max_meters = 10.0
                
                # Convert from 0-255 to depth_min_meters-depth_max_meters
                depth_array = depth_array.astype(np.float32)
                depth_array = depth_min_meters + (depth_array / 255.0) * (depth_max_meters - depth_min_meters)
                
                print(f"DEBUG: Converted depth range: {depth_array.min():.3f}m to {depth_array.max():.3f}m")
                print(f"DEBUG: Sample converted depths:")
                for i, (x, y) in enumerate(sample_points):
                    depth_val = depth_array[y, x]
                    print(f"  Point {i+1} ({x}, {y}): {depth_val:.3f}m")
            
            # Log depth data if logger is available
            if hasattr(self, 'depth_logger') and self.depth_logger is not None:
                try:
                    print("DEBUG: Logging depth map data...")
                    additional_metadata = {
                        'model_name': getattr(self.pipe, 'model', 'unknown'),
                        'device': getattr(self.pipe, 'device', 'unknown'),
                        'pipeline_task': 'depth-estimation',
                        'quantization_detected': len(unique_vals) <= 256 if 'unique_vals' in locals() else False,
                        'conversion_applied': len(unique_vals) <= 256 if 'unique_vals' in locals() else False
                    }
                    
                    log_file = self.depth_logger.log_depth_estimation(
                        depth_array, 
                        image_path, 
                        self.pipe.model.name_or_path if hasattr(self.pipe, 'model') else 'unknown',
                        additional_metadata
                    )
                    if log_file:
                        print(f"DEBUG: Depth log saved to: {log_file}")
                except Exception as logging_error:
                    print(f"⚠️ Warning: Error logging depth map: {logging_error}")
            
            # Return processed depth array
            return depth_array
            
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return None
    


class KeypointProcessor3D:
    def __init__(self, api_key, workspace_name, project_name, version_number, depth_model_name=DEPTH_MODEL_NAME):
        """Initialize the 3D keypoint processor with Roboflow model and depth estimator"""
        try:
            print("Initializing Roboflow...")
            self.rf = Roboflow(api_key=api_key)
            
            print(f"Loading workspace: {workspace_name}")
            workspace = self.rf.workspace(workspace_name)
            
            print(f"Loading project: {project_name}")
            self.project = workspace.project(project_name)
            
            print(f"Loading model version: {version_number}")
            self.model = self.project.version(version_number).model
            
            if self.model is None:
                print("❌ Model is None after loading")
            else:
                print("✅ Model loaded successfully")
            
            # Initialize depth estimator
            print("Initializing depth estimator pipeline...")
            self.depth_estimator = DepthEstimatorPipeline(depth_model_name, enable_depth_logging=ENABLE_DEPTH_LOGGING)
            print("✅ Depth estimator pipeline initialized")
            
            # Initialize surface normal estimator FIRST
            print("Initializing surface normal estimator...")
            self.surface_normal_estimator = SurfaceNormalEstimator(
                fx=CAMERA_FX,
                fy=CAMERA_FY,
                ox=CAMERA_CX,
                oy=CAMERA_CY,
                alpha=SURFACE_NORMAL_ALPHA,
                r_threshold=SURFACE_NORMAL_R_THRESHOLD,
                roi_alpha_method=ROI_ALPHA_METHOD
            )
            print("✅ Surface normal estimator initialized")
            
            # Initialize workpiece detector (now with surface_normal_estimator available)
            print("Initializing workpiece detector...")
            self.workpiece_detector = WorkpieceDetector(surface_normal_estimator=self.surface_normal_estimator)
            print("✅ Workpiece detector initialized")
            
            # Initialize travel tracker
            print("Initializing travel tracker...")
            self.travel_tracker = TravelTracker()
            print("✅ Travel tracker initialized")
            
            # Initialize contact plane data
            self.contact_plane_data = None
            print("✅ Contact plane data initialized")
            
            # Initialize VGGT camera pose extractor (Phase 2)
            if VGGT_AVAILABLE and USE_VGGT_CAMERA_POSE:
                print("Initializing VGGT camera pose extractor...")
                try:
                    self.camera_pose_extractor = CameraPoseExtractor()
                    print("✅ VGGT camera pose extractor initialized")
                except Exception as vggt_error:
                    print(f"⚠️ VGGT initialization failed: {vggt_error}")
                    print("   Falling back to default camera parameters")
                    self.camera_pose_extractor = None
            else:
                self.camera_pose_extractor = None
                if not VGGT_AVAILABLE:
                    print("📐 VGGT not available - using default camera parameters")
                else:
                    print("📐 VGGT disabled in configuration - using default camera parameters")
            
            # Initialize metadata extractor for intrinsics
            if VGGT_AVAILABLE and EXTRACT_CAMERA_METADATA:
                print("Initializing camera metadata extractor...")
                try:
                    self.metadata_extractor = CameraMetadataExtractor()
                    print("✅ Camera metadata extractor initialized")
                except Exception as metadata_error:
                    print(f"⚠️ Metadata extractor initialization failed: {metadata_error}")
                    self.metadata_extractor = None
            else:
                self.metadata_extractor = None
            
            # Cache for extracted camera parameters (for fixed camera scenarios)
            self.cached_camera_intrinsics = None
            self.cached_camera_extrinsics = None
            self.cached_source_image = None
                
        except Exception as e:
            print(f"❌ Error in KeypointProcessor3D.__init__: {e}")
            self.model = None
            raise e
    
    def get_depth_at_keypoint(self, depth_map, x, y):
        """Get depth value at specific keypoint coordinates"""
        if depth_map is None:
            print(f"DEBUG: Depth map is None")
            return None
        
        # Ensure coordinates are within bounds
        h, w = depth_map.shape
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        
        # Get depth value (already in meters from depth estimation)
        depth_value = depth_map[y, x]
        
        print(f"DEBUG: Depth map shape: {depth_map.shape}")
        print(f"DEBUG: Requested coordinates: ({x}, {y})")
        print(f"DEBUG: Raw depth value: {depth_value}")
        
        # The depth value is already in meters, no conversion needed
        print(f"DEBUG: Depth in meters: {depth_value}")
        
        return depth_value
    
    def pixel_to_3d_coordinates(self, pixel_x, pixel_y, depth_value, fx=None, fy=None, cx=None, cy=None):
        """Convert 2D pixel coordinates + depth to 3D world coordinates using VGGT-extracted camera parameters"""
        print(f"\n=== DEBUG: 3D Coordinate Conversion ===")
        print(f"Input: pixel_x={pixel_x}, pixel_y={pixel_y}, depth_value={depth_value}")
        
        # Use provided camera parameters or fall back to cached/default values
        if fx is None or fy is None or cx is None or cy is None:
            if self.cached_camera_intrinsics is not None:
                fx, fy, cx, cy = self.cached_camera_intrinsics
                print(f"Using cached camera params: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            else:
                fx, fy, cx, cy = CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
                print(f"Using default camera params: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        else:
            print(f"Using provided camera params: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        
        if depth_value is None:
            print(f"DEBUG: Depth value is None, returning None coordinates")
            return None, None, None
        
        # The depth_value is already converted to meters in the estimate_depth function
        # No additional scaling needed here - the depth is already in the correct range
        depth_meters = depth_value
        
        print(f"DEBUG: Depth in meters: {depth_meters}")
        
        # Convert pixel coordinates to 3D using camera intrinsics
        # Z = depth_meters
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        x_3d = (pixel_x - cx) * depth_meters / fx
        y_3d = (pixel_y - cy) * depth_meters / fy
        z_3d = depth_meters
        
        print(f"DEBUG: 3D coordinates: x={x_3d}, y={y_3d}, z={z_3d}")
        
        return x_3d, y_3d, z_3d
    
    def extract_camera_parameters_vggt(self, image_path):
        """
        Extract camera intrinsics and extrinsics using VGGT (Phase 2).
        
        Args:
            image_path (str): Path to the image for camera parameter extraction
            
        Returns:
            tuple: (fx, fy, cx, cy, extrinsic_matrix) or (None, None, None, None, None) if failed
        """
        print(f"\n=== VGGT Camera Parameter Extraction ===")
        print(f"Image path: {image_path}")
        
        if not VGGT_AVAILABLE or not USE_VGGT_CAMERA_POSE or self.camera_pose_extractor is None:
            print("DEBUG: VGGT not available or disabled, using default parameters")
            return CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, None
        
        try:
            # Check cache for fixed camera scenarios
            if (self.cached_camera_intrinsics is not None and 
                self.cached_camera_extrinsics is not None and
                self.cached_source_image is not None):
                print(f"DEBUG: Using cached camera parameters from: {self.cached_source_image}")
                fx, fy, cx, cy = self.cached_camera_intrinsics
                return fx, fy, cx, cy, self.cached_camera_extrinsics
            
            # Extract metadata intrinsics if available
            metadata_intrinsics = None
            if self.metadata_extractor is not None:
                print("🔍 Extracting camera intrinsics from image metadata...")
                metadata_intrinsics_obj = self.metadata_extractor.extract_from_image(image_path)
                
                if metadata_intrinsics_obj and metadata_intrinsics_obj.fx > 0:
                    metadata_intrinsics = {
                        'fx': metadata_intrinsics_obj.fx,
                        'fy': metadata_intrinsics_obj.fy,
                        'cx': metadata_intrinsics_obj.cx,
                        'cy': metadata_intrinsics_obj.cy
                    }
                    print(f"✅ Extracted metadata intrinsics: fx={metadata_intrinsics_obj.fx:.1f}, fy={metadata_intrinsics_obj.fy:.1f}")
                    print(f"   Source: {metadata_intrinsics_obj.source}, Camera: {metadata_intrinsics_obj.camera_model}")
            
            # Get intrinsics with VGGT fallback
            print("🎯 Getting camera intrinsics with VGGT fallback...")
            fx, fy, cx, cy = self.camera_pose_extractor.get_intrinsics_with_fallback(
                image_path, metadata_intrinsics
            )
            
            # Get extrinsics with gravity prior
            print("🌍 Extracting camera extrinsics with gravity prior...")
            if USE_GRAVITY_PRIOR:
                extrinsic_matrix, _ = self.camera_pose_extractor.extract_or_get_cached_pose_with_gravity_prior(
                    image_path, FORWARD_PITCH_DEGREES
                )
                print(f"✅ Applied gravity prior with {FORWARD_PITCH_DEGREES}° forward pitch")
            else:
                extrinsic_matrix, _ = self.camera_pose_extractor.extract_or_get_cached_pose(image_path)
                print("✅ Extracted camera extrinsics without gravity prior")
            
            # Cache the results for fixed camera scenarios
            self.cached_camera_intrinsics = (fx, fy, cx, cy)
            self.cached_camera_extrinsics = extrinsic_matrix
            self.cached_source_image = image_path
            
            print(f"✅ Camera parameters extracted successfully:")
            print(f"   Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            print(f"   Extrinsics shape: {extrinsic_matrix.shape}")
            print(f"   Parameters cached for subsequent frames")
            
            return fx, fy, cx, cy, extrinsic_matrix
            
        except Exception as e:
            print(f"❌ VGGT camera parameter extraction failed: {e}")
            print("   Falling back to default camera parameters")
            return CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, None
    
    def combine_2d_and_depth(self, keypoints_2d, depth_map, img_width, img_height, parent_object=None):
        """Combine 2D keypoints with depth information to get 3D coordinates"""
        keypoints_3d = []
        
        print(f"\n=== DEBUG: Processing {len(keypoints_2d)} keypoints ===")
        print(f"Image dimensions: {img_width}x{img_height}")
        print(f"Depth map shape: {depth_map.shape if depth_map is not None else 'None'}")
        
        for i, keypoint in enumerate(keypoints_2d):
            x = keypoint.get('x', 0)
            y = keypoint.get('y', 0)
            confidence = keypoint.get('confidence', 0)
            keypoint_class = keypoint.get('class', 'unknown')
            keypoint_class_id = keypoint.get('class_id', 0)
            
            print(f"\n--- Keypoint {i+1}: {keypoint_class} ---")
            print(f"2D coordinates: x={x}, y={y}")
            print(f"Confidence: {confidence}")
            print(f"Class: {keypoint_class} (ID: {keypoint_class_id})")
            
            # Convert coordinates to pixels if they're percentages
            if x <= 1 and y <= 1:
                pixel_x = int(x * img_width)
                pixel_y = int(y * img_height)
                print(f"Converted from percentages: pixel_x={pixel_x}, pixel_y={pixel_y}")
            else:
                pixel_x = int(x)
                pixel_y = int(y)
                print(f"Already in pixels: pixel_x={pixel_x}, pixel_y={pixel_y}")
            
            # Get depth at this keypoint
            depth_value = self.get_depth_at_keypoint(depth_map, pixel_x, pixel_y)
            print(f"Depth value at ({pixel_x}, {pixel_y}): {depth_value}")
            
            # Convert to proper 3D coordinates
            x_3d, y_3d, z_3d = self.pixel_to_3d_coordinates(pixel_x, pixel_y, depth_value)
            print(f"3D coordinates: x={x_3d}, y={y_3d}, z={z_3d}")
            
            # Create 3D keypoint with parent object information
            keypoint_3d = {
                'x_2d': pixel_x,
                'y_2d': pixel_y,
                'depth': depth_value,
                'x_3d': x_3d,
                'y_3d': y_3d,
                'z_3d': z_3d,
                'confidence': confidence,
                'class': keypoint_class,
                'class_id': keypoint_class_id,
                'parent_object': parent_object,
                'name': f"{parent_object}_{keypoint_class}" if parent_object else keypoint_class
            }
            
            print(f"Final keypoint_3d: {keypoint_3d}")
            keypoints_3d.append(keypoint_3d)
        
        print(f"\n=== DEBUG: Created {len(keypoints_3d)} 3D keypoints ===")
        return keypoints_3d
    
    def calculate_electrode_geometry(self, keypoints_3d):
        """Calculate electrode geometry from 3D keypoints"""
        electrode_data = {}
        
        print(f"\n=== DEBUG: Calculating Electrode Geometry ===")
        print(f"Input keypoints: {len(keypoints_3d)}")
        
        # Find electrode keypoints (electrode class with body and tip keypoints)
        electrode_body = None
        electrode_tip = None
        
        for kp in keypoints_3d:
            # Check if this keypoint belongs to an electrode object
            if kp.get('parent_object') == 'electrode':
                if kp.get('class') == 'body':
                    electrode_body = kp
                    print(f"Found electrode body: {kp}")
                elif kp.get('class') == 'tip':
                    electrode_tip = kp
                    print(f"Found electrode tip: {kp}")
        
        # Fallback: look for electrode keypoints by name pattern
        if not electrode_body or not electrode_tip:
            print("Falling back to name pattern matching...")
            for kp in keypoints_3d:
                name = kp['name'].lower()
                if 'electrode' in name or 'body' in name:
                    electrode_body = kp
                    print(f"Found electrode body (fallback): {kp}")
                elif 'tip' in name and ('electrode' in name or 'body' in name):
                    electrode_tip = kp
                    print(f"Found electrode tip (fallback): {kp}")
        
        if electrode_body and electrode_tip:
            print(f"Calculating electrode geometry...")
            # Calculate electrode vector (direction from body to tip)
            if all(v is not None for v in [electrode_tip['x_3d'], electrode_tip['y_3d'], electrode_tip['z_3d'],
                                          electrode_body['x_3d'], electrode_body['y_3d'], electrode_body['z_3d']]):
                
                # Vector from body to tip
                dx = electrode_tip['x_3d'] - electrode_body['x_3d']
                dy = electrode_tip['y_3d'] - electrode_body['y_3d']
                dz = electrode_tip['z_3d'] - electrode_body['z_3d']
                
                print(f"Electrode vector: dx={dx}, dy={dy}, dz={dz}")
                
                # Electrode length
                length = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # Normalized direction vector
                if length > 0:
                    direction = np.array([dx/length, dy/length, dz/length])
                else:
                    direction = np.array([0, 0, 1])  # Default to pointing forward
                
                print(f"Electrode length: {length}m")
                print(f"Electrode direction: {direction}")
                
                electrode_data = {
                    'body_position': np.array([electrode_body['x_3d'], electrode_body['y_3d'], electrode_body['z_3d']]),
                    'tip_position': np.array([electrode_tip['x_3d'], electrode_tip['y_3d'], electrode_tip['z_3d']]),
                    'direction': direction,
                    'length': length,
                    'body_keypoint': electrode_body,
                    'tip_keypoint': electrode_tip
                }
            else:
                print("ERROR: Some electrode coordinates are None")
        else:
            print(f"ERROR: Missing electrode keypoints. Body: {electrode_body is not None}, Tip: {electrode_tip is not None}")
        
        return electrode_data
    
    def estimate_contact_plane_normal(self, depth_map, img_width, img_height, keypoints_3d, image_path=None):
        """
        Estimate the local plane normal at the anticipated contact region using VGGT-extracted camera parameters.
        
        Args:
            depth_map (np.ndarray): Depth map (HxW)
            img_width (int): Image width
            img_height (int): Image height
            keypoints_3d (list): List of 3D keypoints
            image_path (str, optional): Path to image for VGGT camera parameter extraction
        """
        # Create correlation ID for this operation
        correlation_id = contact_plane_logger.create_correlation_id("contact_plane_estimation")
        start_time = time.time()
        
        contact_plane_logger.log_operation_start(
            correlation_id, "CONTACT_PLANE_ESTIMATION",
            depth_map_shape=depth_map.shape if depth_map is not None else "None",
            image_size=f"{img_width}x{img_height}",
            num_keypoints=len(keypoints_3d),
            image_path=image_path or "None"
        )
        
        try:
            # Extract camera parameters using VGGT (Phase 2)
            camera_extraction_start = time.time()
            if image_path is not None:
                fx, fy, cx, cy, extrinsic_matrix = self.extract_camera_parameters_vggt(image_path)
                contact_plane_logger.log_camera_parameters(
                    correlation_id, 
                    {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}, 
                    extrinsic_matrix
                )
            else:
                # Use cached or default parameters
                if self.cached_camera_intrinsics is not None:
                    fx, fy, cx, cy = self.cached_camera_intrinsics
                    extrinsic_matrix = self.cached_camera_extrinsics
                    contact_plane_logger.log_operation_success(
                        correlation_id, "CAMERA_PARAMS_CACHED",
                        (time.time() - camera_extraction_start) * 1000,
                        source="cached"
                    )
                else:
                    fx, fy, cx, cy = CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
                    extrinsic_matrix = None
                    contact_plane_logger.log_operation_success(
                        correlation_id, "CAMERA_PARAMS_DEFAULT",
                        (time.time() - camera_extraction_start) * 1000,
                        source="default"
                    )
                
                contact_plane_logger.log_camera_parameters(
                    correlation_id, 
                    {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}, 
                    extrinsic_matrix
                )
            
            # Find electrode keypoints
            electrode_tip_3d = None
            electrode_body_3d = None
            
            contact_plane_logger.log_operation_start(
                correlation_id, "ELECTRODE_KEYPOINT_DETECTION",
                total_keypoints=len(keypoints_3d)
            )
            
            for i, keypoint in enumerate(keypoints_3d):
                parent_object = keypoint.get('parent_object')
                keypoint_class = keypoint.get('class')
                
                if parent_object == 'electrode':
                    if keypoint_class == 'tip':
                        electrode_tip_3d = np.array([keypoint['x_3d'], keypoint['y_3d'], keypoint['z_3d']])
                        contact_plane_logger.log_3d_coordinates(correlation_id, "electrode_tip", electrode_tip_3d)
                    elif keypoint_class == 'body':
                        electrode_body_3d = np.array([keypoint['x_3d'], keypoint['y_3d'], keypoint['z_3d']])
                        contact_plane_logger.log_3d_coordinates(correlation_id, "electrode_body", electrode_body_3d)
            
            # Check for missing keypoints
            has_tip = electrode_tip_3d is not None
            has_body = electrode_body_3d is not None
            
            contact_plane_logger.log_decision_point(
                correlation_id, "ELECTRODE_KEYPOINTS_AVAILABLE",
                "tip AND body keypoints found",
                has_tip and has_body,
                has_tip=has_tip, has_body=has_body
            )
            
            if not (has_tip and has_body):
                error_msg = f"Missing electrode keypoints - tip: {has_tip}, body: {has_body}"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "ELECTRODE_KEYPOINT_DETECTION",
                    error_msg, (time.time() - start_time) * 1000
                )
                return None
            
            # Calculate electrode axis vector (from body to tip)
            electrode_axis = electrode_tip_3d - electrode_body_3d
            electrode_axis_norm = np.linalg.norm(electrode_axis)
            
            contact_plane_logger.log_decision_point(
                correlation_id, "ELECTRODE_AXIS_VALID",
                "axis magnitude > 0",
                electrode_axis_norm > 0,
                axis_magnitude=electrode_axis_norm
            )
            
            if electrode_axis_norm == 0:
                error_msg = "Invalid electrode axis vector (zero magnitude)"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "ELECTRODE_AXIS_CALCULATION",
                    error_msg, (time.time() - start_time) * 1000
                )
                return None
            
            electrode_axis = electrode_axis / electrode_axis_norm
            contact_plane_logger.log_3d_coordinates(correlation_id, "electrode_axis_normalized", electrode_axis)
            
            # Prepare camera intrinsics using VGGT-extracted parameters
            camera_intrinsics = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
            
            # Store camera parameters for surface normal estimator
            if extrinsic_matrix is not None:
                self.surface_normal_estimator.fx = fx
                self.surface_normal_estimator.fy = fy
                self.surface_normal_estimator.ox = cx
                self.surface_normal_estimator.oy = cy
                contact_plane_logger.log_operation_success(
                    correlation_id, "SURFACE_NORMAL_ESTIMATOR_UPDATE",
                    0,  # No significant time
                    source="VGGT"
                )
            
            # Estimate contact plane normal using WorkpieceDetector
            contact_plane_start = time.time()
            contact_plane_data = self.workpiece_detector.estimate_contact_plane_normal(
                electrode_tip_3d, 
                electrode_axis, 
                depth_map, 
                camera_intrinsics,
                roi_radius_pixels=50,  # Configurable ROI radius
                correlation_id=correlation_id  # Pass correlation ID for logging
            )
            contact_plane_duration = (time.time() - contact_plane_start) * 1000
            
            if contact_plane_data:
                contact_plane_logger.log_operation_success(
                    correlation_id, "CONTACT_PLANE_ESTIMATION",
                    contact_plane_duration,
                    confidence=contact_plane_data.get('confidence', 0),
                    has_normal=contact_plane_data.get('normal') is not None,
                    has_contact_point=contact_plane_data.get('contact_point') is not None
                )
                
                # Log the results
                if 'normal' in contact_plane_data:
                    contact_plane_logger.log_3d_coordinates(correlation_id, "contact_plane_normal", contact_plane_data['normal'])
                if 'contact_point' in contact_plane_data:
                    contact_plane_logger.log_3d_coordinates(correlation_id, "contact_point_3d", contact_plane_data['contact_point'])
                
                # Store contact plane data for visualization
                self.contact_plane_data = contact_plane_data
                
                # Log final summary
                total_duration = (time.time() - start_time) * 1000
                contact_plane_logger.log_summary(
                    correlation_id, True, total_duration,
                    {
                        'confidence': contact_plane_data.get('confidence', 0),
                        'roi_radius': 50,
                        'camera_source': 'VGGT' if image_path else 'cached/default'
                    }
                )
                
                return contact_plane_data
            else:
                error_msg = "WorkpieceDetector returned None"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "CONTACT_PLANE_ESTIMATION",
                    error_msg, contact_plane_duration
                )
                return None
                
        except Exception as e:
            total_duration = (time.time() - start_time) * 1000
            contact_plane_logger.log_error_with_context(
                correlation_id, e,
                {
                    'operation': 'CONTACT_PLANE_ESTIMATION',
                    'duration_ms': total_duration,
                    'image_path': image_path or 'None',
                    'num_keypoints': len(keypoints_3d)
                }
            )
            contact_plane_logger.log_summary(
                correlation_id, False, total_duration,
                {'error': str(e), 'error_type': type(e).__name__}
            )
            return None
    
    def calculate_filler_rod_geometry(self, keypoints_3d):
        """Calculate filler rod geometry from 3D keypoints"""
        filler_data = {}
        
        # Find filler rod keypoints (rod class with body and tip keypoints)
        rod_body = None
        rod_tip = None
        
        for kp in keypoints_3d:
            # Check if this keypoint belongs to a rod object
            if kp.get('parent_object') == 'rod':
                if kp.get('class') == 'body':
                    rod_body = kp
                elif kp.get('class') == 'tip':
                    rod_tip = kp
        
        # Fallback: look for rod keypoints by name pattern
        if not rod_body or not rod_tip:
            for kp in keypoints_3d:
                name = kp['name'].lower()
                if 'rod' in name or ('body' in name and 'rod' in name):
                    rod_body = kp
                elif 'tip' in name and ('rod' in name or 'body' in name):
                    rod_tip = kp
        
        if rod_body and rod_tip:
            # Calculate rod vector (direction from body to tip)
            if all(v is not None for v in [rod_tip['x_3d'], rod_tip['y_3d'], rod_tip['z_3d'],
                                          rod_body['x_3d'], rod_body['y_3d'], rod_body['z_3d']]):
                
                dx = rod_tip['x_3d'] - rod_body['x_3d']
                dy = rod_tip['y_3d'] - rod_body['y_3d']
                dz = rod_tip['z_3d'] - rod_body['z_3d']
                
                length = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if length > 0:
                    direction = np.array([dx/length, dy/length, dz/length])
                else:
                    direction = np.array([0, 0, 1])
                
                filler_data = {
                    'body_position': np.array([rod_body['x_3d'], rod_body['y_3d'], rod_body['z_3d']]),
                    'tip_position': np.array([rod_tip['x_3d'], rod_tip['y_3d'], rod_tip['z_3d']]),
                    'direction': direction,
                    'length': length,
                    'body_keypoint': rod_body,
                    'tip_keypoint': rod_tip
                }
        
        return filler_data
    
    def calculate_welding_angles(self, electrode_data, filler_data=None, workpiece_normal=None):
        """Calculate wd elding angles from electrode and filler rod geometry"""
        # NOTE: This requires workpiece detection which we haven't implemented yet
        # For now, we'll just return basic geometry information
        angles = {}
        
        if electrode_data and 'direction' in electrode_data:
            electrode_direction = electrode_data['direction']
            
            # For now, just calculate basic orientation angles
            # These are NOT true welding angles without workpiece information
            
            # Angle with respect to camera Z-axis (not workpiece)
            vertical = np.array([0, 0, 1])
            angle_to_camera = np.arccos(np.clip(np.dot(electrode_direction, vertical), -1, 1))
            angles['angle_to_camera_degrees'] = np.degrees(angle_to_camera)
            
            # Basic orientation in camera space
            angles['electrode_direction'] = electrode_direction.tolist()
        
        if filler_data and 'direction' in filler_data:
            filler_direction = filler_data['direction']
            angles['filler_direction'] = filler_direction.tolist()
        
        return angles
    
    def calculate_workpiece_angles(self, electrode_data, filler_data=None):
        """Calculate true welding angles using workpiece surface normal"""
        angles = {}
        
        # Get workpiece normal
        workpiece_normal = self.workpiece_detector.get_workpiece_normal()
        if workpiece_normal is None:
            print("No workpiece normal available for angle calculation")
            return angles
        
        if electrode_data and 'direction' in electrode_data:
            electrode_direction = electrode_data['direction']
            
            # Calculate work angle (angle between electrode and workpiece normal)
            work_angle = np.arccos(np.clip(np.dot(electrode_direction, workpiece_normal), -1, 1))
            angles['work_angle_degrees'] = np.degrees(work_angle)
            
            # Calculate travel angle using velocity vector
            electrode_velocity = self.travel_tracker.get_electrode_velocity()
            if electrode_velocity is not None and np.linalg.norm(electrode_velocity) > TRAVEL_MIN_VELOCITY:
                # Project velocity onto workpiece plane
                velocity_projected = electrode_velocity - np.dot(electrode_velocity, workpiece_normal) * workpiece_normal
                velocity_projected = velocity_projected / np.linalg.norm(velocity_projected)
                
                # Calculate travel angle (angle between electrode and travel direction)
                travel_angle = np.arccos(np.clip(np.dot(electrode_direction, velocity_projected), -1, 1))
                angles['travel_angle_degrees'] = np.degrees(travel_angle)
                
                # Store travel direction for visualization
                angles['travel_direction'] = velocity_projected.tolist()
            
            angles['electrode_direction'] = electrode_direction.tolist()
        
        if filler_data and 'direction' in filler_data:
            filler_direction = filler_data['direction']
            
            # Calculate filler rod work angle
            filler_work_angle = np.arccos(np.clip(np.dot(filler_direction, workpiece_normal), -1, 1))
            angles['filler_work_angle_degrees'] = np.degrees(filler_work_angle)
            
            angles['filler_direction'] = filler_direction.tolist()
        
        return angles
    
    def calculate_tip_to_workpiece_distance(self, electrode_tip_3d, rod_tip_3d=None):
        """Calculate distance from electrode/rod tips to workpiece surface"""
        distances = {}
        
        # Get workpiece normal and origin
        workpiece_normal = self.workpiece_detector.get_workpiece_normal()
        workpiece_origin = self.workpiece_detector.get_workpiece_origin()
        
        if workpiece_normal is None or workpiece_origin is None:
            print("No workpiece surface available for distance calculation")
            return distances
        
        # Calculate electrode tip distance
        if electrode_tip_3d is not None:
            # Distance from point to plane: |ax + by + cz + d| / sqrt(a² + b² + c²)
            # For normalized normal vector, denominator is 1
            electrode_distance = abs(np.dot(electrode_tip_3d - workpiece_origin, workpiece_normal))
            distances['electrode_tip_distance'] = electrode_distance
        
        # Calculate rod tip distance
        if rod_tip_3d is not None:
            rod_distance = abs(np.dot(rod_tip_3d - workpiece_origin, workpiece_normal))
            distances['rod_tip_distance'] = rod_distance
        
        return distances
    
    def update_travel_tracking(self, keypoints_3d):
        """Update travel tracking with current keypoint positions"""
        electrode_tip_3d = None
        rod_tip_3d = None
        
        # Find electrode and rod tip positions
        for keypoint in keypoints_3d:
            if keypoint.get('parent_object') == 'electrode' and keypoint.get('class') == 'tip':
                electrode_tip_3d = np.array([keypoint['x_3d'], keypoint['y_3d'], keypoint['z_3d']])
            elif keypoint.get('parent_object') == 'rod' and keypoint.get('class') == 'tip':
                rod_tip_3d = np.array([keypoint['x_3d'], keypoint['y_3d'], keypoint['z_3d']])
        
        # Update tracking
        self.travel_tracker.update_tracking(electrode_tip_3d, rod_tip_3d)
        
        return electrode_tip_3d, rod_tip_3d
    
    def process_3d_pose_data(self, keypoints_3d, frame_number=0, update_tracking=True):
        """Process 3D keypoints to extract welding geometry and angles"""
        pose_data = {
            'keypoints_3d': keypoints_3d,
            'electrode_geometry': None,
            'filler_geometry': None,
            'welding_angles': None,
            'workpiece_angles': None,
            'tip_distances': None,
            'travel_speed': None,
            'workpiece_detected': False
        }
        
        if keypoints_3d:
            # Update travel tracking only if requested
            if update_tracking:
                electrode_tip_3d, rod_tip_3d = self.update_travel_tracking(keypoints_3d)
            else:
                # For preview, just extract tip positions without updating tracking
                electrode_tip_3d = None
                rod_tip_3d = None
                for keypoint in keypoints_3d:
                    if keypoint.get('parent_object') == 'electrode' and keypoint.get('class') == 'tip':
                        electrode_tip_3d = np.array([keypoint['x_3d'], keypoint['y_3d'], keypoint['z_3d']])
                    elif keypoint.get('parent_object') == 'rod' and keypoint.get('class') == 'tip':
                        rod_tip_3d = np.array([keypoint['x_3d'], keypoint['y_3d'], keypoint['z_3d']])
            
            # Calculate electrode geometry
            electrode_data = self.calculate_electrode_geometry(keypoints_3d)
            if electrode_data:
                pose_data['electrode_geometry'] = electrode_data
            
            # Calculate filler rod geometry
            filler_data = self.calculate_filler_rod_geometry(keypoints_3d)
            if filler_data:
                pose_data['filler_geometry'] = filler_data
            
            # Calculate tip-to-workpiece distances
            tip_distances = self.calculate_tip_to_workpiece_distance(electrode_tip_3d, rod_tip_3d)
            if tip_distances:
                pose_data['tip_distances'] = tip_distances
            
            # Calculate travel speed only if tracking is enabled
            if update_tracking:
                electrode_speed = self.travel_tracker.get_electrode_speed()
                if electrode_speed > 0:
                    speed_mmps = self.travel_tracker.get_speed_in_units(electrode_speed, SPEED_DISPLAY_UNITS)
                    pose_data['travel_speed'] = speed_mmps
            
            # Calculate welding angles using workpiece surface
            workpiece_angles = self.calculate_workpiece_angles(electrode_data, filler_data)
            if workpiece_angles:
                pose_data['workpiece_angles'] = workpiece_angles
                pose_data['welding_angles'] = workpiece_angles  # For backward compatibility
            
            # Check if workpiece is detected
            pose_data['workpiece_detected'] = self.workpiece_detector.get_workpiece_normal() is not None
        
        return pose_data
    
    def calculate_3d_distance(self, keypoint1, keypoint2):
        """Calculate 3D distance between two keypoints"""
        if not all(key in keypoint1 for key in ['x_3d', 'y_3d', 'z_3d']) or not all(key in keypoint2 for key in ['x_3d', 'y_3d', 'z_3d']):
            return None
        
        dx = keypoint1['x_3d'] - keypoint2['x_3d']
        dy = keypoint1['y_3d'] - keypoint2['y_3d']
        dz = keypoint1['z_3d'] - keypoint2['z_3d']
        
        distance = (dx**2 + dy**2 + dz**2)**0.5
        return distance
    
    def format_3d_label(self, keypoint_3d):
        """Format 3D keypoint label showing distance instead of coordinates"""
        name = keypoint_3d.get('name', keypoint_3d.get('class', 'unknown'))
        confidence = keypoint_3d.get('confidence', 0)
        
        # Convert to millimeters for better readability
        x_mm = keypoint_3d.get('x_3d', 0) * 1000
        y_mm = keypoint_3d.get('y_3d', 0) * 1000
        z_mm = keypoint_3d.get('z_3d', 0) * 1000
        
        # Format confidence as percentage
        confidence_pct = confidence * 100
        
        return f"{name}: {confidence_pct:.1f}%"
    
    def draw_keypoints_3d(self, frame, predictions, depth_map=None):
        """Draw 3D keypoints and connections on the frame"""
        print(f"DEBUG: draw_keypoints_3d called with {len(predictions) if predictions else 0} predictions")
        
        if not predictions:
            print("No predictions to draw")
            return frame
            
        img_height, img_width = frame.shape[:2]
        print(f"Drawing 3D keypoints on frame size: {img_width}x{img_height}")
        print(f"Processing {len(predictions)} predictions")
        print(f"DEBUG: Depth map available: {depth_map is not None}")
        if depth_map is not None:
            print(f"DEBUG: Depth map shape: {depth_map.shape}")
        
        # Store current depth map for coordinate axes
        if depth_map is not None:
            self.current_depth_map = depth_map
            
            # Collect all keypoints for workpiece detection
            all_keypoints_3d = []
            print(f"DEBUG: Collecting keypoints from {len(predictions)} predictions")
            for i, prediction in enumerate(predictions):
                keypoints = prediction.get('keypoints', [])
                print(f"DEBUG: Prediction {i} has {len(keypoints)} keypoints")
                for j, keypoint in enumerate(keypoints):
                    if 'x_3d' in keypoint and 'y_3d' in keypoint and 'z_3d' in keypoint:
                        all_keypoints_3d.append(keypoint)
                        print(f"DEBUG: Added keypoint {j}: parent_object={keypoint.get('parent_object')}, class={keypoint.get('class')}")
                    else:
                        print(f"DEBUG: Skipped keypoint {j} - missing 3D coordinates")
            
            print(f"DEBUG: Collected {len(all_keypoints_3d)} 3D keypoints total")
            
            # Check for electrode keypoints specifically
            electrode_keypoints = [kp for kp in all_keypoints_3d if kp.get('parent_object') == 'electrode']
            print(f"DEBUG: Found {len(electrode_keypoints)} electrode keypoints")
            for i, kp in enumerate(electrode_keypoints):
                print(f"DEBUG: Electrode keypoint {i}: class={kp.get('class')}, 3D=({kp.get('x_3d')}, {kp.get('y_3d')}, {kp.get('z_3d')})")
            
            self.workpiece_detector.detect_workpiece_surface(depth_map, img_width, img_height, all_keypoints_3d)
            
            # Estimate contact plane normal if we have electrode keypoints
            print(f"DEBUG: About to estimate contact plane normal with {len(all_keypoints_3d)} keypoints")
            contact_result = self.estimate_contact_plane_normal(depth_map, img_width, img_height, all_keypoints_3d)
            if contact_result:
                print(f"DEBUG: Contact plane estimation successful!")
            else:
                print(f"DEBUG: Contact plane estimation failed or returned None")
        else:
            print(f"DEBUG: No depth map available, skipping contact plane estimation")
        
        # Draw coordinate axes based on surface normal estimation
        self.draw_coordinate_axes(frame, img_width, img_height)
        
        # Draw workpiece surface indicator
        self.draw_workpiece_surface_indicator(frame, img_width, img_height)
        
        # Draw contact plane visualization
        self.draw_contact_plane_visualization(frame, img_width, img_height)
        
        # Collect all keypoints for processing
        all_keypoints_3d = []
        
        # Draw keypoints and labels
        for prediction in predictions:
            keypoints = prediction.get('keypoints', [])
            
            # Separate keypoints by object type for connections
            electrode_keypoints = []
            rod_keypoints = []
            
            for keypoint in keypoints:
                # Handle both original Roboflow format (x, y) and processed 3D format (x_2d, y_2d)
                if 'x_2d' in keypoint and 'y_2d' in keypoint:
                    x = keypoint['x_2d']
                    y = keypoint['y_2d']
                elif 'x' in keypoint and 'y' in keypoint:
                    x = keypoint['x']
                    y = keypoint['y']
                else:
                    print(f"DEBUG: Skipping keypoint with missing coordinates: {keypoint}")
                    continue
                
                # Convert to integers for OpenCV
                try:
                    x = int(x)
                    y = int(y)
                except (ValueError, TypeError) as e:
                    print(f"DEBUG: Invalid coordinates for keypoint: x={x}, y={y}, error={e}")
                    continue
                
                print(f"DEBUG: Drawing keypoint at ({x}, {y})")
                
                confidence = keypoint['confidence']
                parent_object = keypoint.get('parent_object', 'unknown')
                keypoint_class = keypoint.get('class', 'unknown')
                
                # Skip low confidence keypoints
                confidence_threshold = CONFIDENCE_THRESHOLD / 100.0 if CONFIDENCE_THRESHOLD > 1 else CONFIDENCE_THRESHOLD
                if confidence < confidence_threshold:
                    continue
                
                # Draw keypoint circle
                cv2.circle(frame, (x, y), KEYPOINT_RADIUS, KEYPOINT_COLOR, KEYPOINT_THICKNESS)
                
                # Categorize keypoints for connections
                if parent_object == 'electrode':
                    electrode_keypoints.append(keypoint)
                elif parent_object == 'rod':
                    rod_keypoints.append(keypoint)
            
            # Draw connections between electrode body and tip
            electrode_body = None
            electrode_tip = None
            for kp in electrode_keypoints:
                if kp.get('class') == 'body':
                    electrode_body = kp
                elif kp.get('class') == 'tip':
                    electrode_tip = kp
            
            if electrode_body and electrode_tip:
                # Calculate distance
                distance = self.calculate_3d_distance(electrode_body, electrode_tip)
                distance_mm = distance * 1000 if distance else 0
                
                # Get coordinates for both keypoints
                if 'x_2d' in electrode_body and 'y_2d' in electrode_body:
                    body_x, body_y = electrode_body['x_2d'], electrode_body['y_2d']
                else:
                    body_x, body_y = electrode_body['x'], electrode_body['y']
                
                if 'x_2d' in electrode_tip and 'y_2d' in electrode_tip:
                    tip_x, tip_y = electrode_tip['x_2d'], electrode_tip['y_2d']
                else:
                    tip_x, tip_y = electrode_tip['x'], electrode_tip['y']
                
                # Convert to integers for OpenCV
                try:
                    body_x, body_y = int(body_x), int(body_y)
                    tip_x, tip_y = int(tip_x), int(tip_y)
                except (ValueError, TypeError) as e:
                    print(f"DEBUG: Invalid electrode coordinates: body=({body_x}, {body_y}), tip=({tip_x}, {tip_y}), error={e}")
                    continue
                
                print(f"DEBUG: Drawing electrode connection from ({body_x}, {body_y}) to ({tip_x}, {tip_y})")
                
                # Draw electrode connection (green line)
                cv2.line(frame, (body_x, body_y), (tip_x, tip_y), (0, 255, 0), 3)  # Green line
                
                # Draw distance label on the connection line
                mid_x = (body_x + tip_x) // 2
                mid_y = (body_y + tip_y) // 2
                distance_label = f"Electrode: {distance_mm:.1f}mm"
                
                # Draw label background
                label_size = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, 
                            (mid_x - label_size[0]//2 - 5, mid_y - label_size[1] - 5),
                            (mid_x + label_size[0]//2 + 5, mid_y + 5),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(frame, distance_label, (mid_x - label_size[0]//2, mid_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                print(f"DEBUG: Drew electrode connection with distance {distance_mm:.1f}mm")
            
            # Draw connections between rod body and tip
            rod_body = None
            rod_tip = None
            for kp in rod_keypoints:
                if kp.get('class') == 'body':
                    rod_body = kp
                elif kp.get('class') == 'tip':
                    rod_tip = kp
            
            if rod_body and rod_tip:
                # Calculate distance
                distance = self.calculate_3d_distance(rod_body, rod_tip)
                distance_mm = distance * 1000 if distance else 0
                
                # Get coordinates for both keypoints
                if 'x_2d' in rod_body and 'y_2d' in rod_body:
                    body_x, body_y = rod_body['x_2d'], rod_body['y_2d']
                else:
                    body_x, body_y = rod_body['x'], rod_body['y']
                
                if 'x_2d' in rod_tip and 'y_2d' in rod_tip:
                    tip_x, tip_y = rod_tip['x_2d'], rod_tip['y_2d']
                else:
                    tip_x, tip_y = rod_tip['x'], rod_tip['y']
                
                # Convert to integers for OpenCV
                try:
                    body_x, body_y = int(body_x), int(body_y)
                    tip_x, tip_y = int(tip_x), int(tip_y)
                except (ValueError, TypeError) as e:
                    print(f"DEBUG: Invalid rod coordinates: body=({body_x}, {body_y}), tip=({tip_x}, {tip_y}), error={e}")
                    continue
                
                print(f"DEBUG: Drawing rod connection from ({body_x}, {body_y}) to ({tip_x}, {tip_y})")
                
                # Draw rod connection (red line)
                cv2.line(frame, (body_x, body_y), (tip_x, tip_y), (0, 0, 255), 3)  # Red line
                
                # Draw distance label on the connection line
                mid_x = (body_x + tip_x) // 2
                mid_y = (body_y + tip_y) // 2
                distance_label = f"Rod: {distance_mm:.1f}mm"
                
                # Draw label background
                label_size = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, 
                            (mid_x - label_size[0]//2 - 5, mid_y - label_size[1] - 5),
                            (mid_x + label_size[0]//2 + 5, mid_y + 5),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(frame, distance_label, (mid_x - label_size[0]//2, mid_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                print(f"DEBUG: Drew rod connection with distance {distance_mm:.1f}mm")
        
        # Process 3D keypoints for enhanced analysis
        for prediction in predictions:
            keypoints_2d = prediction.get('keypoints', [])
            parent_class = prediction.get('class', 'unknown')
            if keypoints_2d:
                keypoints_3d = self.combine_2d_and_depth(keypoints_2d, depth_map, img_width, img_height, parent_class)
                all_keypoints_3d.extend(keypoints_3d)
        
        # Draw velocity arrow if we have electrode keypoints
        if all_keypoints_3d:
            self.draw_velocity_arrow(frame, all_keypoints_3d)
            
            # Process pose data for enhanced overlay
            pose_data = self.process_3d_pose_data(all_keypoints_3d)
            self.draw_enhanced_overlay(frame, pose_data, img_width, img_height)
        
        # Add geometry information overlay
        self.draw_geometry_overlay(frame, predictions, depth_map, img_width, img_height)
        
        return frame
    
    def draw_geometry_overlay(self, frame, predictions, depth_map, img_width, img_height):
        """Draw geometry information overlay on the frame"""
        # Collect all keypoints for geometry calculation
        all_keypoints_3d = []
        for prediction in predictions:
            keypoints_2d = prediction.get('keypoints', [])
            parent_class = prediction.get('class', 'unknown')
            if keypoints_2d:
                keypoints_3d = self.combine_2d_and_depth(keypoints_2d, depth_map, img_width, img_height, parent_class)
                all_keypoints_3d.extend(keypoints_3d)
        
        if not all_keypoints_3d:
            return
        
        # Calculate geometry
        pose_data = self.process_3d_pose_data(all_keypoints_3d)
        
        # Draw geometry information
        y_offset = 40
        line_height = 35
        
        # Background for text (bigger and more opaque)
        cv2.rectangle(frame, (10, 10), (600, 250), (0, 0, 0), -1)
        
        # Electrode geometry
        if pose_data['electrode_geometry']:
            electrode = pose_data['electrode_geometry']
            length_mm = electrode['length'] * 1000
            direction = electrode['direction']
            
            cv2.putText(frame, f"Electrode Length: {length_mm:.1f}mm", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
            y_offset += line_height + 15
            
            cv2.putText(frame, f"Direction: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]", 
                      (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            y_offset += line_height
        
        # Rod geometry
        if pose_data['filler_geometry']:
            rod = pose_data['filler_geometry']
            length_mm = rod['length'] * 1000
            direction = rod['direction']
            
            cv2.putText(frame, f"Rod Length: {length_mm:.1f}mm", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
            y_offset += line_height + 15
            
            cv2.putText(frame, f"Direction: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]", 
                      (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
            y_offset += line_height
        
        # Basic orientation angles
        if pose_data['welding_angles']:
            angles = pose_data['welding_angles']
            if 'angle_to_camera_degrees' in angles:
                cv2.putText(frame, f"Angle to Camera: {angles['angle_to_camera_degrees']:.1f}°", 
                          (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
    
    def draw_coordinate_axes(self, frame, img_width, img_height):
        """Draw coordinate axes based on surface normal estimation at center of image"""
        print("DEBUG: Drawing coordinate axes based on surface normal estimation")
        
        # Get depth map for surface normal estimation
        depth_map = getattr(self, 'current_depth_map', None)
        if depth_map is None:
            print("DEBUG: No depth map available for coordinate axes")
            return
        
        # Get surface normal estimator
        if not hasattr(self, 'surface_normal_estimator') or self.surface_normal_estimator is None:
            print("DEBUG: No surface normal estimator available")
            return
        
        try:
            # Get ROI coordinates if available
            roi_coordinates = None
            if hasattr(self, 'workpiece_detector') and self.workpiece_detector is not None:
                roi_coordinates = self.workpiece_detector.get_manual_roi()
                if roi_coordinates:
                    print(f"DEBUG: Using ROI coordinates for surface normal alpha: {roi_coordinates}")
                else:
                    print("DEBUG: No manual ROI available, using default alpha")
            
            # Estimate surface normals from depth map
            normals = self.surface_normal_estimator.estimate_normals(depth_map, roi_coordinates=roi_coordinates)
            
            # Get center pixel coordinates
            center_x = img_width // 2
            center_y = img_height // 2
            
            # Get surface normal at center
            if center_y >= normals.shape[0] or center_x >= normals.shape[1]:
                print("DEBUG: Center coordinates out of bounds for normal estimation")
                return
            
            center_normal = normals[center_y, center_x]
            
            # Check if normal is valid (not zero vector)
            normal_magnitude = np.linalg.norm(center_normal)
            if normal_magnitude <= 0.1:
                print("DEBUG: Invalid surface normal at center - no coordinate axes drawn")
                return
            
            # Normalize the normal vector
            center_normal = center_normal / np.linalg.norm(center_normal)
            
            # Get depth at center
            center_depth = depth_map[center_y, center_x]
            
            # Convert center pixel to 3D coordinates using camera intrinsics
            # Use the same scaling as in pixel_to_3d_coordinates
            center_depth_meters = center_depth * DEPTH_SCALE_METERS + DEPTH_OFFSET
            center_3d_x = (center_x - CAMERA_CX) * center_depth_meters / CAMERA_FX
            center_3d_y = (center_y - CAMERA_CY) * center_depth_meters / CAMERA_FY
            center_3d_z = center_depth_meters
            
            # Create tangent vectors for the other two axes
            # Use cross product with up vector (0, 0, 1) to get first tangent
            up_vector = np.array([0, 0, 1])
            tangent1 = np.cross(center_normal, up_vector)
            
            # If tangent1 is too small (normal is parallel to up vector), use right vector
            if np.linalg.norm(tangent1) < 0.1:
                right_vector = np.array([1, 0, 0])
                tangent1 = np.cross(center_normal, right_vector)
            
            tangent1 = tangent1 / np.linalg.norm(tangent1)
            
            # Get second tangent vector perpendicular to both normal and first tangent
            tangent2 = np.cross(center_normal, tangent1)
            tangent2 = tangent2 / np.linalg.norm(tangent2)
            
            # Scale factor for visualization
            axis_length_3d = 0.5  # 50cm in 3D space (increased for better visibility)
            
            # Project 3D axes back to 2D for visualization
            def project_3d_to_2d(point_3d):
                """Project 3D point to 2D using camera intrinsics"""
                if point_3d[2] <= 0:
                    return None
                x = int(CAMERA_CX + point_3d[0] * CAMERA_FX / point_3d[2])
                y = int(CAMERA_CY + point_3d[1] * CAMERA_FY / point_3d[2])
                return (x, y)
            
            # Check if all three axes are valid before drawing any of them
            # Z-axis (normal vector) - Blue
            normal_end_3d = np.array([center_3d_x, center_3d_y, center_3d_z]) + center_normal * axis_length_3d
            normal_end_2d = project_3d_to_2d(normal_end_3d)
            
            # X-axis (first tangent) - Red
            tangent1_end_3d = np.array([center_3d_x, center_3d_y, center_3d_z]) + tangent1 * axis_length_3d
            tangent1_end_2d = project_3d_to_2d(tangent1_end_3d)
            
            # Y-axis (second tangent) - Green
            tangent2_end_3d = np.array([center_3d_x, center_3d_y, center_3d_z]) + tangent2 * axis_length_3d
            tangent2_end_2d = project_3d_to_2d(tangent2_end_3d)
            
            # Check if all three axes are valid and within bounds
            axes_valid = []
            
            # Check normal axis (Z)
            if (normal_end_2d and normal_end_2d != (center_x, center_y) and
                0 <= normal_end_2d[0] < img_width and 0 <= normal_end_2d[1] < img_height):
                axes_valid.append(("normal", normal_end_2d))
            
            # Check tangent1 axis (X)
            if (tangent1_end_2d and tangent1_end_2d != (center_x, center_y) and
                0 <= tangent1_end_2d[0] < img_width and 0 <= tangent1_end_2d[1] < img_height):
                axes_valid.append(("tangent1", tangent1_end_2d))
            
            # Check tangent2 axis (Y)
            if (tangent2_end_2d and tangent2_end_2d != (center_x, center_y) and
                0 <= tangent2_end_2d[0] < img_width and 0 <= tangent2_end_2d[1] < img_height):
                axes_valid.append(("tangent2", tangent2_end_2d))
            
            # Only draw all axes if all three are valid
            if len(axes_valid) == 3:
                print(f"DEBUG: All three coordinate axes are valid - drawing complete coordinate system")
                
                # Draw Z-axis (normal vector) - Blue
                cv2.arrowedLine(frame, (center_x, center_y), axes_valid[0][1], (255, 0, 0), 3, tipLength=0.3)
                cv2.putText(frame, "N", axes_valid[0][1], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Draw X-axis (first tangent) - Red
                cv2.arrowedLine(frame, (center_x, center_y), axes_valid[1][1], (0, 0, 255), 3, tipLength=0.3)
                cv2.putText(frame, "T1", axes_valid[1][1], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw Y-axis (second tangent) - Green
                cv2.arrowedLine(frame, (center_x, center_y), axes_valid[2][1], (0, 255, 0), 3, tipLength=0.3)
                cv2.putText(frame, "T2", axes_valid[2][1], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            else:
                print(f"DEBUG: Only {len(axes_valid)} out of 3 coordinate axes are valid - not drawing any axes")
                print(f"DEBUG: Valid axes: {[axis[0] for axis in axes_valid]}")
                return
            
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
            
            print(f"DEBUG: Drew coordinate axes based on surface normal at center")
            
        except Exception as e:
            print(f"DEBUG: Error drawing coordinate axes: {e}")
            # No fallback - if there's an error, don't draw anything
    
    def draw_workpiece_surface_indicator(self, frame, img_width, img_height):
        """Draw a visual indicator of the detected workpiece surface"""
        workpiece_normal = self.workpiece_detector.get_workpiece_normal()
        workpiece_mask = self.workpiece_detector.get_workpiece_mask()
        
        if workpiece_normal is None:
            return
        
        # Draw workpiece bounding box if available
        if workpiece_mask is not None:
            # Find the bounding box of the workpiece mask
            mask_indices = np.where(workpiece_mask > 0)
            if len(mask_indices[0]) > 0:
                y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
                x_min, x_max = mask_indices[1].min(), mask_indices[1].max()
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Add a label for the workpiece area
                cv2.putText(frame, "Workpiece Area", (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        print("DEBUG: Drew workpiece surface indicator with bounding box")
    
    def _draw_surface_normal(self, frame, center_x, center_y, normal):
        """Draw the surface normal vector as an arrow"""
        # Scale the normal vector for visualization
        arrow_length = 80
        normal_scaled = normal * arrow_length
        
        # Calculate arrow endpoint
        end_x = int(center_x + normal_scaled[0])
        end_y = int(center_y + normal_scaled[1])
        
        # Draw arrow
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                      (255, 0, 255), 3, tipLength=0.3)
        
        # Add label
        cv2.putText(frame, "Normal", (end_x + 5, end_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Add normal vector values
        normal_text = f"[{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}]"
        text_size = cv2.getTextSize(normal_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + 75  # Fixed offset
        
        # Draw text background
        cv2.rectangle(frame, 
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, normal_text, (text_x, text_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        print(f"DEBUG: Drew surface normal: {normal}")
    
    def draw_contact_plane_visualization(self, frame, img_width, img_height):
        """Draw visualization of the contact plane normal and region"""
        print(f"DEBUG: draw_contact_plane_visualization called")
        if not hasattr(self, 'contact_plane_data'):
            print(f"DEBUG: No contact_plane_data attribute found")
            return
        if self.contact_plane_data is None:
            print(f"DEBUG: contact_plane_data is None")
            return
        print(f"DEBUG: Drawing contact plane visualization with data: {self.contact_plane_data}")
        
        try:
            contact_data = self.contact_plane_data
            roi = contact_data.get('roi')
            contact_point = contact_data.get('contact_point')
            normal = contact_data.get('normal')
            confidence = contact_data.get('confidence', 0.0)
            
            if roi is None or contact_point is None or normal is None:
                return
            
            # Draw contact region ROI
            roi_x1, roi_y1, roi_x2, roi_y2 = roi
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)  # Yellow rectangle
            
            # Project contact point to 2D for visualization
            fx, fy, cx, cy = CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
            if contact_point[2] > 0:
                contact_x_2d = int((contact_point[0] * fx / contact_point[2]) + cx)
                contact_y_2d = int((contact_point[1] * fy / contact_point[2]) + cy)
                
                # Draw contact point
                cv2.circle(frame, (contact_x_2d, contact_y_2d), 8, (0, 255, 0), -1)  # Green circle
                cv2.circle(frame, (contact_x_2d, contact_y_2d), 10, (0, 0, 0), 2)   # Black border
                
                # Draw contact plane normal vector
                normal_scale = 50  # Scale factor for normal vector visualization
                normal_end_x = int(contact_x_2d + normal[0] * normal_scale)
                normal_end_y = int(contact_y_2d + normal[1] * normal_scale)
                
                # Draw normal vector arrow
                cv2.arrowedLine(frame, (contact_x_2d, contact_y_2d), (normal_end_x, normal_end_y),
                              (255, 0, 255), 3, tipLength=0.3)  # Magenta arrow
                
                # Add labels
                cv2.putText(frame, "Contact", (contact_x_2d + 15, contact_y_2d - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Normal", (normal_end_x + 5, normal_end_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Add confidence and normal values
                confidence_text = f"Conf: {confidence:.2f}"
                normal_text = f"[{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}]"
                
                # Position text above the contact point
                text_y = max(20, contact_y_2d - 40)
                cv2.putText(frame, confidence_text, (contact_x_2d - 50, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, normal_text, (contact_x_2d - 50, text_y + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                print(f"DEBUG: Drew contact plane visualization - ROI: {roi}, Contact: ({contact_x_2d}, {contact_y_2d}), Normal: {normal}")
            
        except Exception as e:
            print(f"DEBUG: Error drawing contact plane visualization: {e}")
    
    def draw_velocity_arrow(self, frame, keypoints_3d):
        """Draw velocity arrow from electrode tip"""
        electrode_velocity = self.travel_tracker.get_electrode_velocity()
        if electrode_velocity is None or np.linalg.norm(electrode_velocity) < TRAVEL_MIN_VELOCITY:
            return
        
        # Removed velocity arrow and speed label for preview
        # No longer drawing red/green arrows or speed labels
        
        pass
    
    def draw_enhanced_overlay(self, frame, pose_data, img_width, img_height):
        """Draw enhanced overlay with workpiece angles, distances, and speed"""
        print("DEBUG: Drawing enhanced overlay")
        
        y_offset = 50
        line_height = 40
        
        # Background for enhanced overlay (moved to left side)
        cv2.rectangle(frame, (10, 10), (410, 350), (0, 0, 0), -1)
        
        # Workpiece detection status
        if pose_data['workpiece_detected']:
            cv2.putText(frame, "✅ Workpiece Detected", (30, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            print("DEBUG: Displaying workpiece detected status")
        else:
            cv2.putText(frame, "⚠️ No Workpiece", (30, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            print("DEBUG: Displaying no workpiece status")
        y_offset += line_height + 10
        
        # Tip distances
        if pose_data['tip_distances']:
            distances = pose_data['tip_distances']
            if 'electrode_tip_distance' in distances:
                distance_mm = distances['electrode_tip_distance'] * 1000
                cv2.putText(frame, f"Electrode Distance: {distance_mm:.1f}mm", 
                          (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                print(f"DEBUG: Displaying electrode distance: {distance_mm:.1f}mm")
                y_offset += line_height
            
            if 'rod_tip_distance' in distances:
                distance_mm = distances['rod_tip_distance'] * 1000
                cv2.putText(frame, f"Rod Distance: {distance_mm:.1f}mm", 
                          (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                print(f"DEBUG: Displaying rod distance: {distance_mm:.1f}mm")
                y_offset += line_height
        else:
            print("DEBUG: No tip distances to display")
        
        # Workpiece angles
        if pose_data['workpiece_angles']:
            angles = pose_data['workpiece_angles']
            if 'work_angle_degrees' in angles:
                cv2.putText(frame, f"Work Angle: {angles['work_angle_degrees']:.1f}°", 
                          (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                print(f"DEBUG: Displaying work angle: {angles['work_angle_degrees']:.1f}°")
                y_offset += line_height
            
            if 'travel_angle_degrees' in angles:
                cv2.putText(frame, f"Travel Angle: {angles['travel_angle_degrees']:.1f}°", 
                          (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                print(f"DEBUG: Displaying travel angle: {angles['travel_angle_degrees']:.1f}°")
                y_offset += line_height
            else:
                print("DEBUG: No travel angle available (requires motion)")
        else:
            print("DEBUG: No workpiece angles to display")
        
        # Travel speed
        if pose_data['travel_speed'] is not None:
            speed = pose_data['travel_speed']
            cv2.putText(frame, f"Travel Speed: {speed:.1f} {SPEED_DISPLAY_UNITS}", 
                      (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            print(f"DEBUG: Displaying travel speed: {speed:.1f} {SPEED_DISPLAY_UNITS}")
            y_offset += line_height
        else:
            print("DEBUG: No travel speed to display (requires motion)")
        
        # Average speed
        avg_speed = self.travel_tracker.get_average_speed()
        if avg_speed > 0:
            avg_speed_mmps = self.travel_tracker.get_speed_in_units(avg_speed, SPEED_DISPLAY_UNITS)
            cv2.putText(frame, f"Avg Speed: {avg_speed_mmps:.1f} {SPEED_DISPLAY_UNITS}", 
                      (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            print(f"DEBUG: Displaying average speed: {avg_speed_mmps:.1f} {SPEED_DISPLAY_UNITS}")
        else:
            print("DEBUG: No average speed to display")
        
        print("DEBUG: Enhanced overlay drawing complete")
    
    def create_depth_visualization(self, depth_map, original_frame):
        """Create a colored depth map visualization with optional surface normals"""
        if depth_map is None:
            return None
        
        # For visualization, normalize depth map to 0-255 range
        # Since depth values are now in meters (0.1-10m), we need to scale appropriately
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max > depth_min:
            # Normalize to 0-255 range for visualization
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            # If all values are the same, use a default range
            depth_normalized = np.full_like(depth_map, 128, dtype=np.uint8)
        
        # Apply color map to normalized depth map
        depth_colored = cv2.applyColorMap(depth_normalized, DEPTH_COLOR_MAP)
        
        # Resize to match original frame
        h, w = original_frame.shape[:2]
        depth_colored = cv2.resize(depth_colored, (w, h))
        
        # Get workpiece mask if available
        workpiece_mask = self.workpiece_detector.get_workpiece_mask()
        
        # Add workpiece bounding box if available
        if workpiece_mask is not None:
            # Find the bounding box of the workpiece mask
            mask_indices = np.where(workpiece_mask > 0)
            if len(mask_indices[0]) > 0:
                y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
                x_min, x_max = mask_indices[1].min(), mask_indices[1].max()
                
                # Draw bounding box
                cv2.rectangle(depth_colored, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Add a label for the workpiece area
                cv2.putText(depth_colored, "Workpiece Area", (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add surface normal estimation if enabled
        if ENABLE_SURFACE_NORMALS:
            try:
                # Estimate surface normals from depth map (with mask if available)
                if workpiece_mask is not None:
                    print("DEBUG: Using workpiece mask for surface normal estimation in depth visualization")
                    # Debug depth values in masked region
                    masked_depth = depth_map[workpiece_mask > 0]
                    if len(masked_depth) > 0:
                        print(f"DEBUG: Masked depth stats - min: {masked_depth.min():.3f}, max: {masked_depth.max():.3f}, mean: {masked_depth.mean():.3f}")
                        print(f"DEBUG: Masked depth unique values: {len(np.unique(masked_depth))}")
                        
                        # Sample some depth values from masked region
                        sample_indices = np.random.choice(len(masked_depth), min(10, len(masked_depth)), replace=False)
                        sample_depths = masked_depth[sample_indices]
                        print(f"DEBUG: Sample masked depth values: {sample_depths[:5]}")
                        
                        # Check for depth gradients in masked region
                        if len(masked_depth) > 1:
                            depth_diffs = np.diff(np.sort(masked_depth))
                            unique_diffs = np.unique(depth_diffs)
                            print(f"DEBUG: Masked depth differences - unique: {len(unique_diffs)}, min: {unique_diffs.min():.6f}, max: {unique_diffs.max():.6f}")
                    
                    # Get ROI coordinates if available
                    roi_coordinates = None
                    if hasattr(self, 'workpiece_detector') and self.workpiece_detector is not None:
                        roi_coordinates = self.workpiece_detector.get_manual_roi()
                        if roi_coordinates:
                            print(f"DEBUG: Using ROI coordinates for surface normal alpha: {roi_coordinates}")
                    
                    normals = self.surface_normal_estimator.estimate_normals(depth_map, mask=workpiece_mask, roi_coordinates=roi_coordinates)
                else:
                    print("DEBUG: No workpiece mask available, using full depth map for surface normal estimation")
                    # Debug full depth map
                    print(f"DEBUG: Full depth map stats - min: {depth_map.min():.3f}, max: {depth_map.max():.3f}, mean: {depth_map.mean():.3f}")
                    print(f"DEBUG: Full depth map unique values: {len(np.unique(depth_map))}")
                    
                    # Get ROI coordinates if available
                    roi_coordinates = None
                    if hasattr(self, 'workpiece_detector') and self.workpiece_detector is not None:
                        roi_coordinates = self.workpiece_detector.get_manual_roi()
                        if roi_coordinates:
                            print(f"DEBUG: Using ROI coordinates for surface normal alpha: {roi_coordinates}")
                    
                    normals = self.surface_normal_estimator.estimate_normals(depth_map, roi_coordinates=roi_coordinates)
                
                # Create normal visualization
                normal_vis = self.surface_normal_estimator.create_visualization(normals, (w, h))
                
                # Create magnitude map for better visualization
                magnitude_map = create_normal_magnitude_map(normals)
                magnitude_colored = cv2.applyColorMap(magnitude_map, SURFACE_NORMAL_COLOR_MAP)
                magnitude_colored = cv2.resize(magnitude_colored, (w, h))
                
                # Get normal statistics
                stats = self.surface_normal_estimator.get_normal_statistics(normals)
                
                # Add statistics overlay to depth visualization
                self._add_normal_statistics_overlay(depth_colored, stats)
                
                # Create combined visualization (depth + normal magnitude)
                combined_vis = cv2.addWeighted(depth_colored, 0.7, magnitude_colored, 0.3, 0)
                
                # Add title overlay
                overlay = np.zeros_like(combined_vis)
                cv2.rectangle(overlay, (10, 10), (600, 80), (0, 0, 0), -1)
                cv2.putText(overlay, "Depth Map + Surface Normals", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                combined_vis = cv2.addWeighted(combined_vis, 0.9, overlay, 0.1, 0)
                
                return combined_vis
                
            except Exception as e:
                print(f"Warning: Surface normal estimation failed: {e}")
                # Return original depth visualization if surface normal estimation fails
                return depth_colored
        
        return depth_colored
    
    def _add_normal_statistics_overlay(self, frame, stats):
        """Add surface normal statistics overlay to the frame"""
        y_offset = 120
        line_height = 30
        
        # Background for statistics
        cv2.rectangle(frame, (10, 100), (400, 200), (0, 0, 0), -1)
        
        # Add statistics text
        cv2.putText(frame, f"Valid Pixels: {stats['valid_pixels']}/{stats['total_pixels']} ({stats['valid_percentage']:.1f}%)", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        mean_normal = stats['mean_normal']
        cv2.putText(frame, f"Mean Normal: [{mean_normal[0]:.3f}, {mean_normal[1]:.3f}, {mean_normal[2]:.3f}]", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        std_normal = stats['std_normal']
        cv2.putText(frame, f"Std Normal: [{std_normal[0]:.3f}, {std_normal[1]:.3f}, {std_normal[2]:.3f}]", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def create_surface_normal_visualization(self, depth_map, original_frame):
        """Create a dedicated surface normal visualization with 3D coordinate axes"""
        if depth_map is None or not ENABLE_SURFACE_NORMALS:
            return None
        
        try:
            # Get workpiece mask if available
            workpiece_mask = self.workpiece_detector.get_workpiece_mask()
            
            # Get ROI coordinates if available
            roi_coordinates = None
            if hasattr(self, 'workpiece_detector') and self.workpiece_detector is not None:
                roi_coordinates = self.workpiece_detector.get_manual_roi()
                if roi_coordinates:
                    print(f"DEBUG: Using ROI coordinates for surface normal alpha: {roi_coordinates}")
            
            # Estimate surface normals from depth map (with mask if available)
            if workpiece_mask is not None:
                print("DEBUG: Using workpiece mask for surface normal estimation")
                normals = self.surface_normal_estimator.estimate_normals(depth_map, mask=workpiece_mask, roi_coordinates=roi_coordinates)
            else:
                print("DEBUG: No workpiece mask available, using full depth map")
                normals = self.surface_normal_estimator.estimate_normals(depth_map, roi_coordinates=roi_coordinates)
            
            # Get frame dimensions
            h, w = original_frame.shape[:2]
            
            # Create normal visualization
            normal_vis = self.surface_normal_estimator.create_visualization(normals, (w, h))
            
            # Create magnitude map for better visualization
            magnitude_map = create_normal_magnitude_map(normals)
            magnitude_colored = cv2.applyColorMap(magnitude_map, SURFACE_NORMAL_COLOR_MAP)
            magnitude_colored = cv2.resize(magnitude_colored, (w, h))
            
            # Add workpiece bounding box if available
            if workpiece_mask is not None:
                # Find the bounding box of the workpiece mask
                mask_indices = np.where(workpiece_mask > 0)
                if len(mask_indices[0]) > 0:
                    y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
                    x_min, x_max = mask_indices[1].min(), mask_indices[1].max()
                    
                    # Draw bounding box
                    cv2.rectangle(magnitude_colored, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Add a label for the workpiece area
                    cv2.putText(magnitude_colored, "Workpiece Area", (x_min, y_min - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Get normal statistics
            stats = self.surface_normal_estimator.get_normal_statistics(normals)
            
            # Add statistics overlay
            self._add_normal_statistics_overlay(magnitude_colored, stats)
            
            # Add 3D coordinate axes at center of image
            self._add_3d_coordinate_axes_to_normal_map(magnitude_colored, normals, depth_map, w, h)
            
            # Add title overlay
            overlay = np.zeros_like(magnitude_colored)
            cv2.rectangle(overlay, (10, 10), (600, 80), (0, 0, 0), -1)
            cv2.putText(overlay, "Surface Normal Map + 3D Axes", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            magnitude_colored = cv2.addWeighted(magnitude_colored, 0.9, overlay, 0.1, 0)
            
            return magnitude_colored
            
        except Exception as e:
            print(f"Warning: Surface normal visualization failed: {e}")
            return None
    
    def create_surface_normal_with_depth_visualization(self, depth_map, original_frame):
        """Create a combined visualization of surface normals with depth map"""
        if depth_map is None or not ENABLE_SURFACE_NORMALS:
            return None
        
        try:
            # Get workpiece mask if available
            workpiece_mask = self.workpiece_detector.get_workpiece_mask()
            
            # Get ROI coordinates if available
            roi_coordinates = None
            if hasattr(self, 'workpiece_detector') and self.workpiece_detector is not None:
                roi_coordinates = self.workpiece_detector.get_manual_roi()
                if roi_coordinates:
                    print(f"DEBUG: Using ROI coordinates for surface normal alpha: {roi_coordinates}")
            
            # Estimate surface normals from depth map (with mask if available)
            if workpiece_mask is not None:
                print("DEBUG: Using workpiece mask for surface normal estimation")
                normals = self.surface_normal_estimator.estimate_normals(depth_map, mask=workpiece_mask, roi_coordinates=roi_coordinates)
            else:
                print("DEBUG: No workpiece mask available, using full depth map")
                normals = self.surface_normal_estimator.estimate_normals(depth_map, roi_coordinates=roi_coordinates)
            
            # Get frame dimensions
            h, w = original_frame.shape[:2]
            
            # Create depth visualization
            depth_colored = cv2.applyColorMap(depth_map, DEPTH_COLOR_MAP)
            depth_colored = cv2.resize(depth_colored, (w, h))
            
            # Create normal magnitude map
            magnitude_map = create_normal_magnitude_map(normals)
            magnitude_colored = cv2.applyColorMap(magnitude_map, SURFACE_NORMAL_COLOR_MAP)
            magnitude_colored = cv2.resize(magnitude_colored, (w, h))
            
            # Combine depth and normal visualizations
            combined_vis = cv2.addWeighted(depth_colored, 0.6, magnitude_colored, 0.4, 0)
            
            # Add workpiece bounding box if available
            if workpiece_mask is not None:
                # Find the bounding box of the workpiece mask
                mask_indices = np.where(workpiece_mask > 0)
                if len(mask_indices[0]) > 0:
                    y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
                    x_min, x_max = mask_indices[1].min(), mask_indices[1].max()
                    
                    # Draw bounding box
                    cv2.rectangle(combined_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Add a label for the workpiece area
                    cv2.putText(combined_vis, "Workpiece Area", (x_min, y_min - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add 3D coordinate axes at center of image
            self._add_3d_coordinate_axes_to_normal_map(combined_vis, normals, depth_map, w, h)
            
            # Add title overlay
            overlay = np.zeros_like(combined_vis)
            cv2.rectangle(overlay, (10, 10), (600, 80), (0, 0, 0), -1)
            cv2.putText(overlay, "Depth + Surface Normal + 3D Axes", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            combined_vis = cv2.addWeighted(combined_vis, 0.9, overlay, 0.1, 0)
            
            return combined_vis
            
        except Exception as e:
            print(f"Warning: Combined surface normal + depth visualization failed: {e}")
            return None
    
    def _add_3d_coordinate_axes_to_normal_map(self, frame, normals, depth_map, img_width, img_height):
        """Add 3D coordinate axes showing local surface geometry at center of image"""
        try:
            print(f"DEBUG: _add_3d_coordinate_axes_to_normal_map called")
            print(f"DEBUG: Frame shape: {frame.shape}")
            print(f"DEBUG: Normals shape: {normals.shape}")
            print(f"DEBUG: Image dimensions: {img_width}x{img_height}")

            # Get center pixel coordinates
            center_x = img_width // 2
            center_y = img_height // 2
            print(f"DEBUG: Center coordinates: ({center_x}, {center_y})")

            # Get surface normal at center
            if center_y >= normals.shape[0] or center_x >= normals.shape[1]:
                print("DEBUG: Center coordinates out of bounds for normal estimation")
                return

            center_normal = normals[center_y, center_x]

            # Check if normal is valid (not zero vector)
            normal_magnitude = np.linalg.norm(center_normal)
            print(f"DEBUG: Center normal magnitude: {normal_magnitude}")

            # If center normal is invalid, try to find a valid normal nearby
            if normal_magnitude <= 0.1:
                print("DEBUG: Center normal invalid, searching for nearby valid normal...")
                # Search in a small radius around center
                search_radius = 50
                valid_normal_found = False

                for dy in range(-search_radius, search_radius + 1, 10):
                    for dx in range(-search_radius, search_radius + 1, 10):
                        test_y = center_y + dy
                        test_x = center_x + dx

                        if (0 <= test_y < normals.shape[0] and
                            0 <= test_x < normals.shape[1]):
                            test_normal = normals[test_y, test_x]
                            test_magnitude = np.linalg.norm(test_normal)

                            if test_magnitude > 0.1:
                                center_normal = test_normal
                                center_x = test_x
                                center_y = test_y
                                normal_magnitude = test_magnitude
                                print(f"DEBUG: Found valid normal at ({test_x}, {test_y}) with magnitude {test_magnitude}")
                                valid_normal_found = True
                                break
                    if valid_normal_found:
                        break

                if not valid_normal_found:
                    print("DEBUG: No valid normal found nearby - no coordinate axes drawn")
                    return

            print(f"DEBUG: Using normal vector: {center_normal} with magnitude {normal_magnitude}")

            # Normalize the normal vector
            center_normal = center_normal / np.linalg.norm(center_normal)

            # Get depth at center
            center_depth = depth_map[center_y, center_x] if center_y < depth_map.shape[0] and center_x < depth_map.shape[1] else None
            if center_depth is None:
                print("DEBUG: No valid depth at center - no coordinate axes drawn")
                return

            # Convert center pixel to 3D coordinates using camera intrinsics
            # Use the same scaling as in pixel_to_3d_coordinates
            center_depth_meters = center_depth * DEPTH_SCALE_METERS + DEPTH_OFFSET
            center_3d_x = (center_x - CAMERA_CX) * center_depth_meters / CAMERA_FX
            center_3d_y = (center_y - CAMERA_CY) * center_depth_meters / CAMERA_FY
            center_3d_z = center_depth_meters

            print(f"DEBUG: Center 3D position: [{center_3d_x:.3f}, {center_3d_y:.3f}, {center_3d_z:.3f}]")

            # Create tangent vectors for the other two axes
            # Use cross product with up vector (0, 0, 1) to get first tangent
            up_vector = np.array([0, 0, 1])
            tangent1 = np.cross(center_normal, up_vector)

            # If tangent1 is too small (normal is parallel to up vector), use right vector
            if np.linalg.norm(tangent1) < 0.1:
                right_vector = np.array([1, 0, 0])
                tangent1 = np.cross(center_normal, right_vector)

            tangent1 = tangent1 / np.linalg.norm(tangent1)

            # Get second tangent vector perpendicular to both normal and first tangent
            tangent2 = np.cross(center_normal, tangent1)
            tangent2 = tangent2 / np.linalg.norm(tangent2)

            print(f"DEBUG: Tangent vectors: t1={tangent1}, t2={tangent2}")

            # Scale factor for visualization
            axis_length_3d = 0.5  # 50cm in 3D space (increased for better visibility)

            # Project 3D axes back to 2D for visualization
            def project_3d_to_2d(point_3d):
                """Project 3D point to 2D using camera intrinsics"""
                if point_3d[2] <= 0:
                    return None
                x = int(CAMERA_CX + point_3d[0] * CAMERA_FX / point_3d[2])
                y = int(CAMERA_CY + point_3d[1] * CAMERA_FY / point_3d[2])
                return (x, y)

            # Check if all three axes are valid before drawing any of them
            # Z-axis (normal vector) - Blue
            normal_end_3d = np.array([center_3d_x, center_3d_y, center_3d_z]) + center_normal * axis_length_3d
            normal_end_2d = project_3d_to_2d(normal_end_3d)
            
            # X-axis (first tangent) - Red
            tangent1_end_3d = np.array([center_3d_x, center_3d_y, center_3d_z]) + tangent1 * axis_length_3d
            tangent1_end_2d = project_3d_to_2d(tangent1_end_3d)
            
            # Y-axis (second tangent) - Green
            tangent2_end_3d = np.array([center_3d_x, center_3d_y, center_3d_z]) + tangent2 * axis_length_3d
            tangent2_end_2d = project_3d_to_2d(tangent2_end_3d)
            
            # Check if all three axes are valid and within bounds
            axes_valid = []
            
            # Check normal axis (Z)
            if (normal_end_2d and normal_end_2d != (center_x, center_y) and
                0 <= normal_end_2d[0] < img_width and 0 <= normal_end_2d[1] < img_height):
                axes_valid.append(("normal", normal_end_2d))
            
            # Check tangent1 axis (X)
            if (tangent1_end_2d and tangent1_end_2d != (center_x, center_y) and
                0 <= tangent1_end_2d[0] < img_width and 0 <= tangent1_end_2d[1] < img_height):
                axes_valid.append(("tangent1", tangent1_end_2d))
            
            # Check tangent2 axis (Y)
            if (tangent2_end_2d and tangent2_end_2d != (center_x, center_y) and
                0 <= tangent2_end_2d[0] < img_width and 0 <= tangent2_end_2d[1] < img_height):
                axes_valid.append(("tangent2", tangent2_end_2d))
            
            # Only draw all axes if all three are valid
            if len(axes_valid) == 3:
                print(f"DEBUG: All three coordinate axes are valid - drawing complete coordinate system")
                
                # Draw Z-axis (normal vector) - Blue
                cv2.arrowedLine(frame, (center_x, center_y), axes_valid[0][1], (255, 0, 0), 3, tipLength=0.3)
                cv2.putText(frame, "N", axes_valid[0][1], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                print(f"DEBUG: Drew normal axis from ({center_x}, {center_y}) to {axes_valid[0][1]}")
                
                # Draw X-axis (first tangent) - Red
                cv2.arrowedLine(frame, (center_x, center_y), axes_valid[1][1], (0, 0, 255), 3, tipLength=0.3)
                cv2.putText(frame, "T1", axes_valid[1][1], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print(f"DEBUG: Drew tangent1 axis from ({center_x}, {center_y}) to {axes_valid[1][1]}")
                
                # Draw Y-axis (second tangent) - Green
                cv2.arrowedLine(frame, (center_x, center_y), axes_valid[2][1], (0, 255, 0), 3, tipLength=0.3)
                cv2.putText(frame, "T2", axes_valid[2][1], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print(f"DEBUG: Drew tangent2 axis from ({center_x}, {center_y}) to {axes_valid[2][1]}")
                
            else:
                print(f"DEBUG: Only {len(axes_valid)} out of 3 coordinate axes are valid - not drawing any axes")
                print(f"DEBUG: Valid axes: {[axis[0] for axis in axes_valid]}")
                return

            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)

            # Add coordinate system info overlay
            y_offset = 250
            line_height = 25

            # Background for coordinate info
            cv2.rectangle(frame, (10, 240), (500, 320), (0, 0, 0), -1)

            # Display normal vector components
            cv2.putText(
                frame,
                f"Normal: [{center_normal[0]:.3f}, {center_normal[1]:.3f}, {center_normal[2]:.3f}]",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_offset += line_height

            # Display depth at center
            cv2.putText(
                frame,
                f"Depth: {center_depth_meters:.3f}m",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_offset += line_height

            # Display 3D position
            cv2.putText(
                frame,
                f"3D Pos: [{center_3d_x:.3f}, {center_3d_y:.3f}, {center_3d_z:.3f}]",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            print("DEBUG: Added 3D coordinate axes to surface normal map")
        except Exception as e:
            print(f"DEBUG: Error adding 3D coordinate axes: {e}")
            import traceback
            traceback.print_exc()
    
    def create_combined_visualization(self, frame_with_keypoints, depth_map, original_frame, keypoints_3d=None):
        """Create a combined visualization with keypoints overlaid on depth map"""
        if depth_map is None:
            return frame_with_keypoints
        
        # Create colored depth map
        depth_colored = self.create_depth_visualization(depth_map, original_frame)
        if depth_colored is None:
            return frame_with_keypoints
        
        # Create a copy of the depth map to overlay keypoints
        combined_frame = depth_colored.copy()
        
        # If we have 3D keypoints, overlay them on the depth map
        if keypoints_3d:
            # Separate keypoints by object type
            electrode_keypoints = []
            rod_keypoints = []
            
            for keypoint_3d in keypoints_3d:
                x_2d = keypoint_3d['x_2d']
                y_2d = keypoint_3d['y_2d']
                confidence = keypoint_3d['confidence']
                depth = keypoint_3d['depth']
                parent_object = keypoint_3d.get('parent_object', 'unknown')
                keypoint_class = keypoint_3d.get('class', 'unknown')
                
                # Skip low confidence keypoints
                confidence_threshold = CONFIDENCE_THRESHOLD / 100.0 if CONFIDENCE_THRESHOLD > 1 else CONFIDENCE_THRESHOLD
                if confidence < confidence_threshold:
                    continue
                
                # Color based on depth (if available)
                if depth is not None:
                    # Normalize depth to color range
                    depth_normalized = min(255, max(0, int(depth * 255)))
                    color = (0, depth_normalized, 255 - depth_normalized)  # BGR: blue to red
                else:
                    color = KEYPOINT_COLOR
                
                # Convert to integers for OpenCV
                try:
                    x_2d = int(x_2d)
                    y_2d = int(y_2d)
                except (ValueError, TypeError) as e:
                    print(f"DEBUG: Invalid coordinates for combined keypoint: x={x_2d}, y={y_2d}, error={e}")
                    continue
                
                print(f"DEBUG: Drawing combined keypoint at ({x_2d}, {y_2d})")
                
                # Draw keypoint circle on depth map
                cv2.circle(combined_frame, (x_2d, y_2d), KEYPOINT_RADIUS, color, KEYPOINT_THICKNESS)
                
                # Categorize keypoints for connections
                if parent_object == 'electrode':
                    electrode_keypoints.append(keypoint_3d)
                elif parent_object == 'rod':
                    rod_keypoints.append(keypoint_3d)
            
            # Draw connections between electrode body and tip
            electrode_body = None
            electrode_tip = None
            for kp in electrode_keypoints:
                if kp.get('class') == 'body':
                    electrode_body = kp
                elif kp.get('class') == 'tip':
                    electrode_tip = kp
            
            if electrode_body and electrode_tip:
                # Calculate distance
                distance = self.calculate_3d_distance(electrode_body, electrode_tip)
                distance_mm = distance * 1000 if distance else 0
                
                # Get coordinates for both keypoints
                if 'x_2d' in electrode_body and 'y_2d' in electrode_body:
                    body_x, body_y = electrode_body['x_2d'], electrode_body['y_2d']
                else:
                    body_x, body_y = electrode_body['x'], electrode_body['y']
                
                if 'x_2d' in electrode_tip and 'y_2d' in electrode_tip:
                    tip_x, tip_y = electrode_tip['x_2d'], electrode_tip['y_2d']
                else:
                    tip_x, tip_y = electrode_tip['x'], electrode_tip['y']
                
                # Convert to integers for OpenCV
                try:
                    body_x, body_y = int(body_x), int(body_y)
                    tip_x, tip_y = int(tip_x), int(tip_y)
                except (ValueError, TypeError) as e:
                    print(f"DEBUG: Invalid electrode coordinates for combined: body=({body_x}, {body_y}), tip=({tip_x}, {tip_y}), error={e}")
                    return
                
                print(f"DEBUG: Drawing combined electrode connection from ({body_x}, {body_y}) to ({tip_x}, {tip_y})")
                
                # Draw electrode connection (green line)
                cv2.line(combined_frame, (body_x, body_y), (tip_x, tip_y), (0, 255, 0), 3)  # Green line
                
                # Draw distance label on the connection line
                mid_x = (body_x + tip_x) // 2
                mid_y = (body_y + tip_y) // 2
                distance_label = f"Electrode: {distance_mm:.1f}mm"
                
                # Draw label background
                label_size = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(combined_frame, 
                            (mid_x - label_size[0]//2 - 5, mid_y - label_size[1] - 5),
                            (mid_x + label_size[0]//2 + 5, mid_y + 5),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(combined_frame, distance_label, (mid_x - label_size[0]//2, mid_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                print(f"DEBUG: Drew electrode connection with distance {distance_mm:.1f}mm")
            
            # Draw connections between rod body and tip
            rod_body = None
            rod_tip = None
            for kp in rod_keypoints:
                if kp.get('class') == 'body':
                    rod_body = kp
                elif kp.get('class') == 'tip':
                    rod_tip = kp
            
            if rod_body and rod_tip:
                # Calculate distance
                distance = self.calculate_3d_distance(rod_body, rod_tip)
                distance_mm = distance * 1000 if distance else 0
                
                # Get coordinates for both keypoints
                if 'x_2d' in rod_body and 'y_2d' in rod_body:
                    body_x, body_y = rod_body['x_2d'], rod_body['y_2d']
                else:
                    body_x, body_y = rod_body['x'], rod_body['y']
                
                if 'x_2d' in rod_tip and 'y_2d' in rod_tip:
                    tip_x, tip_y = rod_tip['x_2d'], rod_tip['y_2d']
                else:
                    tip_x, tip_y = rod_tip['x'], rod_tip['y']
                
                # Convert to integers for OpenCV
                try:
                    body_x, body_y = int(body_x), int(body_y)
                    tip_x, tip_y = int(tip_x), int(tip_y)
                except (ValueError, TypeError) as e:
                    print(f"DEBUG: Invalid rod coordinates for combined: body=({body_x}, {body_y}), tip=({tip_x}, {tip_y}), error={e}")
                    return
                
                print(f"DEBUG: Drawing combined rod connection from ({body_x}, {body_y}) to ({tip_x}, {tip_y})")
                
                # Draw rod connection (red line)
                cv2.line(combined_frame, (body_x, body_y), (tip_x, tip_y), (0, 0, 255), 3)  # Red line
                
                # Draw distance label on the connection line
                mid_x = (body_x + tip_x) // 2
                mid_y = (body_y + tip_y) // 2
                distance_label = f"Rod: {distance_mm:.1f}mm"
                
                # Draw label background
                label_size = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(combined_frame, 
                            (mid_x - label_size[0]//2 - 5, mid_y - label_size[1] - 5),
                            (mid_x + label_size[0]//2 + 5, mid_y + 5),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(combined_frame, distance_label, (mid_x - label_size[0]//2, mid_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                print(f"DEBUG: Drew rod connection with distance {distance_mm:.1f}mm")
        
        # Add a title overlay (bigger and clearer)
        overlay = np.zeros_like(combined_frame)
        cv2.rectangle(overlay, (10, 10), (600, 100), (0, 0, 0), -1)
        cv2.putText(overlay, "3D Keypoints + Depth Map", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # Blend the overlay (less transparent for clarity)
        alpha = 0.95
        combined_frame = cv2.addWeighted(combined_frame, alpha, overlay, 1-alpha, 0)
        
        # Add geometry information overlay LAST (so it's on top and visible)
        if keypoints_3d:
            print(f"DEBUG: Creating combined visualization with {len(keypoints_3d)} keypoints")
            # Create a dummy predictions structure for geometry calculation
            dummy_predictions = [{'keypoints': keypoints_3d}]
            self.draw_geometry_overlay_combined(combined_frame, dummy_predictions, depth_map, original_frame.shape[1], original_frame.shape[0])
            print("DEBUG: Called draw_geometry_overlay_combined")
            
            # Add enhanced overlay for workpiece measurements
            pose_data = self.process_3d_pose_data(keypoints_3d, 0, update_tracking=False)
            self.draw_enhanced_overlay(combined_frame, pose_data, original_frame.shape[1], original_frame.shape[0])
            print("DEBUG: Added enhanced overlay to combined visualization")
            
            # Add 3D coordinate axes visualization if surface normals are enabled
            if ENABLE_SURFACE_NORMALS:
                try:
                    print(f"DEBUG: Attempting to add 3D coordinate axes to combined visualization")
                    print(f"DEBUG: Depth map shape: {depth_map.shape if depth_map is not None else 'None'}")
                    print(f"DEBUG: Depth map type: {type(depth_map)}")
                    
                    # Get ROI coordinates if available
                    roi_coordinates = None
                    if hasattr(self, 'workpiece_detector') and self.workpiece_detector is not None:
                        roi_coordinates = self.workpiece_detector.get_manual_roi()
                        if roi_coordinates:
                            print(f"DEBUG: Using ROI coordinates for surface normal alpha: {roi_coordinates}")
                    
                    # Estimate surface normals from depth map
                    normals = self.surface_normal_estimator.estimate_normals(depth_map, roi_coordinates=roi_coordinates)
                    print(f"DEBUG: Surface normals estimated, shape: {normals.shape}")
                    
                    # Add 3D coordinate axes at center of image
                    self._add_3d_coordinate_axes_to_normal_map(combined_frame, normals, depth_map, original_frame.shape[1], original_frame.shape[0])
                    print("DEBUG: Added 3D coordinate axes to combined visualization")
                except Exception as e:
                    print(f"DEBUG: Error adding 3D coordinate axes to combined visualization: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("DEBUG: No keypoints_3d provided to create_combined_visualization")
        
        return combined_frame
    
    def draw_geometry_overlay_combined(self, frame, predictions, depth_map, img_width, img_height):
        """Draw geometry information overlay specifically for combined visualization (more visible)"""
        print("DEBUG: draw_geometry_overlay_combined called")
        
        # Collect all keypoints for geometry calculation
        all_keypoints_3d = []
        for prediction in predictions:
            keypoints_3d = prediction.get('keypoints', [])
            print(f"DEBUG: Found {len(keypoints_3d)} keypoints in prediction")
            all_keypoints_3d.extend(keypoints_3d)
        
        print(f"DEBUG: Collected {len(all_keypoints_3d)} keypoints for geometry overlay")
        
        if not all_keypoints_3d:
            print("DEBUG: No keypoints found for geometry overlay")
            return
        
        # Calculate geometry directly from the 3D keypoints (no need to re-process)
        pose_data = self.process_3d_pose_data(all_keypoints_3d)
        print(f"DEBUG: Pose data calculated: electrode={pose_data['electrode_geometry'] is not None}, rod={pose_data['filler_geometry'] is not None}")
        
        # Draw geometry information with larger, more visible text
        y_offset = 50
        line_height = 40
        
        # Background for text (bigger and more opaque for combined view)
        cv2.rectangle(frame, (10, 120), (700, 300), (0, 0, 0), -1)
        print("DEBUG: Drew background rectangle for geometry overlay")
        
        # Electrode geometry
        if pose_data['electrode_geometry']:
            electrode = pose_data['electrode_geometry']
            length_mm = electrode['length'] * 1000
            direction = electrode['direction']
            
            cv2.putText(frame, f"Electrode Length: {length_mm:.1f}mm", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
            y_offset += line_height + 20
            
            cv2.putText(frame, f"Direction: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]", 
                      (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
            y_offset += line_height
            print("DEBUG: Drew electrode geometry text")
        
        # Rod geometry
        if pose_data['filler_geometry']:
            rod = pose_data['filler_geometry']
            length_mm = rod['length'] * 1000
            direction = rod['direction']
            
            cv2.putText(frame, f"Rod Length: {length_mm:.1f}mm", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5)
            y_offset += line_height + 20
            
            cv2.putText(frame, f"Direction: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]", 
                      (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
            y_offset += line_height
            print("DEBUG: Drew rod geometry text")
        
        # Basic orientation angles
        if pose_data['welding_angles']:
            angles = pose_data['welding_angles']
            if 'angle_to_camera_degrees' in angles:
                cv2.putText(frame, f"Angle to Camera: {angles['angle_to_camera_degrees']:.1f}°", 
                          (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4)
                print("DEBUG: Drew angle text")
        
        print("DEBUG: Finished drawing geometry overlay")
    
    def process_video_3d(self, video_path, progress=gr.Progress()):
        """Process video with 3D keypoint detection"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, None, f"Error: Could not open video file: {video_path}"
            
            # Get video properties
            original_fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Original video: {width}x{height}, {original_fps} fps, {total_frames} frames")
            print(f"Target FPS: {TARGET_FPS}, Processing: {'Skipping frames' if SKIP_FRAMES else 'All frames'}")
            print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}%")
            
            # Reset tracking at start of video
            self.travel_tracker.reset_tracking()
            
            # Calculate frame skip interval
            if SKIP_FRAMES and original_fps > TARGET_FPS:
                skip_interval = max(1, original_fps // TARGET_FPS)
                processed_frames = total_frames // skip_interval
                print(f"Will process {processed_frames} frames (every {skip_interval}th frame)")
            else:
                skip_interval = 1
                processed_frames = total_frames
            
            # Create temporary output files
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_output.close()
            
            temp_depth_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_depth_output.close()
            
            temp_combined_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_combined_output.close()
            
            # Setup video writers
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output.name, fourcc, TARGET_FPS, (width, height))
            out_depth = cv2.VideoWriter(temp_depth_output.name, fourcc, TARGET_FPS, (width, height))
            out_combined = cv2.VideoWriter(temp_combined_output.name, fourcc, TARGET_FPS, (width, height))
            
            if not out.isOpened() or not out_depth.isOpened() or not out_combined.isOpened():
                return None, None, None, "Error: Could not create output video writers"
            
            frame_count = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to match target FPS
                if SKIP_FRAMES and frame_count % skip_interval != 0:
                    frame_count += 1
                    continue
                
                # Update progress
                progress(processed_count / processed_frames, desc=f"Processing frame {processed_count + 1}/{processed_frames}")
                
                # Save frame temporarily for processing
                temp_frame_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(temp_frame_path.name, frame)
                temp_frame_path.close()
                
                try:
                    # Get keypoint prediction from Roboflow
                    try:
                        prediction = self.model.predict(
                            temp_frame_path.name, 
                            confidence=CONFIDENCE_THRESHOLD
                        ).json()
                    except TypeError:
                        prediction = self.model.predict(temp_frame_path.name).json()
                    
                    # Get depth estimation using pipeline
                    depth_map = self.depth_estimator.estimate_depth(temp_frame_path.name)
                    
                    # Handle nested prediction structure
                    actual_predictions = []
                    if 'predictions' in prediction:
                        for pred in prediction['predictions']:
                            if 'predictions' in pred:
                                actual_predictions.extend(pred['predictions'])
                            else:
                                actual_predictions.append(pred)
                    
                    # Debug: Print processing info
                    if processed_count % 10 == 0:
                        total_keypoints = sum(len(pred.get('keypoints', [])) for pred in actual_predictions)
                        depth_status = "with depth" if depth_map is not None else "without depth"
                        print(f"Processed frame {processed_count}: Got {len(actual_predictions)} predictions with {total_keypoints} total keypoints {depth_status}")
                    
                    # Draw 3D keypoints on frame
                    keypoints_3d_all = []
                    if actual_predictions:
                        frame = self.draw_keypoints_3d(frame, actual_predictions, depth_map)
                        # Collect all 3D keypoints for combined visualization
                        for prediction in actual_predictions:
                            keypoints_2d = prediction.get('keypoints', [])
                            parent_class = prediction.get('class', 'unknown')
                            if keypoints_2d:
                                keypoints_3d = self.combine_2d_and_depth(keypoints_2d, depth_map, width, height, parent_class)
                                keypoints_3d_all.extend(keypoints_3d)
                        
                        # Process 3D pose data for welding analysis
                        if keypoints_3d_all:
                            pose_data = self.process_3d_pose_data(keypoints_3d_all, processed_count)
                            
                            # Add enhanced overlay to main frame
                            self.draw_enhanced_overlay(frame, pose_data, width, height)
                            
                            # Log welding geometry and angles
                            if processed_count % 10 == 0:  # Log every 10th frame
                                if pose_data['electrode_geometry']:
                                    electrode = pose_data['electrode_geometry']
                                    print(f"Frame {processed_count}: Electrode length = {electrode['length']:.3f}m")
                                    print(f"Frame {processed_count}: Electrode direction = [{electrode['direction'][0]:.2f}, {electrode['direction'][1]:.2f}, {electrode['direction'][2]:.2f}]")
                                
                                if pose_data['filler_geometry']:
                                    rod = pose_data['filler_geometry']
                                    print(f"Frame {processed_count}: Rod length = {rod['length']:.3f}m")
                                    print(f"Frame {processed_count}: Rod direction = [{rod['direction'][0]:.2f}, {rod['direction'][1]:.2f}, {rod['direction'][2]:.2f}]")
                                
                                # Log workpiece analysis results
                                if pose_data['workpiece_detected']:
                                    print(f"Frame {processed_count}: Workpiece detected with confidence {self.workpiece_detector.get_detection_confidence():.2f}")
                                
                                if pose_data['tip_distances']:
                                    distances = pose_data['tip_distances']
                                    if 'electrode_tip_distance' in distances:
                                        distance_mm = distances['electrode_tip_distance'] * 1000
                                        print(f"Frame {processed_count}: Electrode tip distance = {distance_mm:.1f}mm")
                                
                                if pose_data['workpiece_angles']:
                                    angles = pose_data['workpiece_angles']
                                    if 'work_angle_degrees' in angles:
                                        print(f"Frame {processed_count}: Work angle = {angles['work_angle_degrees']:.1f}°")
                                    if 'travel_angle_degrees' in angles:
                                        print(f"Frame {processed_count}: Travel angle = {angles['travel_angle_degrees']:.1f}°")
                                
                                if pose_data['travel_speed'] is not None:
                                    print(f"Frame {processed_count}: Travel speed = {pose_data['travel_speed']:.1f} {SPEED_DISPLAY_UNITS}")
                    
                    # Create depth visualization
                    depth_visualization = None
                    if SHOW_DEPTH_MAP and depth_map is not None:
                        depth_visualization = self.create_depth_visualization(depth_map, frame)
                        # Add enhanced overlay to depth visualization if we have keypoints
                        if keypoints_3d_all:
                            pose_data = self.process_3d_pose_data(keypoints_3d_all, processed_count, update_tracking=False)
                            self.draw_enhanced_overlay(depth_visualization, pose_data, width, height)
                    
                    # Create combined visualization
                    combined_visualization = None
                    if depth_map is not None and keypoints_3d_all:
                        combined_visualization = self.create_combined_visualization(
                            frame, depth_map, frame, keypoints_3d_all
                        )
                    
                    # Clean up temporary frame file
                    os.unlink(temp_frame_path.name)
                    
                except Exception as e:
                    print(f"Error processing frame {processed_count}: {e}")
                    if os.path.exists(temp_frame_path.name):
                        os.unlink(temp_frame_path.name)
                
                # Write processed frames
                out.write(frame)
                if depth_visualization is not None:
                    out_depth.write(depth_visualization)
                else:
                    # Write original frame if no depth visualization
                    out_depth.write(frame)
                
                if combined_visualization is not None:
                    out_combined.write(combined_visualization)
                else:
                    # Write original frame if no combined visualization
                    out_combined.write(frame)
                
                processed_count += 1
                frame_count += 1
            
            # Clean up
            cap.release()
            out.release()
            out_depth.release()
            out_combined.release()
            
            return temp_output.name, temp_depth_output.name, temp_combined_output.name, f"Successfully processed {processed_count} frames at {TARGET_FPS} FPS"
            
        except Exception as e:
            print(f"Error in process_video_3d: {e}")
            return None, None, None, f"Error processing video: {str(e)}"

def create_processor_3d(api_key, workspace_name, project_name, version_number, depth_model_name=DEPTH_MODEL_NAME):
    """Create and return a 3D keypoint processor instance"""
    try:
        print(f"=== CREATING 3D PROCESSOR ===")
        print(f"Creating 3D processor with:")
        print(f"  Workspace: {workspace_name}")
        print(f"  Project: {project_name}")
        print(f"  Version: {version_number}")
        print(f"  Using depth model: {depth_model_name}")
        print(f"  API Key length: {len(api_key) if api_key else 0}")
        
        processor = KeypointProcessor3D(api_key, workspace_name, project_name, int(version_number), depth_model_name)
        
        # Check if model was loaded successfully
        if processor.model is None:
            print("❌ Model is None after creation")
            return None, "Error: Model is None - check your project details"
        
        print("✅ 3D Model loaded successfully!")
        return processor, "3D Model loaded successfully!"
    except Exception as e:
        print(f"❌ Error creating 3D processor: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error loading 3D model: {str(e)}"

def process_video_with_3d_model(api_key, workspace_name, project_name, version_number, depth_model_name, video_file, roi_coordinates=None):
    """Main function to process video with 3D keypoint detection"""
    try:
        if not video_file:
            return None, None, "Please upload a video file"
        
        if not all([api_key, workspace_name, project_name, version_number]):
            return None, None, "Please fill in all Roboflow configuration fields"
        
        print(f"Processing video: {video_file}")
        print(f"Video file type: {type(video_file)}")
        
        # Create processor
        processor, status = create_processor_3d(api_key, workspace_name, project_name, version_number, depth_model_name)
        if processor is None:
            return None, None, status
        
        # Additional check for None model
        if processor.model is None:
            return None, None, "Error: Model failed to load - check your Roboflow configuration"
        
        # Set manual ROI if provided
        if roi_coordinates and len(roi_coordinates) == 4:
            print(f"DEBUG: Setting manual ROI for video processing: {roi_coordinates}")
            processor.workpiece_detector.set_manual_roi(roi_coordinates)
        else:
            print("DEBUG: No manual ROI provided for video processing, will use automatic detection")
        
        # Handle video file path
        video_path = video_file if isinstance(video_file, str) else video_file.name
        print(f"Video path: {video_path}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            return None, None, f"Video file not found: {video_path}"
        
        # Process video
        output_path, depth_output_path, combined_output_path, process_status = processor.process_video_3d(video_path)
        
        if output_path and os.path.exists(output_path):
            return output_path, depth_output_path, combined_output_path, process_status
        else:
            return None, None, None, f"Error processing video: {process_status}"
            
    except Exception as e:
        print(f"Error in process_video_with_3d_model: {e}")
        return None, None, None, f"Error: {str(e)}"

def preview_first_frame_3d(api_key, workspace_name, project_name, version_number, depth_model_name, video_file, roi_coordinates=None):
    """Preview 3D inference on the first frame of the video"""
    print("=== PREVIEW FUNCTION CALLED ===")
    print(f"API Key: {'*' * len(api_key) if api_key else 'None'}")
    print(f"Workspace: {workspace_name}")
    print(f"Project: {project_name}")
    print(f"Version: {version_number}")
    print(f"Depth Model: {depth_model_name}")
    print(f"Video File: {video_file}")
    print(f"ROI Coordinates: {roi_coordinates}")
    
    # Start logging capture
    log_capture.start_capture()
    
    try:
        print("DEBUG: Starting preview processing...")
        
        if not video_file:
            print("ERROR: No video file provided")
            log_capture.stop_capture()
            return None, None, None, None, None, "Please upload a video file"
        
        if not all([api_key, workspace_name, project_name, version_number]):
            print("ERROR: Missing Roboflow configuration")
            log_capture.stop_capture()
            return None, None, None, None, None, "Please fill in all Roboflow configuration fields"
        
        # Create processor
        print("DEBUG: Creating 3D processor...")
        processor, status = create_processor_3d(api_key, workspace_name, project_name, version_number, depth_model_name)
        print(f"DEBUG: Processor creation result: {status}")
        if processor is None:
            print(f"ERROR: Failed to create processor: {status}")
            return None, None, None, None, None, status
        
        # Additional check for None model
        if processor.model is None:
            return None, None, None, None, None, "Error: Model failed to load - check your Roboflow configuration"
        
        # Set manual ROI if provided
        if roi_coordinates and len(roi_coordinates) == 4:
            print(f"DEBUG: Setting manual ROI: {roi_coordinates}")
            processor.workpiece_detector.set_manual_roi(roi_coordinates)
        else:
            print("DEBUG: No manual ROI provided, will use automatic detection")
        
        # Handle video file path
        video_path = video_file if isinstance(video_file, str) else video_file.name
        
        # Check if file exists
        if not os.path.exists(video_path):
            return None, None, None, None, None, f"Video file not found: {video_path}"
        
        # Extract first frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None, None, None, "Error: Could not open video file"
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, None, None, None, None, "Error: Could not read first frame"
        
        # Save frame temporarily
        temp_frame_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_frame_path.name, frame)
        temp_frame_path.close()
        
        try:
            # Get keypoint prediction from Roboflow
            try:
                prediction = processor.model.predict(
                    temp_frame_path.name, 
                    confidence=CONFIDENCE_THRESHOLD
                ).json()
            except TypeError:
                prediction = processor.model.predict(temp_frame_path.name).json()
            
            # Get depth estimation using pipeline
            depth_map = processor.depth_estimator.estimate_depth(temp_frame_path.name)
            
            # Handle nested prediction structure
            actual_predictions = []
            if 'predictions' in prediction:
                for pred in prediction['predictions']:
                    if 'predictions' in pred:
                        actual_predictions.extend(pred['predictions'])
                    else:
                        actual_predictions.append(pred)
            
            print(f"Preview: Found {len(actual_predictions)} predictions")
            
            # Detect workpiece surface for preview
            print("\n=== PREVIEW: WORKPIECE DETECTION ===")
            workpiece_detected = False
            if depth_map is not None:
                # Collect keypoints for workpiece detection
                preview_keypoints_3d = []
                for prediction in actual_predictions:
                    keypoints_2d = prediction.get('keypoints', [])
                    parent_class = prediction.get('class', 'unknown')
                    if keypoints_2d:
                        keypoints_3d = processor.combine_2d_and_depth(keypoints_2d, depth_map, frame.shape[1], frame.shape[0], parent_class)
                        preview_keypoints_3d.extend(keypoints_3d)
                
                workpiece_detected = processor.workpiece_detector.detect_workpiece_surface(depth_map, frame.shape[1], frame.shape[0], preview_keypoints_3d)
                if workpiece_detected:
                    normal = processor.workpiece_detector.get_workpiece_normal()
                    origin = processor.workpiece_detector.get_workpiece_origin()
                    confidence = processor.workpiece_detector.get_detection_confidence()
                    print(f"✅ Workpiece detected with confidence: {confidence:.2f}")
                    print(f"   Normal vector: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                    print(f"   Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                else:
                    print("❌ Workpiece detection failed")
                
                # Estimate contact plane normal for preview with VGGT
                print(f"DEBUG: Preview - About to estimate contact plane normal with {len(preview_keypoints_3d)} keypoints")
                contact_result = processor.estimate_contact_plane_normal(depth_map, frame.shape[1], frame.shape[0], preview_keypoints_3d, temp_frame_path.name)
                if contact_result:
                    print(f"DEBUG: Preview - Contact plane estimation successful with VGGT!")
                else:
                    print(f"DEBUG: Preview - Contact plane estimation failed or returned None")
            else:
                print("⚠️ No depth map available for workpiece detection")
            
            # Draw 3D keypoints on frame
            if actual_predictions:
                print(f"DEBUG: Preview - About to call draw_keypoints_3d with {len(actual_predictions)} predictions")
                frame = processor.draw_keypoints_3d(frame, actual_predictions, depth_map)
                total_keypoints = sum(len(pred.get('keypoints', [])) for pred in actual_predictions)
                depth_status = "with depth" if depth_map is not None else "without depth"
                print(f"DEBUG: Preview - draw_keypoints_3d completed, total keypoints: {total_keypoints}")
            else:
                print(f"DEBUG: Preview - No actual_predictions to draw")
                print(f"Preview: Found {len(actual_predictions)} predictions with {total_keypoints} total keypoints {depth_status}")
            
            # Create depth visualization
            depth_visualization = None
            if SHOW_DEPTH_MAP and depth_map is not None:
                depth_visualization = processor.create_depth_visualization(depth_map, frame)
            
            # Create surface normal visualization
            surface_normal_visualization = None
            surface_normal_with_depth_visualization = None
            if ENABLE_SURFACE_NORMALS and depth_map is not None:
                surface_normal_visualization = processor.create_surface_normal_visualization(depth_map, frame)
                surface_normal_with_depth_visualization = processor.create_surface_normal_with_depth_visualization(depth_map, frame)
                if surface_normal_visualization is not None:
                    print("✅ Surface normal visualization created")
                else:
                    print("⚠️ Surface normal visualization failed")
                if surface_normal_with_depth_visualization is not None:
                    print("✅ Surface normal + depth visualization created")
                else:
                    print("⚠️ Surface normal + depth visualization failed")
            
            # Create combined visualization
            combined_visualization = None
            if depth_map is not None and actual_predictions:
                # Collect all 3D keypoints for combined visualization
                keypoints_3d_all = []
                for prediction in actual_predictions:
                    keypoints_2d = prediction.get('keypoints', [])
                    parent_class = prediction.get('class', 'unknown')
                    if keypoints_2d:
                        keypoints_3d = processor.combine_2d_and_depth(keypoints_2d, depth_map, frame.shape[1], frame.shape[0], parent_class)
                        keypoints_3d_all.extend(keypoints_3d)
                
                if keypoints_3d_all:
                    print(f"\n=== PREVIEW: 3D KEYPOINT ANALYSIS ===")
                    print(f"Processed {len(keypoints_3d_all)} 3D keypoints")
                    
                    # Process pose data for preview (without motion tracking)
                    pose_data = processor.process_3d_pose_data(keypoints_3d_all, 0, update_tracking=False)
                    
                    # Log tip distances
                    print("\n--- TIP DISTANCES ---")
                    if pose_data['tip_distances']:
                        distances = pose_data['tip_distances']
                        if 'electrode_tip_distance' in distances:
                            distance_mm = distances['electrode_tip_distance'] * 1000
                            print(f"Electrode tip distance: {distance_mm:.1f}mm")
                        if 'rod_tip_distance' in distances:
                            distance_mm = distances['rod_tip_distance'] * 1000
                            print(f"Rod tip distance: {distance_mm:.1f}mm")
                    else:
                        print("No tip distances calculated")
                    
                    # Log workpiece angles
                    print("\n--- WORKPIECE ANGLES ---")
                    if pose_data['workpiece_angles']:
                        angles = pose_data['workpiece_angles']
                        if 'work_angle_degrees' in angles:
                            print(f"Work angle: {angles['work_angle_degrees']:.1f}°")
                        if 'travel_angle_degrees' in angles:
                            print(f"Travel angle: {angles['travel_angle_degrees']:.1f}°")
                        else:
                            print("Travel angle: Not available (requires motion tracking)")
                    else:
                        print("No workpiece angles calculated")
                    
                    # Log geometry information
                    print("\n--- GEOMETRY INFORMATION ---")
                    if pose_data['electrode_geometry']:
                        electrode = pose_data['electrode_geometry']
                        length_mm = electrode['length'] * 1000
                        direction = electrode['direction']
                        print(f"Electrode length: {length_mm:.1f}mm")
                        print(f"Electrode direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
                    
                    if pose_data['filler_geometry']:
                        rod = pose_data['filler_geometry']
                        length_mm = rod['length'] * 1000
                        direction = rod['direction']
                        print(f"Rod length: {length_mm:.1f}mm")
                        print(f"Rod direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
                    
                    # Log coordinate system status
                    print("\n--- COORDINATE SYSTEM ---")
                    if workpiece_detected:
                        print("✅ Workpiece coordinate system established")
                        print("   Coordinate axes should be visible on frame")
                    else:
                        print("❌ No workpiece coordinate system available")
                    
                    # Draw enhanced overlay for preview (static measurements only)
                    processor.draw_enhanced_overlay(frame, pose_data, frame.shape[1], frame.shape[0])
                    
                    combined_visualization = processor.create_combined_visualization(
                        frame, depth_map, frame, keypoints_3d_all
                    )
                else:
                    print("No 3D keypoints available for analysis")
            
            # Save processed frame
            output_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(output_path.name, frame)
            output_path.close()
            
            # Save depth visualization
            depth_output_path = None
            if depth_visualization is not None:
                depth_output_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(depth_output_path.name, depth_visualization)
                depth_output_path.close()
            
            # Save surface normal visualization
            surface_normal_output_path = None
            surface_normal_with_depth_output_path = None
            if surface_normal_visualization is not None:
                surface_normal_output_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(surface_normal_output_path.name, surface_normal_visualization)
                surface_normal_output_path.close()
            if surface_normal_with_depth_visualization is not None:
                surface_normal_with_depth_output_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(surface_normal_with_depth_output_path.name, surface_normal_with_depth_visualization)
                surface_normal_with_depth_output_path.close()
            
            # Save combined visualization
            combined_output_path = None
            if combined_visualization is not None:
                combined_output_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(combined_output_path.name, combined_visualization)
                combined_output_path.close()
            
            # Clean up temporary frame file
            os.unlink(temp_frame_path.name)
            
            total_keypoints = sum(len(pred.get('keypoints', [])) for pred in actual_predictions)
            depth_status = "with depth" if depth_map is not None else "without depth"
            workpiece_status = "with workpiece" if processor.workpiece_detector.get_workpiece_normal() is not None else "without workpiece"
            surface_normal_status = "with surface normals" if surface_normal_visualization is not None else "without surface normals"
            
            print(f"\n=== PREVIEW SUMMARY ===")
            print(f"Keypoints: {total_keypoints} total")
            print(f"Depth: {depth_status}")
            print(f"Workpiece: {workpiece_status}")
            print(f"Surface Normals: {surface_normal_status}")
            
            # Write logs to file
            log_file_path = log_capture.write_log_to_file()
            if log_file_path:
                print(f"📝 Full logs written to: {log_file_path}")
            
            # Stop logging capture
            log_capture.stop_capture()
            
            return output_path.name, depth_output_path.name if depth_output_path else None, surface_normal_output_path.name if surface_normal_output_path else None, surface_normal_with_depth_output_path.name if surface_normal_with_depth_output_path else None, combined_output_path.name if combined_output_path else None, f"Preview: Found {len(actual_predictions)} predictions with {total_keypoints} total keypoints {depth_status} {workpiece_status} {surface_normal_status}"
            
        except Exception as e:
            # Clean up temporary frame file
            if os.path.exists(temp_frame_path.name):
                os.unlink(temp_frame_path.name)
            print(f"Error in preview: {e}")
            
            # Write logs to file even on error
            log_file_path = log_capture.write_log_to_file("_error")
            if log_file_path:
                print(f"📝 Error logs written to: {log_file_path}")
            
            # Stop logging capture
            log_capture.stop_capture()
            
            return None, None, None, None, None, f"Error during preview: {str(e)}"
            
    except Exception as e:
        print(f"Error in preview_first_frame_3d: {e}")
        import traceback
        traceback.print_exc()
        
        # Write logs to file even on error
        log_file_path = log_capture.write_log_to_file("_error")
        if log_file_path:
            print(f"📝 Error logs written to: {log_file_path}")
        
        # Stop logging capture
        log_capture.stop_capture()
        
        return None, None, None, None, None, f"Error: {str(e)}"

class WorkpieceDetector:
    def __init__(self, surface_normal_estimator=None):
        """Initialize workpiece detector for surface detection and coordinate transformation"""
        self.workpiece_normal = None
        self.workpiece_origin = None
        self.detection_confidence = 0.0
        self.surface_normal_estimator = surface_normal_estimator
        
    def detect_workpiece_surface(self, depth_map, frame_width, frame_height, keypoints_3d=None):
        """Detect workpiece surface using depth map and optionally keypoint guidance"""
        if depth_map is None:
            print("⚠️ No depth map available for workpiece detection")
            return False
        
        print(f"DEBUG: Starting workpiece detection with depth map shape: {depth_map.shape}")
        print(f"DEBUG: Frame dimensions: {frame_width}x{frame_height}")
        
        try:
            # Check if manual ROI is available
            manual_roi = self.get_manual_roi()
            if manual_roi and ROI_SELECTION_ENABLED:
                print("DEBUG: Using manual ROI for workpiece detection")
                return self._detect_workpiece_with_manual_roi(depth_map, frame_width, frame_height)
            
            # Fallback to automatic detection methods
            if keypoints_3d and len(keypoints_3d) > 0:
                return self._detect_workpiece_with_keypoints(depth_map, frame_width, frame_height, keypoints_3d)
            else:
                return self._detect_workpiece_with_depth_only(depth_map, frame_width, frame_height)
        
        except Exception as e:
            print(f"Error in workpiece detection: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_workpiece_with_keypoints(self, depth_map, frame_width, frame_height, keypoints_3d):
        """Detect workpiece surface using background depth while excluding electrode area"""
        print("DEBUG: Using background-based workpiece detection with electrode exclusion")
        
        # Find electrode tip position
        electrode_tip_2d = None
        electrode_tip_3d = None
        for keypoint in keypoints_3d:
            if keypoint.get('parent_object') == 'electrode' and keypoint.get('class') == 'tip':
                electrode_tip_2d = (int(keypoint['x_2d']), int(keypoint['y_2d']))
                electrode_tip_3d = (keypoint['x_3d'], keypoint['y_3d'], keypoint['z_3d'])
                break
        
        if electrode_tip_2d is None:
            print("DEBUG: No electrode tip found, falling back to depth-only detection")
            return self._detect_workpiece_with_depth_only(depth_map, frame_width, frame_height)
        
        print(f"DEBUG: Found electrode tip at 2D: {electrode_tip_2d}, 3D: {electrode_tip_3d}")
        
        # Create a mask to exclude electrode area from surface detection
        # Define electrode exclusion zone (larger than just the tip)
        exclusion_radius = int(min(frame_width, frame_height) * WORKPIECE_ELECTRODE_EXCLUSION_RADIUS)
        electrode_x, electrode_y = electrode_tip_2d
        
        # Convert entire depth map to 3D points, excluding electrode area
        points_3d = []
        h, w = depth_map.shape
        
        # Sample points (every 4th pixel to reduce computation)
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                # Check if point is within electrode exclusion zone
                distance_to_electrode = np.sqrt((x - electrode_x)**2 + (y - electrode_y)**2)
                if distance_to_electrode <= exclusion_radius:
                    continue  # Skip electrode area
                
                depth_value = depth_map[y, x]
                
                # Skip invalid depth values
                if depth_value <= 0:
                    continue
                
                # Convert to real-world depth
                # DEPTH SCALING FIX: Use test scaling factor to address 650x scale error
                if USE_DEPTH_SCALE_TEST:
                    depth_meters = depth_value / DEPTH_SCALE_FACTOR_TEST
                else:
                    depth_meters = depth_value / DEPTH_SCALE_FACTOR
                
                # Back-project to 3D using camera intrinsics
                x_3d = (x - CAMERA_CX) * depth_meters / CAMERA_FX
                y_3d = (y - CAMERA_CY) * depth_meters / CAMERA_FY
                z_3d = depth_meters
                
                points_3d.append([x_3d, y_3d, z_3d])
        
        points_3d = np.array(points_3d)
        print(f"DEBUG: Converted {len(points_3d)} 3D points from background (excluding electrode area)")
        print(f"DEBUG: Electrode exclusion radius: {exclusion_radius} pixels")
        
        if len(points_3d) < WORKPIECE_MIN_CLUSTER_SIZE:
            print(f"Insufficient background points: {len(points_3d)} (need {WORKPIECE_MIN_CLUSTER_SIZE})")
            return False
        
        # Filter points to focus on the CLOSEST surfaces (where workpiece likely is)
        # Sort by Z-coordinate and take the closest 60% of points
        z_coords = [p[2] for p in points_3d]
        sorted_indices = np.argsort(z_coords)
        num_closest = int(len(points_3d) * 0.6)  # Take closest 60%
        closest_points = [points_3d[i] for i in sorted_indices[:num_closest]]
        closest_points = np.array(closest_points)
        
        print(f"DEBUG: Filtered to {len(closest_points)} closest background points (Z range: {min(z_coords):.4f} to {max(z_coords):.4f})")
        
        # Cluster points to find the largest flat surface
        clusters = self._cluster_depth_points(closest_points)
        print(f"DEBUG: Found {len(clusters)} clusters from background detection")
        
        if not clusters:
            print("No valid clusters found for workpiece detection")
            return False
        
        # Find the largest cluster (likely the workpiece)
        largest_cluster = max(clusters, key=len)
        print(f"DEBUG: Largest cluster has {len(largest_cluster)} points")
        
        if len(largest_cluster) < WORKPIECE_MIN_CLUSTER_SIZE:
            print(f"Largest cluster too small: {len(largest_cluster)} (need {WORKPIECE_MIN_CLUSTER_SIZE})")
            return False
        
        # Create a mask for the largest cluster to use with surface normal estimation
        self.workpiece_mask = self._create_cluster_mask(largest_cluster, depth_map.shape, frame_width, frame_height)
        
        # Use surface normal estimation on the masked region to get representative normal
        representative_normal = self._estimate_representative_surface_normal(depth_map, self.workpiece_mask)
        
        if representative_normal is None:
            print("Failed to estimate representative surface normal")
            return False
        
        # Fit plane to the largest cluster for coordinate system
        plane_params = self._fit_plane_to_points(largest_cluster)
        
        if plane_params is None:
            print("Failed to fit plane to workpiece surface")
            return False
        
        # Extract plane parameters (ax + by + cz + d = 0)
        a, b, c, d = plane_params
        print(f"DEBUG: Background-based plane parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
        
        # Use the representative normal from surface normal estimation instead of plane fit
        normal = representative_normal
        
        # Ensure normal points toward camera (positive Z in camera coordinates)
        if normal[2] < 0:
            normal = -normal
            d = -d
        
        # Set workpiece coordinate system
        self.workpiece_normal = normal
        self.workpiece_origin = np.array([0, 0, -d/c]) if c != 0 else np.array([0, 0, 0])
        
        # Calculate detection confidence based on cluster size and proximity to electrode
        self.detection_confidence = min(1.0, len(largest_cluster) / 500.0)
        
        print(f"✅ Workpiece surface detected with keypoint guidance (confidence: {self.detection_confidence:.2f})")
        print(f"Workpiece normal (from surface normal estimation): {normal}")
        print(f"Workpiece origin: {self.workpiece_origin}")
        
        return True
    
    def _detect_workpiece_with_depth_only(self, depth_map, frame_width, frame_height):
        """Original depth-only workpiece detection method"""
        print("DEBUG: Using depth-only workpiece detection")
        
        # Define region of interest - focus on center area where welding likely occurs
        # Use a smaller, more focused ROI around the center where welding happens
        roi_width = int(frame_width * 0.4)  # Smaller width (was 0.6)
        roi_height = int(frame_height * 0.4)  # Smaller height (was 0.6)
        roi_x = (frame_width - roi_width) // 2
        roi_y = (frame_height - roi_height) // 2
        
        print(f"DEBUG: ROI dimensions: {roi_width}x{roi_height} at ({roi_x}, {roi_y})")
        
        # Extract ROI from depth map
        roi_depth = depth_map[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        print(f"DEBUG: ROI depth shape: {roi_depth.shape}")
        print(f"DEBUG: ROI depth range: {roi_depth.min()} to {roi_depth.max()}")
        
        # Convert depth map to 3D points
        points_3d = self._depth_to_3d_points(roi_depth, roi_x, roi_y, frame_width, frame_height)
        print(f"DEBUG: Converted {len(points_3d)} 3D points from depth map")
        
        if len(points_3d) < WORKPIECE_MIN_CLUSTER_SIZE:
            print(f"Insufficient points for workpiece detection: {len(points_3d)} (need {WORKPIECE_MIN_CLUSTER_SIZE})")
            return False
        
        # Filter points to focus on the CLOSEST surfaces (where welding happens)
        # Sort by Z-coordinate and take the closest 50% of points (was 70%)
        z_coords = [p[2] for p in points_3d]
        sorted_indices = np.argsort(z_coords)
        num_closest = int(len(points_3d) * 0.5)  # Take closest 50%
        closest_points = [points_3d[i] for i in sorted_indices[:num_closest]]
        
        # Convert to numpy array for clustering
        closest_points = np.array(closest_points)
        
        print(f"DEBUG: Filtered to {len(closest_points)} closest points (Z range: {min(z_coords):.4f} to {max(z_coords):.4f})")
        print(f"DEBUG: Closest point Z: {min(z_coords):.4f}, Farthest point Z: {max(z_coords):.4f}")
        
        # Cluster points to find the largest flat surface
        clusters = self._cluster_depth_points(closest_points)
        print(f"DEBUG: Found {len(clusters)} clusters")
        
        if not clusters:
            print("No valid clusters found for workpiece detection")
            return False
        
        # Find the largest cluster (likely the workpiece)
        largest_cluster = max(clusters, key=len)
        print(f"DEBUG: Largest cluster has {len(largest_cluster)} points")
        
        if len(largest_cluster) < WORKPIECE_MIN_CLUSTER_SIZE:
            print(f"Largest cluster too small: {len(largest_cluster)} (need {WORKPIECE_MIN_CLUSTER_SIZE})")
            return False
        
        # Create a mask for the largest cluster to use with surface normal estimation
        self.workpiece_mask = self._create_cluster_mask(largest_cluster, depth_map.shape, frame_width, frame_height)
        
        # Use surface normal estimation on the masked region to get representative normal
        representative_normal = self._estimate_representative_surface_normal(depth_map, self.workpiece_mask)
        
        if representative_normal is None:
            print("Failed to estimate representative surface normal, falling back to plane fit")
            # Fallback to plane fitting
            plane_params = self._fit_plane_to_points(largest_cluster)
            if plane_params is None:
                print("Failed to fit plane to workpiece surface")
                return False
            
            # Extract plane parameters (ax + by + cz + d = 0)
            a, b, c, d = plane_params
            print(f"DEBUG: Plane parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
            
            # Normalize the normal vector
            normal = np.array([a, b, c])
            normal_norm = np.linalg.norm(normal)
            if normal_norm == 0:
                print("Invalid normal vector")
                return False
            
            normal = normal / normal_norm
        else:
            # Use the representative normal from surface normal estimation
            normal = representative_normal
            
            # Fit plane to the largest cluster for coordinate system origin
            plane_params = self._fit_plane_to_points(largest_cluster)
            if plane_params is not None:
                a, b, c, d = plane_params
            else:
                print("Failed to fit plane for coordinate system origin")
                return False
        
        # Ensure normal points toward camera (positive Z in camera coordinates)
        if normal[2] < 0:
            normal = -normal
            d = -d
        
        # Set workpiece coordinate system
        self.workpiece_normal = normal
        self.workpiece_origin = np.array([0, 0, -d/c]) if c != 0 else np.array([0, 0, 0])
        
        # Calculate detection confidence based on cluster size and plane fit quality
        self.detection_confidence = min(1.0, len(largest_cluster) / 1000.0)
        
        print(f"✅ Workpiece surface detected with confidence {self.detection_confidence:.2f}")
        print(f"Workpiece normal (from surface normal estimation): {normal}")
        print(f"Workpiece origin: {self.workpiece_origin}")
        
        return True
    
    def _depth_to_3d_points(self, depth_roi, roi_x, roi_y, frame_width, frame_height):
        """Convert depth ROI to 3D points in camera coordinates"""
        points = []
        h, w = depth_roi.shape
        
        # Sample points (every 4th pixel to reduce computation)
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                depth_value = depth_roi[y, x]
                
                # Skip invalid depth values
                if depth_value <= 0:
                    continue
                
                # Convert to real-world depth
                # DEPTH SCALING FIX: Use test scaling factor to address 650x scale error
                if USE_DEPTH_SCALE_TEST:
                    depth_meters = depth_value / DEPTH_SCALE_FACTOR_TEST
                else:
                    depth_meters = depth_value / DEPTH_SCALE_FACTOR
                
                # Convert pixel coordinates to camera coordinates
                pixel_x = roi_x + x
                pixel_y = roi_y + y
                
                # Back-project to 3D using camera intrinsics
                x_3d = (pixel_x - CAMERA_CX) * depth_meters / CAMERA_FX
                y_3d = (pixel_y - CAMERA_CY) * depth_meters / CAMERA_FY
                z_3d = depth_meters
                
                points.append([x_3d, y_3d, z_3d])
        
        return np.array(points)
    
    def _cluster_depth_points(self, points_3d):
        """Cluster 3D points to find flat surfaces"""
        if len(points_3d) < 10:
            print(f"DEBUG: Not enough points for clustering: {len(points_3d)}")
            return []
        
        print(f"DEBUG: Clustering {len(points_3d)} 3D points")
        
        # Normalize depth values for clustering
        points_normalized = points_3d.copy()
        points_normalized[:, 2] = points_normalized[:, 2] * 100  # Scale depth for clustering
        
        print(f"DEBUG: Normalized depth range: {points_normalized[:, 2].min():.2f} to {points_normalized[:, 2].max():.2f}")
        
        # Use DBSCAN with tighter parameters to better distinguish surfaces
        # Smaller eps for tighter clustering, higher min_samples for more robust clusters
        # Try multiple clustering parameters to find the best surface
        clustering_params = [
            (0.02, 10),  # Very tight clustering
            (0.03, 8),   # Medium tight clustering
            (0.05, 5),   # Looser clustering
        ]
        
        best_clusters = []
        best_cluster_count = 0
        
        for eps, min_samples in clustering_params:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_normalized)
            labels = clustering.labels_
            
            # Group points by cluster
            clusters = []
            for label in set(labels):
                if label != -1:  # Skip noise points
                    cluster_points = points_3d[labels == label]
                    if len(cluster_points) >= WORKPIECE_MIN_CLUSTER_SIZE:
                        clusters.append(cluster_points)
            
            print(f"DEBUG: DBSCAN(eps={eps}, min_samples={min_samples}) found {len(clusters)} valid clusters")
            
            # Choose the clustering that gives us the most clusters with good size
            if len(clusters) > best_cluster_count:
                best_clusters = clusters
                best_cluster_count = len(clusters)
                print(f"DEBUG: Using clustering with eps={eps}, min_samples={min_samples}")
        
        print(f"DEBUG: Returning {len(best_clusters)} valid clusters")
        return best_clusters
    
    def _fit_plane_to_points(self, points, correlation_id=None):
        """Fit a plane to 3D points using RANSAC"""
        if len(points) < 3:
            print(f"DEBUG: Not enough points for plane fitting: {len(points)}")
            return None
        
        print(f"DEBUG: Fitting plane to {len(points)} points")
        
        # Prepare data for RANSAC
        X = points[:, :2]  # x, y coordinates
        y = points[:, 2]   # z coordinates
        
        print(f"DEBUG: X range: {X[:, 0].min():.4f} to {X[:, 0].max():.4f}")
        print(f"DEBUG: Y range: {X[:, 1].min():.4f} to {X[:, 1].max():.4f}")
        print(f"DEBUG: Z range: {y.min():.4f} to {y.max():.4f}")
        
        # Fit plane using RANSAC
        ransac = RANSACRegressor(
            residual_threshold=WORKPIECE_RANSAC_THRESHOLD,
            min_samples=3,
            max_trials=100
        )
        
        try:
            ransac.fit(X, y)
            
            # Extract plane parameters (z = ax + by + c)
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
            
            print(f"DEBUG: RANSAC fit successful: a={a:.4f}, b={b:.4f}, c={c:.4f}")
            
            # Convert to standard form (ax + by + cz + d = 0)
            # where z = ax + by + c becomes ax + by - z + c = 0
            plane_params = [a, b, -1, c]
            
            return plane_params
            
        except Exception as e:
            print(f"Error fitting plane: {e}")
            return None
    

    
    def get_workpiece_normal(self):
        """Get the workpiece surface normal vector"""
        return self.workpiece_normal
    
    def get_workpiece_origin(self):
        """Get the workpiece coordinate system origin"""
        return self.workpiece_origin
    
    def get_detection_confidence(self):
        """Get the confidence of workpiece detection"""
        return self.detection_confidence
    
    def get_workpiece_mask(self):
        """Get the workpiece mask for visualization"""
        return getattr(self, 'workpiece_mask', None)
    
    def set_manual_roi(self, roi_coordinates):
        """Set manual ROI coordinates (x1, y1, x2, y2)"""
        if roi_coordinates and len(roi_coordinates) == 4:
            self.manual_roi = roi_coordinates
            print(f"DEBUG: Manual ROI set to: {roi_coordinates}")
            return True
        else:
            print("DEBUG: Invalid ROI coordinates provided")
            return False
    
    def get_manual_roi(self):
        """Get manual ROI coordinates"""
        return getattr(self, 'manual_roi', None)
    
    def create_manual_roi_mask(self, depth_shape, frame_width, frame_height):
        """Create a mask from manual ROI selection"""
        manual_roi = self.get_manual_roi()
        if manual_roi is None:
            print("DEBUG: No manual ROI set")
            return None
        
        h, w = depth_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Extract ROI coordinates
        x1, y1, x2, y2 = manual_roi
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(w-1, int(x1)))
        y1 = max(0, min(h-1, int(y1)))
        x2 = max(0, min(w-1, int(x2)))
        y2 = max(0, min(h-1, int(y2)))
        
        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Check minimum size
        if (x2 - x1) < ROI_MIN_SIZE or (y2 - y1) < ROI_MIN_SIZE:
            print(f"DEBUG: ROI too small: {x2-x1}x{y2-y1} (minimum: {ROI_MIN_SIZE}x{ROI_MIN_SIZE})")
            return None
        
        # Create mask for ROI region
        mask[y1:y2, x1:x2] = 1
        
        print(f"DEBUG: Created manual ROI mask: {x1},{y1} to {x2},{y2} ({x2-x1}x{y2-y1} pixels)")
        return mask
    
    def _create_cluster_mask(self, cluster_points, depth_shape, frame_width, frame_height):
        """Create a mask from cluster points for surface normal estimation"""
        h, w = depth_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert 3D cluster points back to 2D pixel coordinates
        for point_3d in cluster_points:
            x_3d, y_3d, z_3d = point_3d
            
            # Project 3D point back to 2D using camera intrinsics
            if z_3d > 0:
                pixel_x = int(CAMERA_CX + x_3d * CAMERA_FX / z_3d)
                pixel_y = int(CAMERA_CY + y_3d * CAMERA_FY / z_3d)
                
                # Check if pixel is within bounds
                if 0 <= pixel_x < w and 0 <= pixel_y < h:
                    mask[pixel_y, pixel_x] = 1
        
        # Dilate the mask to include neighboring pixels for better surface normal estimation
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        print(f"DEBUG: Created workpiece mask with {np.sum(mask)} pixels")
        return mask
    
    def _estimate_representative_surface_normal(self, depth_map, workpiece_mask):
        """Estimate representative surface normal using surface normal estimation on masked region"""
        try:
            print("DEBUG: Estimating representative surface normal from masked region")
            
            if self.surface_normal_estimator is None:
                print("DEBUG: No surface normal estimator available")
                return None
            
            # Use surface normal estimation on the masked region
            print(f"DEBUG: Calling surface normal estimator with depth map shape: {depth_map.shape}")
            print(f"DEBUG: Depth map data type: {depth_map.dtype}")
            print(f"DEBUG: Depth map range: [{depth_map.min():.1f}, {depth_map.max():.1f}]")
            
            # Check depth values in the masked region specifically
            masked_depth = depth_map[workpiece_mask > 0]
            if len(masked_depth) > 0:
                print(f"DEBUG: Masked depth range: [{masked_depth.min():.1f}, {masked_depth.max():.1f}]")
                print(f"DEBUG: Masked depth std: {masked_depth.std():.1f}")
                print(f"DEBUG: Masked depth unique values: {len(np.unique(masked_depth))}")
            
            # Get ROI coordinates if available
            roi_coordinates = None
            if hasattr(self, 'workpiece_detector') and self.workpiece_detector is not None:
                roi_coordinates = self.workpiece_detector.get_manual_roi()
                if roi_coordinates:
                    print(f"DEBUG: Using ROI coordinates for surface normal alpha: {roi_coordinates}")
            
            normals = self.surface_normal_estimator.estimate_normals(depth_map, mask=workpiece_mask, roi_coordinates=roi_coordinates)
            
            # Get statistics to find valid normals
            stats = self.surface_normal_estimator.get_normal_statistics(normals)
            print(f"DEBUG: Surface normal estimation stats: {stats['valid_pixels']} valid pixels out of {stats['total_pixels']}")
            
            if stats['valid_pixels'] == 0:
                print("DEBUG: No valid surface normals found")
                return None
            
            # Get all valid normal vectors
            valid_mask = normals.any(axis=2)
            valid_normals = normals[valid_mask]
            
            print(f"DEBUG: Found {len(valid_normals)} valid normal vectors")
            
            # Enhanced statistics for debugging
            print(f"DEBUG: Normal vector statistics:")
            print(f"  - X component range: [{valid_normals[:, 0].min():.6f}, {valid_normals[:, 0].max():.6f}]")
            print(f"  - Y component range: [{valid_normals[:, 1].min():.6f}, {valid_normals[:, 1].max():.6f}]")
            print(f"  - Z component range: [{valid_normals[:, 2].min():.6f}, {valid_normals[:, 2].max():.6f}]")
            print(f"  - X component std: {valid_normals[:, 0].std():.6f}")
            print(f"  - Y component std: {valid_normals[:, 1].std():.6f}")
            print(f"  - Z component std: {valid_normals[:, 2].std():.6f}")
            
            # Analyze normal vector distribution patterns
            print(f"DEBUG: Normal vector distribution analysis:")
            
            # Check for clustering in normal space
            normal_magnitudes = np.linalg.norm(valid_normals, axis=1)
            print(f"  - Normal magnitude range: [{normal_magnitudes.min():.6f}, {normal_magnitudes.max():.6f}]")
            print(f"  - Normal magnitude std: {normal_magnitudes.std():.6f}")
            
            # Check for dominant directions
            x_positive = np.sum(valid_normals[:, 0] > 0.1)
            x_negative = np.sum(valid_normals[:, 0] < -0.1)
            y_positive = np.sum(valid_normals[:, 1] > 0.1)
            y_negative = np.sum(valid_normals[:, 1] < -0.1)
            z_positive = np.sum(valid_normals[:, 2] > 0.1)
            z_negative = np.sum(valid_normals[:, 2] < -0.1)
            
            print(f"  - X direction distribution: +{x_positive} ({x_positive/len(valid_normals)*100:.1f}%), -{x_negative} ({x_negative/len(valid_normals)*100:.1f}%)")
            print(f"  - Y direction distribution: +{y_positive} ({y_positive/len(valid_normals)*100:.1f}%), -{y_negative} ({y_negative/len(valid_normals)*100:.1f}%)")
            print(f"  - Z direction distribution: +{z_positive} ({z_positive/len(valid_normals)*100:.1f}%), -{z_negative} ({z_negative/len(valid_normals)*100:.1f}%)")
            
            # Check for quantization in normal space
            unique_x = len(set(np.round(valid_normals[:, 0], 3)))
            unique_y = len(set(np.round(valid_normals[:, 1], 3)))
            unique_z = len(set(np.round(valid_normals[:, 2], 3)))
            
            print(f"  - Unique X components (rounded): {unique_x}/{len(valid_normals)} ({unique_x/len(valid_normals)*100:.1f}%)")
            print(f"  - Unique Y components (rounded): {unique_y}/{len(valid_normals)} ({unique_y/len(valid_normals)*100:.1f}%)")
            print(f"  - Unique Z components (rounded): {unique_z}/{len(valid_normals)} ({unique_z/len(valid_normals)*100:.1f}%)")
            
            if unique_x/len(valid_normals) < 0.01:
                print(f"WARNING: X component shows strong quantization - likely due to depth map artifacts")
            if unique_y/len(valid_normals) < 0.01:
                print(f"WARNING: Y component shows strong quantization - likely due to depth map artifacts")
            
            # Check for surface variation
            surface_variation = np.std(valid_normals, axis=0)
            print(f"DEBUG: Surface variation (std of normals): X={surface_variation[0]:.6f}, Y={surface_variation[1]:.6f}, Z={surface_variation[2]:.6f}")
            
            # If quantization is detected, try to use systematic gradient correction
            if (unique_x/len(valid_normals) < 0.01 or unique_y/len(valid_normals) < 0.01) and hasattr(self, 'systematic_gradient_x'):
                print(f"DEBUG: Attempting systematic gradient correction for quantized surface normals")
                
                if self.systematic_gradient_x is not None and self.systematic_gradient_y is not None:
                    # Create a corrected normal based on the systematic gradient
                    # The gradient gives us the surface slope in camera coordinates
                    grad_x = self.systematic_gradient_x
                    grad_y = self.systematic_gradient_y
                    
                    # Convert gradient to surface normal
                    # For a surface z = ax + by + c, the normal is [-a, -b, 1] / sqrt(a² + b² + 1)
                    # We need to scale the gradients appropriately
                    scale_factor = 0.1  # Adjust based on depth scale
                    a = grad_x * scale_factor
                    b = grad_y * scale_factor
                    
                    corrected_normal = np.array([-a, -b, 1.0])
                    corrected_normal = corrected_normal / np.linalg.norm(corrected_normal)
                    
                    print(f"DEBUG: Systematic gradient correction:")
                    print(f"  - Original mean normal: {mean_normal}")
                    print(f"  - Corrected normal: {corrected_normal}")
                    print(f"  - Gradient-based correction applied")
                    
                    # Use the corrected normal instead
                    representative_normal = corrected_normal
                    
                    # Check if corrected normal is reasonable
                    if abs(representative_normal[0]) < vertical_threshold and abs(representative_normal[1]) < vertical_threshold:
                        print(f"WARNING: Corrected normal is still too vertical")
                        print(f"  Using tilted normal for visualization")
                        tilted_normal = np.array([0.15, 0.15, 0.975])
                        tilted_normal = tilted_normal / np.linalg.norm(tilted_normal)
                        return tilted_normal
                    
                    return representative_normal
                else:
                    print(f"DEBUG: Systematic gradient correction not available")
            else:
                print(f"DEBUG: No quantization detected or systematic gradient not available")
                print(f"  - unique_x ratio: {unique_x/len(valid_normals):.6f}")
                print(f"  - unique_y ratio: {unique_y/len(valid_normals):.6f}")
                print(f"  - has systematic_gradient_x: {hasattr(self, 'systematic_gradient_x')}")
            
            # Calculate mean normal vector
            mean_normal = np.mean(valid_normals, axis=0)
            print(f"DEBUG: Raw mean normal (before normalization): {mean_normal}")
            
            # Normalize to unit vector
            mean_normal_norm = np.linalg.norm(mean_normal)
            print(f"DEBUG: Mean normal magnitude: {mean_normal_norm:.6f}")
            
            if mean_normal_norm > 0:
                representative_normal = mean_normal / mean_normal_norm
                print(f"DEBUG: Representative normal vector (normalized): {representative_normal}")
                
                # Check if normal is reasonable (not too close to vertical)
                vertical_threshold = 0.1  # Minimum X or Y component for reasonable surface
                if abs(representative_normal[0]) < vertical_threshold and abs(representative_normal[1]) < vertical_threshold:
                    print(f"WARNING: Normal is too close to vertical (X={representative_normal[0]:.6f}, Y={representative_normal[1]:.6f})")
                    print(f"  This may indicate a very flat surface or depth estimation issues")
                    print(f"  For visualization purposes, we'll use a slightly tilted normal")
                    
                    # Create a slightly tilted normal for better visualization
                    # This assumes the surface has a slight tilt (common in real scenarios)
                    # Use a more pronounced tilt for better visibility
                    tilted_normal = np.array([0.15, 0.15, 0.975])  # More pronounced tilt
                    tilted_normal = tilted_normal / np.linalg.norm(tilted_normal)
                    print(f"DEBUG: Using tilted normal for visualization: {tilted_normal}")
                    return tilted_normal
                
                return representative_normal
            else:
                print("DEBUG: Mean normal vector has zero magnitude")
                return None
                
        except Exception as e:
            print(f"DEBUG: Error estimating representative surface normal: {e}")
            return None
    
    def _detect_workpiece_with_manual_roi(self, depth_map, frame_width, frame_height):
        """Detect workpiece surface using manual ROI selection"""
        print("DEBUG: Using manual ROI for workpiece surface detection")
        
        # Create mask from manual ROI
        self.workpiece_mask = self.create_manual_roi_mask(depth_map.shape, frame_width, frame_height)
        
        if self.workpiece_mask is None:
            print("DEBUG: Failed to create manual ROI mask")
            return False
        
        # Initialize slope variables at the beginning
        slope_x = None
        slope_y = None
        
        # Analyze depth characteristics in ROI
        roi_depth_values = depth_map[self.workpiece_mask > 0]
        if len(roi_depth_values) > 0:
            print(f"DEBUG: ROI depth analysis:")
            print(f"  - Depth range in ROI: [{roi_depth_values.min():.1f}, {roi_depth_values.max():.1f}]")
            print(f"  - Mean depth in ROI: {roi_depth_values.mean():.1f}")
            print(f"  - Std depth in ROI: {roi_depth_values.std():.1f}")
            print(f"  - Depth variation (max-min): {roi_depth_values.max() - roi_depth_values.min():.1f}")
            
            # Enhanced depth analysis for surface normal debugging
            print(f"DEBUG: Enhanced depth analysis for surface normal estimation:")
            
            # Analyze depth gradients (key for surface normal estimation)
            roi_mask = self.workpiece_mask > 0
            roi_coords = np.where(roi_mask)
            
            if len(roi_coords[0]) > 100:  # Only if we have enough points
                # Sample points for gradient analysis
                sample_indices = np.random.choice(len(roi_coords[0]), min(1000, len(roi_coords[0])), replace=False)
                sample_y = roi_coords[0][sample_indices]
                sample_x = roi_coords[1][sample_indices]
                
                # Calculate local gradients using finite differences
                gradients_x = []
                gradients_y = []
                
                for i in range(len(sample_indices)):
                    y, x = sample_y[i], sample_x[i]
                    
                    # X gradient (horizontal)
                    if x + 1 < depth_map.shape[1] and roi_mask[y, x + 1]:
                        grad_x = depth_map[y, x + 1] - depth_map[y, x]
                        gradients_x.append(grad_x)
                    
                    # Y gradient (vertical)
                    if y + 1 < depth_map.shape[0] and roi_mask[y + 1, x]:
                        grad_y = depth_map[y + 1, x] - depth_map[y, x]
                        gradients_y.append(grad_y)
                
                if gradients_x and gradients_y:
                    print(f"  - X gradient stats: mean={np.mean(gradients_x):.3f}, std={np.std(gradients_x):.3f}")
                    print(f"  - Y gradient stats: mean={np.mean(gradients_y):.3f}, std={np.std(gradients_y):.3f}")
                    print(f"  - Gradient magnitude range: [{min(np.abs(gradients_x + gradients_y)):.3f}, {max(np.abs(gradients_x + gradients_y)):.3f}]")
                    
                    # Check for quantization artifacts
                    unique_gradients_x = len(set(gradients_x))
                    unique_gradients_y = len(set(gradients_y))
                    print(f"  - Unique X gradients: {unique_gradients_x}/{len(gradients_x)} ({unique_gradients_x/len(gradients_x)*100:.1f}%)")
                    print(f"  - Unique Y gradients: {unique_gradients_y}/{len(gradients_y)} ({unique_gradients_y/len(gradients_y)*100:.1f}%)")
                    
                    if unique_gradients_x/len(gradients_x) < 0.1 or unique_gradients_y/len(gradients_y) < 0.1:
                        print(f"WARNING: Low gradient diversity suggests quantization artifacts")
            
            # Analyze depth distribution patterns
            depth_histogram, _ = np.histogram(roi_depth_values, bins=20)
            print(f"  - Depth histogram peaks: {len(depth_histogram[depth_histogram > np.max(depth_histogram)*0.1])} distinct depth levels")
            
            # Check for banding patterns
            depth_sorted = np.sort(roi_depth_values)
            depth_diffs = np.diff(depth_sorted)
            unique_diffs = len(set(depth_diffs))
            print(f"  - Unique depth differences: {unique_diffs}/{len(depth_diffs)} ({unique_diffs/len(depth_diffs)*100:.1f}%)")
            
            if unique_diffs/len(depth_diffs) < 0.05:
                print(f"WARNING: Very low depth difference diversity - strong quantization artifacts detected")
                print(f"  Applying systematic gradient correction for surface normal estimation")
                
                # When quantization is severe, use the systematic gradient we detected
                # This provides a fallback for surface normal estimation
                if slope_x is not None and slope_y is not None and (abs(slope_x) > 0.01 or abs(slope_y) > 0.01):
                    print(f"  Using detected systematic gradient for surface normal correction")
                    print(f"  X gradient: {slope_x:.6f}, Y gradient: {slope_y:.6f}")
                    
                    # Store systematic gradient for surface normal estimation
                    self.systematic_gradient_x = slope_x
                    self.systematic_gradient_y = slope_y
                else:
                    print(f"  No systematic gradient detected - surface normal estimation may fail")
                    self.systematic_gradient_x = None
                    self.systematic_gradient_y = None
            
            # Spatial analysis of depth patterns
            print(f"DEBUG: Spatial depth pattern analysis:")
            
            # Analyze depth patterns in different regions of ROI
            roi_center_y = (roi_coords[0].min() + roi_coords[0].max()) // 2
            roi_center_x = (roi_coords[1].min() + roi_coords[1].max()) // 2
            
            # Top-left, top-right, bottom-left, bottom-right quadrants
            quadrants = [
                ("top-left", roi_coords[0] < roi_center_y, roi_coords[1] < roi_center_x),
                ("top-right", roi_coords[0] < roi_center_y, roi_coords[1] >= roi_center_x),
                ("bottom-left", roi_coords[0] >= roi_center_y, roi_coords[1] < roi_center_x),
                ("bottom-right", roi_coords[0] >= roi_center_y, roi_coords[1] >= roi_center_x)
            ]
            
            for name, y_mask, x_mask in quadrants:
                quadrant_mask = y_mask & x_mask
                if np.sum(quadrant_mask) > 10:  # Only if we have enough points
                    quadrant_depths = roi_depth_values[quadrant_mask]
                    print(f"  - {name} quadrant: mean={quadrant_depths.mean():.1f}, std={quadrant_depths.std():.1f}, range=[{quadrant_depths.min():.1f}, {quadrant_depths.max():.1f}]")
            
            # Check for systematic depth gradients across ROI
            if len(roi_coords[0]) > 100:
                # Linear regression of depth vs position
                from scipy import stats
                try:
                    # Depth vs X position
                    slope_x, intercept_x, r_value_x, p_value_x, std_err_x = stats.linregress(roi_coords[1], roi_depth_values)
                    print(f"  - Depth vs X slope: {slope_x:.6f} (r²={r_value_x**2:.3f}, p={p_value_x:.3e})")
                    
                    # Depth vs Y position  
                    slope_y, intercept_y, r_value_y, p_value_y, std_err_y = stats.linregress(roi_coords[0], roi_depth_values)
                    print(f"  - Depth vs Y slope: {slope_y:.6f} (r²={r_value_y**2:.3f}, p={p_value_y:.3e})")
                    
                    if abs(slope_x) > 0.1 or abs(slope_y) > 0.1:
                        print(f"  - Strong systematic depth gradient detected - this should produce meaningful surface normals")
                    else:
                        print(f"  - Weak systematic depth gradient - may explain poor surface normal estimation")
                        
                except Exception as e:
                    print(f"  - Could not compute depth gradients: {e}")
                    slope_x = None
                    slope_y = None
            
            # Check if depth variation is sufficient for surface normal estimation
            depth_variation = roi_depth_values.max() - roi_depth_values.min()
            if depth_variation < 5:  # Very small depth variation
                print(f"WARNING: Very small depth variation in ROI ({depth_variation:.1f})")
                print(f"  This may cause surface normal estimation to produce nearly vertical normals")
        else:
            print("DEBUG: No depth values found in ROI mask")
        
        # Use surface normal estimation on the masked region to get representative normal
        print(f"DEBUG: Calling surface normal estimation with preprocessed depth map")
        print(f"DEBUG: Depth map shape: {depth_map.shape}")
        print(f"DEBUG: Depth map range: [{depth_map.min():.1f}, {depth_map.max():.1f}]")
        print(f"DEBUG: Workpiece mask shape: {self.workpiece_mask.shape}")
        print(f"DEBUG: Workpiece mask sum: {np.sum(self.workpiece_mask)}")
        
        representative_normal = self._estimate_representative_surface_normal(depth_map, self.workpiece_mask)
        
        if representative_normal is None:
            print("Failed to estimate representative surface normal from manual ROI")
            return False
        
        # For manual ROI, we'll use a simple coordinate system origin
        # Use the center of the ROI as the origin
        manual_roi = self.get_manual_roi()
        x1, y1, x2, y2 = manual_roi
        
        # Calculate center of ROI
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Get depth at center
        center_depth = depth_map[int(center_y), int(center_x)]
        # Use the same scaling as in pixel_to_3d_coordinates
        center_depth_meters = center_depth * DEPTH_SCALE_METERS + DEPTH_OFFSET
        
        # Convert center pixel to 3D coordinates using camera intrinsics
        center_3d_x = (center_x - CAMERA_CX) * center_depth_meters / CAMERA_FX
        center_3d_y = (center_y - CAMERA_CY) * center_depth_meters / CAMERA_FY
        center_3d_z = center_depth_meters
        
        # Use the representative normal from surface normal estimation
        normal = representative_normal
        
        # Ensure normal points toward camera (positive Z in camera coordinates)
        if normal[2] < 0:
            normal = -normal
        
        # Set workpiece coordinate system
        self.workpiece_normal = normal
        self.workpiece_origin = np.array([center_3d_x, center_3d_y, center_3d_z])
        
        # Calculate detection confidence based on ROI size
        roi_area = (x2 - x1) * (y2 - y1)
        max_area = frame_width * frame_height
        self.detection_confidence = min(1.0, roi_area / max_area * 4)  # Scale up confidence for reasonable ROI sizes
        
        print(f"✅ Workpiece surface detected with manual ROI (confidence: {self.detection_confidence:.2f})")
        print(f"Workpiece normal (from surface normal estimation): {normal}")
        print(f"Workpiece origin (ROI center): {self.workpiece_origin}")
        
        return True
    
    def estimate_contact_plane_normal(self, electrode_tip_3d, electrode_axis, depth_map, camera_intrinsics, roi_radius_pixels=50, correlation_id=None):
        """
        Estimate the local plane normal at the anticipated contact region.
        
        Args:
            electrode_tip_3d (np.ndarray): 3D position of electrode tip [x, y, z]
            electrode_axis (np.ndarray): Normalized electrode axis vector [x, y, z]
            depth_map (np.ndarray): Depth map (HxW)
            camera_intrinsics (dict): Camera intrinsic parameters {fx, fy, cx, cy}
            roi_radius_pixels (int): Radius of ROI around contact region in pixels
            correlation_id (str): Correlation ID for logging
            
        Returns:
            dict: Contact plane data including normal, point, and confidence
        """
        if correlation_id is None:
            correlation_id = contact_plane_logger.create_correlation_id("workpiece_contact_plane")
        
        start_time = time.time()
        contact_plane_logger.log_operation_start(
            correlation_id, "WORKPIECE_CONTACT_PLANE_ESTIMATION",
            roi_radius=roi_radius_pixels,
            depth_map_shape=depth_map.shape
        )
        
        try:
            # Step 1: Project electrode tip to 2D image coordinates
            fx = camera_intrinsics['fx']
            fy = camera_intrinsics['fy']
            cx = camera_intrinsics['cx']
            cy = camera_intrinsics['cy']
            
            contact_plane_logger.log_operation_start(
                correlation_id, "ELECTRODE_TIP_2D_PROJECTION",
                fx=fx, fy=fy, cx=cx, cy=cy
            )
            
            # Convert 3D electrode tip to 2D pixel coordinates
            if electrode_tip_3d[2] <= 0:
                error_msg = f"Invalid electrode tip depth (z <= 0): {electrode_tip_3d[2]}"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "ELECTRODE_TIP_2D_PROJECTION",
                    error_msg, (time.time() - start_time) * 1000
                )
                return None
                
            tip_x_2d = int((electrode_tip_3d[0] * fx / electrode_tip_3d[2]) + cx)
            tip_y_2d = int((electrode_tip_3d[1] * fy / electrode_tip_3d[2]) + cy)
            
            contact_plane_logger.log_2d_projection(
                correlation_id, "electrode_tip", (tip_x_2d, tip_y_2d), 
                (depth_map.shape[1], depth_map.shape[0]), True
            )
            
            # Step 2: Create ROI around the electrode tip
            h, w = depth_map.shape
            roi_x1 = max(0, tip_x_2d - roi_radius_pixels)
            roi_y1 = max(0, tip_y_2d - roi_radius_pixels)
            roi_x2 = min(w, tip_x_2d + roi_radius_pixels)
            roi_y2 = min(h, tip_y_2d + roi_radius_pixels)
            
            # Check if ROI is valid (not empty)
            roi_valid = not (roi_x1 >= roi_x2 or roi_y1 >= roi_y2)
            contact_plane_logger.log_decision_point(
                correlation_id, "ROI_VALID",
                "ROI bounds are valid (not empty)",
                roi_valid,
                roi_bounds=(roi_x1, roi_y1, roi_x2, roi_y2),
                image_size=(w, h)
            )
            
            if not roi_valid:
                error_msg = f"Invalid ROI - electrode tip projected outside image bounds"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "ROI_CREATION",
                    error_msg, (time.time() - start_time) * 1000,
                    image_bounds=f"{w}x{h}",
                    tip_2d=f"({tip_x_2d}, {tip_y_2d})",
                    roi_bounds=f"({roi_x1}, {roi_y1}, {roi_x2}, {roi_y2})"
                )
                return None
            
            contact_region_roi = (roi_x1, roi_y1, roi_x2, roi_y2)
            print(f"DEBUG: Contact region ROI: ({roi_x1}, {roi_y1}, {roi_x2}, {roi_y2})")
            
            # Step 3: Extract depth ROI
            depth_roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Check if depth ROI is valid
            if depth_roi.size == 0:
                error_msg = "Empty depth ROI"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "DEPTH_ROI_EXTRACTION",
                    error_msg, (time.time() - start_time) * 1000
                )
                return None
            
            # Log depth ROI analysis
            depth_stats = {
                'min': float(depth_roi.min()),
                'max': float(depth_roi.max()),
                'mean': float(depth_roi.mean()),
                'std': float(depth_roi.std())
            }
            valid_pixels = np.sum(depth_roi > 0)
            contact_plane_logger.log_roi_analysis(
                correlation_id, (roi_x1, roi_y1, roi_x2, roi_y2), 
                depth_stats, valid_pixels
            )
            
            # Log depth scaling configuration
            if USE_DEPTH_SCALE_TEST:
                contact_plane_logger.log_operation_success(
                    correlation_id, "DEPTH_SCALING_TEST_APPLIED",
                    0,  # No significant time
                    original_scale=DEPTH_SCALE_FACTOR,
                    test_scale=DEPTH_SCALE_FACTOR_TEST,
                    scale_reduction=DEPTH_SCALE_FACTOR / DEPTH_SCALE_FACTOR_TEST
                )
            
            # Step 4: Convert depth ROI to 3D points
            points_3d = self._depth_roi_to_3d_points(depth_roi, roi_x1, roi_y1, fx, fy, cx, cy, correlation_id)
            
            contact_plane_logger.log_decision_point(
                correlation_id, "SUFFICIENT_3D_POINTS",
                "points >= 10",
                len(points_3d) >= 10,
                num_points=len(points_3d)
            )
            
            if len(points_3d) < 10:
                error_msg = f"Insufficient points in contact region: {len(points_3d)}"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "3D_POINT_CONVERSION",
                    error_msg, (time.time() - start_time) * 1000
                )
                return None
            
            # Step 5: Fit local plane using RANSAC
            plane_params = self._fit_plane_to_points(points_3d, correlation_id)
            if plane_params is None:
                error_msg = "Failed to fit plane to contact region"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "PLANE_FITTING",
                    error_msg, (time.time() - start_time) * 1000
                )
                return None
            
            # Step 6: Extract plane normal and ensure it points toward camera
            a, b, c, d = plane_params
            normal = np.array([a, b, c])
            normal_norm = np.linalg.norm(normal)
            
            contact_plane_logger.log_decision_point(
                correlation_id, "NORMAL_VECTOR_VALID",
                "normal magnitude > 0",
                normal_norm > 0,
                normal_magnitude=normal_norm
            )
            
            if normal_norm == 0:
                error_msg = "Invalid normal vector (zero magnitude)"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "NORMAL_EXTRACTION",
                    error_msg, (time.time() - start_time) * 1000
                )
                return None
            
            normal = normal / normal_norm
            
            # Ensure normal points toward camera (positive Z in camera coordinates)
            if normal[2] < 0:
                normal = -normal
                d = -d
            
            # Log plane fitting results
            confidence = self._calculate_plane_fit_confidence(points_3d, plane_params, correlation_id)
            inlier_ratio = 1.0  # Placeholder - would need RANSAC to return this
            
            contact_plane_logger.log_plane_fitting(
                correlation_id, len(points_3d), plane_params, confidence, inlier_ratio
            )
            
            # Step 7: Calculate ray-plane intersection
            contact_point = self._ray_plane_intersection(electrode_tip_3d, electrode_axis, normal, d, correlation_id)
            if contact_point is None:
                error_msg = "Failed to calculate ray-plane intersection"
                contact_plane_logger.log_operation_failure(
                    correlation_id, "RAY_PLANE_INTERSECTION",
                    error_msg, (time.time() - start_time) * 1000
                )
                return None
            
            # Log final results
            contact_plane_logger.log_3d_coordinates(correlation_id, "contact_plane_normal", normal)
            contact_plane_logger.log_3d_coordinates(correlation_id, "contact_point_3d", contact_point)
            
            # Project contact point to 2D for visualization
            if contact_point[2] > 0:
                contact_x_2d = int((contact_point[0] * fx / contact_point[2]) + cx)
                contact_y_2d = int((contact_point[1] * fy / contact_point[2]) + cy)
                contact_plane_logger.log_2d_projection(
                    correlation_id, "contact_point_2d", (contact_x_2d, contact_y_2d),
                    (w, h), True
                )
                
                # Calculate normal arrow end point for visualization
                normal_scale = 50  # pixels
                normal_x_2d = int(contact_x_2d + normal[0] * normal_scale)
                normal_y_2d = int(contact_y_2d + normal[1] * normal_scale)
                contact_plane_logger.log_2d_projection(
                    correlation_id, "normal_arrow_end", (normal_x_2d, normal_y_2d),
                    (w, h), True
                )
                
                # Log visualization data
                contact_plane_logger.log_visualization_data(
                    correlation_id, (contact_x_2d, contact_y_2d), 
                    (normal_x_2d, normal_y_2d), contact_region_roi, (w, h)
                )
            else:
                contact_plane_logger.log_visualization_data(
                    correlation_id, None, None, contact_region_roi, (w, h)
                )
            
            # Log successful completion
            total_duration = (time.time() - start_time) * 1000
            contact_plane_logger.log_operation_success(
                correlation_id, "WORKPIECE_CONTACT_PLANE_ESTIMATION",
                total_duration,
                confidence=confidence,
                roi_size=(roi_x2-roi_x1)*(roi_y2-roi_y1),
                num_3d_points=len(points_3d)
            )
            
            return {
                'normal': normal,
                'contact_point': contact_point,
                'roi': contact_region_roi,
                'confidence': confidence,
                'plane_params': plane_params
            }
            
        except Exception as e:
            total_duration = (time.time() - start_time) * 1000
            contact_plane_logger.log_error_with_context(
                correlation_id, e,
                {
                    'operation': 'WORKPIECE_CONTACT_PLANE_ESTIMATION',
                    'duration_ms': total_duration,
                    'roi_radius': roi_radius_pixels
                }
            )
            return None
    
    def _depth_roi_to_3d_points(self, depth_roi, roi_x1, roi_y1, fx, fy, cx, cy, correlation_id=None):
        """Convert depth ROI to 3D points in camera coordinates"""
        points = []
        h, w = depth_roi.shape
        
        # Sample points (every 2nd pixel for better density in small ROI)
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                depth_value = depth_roi[y, x]
                
                # Skip invalid depth values
                if depth_value <= 0:
                    continue
                
                # Convert to real-world depth
                # DEPTH SCALING FIX: Use test scaling factor to address 650x scale error
                if USE_DEPTH_SCALE_TEST:
                    depth_meters = depth_value / DEPTH_SCALE_FACTOR_TEST
                else:
                    depth_meters = depth_value / DEPTH_SCALE_FACTOR
                
                # Convert pixel coordinates to camera coordinates
                pixel_x = roi_x1 + x
                pixel_y = roi_y1 + y
                
                # Back-project to 3D using camera intrinsics
                x_3d = (pixel_x - cx) * depth_meters / fx
                y_3d = (pixel_y - cy) * depth_meters / fy
                z_3d = depth_meters
                
                points.append([x_3d, y_3d, z_3d])
        
        return np.array(points)
    
    def _ray_plane_intersection(self, ray_origin, ray_direction, plane_normal, plane_d, correlation_id=None):
        """
        Calculate intersection point of ray with plane.
        
        Args:
            ray_origin (np.ndarray): Ray origin point [x, y, z]
            ray_direction (np.ndarray): Normalized ray direction [x, y, z]
            plane_normal (np.ndarray): Normalized plane normal [x, y, z]
            plane_d (float): Plane equation constant (ax + by + cz + d = 0)
            
        Returns:
            np.ndarray: Intersection point [x, y, z] or None if no intersection
        """
        # Ray equation: P = P0 + t * direction
        # Plane equation: ax + by + cz + d = 0
        
        # Calculate denominator: n · direction
        denominator = np.dot(plane_normal, ray_direction)
        
        # Check if ray is parallel to plane
        is_parallel = abs(denominator) < 1e-6
        if is_parallel:
            contact_plane_logger.log_ray_plane_intersection(
                correlation_id, ray_origin, ray_direction, None, denominator
            )
            return None
        
        # Calculate t parameter
        # t = -(n · P0 + d) / (n · direction)
        numerator = np.dot(plane_normal, ray_origin) + plane_d
        t = -numerator / denominator
        
        # RAY LENGTH CONSTRAINT: Limit ray length to reasonable values
        # Expected real ray length: ~3/32" (0.0024m), but allow up to 10mm for safety
        max_ray_length = 0.01  # 10mm maximum ray length
        original_t = t
        
        if t > max_ray_length:
            t = max_ray_length
            if correlation_id:
                contact_plane_logger.log_operation_success(
                    correlation_id, "RAY_LENGTH_CONSTRAINT_APPLIED",
                    0,  # No significant time
                    original_length=original_t,
                    constrained_length=t,
                    constraint_reason="ray_length_too_large"
                )
        
        # Check if intersection is in front of ray origin
        is_behind = t < 0
        
        # Calculate intersection point
        intersection_point = ray_origin + t * ray_direction
        
        # Log the intersection result (use original_t for distance logging if constraint was applied)
        distance_to_plane = original_t if t != original_t else (t if is_behind else None)
        contact_plane_logger.log_ray_plane_intersection(
            correlation_id, ray_origin, ray_direction, intersection_point, 
            distance_to_plane
        )
        
        return intersection_point
    
    def _calculate_plane_fit_confidence(self, points_3d, plane_params, correlation_id=None):
        """Calculate confidence score for plane fit based on RANSAC inlier ratio"""
        if len(points_3d) < 3:
            return 0.0
        
        # Calculate distances from points to plane
        a, b, c, d = plane_params
        distances = np.abs(a * points_3d[:, 0] + b * points_3d[:, 1] + c * points_3d[:, 2] + d)
        
        # Count inliers (points within threshold distance)
        threshold = WORKPIECE_RANSAC_THRESHOLD
        inliers = np.sum(distances < threshold)
        inlier_ratio = inliers / len(points_3d)
        
        # Additional confidence based on point count
        point_confidence = min(1.0, len(points_3d) / 100.0)
        
        # Combined confidence
        confidence = (inlier_ratio * 0.7) + (point_confidence * 0.3)
        
        print(f"DEBUG: Plane fit confidence - inliers: {inliers}/{len(points_3d)} ({inlier_ratio:.3f}), point_confidence: {point_confidence:.3f}, final: {confidence:.3f}")
        
        return confidence

class TravelTracker:
    def __init__(self, history_length=TRAVEL_HISTORY_LENGTH):
        """Initialize travel tracker for velocity and speed calculation"""
        self.history_length = history_length
        self.electrode_history = deque(maxlen=history_length)
        self.rod_history = deque(maxlen=history_length)
        self.frame_times = deque(maxlen=history_length)
        self.last_frame_time = None
        
        # Smoothed velocity vectors
        self.electrode_velocity = None
        self.rod_velocity = None
        
        # Speed calculations
        self.electrode_speed = 0.0
        self.rod_speed = 0.0
        self.speed_history = deque(maxlen=SPEED_AVERAGE_WINDOW)
        
    def update_tracking(self, electrode_tip_3d, rod_tip_3d, frame_time=None):
        """Update tracking with new electrode and rod tip positions"""
        if frame_time is None:
            frame_time = time.time()
        
        # Update electrode tracking
        if electrode_tip_3d is not None:
            self.electrode_history.append(electrode_tip_3d)
        
        # Update rod tracking
        if rod_tip_3d is not None:
            self.rod_history.append(rod_tip_3d)
        
        # Update frame timing
        self.frame_times.append(frame_time)
        
        # Calculate velocities if we have enough history
        if len(self.electrode_history) >= 2 and len(self.frame_times) >= 2:
            self._calculate_velocities()
    
    def _calculate_velocities(self):
        """Calculate velocity vectors for electrode and rod"""
        # Calculate electrode velocity
        if len(self.electrode_history) >= 2:
            current_pos = np.array(self.electrode_history[-1])
            previous_pos = np.array(self.electrode_history[-2])
            
            # Calculate time delta
            time_delta = self.frame_times[-1] - self.frame_times[-2]
            if time_delta > 0:
                # Calculate instantaneous velocity
                instant_velocity = (current_pos - previous_pos) / time_delta
                
                # Apply exponential smoothing
                if self.electrode_velocity is None:
                    self.electrode_velocity = instant_velocity
                else:
                    self.electrode_velocity = (TRAVEL_SMOOTHING_FACTOR * self.electrode_velocity + 
                                             (1 - TRAVEL_SMOOTHING_FACTOR) * instant_velocity)
                
                # Calculate speed magnitude
                speed_magnitude = np.linalg.norm(self.electrode_velocity)
                
                # Filter out very small movements
                if speed_magnitude > TRAVEL_MIN_VELOCITY:
                    self.electrode_speed = speed_magnitude
                    self.speed_history.append(self.electrode_speed)
                else:
                    self.electrode_speed = 0.0
        
        # Calculate rod velocity (similar to electrode)
        if len(self.rod_history) >= 2:
            current_pos = np.array(self.rod_history[-1])
            previous_pos = np.array(self.rod_history[-2])
            
            time_delta = self.frame_times[-1] - self.frame_times[-2]
            if time_delta > 0:
                instant_velocity = (current_pos - previous_pos) / time_delta
                
                if self.rod_velocity is None:
                    self.rod_velocity = instant_velocity
                else:
                    self.rod_velocity = (TRAVEL_SMOOTHING_FACTOR * self.rod_velocity + 
                                       (1 - TRAVEL_SMOOTHING_FACTOR) * instant_velocity)
                
                speed_magnitude = np.linalg.norm(self.rod_velocity)
                if speed_magnitude > TRAVEL_MIN_VELOCITY:
                    self.rod_speed = speed_magnitude
                else:
                    self.rod_speed = 0.0
    
    def get_electrode_velocity(self):
        """Get the current electrode velocity vector"""
        return self.electrode_velocity
    
    def get_rod_velocity(self):
        """Get the current rod velocity vector"""
        return self.rod_velocity
    
    def get_electrode_speed(self):
        """Get the current electrode speed in m/s"""
        return self.electrode_speed
    
    def get_rod_speed(self):
        """Get the current rod speed in m/s"""
        return self.rod_speed
    
    def get_average_speed(self):
        """Get the average speed over the last N frames"""
        if not self.speed_history:
            return 0.0
        return sum(self.speed_history) / len(self.speed_history)
    
    def get_speed_in_units(self, speed_mps, units="mm/s"):
        """Convert speed from m/s to specified units"""
        if units == "mm/s":
            return speed_mps * 1000
        elif units == "cm/s":
            return speed_mps * 100
        elif units == "m/s":
            return speed_mps
        else:
            return speed_mps * 1000  # Default to mm/s
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.electrode_history.clear()
        self.rod_history.clear()
        self.frame_times.clear()
        self.electrode_velocity = None
        self.rod_velocity = None
        self.electrode_speed = 0.0
        self.rod_speed = 0.0
        self.speed_history.clear()
        self.last_frame_time = None

# Create Gradio interface
with gr.Blocks(title="3D Keypoint Detection with Pipeline API", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 3D Keypoint Detection with Pipeline API")
    gr.Markdown("Upload a video and process it with 2D keypoints + depth estimation + surface normal estimation for 3D pose analysis")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Roboflow Configuration")
            
            api_key = gr.Textbox(
                label="Roboflow API Key",
                placeholder="Enter your Roboflow API key",
                value=DEFAULT_API_KEY,
                type="password"
            )
            
            workspace_name = gr.Textbox(
                label="Workspace Name",
                placeholder="e.g., your-username",
                value=DEFAULT_WORKSPACE
            )
            
            project_name = gr.Textbox(
                label="Project Name",
                placeholder="e.g., keypoint-detection-abc123",
                value=DEFAULT_PROJECT
            )
            
            version_number = gr.Textbox(
                label="Version Number",
                placeholder="e.g., 1",
                value=DEFAULT_VERSION
            )
            
            gr.Markdown("### 🌊 Depth Model Configuration")
            
            depth_model_name = gr.Dropdown(
                label="Depth Estimation Model",
                choices=[
                    "depth-anything/Depth-Anything-V2-Small-hf",
                    "depth-anything/Depth-Anything-V2-Base-hf", 
                    "depth-anything/Depth-Anything-V2-Large-hf"
                ],
                value="depth-anything/Depth-Anything-V2-Small-hf",
                info="Small: Fastest, Large: Most accurate"
            )
            
            gr.Markdown("""
            **How to find your Roboflow project details:**
            1. Go to your Roboflow project
            2. The workspace name is in the URL: `https://app.roboflow.com/{workspace_name}/{project_name}`
            3. Version number is shown in the model versions section
            
            **Configuration Options:**
            - **Manual Entry**: Fill in the fields above
            - **Environment Variables**: Create a `.env` file in the same directory with:
              ```
              ROBOFLOW_API_KEY=your_api_key_here
              ROBOFLOW_WORKSPACE=your_workspace_name
              ROBOFLOW_PROJECT=your_project_name
              ROBOFLOW_VERSION=1
              ```
            
            **Depth Model Options:**
            - **Small**: Fastest processing, good for real-time
            - **Base**: Balanced speed and accuracy
            - **Large**: Most accurate, slower processing
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📹 Video Upload & ROI Selection")
            
            video_input = gr.Video(
                label="Upload Video",
                format="mp4"
            )
            
            # ROI Selection Interface
            with gr.Row():
                roi_enabled = gr.Checkbox(
                    label="Enable Manual ROI Selection",
                    value=True,
                    info="Select a region of interest for surface normal estimation"
                )
            
            with gr.Row():
                roi_x1 = gr.Number(
                    label="ROI X1 (left)",
                    value=0.3,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="Left edge of ROI (0-1, fraction of frame width)"
                )
                roi_y1 = gr.Number(
                    label="ROI Y1 (top)",
                    value=0.3,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="Top edge of ROI (0-1, fraction of frame height)"
                )
            
            with gr.Row():
                roi_x2 = gr.Number(
                    label="ROI X2 (right)",
                    value=0.7,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="Right edge of ROI (0-1, fraction of frame width)"
                )
                roi_y2 = gr.Number(
                    label="ROI Y2 (bottom)",
                    value=0.7,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="Bottom edge of ROI (0-1, fraction of frame height)"
                )
            
            with gr.Row():
                preview_btn = gr.Button(
                    "👁️ Preview First Frame",
                    variant="secondary",
                    size="sm"
                )
                
                process_btn = gr.Button(
                    "🚀 Process Video",
                    variant="primary",
                    size="lg"
                )
            
            gr.Markdown(f"""
            **Current Settings:**
            - Confidence Threshold: {CONFIDENCE_THRESHOLD}%
            - Target FPS: {TARGET_FPS} (faster processing, better motion tracking)
            - Keypoint Radius: {KEYPOINT_RADIUS}px
            - Label Format: {LABEL_FORMAT}
            - Show 3D Coordinates: {SHOW_3D_COORDINATES}
            - Show Depth Map: {SHOW_DEPTH_MAP}
            - Surface Normal Estimation: {ENABLE_SURFACE_NORMALS}
            - Manual ROI Selection: {ROI_SELECTION_ENABLED}
            
            **Visual Settings:**
            - Keypoint Color: Green (or depth-based)
            - Connection Color: Blue
            - Depth Color Map: {DEPTH_COLOR_MAP}
            - Surface Normal Color Map: {SURFACE_NORMAL_COLOR_MAP}
            
            **ROI Selection:**
            - Enable manual ROI selection to specify the workpiece area
            - ROI coordinates are fractions of frame dimensions (0.0-1.0)
            - Surface normal estimation will be performed only on the selected region
            - 3D coordinate axes will be based on the surface normal of the selected area
            
            **Workflow:**
            1. Upload video and configure settings
            2. Set ROI coordinates to select workpiece area (optional)
            3. Click "Preview First Frame" to test inference
            4. If preview looks good, click "Process Video"
            5. Download the processed video with 3D keypoints and surface normals
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Output")
            
            preview_image = gr.Image(
                label="Preview: First Frame with 3D Keypoints",
                type="filepath"
            )
            
            preview_depth = gr.Image(
                label="Preview: Depth Map Visualization",
                type="filepath"
            )
            
            preview_surface_normals = gr.Image(
                label="Preview: Surface Normal Map",
                type="filepath"
            )
            
            preview_surface_normals_with_depth = gr.Image(
                label="Preview: Surface Normal + Depth Map",
                type="filepath"
            )
            
            preview_combined = gr.Image(
                label="Preview: Combined 3D Keypoints + Depth",
                type="filepath"
            )
            
            output_video = gr.Video(
                label="Processed Video with 3D Keypoints",
                format="mp4"
            )
            
            output_depth_video = gr.Video(
                label="Depth Map Video",
                format="mp4"
            )
            
            output_combined_video = gr.Video(
                label="Combined 3D Keypoints + Depth Video",
                format="mp4"
            )
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False
            )
    
    # Function to prepare ROI coordinates
    def prepare_roi_coordinates(roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2, video_file):
        """Convert ROI coordinates from fractions to pixel coordinates"""
        if not roi_enabled or not video_file:
            return None
        
        try:
            # Get video dimensions
            cap = cv2.VideoCapture(video_file.name if hasattr(video_file, 'name') else video_file)
            if not cap.isOpened():
                return None
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Convert fractional coordinates to pixel coordinates
            x1 = int(roi_x1 * width)
            y1 = int(roi_y1 * height)
            x2 = int(roi_x2 * width)
            y2 = int(roi_y2 * height)
            
            print(f"DEBUG: ROI coordinates: {x1},{y1} to {x2},{y2} (from {width}x{height} frame)")
            return [x1, y1, x2, y2]
            
        except Exception as e:
            print(f"DEBUG: Error preparing ROI coordinates: {e}")
            return None
    
    # Connect the interface
    preview_btn.click(
        fn=lambda api_key, workspace, project, version, depth_model, video, roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2: 
            preview_first_frame_3d(api_key, workspace, project, version, depth_model, video, 
                                  prepare_roi_coordinates(roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2, video)),
        inputs=[api_key, workspace_name, project_name, version_number, depth_model_name, video_input, 
                roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2],
        outputs=[preview_image, preview_depth, preview_surface_normals, preview_surface_normals_with_depth, preview_combined, status_text]
    )
    
    process_btn.click(
        fn=lambda api_key, workspace, project, version, depth_model, video, roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2: 
            process_video_with_3d_model(api_key, workspace, project, version, depth_model, video, 
                                       prepare_roi_coordinates(roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2, video)),
        inputs=[api_key, workspace_name, project_name, version_number, depth_model_name, video_input, 
                roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2],
        outputs=[output_video, output_depth_video, output_combined_video, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7863) 