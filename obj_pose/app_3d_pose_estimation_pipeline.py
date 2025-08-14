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

# Camera Parameters (you'll need to calibrate these for your camera)
CAMERA_FX = 1512.0  # Focal length in pixels (X direction) - iPhone 15 52mm telephoto
CAMERA_FY = 1512.0  # Focal length in pixels (Y direction) - iPhone 15 52mm telephoto
CAMERA_CX = 1080.0  # Principal point X coordinate (center of 2160px width)
CAMERA_CY = 607.0   # Principal point Y coordinate (center of 1214px height)

# Depth scaling parameters
DEPTH_SCALE_METERS = 0.11  # Convert depth values to meters (11cm per unit) - adjusted for electrode length
DEPTH_OFFSET = 0.0          # Depth offset in meters

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
# END CONFIGURATION PARAMETERS
# ============================================================================

class DepthEstimatorPipeline:
    def __init__(self, model_name=DEPTH_MODEL_NAME):
        """Initialize depth estimator with Hugging Face Pipeline"""
        try:
            print(f"Loading depth estimation pipeline: {model_name}")
            self.pipe = pipeline(
                task="depth-estimation", 
                model=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"✅ Depth estimation pipeline loaded successfully on {self.pipe.device}")
            
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
            result = self.pipe(image)
            depth_map = result["depth"]
            
            print(f"DEBUG: Depth estimation result shape: {depth_map.size}")
            
            # Convert PIL image to numpy array
            depth_array = np.array(depth_map)
            
            print(f"DEBUG: Depth array shape: {depth_array.shape}")
            print(f"DEBUG: Depth array min: {depth_array.min()}, max: {depth_array.max()}")
            
            # Normalize to 0-255 range for visualization
            depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            
            print(f"DEBUG: Normalized depth shape: {depth_normalized.shape}")
            print(f"DEBUG: Normalized depth min: {depth_normalized.min()}, max: {depth_normalized.max()}")
            
            return depth_normalized
            
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
            self.depth_estimator = DepthEstimatorPipeline(depth_model_name)
            print("✅ Depth estimator pipeline initialized")
            
            # Initialize workpiece detector
            print("Initializing workpiece detector...")
            self.workpiece_detector = WorkpieceDetector()
            print("✅ Workpiece detector initialized")
            
            # Initialize travel tracker
            print("Initializing travel tracker...")
            self.travel_tracker = TravelTracker()
            print("✅ Travel tracker initialized")
                
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
        
        # Get depth value (normalized 0-255)
        depth_value = depth_map[y, x]
        
        print(f"DEBUG: Depth map shape: {depth_map.shape}")
        print(f"DEBUG: Requested coordinates: ({x}, {y})")
        print(f"DEBUG: Raw depth value: {depth_value}")
        
        # Convert to real-world depth (adjust scale factor as needed)
        real_depth = depth_value / DEPTH_SCALE_FACTOR
        
        print(f"DEBUG: Real depth (after scaling): {real_depth}")
        
        return real_depth
    
    def pixel_to_3d_coordinates(self, pixel_x, pixel_y, depth_value):
        """Convert 2D pixel coordinates + depth to 3D world coordinates"""
        print(f"\n=== DEBUG: 3D Coordinate Conversion ===")
        print(f"Input: pixel_x={pixel_x}, pixel_y={pixel_y}, depth_value={depth_value}")
        print(f"Camera params: fx={CAMERA_FX}, fy={CAMERA_FY}, cx={CAMERA_CX}, cy={CAMERA_CY}")
        
        if depth_value is None:
            print(f"DEBUG: Depth value is None, returning None coordinates")
            return None, None, None
        
        # The depth_value is already in the range 0-255 (normalized)
        # We need to convert this to real-world meters
        # For Depth Anything models, we need to scale this properly
        
        # Convert normalized depth (0-255) to meters
        # This is the key fix - we need to scale the depth properly
        depth_meters = depth_value * DEPTH_SCALE_METERS + DEPTH_OFFSET
        
        print(f"DEBUG: Depth in meters: {depth_meters}")
        
        # Convert pixel coordinates to 3D using camera intrinsics
        # Z = depth_meters
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        x_3d = (pixel_x - CAMERA_CX) * depth_meters / CAMERA_FX
        y_3d = (pixel_y - CAMERA_CY) * depth_meters / CAMERA_FY
        z_3d = depth_meters
        
        print(f"DEBUG: 3D coordinates: x={x_3d}, y={y_3d}, z={z_3d}")
        
        return x_3d, y_3d, z_3d
    
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
        if not predictions:
            print("No predictions to draw")
            return frame
            
        img_height, img_width = frame.shape[:2]
        print(f"Drawing 3D keypoints on frame size: {img_width}x{img_height}")
        print(f"Processing {len(predictions)} predictions")
        
        # Detect workpiece surface on first frame or when depth map is available
        if depth_map is not None:
            # Collect all keypoints for workpiece detection
            all_keypoints_3d = []
            for prediction in predictions:
                keypoints = prediction.get('keypoints', [])
                for keypoint in keypoints:
                    if 'x_3d' in keypoint and 'y_3d' in keypoint and 'z_3d' in keypoint:
                        all_keypoints_3d.append(keypoint)
            
            self.workpiece_detector.detect_workpiece_surface(depth_map, img_width, img_height, all_keypoints_3d)
        
        # Draw coordinate axes if workpiece is detected
        self.draw_coordinate_axes(frame, img_width, img_height)
        
        # Draw workpiece surface indicator
        self.draw_workpiece_surface_indicator(frame, img_width, img_height)
        
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
        """Draw workpiece coordinate axes on the frame"""
        workpiece_normal = self.workpiece_detector.get_workpiece_normal()
        if workpiece_normal is None:
            print("DEBUG: No workpiece normal available for coordinate axes")
            return
        
        print("DEBUG: Drawing workpiece coordinate axes")
        
        # Get coordinate transformation matrix
        transform_matrix = self.workpiece_detector.coordinate_transform_matrix
        if transform_matrix is None:
            print("DEBUG: No coordinate transformation matrix available")
            return
        
        # Define axis vectors in workpiece coordinates
        x_axis_wp = np.array([COORDINATE_AXIS_LENGTH, 0, 0, 1])  # X-axis
        y_axis_wp = np.array([0, COORDINATE_AXIS_LENGTH, 0, 1])  # Y-axis
        z_axis_wp = np.array([0, 0, COORDINATE_AXIS_LENGTH, 1])  # Z-axis
        
        # Transform to camera coordinates
        try:
            x_axis_cam = np.dot(np.linalg.inv(transform_matrix), x_axis_wp)[:3]
            y_axis_cam = np.dot(np.linalg.inv(transform_matrix), y_axis_wp)[:3]
            z_axis_cam = np.dot(np.linalg.inv(transform_matrix), z_axis_wp)[:3]
        except np.linalg.LinAlgError as e:
            print(f"DEBUG: Linear algebra error in coordinate transformation: {e}")
            return
        
        # Project to 2D image coordinates
        def project_3d_to_2d(point_3d):
            if point_3d[2] <= 0:
                return None
            x = int(CAMERA_CX + point_3d[0] * CAMERA_FX / point_3d[2])
            y = int(CAMERA_CY + point_3d[1] * CAMERA_FY / point_3d[2])
            return (x, y)
        
        # Project origin and axis endpoints
        # Use a point closer to camera for better visibility
        origin_3d = np.array([0, 0, 0.3])  # 30cm from camera
        origin_2d = project_3d_to_2d(origin_3d)
        if origin_2d is None:
            print("DEBUG: Could not project coordinate origin to 2D")
            return
        
        # Check if origin is within frame bounds
        if not (0 <= origin_2d[0] < img_width and 0 <= origin_2d[1] < img_height):
            print(f"DEBUG: Origin outside frame bounds: {origin_2d}")
            return
        
        axes_drawn = 0
        
        # Draw axes with better error handling
        try:
            # X-axis (red)
            x_end_2d = project_3d_to_2d(origin_3d + np.array([0.1, 0, 0]))
            if x_end_2d and 0 <= x_end_2d[0] < img_width and 0 <= x_end_2d[1] < img_height:
                cv2.arrowedLine(frame, origin_2d, x_end_2d, X_AXIS_COLOR, AXIS_THICKNESS, tipLength=0.3)
                cv2.putText(frame, "X", x_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.8, X_AXIS_COLOR, 2)
                axes_drawn += 1
                print("DEBUG: Drew X-axis")
            
            # Y-axis (green)
            y_end_2d = project_3d_to_2d(origin_3d + np.array([0, 0.1, 0]))
            if y_end_2d and 0 <= y_end_2d[0] < img_width and 0 <= y_end_2d[1] < img_height:
                cv2.arrowedLine(frame, origin_2d, y_end_2d, Y_AXIS_COLOR, AXIS_THICKNESS, tipLength=0.3)
                cv2.putText(frame, "Y", y_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.8, Y_AXIS_COLOR, 2)
                axes_drawn += 1
                print("DEBUG: Drew Y-axis")
            
            # Z-axis (blue) - pointing up from workpiece
            z_end_2d = project_3d_to_2d(origin_3d + np.array([0, 0, 0.1]))
            if z_end_2d and 0 <= z_end_2d[0] < img_width and 0 <= z_end_2d[1] < img_height:
                cv2.arrowedLine(frame, origin_2d, z_end_2d, Z_AXIS_COLOR, AXIS_THICKNESS, tipLength=0.3)
                cv2.putText(frame, "Z", z_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.8, Z_AXIS_COLOR, 2)
                axes_drawn += 1
                print("DEBUG: Drew Z-axis")
            
        except Exception as e:
            print(f"DEBUG: Error drawing coordinate axes: {e}")
        
        print(f"DEBUG: Drew {axes_drawn} coordinate axes")
        
        # Also draw a small circle at the origin for reference
        cv2.circle(frame, origin_2d, 5, (255, 255, 255), -1)
        print(f"DEBUG: Drew origin point at {origin_2d}")
        
        # If no axes were drawn, try a simple fallback
        if axes_drawn == 0:
            print("DEBUG: Trying fallback coordinate axes")
            self._draw_simple_coordinate_axes(frame, img_width, img_height)
    
    def _draw_simple_coordinate_axes(self, frame, img_width, img_height):
        """Draw simple coordinate axes as fallback when complex transformation fails"""
        # Draw axes at the center of the frame
        center_x = img_width // 2
        center_y = img_height // 2
        
        # Simple fixed-length axes
        axis_length = 50
        
        # X-axis (red) - horizontal
        x_end = (center_x + axis_length, center_y)
        cv2.arrowedLine(frame, (center_x, center_y), x_end, X_AXIS_COLOR, AXIS_THICKNESS, tipLength=0.3)
        cv2.putText(frame, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, X_AXIS_COLOR, 2)
        
        # Y-axis (green) - vertical
        y_end = (center_x, center_y - axis_length)
        cv2.arrowedLine(frame, (center_x, center_y), y_end, Y_AXIS_COLOR, AXIS_THICKNESS, tipLength=0.3)
        cv2.putText(frame, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, Y_AXIS_COLOR, 2)
        
        # Z-axis (blue) - diagonal (representing depth)
        z_end = (center_x + axis_length//2, center_y - axis_length//2)
        cv2.arrowedLine(frame, (center_x, center_y), z_end, Z_AXIS_COLOR, AXIS_THICKNESS, tipLength=0.3)
        cv2.putText(frame, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, Z_AXIS_COLOR, 2)
        
        # Draw origin point
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        
        print("DEBUG: Drew simple fallback coordinate axes")
    
    def draw_workpiece_surface_indicator(self, frame, img_width, img_height):
        """Draw a visual indicator of the detected workpiece surface"""
        workpiece_normal = self.workpiece_detector.get_workpiece_normal()
        if workpiece_normal is None:
            return
        
        # Removed visual indicators for preview
        # No longer drawing green rectangle, surface normal arrow, or normal vector values
        
        print("DEBUG: Drew workpiece surface indicator")
    
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
        """Create a colored depth map visualization"""
        if depth_map is None:
            return None
        
        # Apply color map to depth map
        depth_colored = cv2.applyColorMap(depth_map, DEPTH_COLOR_MAP)
        
        # Resize to match original frame
        h, w = original_frame.shape[:2]
        depth_colored = cv2.resize(depth_colored, (w, h))
        
        return depth_colored
    
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
        print(f"Creating 3D processor with:")
        print(f"  Workspace: {workspace_name}")
        print(f"  Project: {project_name}")
        print(f"  Version: {version_number}")
        print(f"  Using depth model: {depth_model_name}")
        
        processor = KeypointProcessor3D(api_key, workspace_name, project_name, int(version_number), depth_model_name)
        
        # Check if model was loaded successfully
        if processor.model is None:
            return None, "Error: Model is None - check your project details"
        
        print("✅ 3D Model loaded successfully!")
        return processor, "3D Model loaded successfully!"
    except Exception as e:
        print(f"❌ Error creating 3D processor: {e}")
        return None, f"Error loading 3D model: {str(e)}"

def process_video_with_3d_model(api_key, workspace_name, project_name, version_number, depth_model_name, video_file):
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

def preview_first_frame_3d(api_key, workspace_name, project_name, version_number, depth_model_name, video_file):
    """Preview 3D inference on the first frame of the video"""
    try:
        if not video_file:
            return None, None, None, "Please upload a video file"
        
        if not all([api_key, workspace_name, project_name, version_number]):
            return None, None, None, "Please fill in all Roboflow configuration fields"
        
        # Create processor
        processor, status = create_processor_3d(api_key, workspace_name, project_name, version_number, depth_model_name)
        if processor is None:
            return None, None, None, status
        
        # Additional check for None model
        if processor.model is None:
            return None, None, None, "Error: Model failed to load - check your Roboflow configuration"
        
        # Handle video file path
        video_path = video_file if isinstance(video_file, str) else video_file.name
        
        # Check if file exists
        if not os.path.exists(video_path):
            return None, None, None, f"Video file not found: {video_path}"
        
        # Extract first frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None, "Error: Could not open video file"
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, None, None, "Error: Could not read first frame"
        
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
            else:
                print("⚠️ No depth map available for workpiece detection")
            
            # Draw 3D keypoints on frame
            if actual_predictions:
                frame = processor.draw_keypoints_3d(frame, actual_predictions, depth_map)
                total_keypoints = sum(len(pred.get('keypoints', [])) for pred in actual_predictions)
                depth_status = "with depth" if depth_map is not None else "without depth"
                print(f"Preview: Found {len(actual_predictions)} predictions with {total_keypoints} total keypoints {depth_status}")
            
            # Create depth visualization
            depth_visualization = None
            if SHOW_DEPTH_MAP and depth_map is not None:
                depth_visualization = processor.create_depth_visualization(depth_map, frame)
            
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
            
            print(f"\n=== PREVIEW SUMMARY ===")
            print(f"Keypoints: {total_keypoints} total")
            print(f"Depth: {depth_status}")
            print(f"Workpiece: {workpiece_status}")
            
            return output_path.name, depth_output_path.name if depth_output_path else None, combined_output_path.name if combined_output_path else None, f"Preview: Found {len(actual_predictions)} predictions with {total_keypoints} total keypoints {depth_status} {workpiece_status}"
            
        except Exception as e:
            # Clean up temporary frame file
            if os.path.exists(temp_frame_path.name):
                os.unlink(temp_frame_path.name)
            print(f"Error in preview: {e}")
            return None, None, None, f"Error during preview: {str(e)}"
            
    except Exception as e:
        print(f"Error in preview_first_frame_3d: {e}")
        return None, None, None, f"Error: {str(e)}"

class WorkpieceDetector:
    def __init__(self):
        """Initialize workpiece detector for surface detection and coordinate transformation"""
        self.workpiece_normal = None
        self.workpiece_origin = None
        self.coordinate_transform_matrix = None
        self.detection_confidence = 0.0
        
    def detect_workpiece_surface(self, depth_map, frame_width, frame_height, keypoints_3d=None):
        """Detect workpiece surface using depth map and optionally keypoint guidance"""
        if depth_map is None:
            print("⚠️ No depth map available for workpiece detection")
            return False
        
        print(f"DEBUG: Starting workpiece detection with depth map shape: {depth_map.shape}")
        print(f"DEBUG: Frame dimensions: {frame_width}x{frame_height}")
        
        try:
            # If we have keypoints, use them to guide surface detection
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
        
        # Fit plane to the largest cluster
        plane_params = self._fit_plane_to_points(largest_cluster)
        
        if plane_params is None:
            print("Failed to fit plane to workpiece surface")
            return False
        
        # Extract plane parameters (ax + by + cz + d = 0)
        a, b, c, d = plane_params
        print(f"DEBUG: Background-based plane parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
        
        # Normalize the normal vector
        normal = np.array([a, b, c])
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            print("Invalid normal vector")
            return False
        
        normal = normal / normal_norm
        
        # Ensure normal points toward camera (positive Z in camera coordinates)
        if normal[2] < 0:
            normal = -normal
            d = -d
        
        # Set workpiece coordinate system
        self.workpiece_normal = normal
        self.workpiece_origin = np.array([0, 0, -d/c]) if c != 0 else np.array([0, 0, 0])
        
        # Calculate coordinate transformation matrix
        self._calculate_coordinate_transform()
        
        # Calculate detection confidence based on cluster size and proximity to electrode
        self.detection_confidence = min(1.0, len(largest_cluster) / 500.0)
        
        print(f"✅ Workpiece surface detected with keypoint guidance (confidence: {self.detection_confidence:.2f})")
        print(f"Workpiece normal: {normal}")
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
        
        # Fit plane to the largest cluster
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
        
        # Validate that the detected surface is reasonable
        # For a camera looking down at an angle, the normal should have a significant Z component
        # but not be perfectly aligned with camera Z-axis
        z_component = abs(normal[2])
        if z_component > 0.95:  # Too close to camera Z-axis
            print(f"WARNING: Detected surface normal too close to camera Z-axis: {normal}")
            print("This might be detecting background rather than workpiece surface")
            print("Expected: Normal should reflect camera angle (e.g., [0.2, -0.1, 0.9])")
            print("Actual: Normal is [0, 0, 1] indicating surface perpendicular to camera view")
            print("Trying to detect the actual workpiece surface instead...")
            
            # Try to find a cluster with more variation in Z-coordinates
            clusters_with_variation = []
            for i, cluster in enumerate(clusters):
                if len(cluster) >= WORKPIECE_MIN_CLUSTER_SIZE:
                    z_values = [p[2] for p in cluster]
                    z_variation = max(z_values) - min(z_values)
                    if z_variation > 0.005:  # At least 5mm variation
                        clusters_with_variation.append((i, cluster, z_variation))
                        print(f"DEBUG: Cluster {i} has Z variation: {z_variation:.4f}")
            
            if clusters_with_variation:
                # Use the cluster with the most Z-variation (likely the actual workpiece)
                best_cluster_idx, best_cluster, best_variation = max(clusters_with_variation, key=lambda x: x[2])
                print(f"DEBUG: Using cluster {best_cluster_idx} with Z variation {best_variation:.4f}")
                
                # Fit plane to the best cluster
                plane_params = self._fit_plane_to_points(best_cluster)
                if plane_params is not None:
                    a, b, c, d = plane_params
                    normal = np.array([a, b, c])
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm > 0:
                        normal = normal / normal_norm
                        print(f"DEBUG: New plane parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
                        print(f"DEBUG: New normal vector: {normal}")
        
        # Ensure normal points toward camera (positive Z in camera coordinates)
        if normal[2] < 0:
            normal = -normal
            d = -d
        
        # Set workpiece coordinate system
        self.workpiece_normal = normal
        self.workpiece_origin = np.array([0, 0, -d/c]) if c != 0 else np.array([0, 0, 0])
        
        # Calculate coordinate transformation matrix
        self._calculate_coordinate_transform()
        
        # Calculate detection confidence based on cluster size and plane fit quality
        self.detection_confidence = min(1.0, len(largest_cluster) / 1000.0)
        
        print(f"✅ Workpiece surface detected with confidence {self.detection_confidence:.2f}")
        print(f"Workpiece normal: {normal}")
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
    
    def _fit_plane_to_points(self, points):
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
    
    def _calculate_coordinate_transform(self):
        """Calculate transformation matrix from camera to workpiece coordinates"""
        if self.workpiece_normal is None:
            return
        
        # Workpiece Z-axis is the surface normal
        z_axis = self.workpiece_normal
        
        # Workpiece X-axis: project camera X-axis onto workpiece plane
        camera_x = np.array([1, 0, 0])
        x_axis = camera_x - np.dot(camera_x, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Workpiece Y-axis: cross product of Z and X axes
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Build rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Build transformation matrix (rotation + translation)
        self.coordinate_transform_matrix = np.eye(4)
        self.coordinate_transform_matrix[:3, :3] = rotation_matrix
        self.coordinate_transform_matrix[:3, 3] = -np.dot(rotation_matrix, self.workpiece_origin)
    
    def transform_to_workpiece_coordinates(self, point_3d):
        """Transform a point from camera coordinates to workpiece coordinates"""
        if self.coordinate_transform_matrix is None:
            return point_3d
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(point_3d, 1)
        
        # Transform
        transformed_homogeneous = np.dot(self.coordinate_transform_matrix, point_homogeneous)
        
        return transformed_homogeneous[:3]
    
    def get_workpiece_normal(self):
        """Get the workpiece surface normal vector"""
        return self.workpiece_normal
    
    def get_workpiece_origin(self):
        """Get the workpiece coordinate system origin"""
        return self.workpiece_origin
    
    def get_detection_confidence(self):
        """Get the confidence of workpiece detection"""
        return self.detection_confidence

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
    gr.Markdown("# 🎯 3D Keypoint Detection with Pipeline API")
    gr.Markdown("Upload a video and process it with 2D keypoints + depth estimation for 3D pose analysis")
    
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
            gr.Markdown("### 📹 Video Upload")
            
            video_input = gr.Video(
                label="Upload Video",
                format="mp4"
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
            
            **Visual Settings:**
            - Keypoint Color: Green (or depth-based)
            - Connection Color: Blue
            - Depth Color Map: {DEPTH_COLOR_MAP}
            
            **Workflow:**
            1. Upload video and configure settings
            2. Click "Preview First Frame" to test inference
            3. If preview looks good, click "Process Video"
            4. Download the processed video with 3D keypoints
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
    
    # Connect the interface
    preview_btn.click(
        fn=preview_first_frame_3d,
        inputs=[api_key, workspace_name, project_name, version_number, depth_model_name, video_input],
        outputs=[preview_image, preview_depth, preview_combined, status_text]
    )
    
    process_btn.click(
        fn=process_video_with_3d_model,
        inputs=[api_key, workspace_name, project_name, version_number, depth_model_name, video_input],
        outputs=[output_video, output_depth_video, output_combined_video, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7863) 