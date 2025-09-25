"""
3D Keypoint processing module.
Handles 2D keypoint detection, depth integration, and 3D coordinate conversion.
"""

import numpy as np
from roboflow import Roboflow
from .depth_estimator import DepthEstimatorPipeline
from .workpiece_detector import WorkpieceDetector
from .travel_tracker import TravelTracker
from config.settings import *


class KeypointProcessor3D:
    """Handles 3D keypoint processing with depth estimation and pose analysis"""
    
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
        
        # Convert normalized depth (0-255) to meters
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
