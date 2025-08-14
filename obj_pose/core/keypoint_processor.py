"""
3D Keypoint processing module.
Handles 2D keypoint detection, depth integration, and 3D coordinate conversion.
"""

import cv2
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
    
    def predict_keypoints(self, image_path):
        """Predict keypoints using Roboflow model"""
        try:
            if self.model is None:
                print("❌ Model not loaded")
                return None
            
            print(f"Predicting keypoints for {image_path}")
            result = self.model.predict(image_path, confidence=CONFIDENCE_THRESHOLD, overlap=30)
            
            # Convert to JSON format
            predictions = result.json()
            print(f"✅ Predicted {len(predictions.get('predictions', []))} objects")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error in keypoint prediction: {e}")
            return None
    
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
