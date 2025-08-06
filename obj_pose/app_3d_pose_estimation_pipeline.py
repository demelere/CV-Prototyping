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
        """Calculate welding angles from electrode and filler rod geometry"""
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
    
    def process_3d_pose_data(self, keypoints_3d):
        """Process 3D keypoints to extract welding geometry and angles"""
        pose_data = {
            'keypoints_3d': keypoints_3d,
            'electrode_geometry': None,
            'filler_geometry': None,
            'welding_angles': None
        }
        
        if keypoints_3d:
            # Calculate electrode geometry
            electrode_data = self.calculate_electrode_geometry(keypoints_3d)
            if electrode_data:
                pose_data['electrode_geometry'] = electrode_data
            
            # Calculate filler rod geometry
            filler_data = self.calculate_filler_rod_geometry(keypoints_3d)
            if filler_data:
                pose_data['filler_geometry'] = filler_data
            
            # Calculate welding angles
            # For now, assume workpiece normal is [0, 0, 1] (pointing up)
            # You'll need to detect or calibrate this for your specific setup
            workpiece_normal = np.array([0, 0, 1])
            angles = self.calculate_welding_angles(electrode_data, filler_data, workpiece_normal)
            if angles:
                pose_data['welding_angles'] = angles
        
        return pose_data
    
    def format_3d_label(self, keypoint_3d):
        """Format label for 3D keypoint display"""
        name = keypoint_3d['name']
        confidence = keypoint_3d['confidence']
        x_3d = keypoint_3d['x_3d']
        y_3d = keypoint_3d['y_3d']
        z_3d = keypoint_3d['z_3d']
        
        if SHOW_3D_COORDINATES and all(v is not None for v in [x_3d, y_3d, z_3d]):
            # Convert to millimeters for better readability
            x_mm = x_3d * 1000
            y_mm = y_3d * 1000
            z_mm = z_3d * 1000
            return f"{name}: ({x_mm:.1f}mm, {y_mm:.1f}mm, {z_mm:.1f}mm)"
        elif SHOW_CONFIDENCE:
            return f"{name}: {confidence:.{CONFIDENCE_DECIMAL_PLACES}f}"
        else:
            return name
    
    def draw_keypoints_3d(self, frame, predictions, depth_map=None):
        """Draw 3D keypoints and connections on the frame"""
        if not predictions:
            print("No predictions to draw")
            return frame
            
        img_height, img_width = frame.shape[:2]
        print(f"Drawing 3D keypoints on frame size: {img_width}x{img_height}")
        print(f"Processing {len(predictions)} predictions")
        
        # Process each prediction (could be multiple objects)
        for prediction in predictions:
            keypoints_2d = prediction.get('keypoints', [])
            if not keypoints_2d:
                print("No keypoints in this prediction")
                continue
            
            # Convert 2D keypoints to 3D
            keypoints_3d = self.combine_2d_and_depth(keypoints_2d, depth_map, img_width, img_height)
            
            # Draw 3D keypoints
            for i, keypoint_3d in enumerate(keypoints_3d):
                x_2d = keypoint_3d['x_2d']
                y_2d = keypoint_3d['y_2d']
                confidence = keypoint_3d['confidence']
                depth = keypoint_3d['depth']
                
                # Skip low confidence keypoints
                confidence_threshold = CONFIDENCE_THRESHOLD / 100.0 if CONFIDENCE_THRESHOLD > 1 else CONFIDENCE_THRESHOLD
                if confidence < confidence_threshold:
                    print(f"  Skipping keypoint {i} due to low confidence")
                    continue
                
                # Color based on depth (if available)
                if depth is not None:
                    # Normalize depth to color range
                    depth_normalized = min(255, max(0, int(depth * 255)))
                    color = (0, depth_normalized, 255 - depth_normalized)  # BGR: blue to red
                else:
                    color = KEYPOINT_COLOR
                
                # Draw keypoint circle
                cv2.circle(frame, (x_2d, y_2d), KEYPOINT_RADIUS, color, KEYPOINT_THICKNESS)
                
                # Draw label
                label = self.format_3d_label(keypoint_3d)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)[0]
                
                # Position label above keypoint
                label_x = x_2d - label_size[0] // 2
                label_y = y_2d - KEYPOINT_RADIUS - 8
                
                # Ensure label is within image bounds
                if label_y < label_size[1]:
                    label_y = y_2d + KEYPOINT_RADIUS + label_size[1] + 8
                
                # Draw label background (larger and more opaque)
                cv2.rectangle(frame, 
                            (label_x - 8, label_y - label_size[1] - 8),
                            (label_x + label_size[0] + 8, label_y + 8),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (label_x, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, (255, 255, 255), LABEL_FONT_THICKNESS)
            
            # Draw connections between keypoints if defined
            if KEYPOINT_CONNECTIONS:
                for connection in KEYPOINT_CONNECTIONS:
                    if len(keypoints_3d) > max(connection):
                        kp1 = keypoints_3d[connection[0]]
                        kp2 = keypoints_3d[connection[1]]
                        
                        # Check confidence for both keypoints
                        confidence_threshold = CONFIDENCE_THRESHOLD / 100.0 if CONFIDENCE_THRESHOLD > 1 else CONFIDENCE_THRESHOLD
                        if (kp1['confidence'] >= confidence_threshold and 
                            kp2['confidence'] >= confidence_threshold):
                            
                            # Draw connection line
                            cv2.line(frame, (kp1['x_2d'], kp1['y_2d']), 
                                    (kp2['x_2d'], kp2['y_2d']), 
                                    CONNECTION_COLOR, CONNECTION_THICKNESS)
        
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
            for keypoint_3d in keypoints_3d:
                x_2d = keypoint_3d['x_2d']
                y_2d = keypoint_3d['y_2d']
                confidence = keypoint_3d['confidence']
                depth = keypoint_3d['depth']
                
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
                
                # Draw keypoint circle on depth map
                cv2.circle(combined_frame, (x_2d, y_2d), KEYPOINT_RADIUS, color, KEYPOINT_THICKNESS)
                
                # Draw label
                label = self.format_3d_label(keypoint_3d)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)[0]
                
                # Position label above keypoint
                label_x = x_2d - label_size[0] // 2
                label_y = y_2d - KEYPOINT_RADIUS - 8
                
                # Ensure label is within image bounds
                if label_y < label_size[1]:
                    label_y = y_2d + KEYPOINT_RADIUS + label_size[1] + 8
                
                # Draw label background (larger and more opaque)
                cv2.rectangle(combined_frame, 
                            (label_x - 8, label_y - label_size[1] - 8),
                            (label_x + label_size[0] + 8, label_y + 8),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(combined_frame, label, (label_x, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, (255, 255, 255), LABEL_FONT_THICKNESS)
            
            # Draw connections between keypoints if defined
            if KEYPOINT_CONNECTIONS:
                for connection in KEYPOINT_CONNECTIONS:
                    if len(keypoints_3d) > max(connection):
                        kp1 = keypoints_3d[connection[0]]
                        kp2 = keypoints_3d[connection[1]]
                        
                        # Check confidence for both keypoints
                        confidence_threshold = CONFIDENCE_THRESHOLD / 100.0 if CONFIDENCE_THRESHOLD > 1 else CONFIDENCE_THRESHOLD
                        if (kp1['confidence'] >= confidence_threshold and 
                            kp2['confidence'] >= confidence_threshold):
                            
                            # Draw connection line
                            cv2.line(combined_frame, (kp1['x_2d'], kp1['y_2d']), 
                                    (kp2['x_2d'], kp2['y_2d']), 
                                    CONNECTION_COLOR, CONNECTION_THICKNESS)
        
        # Add a title overlay (bigger and clearer)
        overlay = np.zeros_like(combined_frame)
        cv2.rectangle(overlay, (10, 10), (600, 100), (0, 0, 0), -1)
        cv2.putText(overlay, "3D Keypoints + Depth Map", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # Blend the overlay (less transparent for clarity)
        alpha = 0.95
        combined_frame = cv2.addWeighted(combined_frame, alpha, overlay, 1-alpha, 0)
        
        # Add geometry information overlay (create dummy predictions structure)
        if keypoints_3d:
            # Create a dummy predictions structure for geometry calculation
            dummy_predictions = [{'keypoints': keypoints_3d}]
            self.draw_geometry_overlay(combined_frame, dummy_predictions, depth_map, original_frame.shape[1], original_frame.shape[0])
        
        return combined_frame
    
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
                            pose_data = self.process_3d_pose_data(keypoints_3d_all)
                            
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
                                
                                if pose_data['welding_angles']:
                                    angles = pose_data['welding_angles']
                                    if 'angle_to_camera_degrees' in angles:
                                        print(f"Frame {processed_count}: Electrode angle to camera = {angles['angle_to_camera_degrees']:.1f}°")
                    
                    # Create depth visualization
                    depth_visualization = None
                    if SHOW_DEPTH_MAP and depth_map is not None:
                        depth_visualization = self.create_depth_visualization(depth_map, frame)
                    
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
                    combined_visualization = processor.create_combined_visualization(
                        frame, depth_map, frame, keypoints_3d_all
                    )
            
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
            return output_path.name, depth_output_path.name if depth_output_path else None, combined_output_path.name if combined_output_path else None, f"Preview: Found {len(actual_predictions)} predictions with {total_keypoints} total keypoints {depth_status}"
            
        except Exception as e:
            # Clean up temporary frame file
            if os.path.exists(temp_frame_path.name):
                os.unlink(temp_frame_path.name)
            print(f"Error in preview: {e}")
            return None, None, None, f"Error during preview: {str(e)}"
            
    except Exception as e:
        print(f"Error in preview_first_frame_3d: {e}")
        return None, None, None, f"Error: {str(e)}"

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
                type="password"
            )
            
            workspace_name = gr.Textbox(
                label="Workspace Name",
                placeholder="e.g., your-username"
            )
            
            project_name = gr.Textbox(
                label="Project Name",
                placeholder="e.g., keypoint-detection-abc123"
            )
            
            version_number = gr.Textbox(
                label="Version Number",
                placeholder="e.g., 1",
                value="1"
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