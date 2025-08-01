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

# 3D Visualization Parameters
SHOW_DEPTH_MAP = True              # Show depth map visualization
SHOW_3D_COORDINATES = True         # Show 3D coordinates in labels
DEPTH_SCALE_FACTOR = 1000.0        # Scale factor for depth values (adjust based on your depth model)

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
            
            # Load image with PIL
            image = Image.open(image_path)
            
            # Get depth estimation
            result = self.pipe(image)
            depth_map = result["depth"]
            
            # Convert PIL image to numpy array
            depth_array = np.array(depth_map)
            
            # Normalize to 0-255 range for visualization
            depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            
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
            return None
        
        # Ensure coordinates are within bounds
        h, w = depth_map.shape
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        
        # Get depth value (normalized 0-255)
        depth_value = depth_map[y, x]
        
        # Convert to real-world depth (adjust scale factor as needed)
        real_depth = depth_value / DEPTH_SCALE_FACTOR
        
        return real_depth
    
    def combine_2d_and_depth(self, keypoints_2d, depth_map, img_width, img_height):
        """Combine 2D keypoints with depth information to get 3D coordinates"""
        keypoints_3d = []
        
        for keypoint in keypoints_2d:
            x = keypoint.get('x', 0)
            y = keypoint.get('y', 0)
            confidence = keypoint.get('confidence', 0)
            keypoint_name = keypoint.get('name', 'unknown')
            
            # Convert coordinates to pixels if they're percentages
            if x <= 1 and y <= 1:
                pixel_x = int(x * img_width)
                pixel_y = int(y * img_height)
            else:
                pixel_x = int(x)
                pixel_y = int(y)
            
            # Get depth at this keypoint
            depth = self.get_depth_at_keypoint(depth_map, pixel_x, pixel_y)
            
            # Create 3D keypoint
            keypoint_3d = {
                'x_2d': pixel_x,
                'y_2d': pixel_y,
                'depth': depth,
                'x_3d': pixel_x if depth is not None else None,
                'y_3d': pixel_y if depth is not None else None,
                'z_3d': depth,
                'confidence': confidence,
                'name': keypoint_name
            }
            
            keypoints_3d.append(keypoint_3d)
        
        return keypoints_3d
    
    def format_3d_label(self, keypoint_3d):
        """Format label for 3D keypoint display"""
        name = keypoint_3d['name']
        confidence = keypoint_3d['confidence']
        depth = keypoint_3d['depth']
        
        if SHOW_3D_COORDINATES and depth is not None:
            return f"{name}: ({keypoint_3d['x_2d']}, {keypoint_3d['y_2d']}, {depth:.3f})"
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
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                
                # Position label above keypoint
                label_x = x_2d - label_size[0] // 2
                label_y = y_2d - KEYPOINT_RADIUS - 5
                
                # Ensure label is within image bounds
                if label_y < label_size[1]:
                    label_y = y_2d + KEYPOINT_RADIUS + label_size[1] + 5
                
                # Draw label background
                cv2.rectangle(frame, 
                            (label_x - 2, label_y - label_size[1] - 2),
                            (label_x + label_size[0] + 2, label_y + 2),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (label_x, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
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
        
        return frame
    
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
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                
                # Position label above keypoint
                label_x = x_2d - label_size[0] // 2
                label_y = y_2d - KEYPOINT_RADIUS - 5
                
                # Ensure label is within image bounds
                if label_y < label_size[1]:
                    label_y = y_2d + KEYPOINT_RADIUS + label_size[1] + 5
                
                # Draw label background
                cv2.rectangle(combined_frame, 
                            (label_x - 2, label_y - label_size[1] - 2),
                            (label_x + label_size[0] + 2, label_y + 2),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(combined_frame, label, (label_x, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
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
        
        # Add a title overlay
        overlay = np.zeros_like(combined_frame)
        cv2.rectangle(overlay, (10, 10), (350, 50), (0, 0, 0), -1)
        cv2.putText(overlay, "3D Keypoints + Depth Map", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blend the overlay
        alpha = 0.8
        combined_frame = cv2.addWeighted(combined_frame, alpha, overlay, 1-alpha, 0)
        
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
                            if keypoints_2d:
                                keypoints_3d = self.combine_2d_and_depth(keypoints_2d, depth_map, width, height)
                                keypoints_3d_all.extend(keypoints_3d)
                    
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
                    if keypoints_2d:
                        keypoints_3d = processor.combine_2d_and_depth(keypoints_2d, depth_map, frame.shape[1], frame.shape[0])
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