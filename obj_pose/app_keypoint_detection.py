import gradio as gr
import cv2
import numpy as np
from roboflow import Roboflow
import tempfile
import os
from PIL import Image
import json

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

# Label Display Options
SHOW_CONFIDENCE = True             # Show confidence score in label
SHOW_KEYPOINT_NAME = True          # Show keypoint name in label
CONFIDENCE_DECIMAL_PLACES = 2      # Number of decimal places for confidence
LABEL_FORMAT = "keypoint_confidence"  # Options: "keypoint_confidence", "keypoint_only", "confidence_only", "percentage"

# Keypoint Connection Configuration
# Define connections between keypoints (example for pose estimation)
# Format: [(keypoint1_index, keypoint2_index), ...]
KEYPOINT_CONNECTIONS = [
    # Example connections for pose estimation (adjust based on your model)
    # (0, 1), (1, 2), (2, 3),  # Head to neck to shoulders
    # (3, 4), (4, 5), (5, 6),  # Arms
    # (7, 8), (8, 9), (9, 10), # Legs
]

# ============================================================================
# END CONFIGURATION PARAMETERS
# ============================================================================

class KeypointProcessor:
    def __init__(self, api_key, workspace_name, project_name, version_number):
        """Initialize the keypoint processor with Roboflow model"""
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
            print(f"❌ Error in KeypointProcessor.__init__: {e}")
            self.model = None
            raise e
        
    def format_label(self, keypoint_name, confidence):
        """Format the label based on configuration"""
        if LABEL_FORMAT == "keypoint_confidence":
            if SHOW_KEYPOINT_NAME and SHOW_CONFIDENCE:
                return f"{keypoint_name}: {confidence:.{CONFIDENCE_DECIMAL_PLACES}f}"
            elif SHOW_KEYPOINT_NAME:
                return keypoint_name
            elif SHOW_CONFIDENCE:
                return f"{confidence:.{CONFIDENCE_DECIMAL_PLACES}f}"
        elif LABEL_FORMAT == "keypoint_only":
            return keypoint_name
        elif LABEL_FORMAT == "confidence_only":
            return f"{confidence:.{CONFIDENCE_DECIMAL_PLACES}f}"
        elif LABEL_FORMAT == "percentage":
            return f"{keypoint_name}: {confidence:.1%}"
        
        return f"{keypoint_name}: {confidence:.{CONFIDENCE_DECIMAL_PLACES}f}"
        
    def draw_keypoints(self, frame, predictions):
        """Draw keypoints and connections on the frame"""
        if not predictions:
            print("No predictions to draw")
            return frame
            
        img_height, img_width = frame.shape[:2]
        print(f"Drawing keypoints on frame size: {img_width}x{img_height}")
        print(f"Processing {len(predictions)} predictions")
        
        # Process each prediction (could be multiple people/objects)
        for prediction in predictions:
            keypoints = prediction.get('keypoints', [])
            if not keypoints:
                print("No keypoints in this prediction")
                continue
                
            # Debug: Print keypoint info for first few predictions
            if len([p for p in predictions if p == prediction]) <= 2:
                print(f"Prediction with {len(keypoints)} keypoints")
                print(f"Keypoints data: {keypoints[:2]}")  # Show first 2 keypoints
                
            # Draw keypoints
            for i, keypoint in enumerate(keypoints):
                x = keypoint.get('x', 0)
                y = keypoint.get('y', 0)
                confidence = keypoint.get('confidence', 0)
                keypoint_name = keypoint.get('name', f'kp_{i}')
                
                # Skip low confidence keypoints (handle both percentage and decimal formats)
                confidence_threshold = CONFIDENCE_THRESHOLD / 100.0 if CONFIDENCE_THRESHOLD > 1 else CONFIDENCE_THRESHOLD
                print(f"Keypoint {i} confidence: {confidence:.3f}, threshold: {confidence_threshold:.3f}")
                if confidence < confidence_threshold:
                    print(f"  Skipping keypoint {i} due to low confidence")
                    continue
                
                # Convert coordinates
                # Roboflow returns coordinates as percentages (0-1) or absolute pixels
                print(f"  Original coordinates: x={x}, y={y}")
                if x <= 1 and y <= 1:
                    # Coordinates are percentages (0-1)
                    pixel_x = int(x * img_width)
                    pixel_y = int(y * img_height)
                    print(f"  Converting from percentages to pixels")
                else:
                    # Coordinates are already in pixels
                    pixel_x = int(x)
                    pixel_y = int(y)
                    print(f"  Using coordinates as pixels")
                
                # Ensure coordinates are within image bounds
                pixel_x = max(0, min(img_width - 1, pixel_x))
                pixel_y = max(0, min(img_height - 1, pixel_y))
                
                # Debug: Print converted coordinates
                if len([p for p in predictions if p == prediction]) <= 2 and i < 3:
                    print(f"Keypoint {i}: {keypoint_name} at ({pixel_x}, {pixel_y}) with confidence {confidence:.3f}")
                    print(f"  Original coords: ({x}, {y})")
                    print(f"  Image size: {img_width}x{img_height}")
                    print(f"  Bounds check: x={pixel_x} (0-{img_width-1}), y={pixel_y} (0-{img_height-1})")
                
                # Draw keypoint circle
                print(f"Drawing keypoint at ({pixel_x}, {pixel_y}) with radius {KEYPOINT_RADIUS}")
                cv2.circle(frame, (pixel_x, pixel_y), KEYPOINT_RADIUS, KEYPOINT_COLOR, KEYPOINT_THICKNESS)
                
                # Draw label if enabled
                if SHOW_KEYPOINT_NAME or SHOW_CONFIDENCE:
                    label = self.format_label(keypoint_name, confidence)
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    
                    # Position label above keypoint
                    label_x = pixel_x - label_size[0] // 2
                    label_y = pixel_y - KEYPOINT_RADIUS - 5
                    
                    # Ensure label is within image bounds
                    if label_y < label_size[1]:
                        label_y = pixel_y + KEYPOINT_RADIUS + label_size[1] + 5
                    
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
                    if len(keypoints) > max(connection):
                        kp1 = keypoints[connection[0]]
                        kp2 = keypoints[connection[1]]
                        
                        # Check confidence for both keypoints (handle both percentage and decimal formats)
                        confidence_threshold = CONFIDENCE_THRESHOLD / 100.0 if CONFIDENCE_THRESHOLD > 1 else CONFIDENCE_THRESHOLD
                        if (kp1.get('confidence', 0) >= confidence_threshold and 
                            kp2.get('confidence', 0) >= confidence_threshold):
                            
                            # Get coordinates for both keypoints
                            x1, y1 = kp1.get('x', 0), kp1.get('y', 0)
                            x2, y2 = kp2.get('x', 0), kp2.get('y', 0)
                            
                            # Convert coordinates
                            if x1 <= 1 and y1 <= 1:
                                pixel_x1 = int(x1 * img_width)
                                pixel_y1 = int(y1 * img_height)
                            else:
                                pixel_x1, pixel_y1 = int(x1), int(y1)
                                
                            if x2 <= 1 and y2 <= 1:
                                pixel_x2 = int(x2 * img_width)
                                pixel_y2 = int(y2 * img_height)
                            else:
                                pixel_x2, pixel_y2 = int(x2), int(y2)
                            
                            # Ensure coordinates are within bounds
                            pixel_x1 = max(0, min(img_width - 1, pixel_x1))
                            pixel_y1 = max(0, min(img_height - 1, pixel_y1))
                            pixel_x2 = max(0, min(img_width - 1, pixel_x2))
                            pixel_y2 = max(0, min(img_height - 1, pixel_y2))
                            
                            # Draw connection line
                            cv2.line(frame, (pixel_x1, pixel_y1), (pixel_x2, pixel_y2), 
                                    CONNECTION_COLOR, CONNECTION_THICKNESS)
            
        return frame
    
    def process_video(self, video_path, progress=gr.Progress()):
        """Process video with keypoint detection"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, f"Error: Could not open video file: {video_path}"
            
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
            
            # Create temporary output file
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_output.close()
            
            # Setup video writer with target FPS
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output.name, fourcc, TARGET_FPS, (width, height))
            
            if not out.isOpened():
                return None, "Error: Could not create output video writer"
            
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
                
                # Save frame temporarily for Roboflow
                temp_frame_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(temp_frame_path.name, frame)
                temp_frame_path.close()
                
                try:
                    # Get prediction from Roboflow (keypoint models may not support confidence parameter)
                    try:
                        prediction = self.model.predict(
                            temp_frame_path.name, 
                            confidence=CONFIDENCE_THRESHOLD
                        ).json()
                    except TypeError:
                        # Fallback for models that don't support confidence parameter
                        prediction = self.model.predict(temp_frame_path.name).json()
                    
                    # Handle nested prediction structure
                    actual_predictions = []
                    if 'predictions' in prediction:
                        for pred in prediction['predictions']:
                            if 'predictions' in pred:
                                # Nested structure: prediction['predictions'][0]['predictions']
                                actual_predictions.extend(pred['predictions'])
                            else:
                                # Direct structure: prediction['predictions']
                                actual_predictions.append(pred)
                    
                    # Debug: Print prediction info
                    if processed_count % 10 == 0:  # Print every 10 processed frames
                        total_keypoints = sum(len(pred.get('keypoints', [])) for pred in actual_predictions)
                        print(f"Processed frame {processed_count}: Got {len(actual_predictions)} predictions with {total_keypoints} total keypoints")
                    
                    # Draw keypoints on frame
                    if actual_predictions:
                        frame = self.draw_keypoints(frame, actual_predictions)
                    
                    # Clean up temporary frame file
                    os.unlink(temp_frame_path.name)
                    
                except Exception as e:
                    print(f"Error processing frame {processed_count}: {e}")
                    # Clean up temporary frame file
                    if os.path.exists(temp_frame_path.name):
                        os.unlink(temp_frame_path.name)
                
                # Write processed frame
                out.write(frame)
                processed_count += 1
                frame_count += 1
            
            # Clean up
            cap.release()
            out.release()
            
            return temp_output.name, f"Successfully processed {processed_count} frames at {TARGET_FPS} FPS"
            
        except Exception as e:
            print(f"Error in process_video: {e}")
            return None, f"Error processing video: {str(e)}"

def create_processor(api_key, workspace_name, project_name, version_number):
    """Create and return a keypoint processor instance"""
    try:
        print(f"Creating processor with:")
        print(f"  Workspace: {workspace_name}")
        print(f"  Project: {project_name}")
        print(f"  Version: {version_number}")
        
        processor = KeypointProcessor(api_key, workspace_name, project_name, int(version_number))
        
        # Check if model was loaded successfully
        if processor.model is None:
            return None, "Error: Model is None - check your project details"
        
        print("✅ Model loaded successfully!")
        return processor, "Model loaded successfully!"
    except Exception as e:
        print(f"❌ Error creating processor: {e}")
        return None, f"Error loading model: {str(e)}"

def process_video_with_model(api_key, workspace_name, project_name, version_number, video_file):
    """Main function to process video with keypoint detection"""
    try:
        if not video_file:
            return None, "Please upload a video file"
        
        if not all([api_key, workspace_name, project_name, version_number]):
            return None, "Please fill in all Roboflow configuration fields"
        
        print(f"Processing video: {video_file}")
        print(f"Video file type: {type(video_file)}")
        
        # Create processor
        processor, status = create_processor(api_key, workspace_name, project_name, version_number)
        if processor is None:
            return None, status
        
        # Additional check for None model
        if processor.model is None:
            return None, "Error: Model failed to load - check your Roboflow configuration"
        
        # Handle video file path (Gradio returns string path directly)
        video_path = video_file if isinstance(video_file, str) else video_file.name
        print(f"Video path: {video_path}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            return None, f"Video file not found: {video_path}"
        
        # Process video
        output_path, process_status = processor.process_video(video_path)
        
        if output_path and os.path.exists(output_path):
            return output_path, process_status
        else:
            return None, f"Error processing video: {process_status}"
            
    except Exception as e:
        print(f"Error in process_video_with_model: {e}")
        return None, f"Error: {str(e)}"

def preview_first_frame(api_key, workspace_name, project_name, version_number, video_file):
    """Preview inference on the first frame of the video"""
    try:
        if not video_file:
            return None, "Please upload a video file"
        
        if not all([api_key, workspace_name, project_name, version_number]):
            return None, "Please fill in all Roboflow configuration fields"
        
        # Create processor
        processor, status = create_processor(api_key, workspace_name, project_name, version_number)
        if processor is None:
            return None, status
        
        # Additional check for None model
        if processor.model is None:
            return None, "Error: Model failed to load - check your Roboflow configuration"
        
        # Handle video file path
        video_path = video_file if isinstance(video_file, str) else video_file.name
        
        # Check if file exists
        if not os.path.exists(video_path):
            return None, f"Video file not found: {video_path}"
        
        # Extract first frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error: Could not open video file"
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, "Error: Could not read first frame"
        
        # Save frame temporarily
        temp_frame_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_frame_path.name, frame)
        temp_frame_path.close()
        
        try:
            # Get prediction from Roboflow (keypoint models may not support confidence parameter)
            try:
                prediction = processor.model.predict(
                    temp_frame_path.name, 
                    confidence=CONFIDENCE_THRESHOLD
                ).json()
            except TypeError:
                # Fallback for models that don't support confidence parameter
                prediction = processor.model.predict(temp_frame_path.name).json()
            
            # Debug: Print raw prediction structure
            print(f"Raw prediction keys: {list(prediction.keys())}")
            if 'predictions' in prediction:
                print(f"Number of predictions: {len(prediction['predictions'])}")
                for i, pred in enumerate(prediction['predictions']):
                    print(f"Prediction {i} keys: {list(pred.keys())}")
                    if 'keypoints' in pred:
                        print(f"  Keypoints: {len(pred['keypoints'])}")
                    else:
                        print(f"  No keypoints found in prediction {i}")
            else:
                print("No 'predictions' key found in response")
            
            # Handle nested prediction structure
            actual_predictions = []
            if 'predictions' in prediction:
                for pred in prediction['predictions']:
                    if 'predictions' in pred:
                        # Nested structure: prediction['predictions'][0]['predictions']
                        actual_predictions.extend(pred['predictions'])
                    else:
                        # Direct structure: prediction['predictions']
                        actual_predictions.append(pred)
            
            print(f"Actual predictions to process: {len(actual_predictions)}")
            
            # Draw keypoints on frame
            if actual_predictions:
                print(f"About to draw {len(actual_predictions)} predictions")
                frame = processor.draw_keypoints(frame, actual_predictions)
                total_keypoints = sum(len(pred.get('keypoints', [])) for pred in actual_predictions)
                print(f"Preview: Found {len(actual_predictions)} predictions with {total_keypoints} total keypoints")
                
                # Print details for first few keypoints
                for i, pred in enumerate(actual_predictions[:2]):  # Show first 2 predictions
                    keypoints = pred.get('keypoints', [])
                    print(f"  Prediction {i}: {len(keypoints)} keypoints")
                    for j, kp in enumerate(keypoints[:3]):  # Show first 3 keypoints
                        print(f"    Keypoint {j}: {kp.get('name', f'kp_{j}')} at ({kp.get('x', 0):.3f}, {kp.get('y', 0):.3f}) with confidence {kp.get('confidence', 0):.3f}")
            else:
                print("No actual predictions to draw")
            
            # Save processed frame
            output_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(output_path.name, frame)
            output_path.close()
            
            # Clean up temporary frame file
            os.unlink(temp_frame_path.name)
            
            total_keypoints = sum(len(pred.get('keypoints', [])) for pred in actual_predictions)
            return output_path.name, f"Preview: Found {len(actual_predictions)} predictions with {total_keypoints} total keypoints on first frame"
            
        except Exception as e:
            # Clean up temporary frame file
            if os.path.exists(temp_frame_path.name):
                os.unlink(temp_frame_path.name)
            print(f"Error in preview: {e}")
            return None, f"Error during preview: {str(e)}"
            
    except Exception as e:
        print(f"Error in preview_first_frame: {e}")
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Keypoint Detection Video Processor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Keypoint Detection Video Processor")
    gr.Markdown("Upload a video and process it with your trained keypoint detection model from Roboflow")
    
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
            
            gr.Markdown("""
            **How to find your project details:**
            1. Go to your Roboflow project
            2. The workspace name is in the URL: `https://app.roboflow.com/{workspace_name}/{project_name}`
            3. Version number is shown in the model versions section
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
            
            **Visual Settings:**
            - Keypoint Color: Green
            - Connection Color: Blue
            - Keypoint Radius: {KEYPOINT_RADIUS}px
            - Connection Thickness: {CONNECTION_THICKNESS}px
            
            **Workflow:**
            1. Upload video and configure settings
            2. Click "Preview First Frame" to test inference
            3. If preview looks good, click "Process Video"
            4. Download the processed video with keypoints
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Output")
            
            preview_image = gr.Image(
                label="Preview: First Frame with Keypoints",
                type="filepath"
            )
            
            output_video = gr.Video(
                label="Processed Video with Keypoints",
                format="mp4"
            )
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False
            )
    
    # Connect the interface
    preview_btn.click(
        fn=preview_first_frame,
        inputs=[api_key, workspace_name, project_name, version_number, video_input],
        outputs=[preview_image, status_text]
    )
    
    process_btn.click(
        fn=process_video_with_model,
        inputs=[api_key, workspace_name, project_name, version_number, video_input],
        outputs=[output_video, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861) 