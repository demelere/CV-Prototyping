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
CONFIDENCE_THRESHOLD = 70  # Only show detections with this confidence % or higher
OVERLAP_THRESHOLD = 10     # Overlap threshold for non-maximum suppression

# Processing Parameters
TARGET_FPS = 20            # Target FPS for processing (lower = faster processing, better motion tracking)
SKIP_FRAMES = True         # Whether to skip frames to match target FPS

# Visual Display Parameters
BOUNDING_BOX_THICKNESS = 2         # Thickness of bounding box lines
LABEL_FONT_SCALE = 0.5            # Font size for labels
LABEL_TEXT_COLOR = (255, 255, 255) # White text color (BGR)

# Class-specific colors (BGR format)
CLASS_COLORS = {
    'pool': (0, 0, 255),      # Red
    'electrode': (0, 165, 255), # Orange
    'arc': (255, 255, 0),     # Teal
    'rod': (255, 0, 255),     # Purple
    'default': (0, 255, 0)    # Green (for any other classes)
}

# Label Display Options
SHOW_CONFIDENCE = True             # Show confidence score in label
SHOW_CLASS_NAME = True             # Show class name in label
CONFIDENCE_DECIMAL_PLACES = 2      # Number of decimal places for confidence
LABEL_FORMAT = "class_confidence"  # Options: "class_confidence", "class_only", "confidence_only", "percentage"

# ============================================================================
# END CONFIGURATION PARAMETERS
# ============================================================================

class VideoProcessor:
    def __init__(self, api_key, workspace_name, project_name, version_number):
        """Initialize the video processor with Roboflow model"""
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace(workspace_name).project(project_name)
        self.model = self.project.version(version_number).model
        
    def format_label(self, class_name, confidence):
        """Format the label based on configuration"""
        if LABEL_FORMAT == "class_confidence":
            if SHOW_CLASS_NAME and SHOW_CONFIDENCE:
                return f"{class_name}: {confidence:.{CONFIDENCE_DECIMAL_PLACES}f}"
            elif SHOW_CLASS_NAME:
                return class_name
            elif SHOW_CONFIDENCE:
                return f"{confidence:.{CONFIDENCE_DECIMAL_PLACES}f}"
        elif LABEL_FORMAT == "class_only":
            return class_name
        elif LABEL_FORMAT == "confidence_only":
            return f"{confidence:.{CONFIDENCE_DECIMAL_PLACES}f}"
        elif LABEL_FORMAT == "percentage":
            return f"{class_name}: {confidence:.1%}"
        
        return f"{class_name}: {confidence:.{CONFIDENCE_DECIMAL_PLACES}f}"
        
    def draw_detections(self, frame, predictions):
        """Draw bounding boxes and labels on the frame"""
        if not predictions:
            return frame
            
        for prediction in predictions:
            # Extract prediction data
            x = prediction.get('x', 0)
            y = prediction.get('y', 0)
            width = prediction.get('width', 0)
            height = prediction.get('height', 0)
            confidence = prediction.get('confidence', 0)
            class_name = prediction.get('class', 'Unknown')
            
            # Debug: Print raw coordinates for first few detections
            if len([p for p in predictions if p == prediction]) <= 3:  # Only for first 3 detections
                print(f"Raw prediction: class={class_name}, x={x}, y={y}, w={width}, h={height}, conf={confidence}")
            
            # Convert to pixel coordinates
            img_height, img_width = frame.shape[:2]
            
            # Roboflow returns coordinates as percentages (0-1) or absolute pixels
            # Let's handle both cases
            if x <= 1 and y <= 1 and width <= 1 and height <= 1:
                # Coordinates are percentages (0-1)
                x1 = int((x - width/2) * img_width)
                y1 = int((y - height/2) * img_height)
                x2 = int((x + width/2) * img_width)
                y2 = int((y + height/2) * img_height)
            else:
                # Coordinates are already in pixels
                x1 = int(x - width/2)
                y1 = int(y - height/2)
                x2 = int(x + width/2)
                y2 = int(y + height/2)
            
            # Debug: Print converted coordinates
            if len([p for p in predictions if p == prediction]) <= 3:
                print(f"Converted coordinates: ({x1}, {y1}) to ({x2}, {y2}) for image size {img_width}x{img_height}")
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Get class-specific color
            color = CLASS_COLORS.get(class_name.lower(), CLASS_COLORS['default'])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOUNDING_BOX_THICKNESS)
            
            # Draw label
            label = self.format_label(class_name, confidence)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_TEXT_COLOR, 2)
            
        return frame
    
    def process_video(self, video_path, progress=gr.Progress()):
        """Process video with object detection"""
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
            print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}%, overlap: {OVERLAP_THRESHOLD}%")
            
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
                    # Get prediction from Roboflow with configurable parameters
                    prediction = self.model.predict(
                        temp_frame_path.name, 
                        confidence=CONFIDENCE_THRESHOLD, 
                        overlap=OVERLAP_THRESHOLD
                    ).json()
                    
                    # Debug: Print prediction info
                    if processed_count % 10 == 0:  # Print every 10 processed frames
                        print(f"Processed frame {processed_count}: Got prediction with {len(prediction.get('predictions', []))} detections")
                        if prediction.get('predictions'):
                            for i, pred in enumerate(prediction['predictions'][:3]):  # Show first 3
                                print(f"  Detection {i}: {pred.get('class', 'Unknown')} at ({pred.get('x', 0):.3f}, {pred.get('y', 0):.3f}) with confidence {pred.get('confidence', 0):.3f}")
                    
                    # Draw detections on frame
                    if 'predictions' in prediction:
                        frame = self.draw_detections(frame, prediction['predictions'])
                    
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
    """Create and return a video processor instance"""
    try:
        processor = VideoProcessor(api_key, workspace_name, project_name, int(version_number))
        return processor, "Model loaded successfully!"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def process_video_with_model(api_key, workspace_name, project_name, version_number, video_file):
    """Main function to process video with object detection"""
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
            # Get prediction from Roboflow
            prediction = processor.model.predict(
                temp_frame_path.name, 
                confidence=CONFIDENCE_THRESHOLD, 
                overlap=OVERLAP_THRESHOLD
            ).json()
            
            # Draw detections on frame
            if 'predictions' in prediction:
                frame = processor.draw_detections(frame, prediction['predictions'])
                print(f"Preview: Found {len(prediction['predictions'])} detections")
                for i, pred in enumerate(prediction['predictions']):
                    print(f"  Detection {i}: {pred.get('class', 'Unknown')} at ({pred.get('x', 0):.3f}, {pred.get('y', 0):.3f}) with confidence {pred.get('confidence', 0):.3f}")
            else:
                print("Preview: No detections found")
            
            # Save processed frame
            output_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(output_path.name, frame)
            output_path.close()
            
            # Clean up temporary frame file
            os.unlink(temp_frame_path.name)
            
            return output_path.name, f"Preview: Found {len(prediction.get('predictions', []))} detections on first frame"
            
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
with gr.Blocks(title="RF-DETR Video Object Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 RF-DETR Video Object Detection")
    gr.Markdown("Upload a video and process it with your trained RF-DETR model from Roboflow")
    
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
                placeholder="e.g., object-detection-abc123"
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
            - Overlap Threshold: {OVERLAP_THRESHOLD}%
            - Target FPS: {TARGET_FPS} (faster processing, better motion tracking)
            - Label Format: {LABEL_FORMAT}
            
            **Class Colors:**
            - Pool: Red
            - Electrode: Orange  
            - Arc: Teal
            - Rod: Purple
            - Other classes: Green
            
            **Workflow:**
            1. Upload video and configure settings
            2. Click "Preview First Frame" to test inference
            3. If preview looks good, click "Process Video"
            4. Download the processed video with detections
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Output")
            
            preview_image = gr.Image(
                label="Preview: First Frame with Detections",
                type="filepath"
            )
            
            output_video = gr.Video(
                label="Processed Video with Detections",
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
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860) 