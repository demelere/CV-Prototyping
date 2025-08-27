#!/usr/bin/env python3
"""
Depth Estimation App
A standalone Gradio application for depth estimation.
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time
from datetime import datetime

from core.depth_estimator import DepthEstimatorPipeline

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Depth Model Configuration
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"  # Use the locally downloaded model
# Alternative models:
# "depth-anything/Depth-Anything-V2-Base-hf"  # Medium size
# "depth-anything/Depth-Anything-V2-Large-hf"  # Large, most accurate

# Visualization Parameters
DEPTH_COLOR_MAP = cv2.COLORMAP_VIRIDIS  # Color map for depth visualization

# Processing Parameters
TARGET_FPS = 15            # Target FPS for video processing
SKIP_FRAMES = True         # Whether to skip frames to match target FPS

# ============================================================================
# CORE PROCESSING CLASSES
# ============================================================================

class DepthProcessor:
    def __init__(self, depth_model_name=DEPTH_MODEL_NAME):
        """Initialize the depth processor"""
        try:
            print("Initializing Depth Processor...")
            
            # Initialize depth estimator
            print("Loading depth estimation pipeline...")
            self.depth_estimator = DepthEstimatorPipeline(depth_model_name, enable_logging=True)
            print("✅ Depth estimator pipeline initialized")
            
        except Exception as e:
            print(f"❌ Error initializing processor: {e}")
            raise e
    
    def process_image(self, image_path):
        """Process a single image for depth estimation"""
        try:
            print(f"Processing image: {image_path}")
            
            # Get depth estimation
            depth_map = self.depth_estimator.estimate_depth(image_path)
            if depth_map is None:
                return None, "Error: Depth estimation failed"
            
            # Load original image for size reference
            original_image = cv2.imread(image_path)
            if original_image is None:
                return None, "Error: Could not load image"
            
            h, w = original_image.shape[:2]
            print(f"Original image dimensions: {w}x{h}")
            
            # Create depth visualization
            depth_visualization = self.depth_estimator.create_depth_visualization(depth_map, DEPTH_COLOR_MAP)
            depth_visualization = cv2.resize(depth_visualization, (w, h))
            
            print("✅ Depth estimation completed successfully")
            return depth_visualization, "Processing completed successfully"
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, f"Error: {str(e)}"
    
    def process_video(self, video_path, progress=gr.Progress()):
        """Process video with depth estimation"""
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
            
            print(f"Video: {width}x{height}, {original_fps} fps, {total_frames} frames")
            
            # Calculate frame skip interval
            if SKIP_FRAMES and original_fps > TARGET_FPS:
                skip_interval = max(1, original_fps // TARGET_FPS)
                processed_frames = total_frames // skip_interval
            else:
                skip_interval = 1
                processed_frames = total_frames
            
            # Create temporary output file
            temp_depth_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_depth_output.close()
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_depth = cv2.VideoWriter(temp_depth_output.name, fourcc, TARGET_FPS, (width, height))
            
            if not out_depth.isOpened():
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
                
                # Save frame temporarily for processing
                temp_frame_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(temp_frame_path.name, frame)
                temp_frame_path.close()
                
                try:
                    # Process frame
                    depth_vis, status = self.process_image(temp_frame_path.name)
                    
                    # Clean up temporary frame file
                    os.unlink(temp_frame_path.name)
                    
                    if depth_vis is not None:
                        out_depth.write(depth_vis)
                    else:
                        # Write original frame if processing failed
                        out_depth.write(frame)
                    
                except Exception as e:
                    print(f"Error processing frame {processed_count}: {e}")
                    if os.path.exists(temp_frame_path.name):
                        os.unlink(temp_frame_path.name)
                    # Write original frame on error
                    out_depth.write(frame)
                
                processed_count += 1
                frame_count += 1
            
            # Clean up
            cap.release()
            out_depth.release()
            
            return temp_depth_output.name, f"Successfully processed {processed_count} frames at {TARGET_FPS} FPS"
            
        except Exception as e:
            print(f"Error in process_video: {e}")
            return None, f"Error processing video: {str(e)}"

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def create_processor():
    """Create and return a processor instance"""
    try:
        processor = DepthProcessor(DEPTH_MODEL_NAME)
        return processor, "Processor initialized successfully!"
    except Exception as e:
        print(f"❌ Error creating processor: {e}")
        return None, f"Error initializing processor: {str(e)}"

def preview_image(depth_model_name, image_file):
    """Preview depth estimation on a single image"""
    try:
        if not image_file:
            return None, "Please upload an image file"
        
        # Create processor
        processor, status = create_processor()
        if processor is None:
            return None, status
        
        # Process image
        image_path = image_file.name if hasattr(image_file, 'name') else image_file
        depth_vis, process_status = processor.process_image(image_path)
        
        if depth_vis is not None:
            # Save visualization
            depth_output = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(depth_output.name, depth_vis)
            depth_output.close()
            
            return depth_output.name, process_status
        else:
            return None, process_status
            
    except Exception as e:
        print(f"Error in preview: {e}")
        return None, f"Error: {str(e)}"

def process_video_with_depth(depth_model_name, video_file):
    """Process video with depth estimation"""
    try:
        if not video_file:
            return None, "Please upload a video file"
        
        # Create processor
        processor, status = create_processor()
        if processor is None:
            return None, status
        
        # Process video
        video_path = video_file.name if hasattr(video_file, 'name') else video_file
        depth_output, process_status = processor.process_video(video_path)
        
        return depth_output, process_status
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return None, f"Error: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Create Gradio interface
with gr.Blocks(title="Depth Estimation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Depth Estimation from Monocular Images")
    gr.Markdown("Upload an image or video to estimate depth using Depth-Anything-V2 models")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Configuration")
            
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
            
            gr.Markdown("### 📹 Input")
            
            # File input tabs
            with gr.Tabs():
                with gr.TabItem("Image"):
                    image_input = gr.Image(
                        label="Upload Image",
                        type="filepath"
                    )
                
                with gr.TabItem("Video"):
                    video_input = gr.Video(
                        label="Upload Video",
                        format="mp4"
                    )
            
            with gr.Row():
                preview_btn = gr.Button(
                    "👁️ Preview Image",
                    variant="secondary",
                    size="sm"
                )
                
                process_video_btn = gr.Button(
                    "🚀 Process Video",
                    variant="primary",
                    size="lg"
                )
            
            gr.Markdown(f"""
            **Current Settings:**
            - Target FPS: {TARGET_FPS}
            - Depth Color Map: {DEPTH_COLOR_MAP}
            
            **Workflow:**
            1. Upload image/video
            2. Click "Preview Image" to test on single image
            3. Click "Process Video" for full video processing
            4. Download processed results with depth visualization
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Output")
            
            # Image preview output
            preview_depth = gr.Image(
                label="Preview: Depth Map",
                type="filepath"
            )
            
            # Video processing output
            output_depth_video = gr.Video(
                label="Processed Video: Depth Maps",
                format="mp4"
            )
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False
            )
    
    # Connect the interface
    preview_btn.click(
        fn=preview_image,
        inputs=[depth_model_name, image_input],
        outputs=[preview_depth, status_text]
    )
    
    process_video_btn.click(
        fn=process_video_with_depth,
        inputs=[depth_model_name, video_input],
        outputs=[output_depth_video, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7864)