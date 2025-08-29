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
from core.surface_normal_estimator import GridBasedSurfaceNormalEstimator
from utils.metadata_extractor import CameraMetadataExtractor

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

# Surface Normal Parameters
GRID_SIZE = 40             # Grid size for surface normal arrows (pixels)
ARROW_LENGTH = 60.0        # Length of normal arrows (pixels)
ARROW_THICKNESS = 3        # Thickness of arrow lines
ARROW_COLOR = (0, 255, 0)  # Green arrows (BGR format)

# Camera Intrinsic Parameters (these should ideally be detected from the image metadata)
# Default values based on typical camera setups
CAMERA_FX = 1512.0         # Focal length X (pixels) - adjust based on your camera
CAMERA_FY = 1512.0         # Focal length Y (pixels) - adjust based on your camera  
CAMERA_CX = 1080.0         # Principal point X (pixels) - typically image_width/2
CAMERA_CY = 607.0          # Principal point Y (pixels) - typically image_height/2

# Processing mode selection
USE_3D_BACKPROJECTION = True  # Set to True for proper 3D back-projection, False for simple gradient
EXTRACT_CAMERA_INTRINSICS = True  # Set to True to extract intrinsics from image/video metadata

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
            
            # Initialize metadata extractor
            print("Initializing camera metadata extractor...")
            self.metadata_extractor = CameraMetadataExtractor()
            print("✅ Camera metadata extractor initialized")
            
            # Initialize surface normal estimator with default parameters
            # (will be updated with extracted intrinsics per image/video)
            print("Initializing surface normal estimator...")
            self.surface_normal_estimator = GridBasedSurfaceNormalEstimator(
                fx=CAMERA_FX,
                fy=CAMERA_FY,
                cx=CAMERA_CX,
                cy=CAMERA_CY,
                grid_size=GRID_SIZE,
                arrow_length=ARROW_LENGTH,
                arrow_thickness=ARROW_THICKNESS
            )
            print("✅ Surface normal estimator initialized")
            
            # Store extracted intrinsics for video processing (same intrinsics for all frames)
            self.cached_video_intrinsics = None
            
        except Exception as e:
            print(f"❌ Error initializing processor: {e}")
            raise e
    
    def process_image(self, image_path, use_cached_intrinsics=False):
        """Process a single image for depth estimation and surface normals"""
        try:
            print(f"Processing image: {image_path}")
            
            # Extract camera intrinsics from image metadata if enabled
            if EXTRACT_CAMERA_INTRINSICS and not use_cached_intrinsics:
                print("🔍 Extracting camera intrinsics from image metadata...")
                intrinsics = self.metadata_extractor.extract_from_image(image_path)
                
                # Log extracted intrinsics
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"logs/extracted_intrinsics_{timestamp}.txt"
                self.metadata_extractor.log_intrinsics(intrinsics, log_file)
                
                # Update surface normal estimator with extracted intrinsics
                self.surface_normal_estimator.fx = intrinsics.fx
                self.surface_normal_estimator.fy = intrinsics.fy
                self.surface_normal_estimator.cx = intrinsics.cx
                self.surface_normal_estimator.cy = intrinsics.cy
                
                print(f"✅ Updated camera intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")
                print(f"   Source: {intrinsics.source}, Confidence: {intrinsics.confidence}")
                print(f"   Camera: {intrinsics.camera_model}")
                
            elif use_cached_intrinsics and self.cached_video_intrinsics:
                # Use cached intrinsics from video
                intrinsics = self.cached_video_intrinsics
                self.surface_normal_estimator.fx = intrinsics.fx
                self.surface_normal_estimator.fy = intrinsics.fy
                self.surface_normal_estimator.cx = intrinsics.cx
                self.surface_normal_estimator.cy = intrinsics.cy
            else:
                print("📐 Using default camera intrinsics")
            
            # Get depth estimation
            depth_map = self.depth_estimator.estimate_depth(image_path)
            if depth_map is None:
                return None, None, "Error: Depth estimation failed"
            
            # Load original image for size reference
            original_image = cv2.imread(image_path)
            if original_image is None:
                return None, None, "Error: Could not load image"
            
            h, w = original_image.shape[:2]
            print(f"Original image dimensions: {w}x{h}")
            
            # Create depth visualization
            depth_visualization = self.depth_estimator.create_depth_visualization(depth_map, DEPTH_COLOR_MAP)
            depth_visualization = cv2.resize(depth_visualization, (w, h))
            
            # Estimate surface normals using either 3D back-projection or simple gradient method
            print("Computing surface normals...")
            if USE_3D_BACKPROJECTION:
                print("Using 3D back-projection with camera intrinsics...")
                normal_map, point_cloud_3d, grid_data = self.surface_normal_estimator.estimate_grid_normals_3d(depth_map)
                print(f"✅ Generated 3D point cloud with shape: {point_cloud_3d.shape}")
            else:
                print("Using simple gradient method...")
                normal_map, grid_data = self.surface_normal_estimator.estimate_grid_normals_simple(depth_map)
            
            # Create surface normal arrow visualization using enhanced method
            surface_normal_vis = self.surface_normal_estimator.create_enhanced_arrow_visualization(
                original_image, grid_data, ARROW_COLOR
            )
            
            print("✅ Depth estimation and surface normal computation completed successfully")
            return depth_visualization, surface_normal_vis, "Processing completed successfully"
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None, f"Error: {str(e)}"
    
    def process_video(self, video_path, progress=gr.Progress()):
        """Process video with depth estimation and surface normals"""
        try:
            print(f"Processing video: {video_path}")
            
            # Extract camera intrinsics from video metadata if enabled
            if EXTRACT_CAMERA_INTRINSICS:
                print("🔍 Extracting camera intrinsics from video metadata...")
                self.cached_video_intrinsics = self.metadata_extractor.extract_from_video(video_path)
                
                # Log extracted intrinsics
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"logs/extracted_video_intrinsics_{timestamp}.txt"
                self.metadata_extractor.log_intrinsics(self.cached_video_intrinsics, log_file)
                
                print(f"✅ Extracted video intrinsics: fx={self.cached_video_intrinsics.fx:.1f}, fy={self.cached_video_intrinsics.fy:.1f}")
                print(f"   Source: {self.cached_video_intrinsics.source}, Confidence: {self.cached_video_intrinsics.confidence}")
                print(f"   Camera: {self.cached_video_intrinsics.camera_model}")
                print("   📝 These intrinsics will be used for all frames in this video")
            else:
                print("📐 Using default camera intrinsics for video processing")
                self.cached_video_intrinsics = None
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, None, f"Error: Could not open video file: {video_path}"
            
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
            
            # Create temporary output files
            temp_depth_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_depth_output.close()
            temp_normal_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_normal_output.close()
            
            # Setup video writers
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_depth = cv2.VideoWriter(temp_depth_output.name, fourcc, TARGET_FPS, (width, height))
            out_normal = cv2.VideoWriter(temp_normal_output.name, fourcc, TARGET_FPS, (width, height))
            
            if not out_depth.isOpened() or not out_normal.isOpened():
                return None, None, "Error: Could not create output video writers"
            
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
                    # Process frame using cached video intrinsics
                    depth_vis, normal_vis, status = self.process_image(temp_frame_path.name, use_cached_intrinsics=True)
                    
                    # Clean up temporary frame file
                    os.unlink(temp_frame_path.name)
                    
                    if depth_vis is not None and normal_vis is not None:
                        out_depth.write(depth_vis)
                        out_normal.write(normal_vis)
                    else:
                        # Write original frame if processing failed
                        out_depth.write(frame)
                        out_normal.write(frame)
                    
                except Exception as e:
                    print(f"Error processing frame {processed_count}: {e}")
                    if os.path.exists(temp_frame_path.name):
                        os.unlink(temp_frame_path.name)
                    # Write original frame on error
                    out_depth.write(frame)
                    out_normal.write(frame)
                
                processed_count += 1
                frame_count += 1
            
            # Clean up
            cap.release()
            out_depth.release()
            out_normal.release()
            
            return (temp_depth_output.name, temp_normal_output.name, 
                    f"Successfully processed {processed_count} frames at {TARGET_FPS} FPS")
            
        except Exception as e:
            print(f"Error in process_video: {e}")
            return None, None, f"Error processing video: {str(e)}"

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
    """Preview depth estimation and surface normals on a single image"""
    try:
        if not image_file:
            return None, None, "Please upload an image file"
        
        # Create processor
        processor, status = create_processor()
        if processor is None:
            return None, None, status
        
        # Process image
        image_path = image_file.name if hasattr(image_file, 'name') else image_file
        depth_vis, normal_vis, process_status = processor.process_image(image_path)
        
        if depth_vis is not None and normal_vis is not None:
            # Save visualizations
            depth_output = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(depth_output.name, depth_vis)
            depth_output.close()
            
            normal_output = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(normal_output.name, normal_vis)
            normal_output.close()
            
            return depth_output.name, normal_output.name, process_status
        else:
            return None, None, process_status
            
    except Exception as e:
        print(f"Error in preview: {e}")
        return None, None, f"Error: {str(e)}"

def process_video_with_depth(depth_model_name, video_file):
    """Process video with depth estimation and surface normals"""
    try:
        if not video_file:
            return None, None, "Please upload a video file"
        
        # Create processor
        processor, status = create_processor()
        if processor is None:
            return None, None, status
        
        # Process video
        video_path = video_file.name if hasattr(video_file, 'name') else video_file
        depth_output, normal_output, process_status = processor.process_video(video_path)
        
        return depth_output, normal_output, process_status
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return None, None, f"Error: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Create Gradio interface
with gr.Blocks(title="Depth Estimation & Surface Normals", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Depth Estimation & Surface Normal Visualization")
    gr.Markdown("Upload an image or video to estimate depth and compute surface normals with arrow visualization using Depth-Anything-V2 models")
    
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
            - Grid Size: {GRID_SIZE}px
            - Arrow Length: {ARROW_LENGTH}px
            - 3D Back-Projection: {'Enabled' if USE_3D_BACKPROJECTION else 'Disabled'}
            - Metadata Extraction: {'Enabled' if EXTRACT_CAMERA_INTRINSICS else 'Disabled'}
            
            **Camera Intrinsics:**
            - Auto-Extract from EXIF/Metadata: {'✅ Yes' if EXTRACT_CAMERA_INTRINSICS else '❌ No'}
            - Fallback: fx={CAMERA_FX}, fy={CAMERA_FY}, cx={CAMERA_CX}, cy={CAMERA_CY}
            - Camera Type: Monocular (Depth-Anything-V2)
            - Supports: iPhone, DSLR, generic cameras
            
            **Workflow:**
            1. Upload image/video (EXIF metadata will be automatically extracted)
            2. Click "Preview Image" to see depth & surface normals
            3. Click "Process Video" for full video processing (uses same intrinsics for all frames)
            4. Download processed results with depth & surface normal visualization
            5. Check logs/ directory for extracted camera intrinsics and 3D analysis
            
            **Supported Metadata:**
            - Image: EXIF data (focal length, camera model, sensor size)
            - Video: Metadata tags, first frame EXIF analysis
            - Fallback: Intelligent estimation based on resolution and camera type
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Output")
            
            # Image preview outputs - side by side
            with gr.Row():
                preview_depth = gr.Image(
                    label="Preview: Depth Map",
                    type="filepath"
                )
                preview_normals = gr.Image(
                    label="Preview: Surface Normals",
                    type="filepath"
                )
            
            # Video processing outputs
            with gr.Row():
                output_depth_video = gr.Video(
                    label="Processed Video: Depth Maps",
                    format="mp4"
                )
                output_normals_video = gr.Video(
                    label="Processed Video: Surface Normals",
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
        outputs=[preview_depth, preview_normals, status_text]
    )
    
    process_video_btn.click(
        fn=process_video_with_depth,
        inputs=[depth_model_name, video_input],
        outputs=[output_depth_video, output_normals_video, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7864)