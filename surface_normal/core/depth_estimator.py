"""
Depth estimation module using Hugging Face transformers.
Handles depth estimation from monocular images for surface normal reconstruction.
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import torch
import tempfile
import os


class DepthEstimatorPipeline:
    """Handles depth estimation using Hugging Face Pipeline"""
    
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf", enable_logging=True):
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
            
            self.enable_logging = enable_logging
            
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
                # Assume depth range of 0.1m to 10m (typical for indoor/table scenarios)
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
            
            # Log depth data if logging is enabled
            if self.enable_logging:
                try:
                    print("DEBUG: Logging depth map data...")
                    self._log_depth_data(depth_array, image_path)
                except Exception as logging_error:
                    print(f"⚠️ Warning: Error logging depth map: {logging_error}")
            
            # Return processed depth array
            return depth_array
            
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return None
    
    def _log_depth_data(self, depth_array, image_path):
        """Log depth data to file for analysis"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/depth_log_{timestamp}.txt"
            
            with open(log_file, 'w') as f:
                f.write(f"Depth Estimation Log - {timestamp}\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Source image: {image_path}\n")
                f.write(f"Depth array shape: {depth_array.shape}\n")
                f.write(f"Depth array dtype: {depth_array.dtype}\n")
                f.write(f"Depth range: [{depth_array.min():.3f}, {depth_array.max():.3f}] meters\n")
                f.write(f"Depth mean: {depth_array.mean():.3f} meters\n")
                f.write(f"Depth std: {depth_array.std():.3f} meters\n")
                f.write(f"Unique values: {len(np.unique(depth_array))}\n\n")
                
                # Sample depth values
                h, w = depth_array.shape
                sample_points = [
                    (w//4, h//4, "top-left"),
                    (w//2, h//2, "center"),
                    (3*w//4, h//4, "top-right"),
                    (w//4, 3*h//4, "bottom-left"),
                    (3*w//4, 3*h//4, "bottom-right")
                ]
                
                f.write("Sample depth values:\n")
                for x, y, label in sample_points:
                    depth_val = depth_array[y, x]
                    f.write(f"  {label} ({x}, {y}): {depth_val:.3f}m\n")
                
            print(f"DEBUG: Depth log saved to: {log_file}")
            
        except Exception as e:
            print(f"DEBUG: Failed to log depth data: {e}")
    
    def get_depth_at_point(self, depth_map, x, y):
        """Get depth value at specific pixel coordinates"""
        if depth_map is None:
            return None
        
        height, width = depth_map.shape[:2]
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        return depth_map[y, x]
    
    def create_depth_visualization(self, depth_map, color_map=cv2.COLORMAP_VIRIDIS):
        """Create colored depth visualization"""
        if depth_map is None:
            return None
        
        # For visualization, normalize depth map to 0-255 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max > depth_min:
            # Normalize to 0-255 range for visualization
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            # If all values are the same, use a default range
            depth_normalized = np.full_like(depth_map, 128, dtype=np.uint8)
        
        # Apply color map
        depth_colored = cv2.applyColorMap(depth_normalized, color_map)
        
        return depth_colored
