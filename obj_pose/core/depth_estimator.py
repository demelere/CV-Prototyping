"""
Depth estimation module using Hugging Face transformers.
Handles depth estimation from monocular images for 3D pose reconstruction.
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import torch


class DepthEstimatorPipeline:
    """Handles depth estimation using Hugging Face Pipeline"""
    
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
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
            print(f"❌ Error in depth estimation: {e}")
            return None
    
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
        
        # Apply color map
        depth_colored = cv2.applyColorMap(depth_map, color_map)
        
        return depth_colored
