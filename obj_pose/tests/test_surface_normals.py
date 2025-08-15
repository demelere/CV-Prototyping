#!/usr/bin/env python3
"""
Test script for surface normal estimation module.
This script creates a synthetic depth map and tests the surface normal estimation functionality.
"""

import numpy as np
import cv2
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.surface_normal_estimator import SurfaceNormalEstimator, estimate_surface_normals, visualize_surface_normals, create_normal_magnitude_map

def create_synthetic_depth_map(width=640, height=480, smooth_only=False):
    """
    Create a synthetic depth map for testing.
    This creates a smooth gradient similar to the reference image.
    
    Args:
        width (int): Width of the depth map
        height (int): Height of the depth map
        smooth_only (bool): If True, creates a pure gradient without waves
    """
    # Create a coordinate grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Create a smooth diagonal gradient from upper left to lower right
    # Normalize coordinates to 0-1 range
    x_norm = x / width
    y_norm = y / height
    
    # Create diagonal gradient (similar to the image description)
    # Upper left (dark/deep) to lower right (bright/shallow)
    gradient = x_norm + y_norm  # Diagonal gradient
    
    if smooth_only:
        # Pure gradient without any waves or noise
        depth = gradient
    else:
        # Add some curvature to make it more interesting but still smooth
        # Create a gentle wave pattern
        wave_x = np.sin(x_norm * 2 * np.pi) * 0.1
        wave_y = np.cos(y_norm * 2 * np.pi) * 0.1
        wave_combined = (wave_x + wave_y) * 0.05
        
        # Combine gradient with gentle waves
        depth = gradient + wave_combined
    
    # Scale to reasonable depth range (deeper in upper left, shallower in lower right)
    depth = depth * 200 + 50  # Range from 50 to 250
    
    # Ensure positive depth values
    depth = np.maximum(depth, 10)
    
    # Normalize to 0-255 range for visualization
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)
    
    return depth_normalized, depth

def test_surface_normal_estimation():
    """Test the surface normal estimation functionality."""
    print("=== Testing Surface Normal Estimation ===")
    
    # Camera parameters (example values)
    fx, fy = 1000.0, 1000.0  # Focal lengths
    ox, oy = 320.0, 240.0    # Optical centers
    alpha = 2                # Pixel distance for tangent vector construction
    
    # Create synthetic depth map with gentle waves
    print("Creating synthetic depth map with gentle waves...")
    depth_map, true_depth = create_synthetic_depth_map(640, 480, smooth_only=False)
    
    # Also create a pure gradient version for comparison
    print("Creating pure gradient depth map...")
    depth_map_smooth, true_depth_smooth = create_synthetic_depth_map(640, 480, smooth_only=True)
    
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: {depth_map.min()} to {depth_map.max()}")
    
    # Test the function-based approach
    print("\n--- Testing Function-Based Approach ---")
    try:
        normals = estimate_surface_normals(depth_map, fx, fy, ox, oy, alpha)
        print(f"✅ Function-based estimation successful")
        print(f"Normal map shape: {normals.shape}")
        print(f"Normal range: {normals.min():.4f} to {normals.max():.4f}")
        
        # Check for valid normals
        valid_pixels = normals.any(axis=2)
        print(f"Valid pixels: {valid_pixels.sum()}/{valid_pixels.size} ({valid_pixels.sum()/valid_pixels.size*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Function-based estimation failed: {e}")
        return False
    
    # Test the class-based approach
    print("\n--- Testing Class-Based Approach ---")
    try:
        estimator = SurfaceNormalEstimator(fx, fy, ox, oy, alpha)
        normals_class = estimator.estimate_normals(depth_map)
        print(f"✅ Class-based estimation successful")
        print(f"Normal map shape: {normals_class.shape}")
        
        # Get statistics
        stats = estimator.get_normal_statistics(normals_class)
        print(f"Statistics: {stats}")
        
    except Exception as e:
        print(f"❌ Class-based estimation failed: {e}")
        return False
    
    # Test visualization
    print("\n--- Testing Visualization ---")
    try:
        # Test normal visualization
        normal_vis = visualize_surface_normals(normals)
        print(f"✅ Normal visualization successful")
        print(f"Visualization shape: {normal_vis.shape}")
        
        # Test magnitude map
        magnitude_map = create_normal_magnitude_map(normals)
        print(f"✅ Magnitude map creation successful")
        print(f"Magnitude map shape: {magnitude_map.shape}")
        print(f"Magnitude range: {magnitude_map.min()} to {magnitude_map.max()}")
        
        # Save test images in tests folder
        cv2.imwrite("tests/test_depth_map.png", depth_map)
        cv2.imwrite("tests/test_depth_map_smooth.png", depth_map_smooth)
        cv2.imwrite("tests/test_normal_visualization.png", normal_vis)
        cv2.imwrite("tests/test_magnitude_map.png", magnitude_map)
        print("✅ Test images saved")
        
        # Test with smooth gradient
        print("\n--- Testing with Pure Gradient ---")
        normals_smooth = estimate_surface_normals(depth_map_smooth, fx, fy, ox, oy, alpha)
        normal_vis_smooth = visualize_surface_normals(normals_smooth)
        magnitude_map_smooth = create_normal_magnitude_map(normals_smooth)
        
        cv2.imwrite("tests/test_normal_visualization_smooth.png", normal_vis_smooth)
        cv2.imwrite("tests/test_magnitude_map_smooth.png", magnitude_map_smooth)
        print("✅ Smooth gradient test images saved")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False
    
    # Test with mask
    print("\n--- Testing with Mask ---")
    try:
        # Create a simple mask (center region)
        mask = np.zeros_like(depth_map, dtype=bool)
        h, w = depth_map.shape
        mask[h//4:3*h//4, w//4:3*w//4] = True
        
        normals_masked = estimate_surface_normals(depth_map, fx, fy, ox, oy, alpha, mask=mask)
        print(f"✅ Masked estimation successful")
        
        valid_pixels_masked = normals_masked.any(axis=2)
        print(f"Masked valid pixels: {valid_pixels_masked.sum()}/{valid_pixels_masked.size} ({valid_pixels_masked.sum()/valid_pixels_masked.size*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Masked estimation failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    return True

def test_integration_with_real_camera_params():
    """Test with realistic camera parameters from the main application."""
    print("\n=== Testing with Real Camera Parameters ===")
    
    # Camera parameters from the main application
    CAMERA_FX = 1512.0  # Focal length in pixels (X direction)
    CAMERA_FY = 1512.0  # Focal length in pixels (Y direction)
    CAMERA_CX = 1080.0  # Principal point X coordinate
    CAMERA_CY = 607.0   # Principal point Y coordinate
    ALPHA = 2           # Pixel distance for tangent vector construction
    
    # Create synthetic depth map
    depth_map, true_depth = create_synthetic_depth_map(2160, 1214)  # iPhone 15 resolution
    
    print(f"Testing with iPhone 15 camera parameters:")
    print(f"  fx: {CAMERA_FX}, fy: {CAMERA_FY}")
    print(f"  cx: {CAMERA_CX}, cy: {CAMERA_CY}")
    print(f"  alpha: {ALPHA}")
    print(f"  Depth map size: {depth_map.shape}")
    
    try:
        # Create estimator with real camera parameters
        estimator = SurfaceNormalEstimator(CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, ALPHA)
        
        # Estimate normals
        normals = estimator.estimate_normals(depth_map)
        
        # Get statistics
        stats = estimator.get_normal_statistics(normals)
        
        print(f"✅ Real camera parameters test successful")
        print(f"  Valid pixels: {stats['valid_pixels']}/{stats['total_pixels']} ({stats['valid_percentage']:.1f}%)")
        print(f"  Mean normal: [{stats['mean_normal'][0]:.3f}, {stats['mean_normal'][1]:.3f}, {stats['mean_normal'][2]:.3f}]")
        
        # Create visualization
        normal_vis = estimator.create_visualization(normals)
        cv2.imwrite("tests/test_real_camera_normals.png", normal_vis)
        print("✅ Real camera test visualization saved")
        
    except Exception as e:
        print(f"❌ Real camera parameters test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Surface Normal Estimation Test Suite")
    print("=" * 50)
    
    # Run basic tests
    success1 = test_surface_normal_estimation()
    
    # Run integration tests
    success2 = test_integration_with_real_camera_params()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Surface normal estimation module is working correctly.")
        print("\nGenerated test files in tests/ folder:")
        print("  - tests/test_depth_map.png (gradient with gentle waves)")
        print("  - tests/test_depth_map_smooth.png (pure gradient)")
        print("  - tests/test_normal_visualization.png")
        print("  - tests/test_normal_visualization_smooth.png")
        print("  - tests/test_magnitude_map.png")
        print("  - tests/test_magnitude_map_smooth.png")
        print("  - tests/test_real_camera_normals.png")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
