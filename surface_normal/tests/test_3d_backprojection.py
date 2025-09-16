#!/usr/bin/env python3
"""
Test Script for 3D Back-Projection with Camera Intrinsics
Demonstrates the enhanced surface normal estimation with proper 3D back-projection.
"""

import numpy as np
import cv2
from core.depth_estimator import DepthEstimatorPipeline
from core.surface_normal_estimator import GridBasedSurfaceNormalEstimator

def test_3d_backprojection_with_sample_image():
    """
    Test the 3D back-projection functionality with a sample image.
    This will demonstrate the difference between simple gradient and proper 3D methods.
    """
    
    print("=" * 60)
    print("3D Back-Projection Test with Camera Intrinsics")
    print("=" * 60)
    
    # Initialize depth estimator
    print("\n1. Initializing depth estimation pipeline...")
    depth_estimator = DepthEstimatorPipeline("depth-anything/Depth-Anything-V2-Small-hf", enable_logging=True)
    
    # Camera intrinsics - you should adjust these for your specific camera
    print("\n2. Setting up camera intrinsics...")
    fx, fy = 1512.0, 1512.0  # Focal lengths in pixels
    cx, cy = 1080.0, 607.0   # Principal point (image center approximately)
    
    print(f"   Camera intrinsics:")
    print(f"   - Focal Length: fx={fx}, fy={fy} pixels")
    print(f"   - Principal Point: cx={cx}, cy={cy} pixels")
    print(f"   - Camera Type: Monocular (Depth-Anything-V2)")
    print(f"   - Baseline: N/A (single camera)")
    
    # Initialize surface normal estimator with camera intrinsics
    surface_normal_estimator = GridBasedSurfaceNormalEstimator(
        fx=fx, fy=fy, cx=cx, cy=cy,
        grid_size=32,
        arrow_length=40.0,
        arrow_thickness=2
    )
    
    # Use the sample image if available
    sample_image_path = "IMG_7510.jpeg"
    
    try:
        print(f"\n3. Processing sample image: {sample_image_path}")
        
        # Estimate depth
        print("   Estimating depth...")
        depth_map = depth_estimator.estimate_depth(sample_image_path)
        
        if depth_map is None:
            print("   ❌ Failed to estimate depth")
            return
        
        print(f"   ✅ Depth estimation completed. Shape: {depth_map.shape}")
        print(f"   Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}] meters")
        
        # Compare simple gradient vs 3D back-projection methods
        print("\n4. Comparing surface normal computation methods...")
        
        # Method 1: Simple gradient (old method)
        print("   Method 1: Simple gradient (image space)")
        normal_map_simple, grid_data_simple = surface_normal_estimator.estimate_grid_normals_simple(depth_map)
        
        # Method 2: 3D back-projection (new method)
        print("   Method 2: 3D back-projection (world space)")
        normal_map_3d, point_cloud_3d, grid_data_3d = surface_normal_estimator.estimate_grid_normals_3d(depth_map)
        
        # Analysis and comparison
        print("\n5. Analysis Results:")
        print("   Simple Gradient Method:")
        print(f"   - Valid grid cells: {len(grid_data_simple)}")
        print(f"   - Normal range: X[{normal_map_simple[:,:,0].min():.3f}, {normal_map_simple[:,:,0].max():.3f}]")
        print(f"                   Y[{normal_map_simple[:,:,1].min():.3f}, {normal_map_simple[:,:,1].max():.3f}]")
        print(f"                   Z[{normal_map_simple[:,:,2].min():.3f}, {normal_map_simple[:,:,2].max():.3f}]")
        
        print("\n   3D Back-Projection Method:")
        print(f"   - Valid grid cells: {len(grid_data_3d)}")
        print(f"   - Point cloud shape: {point_cloud_3d.shape}")
        print(f"   - World coordinates:")
        X, Y, Z = point_cloud_3d[:,:,0], point_cloud_3d[:,:,1], point_cloud_3d[:,:,2]
        print(f"     X range: [{X.min():.3f}, {X.max():.3f}] meters")
        print(f"     Y range: [{Y.min():.3f}, {Y.max():.3f}] meters") 
        print(f"     Z range: [{Z.min():.3f}, {Z.max():.3f}] meters")
        print(f"   - Scene dimensions:")
        print(f"     Width:  {X.max() - X.min():.3f} meters")
        print(f"     Height: {Y.max() - Y.min():.3f} meters")
        print(f"     Depth:  {Z.max() - Z.min():.3f} meters")
        
        print(f"   - Normal range: X[{normal_map_3d[:,:,0].min():.3f}, {normal_map_3d[:,:,0].max():.3f}]")
        print(f"                   Y[{normal_map_3d[:,:,1].min():.3f}, {normal_map_3d[:,:,1].max():.3f}]")
        print(f"                   Z[{normal_map_3d[:,:,2].min():.3f}, {normal_map_3d[:,:,2].max():.3f}]")
        
        # Sample comparison of a few grid cells
        print("\n6. Sample Grid Cell Comparison:")
        print("   First 3 grid cells - Simple vs 3D methods:")
        print("   Cell | Simple Normal (nx,ny,nz) | 3D Normal (nx,ny,nz) | 3D World Pos (X,Y,Z)")
        print("   " + "-" * 90)
        
        for i in range(min(3, len(grid_data_simple), len(grid_data_3d))):
            simple_normal = grid_data_simple[i]['normal']
            d3_normal = grid_data_3d[i]['normal']
            d3_world = grid_data_3d[i]['point_3d']
            
            print(f"   {i+1:4d} | ({simple_normal[0]:6.3f},{simple_normal[1]:6.3f},{simple_normal[2]:6.3f}) | "
                  f"({d3_normal[0]:6.3f},{d3_normal[1]:6.3f},{d3_normal[2]:6.3f}) | "
                  f"({d3_world[0]:6.3f},{d3_world[1]:6.3f},{d3_world[2]:6.3f})")
        
        print("\n7. Key Differences:")
        print("   - Simple gradient: Computes normals in image space (pixel units)")
        print("   - 3D back-projection: Computes normals in world space (metric units)")
        print("   - 3D method uses camera intrinsics for proper scale and perspective")
        print("   - 3D method provides actual world coordinates for each pixel")
        print("   - 3D method is geometrically correct for applications requiring real measurements")
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Check the logs/ directory for detailed camera intrinsics analysis.")
        
    except FileNotFoundError:
        print(f"   ⚠️ Sample image '{sample_image_path}' not found.")
        print("   Please place a sample image in the current directory or update the path.")
        
        # Create a synthetic depth map for demonstration
        print("\n   Creating synthetic depth map for demonstration...")
        create_synthetic_depth_demo(surface_normal_estimator)
        
    except Exception as e:
        print(f"   ❌ Error during processing: {e}")

def create_synthetic_depth_demo(surface_normal_estimator):
    """
    Create a synthetic depth map to demonstrate 3D back-projection when no real image is available.
    """
    print("   Generating synthetic depth map...")
    
    # Create a simple synthetic depth map (slanted plane)
    h, w = 480, 640
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Create a slanted plane: depth increases from top-left to bottom-right
    depth_map = 1.0 + 3.0 * (x_coords / w + y_coords / h)  # 1-4 meters depth
    
    print(f"   Synthetic depth map created: {depth_map.shape}")
    print(f"   Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}] meters")
    
    # Test both methods on synthetic data
    print("   Testing simple gradient method...")
    normal_map_simple, grid_data_simple = surface_normal_estimator.estimate_grid_normals_simple(depth_map)
    
    print("   Testing 3D back-projection method...")
    normal_map_3d, point_cloud_3d, grid_data_3d = surface_normal_estimator.estimate_grid_normals_3d(depth_map)
    
    print(f"   ✅ Synthetic test completed!")
    print(f"   - Simple method found {len(grid_data_simple)} valid grid cells")
    print(f"   - 3D method found {len(grid_data_3d)} valid grid cells")
    print(f"   - Generated 3D point cloud with shape: {point_cloud_3d.shape}")

def main():
    """Main test function"""
    print("Starting 3D Back-Projection Test...")
    test_3d_backprojection_with_sample_image()
    print("\nTest completed. Check console output and logs/ directory for results.")

if __name__ == "__main__":
    main()
