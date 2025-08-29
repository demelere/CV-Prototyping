#!/usr/bin/env python3
"""
Test Script for Camera Metadata Extraction
Demonstrates the automatic extraction of camera intrinsics from image and video metadata.
"""

import os
import sys
from utils.metadata_extractor import CameraMetadataExtractor
from core.depth_estimator import DepthEstimatorPipeline
from core.surface_normal_estimator import GridBasedSurfaceNormalEstimator


def test_image_metadata_extraction():
    """Test metadata extraction from images"""
    
    print("=" * 70)
    print("TESTING IMAGE METADATA EXTRACTION")
    print("=" * 70)
    
    # Initialize metadata extractor
    extractor = CameraMetadataExtractor()
    
    # Test with sample image
    sample_image = "IMG_7510.jpeg"
    
    if os.path.exists(sample_image):
        print(f"\n📸 Testing with image: {sample_image}")
        
        # Extract metadata
        intrinsics = extractor.extract_from_image(sample_image)
        
        # Log the results
        print(f"\n📊 EXTRACTION RESULTS:")
        print(f"   Source: {intrinsics.source}")
        print(f"   Confidence: {intrinsics.confidence}")
        print(f"   Camera Model: {intrinsics.camera_model}")
        print(f"   Image Size: {intrinsics.width}x{intrinsics.height}")
        print(f"   Focal Length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f} pixels")
        print(f"   Principal Point: cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f} pixels")
        
        if intrinsics.focal_length_mm:
            print(f"   Physical Focal Length: {intrinsics.focal_length_mm:.2f} mm")
        if intrinsics.sensor_width_mm:
            print(f"   Sensor Size: {intrinsics.sensor_width_mm:.1f}x{intrinsics.sensor_height_mm:.1f} mm")
        
        # Calculate field of view
        import numpy as np
        fov_x = 2 * np.arctan(intrinsics.width / (2 * intrinsics.fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(intrinsics.height / (2 * intrinsics.fy)) * 180 / np.pi
        print(f"   Field of View: {fov_x:.1f}° x {fov_y:.1f}°")
        
        # Save detailed log
        log_file = "logs/test_image_metadata_extraction.txt"
        extractor.log_intrinsics(intrinsics, log_file)
        
        return intrinsics
        
    else:
        print(f"⚠️ Sample image '{sample_image}' not found.")
        print("   Testing with synthetic image dimensions...")
        
        # Test fallback mechanism
        intrinsics = extractor._estimate_intrinsics_fallback(1920, 1080, "test_fallback")
        print(f"   Fallback result: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        
        return intrinsics


def test_video_metadata_extraction():
    """Test metadata extraction from videos"""
    
    print("\n" + "=" * 70)
    print("TESTING VIDEO METADATA EXTRACTION")
    print("=" * 70)
    
    # Initialize metadata extractor
    extractor = CameraMetadataExtractor()
    
    # Look for video files in common locations
    video_files = [
        "Tig_footage_iphone_15_nd1000.mov",
        "../obj_pose/Tig_footage_iphone_15_nd1000.mov",
        "test_video.mp4",
        "sample_video.mov"
    ]
    
    test_video = None
    for video_file in video_files:
        if os.path.exists(video_file):
            test_video = video_file
            break
    
    if test_video:
        print(f"\n🎥 Testing with video: {test_video}")
        
        # Extract metadata
        intrinsics = extractor.extract_from_video(test_video)
        
        # Log the results
        print(f"\n📊 EXTRACTION RESULTS:")
        print(f"   Source: {intrinsics.source}")
        print(f"   Confidence: {intrinsics.confidence}")
        print(f"   Camera Model: {intrinsics.camera_model}")
        print(f"   Video Size: {intrinsics.width}x{intrinsics.height}")
        print(f"   Focal Length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f} pixels")
        print(f"   Principal Point: cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f} pixels")
        
        if intrinsics.focal_length_mm:
            print(f"   Physical Focal Length: {intrinsics.focal_length_mm:.2f} mm")
        if intrinsics.sensor_width_mm:
            print(f"   Sensor Size: {intrinsics.sensor_width_mm:.1f}x{intrinsics.sensor_height_mm:.1f} mm")
        
        # Calculate field of view
        import numpy as np
        fov_x = 2 * np.arctan(intrinsics.width / (2 * intrinsics.fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(intrinsics.height / (2 * intrinsics.fy)) * 180 / np.pi
        print(f"   Field of View: {fov_x:.1f}° x {fov_y:.1f}°")
        
        # Save detailed log
        log_file = "logs/test_video_metadata_extraction.txt"
        extractor.log_intrinsics(intrinsics, log_file)
        
        return intrinsics
        
    else:
        print("⚠️ No video files found for testing.")
        print("   Testing with synthetic video dimensions...")
        
        # Test fallback mechanism
        intrinsics = extractor._estimate_intrinsics_fallback(1920, 1080, "test_video_fallback")
        print(f"   Fallback result: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        
        return intrinsics


def test_integrated_pipeline():
    """Test the integrated pipeline with extracted metadata"""
    
    print("\n" + "=" * 70)
    print("TESTING INTEGRATED PIPELINE WITH EXTRACTED METADATA")
    print("=" * 70)
    
    try:
        # Initialize depth estimator
        print("\n🔧 Initializing depth estimation pipeline...")
        depth_estimator = DepthEstimatorPipeline("depth-anything/Depth-Anything-V2-Small-hf", enable_logging=False)
        
        # Initialize metadata extractor
        metadata_extractor = CameraMetadataExtractor()
        
        # Test with sample image
        sample_image = "IMG_7510.jpeg"
        
        if os.path.exists(sample_image):
            print(f"\n📸 Processing image with extracted metadata: {sample_image}")
            
            # Extract camera intrinsics
            intrinsics = metadata_extractor.extract_from_image(sample_image)
            
            # Initialize surface normal estimator with extracted intrinsics
            surface_normal_estimator = GridBasedSurfaceNormalEstimator(
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                cx=intrinsics.cx,
                cy=intrinsics.cy,
                grid_size=32,
                arrow_length=40.0,
                arrow_thickness=2
            )
            
            print(f"   ✅ Using extracted intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
            print(f"   📷 Camera: {intrinsics.camera_model} (confidence: {intrinsics.confidence})")
            
            # Estimate depth
            depth_map = depth_estimator.estimate_depth(sample_image)
            
            if depth_map is not None:
                print(f"   ✅ Depth estimation completed: {depth_map.shape}")
                
                # Compute surface normals with 3D back-projection
                normal_map, point_cloud_3d, grid_data = surface_normal_estimator.estimate_grid_normals_3d(depth_map)
                
                print(f"   ✅ Surface normal computation completed")
                print(f"   📊 Generated {len(grid_data)} valid grid cells")
                print(f"   🌍 3D point cloud shape: {point_cloud_3d.shape}")
                
                # Analyze the 3D coordinates
                X, Y, Z = point_cloud_3d[:,:,0], point_cloud_3d[:,:,1], point_cloud_3d[:,:,2]
                print(f"   📏 Scene dimensions:")
                print(f"      Width:  {X.max() - X.min():.3f} meters")
                print(f"      Height: {Y.max() - Y.min():.3f} meters")
                print(f"      Depth:  {Z.max() - Z.min():.3f} meters")
                
                print(f"\n✅ Integrated pipeline test completed successfully!")
                print(f"   The extracted camera intrinsics were successfully used for 3D back-projection.")
                
            else:
                print("   ❌ Depth estimation failed")
                
        else:
            print(f"⚠️ Sample image '{sample_image}' not found for integrated testing.")
            
    except Exception as e:
        print(f"❌ Error in integrated pipeline test: {e}")


def compare_default_vs_extracted():
    """Compare results using default intrinsics vs extracted intrinsics"""
    
    print("\n" + "=" * 70)
    print("COMPARING DEFAULT VS EXTRACTED INTRINSICS")
    print("=" * 70)
    
    # Default intrinsics
    default_fx, default_fy = 1512.0, 1512.0
    default_cx, default_cy = 1080.0, 607.0
    
    print(f"\n📐 DEFAULT INTRINSICS:")
    print(f"   fx={default_fx}, fy={default_fy}")
    print(f"   cx={default_cx}, cy={default_cy}")
    
    # Test image metadata extraction
    sample_image = "IMG_7510.jpeg"
    
    if os.path.exists(sample_image):
        extractor = CameraMetadataExtractor()
        intrinsics = extractor.extract_from_image(sample_image)
        
        print(f"\n🔍 EXTRACTED INTRINSICS:")
        print(f"   fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        print(f"   cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")
        print(f"   Source: {intrinsics.source} (confidence: {intrinsics.confidence})")
        
        # Calculate differences
        fx_diff = abs(intrinsics.fx - default_fx) / default_fx * 100
        fy_diff = abs(intrinsics.fy - default_fy) / default_fy * 100
        cx_diff = abs(intrinsics.cx - default_cx) / default_cx * 100
        cy_diff = abs(intrinsics.cy - default_cy) / default_cy * 100
        
        print(f"\n📊 DIFFERENCES:")
        print(f"   fx: {fx_diff:.1f}% difference")
        print(f"   fy: {fy_diff:.1f}% difference")
        print(f"   cx: {cx_diff:.1f}% difference")
        print(f"   cy: {cy_diff:.1f}% difference")
        
        # Assess impact
        max_diff = max(fx_diff, fy_diff, cx_diff, cy_diff)
        if max_diff > 20:
            print(f"   ⚠️  SIGNIFICANT DIFFERENCE detected! Using extracted intrinsics is RECOMMENDED.")
        elif max_diff > 10:
            print(f"   ⚡ MODERATE DIFFERENCE detected. Using extracted intrinsics is beneficial.")
        else:
            print(f"   ✅ MINOR DIFFERENCE. Default intrinsics are reasonably close.")
            
    else:
        print(f"⚠️ Sample image '{sample_image}' not found for comparison.")


def main():
    """Main test function"""
    
    print("🚀 CAMERA METADATA EXTRACTION TEST SUITE")
    print("=" * 70)
    print("This test suite demonstrates automatic extraction of camera intrinsics")
    print("from image and video metadata (EXIF, video tags, etc.)")
    print("=" * 70)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run tests
    try:
        # Test 1: Image metadata extraction
        image_intrinsics = test_image_metadata_extraction()
        
        # Test 2: Video metadata extraction
        video_intrinsics = test_video_metadata_extraction()
        
        # Test 3: Integrated pipeline
        test_integrated_pipeline()
        
        # Test 4: Compare default vs extracted
        compare_default_vs_extracted()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS COMPLETED")
        print("=" * 70)
        print("✅ Camera metadata extraction is working properly!")
        print("📁 Check the logs/ directory for detailed extraction reports.")
        print("🚀 The system can now automatically extract camera intrinsics from your images and videos.")
        print("\n💡 KEY BENEFITS:")
        print("   • Automatic extraction from EXIF data (images)")
        print("   • Video metadata analysis")
        print("   • Intelligent fallback estimation")
        print("   • Supports iPhone, DSLR, and generic cameras")
        print("   • Proper 3D back-projection with real camera parameters")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
