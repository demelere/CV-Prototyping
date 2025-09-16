#!/usr/bin/env python3
"""
Test script for VGGT Camera Pose Extraction

This script tests the camera pose extraction functionality using VGGT.
It can be run with the sample image in the project or any other image.

Usage:
    python test_vggt_camera_pose.py [image_path]
    
If no image path is provided, it will use the default sample image (IMG_7510.jpeg).
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging
from core.camera_pose_extractor import CameraPoseExtractor, extract_camera_pose_from_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vggt_installation():
    """Test if VGGT is properly installed and can be imported."""
    print("="*60)
    print("TESTING VGGT INSTALLATION")
    print("="*60)
    
    try:
        import vggt
        print("✓ VGGT imported successfully")
        
        from vggt.models.vggt import VGGT
        print("✓ VGGT model class imported successfully")
        
        from vggt.utils.load_fn import load_and_preprocess_images
        print("✓ VGGT utilities imported successfully")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ Using device: {device}")
        
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            print(f"✓ CUDA capability: {capability}")
            dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
            print(f"✓ Using dtype: {dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error importing VGGT: {e}")
        return False

def test_camera_pose_extraction(image_path: str):
    """Test camera pose extraction with a given image."""
    print("="*60)
    print("TESTING CAMERA POSE EXTRACTION")
    print("="*60)
    
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False
    
    print(f"Testing with image: {image_path}")
    
    try:
        # Test 1: Quick extraction function
        print("\n--- Test 1: Quick extraction function ---")
        extrinsic, intrinsic = extract_camera_pose_from_image(image_path)
        print(f"✓ Quick extraction successful")
        print(f"  Extrinsic shape: {extrinsic.shape}")
        print(f"  Intrinsic shape: {intrinsic.shape}")
        
        # Test 2: CameraPoseExtractor class
        print("\n--- Test 2: CameraPoseExtractor class ---")
        extractor = CameraPoseExtractor()
        params = extractor.get_camera_parameters_dict(image_path)
        
        print(f"✓ Camera parameters extracted successfully")
        print(f"  Focal lengths: fx={params['fx']:.2f}, fy={params['fy']:.2f}")
        print(f"  Principal point: cx={params['cx']:.2f}, cy={params['cy']:.2f}")
        print(f"  Translation: [{params['translation'][0]:.3f}, {params['translation'][1]:.3f}, {params['translation'][2]:.3f}]")
        
        # Test 3: Save and load parameters
        print("\n--- Test 3: Save and load parameters ---")
        output_path = "test_camera_params.npz"
        extractor.save_camera_parameters(image_path, output_path)
        print(f"✓ Parameters saved to: {output_path}")
        
        from core.camera_pose_extractor import load_camera_parameters
        loaded_params = load_camera_parameters(output_path)
        print(f"✓ Parameters loaded successfully")
        print(f"  Loaded keys: {list(loaded_params.keys())}")
        
        # Clean up
        Path(output_path).unlink()
        print(f"✓ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during camera pose extraction: {e}")
        logger.exception("Full error details:")
        return False

def print_detailed_results(image_path: str):
    """Print detailed camera parameter results."""
    print("="*60)
    print("DETAILED CAMERA PARAMETERS")
    print("="*60)
    
    try:
        extractor = CameraPoseExtractor()
        params = extractor.get_camera_parameters_dict(image_path)
        
        print(f"Image: {image_path}")
        print(f"Image size: {Path(image_path).stat().st_size / (1024*1024):.1f} MB")
        
        print("\nIntrinsic Matrix:")
        intrinsic = params['intrinsic']
        for i in range(3):
            print(f"  [{intrinsic[i,0]:8.2f} {intrinsic[i,1]:8.2f} {intrinsic[i,2]:8.2f}]")
        
        print(f"\nIntrinsic Parameters:")
        print(f"  Focal length X (fx): {params['fx']:.2f}")
        print(f"  Focal length Y (fy): {params['fy']:.2f}")
        print(f"  Principal point X (cx): {params['cx']:.2f}")
        print(f"  Principal point Y (cy): {params['cy']:.2f}")
        
        print("\nExtrinsic Matrix (Camera to World):")
        extrinsic = params['extrinsic']
        for i in range(4):
            print(f"  [{extrinsic[i,0]:8.3f} {extrinsic[i,1]:8.3f} {extrinsic[i,2]:8.3f} {extrinsic[i,3]:8.3f}]")
        
        print(f"\nRotation Matrix:")
        rotation = params['rotation']
        for i in range(3):
            print(f"  [{rotation[i,0]:8.3f} {rotation[i,1]:8.3f} {rotation[i,2]:8.3f}]")
        
        print(f"\nTranslation Vector:")
        translation = params['translation']
        print(f"  [{translation[0]:8.3f} {translation[1]:8.3f} {translation[2]:8.3f}]")
        
        # Camera pose interpretation
        print(f"\nCamera Pose Interpretation:")
        camera_position = -rotation.T @ translation
        print(f"  Camera position in world: [{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]")
        
        # Camera forward direction (negative Z in camera frame)
        forward_direction = rotation[2, :]
        print(f"  Camera forward direction: [{forward_direction[0]:.3f}, {forward_direction[1]:.3f}, {forward_direction[2]:.3f}]")
        
    except Exception as e:
        print(f"✗ Error getting detailed results: {e}")
        logger.exception("Full error details:")

def main():
    """Main test function."""
    print("VGGT Camera Pose Extraction Test")
    print("="*60)
    
    # Determine image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use default sample image
        image_path = "IMG_7510.jpeg"
    
    # Test 1: VGGT installation
    if not test_vggt_installation():
        print("\n✗ VGGT installation test failed. Please check your installation.")
        return False
    
    # Test 2: Camera pose extraction
    if not test_camera_pose_extraction(image_path):
        print(f"\n✗ Camera pose extraction test failed with image: {image_path}")
        return False
    
    # Test 3: Detailed results
    print_detailed_results(image_path)
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED SUCCESSFULLY!")
    print("VGGT camera pose extraction is ready for use.")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)