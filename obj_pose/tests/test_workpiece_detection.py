#!/usr/bin/env python3
"""
Test script for workpiece detection and tracking functionality
"""

import numpy as np
import cv2
from app_3d_pose_estimation_pipeline import WorkpieceDetector, TravelTracker
import time

def test_workpiece_detection():
    """Test workpiece detection with synthetic depth data"""
    print("Testing workpiece detection...")
    
    # Create synthetic depth map (flat surface)
    height, width = 480, 640
    depth_map = np.ones((height, width), dtype=np.uint8) * 128  # Flat surface
    
    # Add some noise
    noise = np.random.normal(0, 5, (height, width))
    depth_map = np.clip(depth_map + noise, 0, 255).astype(np.uint8)
    
    # Initialize detector
    detector = WorkpieceDetector()
    
    # Test detection
    success = detector.detect_workpiece_surface(depth_map, width, height)
    
    if success:
        print("✅ Workpiece detection successful!")
        print(f"Normal vector: {detector.get_workpiece_normal()}")
        print(f"Origin: {detector.get_workpiece_origin()}")
        print(f"Confidence: {detector.get_detection_confidence():.2f}")
    else:
        print("❌ Workpiece detection failed")
    
    return success

def test_travel_tracking():
    """Test travel tracking with synthetic movement data"""
    print("\nTesting travel tracking...")
    
    # Initialize tracker
    tracker = TravelTracker()
    
    # Simulate electrode movement
    positions = [
        np.array([0.0, 0.0, 0.5]),
        np.array([0.01, 0.0, 0.5]),
        np.array([0.02, 0.0, 0.5]),
        np.array([0.03, 0.0, 0.5]),
        np.array([0.04, 0.0, 0.5])
    ]
    
    # Update tracking with positions
    for i, pos in enumerate(positions):
        tracker.update_tracking(pos, None, time.time() + i * 0.1)
        print(f"Frame {i}: Position {pos}")
    
    # Check results
    velocity = tracker.get_electrode_velocity()
    speed = tracker.get_electrode_speed()
    
    if velocity is not None:
        print("✅ Travel tracking successful!")
        print(f"Velocity vector: {velocity}")
        print(f"Speed: {speed:.3f} m/s")
        print(f"Speed in mm/s: {tracker.get_speed_in_units(speed, 'mm/s'):.1f}")
    else:
        print("❌ Travel tracking failed")
    
    return velocity is not None

def test_coordinate_transformation():
    """Test coordinate transformation"""
    print("\nTesting coordinate transformation...")
    
    detector = WorkpieceDetector()
    
    # Create a simple depth map
    depth_map = np.ones((480, 640), dtype=np.uint8) * 128
    success = detector.detect_workpiece_surface(depth_map, 640, 480)
    
    if success:
        # Test point transformation
        test_point = np.array([0.1, 0.2, 0.3])
        transformed_point = detector.transform_to_workpiece_coordinates(test_point)
        
        print("✅ Coordinate transformation successful!")
        print(f"Original point: {test_point}")
        print(f"Transformed point: {transformed_point}")
    else:
        print("❌ Coordinate transformation failed")
    
    return success

if __name__ == "__main__":
    print("=== Testing Workpiece Detection and Tracking ===\n")
    
    # Run tests
    test1 = test_workpiece_detection()
    test2 = test_travel_tracking()
    test3 = test_coordinate_transformation()
    
    print(f"\n=== Test Results ===")
    print(f"Workpiece Detection: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Travel Tracking: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Coordinate Transform: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests passed! Implementation is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the implementation.") 