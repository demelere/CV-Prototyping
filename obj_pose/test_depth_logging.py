#!/usr/bin/env python3
"""
Test script to demonstrate depth logging functionality.
"""

import numpy as np
from utils.depth_logger import create_depth_logger
import tempfile
import os

def test_depth_logging():
    """Test the depth logging functionality with synthetic data."""
    
    print("=== Testing Depth Logging Functionality ===\n")
    
    # Create a depth logger
    logger = create_depth_logger(
        log_dir="logs",
        max_log_files=5,
        sample_strategy="corners"
    )
    
    # Create synthetic depth maps for testing
    test_cases = [
        {
            'name': 'Simple slope',
            'depth_map': create_slope_depth_map(100, 100),
            'description': 'Linear slope from top-left to bottom-right'
        },
        {
            'name': 'Flat surface',
            'depth_map': np.ones((80, 120), dtype=np.float32) * 2.5,
            'description': 'Flat surface at 2.5m depth'
        },
        {
            'name': 'Quantized depth',
            'depth_map': np.random.randint(0, 256, (60, 80), dtype=np.uint8).astype(np.float32),
            'description': 'Quantized depth with limited unique values'
        },
        {
            'name': 'Complex geometry',
            'depth_map': create_complex_depth_map(150, 150),
            'description': 'Complex geometry with multiple depth levels'
        }
    ]
    
    # Test each case
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+1}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Depth map shape: {test_case['depth_map'].shape}")
        print(f"Depth range: [{test_case['depth_map'].min():.3f}, {test_case['depth_map'].max():.3f}]")
        print(f"Unique values: {len(np.unique(test_case['depth_map']))}")
        
        # Log the depth data
        log_file = logger.log_depth_estimation(
            test_case['depth_map'],
            f"test_image_{i+1}.jpg",
            f"test_model_{i+1}",
            additional_metadata={
                'test_case': test_case['name'],
                'description': test_case['description']
            }
        )
        
        if log_file:
            print(f"✅ Log saved to: {log_file}")
        else:
            print("❌ Failed to save log")
        
        print("-" * 50)
        print()

def test_sampling_strategies():
    """Test different sampling strategies."""
    
    print("=== Testing Sampling Strategies ===\n")
    
    # Create a test depth map
    depth_map = create_complex_depth_map(200, 200)
    
    strategies = ['grid', 'random', 'corners', 'edges']
    
    for strategy in strategies:
        print(f"Testing strategy: {strategy}")
        
        logger = create_depth_logger(
            log_dir="logs",
            max_log_files=2,
            sample_strategy=strategy
        )
        
        log_file = logger.log_depth_estimation(
            depth_map,
            "test_sampling.jpg",
            "test_model",
            additional_metadata={'sampling_strategy': strategy}
        )
        
        if log_file:
            print(f"✅ Log saved to: {log_file}")
        else:
            print("❌ Failed to save log")
        
        print()

def test_depth_comparison():
    """Test depth comparison logging."""
    
    print("=== Testing Depth Comparison ===\n")
    
    # Create two depth maps for comparison
    depth_map1 = create_slope_depth_map(100, 100)
    depth_map2 = depth_map1 + np.random.normal(0, 0.1, depth_map1.shape)  # Add noise
    
    logger = create_depth_logger(log_dir="logs", max_log_files=2)
    
    log_file = logger.log_depth_comparison(
        depth_map1,
        depth_map2,
        "test_comparison.jpg",
        ("original_model", "noisy_model"),
        "Original vs noisy depth map"
    )
    
    if log_file:
        print(f"✅ Comparison log saved to: {log_file}")
    else:
        print("❌ Failed to save comparison log")

def create_slope_depth_map(height, width):
    """Create a simple slope depth map."""
    depth_map = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            # Create a slope from top-left (1m) to bottom-right (5m)
            depth_map[y, x] = 1.0 + (x + y) * 4.0 / (width + height)
    return depth_map

def create_complex_depth_map(height, width):
    """Create a complex depth map with multiple features."""
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    # Base depth
    depth_map.fill(2.0)
    
    # Add a central peak
    center_y, center_x = height // 2, width // 2
    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance < min(height, width) // 4:
                depth_map[y, x] = 1.0 + 0.5 * np.cos(distance * np.pi / (min(height, width) // 4))
    
    # Add some random variation
    depth_map += np.random.normal(0, 0.1, depth_map.shape)
    
    # Ensure positive depths
    depth_map = np.maximum(depth_map, 0.1)
    
    return depth_map

def test_file_cleanup():
    """Test the automatic cleanup of old log files."""
    
    print("=== Testing File Cleanup ===\n")
    
    # Create a logger with very low max files
    logger = create_depth_logger(
        log_dir="logs",
        max_log_files=3,
        sample_strategy="corners"
    )
    
    # Create multiple depth maps to trigger cleanup
    for i in range(5):
        depth_map = create_slope_depth_map(50, 50)
        log_file = logger.log_depth_estimation(
            depth_map,
            f"cleanup_test_{i}.jpg",
            "cleanup_test_model"
        )
        print(f"Created log {i+1}: {log_file}")
    
    print("✅ Cleanup test completed")

if __name__ == "__main__":
    test_depth_logging()
    test_sampling_strategies()
    test_depth_comparison()
    test_file_cleanup()
    
    print("=== All Tests Completed ===")
    print("Check the 'logs' directory for generated depth log files.")
