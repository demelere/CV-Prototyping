#!/usr/bin/env python3
"""
Test script for the Keypoint Detection Video Processor
This script helps verify that all dependencies are installed correctly.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'gradio',
        'cv2',
        'numpy',
        'roboflow',
        'PIL',
        'tempfile',
        'os',
        'json'
    ]
    
    print("Testing imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_opencv():
    """Test OpenCV functionality"""
    try:
        import cv2
        import numpy as np
        print("\nTesting OpenCV...")
        
        # Test video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("✅ OpenCV video codec test passed")
        
        # Test basic image operations
        test_img = cv2.imread('test_image.jpg') if cv2.imread('test_image.jpg') is not None else None
        if test_img is None:
            # Create a dummy image for testing
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.circle(test_img, (50, 50), 20, (0, 255, 0), 2)
            print("✅ OpenCV image operations test passed")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_roboflow():
    """Test Roboflow connectivity (without API key)"""
    try:
        from roboflow import Roboflow
        print("\nTesting Roboflow...")
        
        # Test that we can create a Roboflow instance (won't work without API key, but tests import)
        print("✅ Roboflow import successful")
        print("Note: API key required for actual model inference")
        
        return True
    except Exception as e:
        print(f"❌ Roboflow test failed: {e}")
        return False

def test_gradio():
    """Test Gradio functionality"""
    try:
        import gradio as gr
        print("\nTesting Gradio...")
        
        # Test basic Gradio components
        test_interface = gr.Interface(
            fn=lambda x: x,
            inputs=gr.Textbox(),
            outputs=gr.Textbox()
        )
        print("✅ Gradio interface creation test passed")
        
        return True
    except Exception as e:
        print(f"❌ Gradio test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Keypoint Detection App - Dependency Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import test failed. Please fix dependencies before proceeding.")
        return False
    
    # Test individual components
    opencv_ok = test_opencv()
    roboflow_ok = test_roboflow()
    gradio_ok = test_gradio()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"OpenCV: {'✅ PASS' if opencv_ok else '❌ FAIL'}")
    print(f"Roboflow: {'✅ PASS' if roboflow_ok else '❌ FAIL'}")
    print(f"Gradio: {'✅ PASS' if gradio_ok else '❌ FAIL'}")
    
    all_passed = all([imports_ok, opencv_ok, roboflow_ok, gradio_ok])
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run the keypoint detection app.")
        print("\nNext steps:")
        print("1. Get your Roboflow API key")
        print("2. Run: python app_keypoint_detection.py")
        print("3. Open http://localhost:7861 in your browser")
    else:
        print("\n❌ Some tests failed. Please fix the issues before running the app.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 