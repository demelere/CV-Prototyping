import cv2
import subprocess
import json
import sys
from pathlib import Path

def extract_iphone_camera_params(video_path):
    """Extract camera parameters from iPhone video file"""
    
    print(f"Analyzing video: {video_path}")
    
    # Method 1: Try OpenCV
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Video dimensions: {width}x{height}")
        print(f"FPS: {fps}")
    
    # Method 2: Use ffprobe to extract metadata
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Look for video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                print("\n=== Video Stream Metadata ===")
                print(f"Codec: {video_stream.get('codec_name', 'unknown')}")
                print(f"Dimensions: {video_stream.get('width', 'unknown')}x{video_stream.get('height', 'unknown')}")
                print(f"Frame rate: {video_stream.get('r_frame_rate', 'unknown')}")
                
                # Look for camera metadata
                tags = video_stream.get('tags', {})
                print("\n=== Camera Metadata ===")
                for key, value in tags.items():
                    if any(keyword in key.lower() for keyword in ['focal', 'lens', 'camera', 'iphone']):
                        print(f"{key}: {value}")
                
                # Check for rotation/orientation
                rotation = video_stream.get('rotation', 0)
                print(f"Rotation: {rotation}°")
                
                # Look for additional metadata in format
                format_tags = data.get('format', {}).get('tags', {})
                print("\n=== Format Metadata ===")
                for key, value in format_tags.items():
                    if any(keyword in key.lower() for keyword in ['focal', 'lens', 'camera', 'iphone', 'make', 'model']):
                        print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    
    # Method 3: iPhone 15 Pro Max specifications
    print("\n=== iPhone 15 Pro Max Camera Specs ===")
    print("Main Camera: 48MP, f/1.78, 24mm equivalent")
    print("Ultra Wide: 12MP, f/2.2, 13mm equivalent") 
    print("Telephoto: 12MP, f/2.8, 120mm equivalent")
    print("Computational focal lengths: 28mm, 35mm (from 24mm base)")
    
    return width, height

def estimate_camera_params(width, height):
    """Estimate camera parameters for iPhone 15 Pro Max"""
    
    print("\n=== Estimated Camera Parameters ===")
    
    # iPhone 15 Pro Max main camera specs
    # Physical focal length: ~6.86mm (24mm equivalent)
    # Sensor size: ~9.8mm x 7.35mm (approximate)
    
    # For 24mm equivalent focal length:
    # 24mm = 6.86mm physical focal length
    # 28mm equivalent = ~8mm physical focal length (cropped)
    # 35mm equivalent = ~10mm physical focal length (cropped)
    
    # Calculate focal length in pixels
    # fx = fy = focal_length_mm * sensor_width_pixels / sensor_width_mm
    
    # Approximate sensor dimensions in pixels (depends on recording resolution)
    if width >= 3840:  # 4K
        sensor_width_pixels = width
    else:  # 1080p or lower
        sensor_width_pixels = width
    
    # Estimate based on 24mm equivalent (most common for main camera)
    focal_length_mm = 6.86  # Physical focal length for 24mm equivalent
    sensor_width_mm = 9.8   # Approximate sensor width
    
    fx = fy = focal_length_mm * sensor_width_pixels / sensor_width_mm
    
    # Principal point (usually center of image)
    cx = width / 2
    cy = height / 2
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Estimated focal length (fx, fy): {fx:.1f} pixels")
    print(f"Principal point (cx, cy): ({cx:.1f}, {cy:.1f})")
    
    return fx, fy, cx, cy

def suggest_depth_scale():
    """Suggest depth scale parameters"""
    
    print("\n=== Depth Scale Parameters ===")
    print("Depth scale depends on your depth model output range:")
    print("- Depth Anything models typically output values 0-255 (normalized)")
    print("- You may need to experiment with DEPTH_SCALE_METERS")
    print("- Common values: 0.001 to 0.01 meters per unit")
    print("- Start with DEPTH_SCALE_METERS = 0.001")
    print("- Adjust based on your scene scale")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_camera_params.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    width, height = extract_iphone_camera_params(video_path)
    fx, fy, cx, cy = estimate_camera_params(width, height)
    suggest_depth_scale()
    
    print("\n=== Configuration for app_3d_pose_estimation_pipeline.py ===")
    print(f"CAMERA_FX = {fx:.1f}")
    print(f"CAMERA_FY = {fy:.1f}") 
    print(f"CAMERA_CX = {cx:.1f}")
    print(f"CAMERA_CY = {cy:.1f}")
    print("DEPTH_SCALE_METERS = 0.001  # Start with this value") 