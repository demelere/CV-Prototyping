import cv2
import numpy as np
from PIL import Image
import tempfile
import os

def calibrate_depth_scale(video_path, known_distance_m=0.5):
    """
    Calibrate depth scale using a known object at known distance
    
    Args:
        video_path: Path to your video file
        known_distance_m: Known distance to an object in meters (e.g., 0.5m = 50cm)
    """
    
    print(f"=== Depth Calibration ===")
    print(f"Video: {video_path}")
    print(f"Known distance: {known_distance_m}m")
    
    # Extract first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read video frame")
        return
    
    # Save frame temporarily
    temp_frame_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cv2.imwrite(temp_frame_path.name, frame)
    temp_frame_path.close()
    
    # Load depth model (you'll need to import your depth estimator)
    try:
        from transformers import pipeline
        depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        
        # Get depth estimation
        image = Image.open(temp_frame_path.name)
        result = depth_pipe(image)
        depth_map = result["depth"]
        depth_array = np.array(depth_map)
        
        print(f"\n=== Depth Analysis ===")
        print(f"Depth array shape: {depth_array.shape}")
        print(f"Depth min: {depth_array.min()}")
        print(f"Depth max: {depth_array.max()}")
        print(f"Depth mean: {depth_array.mean():.2f}")
        print(f"Depth std: {depth_array.std():.2f}")
        
        # Calculate suggested scale
        # If we assume the mean depth corresponds to the known distance
        mean_depth = depth_array.mean()
        suggested_scale = known_distance_m / mean_depth
        
        print(f"\n=== Calibration Results ===")
        print(f"Mean depth value: {mean_depth:.2f}")
        print(f"Known distance: {known_distance_m}m")
        print(f"Suggested DEPTH_SCALE_METERS: {suggested_scale:.6f}")
        
        # Also try different percentiles
        percentiles = [25, 50, 75, 90]
        print(f"\n=== Alternative Scales (based on percentiles) ===")
        for p in percentiles:
            depth_value = np.percentile(depth_array, p)
            scale = known_distance_m / depth_value
            print(f"{p}th percentile ({depth_value:.2f}): scale = {scale:.6f}")
        
        # Clean up
        os.unlink(temp_frame_path.name)
        
        return suggested_scale
        
    except Exception as e:
        print(f"Error in depth calibration: {e}")
        if os.path.exists(temp_frame_path.name):
            os.unlink(temp_frame_path.name)
        return None

def estimate_object_distances():
    """Helper function to estimate distances in your welding scene"""
    
    print("\n=== Distance Estimation Guide ===")
    print("To calibrate depth scale, you need to estimate distances in your scene:")
    print()
    print("Common welding distances:")
    print("- Electrode tip to workpiece: 2-5mm (0.002-0.005m)")
    print("- Filler rod tip to workpiece: 2-5mm (0.002-0.005m)")
    print("- Camera to workpiece: 20-50cm (0.2-0.5m)")
    print("- Electrode length: 10-15cm (0.1-0.15m)")
    print("- Filler rod length: 10-15cm (0.1-0.15m)")
    print()
    print("Measurement tips:")
    print("1. Use a ruler to measure known distances")
    print("2. Estimate based on standard welding tool sizes")
    print("3. Use the electrode/rod length as reference")
    print("4. Measure from camera to workpiece surface")

if __name__ == "__main__":
    # Example usage
    print("Depth Calibration Tool")
    print("=====================")
    
    # You can run this with your video file
    # calibrate_depth_scale("your_video.mp4", known_distance_m=0.3)
    
    estimate_object_distances()
    
    print("\nTo calibrate:")
    print("1. Measure a known distance in your scene")
    print("2. Run: calibrate_depth_scale('your_video.mp4', known_distance_m=YOUR_MEASURED_DISTANCE)")
    print("3. Use the suggested scale in your app configuration") 