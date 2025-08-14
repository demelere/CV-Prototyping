import cv2
import numpy as np
import glob
import os

def calibrate_camera_from_images(images_path, checkerboard_size=(9,6), square_size=0.025):
    """
    Calibrate camera using checkerboard images
    
    Args:
        images_path: Path to folder containing calibration images
        checkerboard_size: Number of internal corners (width, height)
        square_size: Size of checkerboard squares in meters
    """
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    
    # Find all calibration images
    images = glob.glob(os.path.join(images_path, '*.jpg')) + glob.glob(os.path.join(images_path, '*.png'))
    
    if not images:
        print(f"No images found in {images_path}")
        return None
    
    print(f"Found {len(images)} calibration images")
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            
            # Draw corners
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(500)
        else:
            print(f"Could not find checkerboard in {fname}")
    
    cv2.destroyAllWindows()
    
    if len(objpoints) < 3:
        print("Need at least 3 successful calibration images")
        return None
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if ret:
        print("\n=== Camera Calibration Results ===")
        print(f"Camera Matrix:")
        print(mtx)
        print(f"\nDistortion Coefficients:")
        print(dist)
        
        # Extract parameters
        fx = mtx[0,0]
        fy = mtx[1,1]
        cx = mtx[0,2]
        cy = mtx[1,2]
        
        print(f"\nExtracted Parameters:")
        print(f"fx = {fx:.1f}")
        print(f"fy = {fy:.1f}")
        print(f"cx = {cx:.1f}")
        print(f"cy = {cy:.1f}")
        
        return {
            'camera_matrix': mtx,
            'distortion_coeffs': dist,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
    else:
        print("Calibration failed")
        return None

def estimate_depth_scale(known_distance_m, known_object_pixels, depth_value):
    """
    Estimate depth scale using a known object at known distance
    
    Args:
        known_distance_m: Distance to object in meters
        known_object_pixels: Size of object in pixels
        depth_value: Depth model output value for that object
    """
    
    # If depth_value is the raw model output (0-255), convert to meters
    # This is a rough estimate - you'll need to experiment
    estimated_scale = known_distance_m / depth_value
    
    print(f"\n=== Depth Scale Estimation ===")
    print(f"Known distance: {known_distance_m}m")
    print(f"Depth model output: {depth_value}")
    print(f"Estimated scale: {estimated_scale:.6f} meters per unit")
    print(f"Suggested DEPTH_SCALE_METERS: {estimated_scale:.6f}")
    
    return estimated_scale

if __name__ == "__main__":
    # Example usage
    print("Camera Calibration Tool")
    print("=======================")
    print("1. Print a checkerboard pattern (9x6 internal corners)")
    print("2. Take photos of the checkerboard from different angles")
    print("3. Place images in a folder")
    print("4. Run this script with the folder path")
    
    # Uncomment to run calibration
    # result = calibrate_camera_from_images("calibration_images/")
    # if result:
    #     print("Calibration successful!")
    
    # Example depth scale estimation
    print("\nDepth Scale Estimation:")
    print("Measure a known object in your scene and estimate:")
    known_distance = 0.5  # meters
    depth_output = 128    # depth model output (0-255)
    estimate_depth_scale(known_distance, 100, depth_output) 