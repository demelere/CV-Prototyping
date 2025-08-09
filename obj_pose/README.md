# 3D Keypoint Detection with Depth Estimation

This application combines 2D keypoint detection from Roboflow with depth estimation from Hugging Face to create 3D pose estimation for TIG welding electrode and filler rod tracking.

## Overview

The system works by:
1. 2D Keypoint Detection: Using your trained Roboflow model to detect keypoints on welding tools
2. Depth Estimation: Using Depth Anything v2 from Hugging Face to estimate depth from monocular images
3. 3D Reconstruction: Combining 2D keypoints with depth information to approximate 3D coordinates
4. Pose Analysis: Calculating travel and work angles relative to the workpiece

## Quick Start

### Install Dependencies

```bash
cd obj_pose
pip install -r requirements.txt
```

**Note:** The application now supports `.env` file configuration. If you want to use this feature, make sure `python-dotenv` is installed (it's included in requirements.txt).

### Run the Application

```bash
python app_3d_pose_estimation_pipeline.py
```

The app will be available at `http://localhost:7863`

## Configuration

### Environment Variables (Recommended)

Create a `.env` file in the same directory as the application with your Roboflow credentials:

```bash
# Copy the example file
cp .env.example .env

# Edit with your actual values
nano .env
```

Example `.env` file:
```env
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=your_workspace_name
ROBOFLOW_PROJECT=your_project_name
ROBOFLOW_VERSION=1
```

**Benefits of using .env file:**
- No need to enter credentials every time you run the app
- Keeps sensitive API keys out of your code
- Easy to switch between different projects

### Manual Configuration

If you prefer to enter values manually each time, you can leave the `.env` file empty or delete it. The application will prompt you to enter:
- Roboflow API Key
- Workspace Name  
- Project Name
- Version Number

### Key Parameters

```python
# Detection Parameters
CONFIDENCE_THRESHOLD = 25  # Only show keypoints with this confidence % or higher

# Processing Parameters
TARGET_FPS = 15            # Target FPS for processing (lower = faster processing)
SKIP_FRAMES = True         # Whether to skip frames to match target FPS

# 3D Visualization Parameters
SHOW_DEPTH_MAP = True              # Show depth map visualization
SHOW_3D_COORDINATES = True         # Show 3D coordinates in labels
DEPTH_SCALE_FACTOR = 1000.0        # Scale factor for depth values

# Depth Model Options
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"  # Small: Fastest, Large: Most accurate
```

### Keypoint Connections

Define connections between keypoints for your welding model:

```python
KEYPOINT_CONNECTIONS = [
    # Example for welding electrode
    (0, 1),  # Electrode tip to handle
    (1, 2),  # Handle to connection point
    # Add more connections based on your model
]
```

## Understanding the Output

### 3D Keypoints
- Green circles: 2D keypoints with confidence above threshold
- Color-coded by depth: Blue (closer) to Red (farther)
- Labels: Show 3D coordinates (x, y, depth) and confidence

### Depth Visualization
- Grayscale depth map: Brighter = closer, darker = farther
- Colored depth map: Viridis colormap for better visualization

### Video Outputs
- Processed video: Original video with 3D keypoints overlaid
- Depth video: Depth map visualization for each frame
- Combined video: Keypoints overlaid on depth map for each frame

## For TIG Welding Analysis

### Keypoint Setup
For TIG welding electrode tracking, you'll want keypoints like:
- Electrode tip (primary tracking point)
- Electrode handle (for orientation)
- Filler rod tip (if using filler)
- Workpiece surface (reference point)

### Angle Calculations
Once you have 3D coordinates, you can calculate:

```python
def calculate_travel_angle(electrode_tip_3d, electrode_handle_3d, workpiece_normal):
    """Calculate travel angle relative to workpiece"""
    electrode_vector = electrode_tip_3d - electrode_handle_3d
    # Calculate angle between electrode vector and workpiece normal
    return angle_between_vectors(electrode_vector, workpiece_normal)

def calculate_work_angle(electrode_tip_3d, travel_direction, workpiece_normal):
    """Calculate work angle (perpendicular to travel direction)"""
    # Calculate angle in the plane perpendicular to travel direction
    pass
```

## Workflow

1. Upload Video: Drag and drop your video file
2. Configure Settings: Enter your Roboflow project details
3. Preview: Click "Preview First Frame" to test inference
4. Process: Click "Process Video" to generate the final result
5. Download: Download the processed video with 3D keypoints

## Troubleshooting

### Common Issues

1. Model Loading Errors
   - Check your Roboflow API key and project details
   - Ensure your model supports keypoint detection
   - Verify the model version number

2. Depth Estimation Issues
   - Depth scale may need adjustment (DEPTH_SCALE_FACTOR)
   - Try different depth models if needed
   - Check image quality and lighting

3. Performance Issues
   - Reduce TARGET_FPS for faster processing
   - Increase SKIP_FRAMES for lower computational load
   - Consider using a smaller depth model

### Debug Information

The app provides detailed logging:
- Frame processing status
- Keypoint detection counts
- Depth estimation results
- 3D coordinate calculations

## Model Requirements

Your Roboflow keypoint detection model should return predictions in this format:

```json
{
  "predictions": [
    {
      "keypoints": [
        {
          "x": 0.5,
          "y": 0.3,
          "confidence": 0.95,
          "name": "head"
        }
      ]
    }
  ]
}
```

## Next Steps

### Calibrate for Your Setup
- Adjust DEPTH_SCALE_FACTOR based on your camera setup
- Fine-tune confidence thresholds
- Define appropriate keypoint connections

### Add Workpiece Detection
- Train a separate model for workpiece detection
- Use workpiece surface normal for angle calculations
- Implement workpiece pose estimation

### Real-time Processing
- Consider local model deployment for real-time use
- Optimize frame processing pipeline
- Add angle calculation in real-time

### Advanced Analysis
- Implement travel speed calculation
- Add arc length measurement
- Create welding quality metrics

## Resources

- [Roboflow Keypoint Detection](https://docs.roboflow.com/keypoint-detection)
- [Depth Anything v2 Model](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- [OpenCV Documentation](https://docs.opencv.org/)

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify your API keys and model configurations
3. Check the troubleshooting section above 