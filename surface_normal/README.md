# Surface Normal Estimation

A standalone application for estimating depth and surface normals from monocular images using Depth-Anything-V2 models with grid-based surface normal visualization.

## Features

- 🎯 **Depth Estimation**: Uses Depth-Anything-V2 models for accurate monocular depth estimation
- 📐 **Grid-Based Surface Normals**: Computes surface normals on a grid and visualizes them as upward-pointing arrows
- 🎨 **Multi-Visualization**: Depth maps, surface normal arrows, and combined views

- 📹 **Video Processing**: Process both single images and video sequences
- 🌐 **Web Interface**: Easy-to-use Gradio web interface

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:7864`

## Usage

### Image Processing
1. Upload an image using the "Image" tab
2. Click "Preview Image" to see depth map and surface normal arrows
3. View grid-based surface normal visualization

### Video Processing
1. Upload a video using the "Video" tab
2. Click "Process Video" 
3. Download processed videos with surface normal arrow visualization

## Understanding the Output

### Visualizations
- **Depth Map**: Colored depth visualization showing distance from camera
  - Closer objects appear in warmer colors (red/yellow)
  - Farther objects appear in cooler colors (blue/purple)
- **Surface Normal Arrows**: Green arrows pointing upward from surface grid cells
  - Each arrow represents the surface normal direction at that grid location
  - Arrow length and direction indicate surface orientation
- **Combined View**: Depth map with surface normal arrows overlaid

## Technical Details

### Depth Models
- **Small**: Fastest processing, good for real-time applications
- **Base**: Balanced speed and accuracy
- **Large**: Highest accuracy, slower processing

### Algorithms
- **Depth Estimation**: Uses transformer-based depth estimation from Depth-Anything-V2
- **Surface Normals**: Grid-based finite difference computation of surface normals
  - Divides image into regular grid cells (default 32x32 pixels)
  - Computes local surface normal at each grid center
  - Uses camera intrinsics for proper 3D normal calculation
- Handles various indoor and outdoor scenes
- Optimized for monocular RGB input

## Configuration

Key parameters can be adjusted in `app.py`:

```python
# Grid-Based Surface Normal Parameters
GRID_SIZE = 32                  # Size of each grid cell in pixels
ARROW_LENGTH = 25.0             # Length of surface normal arrows
ARROW_THICKNESS = 2             # Thickness of arrow lines

# Processing Parameters
TARGET_FPS = 15     # Video processing frame rate
SKIP_FRAMES = True  # Skip frames for faster processing
DEPTH_COLOR_MAP = cv2.COLORMAP_VIRIDIS  # Color map for visualization

# Camera Parameters
CAMERA_FX = 1512.0  # Focal length X
CAMERA_FY = 1512.0  # Focal length Y  
CAMERA_CX = 1080.0  # Principal point X
CAMERA_CY = 607.0   # Principal point Y
```

## Project Structure

```
surface_normal/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── core/
│   ├── __init__.py
│   ├── depth_estimator.py         # Depth estimation pipeline
│   └── surface_normal_estimator.py # Grid-based surface normal computation
├── utils/
│   └── __init__.py
└── logs/                          # Processing logs and debug output
```

## Troubleshooting

### Common Issues

1. **GPU Memory**: If running out of GPU memory, use the Small model
2. **Slow Processing**: Enable frame skipping for videos
3. **Poor Depth Quality**: Try a larger model (Base or Large)

### Debug Information
- Check console output for detailed processing information
- Review log files in `logs/` directory for depth analysis

## Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model pipeline
- **OpenCV**: Computer vision operations
- **Gradio**: Web interface framework
- **NumPy**: Scientific computing

## License

This project builds upon:
- Depth-Anything-V2 models (MIT License)
- Depth estimation algorithms from computer vision literature

## Related Projects

- **obj_detection**: Object detection with keypoints
- **obj_pose**: 3D pose estimation pipeline
- **CV-Prototyping**: Parent computer vision project