# Surface Normal Estimation

A standalone application for estimating depth and surface normals from monocular images using Depth-Anything-V2 models with grid-based surface normal visualization.

## Features

- 🎯 **Depth Estimation**: Uses Depth-Anything-V2 models for accurate monocular depth estimation
- 📐 **Grid-Based Surface Normals**: Computes surface normals on a grid and visualizes them as upward-pointing arrows
- 🎯 **Camera Pose Estimation**: Uses VGGT (Visual Geometry Grounded Transformer) for camera extrinsics and intrinsics
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

### 2. VGGT Camera Pose Estimation Setup

For camera pose estimation using VGGT, run the setup script:

```bash
# Install VGGT for camera pose estimation
python setup/setup_vggt.py
```

**What this does:**
1. **Downloads and installs PyTorch** with appropriate CUDA support
2. **Clones VGGT repository** locally to `vggt/` directory  
3. **Installs VGGT as pip package** using `pip install -e .`
4. **Downloads model weights** (~5GB) from Hugging Face automatically

**Test the installation:**
```bash
# Verify VGGT camera pose extraction works
python tests/test_vggt_camera_pose.py
```

### 3. Run the Application

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
- **Camera Pose Estimation**: Uses VGGT (Visual Geometry Grounded Transformer)
  - Extracts camera extrinsic parameters (rotation and translation)
  - Provides camera intrinsic parameters (focal length and principal point)
  - Cached for fixed camera scenarios (extracts once from first frame)
- **Surface Normals**: Grid-based finite difference computation of surface normals
  - Divides image into regular grid cells (default 40x40 pixels)
  - Computes local surface normal at each grid center using 3D back-projection
  - Uses camera intrinsics for proper 3D normal calculation in world coordinates
- Handles various indoor and outdoor scenes
- Optimized for monocular RGB input

## Configuration

Key parameters can be adjusted in `app.py`:

```python
# Grid-Based Surface Normal Parameters
GRID_SIZE = 40                  # Size of each grid cell in pixels
ARROW_LENGTH = 60.0             # Length of surface normal arrows
ARROW_THICKNESS = 3             # Thickness of arrow lines

# Processing Parameters
TARGET_FPS = 15     # Video processing frame rate
SKIP_FRAMES = True  # Skip frames for faster processing
DEPTH_COLOR_MAP = cv2.COLORMAP_VIRIDIS  # Color map for visualization

# Processing Mode Selection
USE_3D_BACKPROJECTION = True     # Use 3D back-projection for surface normals
EXTRACT_CAMERA_INTRINSICS = True # Extract intrinsics from image metadata
USE_VGGT_CAMERA_POSE = True      # Use VGGT for camera pose estimation

# Default Camera Parameters (used as fallback)
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
├── README.md                       # This file
├── core/
│   ├── __init__.py
│   ├── depth_estimator.py          # Depth estimation pipeline
│   ├── surface_normal_estimator.py # Grid-based surface normal computation
│   └── camera_pose_extractor.py    # VGGT camera pose extraction
├── setup/
│   ├── setup.py                    # Package installation setup
│   └── setup_vggt.py               # VGGT installation script
├── tests/
│   ├── test_3d_backprojection.py   # 3D back-projection testing
│   ├── test_metadata_extraction.py # Camera metadata extraction testing
│   └── test_vggt_camera_pose.py    # VGGT installation and functionality test
├── utils/
│   ├── __init__.py
│   └── metadata_extractor.py       # Camera metadata extraction
├── vggt/                           # VGGT repository (after setup)
└── logs/                           # Processing logs and debug output
```

## VGGT Camera Pose Workflow

The camera pose estimation follows this workflow:

### Setup Phase
1. **`setup/setup_vggt.py`** - Downloads PyTorch, clones VGGT repo, installs as pip package
2. **`vggt/`** - Local VGGT repository installed in editable mode (`pip install -e .`)
3. **`tests/test_vggt_camera_pose.py`** - Verifies installation and camera pose extraction

### Runtime Phase
1. **Image Processing**: 
   - Tries to extract intrinsics from image metadata first
   - Falls back to VGGT if metadata unavailable
   - Extracts camera extrinsics using VGGT
   - Caches pose for subsequent processing

2. **Video Processing**:
   - Extracts camera pose from **first frame only** using VGGT
   - Uses cached pose for all subsequent frames (optimized for fixed camera)
   - Combines with metadata intrinsics when available

3. **Surface Normal Computation**:
   - Uses camera intrinsics for 3D back-projection
   - Uses camera extrinsics for world coordinate transformation
   - Renders surface normals as arrows in proper world space orientation

### Key Benefits
- **Single inference**: VGGT runs once per video (first frame only)
- **Metadata fallback**: Uses VGGT intrinsics when image metadata unavailable  
- **World coordinates**: Proper surface normal orientation using camera extrinsics
- **Fixed camera optimization**: Ideal for TIG welding scenarios with stationary camera

## Testing

The project includes comprehensive test scripts in the `tests/` directory:

### Core Functionality Tests

**1. VGGT Camera Pose Extraction Test**
```bash
python tests/test_vggt_camera_pose.py [optional_image_path]
```
- Tests VGGT installation and model loading
- Verifies camera pose extraction functionality
- Tests caching and parameter extraction
- Uses `IMG_7510.jpeg` by default or specify custom image

**2. Camera Metadata Extraction Test**
```bash
python tests/test_metadata_extraction.py
```
- Tests EXIF data extraction from images
- Tests video metadata parsing
- Validates camera intrinsic parameter extraction
- Tests fallback mechanisms

**3. 3D Back-projection Test**
```bash
python tests/test_3d_backprojection.py
```
- Tests 3D point cloud generation from depth maps
- Validates camera intrinsic integration
- Tests coordinate transformations
- Verifies surface normal calculations

### When to Use Each Test

- **Before first use**: Run `test_vggt_camera_pose.py` after VGGT installation
- **Debugging camera issues**: Use `test_metadata_extraction.py` to check EXIF parsing
- **3D reconstruction problems**: Run `test_3d_backprojection.py` to verify coordinate math
- **Installation verification**: Run all tests to ensure complete setup

### Setup and Installation Tests

**Package Installation Setup**
```bash
python setup/setup.py install
```
- Standard package installation using setuptools
- Installs project dependencies and entry points

**VGGT Installation**
```bash
python setup/setup_vggt.py
```
- Downloads PyTorch with CUDA support
- Clones and installs VGGT repository
- Downloads model weights (~5GB)
- Verifies installation automatically

## Troubleshooting

### Common Issues

1. **GPU Memory**: If running out of GPU memory, use the Small model
2. **Slow Processing**: Enable frame skipping for videos
3. **Poor Depth Quality**: Try a larger model (Base or Large)
4. **VGGT Installation Issues**: 
   - Run `python setup/setup_vggt.py` to reinstall
   - Check `~/.cache/huggingface/hub/` for model downloads (~5GB)
   - Test with `python tests/test_vggt_camera_pose.py`
5. **Camera Pose Errors**: Set `USE_VGGT_CAMERA_POSE = False` in app.py to disable

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