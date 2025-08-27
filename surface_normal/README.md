# Depth Estimation

A standalone application for estimating depth from monocular images using Depth-Anything-V2 models.

## Features

- 🎯 **Depth Estimation**: Uses Depth-Anything-V2 models for accurate monocular depth estimation
- 🎨 **Depth Visualization**: Color-coded depth maps for easy interpretation
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
2. Click "Preview Image" to see results
3. View depth map visualization

### Video Processing
1. Upload a video using the "Video" tab
2. Click "Process Video" 
3. Download processed video with depth visualization

## Understanding the Output

### Visualizations
- **Depth Map**: Colored depth visualization showing distance from camera
  - Closer objects appear in warmer colors (red/yellow)
  - Farther objects appear in cooler colors (blue/purple)

## Technical Details

### Depth Models
- **Small**: Fastest processing, good for real-time applications
- **Base**: Balanced speed and accuracy
- **Large**: Highest accuracy, slower processing

### Algorithm
- Uses transformer-based depth estimation from Depth-Anything-V2
- Handles various indoor and outdoor scenes
- Optimized for monocular RGB input

## Configuration

Key parameters can be adjusted in `app.py`:

```python
# Processing Parameters
TARGET_FPS = 15     # Video processing frame rate
SKIP_FRAMES = True  # Skip frames for faster processing
DEPTH_COLOR_MAP = cv2.COLORMAP_VIRIDIS  # Color map for visualization
```

## Project Structure

```
surface_normal/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── core/
│   ├── __init__.py
│   └── depth_estimator.py         # Depth estimation pipeline
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