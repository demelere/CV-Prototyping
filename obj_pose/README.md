# Keypoint Detection Video Processor

A Gradio web application for processing videos with keypoint detection models trained on Roboflow.

## Features

- **Video Processing**: Upload and process videos with keypoint detection
- **Preview Functionality**: Test inference on the first frame before processing
- **Configurable Parameters**: Adjust confidence thresholds, FPS, visual settings
- **Frame Skipping**: Optimize processing speed by skipping frames
- **Keypoint Visualization**: Draw keypoints with customizable colors and labels
- **Connection Lines**: Draw connections between keypoints (configurable)
- **Progress Tracking**: Real-time progress updates during processing

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a trained keypoint detection model on Roboflow.

## Usage

1. **Run the application**:
```bash
python app_keypoint_detection.py
```

2. **Open your browser** and navigate to `http://localhost:7861`

3. **Configure your Roboflow settings**:
   - Enter your Roboflow API key
   - Provide your workspace name
   - Enter your project name
   - Specify the version number

4. **Upload a video** and test the preview functionality

5. **Process the video** to get the final result with keypoints

## Configuration

You can customize the application by modifying the configuration parameters at the top of `app_keypoint_detection.py`:

### Detection Parameters
- `CONFIDENCE_THRESHOLD`: Minimum confidence for keypoint detection (default: 70%)

### Processing Parameters
- `TARGET_FPS`: Target frames per second for processing (default: 20)
- `SKIP_FRAMES`: Whether to skip frames to match target FPS (default: True)

### Visual Display Parameters
- `KEYPOINT_RADIUS`: Radius of keypoint circles (default: 4px)
- `KEYPOINT_THICKNESS`: Thickness of keypoint circles (default: 2px)
- `CONNECTION_THICKNESS`: Thickness of connection lines (default: 2px)
- `KEYPOINT_COLOR`: Color of keypoints in BGR format (default: Green)
- `CONNECTION_COLOR`: Color of connections in BGR format (default: Blue)

### Label Display Options
- `SHOW_CONFIDENCE`: Show confidence scores in labels
- `SHOW_KEYPOINT_NAME`: Show keypoint names in labels
- `CONFIDENCE_DECIMAL_PLACES`: Number of decimal places for confidence
- `LABEL_FORMAT`: Format for labels ("keypoint_confidence", "keypoint_only", etc.)

### Keypoint Connections
Define connections between keypoints by modifying `KEYPOINT_CONNECTIONS`:
```python
KEYPOINT_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),  # Example connections
    (3, 4), (4, 5), (5, 6),  # Adjust based on your model
]
```

## Workflow

1. **Upload Video**: Drag and drop your video file
2. **Configure Settings**: Enter your Roboflow project details
3. **Preview**: Click "Preview First Frame" to test inference
4. **Process**: Click "Process Video" to generate the final result
5. **Download**: Download the processed video with keypoints

## Troubleshooting

- **No keypoints detected**: Check your confidence threshold and model performance
- **Incorrect coordinates**: The app handles both percentage (0-1) and pixel coordinates
- **Slow processing**: Reduce `TARGET_FPS` or enable `SKIP_FRAMES`
- **Memory issues**: Process shorter videos or reduce frame resolution

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

## Port Configuration

The app runs on port 7861 by default. You can change this in the `demo.launch()` call at the bottom of the file. 