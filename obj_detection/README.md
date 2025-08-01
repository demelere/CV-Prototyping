# RF-DETR Video Object Detection App

A local Gradio web application that allows you to process videos with your trained RF-DETR object detection model from Roboflow. The app splits videos into frames, runs inference on each frame, draws detections, and reassembles the video.

## Features

- 🎥 **Video Upload**: Drag and drop video files (MP4 format)
- 🔍 **Object Detection**: Process each frame with your RF-DETR model
- 📊 **Real-time Progress**: See processing progress with frame-by-frame updates
- 🎨 **Visual Detections**: Bounding boxes and labels drawn on detected objects
- 💾 **Video Output**: Download the processed video with detections

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your Roboflow API Key

1. Go to [Roboflow](https://app.roboflow.com/)
2. Sign in to your account
3. Go to your profile settings
4. Copy your API key

### 3. Find Your Project Details

1. Navigate to your RF-DETR project in Roboflow
2. The URL will be: `https://app.roboflow.com/{workspace_name}/{project_name}`
3. Note the version number of your trained model

## Usage

### 1. Start the App

```bash
python app.py
```

The app will be available at `http://localhost:7860`

### 2. Configure Roboflow Settings

Fill in the configuration fields:
- **API Key**: Your Roboflow API key
- **Workspace Name**: Your workspace name (from URL)
- **Project Name**: Your project name (from URL)
- **Version Number**: Model version number (usually 1)

### 3. Upload and Process Video

1. Click "Upload Video" and select your MP4 video file
2. Click "🚀 Process Video" to start processing
3. Wait for the processing to complete (progress bar will show status)
4. Download the processed video with detections

## How It Works

1. **Video Input**: The app accepts MP4 video files
2. **Frame Extraction**: Video is split into individual frames
3. **Inference**: Each frame is sent to your RF-DETR model via Roboflow API
4. **Detection Drawing**: Bounding boxes and labels are drawn on detected objects
5. **Video Reconstruction**: Processed frames are reassembled into a new video
6. **Output**: Download the final video with all detections

## Configuration Options

The app uses these default settings for inference:
- **Confidence Threshold**: 40% (adjustable in code)
- **Overlap Threshold**: 30% (adjustable in code)
- **Detection Color**: Green bounding boxes
- **Label Format**: `{class_name}: {confidence}`

## Troubleshooting

### Common Issues

1. **"Error loading model"**
   - Check your API key is correct
   - Verify workspace and project names match your Roboflow URL
   - Ensure the version number exists

2. **"Could not open video file"**
   - Make sure the video is in MP4 format
   - Check the video file isn't corrupted

3. **Processing is slow**
   - Larger videos take longer to process
   - Each frame requires an API call to Roboflow
   - Consider using shorter videos for testing

4. **No detections shown**
   - Check if your model is trained and deployed
   - Verify the confidence threshold isn't too high
   - Test with a known good image first

### Performance Tips

- Use shorter videos for testing (30 seconds or less)
- Ensure good internet connection for API calls
- Close other applications to free up memory
- Consider processing during off-peak hours

## Customization

### Modifying Detection Parameters

Edit the `process_video` method in `app.py`:

```python
# Change confidence and overlap thresholds
prediction = self.model.predict(temp_frame_path.name, confidence=50, overlap=20).json()
```

### Changing Detection Colors

Edit the `draw_detections` method:

```python
# Change color (BGR format)
color = (0, 0, 255)  # Red instead of green
```

### Adding Custom Labels

Modify the label format in `draw_detections`:

```python
label = f"{class_name} ({confidence:.1%})"  # Different format
```

## File Structure

```
CV-Prototyping/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

- Python 3.7+
- Roboflow account with trained RF-DETR model
- Internet connection for API calls
- Sufficient disk space for temporary files

## License

This project is open source and available under the MIT License.
