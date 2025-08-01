# CV-Prototyping

CV prototyping by specific tasks

## Project Structure

```
CV-Prototyping/
├── obj_detection/     # Object detection 
│   ├── app_configurable.py
│   ├── test_roboflow.py
│   ├── requirements.txt
│   └── README.md
├── pose/             # Object pose estimation 
│   ├── README.md
│   └── requirements.txt
└── README.md         # This file
```

### Object Detection (`obj_detection/`)
- RF-DETR video processing, shows weld pool, arc, tungsten electrode, and filler rod

### Pose Estimation (`obj_pose/`)
- Keypoint detection and object pose estimation

## Getting Started

1. Navigate to the specific subdirectory for your use case
2. Install dependencies: `pip install -r requirements.txt`
3. Run the appropriate application

## Adding New Prototypes

To add a new CV model type:
1. Create a new subdirectory: `mkdir new_model_type`
2. Add a `README.md` and `requirements.txt`
3. Add your prototype applications

## Current Status

- Object Detection: RF-DETR video processing app
- Pose Estimation: Structure ready, apps in development 