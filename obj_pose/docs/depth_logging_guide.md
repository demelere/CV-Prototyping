# Depth Logging Guide

This guide explains the depth logging functionality that helps debug depth estimation outputs by logging raw depth map data in a controlled manner.

## Overview

The depth logging system provides detailed information about depth estimation outputs without overwhelming the logs. It uses smart sampling strategies to capture representative data while keeping log files manageable.

## Features

### 1. Smart Sampling Strategies

The logger supports multiple sampling strategies to capture depth data efficiently:

- **`corners`** (default): Samples corners, center, and edge midpoints (9 points total)
- **`grid`**: Samples on a regular grid across the entire depth map
- **`edges`**: Samples along edges with some interior points
- **`random`**: Random sampling across the depth map

### 2. Comprehensive Analysis

Each log file includes:

- **Metadata**: Timestamp, image path, model name, configuration
- **Statistics**: Min, max, mean, std, unique values
- **Distribution**: Histogram analysis with 20 bins
- **Sampled Values**: Actual depth values at sampled locations
- **Issue Detection**: Automatic detection of potential problems

### 3. Automatic Cleanup

The system automatically manages log files to prevent disk space issues:
- Configurable maximum number of log files
- Automatic removal of oldest files when limit is exceeded

## Configuration

### Main Application Settings

```python
# Depth Logging Parameters
ENABLE_DEPTH_LOGGING = True    # Enable/disable depth logging
DEPTH_LOG_SAMPLE_STRATEGY = 'corners'  # Sampling strategy
DEPTH_LOG_MAX_FILES = 10       # Maximum log files to keep
```

### Sampling Strategy Details

#### Corners Strategy (Recommended)
- **Points**: 9 total (4 corners + center + 4 edge midpoints)
- **Use case**: Quick overview of depth map characteristics
- **File size**: Small (~2-3 KB)

#### Grid Strategy
- **Points**: Up to 100 points on regular grid
- **Use case**: Detailed spatial analysis
- **File size**: Medium (~5-10 KB)

#### Edges Strategy
- **Points**: ~80 points (edges + some interior)
- **Use case**: Boundary analysis and interior sampling
- **File size**: Medium (~8-15 KB)

#### Random Strategy
- **Points**: Up to 100 random points
- **Use case**: Statistical analysis
- **File size**: Medium (~5-10 KB)

## Usage

### Automatic Logging

Depth logging is automatically enabled when `ENABLE_DEPTH_LOGGING = True`. The system will log depth data during:

1. **Initial depth estimation** from the model
2. **Depth processing** (quantization detection, conversion)
3. **Surface normal estimation** (when using ROI-based alpha)

### Manual Logging

You can also use the logger manually:

```python
from utils.depth_logger import create_depth_logger

# Create logger
logger = create_depth_logger(
    log_dir="logs",
    max_log_files=10,
    sample_strategy="corners"
)

# Log depth estimation
log_file = logger.log_depth_estimation(
    depth_array,
    "input_image.jpg",
    "depth-anything-v2",
    additional_metadata={'custom_field': 'value'}
)

# Log depth comparison
logger.log_depth_comparison(
    depth_before,
    depth_after,
    "input_image.jpg",
    ("model_v1", "model_v2"),
    "Before vs after processing"
)
```

## Log File Format

### Depth Estimation Log

```
=== DEPTH ESTIMATION LOG ===
Timestamp: 20250826_143814
Image: /path/to/image.jpg
Model: depth-anything/Depth-Anything-V2-Small-hf
Sample Strategy: corners

=== METADATA ===
timestamp: 20250826_143814
image_path: /path/to/image.jpg
model_name: depth-anything/Depth-Anything-V2-Small-hf
depth_array_shape: (1214, 2160)
depth_array_dtype: float32
depth_min: 0.100000
depth_max: 10.000000
depth_mean: 3.245678
depth_std: 2.123456
unique_values: 256
sample_strategy: corners

=== DEPTH STATISTICS ===
Shape: (1214, 2160)
Data Type: float32
Min: 0.100000
Max: 10.000000
Mean: 3.245678
Std: 2.123456
Unique Values: 256

=== DEPTH DISTRIBUTION ===
Histogram (20 bins):
  Bin  1:    0.100 -    0.595:  12345 pixels
  Bin  2:    0.595 -    1.090:  23456 pixels
  ...

=== SAMPLED DEPTH VALUES ===
Total samples: 9
Format: x, y, depth_value, location(if available)
--------------------------------------------------
  1: (   0,    0) =    1.234 [corner]
  2: (2160,    0) =    2.345 [corner]
  3: (   0, 1214) =    3.456 [corner]
  4: (2160, 1214) =    4.567 [corner]
  5: (1080,  607) =    2.890 [center]
  6: (1080,    0) =    1.567 [top]
  7: (1080, 1214) =    3.789 [bottom]
  8: (   0,  607) =    2.123 [left]
  9: (2160,  607) =    3.456 [right]

=== ANALYSIS ===
Sample depth range: 1.234 - 4.567
Sample depth mean: 2.789
Sample depth std: 1.123

=== POTENTIAL ISSUES ===
⚠️  Limited depth resolution (≤256 unique values) - may indicate quantization
✅ No obvious issues detected
```

### Depth Comparison Log

```
=== DEPTH COMPARISON LOG ===
Timestamp: 20250826_143814
Image: /path/to/image.jpg
Description: Before vs after processing

=== ARRAY 1 ===
Model/Step: original_model
Shape: (1214, 2160)
Min: 0.100000
Max: 10.000000
Mean: 3.245678
Std: 2.123456
Unique Values: 256

=== ARRAY 2 ===
Model/Step: processed_model
Shape: (1214, 2160)
Min: 0.100000
Max: 10.000000
Mean: 3.245678
Std: 2.123456
Unique Values: 256

=== DIFFERENCES ===
Mean difference: 0.000000
Std difference: 0.001234
Max difference: 0.005678
Min difference: -0.003456
```

## Debugging with Depth Logs

### Common Issues and Solutions

#### 1. Quantization Detection
```
⚠️  Limited depth resolution (≤256 unique values) - may indicate quantization
```
**Solution**: Check if depth conversion is working properly

#### 2. Low Variance
```
⚠️  Very low depth variance - may indicate flat or invalid depth map
```
**Solution**: Verify input image quality and model performance

#### 3. Small Depth Range
```
⚠️  Small depth range - may indicate limited depth variation
```
**Solution**: Check if depth scaling is appropriate for your scene

#### 4. NaN/Infinite Values
```
⚠️  NaN values detected in depth map
⚠️  Infinite values detected in depth map
```
**Solution**: Check for model errors or invalid inputs

### Analyzing Depth Patterns

1. **Check corner values**: Look for expected depth variations
2. **Compare center vs edges**: Identify depth gradients
3. **Review histogram**: Understand depth distribution
4. **Examine sample values**: Verify depth magnitudes are reasonable

## Performance Considerations

### File Size Management
- **Corners strategy**: ~2-3 KB per log
- **Grid strategy**: ~5-10 KB per log
- **Edges strategy**: ~8-15 KB per log
- **Random strategy**: ~5-10 KB per log

### Processing Overhead
- **Minimal impact**: Sampling is very fast
- **Configurable**: Can be disabled entirely
- **Automatic cleanup**: Prevents disk space issues

## Best Practices

1. **Use corners strategy** for routine debugging (fastest, smallest files)
2. **Use grid strategy** for detailed analysis when needed
3. **Monitor log file count** and adjust `DEPTH_LOG_MAX_FILES` as needed
4. **Review logs regularly** to catch issues early
5. **Disable logging** in production if not needed

## Testing

Run the test script to verify functionality:

```bash
cd obj_pose
python test_depth_logging.py
```

This will create sample log files demonstrating all features.
