# Surface Normal Estimation Module

This module provides functionality to estimate surface normals from depth maps using the methodology from research papers. It's integrated into the 3D pose estimation pipeline to enhance depth analysis.

## Features

- **Surface Normal Estimation**: Compute surface normals from depth maps using camera intrinsics
- **Visualization**: Create colored visualizations of surface normals and magnitude maps
- **Statistics**: Get detailed statistics about the estimated normals
- **Masking Support**: Apply masks to focus on specific regions of interest
- **Integration**: Seamlessly integrated with the main 3D pose estimation pipeline

## Usage

### Basic Usage

```python
from core.surface_normal_estimator import SurfaceNormalEstimator

# Initialize with camera parameters
estimator = SurfaceNormalEstimator(
    fx=1512.0,  # Focal length X
    fy=1512.0,  # Focal length Y
    ox=1080.0,  # Optical center X
    oy=607.0,   # Optical center Y
    alpha=2,    # Pixel distance for tangent vector construction
    r_threshold=0.1  # Optional: threshold for filtering erroneous normals
)

# Estimate surface normals from depth map
normals = estimator.estimate_normals(depth_map)

# Create visualization
normal_vis = estimator.create_visualization(normals)

# Get statistics
stats = estimator.get_normal_statistics(normals)
```

### Function-Based Usage

```python
from core.surface_normal_estimator import estimate_surface_normals, visualize_surface_normals

# Estimate normals directly
normals = estimate_surface_normals(
    depth_map, 
    fx=1512.0, 
    fy=1512.0, 
    ox=1080.0, 
    oy=607.0, 
    alpha=2
)

# Create visualization
normal_vis = visualize_surface_normals(normals)
```

### With Masking

```python
# Create a mask for region of interest
mask = np.zeros_like(depth_map, dtype=bool)
mask[100:400, 200:600] = True  # Focus on specific region

# Estimate normals only in masked region
normals = estimator.estimate_normals(depth_map, mask=mask)
```

## Parameters

### Camera Parameters
- `fx, fy`: Focal lengths in pixels (X and Y directions)
- `ox, oy`: Optical center coordinates (principal point)
- `alpha`: Pixel distance for tangent vector construction (default: 2)
- `r_threshold`: Threshold for filtering erroneous normals at depth discontinuities (optional)

### Output
- **Normal Map**: 3D array (H×W×3) with normalized surface normal vectors
- **Visualization**: Colored image showing normal components (RGB = XYZ)
- **Magnitude Map**: Grayscale image showing normal vector magnitudes
- **Statistics**: Dictionary with valid pixel count, mean normal, standard deviation

## Integration with 3D Pose Pipeline

The surface normal estimation is automatically integrated into the depth visualization:

1. **Preview Mode**: Shows surface normal map alongside depth map
2. **Video Processing**: Includes surface normal analysis in depth visualization
3. **Statistics Overlay**: Displays normal statistics on the visualization

### Configuration

Enable/disable surface normal estimation in the main configuration:

```python
# In app_3d_pose_estimation_pipeline.py
ENABLE_SURFACE_NORMALS = True  # Enable surface normal estimation
SURFACE_NORMAL_ALPHA = 2       # Pixel distance for tangent vector construction
SURFACE_NORMAL_R_THRESHOLD = 0.1  # Threshold for filtering erroneous normals
SURFACE_NORMAL_COLOR_MAP = cv2.COLORMAP_PLASMA  # Color map for normal visualization
```

## Algorithm Details

The surface normal estimation uses the following approach:

1. **Depth Gradient Calculation**: Compute depth gradients in four cardinal directions
2. **Tangent Vector Construction**: Use depth differences to construct tangent vectors
3. **Normal Vector Computation**: Calculate normal vectors using closed-form expressions
4. **Boundary Filtering**: Filter out erroneous normals at depth discontinuities
5. **Normalization**: Normalize all valid normal vectors to unit length

### Mathematical Formulation

For each pixel (r, c) with depth value d1:

- Get depth values: d2 (top), d3 (right), d4 (bottom), d5 (left)
- Calculate u1 = (c - ox) / fx, v1 = (r - oy) / fy
- Compute normal components:
  - nx = -α/(4×fy) × (d3 + d5) × (d2 - d4)
  - ny = -α/(4×fx) × (d2 + d4) × (d3 - d5)
  - nz = -u1×nx - v1×ny + α²/(4×fx×fy) × (d2 + d4) × (d3 + d5)

## Testing

Run the test script to verify the module works correctly:

```bash
cd obj_pose
python test_surface_normals.py
```

This will:
- Create synthetic depth maps
- Test both function and class-based approaches
- Verify visualization functionality
- Test with real camera parameters
- Generate test images

## Output Files

The test script generates:
- `test_depth_map.png`: Synthetic depth map
- `test_normal_visualization.png`: Normal vector visualization
- `test_magnitude_map.png`: Normal magnitude map
- `test_real_camera_normals.png`: Test with real camera parameters

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `core` directory is in your Python path
2. **Memory Issues**: For large depth maps, consider downsampling before processing
3. **Invalid Normals**: Check camera parameters and depth map quality
4. **Poor Visualization**: Adjust color maps or normalization parameters

### Performance Tips

- Use appropriate `alpha` values (1-3 typically work well)
- Apply masks to focus on regions of interest
- Consider downsampling for real-time applications
- Use `r_threshold` to filter noisy normals

## Dependencies

- NumPy: For numerical computations
- OpenCV: For image processing and visualization
- No additional dependencies beyond the main pipeline requirements
