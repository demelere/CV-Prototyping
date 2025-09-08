# GLB File Viewer

A simple 3D model viewer for GLB/GLTF files built with [Rerun](https://rerun.io/). This viewer provides interactive controls for exploring 3D models including hover detection, selection, rotation, zoom, and pan functionality.

## Features

- **Load GLB/GLTF Files**: Support for standard 3D model formats
- **Interactive Controls**:
  - **Rotate**: Left click + drag to rotate the view
  - **Pan**: Right click + drag to pan around the scene
  - **Zoom**: Use scroll wheel to zoom in/out
  - **Select**: Click on objects to select them
  - **Hover**: Hover over objects for highlighting
- **Multi-geometry Support**: Handles both single meshes and complex scenes
- **Automatic Scene Setup**: Proper coordinate system and bounding box visualization

## Installation

1. Navigate to the viewer directory:
```bash
cd viewer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, you can install dependencies manually:
```bash
pip install rerun-sdk trimesh numpy pillow
```

## Usage

### Basic Usage

```bash
python glb_viewer.py <path_to_glb_file>
```

### Examples

```bash
# View a GLB file
python glb_viewer.py model.glb

# View a GLTF file
python glb_viewer.py /path/to/your/model.gltf
```

### Command Line Options

```bash
python glb_viewer.py --help
```

## Interactive Controls

Once the viewer opens, you can interact with the 3D scene using the following controls:

| Action | Control |
|--------|---------|
| **Rotate View** | Left click + drag |
| **Pan View** | Right click + drag |
| **Zoom** | Mouse scroll wheel |
| **Select Object** | Left click on object |
| **Reset View** | Double-click in empty space |

## Supported File Formats

- `.glb` - Binary glTF files
- `.gltf` - Text-based glTF files (with associated assets)

## Technical Details

### Dependencies

- **rerun-sdk**: For 3D visualization and interaction
- **trimesh**: For loading and processing 3D models
- **numpy**: For numerical operations
- **pillow**: For image processing (textures)

### Scene Structure

The viewer organizes the scene hierarchy as follows:
```
world/                 # Root coordinate system
├── bounds/           # Scene bounding box (red wireframe)
└── scene/            # Loaded 3D models
    ├── mesh1         # Individual meshes
    ├── mesh2
    └── ...
```

### Coordinate System

The viewer uses a right-handed coordinate system with Y-up orientation, which is standard for most 3D modeling applications.

## Troubleshooting

### Common Issues

1. **"GLB file not found"**
   - Check that the file path is correct
   - Ensure the file exists and has read permissions

2. **"Failed to load GLB file"**
   - Verify the file is a valid GLB/GLTF format
   - Check that the file is not corrupted

3. **Viewer window doesn't open**
   - Ensure you have a display available (not running in headless mode)
   - Check that Rerun is properly installed: `pip list | grep rerun`

4. **Performance issues with large models**
   - Large models (>1M vertices) may load slowly
   - Consider using model optimization tools to reduce complexity

### Getting Help

If you encounter issues:
1. Check the [Rerun documentation](https://rerun.io/docs)
2. Open an issue on the [Rerun GitHub repository](https://github.com/rerun-io/rerun)
3. Join the [Rerun Discord server](https://discord.gg/rerun)

## Examples

To test the viewer, you can download sample GLB files from:
- [glTF Sample Models](https://github.com/KhronosGroup/glTF-Sample-Models)
- [Sketchfab](https://sketchfab.com) (free models)
- [Google Poly](https://poly.pizza/)

## License

This project follows the same license as the dependencies it uses. Please check the individual package licenses for more information.
