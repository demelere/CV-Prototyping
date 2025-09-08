#!/usr/bin/env python3
"""
GLB File Viewer using Rerun
A simple viewer for GLB/GLTF files with interactive controls including:
- Hover detection
- Selection
- Rotation
- Zoom
- Pan

Usage:
    python glb_viewer.py <path_to_glb_file>
"""

import sys
import argparse
import numpy as np
import trimesh
import rerun as rr
from pathlib import Path


class GLBViewer:
    def __init__(self, glb_path: str):
        """Initialize the GLB viewer with a file path."""
        self.glb_path = Path(glb_path)
        self.scene = None
        self.mesh_entities = []
        
        if not self.glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")
            
    def load_glb(self):
        """Load the GLB file using trimesh."""
        try:
            # Load the GLB file
            self.scene = trimesh.load(str(self.glb_path))
            print(f"Successfully loaded GLB file: {self.glb_path}")
            
            # Handle different types of loaded objects
            if hasattr(self.scene, 'geometry'):
                # It's a scene with multiple geometries
                print(f"Loaded scene with {len(self.scene.geometry)} geometries")
            elif hasattr(self.scene, 'vertices'):
                # It's a single mesh
                print(f"Loaded single mesh with {len(self.scene.vertices)} vertices")
            else:
                print(f"Loaded object of type: {type(self.scene)}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load GLB file: {e}")
    
    def log_mesh_to_rerun(self, mesh, entity_path: str, color=None):
        """Log a single mesh to Rerun."""
        try:
            # Get vertices and faces
            vertices = np.array(mesh.vertices, dtype=np.float32)
            
            # Get face indices
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                faces = np.array(mesh.faces, dtype=np.uint32)
            else:
                faces = None
            
            # Try to get colors from the mesh first
            vertex_colors = None
            
            # Extract colors from mesh if available
            if hasattr(mesh, 'visual') and mesh.visual is not None:
                try:
                    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                        # Use vertex colors from the mesh
                        vertex_colors = np.array(mesh.visual.vertex_colors)
                        if vertex_colors.shape[1] == 4:  # RGBA
                            vertex_colors = vertex_colors[:, :3]  # Convert to RGB
                        vertex_colors = vertex_colors.astype(np.float32) / 255.0  # Normalize to [0,1]
                        print(f"Using vertex colors from mesh: {vertex_colors.shape}")
                    elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                        # Convert face colors to vertex colors
                        face_colors = np.array(mesh.visual.face_colors)
                        if face_colors.shape[1] == 4:  # RGBA
                            face_colors = face_colors[:, :3]  # Convert to RGB
                        face_colors = face_colors.astype(np.float32) / 255.0  # Normalize to [0,1]
                        
                        # Map face colors to vertices (simple averaging)
                        vertex_colors = np.zeros((len(vertices), 3), dtype=np.float32)
                        if faces is not None:
                            for i, face in enumerate(faces):
                                if i < len(face_colors):
                                    for vertex_idx in face:
                                        vertex_colors[vertex_idx] = face_colors[i]
                        print(f"Converted face colors to vertex colors: {vertex_colors.shape}")
                except Exception as e:
                    print(f"Warning: Could not extract colors from mesh: {e}")
                    vertex_colors = None
            
            # Use provided color or default if no mesh colors found
            if vertex_colors is None:
                if color is None:
                    color = [0.7, 0.7, 0.9, 1.0]  # Light blue-gray default
                vertex_colors = np.tile(color[:3], (len(vertices), 1))
                print(f"Using default color: {color[:3]}")
            
            # Log the mesh to Rerun
            if faces is not None:
                rr.log(
                    entity_path,
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=faces,
                        vertex_colors=vertex_colors
                    )
                )
            else:
                # If no faces, log as points
                rr.log(
                    entity_path,
                    rr.Points3D(
                        positions=vertices,
                        colors=vertex_colors,
                        radii=0.01
                    )
                )
                
            print(f"Logged mesh '{entity_path}' with {len(vertices)} vertices")
            
        except Exception as e:
            print(f"Error logging mesh {entity_path}: {e}")
    
    def setup_scene_transforms(self):
        """Set up coordinate system and initial view."""
        # Set up the coordinate system (Rerun uses RDF by default)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        
        # Set default camera view (front view, right-side up)
        if hasattr(self.scene, 'bounds'):
            bounds = self.scene.bounds
            center = (bounds[0] + bounds[1]) / 2
            size = np.max(bounds[1] - bounds[0])
            
            # Position camera in front of the scene, looking at center
            camera_distance = size * 2.0  # Move camera back by 2x scene size
            camera_pos = center + np.array([0, 0, camera_distance])  # Front view
            
            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=camera_pos,
                    # Look at center (camera points down negative Z)
                    mat3x3=np.eye(3)  # No rotation needed for front view
                ),
                static=True
            )
        
        # Log scene bounds for better initial view
        if hasattr(self.scene, 'bounds'):
            bounds = self.scene.bounds
            center = (bounds[0] + bounds[1]) / 2
            size = np.max(bounds[1] - bounds[0])
            
            # Log bounding box for reference
            rr.log(
                "world/bounds",
                rr.Boxes3D(
                    centers=[center],
                    sizes=[bounds[1] - bounds[0]],
                    colors=[[1.0, 0.0, 0.0, 0.3]]  # Semi-transparent red
                )
            )
            
            print(f"Scene bounds: {bounds}")
            print(f"Scene center: {center}")
            print(f"Scene size: {size}")
    
    def log_scene_to_rerun(self):
        """Log the entire GLB scene to Rerun."""
        if self.scene is None:
            raise RuntimeError("No scene loaded. Call load_glb() first.")
        
        # Set up scene transforms
        self.setup_scene_transforms()
        
        # Handle different scene types
        if hasattr(self.scene, 'geometry'):
            # Multiple geometries in a scene
            for name, geometry in self.scene.geometry.items():
                entity_path = f"world/scene/{name}"
                
                # Get transform if available
                if hasattr(self.scene, 'graph') and name in self.scene.graph.nodes:
                    # Apply transform if available
                    try:
                        transform = self.scene.graph.get(name)
                        if transform is not None and isinstance(transform, np.ndarray) and transform.shape == (4, 4):
                            # Log transform
                            rr.log(
                                entity_path,
                                rr.Transform3D(
                                    mat3x3=transform[:3, :3],
                                    translation=transform[:3, 3]
                                )
                            )
                    except Exception as e:
                        print(f"Warning: Could not apply transform for {name}: {e}")
                
                # Log the geometry
                self.log_mesh_to_rerun(geometry, entity_path)
                self.mesh_entities.append(entity_path)
                
        elif hasattr(self.scene, 'vertices'):
            # Single mesh
            entity_path = "world/scene/mesh"
            self.log_mesh_to_rerun(self.scene, entity_path)
            self.mesh_entities.append(entity_path)
            
        else:
            print(f"Unknown scene type: {type(self.scene)}")
    
    def run_viewer(self):
        """Initialize Rerun and start the viewer."""
        # Initialize Rerun with the GLB filename as the app ID
        app_id = f"glb_viewer_{self.glb_path.stem}"
        rr.init(app_id, spawn=True)
        
        print(f"Starting Rerun viewer for: {self.glb_path}")
        print("Controls:")
        print("- Left click + drag: Rotate view")
        print("- Right click + drag: Pan view")
        print("- Scroll wheel: Zoom in/out")
        print("- Click on objects: Select")
        print("- Hover over objects: Highlight")
        
        # Load and log the GLB file
        self.load_glb()
        self.log_scene_to_rerun()
        
        print("\nViewer is running. Close the viewer window to exit.")
        print(f"Logged {len(self.mesh_entities)} mesh entities to Rerun")


def main():
    """Main entry point for the GLB viewer."""
    parser = argparse.ArgumentParser(
        description="View GLB files using Rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python glb_viewer.py model.glb
    python glb_viewer.py /path/to/your/model.gltf
        """
    )
    
    parser.add_argument(
        "glb_file",
        type=str,
        help="Path to the GLB or GLTF file to view"
    )
    
    args = parser.parse_args()
    
    try:
        viewer = GLBViewer(args.glb_file)
        viewer.run_viewer()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nViewer interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
