"""
Camera Pose Extractor using VGGT (Visual Geometry Grounded Transformer)

This module provides functionality to extract camera extrinsic and intrinsic parameters
from images using the VGGT model. It's designed to work with fixed camera scenarios
where the camera pose remains constant throughout the scene.

For TIG welding scenarios where the camera is fixed and the torch moves laterally,
this module can extract camera pose from a single representative frame.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import logging
from scipy.spatial.transform import Rotation

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError as e:
    logging.error(f"VGGT not installed or not found: {e}")
    raise ImportError("VGGT is required for camera pose extraction. Please install it first.")


class CameraPoseExtractor:
    """
    Extracts camera pose (extrinsic and intrinsic parameters) from images using VGGT.
    
    This class handles loading the VGGT model, preprocessing images, and extracting
    camera parameters suitable for 3D reconstruction tasks. Optimized for fixed camera
    scenarios where pose is extracted once from the first frame and cached.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/VGGT-1B",
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize the camera pose extractor.
        
        Args:
            model_name: Hugging Face model name for VGGT
            device: Device to run inference on (auto-detected if None)
            dtype: Data type for inference (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set appropriate dtype based on GPU capability
        if dtype is None:
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = dtype
            
        self.model_name = model_name
        self.model = None
        
        # Caching for fixed camera scenarios
        self._cached_extrinsic = None
        self._cached_intrinsic = None
        self._cached_source_image = None
        
        self._load_model()
        
        logging.info(f"CameraPoseExtractor initialized on {self.device} with dtype {self.dtype}")
    
    def _load_model(self):
        """Load the VGGT model from Hugging Face."""
        try:
            logging.info(f"Loading VGGT model: {self.model_name}")
            self.model = VGGT.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logging.info("VGGT model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load VGGT model: {e}")
            raise
    
    def extract_camera_pose(self, 
                           image_paths: Union[str, list],
                           return_confidence: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract camera extrinsic and intrinsic parameters from images.
        
        Args:
            image_paths: Path to a single image or list of image paths
            return_confidence: Whether to return confidence scores (not implemented yet)
            
        Returns:
            Tuple of (extrinsic_matrices, intrinsic_matrices)
            - extrinsic_matrices: (N, 4, 4) camera-to-world transformation matrices
            - intrinsic_matrices: (N, 3, 3) camera intrinsic matrices
            
        Note:
            The extrinsic matrices follow OpenCV convention (camera from world).
            For fixed camera scenarios, you typically only need the first matrix.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        # Validate image paths
        for path in image_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Image not found: {path}")
        
        try:
            # Load and preprocess images
            logging.info(f"Processing {len(image_paths)} images")
            images = load_and_preprocess_images(image_paths).to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    # Add batch dimension if needed
                    if images.dim() == 4:  # Single scene
                        images = images[None]
                    
                    # Get aggregated tokens
                    aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                    
                    # Predict camera parameters
                    pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                    
                    # Convert to extrinsic and intrinsic matrices
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_enc, images.shape[-2:]
                    )
            
            # Convert to numpy arrays
            extrinsic_np = extrinsic.cpu().numpy()
            intrinsic_np = intrinsic.cpu().numpy()
            
            logging.info(f"Successfully extracted poses for {len(image_paths)} images")
            logging.info(f"Extrinsic shape: {extrinsic_np.shape}")
            logging.info(f"Intrinsic shape: {intrinsic_np.shape}")
            
            return extrinsic_np, intrinsic_np
            
        except Exception as e:
            logging.error(f"Failed to extract camera pose: {e}")
            raise
    
    def extract_single_camera_pose(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract camera pose from a single image (convenience method).
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (extrinsic_matrix, intrinsic_matrix)
            - extrinsic_matrix: (4, 4) camera-to-world transformation matrix
            - intrinsic_matrix: (3, 3) camera intrinsic matrix
        """
        extrinsic_batch, intrinsic_batch = self.extract_camera_pose([image_path])
        
        # Return single matrices (remove batch dimension)
        return extrinsic_batch[0, 0], intrinsic_batch[0, 0]
    
    def get_camera_parameters_dict(self, 
                                  image_path: str) -> dict:
        """
        Extract camera parameters and return as a dictionary with named components.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing:
            - 'extrinsic': (4, 4) extrinsic matrix
            - 'intrinsic': (3, 3) intrinsic matrix
            - 'rotation': (3, 3) rotation matrix from extrinsic
            - 'translation': (3,) translation vector from extrinsic
            - 'fx', 'fy': focal lengths
            - 'cx', 'cy': principal point coordinates
        """
        extrinsic, intrinsic = self.extract_single_camera_pose(image_path)
        
        # Extract rotation and translation from extrinsic matrix
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        
        # Extract intrinsic parameters
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        return {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'rotation': rotation,
            'translation': translation,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
    
    def save_camera_parameters(self, 
                              image_path: str, 
                              output_path: str):
        """
        Extract camera parameters and save to a file.
        
        Args:
            image_path: Path to the image
            output_path: Path to save the camera parameters (.npz format)
        """
        params = self.get_camera_parameters_dict(image_path)
        
        np.savez(output_path,
                extrinsic=params['extrinsic'],
                intrinsic=params['intrinsic'],
                rotation=params['rotation'],
                translation=params['translation'],
                fx=params['fx'],
                fy=params['fy'],
                cx=params['cx'],
                cy=params['cy'])
        
        logging.info(f"Camera parameters saved to: {output_path}")

    def extract_or_get_cached_pose(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract camera pose from image or return cached result for fixed camera scenarios.
        
        For fixed camera setups (like TIG welding), this method extracts pose from the first
        image and caches it. Subsequent calls return the cached result without re-running VGGT.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (extrinsic_matrix, intrinsic_matrix)
        """
        # Check if we have cached results
        if (self._cached_extrinsic is not None and 
            self._cached_intrinsic is not None and
            self._cached_source_image is not None):
            
            logging.info(f"Using cached camera pose from: {self._cached_source_image}")
            return self._cached_extrinsic, self._cached_intrinsic
        
        # Extract pose for the first time
        logging.info(f"Extracting camera pose from first frame: {image_path}")
        extrinsic, intrinsic = self.extract_single_camera_pose(image_path)
        
        # Cache the results
        self._cached_extrinsic = extrinsic
        self._cached_intrinsic = intrinsic
        self._cached_source_image = image_path
        
        logging.info("Camera pose cached for subsequent frames")
        return extrinsic, intrinsic
    
    def get_intrinsics_with_fallback(self, 
                                   image_path: str, 
                                   metadata_intrinsics: Optional[dict] = None) -> Tuple[float, float, float, float]:
        """
        Get camera intrinsics with fallback to VGGT if metadata is unavailable.
        
        Args:
            image_path: Path to the image
            metadata_intrinsics: Dict with 'fx', 'fy', 'cx', 'cy' from image metadata (optional)
            
        Returns:
            Tuple of (fx, fy, cx, cy) focal lengths and principal point
        """
        if metadata_intrinsics and all(k in metadata_intrinsics for k in ['fx', 'fy', 'cx', 'cy']):
            logging.info("Using camera intrinsics from image metadata")
            return (metadata_intrinsics['fx'], metadata_intrinsics['fy'], 
                   metadata_intrinsics['cx'], metadata_intrinsics['cy'])
        
        # Fallback to VGGT
        logging.info("No metadata intrinsics found, using VGGT as fallback")
        _, intrinsic = self.extract_or_get_cached_pose(image_path)
        
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1] 
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        return fx, fy, cx, cy
    
    def apply_gravity_prior(self, extrinsic_matrix: np.ndarray, 
                           forward_pitch_degrees: float = 15.0) -> np.ndarray:
        """
        Apply gravity prior to camera pose by constraining orientation.
        
        Assumes camera is upright with no roll or yaw, and a small forward pitch.
        This fixes the "up" direction and makes surface normals more realistic.
        
        Args:
            extrinsic_matrix: Original camera extrinsic matrix (4x4 or 3x4)
            forward_pitch_degrees: Forward pitch angle in degrees (default: 15°)
            
        Returns:
            Gravity-constrained extrinsic matrix (4x4)
        """
        # Ensure we have a 4x4 matrix
        if extrinsic_matrix.shape == (3, 4):
            # Convert 3x4 to 4x4
            extrinsic_4x4 = np.eye(4)
            extrinsic_4x4[:3, :] = extrinsic_matrix
        else:
            extrinsic_4x4 = extrinsic_matrix.copy()
        
        # Extract translation (preserve original position)
        translation = extrinsic_4x4[:3, 3]
        
        # Create gravity-constrained rotation matrix
        # World coordinate system: X-right, Y-up, Z-backward (gravity points down in -Y)
        # Camera coordinate system: X-right, Y-down, Z-forward
        
        # Convert pitch to radians
        pitch_rad = np.radians(forward_pitch_degrees)
        
        # Create camera-to-world transformation matrix
        # This transforms from camera coordinates to world coordinates
        # Camera is upright with 15° forward pitch (looking slightly down)
        
        # Step 1: Create base transformation from camera to world coordinates
        # Camera: X-right, Y-down, Z-forward
        # World: X-right, Y-up, Z-backward
        # So we need to flip Y and Z axes
        
        # Base transformation: flip Y and Z axes (camera -> world)
        base_transform = np.array([
            [1,  0,  0],  # X stays the same
            [0, -1,  0],  # Y flips (down -> up)
            [0,  0, -1]   # Z flips (forward -> backward)
        ])
        
        # Apply forward pitch rotation around X-axis
        # For camera-to-world, we need the inverse of the camera's pitch
        pitch_rotation = Rotation.from_euler('x', -pitch_rad).as_matrix()
        
        # Combine: camera -> world transformation
        camera_to_world_rotation = base_transform @ pitch_rotation
        
        # Create new extrinsic matrix with gravity prior
        gravity_extrinsic = np.eye(4)
        gravity_extrinsic[:3, :3] = camera_to_world_rotation
        gravity_extrinsic[:3, 3] = translation
        
        logging.info(f"Applied gravity prior: forward pitch = {forward_pitch_degrees}°")
        logging.info(f"Original translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
        
        return gravity_extrinsic
    
    def extract_or_get_cached_pose_with_gravity_prior(self, 
                                                     image_path: str,
                                                     forward_pitch_degrees: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract camera pose with gravity prior applied.
        
        Args:
            image_path: Path to the image
            forward_pitch_degrees: Forward pitch angle in degrees (default: 15°)
            
        Returns:
            Tuple of (gravity_constrained_extrinsic_matrix, intrinsic_matrix)
        """
        # Get original pose
        extrinsic, intrinsic = self.extract_or_get_cached_pose(image_path)
        
        # Apply gravity prior
        gravity_extrinsic = self.apply_gravity_prior(extrinsic, forward_pitch_degrees)
        
        # Cache the gravity-constrained result
        self._cached_extrinsic = gravity_extrinsic
        
        return gravity_extrinsic, intrinsic
    
    def reset_cache(self):
        """Reset cached camera pose (useful when switching to different camera setup)."""
        self._cached_extrinsic = None
        self._cached_intrinsic = None
        self._cached_source_image = None
        logging.info("Camera pose cache reset")


def load_camera_parameters(file_path: str) -> dict:
    """
    Load camera parameters from a saved .npz file.
    
    Args:
        file_path: Path to the .npz file
        
    Returns:
        Dictionary containing camera parameters
    """
    data = np.load(file_path)
    return {key: data[key] for key in data.files}


# Convenience function for quick usage
def extract_camera_pose_from_image(image_path: str, 
                                  model_name: str = "facebook/VGGT-1B") -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick function to extract camera pose from a single image.
    
    Args:
        image_path: Path to the image
        model_name: VGGT model name
        
    Returns:
        Tuple of (extrinsic_matrix, intrinsic_matrix)
    """
    extractor = CameraPoseExtractor(model_name=model_name)
    return extractor.extract_single_camera_pose(image_path)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample image (replace with actual path)
    sample_image = "sample_image.jpg"
    
    if Path(sample_image).exists():
        try:
            extractor = CameraPoseExtractor()
            params = extractor.get_camera_parameters_dict(sample_image)
            
            print("Camera Parameters:")
            print(f"Focal lengths: fx={params['fx']:.2f}, fy={params['fy']:.2f}")
            print(f"Principal point: cx={params['cx']:.2f}, cy={params['cy']:.2f}")
            print(f"Translation: {params['translation']}")
            print("Rotation matrix:")
            print(params['rotation'])
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Sample image not found: {sample_image}")
        print("Please provide a valid image path to test the camera pose extraction.")