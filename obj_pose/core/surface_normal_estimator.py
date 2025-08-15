import numpy as np

def estimate_surface_normals(depth_map, fx, fy, ox, oy, alpha, mask=None, r_threshold=None):
    """
    Estimates surface normals from a depth map using the methodology from the paper.

    Args:
        depth_map (np.ndarray): The input depth map (HxW).
        fx (float): Focal length in x-direction.
        fy (float): Focal length in y-direction.
        ox (float): Optical center in x-direction.
        oy (float): Optical center in y-direction.
        alpha (int): The pixel distance value for tangent vector construction.
        mask (np.ndarray, optional): A boolean or integer mask (HxW) to apply to the depth map.
                                     Normals will only be computed for pixels where the mask is True or > 0.
                                     Defaults to None, which computes normals for the entire map.
        r_threshold (float, optional): Threshold for the r-component (magnitude) of the normal vector
                                       to filter out erroneous normals at depth discontinuities.
                                       Defaults to None, which disables filtering.

    Returns:
        np.ndarray: A 3D array (HxWx3) representing the surface normal map.
    """
    height, width = depth_map.shape
    normals = np.zeros((height, width, 3), dtype=np.float64)

    # If a mask is provided, use it to select the pixels to process
    if mask is not None:
        if mask.shape != depth_map.shape:
            raise ValueError("Mask must have the same dimensions as the depth map.")
        processed_pixels = np.where(mask > 0)
    else:
        processed_pixels = np.where(np.ones_like(depth_map))

    # Iterate over the valid pixels
    for r, c in zip(*processed_pixels):
        # Check if the neighbors are within bounds
        if not (0 <= r - alpha and r + alpha < height and 0 <= c - alpha and c + alpha < width):
            continue

        # Get depth values for the four cardinal directions
        d1 = depth_map[r, c] # Center point, used for u, v calculations
        d2 = depth_map[r - alpha, c]  # top
        d3 = depth_map[r, c + alpha]  # right
        d4 = depth_map[r + alpha, c]  # bottom
        d5 = depth_map[r, c - alpha]  # left

        # The paper's formulation for u,v values is based on the pixel coordinates
        # and the optical center.
        u1 = (c - ox) / fx
        v1 = (r - oy) / fy
        
        # Calculate normal vector components using the closed-form expressions (Equations 10, 11, and 12)
        nx = -alpha / (4 * fy) * (d3 + d5) * (d2 - d4)
        ny = -alpha / (4 * fx) * (d2 + d4) * (d3 - d5)
        nz = -u1 * nx - v1 * ny + (alpha**2) / (4 * fx * fy) * (d2 + d4) * (d3 + d5)

        # Apply boundary-aware filtering if a threshold is provided (based on equation 13 and section 3.2)
        if r_threshold is not None:
            normal_magnitude = np.sqrt(nx**2 + ny**2 + nz**2)
            if normal_magnitude > r_threshold:
                # Discard erroneous normals by setting them to zero
                nx, ny, nz = 0.0, 0.0, 0.0

        # Store the computed normal vector
        normals[r, c, 0] = nx
        normals[r, c, 1] = ny
        normals[r, c, 2] = nz

    # Normalize the valid normal vectors to unit length
    valid_pixels = normals.any(axis=2)
    if valid_pixels.any():
        normal_magnitudes = np.linalg.norm(normals[valid_pixels], axis=1)
        normals[valid_pixels] /= normal_magnitudes[:, np.newaxis]

    return normals

def apply_mask(depth_map, mask):
    """
    Applies a mask to a depth map, returning only the region of interest.
    Pixels outside the mask are set to zero.

    Args:
        depth_map (np.ndarray): The original depth map.
        mask (np.ndarray): A boolean or integer mask to apply.

    Returns:
        np.ndarray: The masked depth map.
    """
    if mask.shape != depth_map.shape:
        raise ValueError("Mask must have the same dimensions as the depth map.")
    return np.where(mask, depth_map, 0)

def create_inverse_mask(mask):
    """
    Creates an inverse mask from a given mask.

    Args:
        mask (np.ndarray): A boolean or integer mask.

    Returns:
        np.ndarray: The inverted mask.
    """
    return ~mask

def visualize_surface_normals(normals, output_size=None):
    """
    Visualize surface normals as a colored image.
    
    Args:
        normals (np.ndarray): Surface normal map (HxWx3).
        output_size (tuple, optional): Output size (width, height) for resizing.
        
    Returns:
        np.ndarray: Colored visualization of surface normals.
    """
    # Convert normals to RGB visualization
    # Map x, y, z components to R, G, B channels
    # Normalize to 0-1 range and scale to 0-255
    normal_vis = np.zeros_like(normals, dtype=np.uint8)
    
    # Normalize each component to 0-1 range
    for i in range(3):
        component = normals[:, :, i]
        if component.max() != component.min():
            component_norm = (component - component.min()) / (component.max() - component.min())
        else:
            component_norm = component
        normal_vis[:, :, i] = (component_norm * 255).astype(np.uint8)
    
    # Resize if requested
    if output_size is not None:
        import cv2
        normal_vis = cv2.resize(normal_vis, output_size)
    
    return normal_vis

def create_normal_magnitude_map(normals):
    """
    Create a magnitude map from surface normals.
    
    Args:
        normals (np.ndarray): Surface normal map (HxWx3).
        
    Returns:
        np.ndarray: Magnitude map (HxW).
    """
    # Calculate magnitude of each normal vector
    magnitude = np.linalg.norm(normals, axis=2)
    
    # Normalize to 0-255 range
    if magnitude.max() != magnitude.min():
        magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    else:
        magnitude_norm = magnitude
    
    return (magnitude_norm * 255).astype(np.uint8)

class SurfaceNormalEstimator:
    """
    A class to handle surface normal estimation from depth maps with camera parameters.
    """
    
    def __init__(self, fx, fy, ox, oy, alpha=1, r_threshold=None):
        """
        Initialize the surface normal estimator.
        
        Args:
            fx (float): Focal length in x-direction.
            fy (float): Focal length in y-direction.
            ox (float): Optical center in x-direction.
            oy (float): Optical center in y-direction.
            alpha (int): The pixel distance value for tangent vector construction.
            r_threshold (float, optional): Threshold for filtering erroneous normals.
        """
        self.fx = fx
        self.fy = fy
        self.ox = ox
        self.oy = oy
        self.alpha = alpha
        self.r_threshold = r_threshold
    
    def estimate_normals(self, depth_map, mask=None):
        """
        Estimate surface normals from a depth map.
        
        Args:
            depth_map (np.ndarray): The input depth map (HxW).
            mask (np.ndarray, optional): Mask to apply to the depth map.
            
        Returns:
            np.ndarray: Surface normal map (HxWx3).
        """
        return estimate_surface_normals(
            depth_map, 
            self.fx, 
            self.fy, 
            self.ox, 
            self.oy, 
            self.alpha, 
            mask, 
            self.r_threshold
        )
    
    def create_visualization(self, normals, output_size=None):
        """
        Create a visualization of the surface normals.
        
        Args:
            normals (np.ndarray): Surface normal map (HxWx3).
            output_size (tuple, optional): Output size for resizing.
            
        Returns:
            np.ndarray: Colored visualization.
        """
        return visualize_surface_normals(normals, output_size)
    
    def get_normal_statistics(self, normals):
        """
        Get statistics about the surface normals.
        
        Args:
            normals (np.ndarray): Surface normal map (HxWx3).
            
        Returns:
            dict: Statistics about the normals.
        """
        # Calculate statistics for valid normals
        valid_pixels = normals.any(axis=2)
        
        if not valid_pixels.any():
            return {
                'valid_pixels': 0,
                'total_pixels': normals.shape[0] * normals.shape[1],
                'valid_percentage': 0.0,
                'mean_normal': [0, 0, 0],
                'std_normal': [0, 0, 0]
            }
        
        valid_normals = normals[valid_pixels]
        
        stats = {
            'valid_pixels': int(valid_pixels.sum()),
            'total_pixels': normals.shape[0] * normals.shape[1],
            'valid_percentage': float(valid_pixels.sum() / valid_pixels.size * 100),
            'mean_normal': valid_normals.mean(axis=0).tolist(),
            'std_normal': valid_normals.std(axis=0).tolist()
        }
        
        return stats
