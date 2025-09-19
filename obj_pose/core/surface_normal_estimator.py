import numpy as np

def calculate_alpha_from_roi(roi_width, roi_height, method='min'):
    """
    Calculate alpha parameter based on ROI dimensions.
    
    Args:
        roi_width (int): Width of the ROI in pixels
        roi_height (int): Height of the ROI in pixels
        method (str): Method to calculate alpha:
            - 'min': Use minimum of width and height (default)
            - 'max': Use maximum of width and height
            - 'mean': Use average of width and height
            - 'width': Use width only
            - 'height': Use height only
    
    Returns:
        int: Calculated alpha value
    """
    if method == 'min':
        base_alpha = min(roi_width, roi_height)
    elif method == 'max':
        base_alpha = max(roi_width, roi_height)
    elif method == 'mean':
        base_alpha = int((roi_width + roi_height) / 2)
    elif method == 'width':
        base_alpha = roi_width
    elif method == 'height':
        base_alpha = roi_height
    else:
        raise ValueError(f"Unknown method: {method}. Use 'min', 'max', 'mean', 'width', or 'height'")
    
    # Standard alpha scaling
    alpha = max(10, min(base_alpha // 8, 60))  # Scale to 12.5% with max of 60
    
    # Additional check for very small ROIs
    if base_alpha < 100:
        alpha = max(alpha, 15)  # Minimum alpha for small ROIs
    
    print(f"DEBUG: Calculated alpha from ROI: {alpha} (method: {method}, ROI: {roi_width}x{roi_height}, base: {base_alpha})")
    return alpha

def estimate_surface_normals(depth_map, fx, fy, ox, oy, alpha, mask=None, r_threshold=None, roi_coordinates=None, roi_alpha_method='min'):
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
        roi_coordinates (tuple, optional): ROI coordinates (x1, y1, x2, y2) to calculate alpha from.
                                          If provided, this will override the alpha parameter.
        roi_alpha_method (str, optional): Method to calculate alpha from ROI dimensions.
                                         Options: 'min', 'max', 'mean', 'width', 'height'.
                                         Defaults to 'min'.

    Returns:
        np.ndarray: A 3D array (HxWx3) representing the surface normal map.
    """
    print(f"DEBUG: Surface normal estimation input analysis:")
    print(f"  - Depth map shape: {depth_map.shape}")
    print(f"  - Depth map data type: {depth_map.dtype}")
    print(f"  - Depth map range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    print(f"  - Camera parameters: fx={fx}, fy={fy}, ox={ox}, oy={oy}")
    print(f"  - Alpha parameter (initial): {alpha}")
    print(f"  - Mask provided: {mask is not None}")
    if mask is not None:
        print(f"  - Mask shape: {mask.shape}")
        print(f"  - Mask sum: {mask.sum()}")
    
    # Calculate alpha from ROI if provided
    if roi_coordinates is not None and len(roi_coordinates) == 4:
        x1, y1, x2, y2 = roi_coordinates
        roi_width = x2 - x1
        roi_height = y2 - y1
        alpha = calculate_alpha_from_roi(roi_width, roi_height, roi_alpha_method)
        print(f"DEBUG: Using alpha calculated from ROI: {alpha}")
    
    height, width = depth_map.shape
    normals = np.zeros((height, width, 3), dtype=np.float64)

    # If a mask is provided, use it to select the pixels to process
    if mask is not None:
        if mask.shape != depth_map.shape:
            raise ValueError("Mask must have the same dimensions as the depth map.")
        processed_pixels = np.where(mask > 0)
        print(f"DEBUG: Using mask - processing {len(processed_pixels[0])} pixels")
    else:
        processed_pixels = np.where(np.ones_like(depth_map))
        print(f"DEBUG: No mask - processing all {len(processed_pixels[0])} pixels")

    # Sample analysis for debugging
    sample_count = 0
    max_samples = 5
    sample_data = []
    
    print(f"DEBUG: Starting normal calculation for {len(processed_pixels[0])} pixels...")

    # Count boundary rejections for debugging
    boundary_rejected = 0
    processed_count = 0
    
    # Iterate over the valid pixels
    for r, c in zip(*processed_pixels):
        # Check if the neighbors are within bounds
        if not (0 <= r - alpha and r + alpha < height and 0 <= c - alpha and c + alpha < width):
            boundary_rejected += 1
            continue

        processed_count += 1
        
        # Get depth values for the four cardinal directions
        d1 = depth_map[r, c] # Center point, used for u, v calculations
        d2 = depth_map[r - alpha, c]  # top
        d3 = depth_map[r, c + alpha]  # right
        d4 = depth_map[r + alpha, c]  # bottom
        d5 = depth_map[r, c - alpha]  # left

        # Sample analysis for debugging
        if sample_count < max_samples:
            sample_data.append({
                'position': (r, c),
                'depths': [d1, d2, d3, d4, d5],
                'depth_diffs': [d2-d4, d3-d5, d2+d4, d3+d5]
            })
            sample_count += 1

        # The paper's formulation for u,v values is based on the pixel coordinates
        # and the optical center.
        u1 = (c - ox) / fx
        v1 = (r - oy) / fy
        
        # Calculate normal vector components using the closed-form expressions (Equations 10, 11, and 12)
        # For quantized depth data, we need to scale the gradients appropriately
        
        # Calculate depth gradients
        grad_x = (d3 - d5) / (2 * alpha)  # Right - left
        grad_y = (d2 - d4) / (2 * alpha)  # Top - bottom
        
        # Original formulation (modified for better numerical stability)
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
        # Note: For table surface visualization, we want normals pointing toward camera
        # The paper's formulation gives normals pointing away from camera (positive Z)
        # So we flip the Z component for proper visualization
        normals[r, c, 0] = nx
        normals[r, c, 1] = ny
        normals[r, c, 2] = -nz  # Flip Z to point toward camera

    # Print sample analysis
    if sample_data:
        print(f"DEBUG: Sample normal calculation analysis:")
        for i, sample in enumerate(sample_data):
            r, c = sample['position']
            depths = sample['depths']
            diffs = sample['depth_diffs']
            u1 = (c - ox) / fx
            v1 = (r - oy) / fy
            
            # Recalculate normal for this sample
            nx = -alpha / (4 * fy) * (depths[2] + depths[4]) * (depths[1] - depths[3])
            ny = -alpha / (4 * fx) * (depths[1] + depths[3]) * (depths[2] - depths[4])
            nz = -u1 * nx - v1 * ny + (alpha**2) / (4 * fx * fy) * (depths[1] + depths[3]) * (depths[2] + depths[4])
            
            print(f"  Sample {i+1} at ({r}, {c}):")
            print(f"    Depths: center={depths[0]:.3f}, top={depths[1]:.3f}, right={depths[2]:.3f}, bottom={depths[3]:.3f}, left={depths[4]:.3f}")
            print(f"    Depth diffs: top-bottom={diffs[0]:.3f}, right-left={diffs[1]:.3f}, top+bottom={diffs[2]:.3f}, right+left={diffs[3]:.3f}")
            print(f"    u1={u1:.3f}, v1={v1:.3f}")
            print(f"    Raw normal: [{nx:.6f}, {ny:.6f}, {nz:.6f}]")
            print(f"    Normal magnitude: {np.sqrt(nx**2 + ny**2 + nz**2):.6f}")

    # Print processing statistics
    print(f"DEBUG: Processing statistics:")
    print(f"  - Total pixels to check: {len(processed_pixels[0])}")
    print(f"  - Boundary rejected: {boundary_rejected}")
    print(f"  - Actually processed: {processed_count}")
    print(f"  - Boundary rejection rate: {boundary_rejected / len(processed_pixels[0]) * 100:.2f}%")
    
    # Normalize the valid normal vectors to unit length
    valid_pixels = normals.any(axis=2)
    if valid_pixels.any():
        normal_magnitudes = np.linalg.norm(normals[valid_pixels], axis=1)
        normals[valid_pixels] /= normal_magnitudes[:, np.newaxis]

    # Final analysis
    print(f"DEBUG: Surface normal estimation output analysis:")
    print(f"  - Valid pixels: {valid_pixels.sum()} out of {valid_pixels.size}")
    print(f"  - Valid percentage: {valid_pixels.sum() / valid_pixels.size * 100:.2f}%")
    
    if valid_pixels.any():
        valid_normals = normals[valid_pixels]
        print(f"  - Normal X range: [{valid_normals[:, 0].min():.6f}, {valid_normals[:, 0].max():.6f}]")
        print(f"  - Normal Y range: [{valid_normals[:, 1].min():.6f}, {valid_normals[:, 1].max():.6f}]")
        print(f"  - Normal Z range: [{valid_normals[:, 2].min():.6f}, {valid_normals[:, 2].max():.6f}]")
        print(f"  - Normal X std: {valid_normals[:, 0].std():.6f}")
        print(f"  - Normal Y std: {valid_normals[:, 1].std():.6f}")
        print(f"  - Normal Z std: {valid_normals[:, 2].std():.6f}")
        
        # Check for vertical normals (indicating flat surfaces)
        vertical_threshold = 0.99
        vertical_count = np.sum(valid_normals[:, 2] > vertical_threshold)
        print(f"  - Nearly vertical normals (Z > {vertical_threshold}): {vertical_count} ({vertical_count/len(valid_normals)*100:.1f}%)")
    
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
    
    def __init__(self, fx, fy, ox, oy, alpha=100, r_threshold=None, roi_alpha_method='min'):
        """
        Initialize the surface normal estimator.
        
        Args:
            fx (float): Focal length in x-direction.
            fy (float): Focal length in y-direction.
            ox (float): Optical center in x-direction.
            oy (float): Optical center in y-direction.
            alpha (int): The pixel distance value for tangent vector construction.
            r_threshold (float, optional): Threshold for filtering erroneous normals.
            roi_alpha_method (str, optional): Method to calculate alpha from ROI dimensions.
                                             Options: 'min', 'max', 'mean', 'width', 'height'.
                                             Defaults to 'min'.
        """
        self.fx = fx
        self.fy = fy
        self.ox = ox
        self.oy = oy
        self.alpha = alpha
        self.r_threshold = r_threshold
        self.roi_alpha_method = roi_alpha_method
    
    def estimate_normals(self, depth_map, mask=None, roi_coordinates=None):
        """
        Estimate surface normals from a depth map.
        
        Args:
            depth_map (np.ndarray): The input depth map (HxW).
            mask (np.ndarray, optional): Mask to apply to the depth map.
            roi_coordinates (tuple, optional): ROI coordinates (x1, y1, x2, y2) to calculate alpha from.
                                             If provided, this will override the default alpha parameter.
            
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
            self.r_threshold,
            roi_coordinates,
            self.roi_alpha_method
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
