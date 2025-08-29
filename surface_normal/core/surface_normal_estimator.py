#!/usr/bin/env python3
"""
Surface Normal Estimation Module
Implements grid-based surface normal estimation from depth maps with arrow visualization.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict


class GridBasedSurfaceNormalEstimator:
    """
    Grid-based surface normal estimator that computes normals for subregions
    and visualizes them as upward-pointing arrows from the surface.
    """
    
    def __init__(self, 
                 fx: float = 1512.0, 
                 fy: float = 1512.0, 
                 cx: float = 1080.0, 
                 cy: float = 607.0,
                 grid_size: int = 32,
                 arrow_length: float = 30.0,
                 arrow_thickness: int = 2):
        """
        Initialize the grid-based surface normal estimator.
        
        Args:
            fx, fy: Camera focal lengths in pixels
            cx, cy: Camera principal point coordinates
            grid_size: Size of each grid cell in pixels
            arrow_length: Length of normal arrows in pixels
            arrow_thickness: Thickness of arrow lines
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.grid_size = grid_size
        self.arrow_length = arrow_length
        self.arrow_thickness = arrow_thickness
        
        print(f"Initialized GridBasedSurfaceNormalEstimator:")
        print(f"  Camera params: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print(f"  Grid size: {grid_size}px")
        print(f"  Arrow length: {arrow_length}px, thickness: {arrow_thickness}")
    
    def compute_surface_normal_at_point(self, 
                                      depth_map: np.ndarray, 
                                      x: int, 
                                      y: int, 
                                      neighborhood_size: int = 5) -> Optional[np.ndarray]:
        """
        Compute surface normal at a specific point using local neighborhood.
        
        Args:
            depth_map: Input depth map
            x, y: Pixel coordinates
            neighborhood_size: Size of neighborhood for gradient computation
            
        Returns:
            3D surface normal vector or None if computation fails
        """
        h, w = depth_map.shape
        
        # Ensure we have enough boundary space
        if (x < neighborhood_size or x >= w - neighborhood_size or 
            y < neighborhood_size or y >= h - neighborhood_size):
            return None
        
        try:
            # Get depth value at center point
            depth_center = depth_map[y, x]
            if depth_center <= 0:
                return None
            
            # Compute gradients using finite differences
            # X-gradient (horizontal)
            depth_left = depth_map[y, x - neighborhood_size]
            depth_right = depth_map[y, x + neighborhood_size]
            
            # Y-gradient (vertical) 
            depth_up = depth_map[y - neighborhood_size, x]
            depth_down = depth_map[y + neighborhood_size, x]
            
            # Calculate gradients in depth
            grad_x = (depth_right - depth_left) / (2 * neighborhood_size)
            grad_y = (depth_down - depth_up) / (2 * neighborhood_size)
            
            # Convert pixel coordinates to normalized camera coordinates
            u = (x - self.cx) / self.fx
            v = (y - self.cy) / self.fy
            
            # Compute 3D gradients
            # For a point P = (X, Y, Z) where Z is depth:
            # X = u * Z, Y = v * Z, so:
            # dX/dx = Z/fx + u * dZ/dx
            # dY/dy = Z/fy + v * dZ/dy
            
            grad_3d_x = depth_center / self.fx + u * grad_x
            grad_3d_y = depth_center / self.fy + v * grad_y
            
            # Surface normal is perpendicular to the surface
            # If surface is z = f(x,y), then normal = (-df/dx, -df/dy, 1)
            normal = np.array([-grad_3d_x, -grad_3d_y, 1.0])
            
            # Normalize to unit vector
            normal_magnitude = np.linalg.norm(normal)
            if normal_magnitude > 1e-6:
                normal = normal / normal_magnitude
                # Ensure normal points upward (positive Z component)
                if normal[2] < 0:
                    normal = -normal
                return normal
            
        except Exception as e:
            print(f"Error computing normal at ({x}, {y}): {e}")
            
        return None
    
    def compute_normals_from_depth_gradient(self, 
                                          depth_map: np.ndarray, 
                                          use_sobel: bool = False,
                                          sobel_ksize: int = 5) -> np.ndarray:
        """
        Compute surface normals using numpy gradient method or Sobel filters.
        
        Args:
            depth_map: Input depth map (H x W)
            use_sobel: If True, use Sobel filters instead of numpy gradient
            sobel_ksize: Kernel size for Sobel filters (must be odd)
            
        Returns:
            Surface normal map (H x W x 3) with unit normal vectors
        """
        print(f"Computing surface normals using {'Sobel filters' if use_sobel else 'numpy gradient'}")
        
        if use_sobel:
            # Use Sobel filters for smoother gradient computation
            zx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            zy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            print(f"Using Sobel filters with kernel size: {sobel_ksize}")
        else:
            # Use numpy gradient for fast computation
            zy, zx = np.gradient(depth_map)
            print("Using numpy gradient method")
        
        # Construct normal vectors: cross product of surface tangents
        # Surface tangent vectors are [1, 0, zx] and [0, 1, zy]
        # Their cross product gives normal [-zx, -zy, 1]
        normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
        
        # Normalize to unit vectors
        n = np.linalg.norm(normal, axis=2, keepdims=True)
        # Avoid division by zero
        n = np.where(n > 1e-6, n, 1.0)
        normal = normal / n
        
        # Ensure normals point upward (positive Z component)
        normal[:, :, 2] = np.abs(normal[:, :, 2])
        
        print(f"Computed normals with shape: {normal.shape}")
        print(f"Normal range: X[{normal[:,:,0].min():.3f}, {normal[:,:,0].max():.3f}], "
              f"Y[{normal[:,:,1].min():.3f}, {normal[:,:,1].max():.3f}], "
              f"Z[{normal[:,:,2].min():.3f}, {normal[:,:,2].max():.3f}]")
        
        return normal
    
    def compute_normals_simple_gradient(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Simple implementation of surface normal computation using numpy gradient.
        This is the standalone version matching the user's requested function.
        
        Args:
            depth_map: Input depth map (H x W)
            
        Returns:
            Surface normal map (H x W x 3) with unit normal vectors
        """
        print("Computing surface normals using simple numpy gradient method")
        
        # Compute gradients
        zy, zx = np.gradient(depth_map)
        
        # Alternative: Use Sobel filters for smoother results
        # zx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)  
        # zy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
        
        # Construct normal vectors: cross product of surface tangents
        normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
        
        # Normalize to unit vectors
        n = np.linalg.norm(normal, axis=2, keepdims=True)
        # Avoid division by zero
        n = np.where(n > 1e-6, n, 1.0)
        normal = normal / n
        
        print(f"Computed normals with shape: {normal.shape}")
        print(f"Normal range: X[{normal[:,:,0].min():.3f}, {normal[:,:,0].max():.3f}], "
              f"Y[{normal[:,:,1].min():.3f}, {normal[:,:,1].max():.3f}], "
              f"Z[{normal[:,:,2].min():.3f}, {normal[:,:,2].max():.3f}]")
        
        return normal
    
    def estimate_grid_normals(self, depth_map: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Estimate surface normals on a grid across the image.
        
        Args:
            depth_map: Input depth map (H x W)
            
        Returns:
            Tuple of (normal_map, grid_data) where:
            - normal_map: 3D array (H x W x 3) of surface normals
            - grid_data: List of dicts with grid cell information
        """
        h, w = depth_map.shape
        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        grid_data = []
        
        print(f"Computing grid-based surface normals for {w}x{h} image")
        print(f"Grid size: {self.grid_size}px")
        
        # Iterate through grid
        valid_normals = 0
        total_cells = 0
        
        for grid_y in range(0, h, self.grid_size):
            for grid_x in range(0, w, self.grid_size):
                total_cells += 1
                
                # Get center of grid cell
                center_x = min(grid_x + self.grid_size // 2, w - 1)
                center_y = min(grid_y + self.grid_size // 2, h - 1)
                
                # Compute surface normal at grid center
                normal = self.compute_surface_normal_at_point(
                    depth_map, center_x, center_y, 
                    neighborhood_size=min(self.grid_size // 4, 5)
                )
                
                if normal is not None:
                    valid_normals += 1
                    
                    # Fill the grid cell with this normal
                    end_x = min(grid_x + self.grid_size, w)
                    end_y = min(grid_y + self.grid_size, h)
                    
                    normal_map[grid_y:end_y, grid_x:end_x] = normal
                    
                    # Store grid cell data for visualization
                    grid_data.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'normal': normal,
                        'depth': depth_map[center_y, center_x],
                        'grid_bounds': (grid_x, grid_y, end_x, end_y)
                    })
        
        print(f"Computed {valid_normals}/{total_cells} valid surface normals")
        return normal_map, grid_data
    
    def estimate_grid_normals_gradient(self, 
                                     depth_map: np.ndarray,
                                     use_sobel: bool = False,
                                     sobel_ksize: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        """
        Estimate surface normals using gradient method and organize on a grid for visualization.
        
        Args:
            depth_map: Input depth map (H x W)
            use_sobel: If True, use Sobel filters instead of numpy gradient
            sobel_ksize: Kernel size for Sobel filters
            
        Returns:
            Tuple of (normal_map, grid_data) where:
            - normal_map: 3D array (H x W x 3) of surface normals
            - grid_data: List of dicts with grid cell information
        """
        # Compute normals for entire image using gradient method
        normal_map = self.compute_normals_from_depth_gradient(
            depth_map, use_sobel=use_sobel, sobel_ksize=sobel_ksize
        )
        
        h, w = depth_map.shape
        grid_data = []
        
        print(f"Organizing gradient-based normals on {self.grid_size}px grid")
        
        # Sample normals on grid for visualization
        valid_cells = 0
        total_cells = 0
        
        for grid_y in range(0, h, self.grid_size):
            for grid_x in range(0, w, self.grid_size):
                total_cells += 1
                
                # Get center of grid cell
                center_x = min(grid_x + self.grid_size // 2, w - 1)
                center_y = min(grid_y + self.grid_size // 2, h - 1)
                
                # Get normal at grid center
                normal = normal_map[center_y, center_x]
                depth_val = depth_map[center_y, center_x]
                
                # Check if normal is valid (not zero)
                if np.linalg.norm(normal) > 1e-6 and depth_val > 0:
                    valid_cells += 1
                    
                    # Store grid cell data for visualization
                    end_x = min(grid_x + self.grid_size, w)
                    end_y = min(grid_y + self.grid_size, h)
                    
                    grid_data.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'normal': normal,
                        'depth': depth_val,
                        'grid_bounds': (grid_x, grid_y, end_x, end_y)
                    })
        
        print(f"Sampled {valid_cells}/{total_cells} valid grid cells for visualization")
        return normal_map, grid_data
    
    def estimate_grid_normals_simple(self, depth_map: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Estimate surface normals using simple numpy gradient and organize on a grid.
        This method uses the user's requested simple gradient computation.
        
        Args:
            depth_map: Input depth map (H x W)
            
        Returns:
            Tuple of (normal_map, grid_data) where:
            - normal_map: 3D array (H x W x 3) of surface normals
            - grid_data: List of dicts with grid cell information
        """
        # Compute normals for entire image using simple gradient method
        normal_map = self.compute_normals_simple_gradient(depth_map)
        
        h, w = depth_map.shape
        grid_data = []
        
        print(f"Organizing simple gradient normals on {self.grid_size}px grid")
        
        # Sample normals on grid for visualization
        valid_cells = 0
        total_cells = 0
        
        for grid_y in range(0, h, self.grid_size):
            for grid_x in range(0, w, self.grid_size):
                total_cells += 1
                
                # Get center of grid cell
                center_x = min(grid_x + self.grid_size // 2, w - 1)
                center_y = min(grid_y + self.grid_size // 2, h - 1)
                
                # Get normal at grid center
                normal = normal_map[center_y, center_x]
                depth_val = depth_map[center_y, center_x]
                
                # Check if normal is valid (not zero)
                if np.linalg.norm(normal) > 1e-6 and depth_val > 0:
                    valid_cells += 1
                    
                    # Store grid cell data for visualization
                    end_x = min(grid_x + self.grid_size, w)
                    end_y = min(grid_y + self.grid_size, h)
                    
                    grid_data.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'normal': normal,
                        'depth': depth_val,
                        'grid_bounds': (grid_x, grid_y, end_x, end_y)
                    })
        
        print(f"Sampled {valid_cells}/{total_cells} valid grid cells for visualization")
        return normal_map, grid_data
    
    def create_arrow_visualization(self, 
                                 image: np.ndarray, 
                                 grid_data: List[Dict],
                                 arrow_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Create visualization with arrows showing surface normals.
        
        Args:
            image: Base image for visualization (H x W x 3)
            grid_data: Grid cell data from estimate_grid_normals
            arrow_color: RGB color for arrows
            
        Returns:
            Visualization image with arrows
        """
        vis_image = image.copy()
        
        print(f"Drawing {len(grid_data)} surface normal arrows")
        arrows_drawn = 0
        
        for cell in grid_data:
            center_x = cell['center_x']
            center_y = cell['center_y']
            normal = cell['normal']
            
            # Amplify the normal components for better visibility
            # Scale up X and Y components while preserving direction
            scale_factor = 3.0  # Increase this to make arrows more visible
            
            # Calculate arrow end point with amplified normal components
            arrow_end_x = int(center_x + normal[0] * self.arrow_length * scale_factor)
            arrow_end_y = int(center_y + normal[1] * self.arrow_length * scale_factor)
            
            # Ensure arrow end is within image bounds
            h, w = vis_image.shape[:2]
            arrow_end_x = max(0, min(arrow_end_x, w - 1))
            arrow_end_y = max(0, min(arrow_end_y, h - 1))
            
            # Only draw arrow if it has significant length
            arrow_length_actual = np.sqrt((arrow_end_x - center_x)**2 + (arrow_end_y - center_y)**2)
            
            if arrow_length_actual > 5:  # Minimum arrow length threshold
                # Draw arrow
                cv2.arrowedLine(
                    vis_image,
                    (center_x, center_y),
                    (arrow_end_x, arrow_end_y),
                    arrow_color,
                    self.arrow_thickness,
                    tipLength=0.4
                )
                arrows_drawn += 1
            
            # Draw circle at arrow base (color-coded by Z component)
            # Green for upward normals, red for downward
            z_component = normal[2]
            if z_component > 0.7:  # Mostly upward
                circle_color = (0, 255, 0)  # Green
                circle_radius = 4
            elif z_component > 0.3:  # Moderately upward
                circle_color = (0, 255, 255)  # Yellow
                circle_radius = 3
            else:  # More tilted
                circle_color = (0, 0, 255)  # Red
                circle_radius = 2
            
            cv2.circle(vis_image, (center_x, center_y), circle_radius, circle_color, -1)
        
        print(f"Drew {arrows_drawn}/{len(grid_data)} visible arrows")
        return vis_image
    
    def create_enhanced_arrow_visualization(self, 
                                          image: np.ndarray, 
                                          grid_data: List[Dict],
                                          arrow_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Create enhanced visualization with more visible arrows showing surface normals.
        Uses a needle map approach where arrows are always visible.
        
        Args:
            image: Base image for visualization (H x W x 3)
            grid_data: Grid cell data from estimate_grid_normals
            arrow_color: RGB color for arrows
            
        Returns:
            Visualization image with enhanced arrows
        """
        vis_image = image.copy()
        
        print(f"Drawing {len(grid_data)} enhanced surface normal arrows")
        
        for cell in grid_data:
            center_x = cell['center_x']
            center_y = cell['center_y']
            normal = cell['normal']
            
            # Use a needle map approach: always show arrows with consistent length
            # Direction represents tilt, length is constant for visibility
            
            # Normalize X,Y components and scale to fixed arrow length
            xy_magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
            
            if xy_magnitude > 1e-6:  # Has horizontal component
                # Scale to arrow length
                arrow_end_x = int(center_x + (normal[0] / xy_magnitude) * self.arrow_length)
                arrow_end_y = int(center_y + (normal[1] / xy_magnitude) * self.arrow_length)
            else:
                # Pure vertical normal - show as small cross
                arrow_end_x = center_x
                arrow_end_y = center_y
            
            # Ensure arrow end is within image bounds
            h, w = vis_image.shape[:2]
            arrow_end_x = max(0, min(arrow_end_x, w - 1))
            arrow_end_y = max(0, min(arrow_end_y, h - 1))
            
            # Color and thickness based on surface angle
            z_component = abs(normal[2])
            
            if z_component > 0.9:  # Nearly flat surface
                color = (255, 255, 255)  # White - flat surface
                thickness = 1
                circle_radius = 2
            elif z_component > 0.7:  # Mildly tilted
                color = (0, 255, 0)  # Green
                thickness = 2
                circle_radius = 3
            elif z_component > 0.5:  # Moderately tilted  
                color = (0, 255, 255)  # Yellow
                thickness = 3
                circle_radius = 4
            else:  # Highly tilted
                color = (0, 0, 255)  # Red
                thickness = 4
                circle_radius = 5
            
            # Draw arrow if it has meaningful direction
            if np.sqrt((arrow_end_x - center_x)**2 + (arrow_end_y - center_y)**2) > 3:
                cv2.arrowedLine(
                    vis_image,
                    (center_x, center_y),
                    (arrow_end_x, arrow_end_y),
                    color,
                    thickness,
                    tipLength=0.3
                )
            else:
                # For vertical normals, draw a cross to indicate flatness
                cross_size = 8
                cv2.line(vis_image, 
                        (center_x - cross_size//2, center_y - cross_size//2),
                        (center_x + cross_size//2, center_y + cross_size//2),
                        color, thickness)
                cv2.line(vis_image, 
                        (center_x - cross_size//2, center_y + cross_size//2),
                        (center_x + cross_size//2, center_y - cross_size//2),
                        color, thickness)
            
            # Draw circle at center
            cv2.circle(vis_image, (center_x, center_y), circle_radius, color, -1)
        
        print(f"Enhanced arrows visualization completed")
        return vis_image
    
    def create_normal_magnitude_visualization(self, normal_map: np.ndarray) -> np.ndarray:
        """
        Create a color-coded visualization of normal magnitudes.
        
        Args:
            normal_map: Surface normal map (H x W x 3)
            
        Returns:
            Color visualization of normal magnitudes
        """
        # Calculate magnitude of normals
        magnitude = np.linalg.norm(normal_map, axis=2)
        
        # Normalize to 0-255 range
        if magnitude.max() > magnitude.min():
            magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        else:
            magnitude_norm = magnitude
        
        magnitude_uint8 = (magnitude_norm * 255).astype(np.uint8)
        
        # Apply colormap
        colored_magnitude = cv2.applyColorMap(magnitude_uint8, cv2.COLORMAP_VIRIDIS)
        
        return colored_magnitude
    
    def create_normal_direction_visualization(self, normal_map: np.ndarray) -> np.ndarray:
        """
        Create RGB visualization where color represents normal direction.
        
        Args:
            normal_map: Surface normal map (H x W x 3)
            
        Returns:
            RGB visualization of normal directions
        """
        # Map normal components to RGB channels
        # X -> Red, Y -> Green, Z -> Blue
        vis_image = np.zeros_like(normal_map, dtype=np.uint8)
        
        for i in range(3):
            component = normal_map[:, :, i]
            # Map from [-1, 1] to [0, 255]
            component_mapped = ((component + 1) / 2 * 255).astype(np.uint8)
            vis_image[:, :, i] = component_mapped
        
        return vis_image
    
    def get_statistics(self, normal_map: np.ndarray, grid_data: List[Dict]) -> Dict:
        """
        Compute statistics about the estimated surface normals.
        
        Args:
            normal_map: Surface normal map
            grid_data: Grid cell data
            
        Returns:
            Dictionary of statistics
        """
        valid_mask = np.any(normal_map != 0, axis=2)
        valid_normals = normal_map[valid_mask]
        
        if len(valid_normals) == 0:
            return {
                'total_grid_cells': len(grid_data),
                'valid_cells': 0,
                'coverage_percentage': 0.0,
                'mean_normal': [0, 0, 0],
                'std_normal': [0, 0, 0]
            }
        
        stats = {
            'total_grid_cells': len(grid_data),
            'valid_cells': len(grid_data),
            'coverage_percentage': len(grid_data) / (normal_map.shape[0] * normal_map.shape[1] / (self.grid_size ** 2)) * 100,
            'mean_normal': valid_normals.mean(axis=0).tolist(),
            'std_normal': valid_normals.std(axis=0).tolist(),
            'mean_upward_component': valid_normals[:, 2].mean(),
            'grid_size': self.grid_size
        }
        
        return stats


def visualize_surface_normals(normals: np.ndarray, output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    """
    estimator = GridBasedSurfaceNormalEstimator()
    return estimator.create_normal_direction_visualization(normals)


def create_normal_magnitude_map(normals: np.ndarray) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    """
    estimator = GridBasedSurfaceNormalEstimator()
    return estimator.create_normal_magnitude_visualization(normals)


# Standalone function matching user's exact requirements
def compute_normals_from_depth(depth_map):
    """
    Simple implementation of surface normal computation using numpy gradient.
    This is the exact function the user requested.
    
    Args:
        depth_map: Input depth map (H x W)
        
    Returns:
        Surface normal map (H x W x 3) with unit normal vectors
    """
    # Compute gradients
    zy, zx = np.gradient(depth_map)
    
    # Alternative: Use Sobel filters for smoother results
    # zx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)  
    # zy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    
    # Construct normal vectors: cross product of surface tangents
    normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
    
    # Normalize to unit vectors
    n = np.linalg.norm(normal, axis=2, keepdims=True)
    # Avoid division by zero
    n = np.where(n > 1e-6, n, 1.0)
    normal = normal / n
    
    return normal


# Legacy class for backward compatibility
class SurfaceNormalEstimator:
    """
    Legacy surface normal estimator class for backward compatibility.
    """
    
    def __init__(self, fx, fy, ox, oy, alpha=100, r_threshold=None):
        self.grid_estimator = GridBasedSurfaceNormalEstimator(
            fx=fx, fy=fy, cx=ox, cy=oy, 
            grid_size=alpha, arrow_length=30.0
        )
        self.r_threshold = r_threshold
    
    def estimate_normals(self, depth_map, mask=None):
        normal_map, _ = self.grid_estimator.estimate_grid_normals(depth_map)
        return normal_map
    
    def create_visualization(self, normals, output_size=None):
        return self.grid_estimator.create_normal_direction_visualization(normals)
    
    def get_normal_statistics(self, normals):
        valid_mask = np.any(normals != 0, axis=2)
        if not valid_mask.any():
            return {
                'valid_pixels': 0,
                'total_pixels': normals.shape[0] * normals.shape[1],
                'valid_percentage': 0.0,
                'mean_normal': [0, 0, 0]
            }
        
        valid_normals = normals[valid_mask]
        return {
            'valid_pixels': int(valid_mask.sum()),
            'total_pixels': normals.shape[0] * normals.shape[1],
            'valid_percentage': float(valid_mask.sum() / valid_mask.size * 100),
            'mean_normal': valid_normals.mean(axis=0).tolist()
        }
