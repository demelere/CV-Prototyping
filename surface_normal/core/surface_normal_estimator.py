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
