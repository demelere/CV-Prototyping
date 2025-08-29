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
    
    def compute_normals_with_backprojection(self, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute surface normals using proper 3D back-projection with camera intrinsics.
        This method converts pixels to 3D world coordinates before computing normals.
        
        Args:
            depth_map: Input depth map (H x W) in meters
            
        Returns:
            Tuple of (normal_map, point_cloud_3d) where:
            - normal_map: Surface normal map (H x W x 3) with unit normal vectors in world space
            - point_cloud_3d: 3D point cloud (H x W x 3) with world coordinates in meters
        """
        print("Computing surface normals using 3D back-projection with camera intrinsics")
        print(f"Using camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        
        h, w = depth_map.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Back-project pixels to 3D world coordinates
        X = (u - self.cx) * depth_map / self.fx
        Y = (v - self.cy) * depth_map / self.fy
        Z = depth_map
        
        # Stack to create 3D point cloud
        point_cloud_3d = np.stack([X, Y, Z], axis=2)
        
        print(f"3D point cloud created with shape: {point_cloud_3d.shape}")
        print(f"World coordinate ranges:")
        print(f"  X: [{X.min():.3f}, {X.max():.3f}] meters")
        print(f"  Y: [{Y.min():.3f}, {Y.max():.3f}] meters") 
        print(f"  Z: [{Z.min():.3f}, {Z.max():.3f}] meters")
        
        # Compute gradients in world space
        dX_du, dX_dv = np.gradient(X)
        dY_du, dY_dv = np.gradient(Y)
        dZ_du, dZ_dv = np.gradient(Z)
        
        # Compute surface tangent vectors
        tangent_u = np.stack([dX_du, dY_du, dZ_du], axis=2)
        tangent_v = np.stack([dX_dv, dY_dv, dZ_dv], axis=2)
        
        # Compute normal as cross product of tangent vectors
        normal = np.cross(tangent_u, tangent_v)
        
        # Normalize to unit vectors
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        norm = np.where(norm > 1e-6, norm, 1.0)
        normal = normal / norm
        
        # Ensure normals point towards camera (negative Z direction)
        # Flip normals that point away from camera
        flip_mask = normal[:, :, 2] > 0
        normal[flip_mask] = -normal[flip_mask]
        
        print(f"Computed 3D surface normals with shape: {normal.shape}")
        print(f"Normal range in world space:")
        print(f"  X: [{normal[:,:,0].min():.3f}, {normal[:,:,0].max():.3f}]")
        print(f"  Y: [{normal[:,:,1].min():.3f}, {normal[:,:,1].max():.3f}]")
        print(f"  Z: [{normal[:,:,2].min():.3f}, {normal[:,:,2].max():.3f}]")
        
        return normal, point_cloud_3d
    

    
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
    
    def estimate_grid_normals_3d(self, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Estimate surface normals using proper 3D back-projection with camera intrinsics.
        This method provides geometrically correct surface normals in world space.
        
        Args:
            depth_map: Input depth map (H x W) in meters
            
        Returns:
            Tuple of (normal_map, point_cloud_3d, grid_data) where:
            - normal_map: 3D array (H x W x 3) of surface normals in world space
            - point_cloud_3d: 3D array (H x W x 3) of world coordinates in meters
            - grid_data: List of dicts with grid cell information including 3D coordinates
        """
        # Compute normals using 3D back-projection
        normal_map, point_cloud_3d = self.compute_normals_with_backprojection(depth_map)
        
        h, w = depth_map.shape
        grid_data = []
        
        print(f"Organizing 3D back-projected normals on {self.grid_size}px grid")
        
        # Sample normals and 3D points on grid for visualization
        valid_cells = 0
        total_cells = 0
        
        for grid_y in range(0, h, self.grid_size):
            for grid_x in range(0, w, self.grid_size):
                total_cells += 1
                
                # Get center of grid cell
                center_x = min(grid_x + self.grid_size // 2, w - 1)
                center_y = min(grid_y + self.grid_size // 2, h - 1)
                
                # Get normal and 3D point at grid center
                normal = normal_map[center_y, center_x]
                point_3d = point_cloud_3d[center_y, center_x]
                depth_val = depth_map[center_y, center_x]
                
                # Check if normal and point are valid
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
                        'point_3d': point_3d,
                        'world_x': point_3d[0],
                        'world_y': point_3d[1], 
                        'world_z': point_3d[2],
                        'grid_bounds': (grid_x, grid_y, end_x, end_y)
                    })
        
        print(f"Sampled {valid_cells}/{total_cells} valid grid cells with 3D coordinates")
        
        # Log camera intrinsics and 3D coordinate analysis
        self._log_camera_intrinsics_and_3d_analysis(depth_map, point_cloud_3d, normal_map, grid_data)
        
        return normal_map, point_cloud_3d, grid_data
    
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
    
    def _log_camera_intrinsics_and_3d_analysis(self, depth_map: np.ndarray, point_cloud_3d: np.ndarray, 
                                              normal_map: np.ndarray, grid_data: List[Dict]) -> None:
        """
        Log comprehensive camera intrinsics and 3D coordinate analysis.
        
        Args:
            depth_map: Input depth map
            point_cloud_3d: 3D point cloud
            normal_map: Surface normal map
            grid_data: Grid cell data
        """
        try:
            from datetime import datetime
            import os
            
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/camera_intrinsics_3d_analysis_{timestamp}.txt"
            
            with open(log_file, 'w') as f:
                f.write("Camera Intrinsics and 3D Back-Projection Analysis\n")
                f.write("=" * 60 + "\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                # Camera intrinsics section
                f.write("CAMERA INTRINSICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Camera Type: Monocular (Depth-Anything-V2)\n")
                f.write(f"Focal Length X (fx): {self.fx:.2f} pixels\n")
                f.write(f"Focal Length Y (fy): {self.fy:.2f} pixels\n")
                f.write(f"Principal Point X (cx): {self.cx:.2f} pixels\n")
                f.write(f"Principal Point Y (cy): {self.cy:.2f} pixels\n")
                f.write(f"Baseline: N/A (monocular system)\n")
                
                # Calculate field of view
                h, w = depth_map.shape
                fov_x = 2 * np.arctan(w / (2 * self.fx)) * 180 / np.pi
                fov_y = 2 * np.arctan(h / (2 * self.fy)) * 180 / np.pi
                f.write(f"Field of View X: {fov_x:.1f} degrees\n")
                f.write(f"Field of View Y: {fov_y:.1f} degrees\n")
                f.write(f"Image dimensions: {w} x {h} pixels\n\n")
                
                # Depth map analysis
                f.write("DEPTH MAP ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}] meters\n")
                f.write(f"Depth mean: {depth_map.mean():.3f} meters\n")
                f.write(f"Depth std: {depth_map.std():.3f} meters\n")
                f.write(f"Valid depth pixels: {np.sum(depth_map > 0)}/{depth_map.size} ({100*np.sum(depth_map > 0)/depth_map.size:.1f}%)\n\n")
                
                # 3D coordinate analysis
                f.write("3D WORLD COORDINATES:\n")
                f.write("-" * 20 + "\n")
                X = point_cloud_3d[:, :, 0]
                Y = point_cloud_3d[:, :, 1] 
                Z = point_cloud_3d[:, :, 2]
                
                f.write(f"X coordinates: [{X.min():.3f}, {X.max():.3f}] meters\n")
                f.write(f"Y coordinates: [{Y.min():.3f}, {Y.max():.3f}] meters\n")
                f.write(f"Z coordinates: [{Z.min():.3f}, {Z.max():.3f}] meters\n")
                f.write(f"Scene width (X): {X.max() - X.min():.3f} meters\n")
                f.write(f"Scene height (Y): {Y.max() - Y.min():.3f} meters\n")
                f.write(f"Scene depth (Z): {Z.max() - Z.min():.3f} meters\n\n")
                
                # Surface normal analysis
                f.write("SURFACE NORMAL ANALYSIS:\n")
                f.write("-" * 25 + "\n")
                nx = normal_map[:, :, 0]
                ny = normal_map[:, :, 1]
                nz = normal_map[:, :, 2]
                
                f.write(f"Normal X range: [{nx.min():.3f}, {nx.max():.3f}]\n")
                f.write(f"Normal Y range: [{ny.min():.3f}, {ny.max():.3f}]\n")
                f.write(f"Normal Z range: [{nz.min():.3f}, {nz.max():.3f}]\n")
                
                # Analyze surface orientations
                # Flat surfaces (high Z component)
                flat_mask = np.abs(nz) > 0.9
                flat_percentage = 100 * np.sum(flat_mask) / flat_mask.size
                
                # Vertical surfaces (low Z component)
                vertical_mask = np.abs(nz) < 0.3
                vertical_percentage = 100 * np.sum(vertical_mask) / vertical_mask.size
                
                f.write(f"Flat surfaces (|nz| > 0.9): {flat_percentage:.1f}%\n")
                f.write(f"Vertical surfaces (|nz| < 0.3): {vertical_percentage:.1f}%\n\n")
                
                # Grid analysis
                f.write("GRID CELL ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Grid size: {self.grid_size} x {self.grid_size} pixels\n")
                f.write(f"Total grid cells: {len(grid_data)}\n")
                
                if grid_data:
                    # Sample 10 grid cells for detailed analysis
                    sample_size = min(10, len(grid_data))
                    f.write(f"\nSample of {sample_size} grid cells:\n")
                    f.write("Cell | Pixel (x,y) | Depth (m) | World (X,Y,Z) | Normal (nx,ny,nz)\n")
                    f.write("-" * 80 + "\n")
                    
                    for i in range(sample_size):
                        cell = grid_data[i]
                        px, py = cell['center_x'], cell['center_y']
                        depth = cell['depth']
                        wx, wy, wz = cell['world_x'], cell['world_y'], cell['world_z']
                        nx, ny, nz = cell['normal']
                        
                        f.write(f"{i+1:4d} | ({px:4d},{py:4d}) | {depth:8.3f} | "
                               f"({wx:6.3f},{wy:6.3f},{wz:6.3f}) | "
                               f"({nx:6.3f},{ny:6.3f},{nz:6.3f})\n")
                
                f.write(f"\n3D BACK-PROJECTION VERIFICATION:\n")
                f.write("-" * 35 + "\n")
                f.write("The following formula was used for back-projection:\n")
                f.write("X = (u - cx) * Z / fx\n")
                f.write("Y = (v - cy) * Z / fy\n")
                f.write("Z = depth_value\n")
                f.write("Where (u,v) are pixel coordinates and (X,Y,Z) are world coordinates.\n\n")
                
                # Verify a sample point
                if grid_data:
                    sample = grid_data[0]
                    u, v = sample['center_x'], sample['center_y']
                    Z = sample['depth']
                    X_calc = (u - self.cx) * Z / self.fx
                    Y_calc = (v - self.cy) * Z / self.fy
                    
                    f.write("Verification for first grid cell:\n")
                    f.write(f"Pixel coordinates: ({u}, {v})\n")
                    f.write(f"Depth: {Z:.3f} m\n")
                    f.write(f"Calculated X: ({u} - {self.cx}) * {Z:.3f} / {self.fx} = {X_calc:.3f} m\n")
                    f.write(f"Calculated Y: ({v} - {self.cy}) * {Z:.3f} / {self.fy} = {Y_calc:.3f} m\n")
                    f.write(f"Stored world coordinates: ({sample['world_x']:.3f}, {sample['world_y']:.3f}, {sample['world_z']:.3f})\n")
                    f.write(f"Match: {'✓' if abs(X_calc - sample['world_x']) < 1e-6 and abs(Y_calc - sample['world_y']) < 1e-6 else '✗'}\n")
            
            print(f"✅ Camera intrinsics and 3D analysis logged to: {log_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to log camera intrinsics analysis: {e}")
