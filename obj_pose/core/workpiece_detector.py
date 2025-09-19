"""
Workpiece detection module.
Handles surface detection, plane fitting, and coordinate transformation for welding analysis.
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from config.settings import *


class WorkpieceDetector:
    """Detects workpiece surface and provides coordinate transformation"""
    
    def __init__(self):
        """Initialize workpiece detector for surface detection and coordinate transformation"""
        self.workpiece_normal = None
        self.workpiece_origin = None
        self.coordinate_transform_matrix = None
        self.detection_confidence = 0.0
        
        # Contact plane estimation
        self.contact_plane_normal = None
        self.contact_point_3d = None
        self.contact_region_roi = None
        self.contact_plane_confidence = 0.0
        
    def detect_workpiece_surface(self, depth_map, frame_width, frame_height, keypoints_3d=None):
        """Detect workpiece surface using depth map and optionally keypoint guidance"""
        if depth_map is None:
            print("⚠️ No depth map available for workpiece detection")
            return False
        
        print(f"DEBUG: Starting workpiece detection with depth map shape: {depth_map.shape}")
        print(f"DEBUG: Frame dimensions: {frame_width}x{frame_height}")
        
        try:
            # If we have keypoints, use them to guide surface detection
            if keypoints_3d and len(keypoints_3d) > 0:
                return self._detect_workpiece_with_keypoints(depth_map, frame_width, frame_height, keypoints_3d)
            else:
                return self._detect_workpiece_with_depth_only(depth_map, frame_width, frame_height)
        
        except Exception as e:
            print(f"Error in workpiece detection: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_workpiece_with_keypoints(self, depth_map, frame_width, frame_height, keypoints_3d):
        """Detect workpiece surface using background depth while excluding electrode area"""
        print("DEBUG: Using background-based workpiece detection with electrode exclusion")
        
        # Find electrode tip position
        electrode_tip_2d = None
        electrode_tip_3d = None
        for keypoint in keypoints_3d:
            if keypoint.get('parent_object') == 'electrode' and keypoint.get('class') == 'tip':
                electrode_tip_2d = (int(keypoint['x_2d']), int(keypoint['y_2d']))
                electrode_tip_3d = (keypoint['x_3d'], keypoint['y_3d'], keypoint['z_3d'])
                break
        
        if electrode_tip_2d is None:
            print("DEBUG: No electrode tip found, falling back to depth-only detection")
            return self._detect_workpiece_with_depth_only(depth_map, frame_width, frame_height)
        
        print(f"DEBUG: Found electrode tip at 2D: {electrode_tip_2d}, 3D: {electrode_tip_3d}")
        
        # Create a mask to exclude electrode area from surface detection
        # Define electrode exclusion zone (larger than just the tip)
        exclusion_radius = int(min(frame_width, frame_height) * WORKPIECE_ELECTRODE_EXCLUSION_RADIUS)
        electrode_x, electrode_y = electrode_tip_2d
        
        # Convert entire depth map to 3D points, excluding electrode area
        points_3d = []
        h, w = depth_map.shape
        
        # Sample points (every 4th pixel to reduce computation)
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                # Check if point is within electrode exclusion zone
                distance_to_electrode = np.sqrt((x - electrode_x)**2 + (y - electrode_y)**2)
                if distance_to_electrode <= exclusion_radius:
                    continue  # Skip electrode area
                
                depth_value = depth_map[y, x]
                
                # Skip invalid depth values
                if depth_value <= 0:
                    continue
                
                # Convert to real-world depth
                depth_meters = depth_value / DEPTH_SCALE_FACTOR
                
                # Back-project to 3D using camera intrinsics
                x_3d = (x - CAMERA_CX) * depth_meters / CAMERA_FX
                y_3d = (y - CAMERA_CY) * depth_meters / CAMERA_FY
                z_3d = depth_meters
                
                points_3d.append([x_3d, y_3d, z_3d])
        
        points_3d = np.array(points_3d)
        print(f"DEBUG: Converted {len(points_3d)} 3D points from background (excluding electrode area)")
        print(f"DEBUG: Electrode exclusion radius: {exclusion_radius} pixels")
        
        if len(points_3d) < WORKPIECE_MIN_CLUSTER_SIZE:
            print(f"Insufficient background points: {len(points_3d)} (need {WORKPIECE_MIN_CLUSTER_SIZE})")
            return False
        
        # Filter points to focus on the CLOSEST surfaces (where workpiece likely is)
        # Sort by Z-coordinate and take the closest 60% of points
        z_coords = [p[2] for p in points_3d]
        sorted_indices = np.argsort(z_coords)
        num_closest = int(len(points_3d) * 0.6)  # Take closest 60%
        closest_points = [points_3d[i] for i in sorted_indices[:num_closest]]
        closest_points = np.array(closest_points)
        
        print(f"DEBUG: Filtered to {len(closest_points)} closest background points (Z range: {min(z_coords):.4f} to {max(z_coords):.4f})")
        
        # Cluster points to find the largest flat surface
        clusters = self._cluster_depth_points(closest_points)
        print(f"DEBUG: Found {len(clusters)} clusters from background detection")
        
        if not clusters:
            print("No valid clusters found for workpiece detection")
            return False
        
        # Find the largest cluster (likely the workpiece)
        largest_cluster = max(clusters, key=len)
        print(f"DEBUG: Largest cluster has {len(largest_cluster)} points")
        
        if len(largest_cluster) < WORKPIECE_MIN_CLUSTER_SIZE:
            print(f"Largest cluster too small: {len(largest_cluster)} (need {WORKPIECE_MIN_CLUSTER_SIZE})")
            return False
        
        # Fit plane to the largest cluster
        plane_params = self._fit_plane_to_points(largest_cluster)
        
        if plane_params is None:
            print("Failed to fit plane to workpiece surface")
            return False
        
        # Extract plane parameters (ax + by + cz + d = 0)
        a, b, c, d = plane_params
        print(f"DEBUG: Background-based plane parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
        
        # Normalize the normal vector
        normal = np.array([a, b, c])
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            print("Invalid normal vector")
            return False
        
        normal = normal / normal_norm
        
        # Ensure normal points toward camera (positive Z in camera coordinates)
        if normal[2] < 0:
            normal = -normal
            d = -d
        
        # Set workpiece coordinate system
        self.workpiece_normal = normal
        self.workpiece_origin = np.array([0, 0, -d/c]) if c != 0 else np.array([0, 0, 0])
        
        # Calculate coordinate transformation matrix
        self._calculate_coordinate_transform()
        
        # Calculate detection confidence based on cluster size and proximity to electrode
        self.detection_confidence = min(1.0, len(largest_cluster) / 500.0)
        
        print(f"✅ Workpiece surface detected with keypoint guidance (confidence: {self.detection_confidence:.2f})")
        print(f"Workpiece normal: {normal}")
        print(f"Workpiece origin: {self.workpiece_origin}")
        
        return True
    
    def _detect_workpiece_with_depth_only(self, depth_map, frame_width, frame_height):
        """Original depth-only workpiece detection method"""
        print("DEBUG: Using depth-only workpiece detection")
        
        # Define region of interest - focus on center area where welding likely occurs
        # Use a smaller, more focused ROI around the center where welding happens
        roi_width = int(frame_width * 0.4)  # Smaller width (was 0.6)
        roi_height = int(frame_height * 0.4)  # Smaller height (was 0.6)
        roi_x = (frame_width - roi_width) // 2
        roi_y = (frame_height - roi_height) // 2
        
        print(f"DEBUG: ROI dimensions: {roi_width}x{roi_height} at ({roi_x}, {roi_y})")
        
        # Extract ROI from depth map
        roi_depth = depth_map[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        print(f"DEBUG: ROI depth shape: {roi_depth.shape}")
        print(f"DEBUG: ROI depth range: {roi_depth.min()} to {roi_depth.max()}")
        
        # Convert depth map to 3D points
        points_3d = self._depth_to_3d_points(roi_depth, roi_x, roi_y, frame_width, frame_height)
        print(f"DEBUG: Converted {len(points_3d)} 3D points from depth map")
        
        if len(points_3d) < WORKPIECE_MIN_CLUSTER_SIZE:
            print(f"Insufficient points for workpiece detection: {len(points_3d)} (need {WORKPIECE_MIN_CLUSTER_SIZE})")
            return False
        
        # Filter points to focus on the CLOSEST surfaces (where welding happens)
        # Sort by Z-coordinate and take the closest 50% of points (was 70%)
        z_coords = [p[2] for p in points_3d]
        sorted_indices = np.argsort(z_coords)
        num_closest = int(len(points_3d) * 0.5)  # Take closest 50%
        closest_points = [points_3d[i] for i in sorted_indices[:num_closest]]
        
        # Convert to numpy array for clustering
        closest_points = np.array(closest_points)
        
        print(f"DEBUG: Filtered to {len(closest_points)} closest points (Z range: {min(z_coords):.4f} to {max(z_coords):.4f})")
        print(f"DEBUG: Closest point Z: {min(z_coords):.4f}, Farthest point Z: {max(z_coords):.4f}")
        
        # Cluster points to find the largest flat surface
        clusters = self._cluster_depth_points(closest_points)
        print(f"DEBUG: Found {len(clusters)} clusters")
        
        if not clusters:
            print("No valid clusters found for workpiece detection")
            return False
        
        # Find the largest cluster (likely the workpiece)
        largest_cluster = max(clusters, key=len)
        print(f"DEBUG: Largest cluster has {len(largest_cluster)} points")
        
        if len(largest_cluster) < WORKPIECE_MIN_CLUSTER_SIZE:
            print(f"Largest cluster too small: {len(largest_cluster)} (need {WORKPIECE_MIN_CLUSTER_SIZE})")
            return False
        
        # Fit plane to the largest cluster
        plane_params = self._fit_plane_to_points(largest_cluster)
        
        if plane_params is None:
            print("Failed to fit plane to workpiece surface")
            return False
        
        # Extract plane parameters (ax + by + cz + d = 0)
        a, b, c, d = plane_params
        print(f"DEBUG: Plane parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
        
        # Normalize the normal vector
        normal = np.array([a, b, c])
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            print("Invalid normal vector")
            return False
        
        normal = normal / normal_norm
        
        # Validate that the detected surface is reasonable
        # For a camera looking down at an angle, the normal should have a significant Z component
        # but not be perfectly aligned with camera Z-axis
        z_component = abs(normal[2])
        if z_component > 0.95:  # Too close to camera Z-axis
            print(f"WARNING: Detected surface normal too close to camera Z-axis: {normal}")
            print("This might be detecting background rather than workpiece surface")
            print("Expected: Normal should reflect camera angle (e.g., [0.2, -0.1, 0.9])")
            print("Actual: Normal is [0, 0, 1] indicating surface perpendicular to camera view")
            print("Trying to detect the actual workpiece surface instead...")
            
            # Try to find a cluster with more variation in Z-coordinates
            clusters_with_variation = []
            for i, cluster in enumerate(clusters):
                if len(cluster) >= WORKPIECE_MIN_CLUSTER_SIZE:
                    z_values = [p[2] for p in cluster]
                    z_variation = max(z_values) - min(z_values)
                    if z_variation > 0.005:  # At least 5mm variation
                        clusters_with_variation.append((i, cluster, z_variation))
                        print(f"DEBUG: Cluster {i} has Z variation: {z_variation:.4f}")
            
            if clusters_with_variation:
                # Use the cluster with the most Z-variation (likely the actual workpiece)
                best_cluster_idx, best_cluster, best_variation = max(clusters_with_variation, key=lambda x: x[2])
                print(f"DEBUG: Using cluster {best_cluster_idx} with Z variation {best_variation:.4f}")
                
                # Fit plane to the best cluster
                plane_params = self._fit_plane_to_points(best_cluster)
                if plane_params is not None:
                    a, b, c, d = plane_params
                    normal = np.array([a, b, c])
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm > 0:
                        normal = normal / normal_norm
                        print(f"DEBUG: New plane parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
                        print(f"DEBUG: New normal vector: {normal}")
        
        # Ensure normal points toward camera (positive Z in camera coordinates)
        if normal[2] < 0:
            normal = -normal
            d = -d
        
        # Set workpiece coordinate system
        self.workpiece_normal = normal
        self.workpiece_origin = np.array([0, 0, -d/c]) if c != 0 else np.array([0, 0, 0])
        
        # Calculate coordinate transformation matrix
        self._calculate_coordinate_transform()
        
        # Calculate detection confidence based on cluster size and plane fit quality
        self.detection_confidence = min(1.0, len(largest_cluster) / 1000.0)
        
        print(f"✅ Workpiece surface detected with confidence {self.detection_confidence:.2f}")
        print(f"Workpiece normal: {normal}")
        print(f"Workpiece origin: {self.workpiece_origin}")
        
        return True
    
    def _depth_to_3d_points(self, depth_roi, roi_x, roi_y, frame_width, frame_height):
        """Convert depth ROI to 3D points in camera coordinates"""
        points = []
        h, w = depth_roi.shape
        
        # Sample points (every 4th pixel to reduce computation)
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                depth_value = depth_roi[y, x]
                
                # Skip invalid depth values
                if depth_value <= 0:
                    continue
                
                # Convert to real-world depth
                depth_meters = depth_value / DEPTH_SCALE_FACTOR
                
                # Convert pixel coordinates to camera coordinates
                pixel_x = roi_x + x
                pixel_y = roi_y + y
                
                # Back-project to 3D using camera intrinsics
                x_3d = (pixel_x - CAMERA_CX) * depth_meters / CAMERA_FX
                y_3d = (pixel_y - CAMERA_CY) * depth_meters / CAMERA_FY
                z_3d = depth_meters
                
                points.append([x_3d, y_3d, z_3d])
        
        return np.array(points)
    
    def _cluster_depth_points(self, points_3d):
        """Cluster 3D points to find flat surfaces"""
        if len(points_3d) < 10:
            print(f"DEBUG: Not enough points for clustering: {len(points_3d)}")
            return []
        
        print(f"DEBUG: Clustering {len(points_3d)} 3D points")
        
        # Normalize depth values for clustering
        points_normalized = points_3d.copy()
        points_normalized[:, 2] = points_normalized[:, 2] * 100  # Scale depth for clustering
        
        print(f"DEBUG: Normalized depth range: {points_normalized[:, 2].min():.2f} to {points_normalized[:, 2].max():.2f}")
        
        # Use DBSCAN with tighter parameters to better distinguish surfaces
        # Smaller eps for tighter clustering, higher min_samples for more robust clusters
        # Try multiple clustering parameters to find the best surface
        clustering_params = [
            (0.02, 10),  # Very tight clustering
            (0.03, 8),   # Medium tight clustering
            (0.05, 5),   # Looser clustering
        ]
        
        best_clusters = []
        best_cluster_count = 0
        
        for eps, min_samples in clustering_params:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_normalized)
            labels = clustering.labels_
            
            # Group points by cluster
            clusters = []
            for label in set(labels):
                if label != -1:  # Skip noise points
                    cluster_points = points_3d[labels == label]
                    if len(cluster_points) >= WORKPIECE_MIN_CLUSTER_SIZE:
                        clusters.append(cluster_points)
            
            print(f"DEBUG: DBSCAN(eps={eps}, min_samples={min_samples}) found {len(clusters)} valid clusters")
            
            # Choose the clustering that gives us the most clusters with good size
            if len(clusters) > best_cluster_count:
                best_clusters = clusters
                best_cluster_count = len(clusters)
                print(f"DEBUG: Using clustering with eps={eps}, min_samples={min_samples}")
        
        print(f"DEBUG: Returning {len(best_clusters)} valid clusters")
        return best_clusters
    
    def _fit_plane_to_points(self, points):
        """Fit a plane to 3D points using RANSAC"""
        if len(points) < 3:
            print(f"DEBUG: Not enough points for plane fitting: {len(points)}")
            return None
        
        print(f"DEBUG: Fitting plane to {len(points)} points")
        
        # Prepare data for RANSAC
        X = points[:, :2]  # x, y coordinates
        y = points[:, 2]   # z coordinates
        
        print(f"DEBUG: X range: {X[:, 0].min():.4f} to {X[:, 0].max():.4f}")
        print(f"DEBUG: Y range: {X[:, 1].min():.4f} to {X[:, 1].max():.4f}")
        print(f"DEBUG: Z range: {y.min():.4f} to {y.max():.4f}")
        
        # Fit plane using RANSAC
        ransac = RANSACRegressor(
            residual_threshold=WORKPIECE_RANSAC_THRESHOLD,
            min_samples=3,
            max_trials=100
        )
        
        try:
            ransac.fit(X, y)
            
            # Extract plane parameters (z = ax + by + c)
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
            
            print(f"DEBUG: RANSAC fit successful: a={a:.4f}, b={b:.4f}, c={c:.4f}")
            
            # Convert to standard form (ax + by + cz + d = 0)
            # where z = ax + by + c becomes ax + by - z + c = 0
            plane_params = [a, b, -1, c]
            
            return plane_params
            
        except Exception as e:
            print(f"Error fitting plane: {e}")
            return None
    
    def _calculate_coordinate_transform(self):
        """Calculate transformation matrix from camera to workpiece coordinates"""
        if self.workpiece_normal is None:
            print("DEBUG: No workpiece normal available for coordinate transformation")
            return
        
        # Create coordinate system aligned with workpiece surface
        # Z-axis: workpiece normal (pointing toward camera)
        z_axis = self.workpiece_normal
        
        # X-axis: perpendicular to Z and world Y (up)
        world_y = np.array([0, 1, 0])
        x_axis = np.cross(world_y, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            # If Z is parallel to world Y, use world X instead
            world_x = np.array([1, 0, 0])
            x_axis = np.cross(world_x, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis: perpendicular to X and Z
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Create transformation matrix (4x4)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = self.workpiece_origin
        
        self.coordinate_transform_matrix = transform_matrix
        
        print(f"DEBUG: Coordinate transformation matrix calculated")
        print(f"DEBUG: X-axis: {x_axis}")
        print(f"DEBUG: Y-axis: {y_axis}")
        print(f"DEBUG: Z-axis: {z_axis}")
    
    def get_workpiece_normal(self):
        """Get the workpiece surface normal vector"""
        return self.workpiece_normal
    
    def get_workpiece_origin(self):
        """Get the workpiece coordinate system origin"""
        return self.workpiece_origin
    
    def get_detection_confidence(self):
        """Get the confidence of workpiece detection"""
        return self.detection_confidence
    
    def transform_to_workpiece_coordinates(self, point_3d):
        """Transform a 3D point from camera coordinates to workpiece coordinates"""
        if self.coordinate_transform_matrix is None:
            return point_3d
        
        # Convert point to homogeneous coordinates
        point_homogeneous = np.append(point_3d, 1)
        
        # Apply transformation
        transformed_point = np.dot(self.coordinate_transform_matrix, point_homogeneous)
        
        return transformed_point[:3]
    
    def estimate_contact_plane_normal(self, electrode_tip_3d, electrode_axis, depth_map, camera_intrinsics, roi_radius_pixels=50):
        """
        Estimate the local plane normal at the anticipated contact region.
        
        Args:
            electrode_tip_3d (np.ndarray): 3D position of electrode tip [x, y, z]
            electrode_axis (np.ndarray): Normalized electrode axis vector [x, y, z]
            depth_map (np.ndarray): Depth map (HxW)
            camera_intrinsics (dict): Camera intrinsic parameters {fx, fy, cx, cy}
            roi_radius_pixels (int): Radius of ROI around contact region in pixels
            
        Returns:
            dict: Contact plane data including normal, point, and confidence
        """
        print(f"DEBUG: Starting contact plane normal estimation")
        print(f"DEBUG: Electrode tip 3D: {electrode_tip_3d}")
        print(f"DEBUG: Electrode axis: {electrode_axis}")
        print(f"DEBUG: ROI radius: {roi_radius_pixels} pixels")
        
        try:
            # Step 1: Project electrode tip to 2D image coordinates
            fx = camera_intrinsics['fx']
            fy = camera_intrinsics['fy']
            cx = camera_intrinsics['cx']
            cy = camera_intrinsics['cy']
            
            # Convert 3D electrode tip to 2D pixel coordinates
            if electrode_tip_3d[2] <= 0:
                print("ERROR: Invalid electrode tip depth (z <= 0)")
                return None
                
            tip_x_2d = int((electrode_tip_3d[0] * fx / electrode_tip_3d[2]) + cx)
            tip_y_2d = int((electrode_tip_3d[1] * fy / electrode_tip_3d[2]) + cy)
            
            print(f"DEBUG: Electrode tip 2D: ({tip_x_2d}, {tip_y_2d})")
            
            # Step 2: Create ROI around the electrode tip
            h, w = depth_map.shape
            roi_x1 = max(0, tip_x_2d - roi_radius_pixels)
            roi_y1 = max(0, tip_y_2d - roi_radius_pixels)
            roi_x2 = min(w, tip_x_2d + roi_radius_pixels)
            roi_y2 = min(h, tip_y_2d + roi_radius_pixels)
            
            # Check if ROI is valid (not empty)
            if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
                print(f"ERROR: Invalid ROI - electrode tip projected outside image bounds")
                print(f"DEBUG: Image bounds: {w}x{h}, Tip 2D: ({tip_x_2d}, {tip_y_2d}), ROI: ({roi_x1}, {roi_y1}, {roi_x2}, {roi_y2})")
                return None
            
            self.contact_region_roi = (roi_x1, roi_y1, roi_x2, roi_y2)
            print(f"DEBUG: Contact region ROI: ({roi_x1}, {roi_y1}, {roi_x2}, {roi_y2})")
            
            # Step 3: Extract depth ROI
            depth_roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
            print(f"DEBUG: Depth ROI shape: {depth_roi.shape}")
            
            # Check if depth ROI is valid
            if depth_roi.size == 0:
                print(f"ERROR: Empty depth ROI")
                return None
                
            print(f"DEBUG: Depth ROI range: {depth_roi.min():.3f} to {depth_roi.max():.3f}")
            
            # Step 4: Convert depth ROI to 3D points
            points_3d = self._depth_roi_to_3d_points(depth_roi, roi_x1, roi_y1, fx, fy, cx, cy)
            print(f"DEBUG: Converted {len(points_3d)} 3D points from depth ROI")
            
            if len(points_3d) < 10:
                print(f"ERROR: Insufficient points in contact region: {len(points_3d)}")
                return None
            
            # Step 5: Fit local plane using RANSAC
            plane_params = self._fit_plane_to_points(points_3d)
            if plane_params is None:
                print("ERROR: Failed to fit plane to contact region")
                return None
            
            # Step 6: Extract plane normal and ensure it points toward camera
            a, b, c, d = plane_params
            normal = np.array([a, b, c])
            normal_norm = np.linalg.norm(normal)
            if normal_norm == 0:
                print("ERROR: Invalid normal vector")
                return None
            
            normal = normal / normal_norm
            
            # Ensure normal points toward camera (positive Z in camera coordinates)
            if normal[2] < 0:
                normal = -normal
                d = -d
            
            # Step 7: Calculate ray-plane intersection
            contact_point = self._ray_plane_intersection(electrode_tip_3d, electrode_axis, normal, d)
            if contact_point is None:
                print("ERROR: Failed to calculate ray-plane intersection")
                return None
            
            # Step 8: Calculate confidence based on plane fit quality
            confidence = self._calculate_plane_fit_confidence(points_3d, plane_params)
            
            # Store results
            self.contact_plane_normal = normal
            self.contact_point_3d = contact_point
            self.contact_plane_confidence = confidence
            
            print(f"✅ Contact plane normal estimated successfully")
            print(f"Contact plane normal: {normal}")
            print(f"Contact point 3D: {contact_point}")
            print(f"Confidence: {confidence:.3f}")
            
            return {
                'normal': normal,
                'contact_point': contact_point,
                'roi': self.contact_region_roi,
                'confidence': confidence,
                'plane_params': plane_params
            }
            
        except Exception as e:
            print(f"ERROR in contact plane estimation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _depth_roi_to_3d_points(self, depth_roi, roi_x1, roi_y1, fx, fy, cx, cy):
        """Convert depth ROI to 3D points in camera coordinates"""
        points = []
        h, w = depth_roi.shape
        
        # Sample points (every 2nd pixel for better density in small ROI)
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                depth_value = depth_roi[y, x]
                
                # Skip invalid depth values
                if depth_value <= 0:
                    continue
                
                # Convert to real-world depth
                depth_meters = depth_value / DEPTH_SCALE_FACTOR
                
                # Convert pixel coordinates to camera coordinates
                pixel_x = roi_x1 + x
                pixel_y = roi_y1 + y
                
                # Back-project to 3D using camera intrinsics
                x_3d = (pixel_x - cx) * depth_meters / fx
                y_3d = (pixel_y - cy) * depth_meters / fy
                z_3d = depth_meters
                
                points.append([x_3d, y_3d, z_3d])
        
        return np.array(points)
    
    def _ray_plane_intersection(self, ray_origin, ray_direction, plane_normal, plane_d):
        """
        Calculate intersection point of ray with plane.
        
        Args:
            ray_origin (np.ndarray): Ray origin point [x, y, z]
            ray_direction (np.ndarray): Normalized ray direction [x, y, z]
            plane_normal (np.ndarray): Normalized plane normal [x, y, z]
            plane_d (float): Plane equation constant (ax + by + cz + d = 0)
            
        Returns:
            np.ndarray: Intersection point [x, y, z] or None if no intersection
        """
        # Ray equation: P = P0 + t * direction
        # Plane equation: ax + by + cz + d = 0
        
        # Calculate denominator: n · direction
        denominator = np.dot(plane_normal, ray_direction)
        
        if abs(denominator) < 1e-6:
            print("ERROR: Ray is parallel to plane")
            return None
        
        # Calculate t parameter
        # t = -(n · P0 + d) / (n · direction)
        numerator = np.dot(plane_normal, ray_origin) + plane_d
        t = -numerator / denominator
        
        # Check if intersection is in front of ray origin
        if t < 0:
            print(f"WARNING: Intersection is behind ray origin (t={t})")
            # Still return the point for debugging
        
        # Calculate intersection point
        intersection_point = ray_origin + t * ray_direction
        
        print(f"DEBUG: Ray-plane intersection: t={t:.4f}, point={intersection_point}")
        
        return intersection_point
    
    def _calculate_plane_fit_confidence(self, points_3d, plane_params):
        """Calculate confidence score for plane fit based on RANSAC inlier ratio"""
        if len(points_3d) < 3:
            return 0.0
        
        # Calculate distances from points to plane
        a, b, c, d = plane_params
        distances = np.abs(a * points_3d[:, 0] + b * points_3d[:, 1] + c * points_3d[:, 2] + d)
        
        # Count inliers (points within threshold distance)
        threshold = WORKPIECE_RANSAC_THRESHOLD
        inliers = np.sum(distances < threshold)
        inlier_ratio = inliers / len(points_3d)
        
        # Additional confidence based on point count
        point_confidence = min(1.0, len(points_3d) / 100.0)
        
        # Combined confidence
        confidence = (inlier_ratio * 0.7) + (point_confidence * 0.3)
        
        print(f"DEBUG: Plane fit confidence - inliers: {inliers}/{len(points_3d)} ({inlier_ratio:.3f}), point_confidence: {point_confidence:.3f}, final: {confidence:.3f}")
        
        return confidence
    
    def get_contact_plane_normal(self):
        """Get the contact plane normal vector"""
        return self.contact_plane_normal
    
    def get_contact_point_3d(self):
        """Get the 3D contact point"""
        return self.contact_point_3d
    
    def get_contact_region_roi(self):
        """Get the contact region ROI coordinates"""
        return self.contact_region_roi
    
    def get_contact_plane_confidence(self):
        """Get the confidence of contact plane estimation"""
        return self.contact_plane_confidence
