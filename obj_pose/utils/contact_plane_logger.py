#!/usr/bin/env python3
"""
Contact Plane Estimation Logger
Comprehensive logging for debugging contact plane estimation with correlation IDs, units, and failure detection.
"""

import logging
import time
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import json

class ContactPlaneLogger:
    """
    Structured logger for contact plane estimation with correlation tracking.
    Provides visibility into the entire pipeline with proper units and failure detection.
    """
    
    def __init__(self, log_level=logging.INFO):
        """Initialize the contact plane logger"""
        self.logger = logging.getLogger('ContactPlaneEstimation')
        self.logger.setLevel(log_level)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%H:%M:%S.%f'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Session tracking
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        
        self.logger.info(f"🔍 ContactPlaneLogger initialized | Session: {self.session_id}")
    
    def create_correlation_id(self, operation: str) -> str:
        """Create a correlation ID for tracking operations"""
        return f"{self.session_id}-{operation}-{int(time.time() * 1000) % 10000:04d}"
    
    def log_operation_start(self, correlation_id: str, operation: str, **kwargs):
        """Log the start of an operation with context"""
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"🚀 START {operation} | ID: {correlation_id} | {context}")
    
    def log_operation_success(self, correlation_id: str, operation: str, duration_ms: float, **kwargs):
        """Log successful operation completion"""
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"✅ SUCCESS {operation} | ID: {correlation_id} | Duration: {duration_ms:.1f}ms | {context}")
    
    def log_operation_failure(self, correlation_id: str, operation: str, error: str, duration_ms: float, **kwargs):
        """Log operation failure with error details"""
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.error(f"❌ FAILURE {operation} | ID: {correlation_id} | Duration: {duration_ms:.1f}ms | Error: {error} | {context}")
    
    def log_camera_parameters(self, correlation_id: str, intrinsics: Dict[str, float], extrinsics: Optional[np.ndarray] = None):
        """Log camera parameters with proper units"""
        self.logger.info(f"📐 CAMERA_PARAMS | ID: {correlation_id} | "
                        f"fx={intrinsics['fx']:.1f}px, fy={intrinsics['fy']:.1f}px, "
                        f"cx={intrinsics['cx']:.1f}px, cy={intrinsics['cy']:.1f}px")
        
        if extrinsics is not None:
            rotation = extrinsics[:3, :3]
            translation = extrinsics[:3, 3]
            self.logger.info(f"📐 CAMERA_EXTRINSICS | ID: {correlation_id} | "
                           f"Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]m | "
                           f"Rotation det: {np.linalg.det(rotation):.6f}")
    
    def log_3d_coordinates(self, correlation_id: str, point_name: str, coords_3d: np.ndarray, units: str = "m"):
        """Log 3D coordinates with units"""
        self.logger.info(f"🎯 3D_COORDS | ID: {correlation_id} | {point_name}: "
                        f"[{coords_3d[0]:.3f}, {coords_3d[1]:.3f}, {coords_3d[2]:.3f}]{units}")
    
    def log_2d_projection(self, correlation_id: str, point_name: str, coords_2d: Tuple[int, int], 
                         image_size: Tuple[int, int], valid: bool):
        """Log 2D projection with bounds checking"""
        x, y = coords_2d
        w, h = image_size
        bounds_check = "✅" if (0 <= x < w and 0 <= y < h) else "❌"
        validity = "✅" if valid else "❌"
        
        self.logger.info(f"🖼️ 2D_PROJECTION | ID: {correlation_id} | {point_name}: "
                        f"({x}, {y})px | Bounds: {bounds_check} | Valid: {validity} | "
                        f"Image: {w}x{h}px")
    
    def log_roi_analysis(self, correlation_id: str, roi_bounds: Tuple[int, int, int, int], 
                        depth_stats: Dict[str, float], valid_pixels: int):
        """Log ROI analysis with depth statistics"""
        x1, y1, x2, y2 = roi_bounds
        roi_size = (x2 - x1) * (y2 - y1)
        coverage = (valid_pixels / roi_size * 100) if roi_size > 0 else 0
        
        self.logger.info(f"📦 ROI_ANALYSIS | ID: {correlation_id} | "
                        f"Bounds: ({x1}, {y1}, {x2}, {y2}) | Size: {roi_size}px | "
                        f"Valid: {valid_pixels}px ({coverage:.1f}%) | "
                        f"Depth: [{depth_stats['min']:.3f}, {depth_stats['max']:.3f}]m | "
                        f"Mean: {depth_stats['mean']:.3f}m")
    
    def log_plane_fitting(self, correlation_id: str, num_points: int, plane_params: np.ndarray, 
                         confidence: float, inlier_ratio: float):
        """Log plane fitting results"""
        normal = plane_params[:3]
        d = plane_params[3]
        normal_magnitude = np.linalg.norm(normal)
        
        self.logger.info(f"📐 PLANE_FITTING | ID: {correlation_id} | "
                        f"Points: {num_points} | Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}] | "
                        f"Magnitude: {normal_magnitude:.6f} | D: {d:.3f} | "
                        f"Confidence: {confidence:.3f} | Inliers: {inlier_ratio:.1%}")
    
    def log_ray_plane_intersection(self, correlation_id: str, ray_origin: np.ndarray, 
                                  ray_direction: np.ndarray, intersection: Optional[np.ndarray],
                                  distance_to_plane: Optional[float]):
        """Log ray-plane intersection with geometric analysis"""
        if intersection is not None:
            ray_length = np.linalg.norm(intersection - ray_origin)
            self.logger.info(f"🔗 RAY_INTERSECTION | ID: {correlation_id} | "
                           f"Origin: [{ray_origin[0]:.3f}, {ray_origin[1]:.3f}, {ray_origin[2]:.3f}]m | "
                           f"Direction: [{ray_direction[0]:.3f}, {ray_direction[1]:.3f}, {ray_direction[2]:.3f}] | "
                           f"Intersection: [{intersection[0]:.3f}, {intersection[1]:.3f}, {intersection[2]:.3f}]m | "
                           f"Ray length: {ray_length:.3f}m")
        else:
            self.logger.warning(f"⚠️ RAY_INTERSECTION_FAILED | ID: {correlation_id} | "
                              f"Origin: [{ray_origin[0]:.3f}, {ray_origin[1]:.3f}, {ray_origin[2]:.3f}]m | "
                              f"Direction: [{ray_direction[0]:.3f}, {ray_direction[1]:.3f}, {ray_direction[2]:.3f}] | "
                              f"Distance to plane: {distance_to_plane:.3f}m")
    
    def log_visualization_data(self, correlation_id: str, contact_point_2d: Optional[Tuple[int, int]],
                              normal_2d: Optional[Tuple[int, int]], roi_bounds: Optional[Tuple[int, int, int, int]],
                              image_size: Tuple[int, int]):
        """Log visualization data with bounds checking"""
        w, h = image_size
        
        if contact_point_2d is not None:
            x, y = contact_point_2d
            in_bounds = 0 <= x < w and 0 <= y < h
            self.logger.info(f"🎨 VIS_CONTACT_POINT | ID: {correlation_id} | "
                           f"2D: ({x}, {y})px | In bounds: {'✅' if in_bounds else '❌'} | "
                           f"Image: {w}x{h}px")
        else:
            self.logger.warning(f"⚠️ VIS_CONTACT_POINT_MISSING | ID: {correlation_id}")
        
        if normal_2d is not None:
            x, y = normal_2d
            in_bounds = 0 <= x < w and 0 <= y < h
            self.logger.info(f"🎨 VIS_NORMAL_ARROW | ID: {correlation_id} | "
                           f"End: ({x}, {y})px | In bounds: {'✅' if in_bounds else '❌'}")
        else:
            self.logger.warning(f"⚠️ VIS_NORMAL_ARROW_MISSING | ID: {correlation_id}")
        
        if roi_bounds is not None:
            x1, y1, x2, y2 = roi_bounds
            in_bounds = (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h)
            self.logger.info(f"🎨 VIS_ROI | ID: {correlation_id} | "
                           f"Bounds: ({x1}, {y1}, {x2}, {y2}) | In bounds: {'✅' if in_bounds else '❌'}")
        else:
            self.logger.warning(f"⚠️ VIS_ROI_MISSING | ID: {correlation_id}")
    
    def log_performance_metrics(self, correlation_id: str, operation: str, metrics: Dict[str, float]):
        """Log performance metrics with units"""
        metric_str = " | ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
        self.logger.info(f"⚡ PERFORMANCE | ID: {correlation_id} | {operation} | {metric_str}")
    
    def log_decision_point(self, correlation_id: str, decision: str, condition: str, result: bool, **context):
        """Log decision points for debugging logic flow"""
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
        result_icon = "✅" if result else "❌"
        self.logger.info(f"🤔 DECISION | ID: {correlation_id} | {decision} | "
                        f"Condition: {condition} | Result: {result_icon} | {context_str}")
    
    def log_error_with_context(self, correlation_id: str, error: Exception, context: Dict[str, Any]):
        """Log errors with full context for debugging"""
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.error(f"💥 ERROR | ID: {correlation_id} | {type(error).__name__}: {str(error)} | {context_str}")
        
        # Log stack trace for debugging
        import traceback
        self.logger.debug(f"STACK_TRACE | ID: {correlation_id} | {traceback.format_exc()}")
    
    def log_summary(self, correlation_id: str, success: bool, total_duration_ms: float, 
                   key_metrics: Dict[str, Any]):
        """Log operation summary"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        metrics_str = " | ".join([f"{k}={v}" for k, v in key_metrics.items()])
        
        self.logger.info(f"📊 SUMMARY | ID: {correlation_id} | {status} | "
                        f"Duration: {total_duration_ms:.1f}ms | {metrics_str}")

# Global logger instance
contact_plane_logger = ContactPlaneLogger()
