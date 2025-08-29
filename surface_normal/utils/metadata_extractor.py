#!/usr/bin/env python3
"""
Camera Metadata Extractor
Extracts camera intrinsic parameters from image and video metadata (EXIF, etc.)
"""

import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import json
import os
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import subprocess


@dataclass
class CameraIntrinsics:
    """Data class for camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = "unknown"
    confidence: str = "low"  # low, medium, high
    camera_model: str = "unknown"
    focal_length_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None


class CameraMetadataExtractor:
    """Extract camera intrinsics from image and video metadata"""
    
    def __init__(self):
        """Initialize the metadata extractor"""
        # Common camera sensor sizes (in mm) for focal length conversion
        self.sensor_database = {
            # iPhone models
            "iPhone 15": {"sensor_width": 5.7, "sensor_height": 4.28},
            "iPhone 14": {"sensor_width": 5.7, "sensor_height": 4.28},
            "iPhone 13": {"sensor_width": 5.7, "sensor_height": 4.28},
            "iPhone 12": {"sensor_width": 5.7, "sensor_height": 4.28},
            "iPhone 11": {"sensor_width": 5.7, "sensor_height": 4.28},
            
            # Common DSLR/mirrorless sensors
            "Full Frame": {"sensor_width": 36.0, "sensor_height": 24.0},
            "APS-C Canon": {"sensor_width": 22.3, "sensor_height": 14.9},
            "APS-C Nikon": {"sensor_width": 23.5, "sensor_height": 15.6},
            "Micro 4/3": {"sensor_width": 17.3, "sensor_height": 13.0},
            
            # Default fallbacks
            "Generic Mobile": {"sensor_width": 5.7, "sensor_height": 4.28},
            "Generic Camera": {"sensor_width": 23.5, "sensor_height": 15.6}
        }
    
    def extract_from_image(self, image_path: str) -> CameraIntrinsics:
        """
        Extract camera intrinsics from image metadata
        
        Args:
            image_path: Path to the image file
            
        Returns:
            CameraIntrinsics object with extracted parameters
        """
        print(f"Extracting camera metadata from image: {image_path}")
        
        try:
            # Load image to get dimensions
            image = Image.open(image_path)
            width, height = image.size
            
            print(f"Image dimensions: {width} x {height}")
            
            # Extract EXIF data
            exif_data = self._extract_exif_data(image)
            
            # Try to extract camera intrinsics from EXIF
            intrinsics = self._parse_camera_intrinsics_from_exif(exif_data, width, height)
            
            if intrinsics:
                print(f"✅ Successfully extracted intrinsics from EXIF metadata")
                return intrinsics
            
            # Fallback: estimate from image dimensions and common assumptions
            print("⚠️ No usable EXIF data found, using fallback estimation")
            return self._estimate_intrinsics_fallback(width, height, "image")
            
        except Exception as e:
            print(f"❌ Error extracting metadata from image: {e}")
            # Final fallback
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    height, width = img.shape[:2]
                    return self._estimate_intrinsics_fallback(width, height, "image_opencv")
                else:
                    # Use default dimensions if all else fails
                    return self._estimate_intrinsics_fallback(1920, 1080, "default")
            except:
                return self._estimate_intrinsics_fallback(1920, 1080, "default")
    
    def extract_from_video(self, video_path: str) -> CameraIntrinsics:
        """
        Extract camera intrinsics from video metadata
        
        Args:
            video_path: Path to the video file
            
        Returns:
            CameraIntrinsics object with extracted parameters
        """
        print(f"Extracting camera metadata from video: {video_path}")
        
        try:
            # Open video to get basic properties
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            print(f"Video properties: {width}x{height} @ {fps:.2f}fps")
            
            # Try to extract metadata using ffprobe
            video_metadata = self._extract_video_metadata_ffprobe(video_path)
            
            # Parse camera intrinsics from video metadata
            intrinsics = self._parse_camera_intrinsics_from_video_metadata(video_metadata, width, height)
            
            if intrinsics:
                print(f"✅ Successfully extracted intrinsics from video metadata")
                return intrinsics
            
            # Try extracting first frame and analyzing as image
            print("Attempting to extract metadata from first video frame...")
            first_frame_intrinsics = self._extract_from_video_frame(video_path, width, height)
            
            if first_frame_intrinsics:
                return first_frame_intrinsics
            
            # Fallback: estimate from video dimensions
            print("⚠️ No usable video metadata found, using fallback estimation")
            return self._estimate_intrinsics_fallback(width, height, "video")
            
        except Exception as e:
            print(f"❌ Error extracting metadata from video: {e}")
            return self._estimate_intrinsics_fallback(1920, 1080, "default")
    
    def _extract_exif_data(self, image: Image.Image) -> Dict[str, Any]:
        """Extract EXIF data from PIL Image"""
        exif_data = {}
        
        try:
            if hasattr(image, '_getexif') and image._getexif() is not None:
                exif = image._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
                    
            print(f"Extracted {len(exif_data)} EXIF tags")
            
            # Log some key metadata for debugging
            key_tags = ['Make', 'Model', 'FocalLength', 'FocalLengthIn35mmFilm', 
                       'ExifImageWidth', 'ExifImageHeight', 'XResolution', 'YResolution']
            
            for tag in key_tags:
                if tag in exif_data:
                    print(f"  {tag}: {exif_data[tag]}")
                    
        except Exception as e:
            print(f"Warning: Could not extract EXIF data: {e}")
            
        return exif_data
    
    def _parse_camera_intrinsics_from_exif(self, exif_data: Dict[str, Any], 
                                         width: int, height: int) -> Optional[CameraIntrinsics]:
        """Parse camera intrinsics from EXIF data"""
        
        if not exif_data:
            return None
        
        try:
            # Extract camera make and model
            camera_make = exif_data.get('Make', 'Unknown')
            camera_model = exif_data.get('Model', 'Unknown')
            full_camera_name = f"{camera_make} {camera_model}".strip()
            
            print(f"Camera: {full_camera_name}")
            
            # Extract focal length information
            focal_length_mm = None
            focal_length_35mm = exif_data.get('FocalLengthIn35mmFilm')
            
            if 'FocalLength' in exif_data:
                fl_value = exif_data['FocalLength']
                if isinstance(fl_value, tuple) and len(fl_value) == 2:
                    focal_length_mm = fl_value[0] / fl_value[1]
                else:
                    focal_length_mm = float(fl_value)
                    
            print(f"Focal length: {focal_length_mm}mm (35mm equiv: {focal_length_35mm}mm)")
            
            # Determine sensor size
            sensor_info = self._get_sensor_info_for_camera(full_camera_name)
            
            if focal_length_mm and sensor_info:
                # Calculate focal length in pixels
                fx = (focal_length_mm * width) / sensor_info['sensor_width']
                fy = (focal_length_mm * height) / sensor_info['sensor_height']
                
                # Principal point (usually image center)
                cx = width / 2.0
                cy = height / 2.0
                
                return CameraIntrinsics(
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    width=width, height=height,
                    source="exif_metadata",
                    confidence="high",
                    camera_model=full_camera_name,
                    focal_length_mm=focal_length_mm,
                    sensor_width_mm=sensor_info['sensor_width'],
                    sensor_height_mm=sensor_info['sensor_height']
                )
            
            # If we have 35mm equivalent focal length, use that
            if focal_length_35mm:
                # Use full frame sensor as reference
                full_frame = self.sensor_database["Full Frame"]
                fx = (focal_length_35mm * width) / full_frame['sensor_width']
                fy = (focal_length_35mm * height) / full_frame['sensor_height']
                cx = width / 2.0
                cy = height / 2.0
                
                return CameraIntrinsics(
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    width=width, height=height,
                    source="exif_35mm_equivalent",
                    confidence="medium",
                    camera_model=full_camera_name,
                    focal_length_mm=focal_length_35mm,
                    sensor_width_mm=full_frame['sensor_width'],
                    sensor_height_mm=full_frame['sensor_height']
                )
                
        except Exception as e:
            print(f"Warning: Error parsing EXIF intrinsics: {e}")
            
        return None
    
    def _extract_video_metadata_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using ffprobe"""
        metadata = {}
        
        try:
            # Use ffprobe to extract metadata
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                print(f"Successfully extracted video metadata using ffprobe")
            else:
                print(f"ffprobe failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("ffprobe timed out")
        except FileNotFoundError:
            print("ffprobe not found - install ffmpeg for better metadata extraction")
        except Exception as e:
            print(f"Error running ffprobe: {e}")
            
        return metadata
    
    def _parse_camera_intrinsics_from_video_metadata(self, metadata: Dict[str, Any], 
                                                   width: int, height: int) -> Optional[CameraIntrinsics]:
        """Parse camera intrinsics from video metadata"""
        
        if not metadata:
            return None
        
        try:
            # Extract format tags (often contains camera info)
            format_info = metadata.get('format', {})
            tags = format_info.get('tags', {})
            
            # Look for camera-related tags
            camera_info = {}
            for key, value in tags.items():
                key_lower = key.lower()
                if any(cam_key in key_lower for cam_key in ['make', 'model', 'camera', 'device']):
                    camera_info[key] = value
                    print(f"Video metadata - {key}: {value}")
            
            # Check streams for additional metadata
            streams = metadata.get('streams', [])
            for stream in streams:
                if stream.get('codec_type') == 'video':
                    stream_tags = stream.get('tags', {})
                    for key, value in stream_tags.items():
                        key_lower = key.lower()
                        if any(cam_key in key_lower for cam_key in ['make', 'model', 'camera', 'device']):
                            camera_info[key] = value
                            print(f"Video stream metadata - {key}: {value}")
            
            # If we found camera information, try to estimate intrinsics
            if camera_info:
                camera_model = self._extract_camera_model_from_metadata(camera_info)
                sensor_info = self._get_sensor_info_for_camera(camera_model)
                
                if sensor_info:
                    # Estimate focal length based on common video recording settings
                    estimated_focal_length_mm = self._estimate_video_focal_length(camera_model, width, height)
                    
                    if estimated_focal_length_mm:
                        fx = (estimated_focal_length_mm * width) / sensor_info['sensor_width']
                        fy = (estimated_focal_length_mm * height) / sensor_info['sensor_height']
                        cx = width / 2.0
                        cy = height / 2.0
                        
                        return CameraIntrinsics(
                            fx=fx, fy=fy, cx=cx, cy=cy,
                            width=width, height=height,
                            source="video_metadata",
                            confidence="medium",
                            camera_model=camera_model,
                            focal_length_mm=estimated_focal_length_mm,
                            sensor_width_mm=sensor_info['sensor_width'],
                            sensor_height_mm=sensor_info['sensor_height']
                        )
                        
        except Exception as e:
            print(f"Warning: Error parsing video metadata: {e}")
            
        return None
    
    def _extract_from_video_frame(self, video_path: str, width: int, height: int) -> Optional[CameraIntrinsics]:
        """Extract metadata from the first frame of the video"""
        
        try:
            # Extract first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Save frame temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    cv2.imwrite(temp_file.name, frame)
                    
                    # Extract metadata from frame
                    intrinsics = self.extract_from_image(temp_file.name)
                    
                    # Clean up
                    os.unlink(temp_file.name)
                    
                    # Update source to indicate it came from video frame
                    intrinsics.source = f"video_frame_{intrinsics.source}"
                    
                    return intrinsics
                    
        except Exception as e:
            print(f"Warning: Could not extract metadata from video frame: {e}")
            
        return None
    
    def _get_sensor_info_for_camera(self, camera_model: str) -> Optional[Dict[str, float]]:
        """Get sensor information for a specific camera model"""
        
        camera_model_lower = camera_model.lower()
        
        # Direct match
        if camera_model in self.sensor_database:
            return self.sensor_database[camera_model]
        
        # Pattern matching
        if 'iphone' in camera_model_lower:
            # Extract iPhone model number
            for model_name in self.sensor_database:
                if 'iphone' in model_name.lower() and any(num in camera_model_lower for num in ['15', '14', '13', '12', '11']):
                    return self.sensor_database[model_name]
            return self.sensor_database["iPhone 13"]  # Default iPhone
        
        # Generic patterns
        if any(mobile in camera_model_lower for mobile in ['phone', 'mobile', 'android']):
            return self.sensor_database["Generic Mobile"]
        
        if any(dslr in camera_model_lower for dslr in ['canon', 'nikon', 'sony', 'dslr']):
            return self.sensor_database["APS-C Canon"]  # Default DSLR
        
        # Default fallback
        return self.sensor_database["Generic Camera"]
    
    def _extract_camera_model_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Extract camera model from metadata dictionary"""
        
        make = ""
        model = ""
        
        for key, value in metadata.items():
            key_lower = key.lower()
            if 'make' in key_lower:
                make = str(value)
            elif 'model' in key_lower:
                model = str(value)
        
        return f"{make} {model}".strip() or "Unknown Camera"
    
    def _estimate_video_focal_length(self, camera_model: str, width: int, height: int) -> Optional[float]:
        """Estimate focal length for video recording based on camera model and resolution"""
        
        camera_model_lower = camera_model.lower()
        
        # iPhone video focal lengths (approximate)
        if 'iphone' in camera_model_lower:
            # iPhones typically use wide lens for video
            return 4.25  # Approximate wide lens focal length
        
        # Generic mobile device
        if any(mobile in camera_model_lower for mobile in ['phone', 'mobile', 'android']):
            return 4.0  # Typical mobile wide lens
        
        # DSLR/mirrorless - assume standard lens
        if any(camera in camera_model_lower for camera in ['canon', 'nikon', 'sony']):
            return 35.0  # Typical wide-normal lens
        
        # Default fallback based on resolution
        if width >= 3840:  # 4K
            return 28.0
        elif width >= 1920:  # 1080p
            return 35.0
        else:  # 720p or lower
            return 50.0
    
    def _estimate_intrinsics_fallback(self, width: int, height: int, source: str) -> CameraIntrinsics:
        """Fallback estimation when no metadata is available"""
        
        print(f"Using fallback intrinsic estimation for {width}x{height}")
        
        # Estimate focal length based on common field of view assumptions
        # Typical smartphone/camera FOV is around 60-70 degrees
        diagonal_pixels = np.sqrt(width**2 + height**2)
        
        # Assume 65 degree field of view (typical for smartphones)
        fov_radians = np.radians(65)
        focal_length_pixels = diagonal_pixels / (2 * np.tan(fov_radians / 2))
        
        # Adjust for aspect ratio
        fx = focal_length_pixels * (width / diagonal_pixels)
        fy = focal_length_pixels * (height / diagonal_pixels)
        
        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        # Determine confidence based on source
        confidence = "low"
        if "image" in source:
            confidence = "medium"  # Images might have better estimation
        
        return CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height,
            source=f"fallback_{source}",
            confidence=confidence,
            camera_model="Estimated",
            focal_length_mm=None,
            sensor_width_mm=None,
            sensor_height_mm=None
        )
    
    def log_intrinsics(self, intrinsics: CameraIntrinsics, output_file: str = None) -> None:
        """Log camera intrinsics to file and console"""
        
        log_content = f"""
Camera Intrinsics Extraction Report
===================================
Source: {intrinsics.source}
Confidence: {intrinsics.confidence}
Camera Model: {intrinsics.camera_model}

Intrinsic Parameters:
- Focal Length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f} pixels
- Principal Point: cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f} pixels
- Image Dimensions: {intrinsics.width}x{intrinsics.height} pixels

Physical Parameters:
- Focal Length: {intrinsics.focal_length_mm:.2f} mm (if available)
- Sensor Size: {intrinsics.sensor_width_mm}x{intrinsics.sensor_height_mm} mm (if available)

Camera Matrix K:
[{intrinsics.fx:8.2f}  0.00  {intrinsics.cx:8.2f}]
[     0.00  {intrinsics.fy:8.2f}  {intrinsics.cy:8.2f}]
[     0.00       0.00       1.00]

Field of View:
- Horizontal: {2 * np.arctan(intrinsics.width / (2 * intrinsics.fx)) * 180 / np.pi:.1f}°
- Vertical: {2 * np.arctan(intrinsics.height / (2 * intrinsics.fy)) * 180 / np.pi:.1f}°
"""
        
        print(log_content)
        
        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(log_content)
                print(f"✅ Intrinsics logged to: {output_file}")
            except Exception as e:
                print(f"⚠️ Warning: Could not write log file: {e}")
