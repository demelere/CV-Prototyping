"""
Depth logging utility for debugging depth estimation outputs.
Provides controlled logging of raw depth map data without overwhelming the logs.
"""

import numpy as np
from datetime import datetime
from pathlib import Path


class DepthLogger:
    """Logs depth estimation outputs for debugging purposes"""
    
    def __init__(self, log_dir="logs", max_log_files=10, sample_strategy="grid"):
        """
        Initialize depth logger.
        
        Args:
            log_dir (str): Directory to store log files
            max_log_files (int): Maximum number of depth log files to keep
            sample_strategy (str): Strategy for sampling depth data:
                - 'grid': Sample on a regular grid
                - 'random': Random sampling
                - 'corners': Sample corners and center
                - 'edges': Sample along edges
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.max_log_files = max_log_files
        self.sample_strategy = sample_strategy
        
        # Clean up old log files
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove old depth log files to prevent disk space issues"""
        depth_logs = list(self.log_dir.glob("depth_log_*.txt"))
        if len(depth_logs) > self.max_log_files:
            # Sort by modification time and remove oldest
            depth_logs.sort(key=lambda x: x.stat().st_mtime)
            for old_log in depth_logs[:-self.max_log_files]:
                try:
                    old_log.unlink()
                    print(f"DEBUG: Removed old depth log: {old_log}")
                except Exception as e:
                    print(f"Warning: Could not remove old log {old_log}: {e}")
    
    def _sample_depth_data(self, depth_array, max_samples=100):
        """
        Sample depth data using the specified strategy.
        
        Args:
            depth_array (np.ndarray): Depth array to sample
            max_samples (int): Maximum number of samples to take
            
        Returns:
            dict: Sampled depth data with metadata
        """
        height, width = depth_array.shape
        
        if self.sample_strategy == "grid":
            # Sample on a regular grid
            step_y = max(1, height // int(np.sqrt(max_samples)))
            step_x = max(1, width // int(np.sqrt(max_samples)))
            
            samples = []
            for y in range(0, height, step_y):
                for x in range(0, width, step_x):
                    if len(samples) >= max_samples:
                        break
                    samples.append({
                        'x': x, 'y': y, 
                        'depth': float(depth_array[y, x])
                    })
                if len(samples) >= max_samples:
                    break
                    
        elif self.sample_strategy == "corners":
            # Sample corners, center, and midpoints
            samples = []
            
            # Corners
            corners = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
            for x, y in corners:
                samples.append({
                    'x': x, 'y': y, 
                    'depth': float(depth_array[y, x]),
                    'location': 'corner'
                })
            
            # Center
            center_x, center_y = width // 2, height // 2
            samples.append({
                'x': center_x, 'y': center_y,
                'depth': float(depth_array[center_y, center_x]),
                'location': 'center'
            })
            
            # Midpoints of edges
            edge_points = [
                (width // 2, 0, 'top'),
                (width // 2, height - 1, 'bottom'),
                (0, height // 2, 'left'),
                (width - 1, height // 2, 'right')
            ]
            for x, y, location in edge_points:
                samples.append({
                    'x': x, 'y': y,
                    'depth': float(depth_array[y, x]),
                    'location': location
                })
                
        elif self.sample_strategy == "edges":
            # Sample along edges with some interior points
            samples = []
            
            # Sample along top and bottom edges
            for x in np.linspace(0, width-1, min(20, width), dtype=int):
                samples.append({
                    'x': x, 'y': 0,
                    'depth': float(depth_array[0, x]),
                    'location': 'top_edge'
                })
                samples.append({
                    'x': x, 'y': height-1,
                    'depth': float(depth_array[height-1, x]),
                    'location': 'bottom_edge'
                })
            
            # Sample along left and right edges
            for y in np.linspace(0, height-1, min(20, height), dtype=int):
                samples.append({
                    'x': 0, 'y': y,
                    'depth': float(depth_array[y, 0]),
                    'location': 'left_edge'
                })
                samples.append({
                    'x': width-1, 'y': y,
                    'depth': float(depth_array[y, width-1]),
                    'location': 'right_edge'
                })
            
            # Add some interior points
            for i in range(0, min(20, max_samples - len(samples))):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                samples.append({
                    'x': x, 'y': y,
                    'depth': float(depth_array[y, x]),
                    'location': 'interior'
                })
                
        else:  # random
            # Random sampling
            indices = np.random.choice(height * width, min(max_samples, height * width), replace=False)
            samples = []
            for idx in indices:
                y, x = np.unravel_index(idx, (height, width))
                samples.append({
                    'x': int(x), 'y': int(y),
                    'depth': float(depth_array[y, x])
                })
        
        return samples
    
    def log_depth_estimation(self, depth_array, image_path, model_name, additional_metadata=None):
        """
        Log depth estimation results to a text file.
        
        Args:
            depth_array (np.ndarray): Raw depth array from model
            image_path (str): Path to input image
            model_name (str): Name of the depth model used
            additional_metadata (dict, optional): Additional metadata to log
            
        Returns:
            str: Path to the created log file
        """
        try:
            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"depth_log_{timestamp}.txt"
            log_path = self.log_dir / log_filename
            
            # Prepare metadata
            metadata = {
                'timestamp': timestamp,
                'image_path': str(image_path),
                'model_name': model_name,
                'depth_array_shape': depth_array.shape,
                'depth_array_dtype': str(depth_array.dtype),
                'depth_min': float(depth_array.min()),
                'depth_max': float(depth_array.max()),
                'depth_mean': float(depth_array.mean()),
                'depth_std': float(depth_array.std()),
                'unique_values': int(len(np.unique(depth_array))),
                'sample_strategy': self.sample_strategy
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Sample depth data
            samples = self._sample_depth_data(depth_array)
            
            # Write to log file
            with open(log_path, 'w') as f:
                f.write("=== DEPTH ESTIMATION LOG ===\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Sample Strategy: {self.sample_strategy}\n\n")
                
                f.write("=== METADATA ===\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("=== DEPTH STATISTICS ===\n")
                f.write(f"Shape: {depth_array.shape}\n")
                f.write(f"Data Type: {depth_array.dtype}\n")
                f.write(f"Min: {depth_array.min():.6f}\n")
                f.write(f"Max: {depth_array.max():.6f}\n")
                f.write(f"Mean: {depth_array.mean():.6f}\n")
                f.write(f"Std: {depth_array.std():.6f}\n")
                f.write(f"Unique Values: {len(np.unique(depth_array))}\n")
                f.write("\n")
                
                # Histogram analysis
                f.write("=== DEPTH DISTRIBUTION ===\n")
                hist, bins = np.histogram(depth_array, bins=20)
                f.write("Histogram (20 bins):\n")
                for i, (count, bin_edge) in enumerate(zip(hist, bins[:-1])):
                    f.write(f"  Bin {i+1:2d}: {bin_edge:8.3f} - {bins[i+1]:8.3f}: {count:6d} pixels\n")
                f.write("\n")
                
                f.write("=== SAMPLED DEPTH VALUES ===\n")
                f.write(f"Total samples: {len(samples)}\n")
                f.write("Format: x, y, depth_value, location(if available)\n")
                f.write("-" * 50 + "\n")
                
                for i, sample in enumerate(samples):
                    location = sample.get('location', '')
                    f.write(f"{i+1:3d}: ({sample['x']:4d}, {sample['y']:4d}) = {sample['depth']:8.3f}")
                    if location:
                        f.write(f" [{location}]")
                    f.write("\n")
                
                # Add some analysis
                f.write("\n=== ANALYSIS ===\n")
                sample_depths = [s['depth'] for s in samples]
                f.write(f"Sample depth range: {min(sample_depths):.3f} - {max(sample_depths):.3f}\n")
                f.write(f"Sample depth mean: {np.mean(sample_depths):.3f}\n")
                f.write(f"Sample depth std: {np.std(sample_depths):.3f}\n")
                
                # Check for potential issues
                f.write("\n=== POTENTIAL ISSUES ===\n")
                if len(np.unique(depth_array)) <= 256:
                    f.write("⚠️  Limited depth resolution (≤256 unique values) - may indicate quantization\n")
                if depth_array.std() < 0.001:
                    f.write("⚠️  Very low depth variance - may indicate flat or invalid depth map\n")
                if depth_array.max() - depth_array.min() < 0.1:
                    f.write("⚠️  Small depth range - may indicate limited depth variation\n")
                
                # Check for NaN or infinite values
                if np.any(np.isnan(depth_array)):
                    f.write("⚠️  NaN values detected in depth map\n")
                if np.any(np.isinf(depth_array)):
                    f.write("⚠️  Infinite values detected in depth map\n")
                
                if not any([len(np.unique(depth_array)) <= 256, depth_array.std() < 0.001, 
                           depth_array.max() - depth_array.min() < 0.1,
                           np.any(np.isnan(depth_array)), np.any(np.isinf(depth_array))]):
                    f.write("✅ No obvious issues detected\n")
            
            print(f"DEBUG: Depth log saved to: {log_path}")
            return str(log_path)
            
        except Exception as e:
            print(f"Error logging depth data: {e}")
            return None
    
    def log_depth_comparison(self, depth_array1, depth_array2, image_path, model_names, description=""):
        """
        Log comparison between two depth arrays (e.g., before/after processing).
        
        Args:
            depth_array1 (np.ndarray): First depth array
            depth_array2 (np.ndarray): Second depth array
            image_path (str): Path to input image
            model_names (tuple): Names of the models/processing steps
            description (str): Description of the comparison
            
        Returns:
            str: Path to the created log file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"depth_comparison_{timestamp}.txt"
            log_path = self.log_dir / log_filename
            
            with open(log_path, 'w') as f:
                f.write("=== DEPTH COMPARISON LOG ===\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Description: {description}\n\n")
                
                f.write("=== ARRAY 1 ===\n")
                f.write(f"Model/Step: {model_names[0]}\n")
                f.write(f"Shape: {depth_array1.shape}\n")
                f.write(f"Min: {depth_array1.min():.6f}\n")
                f.write(f"Max: {depth_array1.max():.6f}\n")
                f.write(f"Mean: {depth_array1.mean():.6f}\n")
                f.write(f"Std: {depth_array1.std():.6f}\n")
                f.write(f"Unique Values: {len(np.unique(depth_array1))}\n\n")
                
                f.write("=== ARRAY 2 ===\n")
                f.write(f"Model/Step: {model_names[1]}\n")
                f.write(f"Shape: {depth_array2.shape}\n")
                f.write(f"Min: {depth_array2.min():.6f}\n")
                f.write(f"Max: {depth_array2.max():.6f}\n")
                f.write(f"Mean: {depth_array2.mean():.6f}\n")
                f.write(f"Std: {depth_array2.std():.6f}\n")
                f.write(f"Unique Values: {len(np.unique(depth_array2))}\n\n")
                
                f.write("=== DIFFERENCES ===\n")
                if depth_array1.shape == depth_array2.shape:
                    diff = depth_array2.astype(float) - depth_array1.astype(float)
                    f.write(f"Mean difference: {diff.mean():.6f}\n")
                    f.write(f"Std difference: {diff.std():.6f}\n")
                    f.write(f"Max difference: {diff.max():.6f}\n")
                    f.write(f"Min difference: {diff.min():.6f}\n")
                else:
                    f.write("⚠️  Arrays have different shapes - cannot compute differences\n")
            
            print(f"DEBUG: Depth comparison log saved to: {log_path}")
            return str(log_path)
            
        except Exception as e:
            print(f"Error logging depth comparison: {e}")
            return None


def create_depth_logger(log_dir="logs", max_log_files=10, sample_strategy="corners"):
    """
    Create a depth logger instance with sensible defaults.
    
    Args:
        log_dir (str): Directory to store log files
        max_log_files (int): Maximum number of depth log files to keep
        sample_strategy (str): Strategy for sampling depth data
        
    Returns:
        DepthLogger: Configured depth logger instance
    """
    return DepthLogger(log_dir=log_dir, max_log_files=max_log_files, sample_strategy=sample_strategy)
