"""
Log Capture Module

This module provides functionality to capture all print statements and log them to files.
Useful for debugging and tracking the execution flow of the 3D pose estimation pipeline.
"""

import os
import sys
from datetime import datetime
from io import StringIO


class LogCapture:
    """Capture all print statements and log them to a file"""
    
    def __init__(self, logs_dir=None):
        """
        Initialize the log capture system
        
        Args:
            logs_dir (str, optional): Directory to store log files. 
                                    If None, creates a 'logs' directory in the current file's directory.
        """
        if logs_dir is None:
            # Get the directory of the current file (utils/log_capture.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the obj_pose directory
            parent_dir = os.path.dirname(current_dir)
            self.logs_dir = os.path.join(parent_dir, 'logs')
        else:
            self.logs_dir = logs_dir
            
        self.ensure_logs_directory()
        self.original_stdout = sys.stdout
        self.log_buffer = StringIO()
        self.is_capturing = False
        
    def ensure_logs_directory(self):
        """Create logs directory if it doesn't exist"""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            print(f"Created logs directory: {self.logs_dir}")
    
    def start_capture(self):
        """Start capturing all print statements"""
        if not self.is_capturing:
            self.is_capturing = True
            self.log_buffer = StringIO()
            sys.stdout = self.log_buffer
            print(f"=== LOGGING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    def stop_capture(self):
        """Stop capturing and restore original stdout"""
        if self.is_capturing:
            self.is_capturing = False
            sys.stdout = self.original_stdout
    
    def write_log_to_file(self, filename_suffix=""):
        """
        Write captured logs to a file
        
        Args:
            filename_suffix (str): Additional suffix to add to the filename
            
        Returns:
            str or None: Path to the written log file, or None if failed
        """
        if not self.is_capturing:
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"preview_log_{timestamp}{filename_suffix}.txt"
        log_file_path = os.path.join(self.logs_dir, filename)
        
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(self.log_buffer.getvalue())
            print(f"✅ Logs written to: {log_file_path}")
            return log_file_path
        except Exception as e:
            print(f"❌ Error writing log file: {e}")
            return None
    
    def get_captured_logs(self):
        """
        Get the currently captured logs as a string
        
        Returns:
            str: The captured log content
        """
        return self.log_buffer.getvalue() if self.is_capturing else ""
    
    def clear_buffer(self):
        """Clear the current log buffer"""
        if self.is_capturing:
            self.log_buffer = StringIO()
    
    def get_logs_directory(self):
        """
        Get the logs directory path
        
        Returns:
            str: Path to the logs directory
        """
        return self.logs_dir


# Global log capture instance for easy access
log_capture = LogCapture()
