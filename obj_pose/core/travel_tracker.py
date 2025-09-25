"""
Travel tracking module.
Handles velocity and speed calculation for electrode and filler rod movement.
"""

import numpy as np
import time
from collections import deque
from config.settings import *


class TravelTracker:
    """Tracks electrode and filler rod movement for velocity and speed calculation"""
    
    def __init__(self, history_length=TRAVEL_HISTORY_LENGTH):
        """Initialize travel tracker for velocity and speed calculation"""
        self.history_length = history_length
        self.electrode_history = deque(maxlen=history_length)
        self.rod_history = deque(maxlen=history_length)
        self.frame_times = deque(maxlen=history_length)
        
        # Smoothed velocity vectors
        self.electrode_velocity = None
        self.rod_velocity = None
        
        # Speed calculations
        self.electrode_speed = 0.0
        self.rod_speed = 0.0
        self.speed_history = deque(maxlen=SPEED_AVERAGE_WINDOW)
        
    def update_tracking(self, electrode_tip_3d, rod_tip_3d, frame_time=None):
        """Update tracking with new electrode and rod tip positions"""
        if frame_time is None:
            frame_time = time.time()
        
        # Update electrode tracking
        if electrode_tip_3d is not None:
            self.electrode_history.append(electrode_tip_3d)
        
        # Update rod tracking
        if rod_tip_3d is not None:
            self.rod_history.append(rod_tip_3d)
        
        # Update frame timing
        self.frame_times.append(frame_time)
        
        # Calculate velocities if we have enough history
        if len(self.electrode_history) >= 2 and len(self.frame_times) >= 2:
            self._calculate_velocities()
    
    def _calculate_velocities(self):
        """Calculate velocity vectors for electrode and rod"""
        # Calculate electrode velocity
        if len(self.electrode_history) >= 2:
            current_pos = np.array(self.electrode_history[-1])
            previous_pos = np.array(self.electrode_history[-2])
            
            # Calculate time delta
            time_delta = self.frame_times[-1] - self.frame_times[-2]
            if time_delta > 0:
                # Calculate instantaneous velocity
                instant_velocity = (current_pos - previous_pos) / time_delta
                
                # Apply exponential smoothing
                if self.electrode_velocity is None:
                    self.electrode_velocity = instant_velocity
                else:
                    self.electrode_velocity = (TRAVEL_SMOOTHING_FACTOR * self.electrode_velocity + 
                                             (1 - TRAVEL_SMOOTHING_FACTOR) * instant_velocity)
                
                # Calculate speed magnitude
                speed_magnitude = np.linalg.norm(self.electrode_velocity)
                
                # Filter out very small movements
                if speed_magnitude > TRAVEL_MIN_VELOCITY:
                    self.electrode_speed = speed_magnitude
                    self.speed_history.append(self.electrode_speed)
                else:
                    self.electrode_speed = 0.0
        
        # Calculate rod velocity (similar to electrode)
        if len(self.rod_history) >= 2:
            current_pos = np.array(self.rod_history[-1])
            previous_pos = np.array(self.rod_history[-2])
            
            time_delta = self.frame_times[-1] - self.frame_times[-2]
            if time_delta > 0:
                instant_velocity = (current_pos - previous_pos) / time_delta
                
                if self.rod_velocity is None:
                    self.rod_velocity = instant_velocity
                else:
                    self.rod_velocity = (TRAVEL_SMOOTHING_FACTOR * self.rod_velocity + 
                                       (1 - TRAVEL_SMOOTHING_FACTOR) * instant_velocity)
                
                speed_magnitude = np.linalg.norm(self.rod_velocity)
                if speed_magnitude > TRAVEL_MIN_VELOCITY:
                    self.rod_speed = speed_magnitude
                else:
                    self.rod_speed = 0.0
    
    def get_electrode_velocity(self):
        """Get the current electrode velocity vector"""
        return self.electrode_velocity
    
    def get_electrode_speed(self):
        """Get the current electrode speed in m/s"""
        return self.electrode_speed
    
    def get_average_speed(self):
        """Get the average speed over the last N frames"""
        if not self.speed_history:
            return 0.0
        return sum(self.speed_history) / len(self.speed_history)
    
    def get_speed_in_units(self, speed_mps, units="mm/s"):
        """Convert speed from m/s to specified units"""
        if units == "mm/s":
            return speed_mps * 1000
        elif units == "cm/s":
            return speed_mps * 100
        elif units == "m/s":
            return speed_mps
        else:
            return speed_mps * 1000  # Default to mm/s
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.electrode_history.clear()
        self.rod_history.clear()
        self.frame_times.clear()
        self.electrode_velocity = None
        self.rod_velocity = None
        self.electrode_speed = 0.0
        self.rod_speed = 0.0
        self.speed_history.clear()
