# =============================================================================
# L3 World Model - Simple Sensor Fusion
# =============================================================================

import numpy as np
from typing import List, Dict, Tuple
from collections import deque

class SimpleSensorFusion:
    """
    Simple probabilistic sensor fusion for obstacle tracking.
    Combines LiDAR measurements with motion predictions.
    """
    
    def __init__(self, max_range: float = 30.0, 
                 process_noise: float = 0.1,
                 measurement_noise: float = 0.05,
                 max_track_age: int = 10,
                 confidence_threshold: float = 0.7):
        """
        Initialize sensor fusion.
        
        Args:
            max_range: Maximum sensor range
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
            max_track_age: Frames before removing unobserved tracks
            confidence_threshold: Minimum confidence to consider track valid
        """
        self.max_range = max_range
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_track_age = max_track_age
        self.confidence_threshold = confidence_threshold
        
        # Tracked obstacles with state: [x, y, vx, vy, confidence, age]
        self.tracks: Dict[int, dict] = {}
        self.next_track_id = 0
        
        # Measurement history for consistency checking
        self.measurement_history = deque(maxlen=20)
        
    def cartesian_to_polar(self, x: float, y: float) -> Tuple[float, float]:
        """Convert Cartesian coordinates to polar."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta
    
    def polar_to_cartesian(self, r: float, theta: float) -> Tuple[float, float]:
        """Convert polar coordinates to Cartesian."""
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
    
    def predict(self, dt: float = 0.1):
        """
        Predict next state for all tracks using constant velocity model.
        """
        for track_id, track in self.tracks.items():
            # Apply constant velocity motion model
            track['state'][0] += track['state'][2] * dt  # x += vx * dt
            track['state'][1] += track['state'][3] * dt  # y += vy * dt
            
            # Add process noise
            track['state'][0] += np.random.normal(0, self.process_noise * dt)
            track['state'][1] += np.random.normal(0, self.process_noise * dt)
            
            # Slightly reduce velocity (damping)
            track['state'][2] *= 0.98
            track['state'][3] *= 0.98
            
            # Increase age
            track['age'] += 1
            track['frames_since_update'] += 1
            
            # Decay confidence for unobserved tracks
            if track['frames_since_update'] > 2:
                track['confidence'] *= 0.9
    
    def update(self, lidar_ranges: np.ndarray, lidar_angles: np.ndarray,
               agv_pos: np.ndarray, agv_heading: float) -> List[dict]:
        """
        Update tracks with new LiDAR measurements.
        
        Args:
            lidar_ranges: LiDAR range measurements
            lidar_angles: LiDAR angle measurements (relative to AGV heading)
            agv_pos: AGV position in world coordinates
            agv_heading: AGV heading in radians
            
        Returns:
            List of fused obstacle estimates
        """
        # Convert LiDAR measurements to world coordinates
        measurements = self._lidar_to_world_coords(
            lidar_ranges, lidar_angles, agv_pos, agv_heading
        )
        
        self.measurement_history.append(measurements)
        
        # Predict existing tracks
        self.predict()
        
        # Associate measurements with existing tracks
        assigned_measurements = set()
        
        for track_id, track in list(self.tracks.items()):
            # Skip if track is too old or low confidence
            if (track['frames_since_update'] > self.max_track_age or 
                track['confidence'] < 0.1):
                del self.tracks[track_id]
                continue
            
            # Find closest measurement
            min_distance = float('inf')
            best_measurement_idx = -1
            
            for i, meas in enumerate(measurements):
                if i in assigned_measurements:
                    continue
                    
                distance = np.linalg.norm(
                    track['state'][:2] - meas[:2]
                )
                
                if distance < min_distance and distance < 1.0:  # Association threshold
                    min_distance = distance
                    best_measurement_idx = i
            
            if best_measurement_idx >= 0:
                # Update track with measurement
                meas = measurements[best_measurement_idx]
                self._update_track(track, meas)
                assigned_measurements.add(best_measurement_idx)
            else:
                # No measurement, reduce confidence
                track['confidence'] *= 0.8
        
        # Create new tracks for unassigned measurements
        for i, meas in enumerate(measurements):
            if i not in assigned_measurements and meas[2] > 0.3:  # Minimum range
                self._create_new_track(meas)
        
        # Return fused estimates
        return self._get_fused_estimates()
    
    def _lidar_to_world_coords(self, ranges: np.ndarray, angles: np.ndarray,
                               agv_pos: np.ndarray, agv_heading: float) -> List[np.ndarray]:
        """Convert LiDAR data to world coordinates."""
        measurements = []
        
        for r, theta in zip(ranges, angles):
            if r < self.max_range * 0.95:  # Ignore max range readings
                # Relative to AGV
                x_rel = r * np.cos(theta + agv_heading)
                y_rel = r * np.sin(theta + agv_heading)
                
                # World coordinates
                x_world = agv_pos[0] + x_rel
                y_world = agv_pos[1] + y_rel
                
                # Simple velocity estimate using history
                velocity = self._estimate_velocity(x_world, y_world)
                
                # Initial confidence based on range (closer = more confident)
                confidence = 1.0 - (r / self.max_range) * 0.5
                
                measurements.append(np.array([
                    x_world, y_world, velocity[0], velocity[1], confidence
                ]))
        
        return measurements
    
    def _estimate_velocity(self, x: float, y: float) -> np.ndarray:
        """Estimate velocity using measurement history."""
        if len(self.measurement_history) < 2:
            return np.array([0.0, 0.0])
        
        # Find closest point in previous frame
        prev_measurements = self.measurement_history[-2]
        min_distance = float('inf')
        best_velocity = np.array([0.0, 0.0])
        
        for meas in prev_measurements:
            distance = np.sqrt((x - meas[0])**2 + (y - meas[1])**2)
            if distance < min_distance and distance < 0.5:  # Matching threshold
                min_distance = distance
                # Simple velocity = displacement / dt (dt = 0.1s default)
                best_velocity = np.array([
                    (x - meas[0]) / 0.1,
                    (y - meas[1]) / 0.1
                ])
        
        return best_velocity
    
    def _update_track(self, track: dict, measurement: np.ndarray):
        """Update a track with a new measurement using simple Kalman-like fusion."""
        # Measurement weights based on confidence
        meas_conf = measurement[4]
        track_conf = track['confidence']
        
        # Combined weight
        total_conf = meas_conf + track_conf
        meas_weight = meas_conf / total_conf
        track_weight = track_conf / total_conf
        
        # Fuse position (weighted average)
        track['state'][0] = (track_weight * track['state'][0] + 
                           meas_weight * measurement[0])
        track['state'][1] = (track_weight * track['state'][1] + 
                           meas_weight * measurement[1])
        
        # Fuse velocity with some inertia
        track['state'][2] = 0.7 * track['state'][2] + 0.3 * measurement[2]
        track['state'][3] = 0.7 * track['state'][3] + 0.3 * measurement[3]
        
        # Update confidence
        track['confidence'] = min(1.0, track['confidence'] * 0.9 + meas_conf * 0.2)
        track['frames_since_update'] = 0
        track['last_update'] = measurement[:2]
    
    def _create_new_track(self, measurement: np.ndarray):
        """Create a new track from a measurement."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        self.tracks[track_id] = {
            'state': measurement.copy(),  # [x, y, vx, vy, confidence]
            'confidence': measurement[4],
            'age': 0,
            'frames_since_update': 0,
            'last_update': measurement[:2],
            'type': 'dynamic' if np.linalg.norm(measurement[2:4]) > 0.3 else 'static'
        }
    
    def _get_fused_estimates(self) -> List[dict]:
        """Get current fused obstacle estimates."""
        estimates = []
        
        for track_id, track in self.tracks.items():
            if track['confidence'] >= self.confidence_threshold:
                # Convert to standard obstacle format
                estimates.append({
                    'center': track['state'][:2].copy(),
                    'velocity': track['state'][2:4].copy(),
                    'confidence': track['confidence'],
                    'type': track['type'],
                    'track_id': track_id
                })
        
        return estimates
    
    def clear_tracks(self):
        """Clear all tracks."""
        self.tracks.clear()
        self.measurement_history.clear()
        self.next_track_id = 0
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracks)