# =============================================================================
# L4 Detection - Extended Kalman Filter
# =============================================================================
# EKF with Constant Velocity model for obstacle tracking.
# =============================================================================

import numpy as np
from typing import Tuple

from .config import (
    EKF_PROCESS_NOISE,
    EKF_MEASUREMENT_NOISE,
    EKF_INITIAL_COVARIANCE,
    EKF_MAHALANOBIS_THRESHOLD,
    EKF_INITIAL_DAMPING,
    EKF_INTERMEDIATE_DAMPING,
    EKF_CONVERGED_DAMPING,
    EKF_INNOVATION_THRESHOLD,
    EKF_POSITION_HISTORY_LENGTH,
    EKF_INNOVATION_HISTORY_LENGTH
)


class ExtendedKalmanFilterCV:
    """
    Extended Kalman Filter with Constant Velocity model.
    
    State: [x, y, vx, vy]
    Observation: [x, y]
    
    Used to track obstacle position and estimate velocity.
    """
    
    def __init__(self, dt: float, 
                 process_noise: float = EKF_PROCESS_NOISE, 
                 measurement_noise: float = EKF_MEASUREMENT_NOISE):
        """
        Initialize the EKF.
        
        Args:
            dt: Delta time between measurements
            process_noise: Process noise parameter
            measurement_noise: Measurement noise parameter
        """
        self.dt = dt
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = process_noise
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ]) * 0.3
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # State covariance
        self.P = np.eye(4) * EKF_INITIAL_COVARIANCE
        
        # State vector [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        
        # Flags and counters
        self.initialized = False
        self.update_count = 0
        
        # History for analysis
        self.innovation_history = []
        self.position_history = []

    def predict(self):
        """Prediction step: propagate state using motion model."""
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update step: incorporate new measurement.
        
        Args:
            z: Measurement [x, y]
            
        Returns:
            Updated state vector
        """
        z = np.array(z).reshape(2, 1)
        
        if not self.initialized:
            self.x[0:2] = z
            self.x[2:4] = 0  # Initial velocity zero
            self.initialized = True
            self.update_count = 1
            self.position_history = [z.copy()]
            return self.x.copy()
        
        self.update_count += 1
        
        # Estimate initial velocity from first 2 frames
        if self.update_count == 2:
            delta_pos = z - self.position_history[0]
            estimated_vel = delta_pos / self.dt
            self.x[2:4] = estimated_vel
        
        # Save position for initial frames
        if len(self.position_history) < 3:
            self.position_history.append(z.copy())
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        try:
            S_inv = np.linalg.inv(S)
            mahal_dist = np.sqrt(y.T @ S_inv @ y)[0, 0]
        except:
            S_inv = np.eye(2) * 0.1
            mahal_dist = 0
        
        # Gating: reject outlier measurements
        if mahal_dist > EKF_MAHALANOBIS_THRESHOLD:
            return self.x.copy()
        
        # Kalman Gain
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        
        # Update innovation history
        self.innovation_history.append(np.linalg.norm(y))
        if len(self.innovation_history) > EKF_INNOVATION_HISTORY_LENGTH:
            self.innovation_history.pop(0)
        
        # Adaptive velocity damping
        avg_innovation = np.mean(self.innovation_history) if self.innovation_history else 0
        
        if self.update_count <= 3:
            damping = EKF_INITIAL_DAMPING
        elif self.update_count < 8:
            if avg_innovation < EKF_INNOVATION_THRESHOLD:
                damping = EKF_INTERMEDIATE_DAMPING
            else:
                damping = 1.0
        else:
            if avg_innovation < EKF_INNOVATION_THRESHOLD:
                damping = EKF_CONVERGED_DAMPING
            else:
                damping = 1.0
        
        self.x[2:4] *= damping
        
        # Update covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x.copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get estimated position and velocity.
        
        Returns:
            Tuple (position [x,y], velocity [vx,vy])
        """
        pos = self.x[0:2].flatten()
        vel = self.x[2:4].flatten()
        return pos, vel
    
    def get_position(self) -> np.ndarray:
        """Returns estimated position [x, y]."""
        return self.x[0:2].flatten()
    
    def get_velocity(self) -> np.ndarray:
        """Returns estimated velocity [vx, vy]."""
        return self.x[2:4].flatten()
    
    def get_velocity_magnitude(self) -> float:
        """Returns velocity magnitude (speed)."""
        return np.linalg.norm(self.x[2:4])
