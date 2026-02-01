# =============================================================================
# L4 Detection - HySDG-ESD Calculator and Obstacle Classifier
# =============================================================================
# Core algorithms for obstacle analysis:
# - HySDG-ESD: Equivalent distance calculation with velocity consideration
# - Classifier: Static/Dynamic/Unknown obstacle classification
# =============================================================================

import numpy as np
from typing import List, Optional

from .types import ObstacleState

from .config import (
    HYSDG_LAMBDA_ESD,
    HYSDG_STATIC_VELOCITY_THRESHOLD,
    HYSDG_STATIC_DDOT_THRESHOLD,
    CLASSIFIER_MIN_FRAMES,
    CLASSIFIER_ACCURATE_THRESHOLD,
    CLASSIFIER_FAST_DYNAMIC_AVG_THRESHOLD,
    CLASSIFIER_FAST_DYNAMIC_MAX_THRESHOLD,
    CLASSIFIER_FAST_STATIC_AVG_THRESHOLD,
    CLASSIFIER_FAST_STATIC_MAX_THRESHOLD,
    CLASSIFIER_STATIC_VEL_THRESHOLD,
    CLASSIFIER_DYNAMIC_VEL_THRESHOLD,
    CLASSIFIER_STATIC_MAX_VEL,
    CLASSIFIER_STATIC_STD_VEL,
    CLASSIFIER_DYNAMIC_MAX_VEL,
    CLASSIFIER_DYNAMIC_STD_VEL,
    CLASSIFIER_VOTE_MARGIN,
    CLASSIFIER_HISTORY_WINDOW
)


class HySDGCalculator:
    """
    Calculates HySDG-ESD equivalent distance.
    
    Takes into account relative velocity between AGV and obstacle
    to compute a distance metric that captures approaching/receding dynamics.
    
    The equivalent distance d_eq is smaller when an obstacle is approaching
    (more dangerous) and larger when it's receding (less dangerous).
    """
    
    def __init__(self, lambda_esd: float = HYSDG_LAMBDA_ESD):
        """
        Args:
            lambda_esd: Scaling parameter for velocity component.
        """
        self.lambda_esd = lambda_esd
    
    def compute(self, obs_pos: np.ndarray, obs_vel: np.ndarray,
                agv_pos: np.ndarray, agv_vel: np.ndarray,
                prev_d_eq: Optional[float] = None,
                dt: float = 0.1) -> dict:
        """
        Calculate equivalent distance and its derivative.
        
        Args:
            obs_pos: Obstacle position in world frame
            obs_vel: Obstacle velocity
            agv_pos: AGV position
            agv_vel: AGV velocity
            prev_d_eq: Previous equivalent distance (for derivative)
            dt: Delta time
            
        Returns:
            Dictionary with d_eq, d_dot, state, distance, relative_velocity
        """
        # Relative position vector (obstacle relative to AGV)
        r_t = obs_pos - agv_pos
        
        # Relative velocity vector
        u_t = obs_vel - agv_vel
        
        # Euclidean distance
        d = np.linalg.norm(r_t)
        u_norm = np.linalg.norm(u_t)

        # Equivalent distance
        if u_norm < 1e-6:
            d_eq = d
        else:
            d_eq = d - self.lambda_esd * np.dot(r_t, u_t) / u_norm

        # Equivalent distance derivative
        if prev_d_eq is None:
            d_dot = 0.0
        else:
            d_dot = (d_eq - prev_d_eq) / dt

        # Preliminary classification based on velocity
        if u_norm < HYSDG_STATIC_VELOCITY_THRESHOLD and abs(d_dot) < HYSDG_STATIC_DDOT_THRESHOLD:
            state = ObstacleState.STATIC
        else:
            state = ObstacleState.DYNAMIC

        return {
            'd_eq': d_eq, 
            'd_dot': d_dot, 
            'state': state,
            'distance': d,
            'relative_velocity': u_norm
        }


class ObstacleClassifier:
    """
    Classifies obstacles as STATIC, DYNAMIC or UNKNOWN.
    
    Uses velocity and state history for robust decisions.
    Employs a multi-level classification strategy:
    1. Fast decision (4-7 frames): Quick threshold-based
    2. Accurate decision (8+ frames): Robust with voting
    """
    
    def __init__(self, 
                 min_frames_for_decision: int = CLASSIFIER_MIN_FRAMES,
                 static_vel_threshold: float = CLASSIFIER_STATIC_VEL_THRESHOLD,
                 dynamic_vel_threshold: float = CLASSIFIER_DYNAMIC_VEL_THRESHOLD):
        """
        Args:
            min_frames_for_decision: Minimum frames before classification
            static_vel_threshold: Velocity below which obstacle is STATIC
            dynamic_vel_threshold: Velocity above which obstacle is DYNAMIC
        """
        self.min_frames = min_frames_for_decision
        self.static_threshold = static_vel_threshold
        self.dynamic_threshold = dynamic_vel_threshold
    
    def classify(self, velocity_history: List[float], 
                 state_history: List[ObstacleState]) -> ObstacleState:
        """
        Classify an obstacle based on its history.
        
        Args:
            velocity_history: History of velocity magnitudes
            state_history: History of preliminary state classifications
            
        Returns:
            Final classified state
        """
        # Insufficient frames
        if len(velocity_history) < self.min_frames:
            return ObstacleState.UNKNOWN
        
        # Fast decision (4-7 frames)
        if len(velocity_history) < CLASSIFIER_ACCURATE_THRESHOLD:
            recent_velocities = velocity_history[-self.min_frames:]
            avg_velocity = np.mean(recent_velocities)
            max_velocity = np.max(recent_velocities)
            
            if avg_velocity > CLASSIFIER_FAST_DYNAMIC_AVG_THRESHOLD or \
               max_velocity > CLASSIFIER_FAST_DYNAMIC_MAX_THRESHOLD:
                return ObstacleState.DYNAMIC
            elif avg_velocity < CLASSIFIER_FAST_STATIC_AVG_THRESHOLD and \
                 max_velocity < CLASSIFIER_FAST_STATIC_MAX_THRESHOLD:
                return ObstacleState.STATIC
            else:
                return ObstacleState.UNKNOWN
        
        # Accurate decision (8+ frames)
        recent_velocities = velocity_history[-CLASSIFIER_HISTORY_WINDOW:]
        avg_velocity = np.mean(recent_velocities)
        std_velocity = np.std(recent_velocities)
        max_velocity = np.max(recent_velocities)
        
        # State vote count
        if len(state_history) >= CLASSIFIER_ACCURATE_THRESHOLD:
            recent_history = state_history[-CLASSIFIER_HISTORY_WINDOW:]
            static_count = sum(1 for s in recent_history if s == ObstacleState.STATIC)
            dynamic_count = sum(1 for s in recent_history if s == ObstacleState.DYNAMIC)
        else:
            static_count = 0
            dynamic_count = 0
        
        # Level 1: Definitely STATIC
        if (avg_velocity < CLASSIFIER_FAST_STATIC_AVG_THRESHOLD and 
            max_velocity < CLASSIFIER_STATIC_MAX_VEL and 
            std_velocity < CLASSIFIER_STATIC_STD_VEL):
            return ObstacleState.STATIC
        
        # Level 2: Definitely DYNAMIC
        elif (avg_velocity > self.dynamic_threshold or 
              max_velocity > CLASSIFIER_DYNAMIC_MAX_VEL or
              (avg_velocity > 0.30 and std_velocity > CLASSIFIER_DYNAMIC_STD_VEL)):
            return ObstacleState.DYNAMIC
        
        # Level 3: Vote based on history
        elif static_count > dynamic_count + CLASSIFIER_VOTE_MARGIN:
            return ObstacleState.STATIC
        elif dynamic_count > static_count + CLASSIFIER_VOTE_MARGIN:
            return ObstacleState.DYNAMIC
        
        # Level 4: Fall back to average velocity
        elif avg_velocity < self.static_threshold:
            return ObstacleState.STATIC
        else:
            return ObstacleState.DYNAMIC
