# =============================================================================
# L5 Decision - Base Decision Maker
# =============================================================================
# Abstract base class with common functionality shared by all algorithms.
# Eliminates code duplication for: stuck detection, wall following,
# angle normalization, safety score calculation.
# =============================================================================

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from .types import (
    TrackedObstacle,
    NavigationDecision,
    NavigationAction,
    ObstacleState
)

# Import configuration
from .config import (
    ROBOT_RADIUS,
    ROBOT_EFFECTIVE_RADIUS,
    MAX_LINEAR_VELOCITY,
    MIN_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    SIMULATION_DT,
    GOAL_TOLERANCE,
    LIDAR_VIRTUAL_RANGE
)


class BaseDecisionMaker(ABC):
    """
    Abstract base class for all navigation decision makers.
    
    Provides common functionality:
    - Stuck detection and recovery triggering
    - Wall-following behavior
    - Angle normalization
    - Safety score calculation
    - Velocity management
    
    Subclasses must implement the decide() method with their specific algorithm.
    """
    
    def __init__(self,
                 max_speed: float = MAX_LINEAR_VELOCITY,
                 min_speed: float = MIN_LINEAR_VELOCITY,
                 max_angular: float = MAX_ANGULAR_VELOCITY,
                 stuck_threshold: int = 15,
                 no_progress_threshold: int = 40,
                 wall_follow_distance: float = 1.0):
        """
        Initialize base decision maker.
        
        Args:
            max_speed: Maximum linear velocity (m/s)
            min_speed: Minimum linear velocity (m/s)
            max_angular: Maximum angular velocity (rad/s)
            stuck_threshold: Frames without movement to trigger recovery
            no_progress_threshold: Frames without goal progress to trigger recovery
            wall_follow_distance: Target distance from wall during recovery (m)
        """
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_angular = max_angular
        self.stuck_threshold = stuck_threshold
        self.no_progress_threshold = no_progress_threshold
        self.wall_follow_distance = wall_follow_distance
        
        # Current velocities
        self.current_v = 0.0
        self.current_w = 0.0
        
        # Recovery state
        self.in_recovery = False
        self.recovery_counter = 0
        self.recovery_direction = True  # True = left
        
        # Stuck detection
        self.stuck_counter = 0
        self.no_progress_counter = 0
        self.prev_pos: Optional[np.ndarray] = None
        self.prev_goal_dist: Optional[float] = None
        
    def reset(self):
        """Reset internal state to initial values."""
        self.current_v = 0.0
        self.current_w = 0.0
        self.in_recovery = False
        self.recovery_counter = 0
        self.stuck_counter = 0
        self.no_progress_counter = 0
        self.prev_pos = None
        self.prev_goal_dist = None

    # In L5_decision/base.py -> Inside the BaseDecisionMaker class

    def get_fused_obstacles(self, obstacles: List[TrackedObstacle]) -> List[TrackedObstacle]:
        """
        Sensor Fusion Filter:
        Only returns obstacles that have passed the Kinematic Consensus 
        (agreement between Raw Sensor and Kalman Filter).
        """
        # We only keep obstacles with confidence > 0.5
        # Objects with low confidence are usually sensor noise or 'disagreements'
        return [obs for obs in obstacles if obs.confidence > 0.5]
        
    # =========================================================================
    # Utility Methods (shared by all algorithms)
    # =========================================================================
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def update_stuck_detection(self,
                                pos: np.ndarray,
                                goal_dist: float,
                                goal_dir: float,
                                heading: float) -> bool:
        """
        Update stuck detection counters and determine if recovery is needed.
        
        Args:
            pos: Current robot position
            goal_dist: Current distance to goal
            goal_dir: Direction to goal (radians)
            heading: Current robot heading
            
        Returns:
            True if recovery mode should be activated
        """
        need_recovery = False
        
        # Check if robot is moving
        if self.prev_pos is not None:
            movement = np.linalg.norm(pos - self.prev_pos)
            if movement < 0.01:  # Less than 1cm movement
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
        # Check if making progress toward goal
        if self.prev_goal_dist is not None:
            progress = self.prev_goal_dist - goal_dist
            if progress < 0.02:  # Less than 2cm progress
                self.no_progress_counter += 1
            else:
                self.no_progress_counter = 0
                # Making progress - consider exiting recovery
                if self.in_recovery:
                    self.recovery_counter += 1
                    if self.recovery_counter > 30:
                        self.in_recovery = False
                        
        # Trigger recovery if stuck
        if (self.stuck_counter > self.stuck_threshold or 
            self.no_progress_counter > self.no_progress_threshold):
            need_recovery = True
            
            # Choose recovery direction based on goal (go opposite side)
            if not self.in_recovery:
                rel_goal = self.normalize_angle(goal_dir - heading)
                self.recovery_direction = rel_goal < 0
                
        # Update state for next call
        self.prev_pos = pos.copy()
        self.prev_goal_dist = goal_dist
        
        return need_recovery
    
    def wall_follow_control(self,
                            obstacles: List[TrackedObstacle],
                            pos: np.ndarray,
                            heading: float,
                            follow_left: bool = True) -> Tuple[float, float]:
        """
        Wall-following behavior for recovery mode.
        
        Finds the closest obstacle and computes a tangent direction
        to follow along its contour while maintaining target distance.
        
        Args:
            obstacles: List of tracked obstacles
            pos: Robot position
            heading: Robot heading
            follow_left: If True, keep wall on left side
            
        Returns:
            (v, w): Linear and angular velocities
        """
        if not obstacles:
            return self.max_speed * 0.5, 0.0
            
        # Find closest obstacle
        min_dist = float('inf')
        wall_angle = heading
        
        for obs in obstacles:
            rel_pos = obs.center - pos
            dist = np.linalg.norm(rel_pos)
            if dist < min_dist:
                min_dist = dist
                wall_angle = np.arctan2(rel_pos[1], rel_pos[0])
                
        # Tangent direction (perpendicular to wall direction)
        if follow_left:
            follow_angle = wall_angle + np.pi / 2
        else:
            follow_angle = wall_angle - np.pi / 2
            
        # Proportional correction to maintain desired distance
        dist_error = min_dist - self.wall_follow_distance
        correction = np.clip(dist_error * 0.4, -np.pi / 8, np.pi / 8)
        
        if follow_left:
            follow_angle -= correction
        else:
            follow_angle += correction
            
        # Angular velocity towards follow direction
        heading_diff = self.normalize_angle(follow_angle - heading)
        w = np.clip(heading_diff / SIMULATION_DT, -self.max_angular, self.max_angular)
        
        # Speed based on clearance
        speed_factor = np.clip((min_dist - ROBOT_EFFECTIVE_RADIUS) / 1.0, 0.2, 1.0)
        v = self.max_speed * 0.5 * speed_factor
        
        return v, w
    
    def compute_safety_score(self, 
                             obstacles: List[TrackedObstacle],
                             pos: np.ndarray) -> float:
        """
        Calculate a safety score (0-1).
        
        1.0 = very safe (no obstacles nearby)
        0.0 = very dangerous (collision imminent)
        
        Args:
            obstacles: List of tracked obstacles
            pos: Robot position
            
        Returns:
            Safety score between 0 and 1
        """
        if not obstacles:
            return 1.0
            
        min_dist = min(np.linalg.norm(obs.center - pos) for obs in obstacles)
        
        critical_dist = ROBOT_EFFECTIVE_RADIUS * 1.5
        warning_dist = ROBOT_EFFECTIVE_RADIUS * 4
        
        if min_dist < critical_dist:
            return 0.2
        elif min_dist < warning_dist:
            return 0.5
        else:
            return 0.9
    
    def get_clearance_in_direction(self,
                                   obstacles: List[TrackedObstacle],
                                   pos: np.ndarray,
                                   direction: float,
                                   cone_width: float = np.deg2rad(25)) -> float:
        """
        Get minimum clearance in a directional cone.
        
        Args:
            obstacles: List of tracked obstacles
            pos: Robot position
            direction: Direction to check (radians)
            cone_width: Width of the cone (radians)
            
        Returns:
            Minimum clearance in the specified direction
        """
        min_clearance = LIDAR_VIRTUAL_RANGE
        
        for obs in obstacles:
            rel_pos = obs.center - pos
            dist = np.linalg.norm(rel_pos)
            obs_angle = np.arctan2(rel_pos[1], rel_pos[0])
            
            angle_diff = abs(self.normalize_angle(obs_angle - direction))
            if angle_diff < cone_width:
                clearance = dist - ROBOT_EFFECTIVE_RADIUS
                min_clearance = min(min_clearance, clearance)
                
        return min_clearance
    
    def predict_trajectory(self,
                           pos: np.ndarray,
                           heading: float,
                           v: float,
                           w: float,
                           predict_time: float = 2.0,
                           dt: float = SIMULATION_DT) -> List[Tuple[float, float, float]]:
        """
        Predict trajectory for given velocity commands.
        
        Args:
            pos: Starting position [x, y]
            heading: Starting heading (radians)
            v: Linear velocity (m/s)
            w: Angular velocity (rad/s)
            predict_time: How far ahead to predict (seconds)
            dt: Time step for prediction
            
        Returns:
            List of (x, y, theta) along predicted trajectory
        """
        trajectory = []
        x, y, theta = pos[0], pos[1], heading
        
        steps = int(predict_time / dt)
        for _ in range(steps):
            theta += w * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            trajectory.append((x, y, theta))
            
        return trajectory
    
    def check_trajectory_clearance(self,
                                    trajectory: List[Tuple[float, float, float]],
                                    obstacles: List[TrackedObstacle]) -> Tuple[bool, float]:
        """
        Check trajectory clearance against obstacles.
        
        Args:
            trajectory: List of (x, y, theta) points
            obstacles: List of tracked obstacles
            
        Returns:
            (is_valid, min_clearance): Whether trajectory is collision-free
                                       and the minimum clearance found
        """
        if not obstacles:
            return True, float('inf')
            
        min_clearance = float('inf')
        
        for (x, y, _) in trajectory:
            traj_pos = np.array([x, y])
            
            for obs in obstacles:
                dist = np.linalg.norm(traj_pos - obs.center)
                clearance = dist - ROBOT_EFFECTIVE_RADIUS
                
                if clearance < 0:
                    return False, 0  # Collision
                    
                min_clearance = min(min_clearance, clearance)
                
        return True, min_clearance
    
    def compute_evasion_control(self,
                                obstacles: List[TrackedObstacle],
                                pos: np.ndarray,
                                heading: float) -> Tuple[float, float, float]:
        """
        Compute emergency evasion control when too close to obstacles.
        
        Args:
            obstacles: List of tracked obstacles
            pos: Robot position
            heading: Robot heading
            
        Returns:
            (v, w, heading_change): Velocities and heading change for evasion
        """
        if not obstacles:
            return self.min_speed, 0.0, 0.0
            
        # Find closest obstacle
        closest_obs = min(obstacles, key=lambda o: np.linalg.norm(o.center - pos))
        
        # Direction away from obstacle
        away_dir = pos - closest_obs.center
        away_angle = np.arctan2(away_dir[1], away_dir[0])
        
        # Rotate towards escape direction
        heading_diff = self.normalize_angle(away_angle - heading)
        evasion_w = np.clip(heading_diff * 2.0, -self.max_angular, self.max_angular)
        heading_change = evasion_w * SIMULATION_DT
        
        # Minimal speed to allow rotation
        evasion_v = max(self.min_speed, 0.05)
        
        return evasion_v, evasion_w, heading_change
    
    # =========================================================================
    # Abstract Method (must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def decide(self,
               obstacles: List[TrackedObstacle],
               agv_pos: np.ndarray,
               agv_heading: float,
               goal_pos: Optional[np.ndarray] = None,
               current_vel: Optional[Tuple[float, float]] = None) -> NavigationDecision:
        """
        Make a navigation decision based on obstacles and goal.
        
        This method must be implemented by each specific algorithm
        (DWA, VFH, GapNav, etc.).
        
        Args:
            obstacles: List of tracked obstacles from tracker
            agv_pos: AGV position [x, y]
            agv_heading: AGV heading in radians
            goal_pos: Goal position [x, y] (optional)
            current_vel: Current (linear, angular) velocities
            
        Returns:
            NavigationDecision (or subclass) with action and parameters
        """
        pass
