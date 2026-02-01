# =============================================================================
# L5 Decision - DWA (Dynamic Window Approach) Algorithm
# =============================================================================
# Samples velocities within a dynamic window and evaluates trajectories
# based on obstacle clearance, goal heading, and velocity preference.
# =============================================================================

import numpy as np
from typing import List, Optional, Tuple

from ..base import BaseDecisionMaker
from ..types import (
    TrackedObstacle,
    DWANavigationDecision,
    NavigationAction
)

# Import configuration
from ..config import (
    ROBOT_EFFECTIVE_RADIUS,
    MAX_LINEAR_VELOCITY,
    MIN_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    MAX_LINEAR_ACCEL,
    MAX_ANGULAR_ACCEL,
    DWA_VELOCITY_SAMPLES,
    DWA_ANGULAR_SAMPLES,
    DWA_PREDICT_TIME,
    DWA_PREDICT_DT,
    DWA_WEIGHT_HEADING,
    DWA_WEIGHT_CLEARANCE,
    DWA_WEIGHT_VELOCITY,
    DWA_WEIGHT_PROGRESS,
    DWA_WALL_FOLLOW_DISTANCE,
    DWA_STUCK_THRESHOLD,
    DWA_NO_PROGRESS_THRESHOLD,
    SIMULATION_DT,
    GOAL_TOLERANCE
)


class DWADecisionMaker(BaseDecisionMaker):
    """
    Dynamic Window Approach based navigation decision maker.
    
    DWA samples velocity pairs (v, w) within a dynamic window constrained
    by acceleration limits and evaluates predicted trajectories based on:
    - Heading towards goal
    - Clearance from obstacles
    - Velocity preference (faster is better)
    - Progress towards goal
    
    Includes wall-following recovery for local minima escape.
    
    Best for: Environments requiring smooth trajectory planning with
              consideration for robot dynamics.
    """
    
    def __init__(self,
                 max_speed: float = MAX_LINEAR_VELOCITY,
                 min_speed: float = MIN_LINEAR_VELOCITY,
                 max_angular: float = MAX_ANGULAR_VELOCITY,
                 max_accel: float = MAX_LINEAR_ACCEL,
                 max_angular_accel: float = MAX_ANGULAR_ACCEL,
                 velocity_samples: int = DWA_VELOCITY_SAMPLES,
                 angular_samples: int = DWA_ANGULAR_SAMPLES,
                 predict_time: float = DWA_PREDICT_TIME):
        """
        Initialize DWA decision maker.
        
        Args:
            max_speed: Maximum linear velocity (m/s)
            min_speed: Minimum linear velocity (m/s)
            max_angular: Maximum angular velocity (rad/s)
            max_accel: Maximum linear acceleration (m/s²)
            max_angular_accel: Maximum angular acceleration (rad/s²)
            velocity_samples: Number of linear velocity samples
            angular_samples: Number of angular velocity samples
            predict_time: Trajectory prediction horizon (seconds)
        """
        super().__init__(
            max_speed=max_speed,
            min_speed=min_speed,
            max_angular=max_angular,
            stuck_threshold=DWA_STUCK_THRESHOLD,
            no_progress_threshold=DWA_NO_PROGRESS_THRESHOLD,
            wall_follow_distance=DWA_WALL_FOLLOW_DISTANCE
        )
        
        self.max_accel = max_accel
        self.max_angular_accel = max_angular_accel
        self.velocity_samples = velocity_samples
        self.angular_samples = angular_samples
        self.predict_time = predict_time
        
    def get_dynamic_window(self) -> Tuple[float, float, float, float]:
        """
        Calculate the dynamic window based on current velocities
        and acceleration limits.
        
        Returns:
            (v_min, v_max, w_min, w_max): Velocity bounds
        """
        dt = SIMULATION_DT
        accel_factor = 2.0  # Allow faster velocity changes
        
        v_min = max(self.min_speed, self.current_v - self.max_accel * dt * accel_factor)
        v_max = min(self.max_speed, self.current_v + self.max_accel * dt * accel_factor)
        w_min = max(-self.max_angular, self.current_w - self.max_angular_accel * dt * accel_factor)
        w_max = min(self.max_angular, self.current_w + self.max_angular_accel * dt * accel_factor)
        
        # Ensure minimum velocity range
        if v_max < self.max_speed * 0.5:
            v_max = min(self.max_speed, v_max + self.max_accel * dt)
        
        return v_min, v_max, w_min, w_max
    
    def evaluate_trajectory(self,
                            trajectory: List[Tuple[float, float, float]],
                            v: float,
                            goal_pos: np.ndarray,
                            pos: np.ndarray,
                            min_clearance: float) -> float:
        """
        Evaluate a trajectory using DWA scoring function.
        
        Args:
            trajectory: Predicted trajectory
            v: Linear velocity
            goal_pos: Goal position
            pos: Current position
            min_clearance: Minimum clearance along trajectory
            
        Returns:
            Combined score (higher is better)
        """
        if not trajectory:
            return -float('inf')
            
        final_x, final_y, final_heading = trajectory[-1]
        final_pos = np.array([final_x, final_y])
        
        # Goal direction
        goal_dir = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        dist_to_goal = np.linalg.norm(goal_pos - pos)
        
        # 1. Heading score: alignment with goal
        heading_diff = abs(self.normalize_angle(goal_dir - final_heading))
        heading_score = (np.pi - heading_diff) / np.pi
        
        # 2. Clearance score
        clearance_score = np.clip(min_clearance / 2.0, 0, 1)
        
        # 3. Velocity score: faster is better
        velocity_score = v / self.max_speed
        
        # 4. Progress score: getting closer to goal
        final_goal_dist = np.linalg.norm(goal_pos - final_pos)
        progress = (dist_to_goal - final_goal_dist) / max(dist_to_goal, 0.1)
        progress_score = np.clip(progress, -1, 1)
        
        # Combined score
        return (
            DWA_WEIGHT_HEADING * heading_score +
            DWA_WEIGHT_CLEARANCE * clearance_score +
            DWA_WEIGHT_VELOCITY * velocity_score +
            DWA_WEIGHT_PROGRESS * progress_score
        )
    
    def decide(self,
               obstacles: List[TrackedObstacle],
               agv_pos: np.ndarray,
               agv_heading: float,
               goal_pos: Optional[np.ndarray] = None,
               current_vel: Optional[Tuple[float, float]] = None) -> DWANavigationDecision:
        """
        Make a navigation decision using DWA algorithm.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position [x, y]
            agv_heading: AGV heading in radians
            goal_pos: Goal position [x, y]
            current_vel: Current (linear, angular) velocities
            
        Returns:
            DWANavigationDecision with action and parameters
        """
        # Update current velocities
        if current_vel is not None:
            self.current_v, self.current_w = current_vel
            
        # Default goal is forward
        if goal_pos is None:
            goal_pos = agv_pos + 10.0 * np.array([np.cos(agv_heading), np.sin(agv_heading)])
            
        goal_dir = np.arctan2(goal_pos[1] - agv_pos[1], goal_pos[0] - agv_pos[0])
        goal_dist = np.linalg.norm(goal_pos - agv_pos)
        
        # Goal reached check
        if goal_dist < GOAL_TOLERANCE:
            return DWANavigationDecision(
                action=NavigationAction.CONTINUE,
                target_speed=0.0,
                target_heading_change=0.0,
                reason="Goal reached",
                critical_obstacles=[],
                safety_score=1.0,
                linear_velocity=0.0,
                angular_velocity=0.0,
                in_recovery=False
            )
        
        # Fast path: No obstacles - go straight to goal
        min_obs_dist = min((np.linalg.norm(obs.center - agv_pos) for obs in obstacles),
                           default=float('inf'))
        if not obstacles or min_obs_dist > 3.0:
            heading_diff = self.normalize_angle(goal_dir - agv_heading)
            heading_change = np.clip(heading_diff, -self.max_angular * SIMULATION_DT, 
                                     self.max_angular * SIMULATION_DT)
            self.current_v = self.max_speed
            self.current_w = heading_change / SIMULATION_DT
            return DWANavigationDecision(
                action=NavigationAction.CONTINUE,
                target_speed=self.max_speed,
                target_heading_change=heading_change,
                reason="No obstacles - direct to goal",
                critical_obstacles=[],
                safety_score=1.0,
                linear_velocity=self.max_speed,
                angular_velocity=self.current_w,
                in_recovery=False
            )
            
        # Update stuck detection
        need_recovery = self.update_stuck_detection(agv_pos, goal_dist, goal_dir, agv_heading)
        
        # Enter recovery mode if needed
        if need_recovery and not self.in_recovery:
            self.in_recovery = True
            self.recovery_counter = 0
            self.stuck_counter = 0
            self.no_progress_counter = 0
            
        # Recovery mode timeout
        if self.in_recovery:
            self.recovery_counter += 1
            if self.recovery_counter > 80:
                self.recovery_direction = not self.recovery_direction
                self.recovery_counter = 0
                
        # Critical obstacles
        critical_obs = [obs.id for obs in obstacles 
                       if np.linalg.norm(obs.center - agv_pos) < ROBOT_EFFECTIVE_RADIUS * 3]
        
        # Control selection
        if self.in_recovery:
            # Wall-following control
            best_v, best_w = self.wall_follow_control(
                obstacles, agv_pos, agv_heading, self.recovery_direction
            )
            best_trajectory = self.predict_trajectory(agv_pos, agv_heading, best_v, best_w,
                                                      self.predict_time, DWA_PREDICT_DT)
            reason = f"Recovery: Wall-follow ({'left' if self.recovery_direction else 'right'})"
        else:
            # DWA control
            v_min, v_max, w_min, w_max = self.get_dynamic_window()
            
            v_samples = np.linspace(v_min, v_max, self.velocity_samples)
            w_samples = np.linspace(w_min, w_max, self.angular_samples)
            
            best_score = -float('inf')
            best_v, best_w = 0.0, 0.0
            best_trajectory = []
            
            for v in v_samples:
                for w in w_samples:
                    # Predict trajectory
                    trajectory = self.predict_trajectory(agv_pos, agv_heading, v, w,
                                                        self.predict_time, DWA_PREDICT_DT)
                    
                    # Check clearance
                    valid, min_clearance = self.check_trajectory_clearance(trajectory, obstacles)
                    
                    if not valid:
                        continue
                        
                    # Evaluate
                    score = self.evaluate_trajectory(trajectory, v, goal_pos, agv_pos, min_clearance)
                    
                    if score > best_score:
                        best_score = score
                        best_v, best_w = v, w
                        best_trajectory = trajectory
                        
            if best_score == -float('inf'):
                # No valid trajectory - trigger recovery
                self.in_recovery = True
                self.recovery_counter = 0
                best_v, best_w = self.wall_follow_control(
                    obstacles, agv_pos, agv_heading, self.recovery_direction
                )
                best_trajectory = self.predict_trajectory(agv_pos, agv_heading, best_v, best_w,
                                                          self.predict_time, DWA_PREDICT_DT)
                reason = "DWA: No valid trajectory, entering recovery"
            else:
                reason = f"DWA: v={best_v:.2f}, w={np.degrees(best_w):.1f}°/s"
                
        # Update current velocities
        self.current_v = best_v
        self.current_w = best_w
        
        # Calculate heading change
        heading_change = best_w * SIMULATION_DT
        
        # Emergency evasion if too close
        if min_obs_dist < ROBOT_EFFECTIVE_RADIUS * 1.5:
            best_v, evasion_w, heading_change = self.compute_evasion_control(
                obstacles, agv_pos, agv_heading
            )
            best_w = evasion_w
            
            if not self.in_recovery:
                self.in_recovery = True
                self.recovery_counter = 0
                
            action = NavigationAction.TURN_LEFT if evasion_w > 0 else NavigationAction.TURN_RIGHT
            reason = "Evading obstacle - rotating away"
        elif abs(best_w) > np.deg2rad(30):
            action = NavigationAction.TURN_LEFT if best_w > 0 else NavigationAction.TURN_RIGHT
        elif best_v < self.max_speed * 0.3:
            action = NavigationAction.SLOW_DOWN
        else:
            action = NavigationAction.CONTINUE
            
        # Safety score
        safety_score = self.compute_safety_score(obstacles, agv_pos)
            
        return DWANavigationDecision(
            action=action,
            target_speed=best_v,
            target_heading_change=heading_change,
            reason=reason,
            critical_obstacles=critical_obs,
            safety_score=safety_score,
            linear_velocity=best_v,
            angular_velocity=best_w,
            predicted_trajectory=best_trajectory,
            in_recovery=self.in_recovery
        )
