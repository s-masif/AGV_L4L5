# =============================================================================
# L5 Alternative Decision Layer - DWA (Dynamic Window Approach)
# =============================================================================
# This module implements DWA-based navigation as an alternative to the default
# NavigationDecisionMaker. DWA samples velocities within a dynamic window
# and evaluates trajectories based on obstacle clearance, goal heading,
# and velocity preference.
#
# Usage:
#   from L5_decision_dwa import DWADecisionMaker, DWANavigationDecision
#   dwa = DWADecisionMaker()
#   decision = dwa.decide(tracked_obstacles, agv_pos, agv_heading, agv_vel)
# =============================================================================

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

# Import from L5 for compatibility
from L5_decision_layer import (
    TrackedObstacle,
    NavigationAction,
    ObstacleState
)

# Import configuration
from config_L5_alternatives import (
    ROBOT_RADIUS,
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
    DWA_MIN_CLEARANCE_THRESHOLD,
    DWA_STUCK_THRESHOLD,
    DWA_NO_PROGRESS_THRESHOLD,
    SIMULATION_DT,
    GOAL_TOLERANCE
)


# =============================================================================
# DWA Navigation Decision
# =============================================================================

@dataclass
class DWANavigationDecision:
    """DWA-specific navigation decision with trajectory info."""
    action: NavigationAction
    target_speed: float
    target_heading_change: float
    reason: str
    critical_obstacles: List[int]
    safety_score: float
    linear_velocity: float = 0.0      # Recommended linear velocity
    angular_velocity: float = 0.0     # Recommended angular velocity
    predicted_trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    in_recovery: bool = False


# =============================================================================
# DWA Decision Maker
# =============================================================================

class DWADecisionMaker:
    """
    Dynamic Window Approach based navigation decision maker.
    
    DWA samples velocity pairs (v, w) within a dynamic window constrained
    by acceleration limits and evaluates predicted trajectories based on:
    - Heading towards goal
    - Clearance from obstacles
    - Velocity preference
    - Progress towards goal
    
    Includes wall-following recovery for local minima escape.
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
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_angular = max_angular
        self.max_accel = max_accel
        self.max_angular_accel = max_angular_accel
        self.velocity_samples = velocity_samples
        self.angular_samples = angular_samples
        self.predict_time = predict_time
        
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
        self.prev_pos = None
        self.prev_goal_dist = None
        
    def reset(self):
        """Reset internal state."""
        self.current_v = 0.0
        self.current_w = 0.0
        self.in_recovery = False
        self.recovery_counter = 0
        self.stuck_counter = 0
        self.no_progress_counter = 0
        self.prev_pos = None
        self.prev_goal_dist = None
        
    def get_dynamic_window(self) -> Tuple[float, float, float, float]:
        """
        Calculate the dynamic window based on current velocities
        and acceleration limits.
        
        Returns:
            (v_min, v_max, w_min, w_max): Velocity bounds
        """
        dt = SIMULATION_DT
        
        # Use larger acceleration window for more responsive control
        accel_factor = 2.0  # Allow faster velocity changes
        
        v_min = max(self.min_speed, self.current_v - self.max_accel * dt * accel_factor)
        v_max = min(self.max_speed, self.current_v + self.max_accel * dt * accel_factor)
        w_min = max(-self.max_angular, self.current_w - self.max_angular_accel * dt * accel_factor)
        w_max = min(self.max_angular, self.current_w + self.max_angular_accel * dt * accel_factor)
        
        # Ensure we always have some minimum velocity range to sample
        if v_max < self.max_speed * 0.5:
            v_max = min(self.max_speed, v_max + self.max_accel * dt)
        
        return v_min, v_max, w_min, w_max
    
    def predict_trajectory(self,
                           pos: np.ndarray,
                           heading: float,
                           v: float,
                           w: float) -> List[Tuple[float, float, float]]:
        """
        Predict trajectory for given velocity commands.
        
        Args:
            pos: Starting position [x, y]
            heading: Starting heading (radians)
            v: Linear velocity (m/s)
            w: Angular velocity (rad/s)
            
        Returns:
            List of (x, y, theta) along predicted trajectory
        """
        trajectory = []
        x, y, theta = pos[0], pos[1], heading
        
        steps = int(self.predict_time / DWA_PREDICT_DT)
        for _ in range(steps):
            theta += w * DWA_PREDICT_DT
            x += v * np.cos(theta) * DWA_PREDICT_DT
            y += v * np.sin(theta) * DWA_PREDICT_DT
            trajectory.append((x, y, theta))
            
        return trajectory
    
    def check_trajectory_clearance(self,
                                    trajectory: List[Tuple[float, float, float]],
                                    obstacles: List[TrackedObstacle],
                                    pos: np.ndarray) -> Tuple[bool, float]:
        """
        Check trajectory clearance against obstacles.
        
        Args:
            trajectory: List of (x, y, theta) points
            obstacles: List of tracked obstacles
            pos: Current robot position
            
        Returns:
            (is_valid, min_clearance): Validity and minimum clearance
        """
        if not obstacles:
            return True, float('inf')
            
        min_clearance = float('inf')
        
        for (x, y, theta) in trajectory:
            traj_pos = np.array([x, y])
            
            for obs in obstacles:
                dist = np.linalg.norm(traj_pos - obs.center)
                clearance = dist - ROBOT_EFFECTIVE_RADIUS
                
                if clearance < 0:
                    return False, 0  # Collision
                    
                min_clearance = min(min_clearance, clearance)
                
        return True, min_clearance
    
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
        heading_diff = abs(self._normalize_angle(goal_dir - final_heading))
        heading_score = (np.pi - heading_diff) / np.pi
        
        # 2. Clearance score: distance from obstacles
        clearance_score = np.clip(min_clearance / 2.0, 0, 1)
        
        # 3. Velocity score: faster is better
        velocity_score = v / self.max_speed
        
        # 4. Progress score: getting closer to goal
        final_goal_dist = np.linalg.norm(goal_pos - final_pos)
        progress = (dist_to_goal - final_goal_dist) / max(dist_to_goal, 0.1)
        progress_score = np.clip(progress, -1, 1)
        
        # Combined score
        score = (
            DWA_WEIGHT_HEADING * heading_score +
            DWA_WEIGHT_CLEARANCE * clearance_score +
            DWA_WEIGHT_VELOCITY * velocity_score +
            DWA_WEIGHT_PROGRESS * progress_score
        )
        
        return score
    
    def wall_follow_control(self,
                            obstacles: List[TrackedObstacle],
                            pos: np.ndarray,
                            heading: float,
                            follow_left: bool = True) -> Tuple[float, float]:
        """
        Wall-following behavior for recovery mode.
        
        Args:
            obstacles: List of tracked obstacles
            pos: Robot position
            heading: Robot heading
            follow_left: Follow wall on left if True
            
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
                
        # Tangent direction
        if follow_left:
            follow_angle = wall_angle + np.pi / 2
        else:
            follow_angle = wall_angle - np.pi / 2
            
        # Correction to maintain distance
        dist_error = min_dist - DWA_WALL_FOLLOW_DISTANCE
        correction = np.clip(dist_error * 0.4, -np.pi / 8, np.pi / 8)
        
        if follow_left:
            follow_angle -= correction
        else:
            follow_angle += correction
            
        # Angular velocity towards follow direction
        heading_diff = self._normalize_angle(follow_angle - heading)
        w = np.clip(heading_diff / SIMULATION_DT, -self.max_angular, self.max_angular)
        
        # Speed based on clearance
        speed_factor = np.clip((min_dist - ROBOT_EFFECTIVE_RADIUS) / 1.0, 0.2, 1.0)
        v = self.max_speed * 0.5 * speed_factor
        
        return v, w
    
    def _update_stuck_detection(self,
                                 pos: np.ndarray,
                                 goal_dist: float,
                                 goal_dir: float,
                                 heading: float) -> bool:
        """
        Update stuck detection and determine if recovery needed.
        
        Returns:
            True if recovery mode should be activated
        """
        need_recovery = False
        
        # Check movement
        if self.prev_pos is not None:
            if np.linalg.norm(pos - self.prev_pos) < 0.01:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
        # Check progress
        if self.prev_goal_dist is not None:
            if goal_dist >= self.prev_goal_dist - 0.02:
                self.no_progress_counter += 1
            else:
                self.no_progress_counter = 0
                # Progress made, consider exiting recovery
                if self.in_recovery:
                    self.recovery_counter += 1
                    if self.recovery_counter > 30:
                        self.in_recovery = False
                        
        # Need recovery?
        if (self.stuck_counter > DWA_STUCK_THRESHOLD or 
            self.no_progress_counter > DWA_NO_PROGRESS_THRESHOLD):
            need_recovery = True
            
            # Choose recovery direction based on goal
            if not self.in_recovery:
                rel_goal = self._normalize_angle(goal_dir - heading)
                self.recovery_direction = rel_goal < 0  # Opposite side
                
        self.prev_pos = pos.copy()
        self.prev_goal_dist = goal_dist
        
        return need_recovery
    
    def decide(self,
               obstacles: List[TrackedObstacle],
               agv_pos: np.ndarray,
               agv_heading: float,
               goal_pos: Optional[np.ndarray] = None,
               current_vel: Optional[Tuple[float, float]] = None) -> DWANavigationDecision:
        """
        Make a navigation decision using DWA algorithm.
        
        Args:
            obstacles: List of tracked obstacles from L5
            agv_pos: AGV position [x, y]
            agv_heading: AGV heading in radians
            goal_pos: Goal position [x, y]
            current_vel: Current (linear, angular) velocities
            
        Returns:
            DWANavigationDecision with action and parameters
        """
        # Update current velocities if provided
        if current_vel is not None:
            self.current_v, self.current_w = current_vel
            
        # Default goal is forward
        if goal_pos is None:
            goal_pos = agv_pos + 10.0 * np.array([np.cos(agv_heading), np.sin(agv_heading)])
            
        goal_dir = np.arctan2(goal_pos[1] - agv_pos[1], goal_pos[0] - agv_pos[0])
        goal_dist = np.linalg.norm(goal_pos - agv_pos)
        
        # Check if goal reached
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
        
        # Fast path: No obstacles or all far away - go straight to goal at max speed
        min_obs_dist = min((np.linalg.norm(obs.center - agv_pos) for obs in obstacles),
                           default=float('inf'))
        if not obstacles or min_obs_dist > 3.0:
            heading_diff = self._normalize_angle(goal_dir - agv_heading)
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
        need_recovery = self._update_stuck_detection(agv_pos, goal_dist, goal_dir, agv_heading)
        
        # Enter recovery mode if needed
        if need_recovery and not self.in_recovery:
            self.in_recovery = True
            self.recovery_counter = 0
            self.stuck_counter = 0
            self.no_progress_counter = 0
            
        # Recovery mode handling
        if self.in_recovery:
            self.recovery_counter += 1
            if self.recovery_counter > 80:
                # Switch direction
                self.recovery_direction = not self.recovery_direction
                self.recovery_counter = 0
                
        # Critical obstacles
        critical_obs = [obs.id for obs in obstacles 
                       if np.linalg.norm(obs.center - agv_pos) < ROBOT_EFFECTIVE_RADIUS * 3]
                       
        # Minimum distance for safety
        min_dist = min((np.linalg.norm(obs.center - agv_pos) for obs in obstacles),
                       default=float('inf'))
        
        # Control selection
        if self.in_recovery:
            # Wall-following control
            best_v, best_w = self.wall_follow_control(
                obstacles, agv_pos, agv_heading, self.recovery_direction
            )
            best_trajectory = self.predict_trajectory(agv_pos, agv_heading, best_v, best_w)
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
                    trajectory = self.predict_trajectory(agv_pos, agv_heading, v, w)
                    
                    # Check clearance
                    valid, min_clearance = self.check_trajectory_clearance(
                        trajectory, obstacles, agv_pos
                    )
                    
                    if not valid:
                        continue
                        
                    # Evaluate
                    score = self.evaluate_trajectory(
                        trajectory, v, goal_pos, agv_pos, min_clearance
                    )
                    
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
                best_trajectory = self.predict_trajectory(agv_pos, agv_heading, best_v, best_w)
                reason = "DWA: No valid trajectory, entering recovery"
            else:
                reason = f"DWA: v={best_v:.2f}, w={np.degrees(best_w):.1f}°/s"
                
        # Update current velocities
        self.current_v = best_v
        self.current_w = best_w
        
        # Calculate heading change
        heading_change = best_w * SIMULATION_DT
        
        # Determine action - handle emergency evasion instead of full stop
        if min_dist < ROBOT_EFFECTIVE_RADIUS * 1.5:
            # Instead of stopping completely, evade the closest obstacle
            closest_obs = min(obstacles, key=lambda o: np.linalg.norm(o.center - agv_pos))
            away_dir = agv_pos - closest_obs.center
            away_angle = np.arctan2(away_dir[1], away_dir[0])
            
            # Rotate towards the direction away from obstacle
            heading_diff = self._normalize_angle(away_angle - agv_heading)
            evasion_w = np.clip(heading_diff * 2.0, -self.max_angular, self.max_angular)
            heading_change = evasion_w * SIMULATION_DT
            
            # Use minimal speed to allow rotation/escape
            best_v = self.min_speed if self.min_speed > 0 else 0.05
            best_w = evasion_w
            
            # Trigger recovery mode
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
        if min_dist < ROBOT_EFFECTIVE_RADIUS * 2:
            safety_score = 0.2
        elif min_dist < ROBOT_EFFECTIVE_RADIUS * 4:
            safety_score = 0.5
        else:
            safety_score = 0.9
            
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
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# =============================================================================
# DWA Decision Layer (Complete L5 Alternative)
# =============================================================================

class DWADecisionLayer:
    """
    Complete DWA-based decision layer as alternative to default DecisionLayer.
    
    Integrates with existing L5 tracking infrastructure while using DWA
    for navigation decisions.
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize DWA decision layer.
        
        Args:
            dt: Delta time for simulation
        """
        from L5_decision_layer import DecisionTracker
        from L4_detection_layer import LidarProcessor
        
        self.dt = dt
        self.tracker = DecisionTracker(dt)
        self.navigator = DWADecisionMaker()
        self.lidar_processor = LidarProcessor()
        self.goal_pos = None
        self.current_vel = (0.0, 0.0)
        
    def set_goal(self, goal_pos: np.ndarray):
        """Set navigation goal position."""
        self.goal_pos = goal_pos.copy()
        
    def set_velocity(self, linear: float, angular: float):
        """Set current robot velocities."""
        self.current_vel = (linear, angular)
        
    def process_scan(self, ranges: np.ndarray, angles: np.ndarray,
                     agv_pos: np.ndarray, agv_vel: np.ndarray,
                     agv_heading: float) -> List[TrackedObstacle]:
        """
        Process LiDAR scan and update tracker.
        
        Returns:
            List of tracked obstacles
        """
        points = self.lidar_processor.parse_scan(ranges, angles)
        clusters = self.lidar_processor.cluster_points(points)
        self.tracker.update(clusters, agv_pos, agv_vel, agv_heading)
        return self.tracker.get_obstacles()
    
    def get_navigation_decision(self, agv_pos: np.ndarray,
                                 agv_heading: float) -> DWANavigationDecision:
        """
        Get DWA-based navigation decision.
        """
        return self.navigator.decide(
            self.tracker.get_obstacles(),
            agv_pos,
            agv_heading,
            self.goal_pos,
            self.current_vel
        )
    
    def get_critical_obstacles(self, safety_distance: float = 2.0) -> List[TrackedObstacle]:
        """Returns critical obstacles."""
        return self.tracker.get_critical_obstacles(safety_distance)
    
    def get_all_obstacles(self) -> List[TrackedObstacle]:
        """Returns all obstacles."""
        return self.tracker.get_obstacles()
    
    def get_predicted_trajectory(self, agv_pos: np.ndarray, 
                                  agv_heading: float) -> List[Tuple[float, float, float]]:
        """Get predicted trajectory for visualization."""
        decision = self.navigator.decide(
            self.tracker.get_obstacles(),
            agv_pos,
            agv_heading,
            self.goal_pos,
            self.current_vel
        )
        return decision.predicted_trajectory
    
    def reset(self):
        """Reset the decision layer."""
        from L5_decision_layer import DecisionTracker
        self.tracker = DecisionTracker(self.dt)
        self.navigator.reset()
        self.current_vel = (0.0, 0.0)
    
    def export_state(self) -> List[dict]:
        """
        Exports complete state for analysis/debug.
        
        Returns:
            List of dictionaries with info on each obstacle
        """
        from L5_decision_layer import ObstacleState
        obstacles = self.tracker.get_obstacles()
        
        export_data = []
        for obs in obstacles:
            export_data.append({
                "id": obs.id,
                "pos": obs.center.tolist(),
                "vel": obs.velocity.tolist(),
                "state": obs.state.value,
                "d_eq": float(obs.d_eq),
                "d_dot": float(obs.d_dot),
                "confidence": float(obs.confidence),
                "velocity_magnitude": float(np.linalg.norm(obs.velocity)),
                "velocity_history": [float(v) for v in obs.velocity_history],
                "state_history": [s.value for s in obs.state_history],
                "last_seen": int(obs.last_seen),
                "num_points": len(obs.points)
            })
        
        return export_data
    
    def get_statistics(self) -> dict:
        """
        Returns system statistics.
        """
        from L5_decision_layer import ObstacleState
        obstacles = self.tracker.get_obstacles()
        
        static_obs = [o for o in obstacles if o.state == ObstacleState.STATIC]
        dynamic_obs = [o for o in obstacles if o.state == ObstacleState.DYNAMIC]
        unknown_obs = [o for o in obstacles if o.state == ObstacleState.UNKNOWN]
        
        return {
            "total_obstacles": len(obstacles),
            "static_count": len(static_obs),
            "dynamic_count": len(dynamic_obs),
            "unknown_count": len(unknown_obs),
            "average_confidence": np.mean([o.confidence for o in obstacles]) if obstacles else 0.0,
            "critical_obstacles": len([o for o in obstacles if o.d_eq < 2.0]),
            "tracker_time": self.tracker.current_time,
            "active_kalman_filters": len(self.tracker.kalman_filters),
            "dwa_in_recovery": self.navigator.in_recovery
        }
