# =============================================================================
# L5 Alternative Decision Layer - VFH (Vector Field Histogram)
# =============================================================================
# This module implements VFH-based navigation as an alternative to the default
# NavigationDecisionMaker. It uses polar histograms to find obstacle-free
# directions and includes wall-following recovery for local minima.
#
# Usage:
#   from L5_decision_vfh import VFHDecisionMaker, VFHNavigationDecision
#   vfh = VFHDecisionMaker()
#   decision = vfh.decide(tracked_obstacles, agv_pos, agv_heading)
# =============================================================================

import numpy as np
from dataclasses import dataclass
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
    VFH_NUM_SECTORS,
    VFH_OBSTACLE_THRESHOLD,
    VFH_SAFETY_MARGIN,
    VFH_WALL_FOLLOW_DISTANCE,
    VFH_STUCK_THRESHOLD,
    VFH_NO_PROGRESS_THRESHOLD,
    VFH_RECOVERY_EXIT_STEPS,
    VFH_RECOVERY_TIMEOUT,
    VFH_MIN_SAFE_DISTANCE,
    SIMULATION_DT,
    GOAL_TOLERANCE,
    LIDAR_VIRTUAL_RANGE
)


# =============================================================================
# VFH Navigation Decision
# =============================================================================

@dataclass
class VFHNavigationDecision:
    """VFH-specific navigation decision with histogram data."""
    action: NavigationAction
    target_speed: float
    target_heading_change: float
    reason: str
    critical_obstacles: List[int]
    safety_score: float
    histogram: Optional[np.ndarray] = None  # Polar histogram for visualization
    best_sector: int = 0                     # Selected sector index
    in_recovery: bool = False                # Wall-following active


# =============================================================================
# VFH Decision Maker
# =============================================================================

class VFHDecisionMaker:
    """
    Vector Field Histogram based navigation decision maker.
    
    VFH builds a polar histogram of obstacle density around the robot
    and selects the clearest direction towards the goal. Includes
    wall-following recovery for escaping local minima.
    """
    
    def __init__(self,
                 num_sectors: int = VFH_NUM_SECTORS,
                 obstacle_threshold: float = VFH_OBSTACLE_THRESHOLD,
                 safety_margin: float = VFH_SAFETY_MARGIN,
                 max_speed: float = MAX_LINEAR_VELOCITY,
                 min_speed: float = MIN_LINEAR_VELOCITY):
        """
        Initialize VFH decision maker.
        
        Args:
            num_sectors: Number of sectors in polar histogram (default 72 = 5°/sector)
            obstacle_threshold: Density threshold for blocked sectors
            safety_margin: Angular footprint safety margin
            max_speed: Maximum robot speed (m/s)
            min_speed: Minimum robot speed (m/s)
        """
        self.num_sectors = num_sectors
        self.sector_size = 2 * np.pi / num_sectors
        self.obstacle_threshold = obstacle_threshold
        self.safety_margin = safety_margin
        self.max_speed = max_speed
        self.min_speed = min_speed
        
        # Recovery state
        self.wall_follow_mode = False
        self.wall_follow_counter = 0
        self.wall_follow_left = True
        
        # Stuck detection
        self.stuck_counter = 0
        self.no_progress_counter = 0
        self.prev_pos = None
        self.prev_goal_dist = None
        
    def reset(self):
        """Reset internal state."""
        self.wall_follow_mode = False
        self.wall_follow_counter = 0
        self.stuck_counter = 0
        self.no_progress_counter = 0
        self.prev_pos = None
        self.prev_goal_dist = None
        
    def build_polar_histogram(self, 
                               obstacles: List[TrackedObstacle],
                               agv_pos: np.ndarray,
                               agv_heading: float) -> np.ndarray:
        """
        Build a polar histogram of obstacle density.
        
        Each sector accumulates a weight based on obstacle proximity,
        where closer obstacles contribute more (quadratic weighting).
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position [x, y]
            agv_heading: AGV heading in radians
            
        Returns:
            histogram: Array of obstacle density per sector
        """
        histogram = np.zeros(self.num_sectors)
        
        for obs in obstacles:
            # Calculate relative position
            rel_pos = obs.center - agv_pos
            dist = np.linalg.norm(rel_pos)
            
            if dist > LIDAR_VIRTUAL_RANGE or dist < 0.01:
                continue
                
            # Angle to obstacle (global frame)
            angle = np.arctan2(rel_pos[1], rel_pos[0])
            
            # Calculate angular footprint (closer = wider shadow)
            gamma = np.arctan2(ROBOT_RADIUS + self.safety_margin, max(dist, 0.1))
            
            # Normalize angle relative to heading
            rel_angle = (angle - agv_heading) % (2 * np.pi)
            
            # Distribute weight across sectors covered by footprint
            for a in np.linspace(rel_angle - gamma, rel_angle + gamma, 5):
                sector_idx = int((a % (2 * np.pi)) / self.sector_size)
                sector_idx = np.clip(sector_idx, 0, self.num_sectors - 1)
                
                # Quadratic weight: closer obstacles weigh much more
                weight = ((LIDAR_VIRTUAL_RANGE - dist) / LIDAR_VIRTUAL_RANGE) ** 2
                
                # Dynamic obstacles get extra weight
                if obs.state == ObstacleState.DYNAMIC:
                    weight *= 1.5
                    
                histogram[sector_idx] += weight
                
        return histogram
    
    def find_best_direction(self,
                            histogram: np.ndarray,
                            goal_direction: float,
                            agv_heading: float) -> Tuple[float, bool, int, float]:
        """
        Find the best movement direction using VFH algorithm.
        
        Selects the open sector (below threshold) closest to the goal direction.
        If all sectors are blocked, falls back to the least dense sector.
        
        Args:
            histogram: Polar histogram of obstacle density
            goal_direction: Goal direction in radians (global frame)
            agv_heading: Current AGV heading in radians
            
        Returns:
            Tuple of:
                - best_angle: Recommended direction (radians)
                - path_clear: True if open sector found
                - best_sector: Index of selected sector
                - sector_density: Density of selected sector
        """
        # Calculate which sector contains the goal
        goal_rel = (goal_direction - agv_heading) % (2 * np.pi)
        goal_sector = int(goal_rel / self.sector_size)
        goal_sector = np.clip(goal_sector, 0, self.num_sectors - 1)
        
        # Find open sectors (below threshold)
        open_sectors = np.where(histogram < self.obstacle_threshold)[0]
        
        if len(open_sectors) == 0:
            # RECOVERY: all sectors blocked, choose least dense
            best_sector = np.argmin(histogram)
            best_angle = (best_sector * self.sector_size) + agv_heading
            return best_angle, False, best_sector, histogram[best_sector]
        
        # Find open sector closest to goal (considering wraparound)
        distances_to_goal = np.minimum(
            np.abs(open_sectors - goal_sector),
            self.num_sectors - np.abs(open_sectors - goal_sector)
        )
        best_idx = np.argmin(distances_to_goal)
        best_sector = open_sectors[best_idx]
        best_angle = (best_sector * self.sector_size) + agv_heading
        
        return best_angle, True, best_sector, histogram[best_sector]
    
    def wall_follow_direction(self,
                               obstacles: List[TrackedObstacle],
                               agv_pos: np.ndarray,
                               agv_heading: float,
                               follow_left: bool = True) -> float:
        """
        Calculate wall-following direction for recovery mode.
        
        Finds the closest obstacle and computes a tangent direction
        to follow along its contour.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            agv_heading: AGV heading
            follow_left: True = keep wall on left, False = right
            
        Returns:
            follow_angle: Direction to follow (radians)
        """
        if not obstacles:
            return agv_heading
            
        # Find closest obstacle
        min_dist = float('inf')
        closest_angle = agv_heading
        
        for obs in obstacles:
            rel_pos = obs.center - agv_pos
            dist = np.linalg.norm(rel_pos)
            if dist < min_dist:
                min_dist = dist
                closest_angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        # Tangent direction (perpendicular to wall direction)
        if follow_left:
            follow_angle = closest_angle + np.pi / 2
        else:
            follow_angle = closest_angle - np.pi / 2
            
        # Proportional correction to maintain desired distance
        dist_error = min_dist - VFH_WALL_FOLLOW_DISTANCE
        correction = np.clip(dist_error * 0.3, -np.pi / 12, np.pi / 12)
        
        if follow_left:
            follow_angle -= correction
        else:
            follow_angle += correction
            
        return follow_angle
    
    def _update_stuck_detection(self,
                                 agv_pos: np.ndarray,
                                 goal_dist: float) -> bool:
        """
        Update stuck detection counters.
        
        Returns:
            True if robot appears stuck and needs recovery
        """
        need_recovery = False
        
        # Check if robot is moving
        if self.prev_pos is not None:
            if np.linalg.norm(agv_pos - self.prev_pos) < 0.005:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
        # Check if making progress toward goal
        if self.prev_goal_dist is not None:
            if goal_dist >= self.prev_goal_dist - 0.01:
                self.no_progress_counter += 1
            else:
                self.no_progress_counter = 0
                # Making progress, consider exiting recovery
                if self.wall_follow_mode:
                    self.wall_follow_counter += 1
                    if self.wall_follow_counter > VFH_RECOVERY_EXIT_STEPS:
                        self.wall_follow_mode = False
                        
        # Activate recovery if stuck
        if (self.stuck_counter > VFH_STUCK_THRESHOLD or 
            self.no_progress_counter > VFH_NO_PROGRESS_THRESHOLD):
            need_recovery = True
            
        self.prev_pos = agv_pos.copy()
        self.prev_goal_dist = goal_dist
        
        return need_recovery
    
    def decide(self,
               obstacles: List[TrackedObstacle],
               agv_pos: np.ndarray,
               agv_heading: float,
               goal_pos: Optional[np.ndarray] = None) -> VFHNavigationDecision:
        """
        Make a navigation decision using VFH algorithm.
        
        Args:
            obstacles: List of tracked obstacles from L5
            agv_pos: AGV position [x, y]
            agv_heading: AGV heading in radians
            goal_pos: Goal position [x, y] (optional, uses forward if not provided)
            
        Returns:
            VFHNavigationDecision with action and parameters
        """
        # Default goal is forward if not specified
        if goal_pos is None:
            goal_pos = agv_pos + 10.0 * np.array([np.cos(agv_heading), np.sin(agv_heading)])
            
        goal_direction = np.arctan2(goal_pos[1] - agv_pos[1], goal_pos[0] - agv_pos[0])
        goal_dist = np.linalg.norm(goal_pos - agv_pos)
        
        # Check if goal reached
        if goal_dist < GOAL_TOLERANCE:
            return VFHNavigationDecision(
                action=NavigationAction.CONTINUE,
                target_speed=0.0,
                target_heading_change=0.0,
                reason="Goal reached",
                critical_obstacles=[],
                safety_score=1.0,
                in_recovery=False
            )
        
        # No obstacles case
        if not obstacles:
            heading_change = self._normalize_angle(goal_direction - agv_heading)
            return VFHNavigationDecision(
                action=NavigationAction.CONTINUE,
                target_speed=self.max_speed,
                target_heading_change=heading_change,
                reason="No obstacles detected",
                critical_obstacles=[],
                safety_score=1.0,
                histogram=np.zeros(self.num_sectors),
                in_recovery=False
            )
        
        # Update stuck detection
        need_recovery = self._update_stuck_detection(agv_pos, goal_dist)
        
        # Enter wall-follow mode if stuck
        if need_recovery and not self.wall_follow_mode:
            self.wall_follow_mode = True
            self.wall_follow_counter = 0
            self.wall_follow_left = np.random.choice([True, False])
            self.stuck_counter = 0
            self.no_progress_counter = 0
            
        # Check for recovery timeout
        if self.wall_follow_mode and self.wall_follow_counter > VFH_RECOVERY_TIMEOUT:
            # Stuck too long, try other direction
            self.wall_follow_left = not self.wall_follow_left
            self.wall_follow_counter = 0
            
        # Build histogram
        histogram = self.build_polar_histogram(obstacles, agv_pos, agv_heading)
        
        # Find minimum distance for speed control
        min_dist = min((np.linalg.norm(obs.center - agv_pos) for obs in obstacles), 
                       default=LIDAR_VIRTUAL_RANGE)
        
        # Critical obstacles
        critical_obs = [obs.id for obs in obstacles 
                       if np.linalg.norm(obs.center - agv_pos) < ROBOT_EFFECTIVE_RADIUS * 3]
        
        # Determine target heading
        if self.wall_follow_mode:
            target_heading = self.wall_follow_direction(
                obstacles, agv_pos, agv_heading, self.wall_follow_left
            )
            self.wall_follow_counter += 1
            path_clear = False
            best_sector = 0
            reason = f"Wall-following ({'left' if self.wall_follow_left else 'right'})"
        else:
            target_heading, path_clear, best_sector, sector_density = self.find_best_direction(
                histogram, goal_direction, agv_heading
            )
            if path_clear:
                reason = f"VFH: Clear path in sector {best_sector}"
            else:
                reason = f"VFH: All blocked, using least dense sector {best_sector}"
                
        # Calculate heading change (normalized)
        heading_diff = self._normalize_angle(target_heading - agv_heading)
        
        # Limit rotation rate
        max_turn = MAX_ANGULAR_VELOCITY * SIMULATION_DT
        if self.wall_follow_mode:
            max_turn *= 0.6  # Smoother turns during wall-follow
        heading_change = np.clip(heading_diff, -max_turn, max_turn)
        
        # Calculate speed based on obstacle proximity
        speed_factor = np.clip((min_dist - ROBOT_RADIUS) / VFH_MIN_SAFE_DISTANCE, 0.15, 1.0)
        target_speed = self.max_speed * speed_factor
        
        # Determine action
        if min_dist < ROBOT_EFFECTIVE_RADIUS * 1.5:
            # Very close to obstacle - slow down but keep moving to allow rotation
            action = NavigationAction.SLOW_DOWN
            target_speed = self.min_speed + 0.05  # Minimal speed to allow rotation
            
            # Force larger rotation away from the closest obstacle direction
            # Find the closest obstacle direction
            closest_obs = min(obstacles, key=lambda o: np.linalg.norm(o.center - agv_pos))
            obs_angle = np.arctan2(closest_obs.center[1] - agv_pos[1], 
                                   closest_obs.center[0] - agv_pos[0])
            # Rotate away (opposite direction)
            escape_angle = self._normalize_angle(obs_angle + np.pi)
            heading_diff_escape = self._normalize_angle(escape_angle - agv_heading)
            heading_change = np.clip(heading_diff_escape, -max_turn * 1.5, max_turn * 1.5)
            
            reason = f"Evading obstacle - rotating away"
            
            # Force wall-follow recovery if critically close
            if min_dist < ROBOT_EFFECTIVE_RADIUS * 1.2 and not self.wall_follow_mode:
                self.wall_follow_mode = True
                self.wall_follow_counter = 0
                self.wall_follow_left = heading_diff_escape > 0  # Follow in escape direction
                
        elif abs(heading_change) > np.deg2rad(30):
            action = NavigationAction.TURN_LEFT if heading_change > 0 else NavigationAction.TURN_RIGHT
        elif speed_factor < 0.5:
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
            
        return VFHNavigationDecision(
            action=action,
            target_speed=target_speed,
            target_heading_change=heading_change,
            reason=reason,
            critical_obstacles=critical_obs,
            safety_score=safety_score,
            histogram=histogram,
            best_sector=best_sector,
            in_recovery=self.wall_follow_mode
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
# VFH Decision Layer (Complete L5 Alternative)
# =============================================================================

class VFHDecisionLayer:
    """
    Complete VFH-based decision layer as alternative to default DecisionLayer.
    
    Integrates with existing L5 tracking infrastructure while using VFH
    for navigation decisions instead of the default algorithm.
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize VFH decision layer.
        
        Args:
            dt: Delta time for simulation
        """
        from L5_decision_layer import DecisionTracker
        from L4_detection_layer import LidarProcessor
        
        self.dt = dt
        self.tracker = DecisionTracker(dt)
        self.navigator = VFHDecisionMaker()
        self.lidar_processor = LidarProcessor()
        self.goal_pos = None
        
    def set_goal(self, goal_pos: np.ndarray):
        """Set navigation goal position."""
        self.goal_pos = goal_pos.copy()
        
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
                                 agv_heading: float) -> VFHNavigationDecision:
        """
        Get VFH-based navigation decision.
        """
        return self.navigator.decide(
            self.tracker.get_obstacles(),
            agv_pos,
            agv_heading,
            self.goal_pos
        )
    
    def get_critical_obstacles(self, safety_distance: float = 2.0) -> List[TrackedObstacle]:
        """Returns critical obstacles."""
        return self.tracker.get_critical_obstacles(safety_distance)
    
    def get_all_obstacles(self) -> List[TrackedObstacle]:
        """Returns all obstacles."""
        return self.tracker.get_obstacles()
    
    def get_histogram(self, agv_pos: np.ndarray, agv_heading: float) -> np.ndarray:
        """Get current VFH polar histogram for visualization."""
        return self.navigator.build_polar_histogram(
            self.tracker.get_obstacles(), agv_pos, agv_heading
        )
    
    def reset(self):
        """Reset the decision layer."""
        from L5_decision_layer import DecisionTracker
        self.tracker = DecisionTracker(self.dt)
        self.navigator.reset()
    
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
            "vfh_in_recovery": self.navigator.wall_follow_mode
        }
