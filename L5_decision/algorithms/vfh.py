# =============================================================================
# L5 Decision - VFH (Vector Field Histogram) Algorithm
# =============================================================================
# Uses polar histograms to find obstacle-free directions and includes
# wall-following recovery for local minima.
# =============================================================================

import numpy as np
from typing import List, Optional, Tuple

from ..base import BaseDecisionMaker
from ..types import (
    TrackedObstacle,
    VFHNavigationDecision,
    NavigationAction,
    ObstacleState
)

# Import configuration
from ..config import (
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


class VFHDecisionMaker(BaseDecisionMaker):
    """
    Vector Field Histogram based navigation decision maker.
    
    VFH builds a polar histogram of obstacle density around the robot
    and selects the clearest direction towards the goal.
    
    Features:
    - Polar histogram with configurable sectors
    - Quadratic weighting (closer obstacles = higher weight)
    - Dynamic obstacle bonus weighting
    - Wall-following recovery mode
    
    Best for: Cluttered environments where finding gaps is important.
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
            num_sectors: Number of sectors in polar histogram (72 = 5°/sector)
            obstacle_threshold: Density threshold for blocked sectors
            safety_margin: Angular footprint safety margin (m)
            max_speed: Maximum robot speed (m/s)
            min_speed: Minimum robot speed (m/s)
        """
        super().__init__(
            max_speed=max_speed,
            min_speed=min_speed,
            max_angular=MAX_ANGULAR_VELOCITY,
            stuck_threshold=VFH_STUCK_THRESHOLD,
            no_progress_threshold=VFH_NO_PROGRESS_THRESHOLD,
            wall_follow_distance=VFH_WALL_FOLLOW_DISTANCE
        )
        
        self.num_sectors = num_sectors
        self.sector_size = 2 * np.pi / num_sectors
        self.obstacle_threshold = obstacle_threshold
        self.safety_margin = safety_margin
        
        # VFH-specific recovery state
        self.wall_follow_mode = False
        self.wall_follow_counter = 0
        self.wall_follow_left = True
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.wall_follow_mode = False
        self.wall_follow_counter = 0
        
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
            rel_pos = obs.center - agv_pos
            dist = np.linalg.norm(rel_pos)
            
            if dist > LIDAR_VIRTUAL_RANGE or dist < 0.01:
                continue
                
            # Angle to obstacle
            angle = np.arctan2(rel_pos[1], rel_pos[0])
            
            # Angular footprint (closer = wider shadow)
            gamma = np.arctan2(ROBOT_RADIUS + self.safety_margin, max(dist, 0.1))
            
            # Relative angle to heading
            rel_angle = (angle - agv_heading) % (2 * np.pi)
            
            # Distribute weight across covered sectors
            for a in np.linspace(rel_angle - gamma, rel_angle + gamma, 5):
                sector_idx = int((a % (2 * np.pi)) / self.sector_size)
                sector_idx = np.clip(sector_idx, 0, self.num_sectors - 1)
                
                # Quadratic weight
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
        
        Selects the open sector (below threshold) closest to goal direction.
        Falls back to least dense sector if all blocked.
        
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
        # Goal sector
        goal_rel = (goal_direction - agv_heading) % (2 * np.pi)
        goal_sector = int(goal_rel / self.sector_size)
        goal_sector = np.clip(goal_sector, 0, self.num_sectors - 1)
        
        # Find open sectors
        open_sectors = np.where(histogram < self.obstacle_threshold)[0]
        
        if len(open_sectors) == 0:
            # All blocked: choose least dense
            best_sector = np.argmin(histogram)
            best_angle = (best_sector * self.sector_size) + agv_heading
            return best_angle, False, best_sector, histogram[best_sector]
        
        # Find open sector closest to goal (with wraparound)
        distances_to_goal = np.minimum(
            np.abs(open_sectors - goal_sector),
            self.num_sectors - np.abs(open_sectors - goal_sector)
        )
        best_idx = np.argmin(distances_to_goal)
        best_sector = open_sectors[best_idx]
        best_angle = (best_sector * self.sector_size) + agv_heading
        
        return best_angle, True, best_sector, histogram[best_sector]
    
    def _update_vfh_stuck_detection(self, agv_pos: np.ndarray, goal_dist: float) -> bool:
        """
        Update VFH-specific stuck detection.
        
        Returns:
            True if robot appears stuck and needs recovery
        """
        need_recovery = False
        
        # Check movement
        if self.prev_pos is not None:
            if np.linalg.norm(agv_pos - self.prev_pos) < 0.005:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
        # Check goal progress
        if self.prev_goal_dist is not None:
            if goal_dist >= self.prev_goal_dist - 0.01:
                self.no_progress_counter += 1
            else:
                self.no_progress_counter = 0
                # Making progress - consider exiting recovery
                if self.wall_follow_mode:
                    self.wall_follow_counter += 1
                    if self.wall_follow_counter > VFH_RECOVERY_EXIT_STEPS:
                        self.wall_follow_mode = False
                        
        # Activate recovery
        if (self.stuck_counter > self.stuck_threshold or 
            self.no_progress_counter > self.no_progress_threshold):
            need_recovery = True
            
        self.prev_pos = agv_pos.copy()
        self.prev_goal_dist = goal_dist
        
        return need_recovery
    
    def decide(self,
               obstacles: List[TrackedObstacle],
               agv_pos: np.ndarray,
               agv_heading: float,
               goal_pos: Optional[np.ndarray] = None,
               current_vel: Optional[Tuple[float, float]] = None) -> VFHNavigationDecision:
        """
        Make a navigation decision using VFH algorithm.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position [x, y]
            agv_heading: AGV heading in radians
            goal_pos: Goal position [x, y]
            current_vel: Current velocities (not used in VFH)
            
        Returns:
            VFHNavigationDecision with action and parameters
        """
        # Default goal
        if goal_pos is None:
            goal_pos = agv_pos + 10.0 * np.array([np.cos(agv_heading), np.sin(agv_heading)])
            
        goal_direction = np.arctan2(goal_pos[1] - agv_pos[1], goal_pos[0] - agv_pos[0])
        goal_dist = np.linalg.norm(goal_pos - agv_pos)
        
        # Goal reached
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
        
        # No obstacles
        if not obstacles:
            heading_change = self.normalize_angle(goal_direction - agv_heading)
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
        
        # Stuck detection
        need_recovery = self._update_vfh_stuck_detection(agv_pos, goal_dist)
        
        # Enter wall-follow if stuck
        if need_recovery and not self.wall_follow_mode:
            self.wall_follow_mode = True
            self.wall_follow_counter = 0
            self.wall_follow_left = np.random.choice([True, False])
            self.stuck_counter = 0
            self.no_progress_counter = 0
            
        # Recovery timeout
        if self.wall_follow_mode and self.wall_follow_counter > VFH_RECOVERY_TIMEOUT:
            self.wall_follow_left = not self.wall_follow_left
            self.wall_follow_counter = 0
            
        # Build histogram
        histogram = self.build_polar_histogram(obstacles, agv_pos, agv_heading)
        
        # Minimum distance
        min_dist = min((np.linalg.norm(obs.center - agv_pos) for obs in obstacles), 
                       default=LIDAR_VIRTUAL_RANGE)
        
        # Critical obstacles
        critical_obs = [obs.id for obs in obstacles 
                       if np.linalg.norm(obs.center - agv_pos) < ROBOT_EFFECTIVE_RADIUS * 3]
        
        # Determine target heading
        if self.wall_follow_mode:
            target_heading = self._compute_wall_follow_heading(obstacles, agv_pos, agv_heading)
            self.wall_follow_counter += 1
            path_clear = False
            best_sector = 0
            reason = f"Wall-following ({'left' if self.wall_follow_left else 'right'})"
        else:
            target_heading, path_clear, best_sector, _ = self.find_best_direction(
                histogram, goal_direction, agv_heading
            )
            if path_clear:
                reason = f"VFH: Clear path in sector {best_sector}"
            else:
                reason = f"VFH: All blocked, using least dense sector {best_sector}"
                
        # Heading change
        heading_diff = self.normalize_angle(target_heading - agv_heading)
        max_turn = MAX_ANGULAR_VELOCITY * SIMULATION_DT
        if self.wall_follow_mode:
            max_turn *= 0.6
        heading_change = np.clip(heading_diff, -max_turn, max_turn)
        
        # Speed
        speed_factor = np.clip((min_dist - ROBOT_RADIUS) / VFH_MIN_SAFE_DISTANCE, 0.15, 1.0)
        target_speed = self.max_speed * speed_factor
        
        # Determine action
        if min_dist < ROBOT_EFFECTIVE_RADIUS * 1.5:
            action = NavigationAction.SLOW_DOWN
            target_speed = self.min_speed + 0.05
            
            # Escape direction
            closest_obs = min(obstacles, key=lambda o: np.linalg.norm(o.center - agv_pos))
            obs_angle = np.arctan2(closest_obs.center[1] - agv_pos[1], 
                                   closest_obs.center[0] - agv_pos[0])
            escape_angle = self.normalize_angle(obs_angle + np.pi)
            heading_diff_escape = self.normalize_angle(escape_angle - agv_heading)
            heading_change = np.clip(heading_diff_escape, -max_turn * 1.5, max_turn * 1.5)
            reason = "Evading obstacle - rotating away"
            
            if min_dist < ROBOT_EFFECTIVE_RADIUS * 1.2 and not self.wall_follow_mode:
                self.wall_follow_mode = True
                self.wall_follow_counter = 0
                self.wall_follow_left = heading_diff_escape > 0
                
        elif abs(heading_change) > np.deg2rad(30):
            action = NavigationAction.TURN_LEFT if heading_change > 0 else NavigationAction.TURN_RIGHT
        elif speed_factor < 0.5:
            action = NavigationAction.SLOW_DOWN
        else:
            action = NavigationAction.CONTINUE
            
        # Safety score
        safety_score = self.compute_safety_score(obstacles, agv_pos)
            
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
    
    def _compute_wall_follow_heading(self,
                                     obstacles: List[TrackedObstacle],
                                     agv_pos: np.ndarray,
                                     agv_heading: float) -> float:
        """Compute wall-following heading direction."""
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
        
        # Tangent direction
        if self.wall_follow_left:
            follow_angle = closest_angle + np.pi / 2
        else:
            follow_angle = closest_angle - np.pi / 2
            
        # Distance correction
        dist_error = min_dist - self.wall_follow_distance
        correction = np.clip(dist_error * 0.3, -np.pi / 12, np.pi / 12)
        
        if self.wall_follow_left:
            follow_angle -= correction
        else:
            follow_angle += correction
            
        return follow_angle
