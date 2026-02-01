# =============================================================================
# L5 Alternative Decision Layer - GapNav (Gap-Based Navigation + APF)
# =============================================================================
# This module implements state-of-the-art Gap-Based Navigation combined with
# Artificial Potential Fields and Enhanced DWA as an alternative to the default
# NavigationDecisionMaker.
#
# Key features:
# - Gap Detection: Finds passable corridors in obstacle field
# - APF Integration: Smooth obstacle repulsion + goal attraction
# - Direct Path Priority: Goes straight when path is clear
# - Multi-Layer Fallback: Wall-follow → Reverse → Random escape
#
# Usage:
#   from L5_decision_gapnav import GapNavDecisionMaker, GapNavNavigationDecision
#   gapnav = GapNavDecisionMaker()
#   decision = gapnav.decide(tracked_obstacles, agv_pos, agv_heading)
# =============================================================================

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
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
    GAPNAV_MIN_GAP_WIDTH,
    GAPNAV_FREE_THRESHOLD_FACTOR,
    GAPNAV_MAX_FREE_THRESHOLD,
    APF_ATTRACT_GAIN,
    APF_REPEL_GAIN,
    APF_REPEL_THRESHOLD,
    GAPNAV_WEIGHT_HEADING,
    GAPNAV_WEIGHT_CLEARANCE,
    GAPNAV_WEIGHT_VELOCITY,
    GAPNAV_WEIGHT_PROGRESS,
    GAPNAV_WEIGHT_APF,
    GAPNAV_VELOCITY_SAMPLES,
    GAPNAV_ANGULAR_SAMPLES,
    GAPNAV_PREDICT_TIME,
    GAPNAV_WALL_FOLLOW_DISTANCE,
    GAPNAV_REVERSE_DISTANCE,
    GAPNAV_FINAL_APPROACH_DISTANCE,
    GAPNAV_FINAL_APPROACH_TOLERANCE,
    GAPNAV_WALL_FOLLOW_TIMEOUT,
    GAPNAV_REVERSE_TIMEOUT,
    GAPNAV_RANDOM_TIMEOUT,
    DIRECT_PATH_SAFETY_MARGIN,
    DIRECT_PATH_MIN_CLEARANCE,
    DIRECT_PATH_CORRIDOR_MARGIN,
    SIMULATION_DT,
    GOAL_TOLERANCE,
    LIDAR_VIRTUAL_RANGE
)


# =============================================================================
# Recovery Mode Enum
# =============================================================================

class RecoveryMode(Enum):
    """Recovery mode states for hierarchical fallback."""
    NORMAL = 0
    WALL_FOLLOW = 1
    REVERSE_ESCAPE = 2
    RANDOM_ESCAPE = 3


# =============================================================================
# Gap Data Structure
# =============================================================================

@dataclass
class DetectedGap:
    """Represents a detected navigable gap."""
    center_angle: float         # Center angle of the gap (radians)
    angular_width: float        # Angular width (radians)
    linear_width: float         # Estimated linear width (meters)
    depth: float                # Average depth (distance to obstacles)
    min_depth: float            # Minimum depth in the gap
    score: float = 0.0          # Computed score for selection


# =============================================================================
# GapNav Navigation Decision
# =============================================================================

@dataclass
class GapNavNavigationDecision:
    """GapNav-specific navigation decision with rich info."""
    action: NavigationAction
    target_speed: float
    target_heading_change: float
    reason: str
    critical_obstacles: List[int]
    safety_score: float
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    predicted_trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    detected_gaps: List[DetectedGap] = field(default_factory=list)
    selected_gap: Optional[DetectedGap] = None
    apf_force: Optional[np.ndarray] = None
    recovery_mode: RecoveryMode = RecoveryMode.NORMAL
    using_direct_path: bool = False


# =============================================================================
# GapNav Decision Maker
# =============================================================================

class GapNavDecisionMaker:
    """
    Gap-Based Navigation with APF and Enhanced DWA decision maker.
    
    This is a state-of-the-art approach that:
    1. First checks for direct path to goal (priority)
    2. Detects navigable gaps in obstacle field
    3. Uses APF for smooth obstacle repulsion
    4. Applies enhanced DWA for trajectory selection
    5. Has multi-layer fallback for recovery
    """
    
    def __init__(self,
                 max_speed: float = MAX_LINEAR_VELOCITY,
                 min_speed: float = MIN_LINEAR_VELOCITY,
                 max_angular: float = MAX_ANGULAR_VELOCITY,
                 min_gap_width: float = GAPNAV_MIN_GAP_WIDTH):
        """
        Initialize GapNav decision maker.
        
        Args:
            max_speed: Maximum linear velocity (m/s)
            min_speed: Minimum linear velocity (m/s)
            max_angular: Maximum angular velocity (rad/s)
            min_gap_width: Minimum gap width for passage (meters)
        """
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_angular = max_angular
        self.min_gap_width = min_gap_width
        
        # Current velocities
        self.current_v = 0.0
        self.current_w = 0.0
        
        # Recovery state machine
        self.recovery_mode = RecoveryMode.NORMAL
        self.recovery_counter = 0
        self.recovery_direction = True  # True = left
        
        # Stuck detection
        self.stuck_counter = 0
        self.no_progress_counter = 0
        self.prev_pos = None
        self.prev_goal_dist = None
        
        # Random generator for random escape
        self.rng = np.random.default_rng(42)
        
    def reset(self):
        """Reset internal state."""
        self.current_v = 0.0
        self.current_w = 0.0
        self.recovery_mode = RecoveryMode.NORMAL
        self.recovery_counter = 0
        self.stuck_counter = 0
        self.no_progress_counter = 0
        self.prev_pos = None
        self.prev_goal_dist = None
        
    # =========================================================================
    # Direct Path Check (Priority 1)
    # =========================================================================
    
    def check_direct_path(self,
                          obstacles: List[TrackedObstacle],
                          agv_pos: np.ndarray,
                          agv_heading: float,
                          goal_pos: np.ndarray) -> Tuple[bool, float, float]:
        """
        Check if there's a clear direct path to the goal.
        This is checked FIRST at every decision step.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            agv_heading: AGV heading
            goal_pos: Goal position
            
        Returns:
            (is_clear, goal_direction, min_clearance)
        """
        goal_dir = np.arctan2(goal_pos[1] - agv_pos[1], goal_pos[0] - agv_pos[0])
        goal_dist = np.linalg.norm(goal_pos - agv_pos)
        
        if not obstacles:
            return True, goal_dir, float('inf')
            
        # Check clearance in cone towards goal
        cone_width = np.deg2rad(max(15, min(45, 300 / max(goal_dist, 1))))
        
        min_clearance = float('inf')
        for obs in obstacles:
            rel_pos = obs.center - agv_pos
            obs_dist = np.linalg.norm(rel_pos)
            obs_angle = np.arctan2(rel_pos[1], rel_pos[0])
            
            # Check if obstacle is in the cone
            angle_diff = abs(self._normalize_angle(obs_angle - goal_dir))
            if angle_diff < cone_width:
                clearance = obs_dist - ROBOT_EFFECTIVE_RADIUS
                min_clearance = min(min_clearance, clearance)
                
        # Check corridor sides
        left_angle = goal_dir + np.deg2rad(15)
        right_angle = goal_dir - np.deg2rad(15)
        
        left_clearance = self._get_clearance_in_direction(obstacles, agv_pos, left_angle)
        right_clearance = self._get_clearance_in_direction(obstacles, agv_pos, right_angle)
        
        corridor_clear = min(left_clearance, right_clearance) > DIRECT_PATH_CORRIDOR_MARGIN
        
        # Path is clear if:
        # 1. Clearance > goal distance (nothing between us and goal), OR
        # 2. Clearance is large enough for safe travel
        goal_visible = min_clearance >= goal_dist
        path_safe = min_clearance > DIRECT_PATH_MIN_CLEARANCE
        
        is_clear = (goal_visible or path_safe) and corridor_clear
        
        return is_clear, goal_dir, min(min_clearance, left_clearance, right_clearance)
    
    def go_straight_control(self,
                            agv_pos: np.ndarray,
                            agv_heading: float,
                            goal_pos: np.ndarray,
                            clearance: float) -> Tuple[float, float]:
        """
        Simple control to go straight towards goal.
        Used when direct path is clear.
        
        Returns:
            (v, w): Linear and angular velocities
        """
        goal_dir = np.arctan2(goal_pos[1] - agv_pos[1], goal_pos[0] - agv_pos[0])
        goal_dist = np.linalg.norm(goal_pos - agv_pos)
        
        # Angular error
        angle_error = self._normalize_angle(goal_dir - agv_heading)
        
        # Angular velocity with proportional control
        w = np.clip(angle_error * 2.0, -self.max_angular, self.max_angular)
        
        # Speed factors
        alignment_factor = max(0.3, np.cos(angle_error))
        clearance_factor = np.clip((clearance - ROBOT_EFFECTIVE_RADIUS) / 2.0, 0.3, 1.0)
        distance_factor = np.clip(goal_dist / 2.0, 0.3, 1.0)
        
        v = self.max_speed * alignment_factor * clearance_factor * distance_factor
        
        return v, w
    
    # =========================================================================
    # Gap Detection (Priority 2)
    # =========================================================================
    
    def detect_gaps(self,
                    obstacles: List[TrackedObstacle],
                    agv_pos: np.ndarray,
                    agv_heading: float) -> List[DetectedGap]:
        """
        Detect navigable gaps from obstacle positions.
        
        Analyzes the angular distribution of obstacles and finds
        regions with no obstacles (gaps) that are wide enough
        for the robot to pass through.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            agv_heading: AGV heading
            
        Returns:
            List of detected gaps sorted by score
        """
        if not obstacles:
            # No obstacles = one big gap in front
            return [DetectedGap(
                center_angle=agv_heading,
                angular_width=2 * np.pi,
                linear_width=float('inf'),
                depth=LIDAR_VIRTUAL_RANGE,
                min_depth=LIDAR_VIRTUAL_RANGE,
                score=1.0
            )]
            
        # Create angular histogram of obstacles
        num_sectors = 72  # 5 degrees per sector
        sector_size = 2 * np.pi / num_sectors
        obstacle_in_sector = [False] * num_sectors
        sector_distances = [LIDAR_VIRTUAL_RANGE] * num_sectors
        
        free_threshold = min(LIDAR_VIRTUAL_RANGE * GAPNAV_FREE_THRESHOLD_FACTOR, 
                            GAPNAV_MAX_FREE_THRESHOLD)
        
        for obs in obstacles:
            rel_pos = obs.center - agv_pos
            dist = np.linalg.norm(rel_pos)
            
            if dist > LIDAR_VIRTUAL_RANGE:
                continue
                
            angle = np.arctan2(rel_pos[1], rel_pos[0])
            
            # Mark sectors occupied by this obstacle
            angular_footprint = np.arctan2(ROBOT_EFFECTIVE_RADIUS * 1.5, max(dist, 0.5))
            
            for a in np.linspace(angle - angular_footprint, angle + angular_footprint, 5):
                sector_idx = int(((a + np.pi) % (2 * np.pi)) / sector_size)
                sector_idx = np.clip(sector_idx, 0, num_sectors - 1)
                
                if dist < free_threshold:
                    obstacle_in_sector[sector_idx] = True
                    
                sector_distances[sector_idx] = min(sector_distances[sector_idx], dist)
                
        # Find gaps (contiguous free sectors)
        gaps = []
        in_gap = False
        gap_start = 0
        
        # Handle wraparound by checking extended array
        extended_occupied = obstacle_in_sector + obstacle_in_sector
        
        for i in range(num_sectors * 2):
            idx = i % num_sectors
            
            if not extended_occupied[i] and not in_gap:
                in_gap = True
                gap_start = i
            elif extended_occupied[i] and in_gap:
                in_gap = False
                gap_end = i - 1
                
                # Only process gaps that start in first half
                if gap_start < num_sectors:
                    self._process_gap(gaps, gap_start, gap_end, num_sectors, 
                                     sector_size, sector_distances)
                    
        # Process final gap if still open
        if in_gap and gap_start < num_sectors:
            self._process_gap(gaps, gap_start, num_sectors * 2 - 1, num_sectors,
                             sector_size, sector_distances)
            
        return gaps
    
    def _process_gap(self, gaps: List[DetectedGap], start: int, end: int,
                     num_sectors: int, sector_size: float,
                     sector_distances: List[float]):
        """Process and add a detected gap to the list."""
        # Calculate gap properties
        gap_sectors = list(range(start, end + 1))
        
        # Adjust for wraparound
        gap_angles = [(s % num_sectors) * sector_size - np.pi for s in gap_sectors]
        gap_dists = [sector_distances[s % num_sectors] for s in gap_sectors]
        
        if len(gap_angles) < 2:
            return
            
        # Center angle (weighted by distance)
        weights = np.array(gap_dists) / max(np.sum(gap_dists), 0.1)
        center_angle = np.sum(np.array(gap_angles) * weights)
        
        # Angular width
        angular_width = len(gap_sectors) * sector_size
        
        # Estimate linear width at average depth
        avg_depth = np.mean(gap_dists)
        linear_width = 2 * avg_depth * np.tan(angular_width / 2)
        
        if linear_width >= self.min_gap_width:
            gaps.append(DetectedGap(
                center_angle=center_angle,
                angular_width=angular_width,
                linear_width=linear_width,
                depth=avg_depth,
                min_depth=np.min(gap_dists)
            ))
            
    def select_best_gap(self,
                        gaps: List[DetectedGap],
                        goal_direction: float,
                        agv_heading: float) -> Optional[DetectedGap]:
        """
        Select the best gap for navigation.
        
        Prioritizes gaps that are:
        1. Aligned with goal direction
        2. Wide enough for safe passage
        3. Deep (far obstacles)
        
        Returns:
            Best gap or None if no gaps
        """
        if not gaps:
            return None
            
        best_gap = None
        best_score = -float('inf')
        
        for gap in gaps:
            # Alignment with goal
            angle_to_goal = abs(self._normalize_angle(gap.center_angle - goal_direction))
            alignment_score = (np.pi - angle_to_goal) / np.pi
            
            # Width score
            width_score = np.clip(gap.linear_width / 3.0, 0, 1)
            
            # Depth score
            depth_score = np.clip(gap.depth / LIDAR_VIRTUAL_RANGE, 0, 1)
            
            # Combined score
            score = 0.5 * alignment_score + 0.3 * width_score + 0.2 * depth_score
            gap.score = score
            
            if score > best_score:
                best_score = score
                best_gap = gap
                
        return best_gap
    
    # =========================================================================
    # Artificial Potential Field (APF)
    # =========================================================================
    
    def compute_apf_force(self,
                          obstacles: List[TrackedObstacle],
                          agv_pos: np.ndarray,
                          subgoal_angle: float) -> Tuple[np.ndarray, float]:
        """
        Compute Artificial Potential Field force.
        
        Combines:
        - Attractive force: pulls towards subgoal
        - Repulsive force: pushes away from obstacles
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            subgoal_angle: Direction towards subgoal (radians)
            
        Returns:
            (total_force, apf_angle): Force vector and recommended angle
        """
        # Attractive force (towards subgoal)
        attract_dir = np.array([np.cos(subgoal_angle), np.sin(subgoal_angle)])
        F_attract = APF_ATTRACT_GAIN * attract_dir
        
        # Repulsive force (away from obstacles)
        F_repel = np.zeros(2)
        
        for obs in obstacles:
            diff = agv_pos - obs.center
            dist = np.linalg.norm(diff)
            
            if dist < APF_REPEL_THRESHOLD and dist > 0.01:
                # Inverse-square repulsion
                repel_magnitude = APF_REPEL_GAIN * (1.0 / dist - 1.0 / APF_REPEL_THRESHOLD) / (dist ** 2)
                repel_magnitude = np.clip(repel_magnitude, 0, 2.0)
                
                # Extra repulsion for dynamic obstacles
                if obs.state == ObstacleState.DYNAMIC:
                    repel_magnitude *= 1.5
                    
                F_repel += repel_magnitude * (diff / dist)
                
        # Total force
        F_total = F_attract + F_repel
        
        # Compute recommended angle
        if np.linalg.norm(F_total) > 0.01:
            apf_angle = np.arctan2(F_total[1], F_total[0])
        else:
            apf_angle = subgoal_angle
            
        return F_total, apf_angle
    
    # =========================================================================
    # Enhanced DWA Control
    # =========================================================================
    
    def enhanced_dwa_control(self,
                              obstacles: List[TrackedObstacle],
                              agv_pos: np.ndarray,
                              agv_heading: float,
                              goal_pos: np.ndarray,
                              subgoal_angle: float,
                              apf_angle: float) -> Tuple[float, float, List, bool]:
        """
        Enhanced DWA control with gap and APF integration.
        
        Returns:
            (v, w, trajectory, need_recovery)
        """
        # Dynamic window
        dt = SIMULATION_DT
        v_min = max(self.min_speed, self.current_v - MAX_LINEAR_ACCEL * dt)
        v_max = min(self.max_speed, self.current_v + MAX_LINEAR_ACCEL * dt)
        w_min = max(-self.max_angular, self.current_w - MAX_ANGULAR_ACCEL * dt)
        w_max = min(self.max_angular, self.current_w + MAX_ANGULAR_ACCEL * dt)
        
        v_samples = np.linspace(v_min, v_max, GAPNAV_VELOCITY_SAMPLES)
        w_samples = np.linspace(w_min, w_max, GAPNAV_ANGULAR_SAMPLES)
        
        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0
        best_trajectory = []
        
        goal_dir = np.arctan2(goal_pos[1] - agv_pos[1], goal_pos[0] - agv_pos[0])
        dist_to_goal = np.linalg.norm(goal_pos - agv_pos)
        
        for v in v_samples:
            for w in w_samples:
                # Predict trajectory
                trajectory = self._predict_trajectory(agv_pos, agv_heading, v, w)
                
                if not trajectory:
                    continue
                    
                # Check clearance
                valid, min_clearance = self._check_trajectory_clearance(
                    trajectory, obstacles, agv_pos
                )
                
                if not valid:
                    continue
                    
                final_x, final_y, final_heading = trajectory[-1]
                final_pos = np.array([final_x, final_y])
                
                # Score components
                # 1. Heading towards subgoal
                heading_diff = abs(self._normalize_angle(subgoal_angle - final_heading))
                heading_score = (np.pi - heading_diff) / np.pi
                
                # 2. Clearance
                clearance_score = np.clip(min_clearance / 2.5, 0, 1)
                
                # 3. Velocity
                velocity_score = v / self.max_speed
                
                # 4. Progress
                final_goal_dist = np.linalg.norm(goal_pos - final_pos)
                progress = (dist_to_goal - final_goal_dist) / max(dist_to_goal, 0.1)
                progress_score = np.clip(progress, -1, 1)
                
                # 5. APF alignment
                apf_diff = abs(self._normalize_angle(apf_angle - final_heading))
                apf_score = (np.pi - apf_diff) / np.pi
                
                # Combined score
                score = (
                    GAPNAV_WEIGHT_HEADING * heading_score +
                    GAPNAV_WEIGHT_CLEARANCE * clearance_score +
                    GAPNAV_WEIGHT_VELOCITY * velocity_score +
                    GAPNAV_WEIGHT_PROGRESS * progress_score +
                    GAPNAV_WEIGHT_APF * apf_score
                )
                
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
                    best_trajectory = trajectory
                    
        need_recovery = best_score == -float('inf')
        
        return best_v, best_w, best_trajectory, need_recovery
    
    def _predict_trajectory(self, pos: np.ndarray, heading: float,
                            v: float, w: float) -> List[Tuple[float, float, float]]:
        """Predict trajectory for given velocity commands."""
        trajectory = []
        x, y, theta = pos[0], pos[1], heading
        
        steps = int(GAPNAV_PREDICT_TIME / SIMULATION_DT)
        for _ in range(steps):
            theta += w * SIMULATION_DT
            x += v * np.cos(theta) * SIMULATION_DT
            y += v * np.sin(theta) * SIMULATION_DT
            trajectory.append((x, y, theta))
            
        return trajectory
    
    def _check_trajectory_clearance(self, trajectory: List[Tuple[float, float, float]],
                                     obstacles: List[TrackedObstacle],
                                     pos: np.ndarray) -> Tuple[bool, float]:
        """Check trajectory clearance against obstacles."""
        if not obstacles:
            return True, float('inf')
            
        min_clearance = float('inf')
        
        for (x, y, theta) in trajectory:
            traj_pos = np.array([x, y])
            
            for obs in obstacles:
                dist = np.linalg.norm(traj_pos - obs.center)
                clearance = dist - ROBOT_EFFECTIVE_RADIUS
                
                if clearance < 0:
                    return False, 0
                    
                min_clearance = min(min_clearance, clearance)
                
        return True, min_clearance
    
    # =========================================================================
    # Recovery Behaviors
    # =========================================================================
    
    def wall_follow_control(self, obstacles: List[TrackedObstacle],
                            agv_pos: np.ndarray, agv_heading: float,
                            follow_left: bool) -> Tuple[float, float]:
        """Wall-following behavior for recovery."""
        if not obstacles:
            return self.max_speed * 0.5, 0.0
            
        # Find closest obstacle
        min_dist = float('inf')
        wall_angle = agv_heading
        
        for obs in obstacles:
            rel_pos = obs.center - agv_pos
            dist = np.linalg.norm(rel_pos)
            if dist < min_dist:
                min_dist = dist
                wall_angle = np.arctan2(rel_pos[1], rel_pos[0])
                
        # Tangent direction
        if follow_left:
            follow_angle = wall_angle + np.pi / 2
        else:
            follow_angle = wall_angle - np.pi / 2
            
        # Distance correction
        dist_error = min_dist - GAPNAV_WALL_FOLLOW_DISTANCE
        correction = np.clip(dist_error * 0.4, -np.pi / 8, np.pi / 8)
        
        if follow_left:
            follow_angle -= correction
        else:
            follow_angle += correction
            
        # Angular velocity
        heading_diff = self._normalize_angle(follow_angle - agv_heading)
        w = np.clip(heading_diff / SIMULATION_DT, -self.max_angular, self.max_angular)
        
        # Speed
        clearance = self._get_clearance_in_direction(obstacles, agv_pos, agv_heading)
        speed_factor = np.clip((clearance - ROBOT_EFFECTIVE_RADIUS) / 1.0, 0.2, 1.0)
        v = self.max_speed * 0.5 * speed_factor
        
        return v, w
    
    def reverse_escape_control(self, obstacles: List[TrackedObstacle],
                                agv_pos: np.ndarray,
                                agv_heading: float) -> Tuple[float, float]:
        """Reverse and escape behavior."""
        # Find clearest direction
        best_angle = agv_heading
        best_clearance = 0
        
        for angle_offset in np.linspace(-np.pi, np.pi, 36):
            test_angle = agv_heading + angle_offset
            clearance = self._get_clearance_in_direction(obstacles, agv_pos, test_angle)
            if clearance > best_clearance:
                best_clearance = clearance
                best_angle = test_angle
                
        # Check if reversing is safe
        reverse_angle = agv_heading + np.pi
        reverse_clearance = self._get_clearance_in_direction(obstacles, agv_pos, reverse_angle)
        
        if reverse_clearance > ROBOT_EFFECTIVE_RADIUS + 0.3:
            v = -0.3  # Slow reverse
            angle_diff = self._normalize_angle(best_angle - agv_heading)
            w = np.clip(angle_diff * 0.5, -self.max_angular * 0.5, self.max_angular * 0.5)
        else:
            v = 0.0
            angle_diff = self._normalize_angle(best_angle - agv_heading)
            w = np.clip(angle_diff, -self.max_angular, self.max_angular)
            
        return v, w
    
    def random_escape_control(self, agv_heading: float) -> Tuple[float, float]:
        """Random rotation to break deadlock."""
        v = 0.0
        w = self.rng.choice([-1, 1]) * self.max_angular * 0.8
        return v, w
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _get_clearance_in_direction(self, obstacles: List[TrackedObstacle],
                                     agv_pos: np.ndarray,
                                     direction: float,
                                     cone_width: float = np.deg2rad(25)) -> float:
        """Get minimum clearance in a directional cone."""
        min_clearance = LIDAR_VIRTUAL_RANGE
        
        for obs in obstacles:
            rel_pos = obs.center - agv_pos
            dist = np.linalg.norm(rel_pos)
            obs_angle = np.arctan2(rel_pos[1], rel_pos[0])
            
            angle_diff = abs(self._normalize_angle(obs_angle - direction))
            if angle_diff < cone_width:
                clearance = dist - ROBOT_EFFECTIVE_RADIUS
                min_clearance = min(min_clearance, clearance)
                
        return min_clearance
    
    def _update_stuck_detection(self, agv_pos: np.ndarray, goal_dist: float,
                                 goal_dir: float, agv_heading: float) -> bool:
        """Update stuck detection and return if recovery needed."""
        need_recovery = False
        
        # Movement check
        if self.prev_pos is not None:
            if np.linalg.norm(agv_pos - self.prev_pos) < 0.01:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
        # Progress check
        if self.prev_goal_dist is not None:
            if goal_dist >= self.prev_goal_dist - 0.02:
                self.no_progress_counter += 1
            else:
                self.no_progress_counter = 0
                # Exit recovery if making progress
                if self.recovery_mode != RecoveryMode.NORMAL:
                    self.recovery_counter += 1
                    if self.recovery_counter > 30:
                        self.recovery_mode = RecoveryMode.NORMAL
                        
        # Need recovery?
        if self.stuck_counter > 15 or self.no_progress_counter > 40:
            need_recovery = True
            if self.recovery_mode == RecoveryMode.NORMAL:
                rel_goal = self._normalize_angle(goal_dir - agv_heading)
                self.recovery_direction = rel_goal < 0
                
        self.prev_pos = agv_pos.copy()
        self.prev_goal_dist = goal_dist
        
        return need_recovery
    
    def _update_recovery_state(self, need_recovery: bool, in_final_approach: bool):
        """Update recovery state machine."""
        if self.recovery_mode == RecoveryMode.NORMAL and need_recovery:
            self.recovery_mode = RecoveryMode.WALL_FOLLOW
            self.recovery_counter = 0
            self.stuck_counter = 0
            self.no_progress_counter = 0
            
        elif self.recovery_mode == RecoveryMode.WALL_FOLLOW:
            self.recovery_counter += 1
            timeout = GAPNAV_WALL_FOLLOW_TIMEOUT // 2 if in_final_approach else GAPNAV_WALL_FOLLOW_TIMEOUT
            if self.recovery_counter > timeout:
                self.recovery_mode = RecoveryMode.REVERSE_ESCAPE
                self.recovery_counter = 0
                
        elif self.recovery_mode == RecoveryMode.REVERSE_ESCAPE:
            self.recovery_counter += 1
            timeout = GAPNAV_REVERSE_TIMEOUT // 2 if in_final_approach else GAPNAV_REVERSE_TIMEOUT
            if self.recovery_counter > timeout:
                self.recovery_mode = RecoveryMode.RANDOM_ESCAPE
                self.recovery_counter = 0
                
        elif self.recovery_mode == RecoveryMode.RANDOM_ESCAPE:
            self.recovery_counter += 1
            timeout = GAPNAV_RANDOM_TIMEOUT // 2 if in_final_approach else GAPNAV_RANDOM_TIMEOUT
            if self.recovery_counter > timeout:
                self.recovery_mode = RecoveryMode.NORMAL
                self.recovery_counter = 0
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    # =========================================================================
    # Main Decision Method
    # =========================================================================
    
    def decide(self,
               obstacles: List[TrackedObstacle],
               agv_pos: np.ndarray,
               agv_heading: float,
               goal_pos: Optional[np.ndarray] = None,
               current_vel: Optional[Tuple[float, float]] = None) -> GapNavNavigationDecision:
        """
        Make a navigation decision using GapNav + APF + Enhanced DWA.
        
        Priority order:
        1. Direct path to goal (if clear)
        2. Gap-based navigation with APF
        3. Recovery modes (wall-follow → reverse → random)
        
        Args:
            obstacles: List of tracked obstacles from L5
            agv_pos: AGV position [x, y]
            agv_heading: AGV heading in radians
            goal_pos: Goal position [x, y]
            current_vel: Current (linear, angular) velocities
            
        Returns:
            GapNavNavigationDecision with full navigation info
        """
        # Update velocities
        if current_vel is not None:
            self.current_v, self.current_w = current_vel
            
        # Default goal (ensure goal_pos is not None for type checker)
        goal: np.ndarray
        if goal_pos is None:
            goal = agv_pos + 10.0 * np.array([np.cos(agv_heading), np.sin(agv_heading)])
        else:
            goal = goal_pos
        
        # Now goal is guaranteed to be a valid array
        goal_dir = np.arctan2(goal[1] - agv_pos[1], goal[0] - agv_pos[0])
        goal_dist = np.linalg.norm(goal - agv_pos)
        
        # Final approach mode
        in_final_approach = goal_dist < GAPNAV_FINAL_APPROACH_DISTANCE
        effective_tolerance = GAPNAV_FINAL_APPROACH_TOLERANCE if in_final_approach else GOAL_TOLERANCE
        
        # Goal reached check
        if goal_dist < effective_tolerance:
            return GapNavNavigationDecision(
                action=NavigationAction.CONTINUE,
                target_speed=0.0,
                target_heading_change=0.0,
                reason="Goal reached",
                critical_obstacles=[],
                safety_score=1.0,
                recovery_mode=RecoveryMode.NORMAL
            )
            
        # Stuck detection
        need_recovery = self._update_stuck_detection(agv_pos, goal_dist, goal_dir, agv_heading)
        self._update_recovery_state(need_recovery, in_final_approach)
        
        # Critical obstacles
        critical_obs = [obs.id for obs in obstacles 
                       if np.linalg.norm(obs.center - agv_pos) < ROBOT_EFFECTIVE_RADIUS * 3]
        
        # Minimum distance
        min_dist = min((np.linalg.norm(obs.center - agv_pos) for obs in obstacles),
                       default=float('inf'))
        
        # Initialize decision variables
        using_direct_path = False
        detected_gaps = []
        selected_gap = None
        apf_force = None
        trajectory = []
        
        # Initialize control outputs with safe defaults
        best_v: float = 0.0
        best_w: float = 0.0
        reason: str = "Initializing"
        
        # =====================================================================
        # PRIORITY 1: Direct Path Check
        # =====================================================================
        if self.recovery_mode == RecoveryMode.NORMAL:
            path_clear, path_dir, path_clearance = self.check_direct_path(
                obstacles, agv_pos, agv_heading, goal
            )
            
            if path_clear:
                using_direct_path = True
                best_v, best_w = self.go_straight_control(
                    agv_pos, agv_heading, goal, path_clearance
                )
                trajectory = self._predict_trajectory(agv_pos, agv_heading, best_v, best_w)
                reason = "Direct path clear - going straight"
                
        # =====================================================================
        # PRIORITY 2: Gap-Based Navigation with APF
        # =====================================================================
        if not using_direct_path and self.recovery_mode == RecoveryMode.NORMAL:
            # Detect gaps
            detected_gaps = self.detect_gaps(obstacles, agv_pos, agv_heading)
            selected_gap = self.select_best_gap(detected_gaps, goal_dir, agv_heading)
            
            subgoal_angle = selected_gap.center_angle if selected_gap else goal_dir
            
            # Compute APF
            apf_force, apf_angle = self.compute_apf_force(obstacles, agv_pos, subgoal_angle)
            
            # Enhanced DWA
            best_v, best_w, trajectory, dwa_need_recovery = self.enhanced_dwa_control(
                obstacles, agv_pos, agv_heading, goal, subgoal_angle, apf_angle
            )
            
            if dwa_need_recovery:
                self.recovery_mode = RecoveryMode.WALL_FOLLOW
                self.recovery_counter = 0
                reason = "DWA: No valid trajectory, entering recovery"
            else:
                if selected_gap:
                    reason = f"GapNav: Using gap at {np.degrees(selected_gap.center_angle):.0f}°"
                else:
                    reason = "GapNav: Following APF direction"
                    
        # =====================================================================
        # PRIORITY 3: Recovery Modes
        # =====================================================================
        if self.recovery_mode == RecoveryMode.WALL_FOLLOW:
            best_v, best_w = self.wall_follow_control(
                obstacles, agv_pos, agv_heading, self.recovery_direction
            )
            trajectory = self._predict_trajectory(agv_pos, agv_heading, best_v, best_w)
            direction = "left" if self.recovery_direction else "right"
            reason = f"Recovery: Wall-following ({direction})"
            
        elif self.recovery_mode == RecoveryMode.REVERSE_ESCAPE:
            best_v, best_w = self.reverse_escape_control(obstacles, agv_pos, agv_heading)
            trajectory = self._predict_trajectory(agv_pos, agv_heading, best_v, best_w)
            reason = "Recovery: Reverse escape"
            
        elif self.recovery_mode == RecoveryMode.RANDOM_ESCAPE:
            best_v, best_w = self.random_escape_control(agv_heading)
            trajectory = self._predict_trajectory(agv_pos, agv_heading, best_v, best_w)
            reason = "Recovery: Random rotation"
            
        # Update velocities
        self.current_v = best_v
        self.current_w = best_w
        
        # Heading change
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
            best_v = 0.05
            best_w = evasion_w
            
            # Trigger recovery mode
            if self.recovery_mode == RecoveryMode.NORMAL:
                self.recovery_mode = RecoveryMode.WALL_FOLLOW
                self.recovery_counter = 0
                
            action = NavigationAction.TURN_LEFT if evasion_w > 0 else NavigationAction.TURN_RIGHT
            reason = "Evading obstacle - rotating away"
        elif abs(best_w) > np.deg2rad(30):
            action = NavigationAction.TURN_LEFT if best_w > 0 else NavigationAction.TURN_RIGHT
        elif best_v < 0:
            action = NavigationAction.REVERSE
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
            
        return GapNavNavigationDecision(
            action=action,
            target_speed=abs(best_v),
            target_heading_change=heading_change,
            reason=reason,
            critical_obstacles=critical_obs,
            safety_score=safety_score,
            linear_velocity=best_v,
            angular_velocity=best_w,
            predicted_trajectory=trajectory,
            detected_gaps=detected_gaps,
            selected_gap=selected_gap,
            apf_force=apf_force,
            recovery_mode=self.recovery_mode,
            using_direct_path=using_direct_path
        )


# =============================================================================
# GapNav Decision Layer (Complete L5 Alternative)
# =============================================================================

class GapNavDecisionLayer:
    """
    Complete GapNav-based decision layer as alternative to default DecisionLayer.
    
    State-of-the-art navigation combining gap detection, APF, and enhanced DWA
    with multi-layer recovery fallback.
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize GapNav decision layer.
        
        Args:
            dt: Delta time for simulation
        """
        from L5_decision_layer import DecisionTracker
        from L4_detection_layer import LidarProcessor
        
        self.dt = dt
        self.tracker = DecisionTracker(dt)
        self.navigator = GapNavDecisionMaker()
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
                                 agv_heading: float) -> GapNavNavigationDecision:
        """
        Get GapNav-based navigation decision.
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
    
    def get_detected_gaps(self, agv_pos: np.ndarray,
                          agv_heading: float) -> List[DetectedGap]:
        """Get detected gaps for visualization."""
        return self.navigator.detect_gaps(
            self.tracker.get_obstacles(), agv_pos, agv_heading
        )
    
    def get_recovery_mode(self) -> RecoveryMode:
        """Get current recovery mode."""
        return self.navigator.recovery_mode
    
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
            "gapnav_recovery_mode": self.navigator.recovery_mode.value
        }
