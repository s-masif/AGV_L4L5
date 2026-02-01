# =============================================================================
# L5 Decision - GapNav (Gap-Based Navigation + APF)
# =============================================================================
# State-of-the-art algorithm combining:
# - Gap Detection: Finds passable corridors
# - APF Integration: Smooth obstacle repulsion + goal attraction
# - Direct Path Priority: Goes straight when clear
# - Multi-Layer Fallback: Wall-follow → Reverse → Random escape
# =============================================================================

import numpy as np
from typing import List, Optional, Tuple

from ..base import BaseDecisionMaker
from ..types import (
    TrackedObstacle,
    GapNavNavigationDecision,
    DetectedGap,
    NavigationAction,
    ObstacleState,
    RecoveryMode
)

# Import configuration
from ..config import (
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


class GapNavDecisionMaker(BaseDecisionMaker):
    """
    Gap-Based Navigation with APF and Enhanced DWA decision maker.
    
    This is a state-of-the-art approach that:
    1. First checks for direct path to goal (priority)
    2. Detects navigable gaps in obstacle field
    3. Uses APF for smooth obstacle repulsion
    4. Applies enhanced DWA for trajectory selection
    5. Has multi-layer fallback for recovery
    
    Best for: Complex environments with narrow passages and
              dynamic obstacle avoidance requirements.
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
        super().__init__(
            max_speed=max_speed,
            min_speed=min_speed,
            max_angular=max_angular,
            wall_follow_distance=GAPNAV_WALL_FOLLOW_DISTANCE
        )
        
        self.min_gap_width = min_gap_width
        
        # Recovery state machine
        self.recovery_mode = RecoveryMode.NORMAL
        
        # Random generator for random escape
        self.rng = np.random.default_rng(42)
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.recovery_mode = RecoveryMode.NORMAL
        
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
            
            angle_diff = abs(self.normalize_angle(obs_angle - goal_dir))
            if angle_diff < cone_width:
                clearance = obs_dist - ROBOT_EFFECTIVE_RADIUS
                min_clearance = min(min_clearance, clearance)
                
        # Check corridor sides
        left_angle = goal_dir + np.deg2rad(15)
        right_angle = goal_dir - np.deg2rad(15)
        
        left_clearance = self.get_clearance_in_direction(obstacles, agv_pos, left_angle)
        right_clearance = self.get_clearance_in_direction(obstacles, agv_pos, right_angle)
        
        corridor_clear = min(left_clearance, right_clearance) > DIRECT_PATH_CORRIDOR_MARGIN
        
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
        
        Returns:
            (v, w): Linear and angular velocities
        """
        goal_dir = np.arctan2(goal_pos[1] - agv_pos[1], goal_pos[0] - agv_pos[0])
        goal_dist = np.linalg.norm(goal_pos - agv_pos)
        
        angle_error = self.normalize_angle(goal_dir - agv_heading)
        w = np.clip(angle_error * 2.0, -self.max_angular, self.max_angular)
        
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
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            agv_heading: AGV heading
            
        Returns:
            List of detected gaps sorted by score
        """
        if not obstacles:
            return [DetectedGap(
                center_angle=agv_heading,
                angular_width=2 * np.pi,
                linear_width=float('inf'),
                depth=LIDAR_VIRTUAL_RANGE,
                min_depth=LIDAR_VIRTUAL_RANGE,
                score=1.0
            )]
            
        # Angular histogram
        num_sectors = 72
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
            angular_footprint = np.arctan2(ROBOT_EFFECTIVE_RADIUS * 1.5, max(dist, 0.5))
            
            for a in np.linspace(angle - angular_footprint, angle + angular_footprint, 5):
                sector_idx = int(((a + np.pi) % (2 * np.pi)) / sector_size)
                sector_idx = np.clip(sector_idx, 0, num_sectors - 1)
                
                if dist < free_threshold:
                    obstacle_in_sector[sector_idx] = True
                    
                sector_distances[sector_idx] = min(sector_distances[sector_idx], dist)
                
        # Find gaps
        gaps = []
        in_gap = False
        gap_start = 0
        
        extended_occupied = obstacle_in_sector + obstacle_in_sector
        
        for i in range(num_sectors * 2):
            if not extended_occupied[i] and not in_gap:
                in_gap = True
                gap_start = i
            elif extended_occupied[i] and in_gap:
                in_gap = False
                if gap_start < num_sectors:
                    self._process_gap(gaps, gap_start, i - 1, num_sectors, 
                                     sector_size, sector_distances)
                    
        if in_gap and gap_start < num_sectors:
            self._process_gap(gaps, gap_start, num_sectors * 2 - 1, num_sectors,
                             sector_size, sector_distances)
            
        return gaps
    
    def _process_gap(self, gaps: List[DetectedGap], start: int, end: int,
                     num_sectors: int, sector_size: float,
                     sector_distances: List[float]):
        """Process and add a detected gap."""
        gap_sectors = list(range(start, end + 1))
        gap_angles = [(s % num_sectors) * sector_size - np.pi for s in gap_sectors]
        gap_dists = [sector_distances[s % num_sectors] for s in gap_sectors]
        
        if len(gap_angles) < 2:
            return
            
        weights = np.array(gap_dists) / max(np.sum(gap_dists), 0.1)
        center_angle = np.sum(np.array(gap_angles) * weights)
        angular_width = len(gap_sectors) * sector_size
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
        """Select the best gap for navigation."""
        if not gaps:
            return None
            
        best_gap = None
        best_score = -float('inf')
        
        for gap in gaps:
            angle_to_goal = abs(self.normalize_angle(gap.center_angle - goal_direction))
            alignment_score = (np.pi - angle_to_goal) / np.pi
            width_score = np.clip(gap.linear_width / 3.0, 0, 1)
            depth_score = np.clip(gap.depth / LIDAR_VIRTUAL_RANGE, 0, 1)
            
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
        
        Returns:
            (total_force, apf_angle): Force vector and recommended angle
        """
        # Attractive force
        attract_dir = np.array([np.cos(subgoal_angle), np.sin(subgoal_angle)])
        F_attract = APF_ATTRACT_GAIN * attract_dir
        
        # Repulsive force
        F_repel = np.zeros(2)
        
        for obs in obstacles:
            diff = agv_pos - obs.center
            dist = np.linalg.norm(diff)
            
            if dist < APF_REPEL_THRESHOLD and dist > 0.01:
                repel_magnitude = APF_REPEL_GAIN * (1.0 / dist - 1.0 / APF_REPEL_THRESHOLD) / (dist ** 2)
                repel_magnitude = np.clip(repel_magnitude, 0, 2.0)
                
                if obs.state == ObstacleState.DYNAMIC:
                    repel_magnitude *= 1.5
                    
                F_repel += repel_magnitude * (diff / dist)
                
        F_total = F_attract + F_repel
        
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
        
        dist_to_goal = np.linalg.norm(goal_pos - agv_pos)
        
        for v in v_samples:
            for w in w_samples:
                trajectory = self.predict_trajectory(agv_pos, agv_heading, v, w, GAPNAV_PREDICT_TIME)
                
                if not trajectory:
                    continue
                    
                valid, min_clearance = self.check_trajectory_clearance(trajectory, obstacles)
                
                if not valid:
                    continue
                    
                final_x, final_y, final_heading = trajectory[-1]
                final_pos = np.array([final_x, final_y])
                
                # Scores
                heading_diff = abs(self.normalize_angle(subgoal_angle - final_heading))
                heading_score = (np.pi - heading_diff) / np.pi
                clearance_score = np.clip(min_clearance / 2.5, 0, 1)
                velocity_score = v / self.max_speed
                
                final_goal_dist = np.linalg.norm(goal_pos - final_pos)
                progress = (dist_to_goal - final_goal_dist) / max(dist_to_goal, 0.1)
                progress_score = np.clip(progress, -1, 1)
                
                apf_diff = abs(self.normalize_angle(apf_angle - final_heading))
                apf_score = (np.pi - apf_diff) / np.pi
                
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
                    
        return best_v, best_w, best_trajectory, best_score == -float('inf')
    
    # =========================================================================
    # Recovery Behaviors
    # =========================================================================
    
    def reverse_escape_control(self, obstacles: List[TrackedObstacle],
                                agv_pos: np.ndarray,
                                agv_heading: float) -> Tuple[float, float]:
        """Reverse and escape behavior."""
        best_angle = agv_heading
        best_clearance = 0
        
        for angle_offset in np.linspace(-np.pi, np.pi, 36):
            test_angle = agv_heading + angle_offset
            clearance = self.get_clearance_in_direction(obstacles, agv_pos, test_angle)
            if clearance > best_clearance:
                best_clearance = clearance
                best_angle = test_angle
                
        reverse_angle = agv_heading + np.pi
        reverse_clearance = self.get_clearance_in_direction(obstacles, agv_pos, reverse_angle)
        
        if reverse_clearance > ROBOT_EFFECTIVE_RADIUS + 0.3:
            v = -0.3
            angle_diff = self.normalize_angle(best_angle - agv_heading)
            w = np.clip(angle_diff * 0.5, -self.max_angular * 0.5, self.max_angular * 0.5)
        else:
            v = 0.0
            angle_diff = self.normalize_angle(best_angle - agv_heading)
            w = np.clip(angle_diff, -self.max_angular, self.max_angular)
            
        return v, w
    
    def random_escape_control(self, agv_heading: float) -> Tuple[float, float]:
        """Random rotation to break deadlock."""
        v = 0.0
        w = self.rng.choice([-1, 1]) * self.max_angular * 0.8
        return v, w
    
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
        """
        if current_vel is not None:
            self.current_v, self.current_w = current_vel
            
        # Default goal
        goal: np.ndarray
        if goal_pos is None:
            goal = agv_pos + 10.0 * np.array([np.cos(agv_heading), np.sin(agv_heading)])
        else:
            goal = goal_pos
        
        goal_dir = np.arctan2(goal[1] - agv_pos[1], goal[0] - agv_pos[0])
        goal_dist = np.linalg.norm(goal - agv_pos)
        
        in_final_approach = goal_dist < GAPNAV_FINAL_APPROACH_DISTANCE
        effective_tolerance = GAPNAV_FINAL_APPROACH_TOLERANCE if in_final_approach else GOAL_TOLERANCE
        
        # Goal reached
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
        need_recovery = self.update_stuck_detection(agv_pos, goal_dist, goal_dir, agv_heading)
        self._update_recovery_state(need_recovery, in_final_approach)
        
        # Critical obstacles
        critical_obs = [obs.id for obs in obstacles 
                       if np.linalg.norm(obs.center - agv_pos) < ROBOT_EFFECTIVE_RADIUS * 3]
        
        min_dist = min((np.linalg.norm(obs.center - agv_pos) for obs in obstacles),
                       default=float('inf'))
        
        # Initialize
        using_direct_path = False
        detected_gaps = []
        selected_gap = None
        apf_force = None
        trajectory = []
        best_v: float = 0.0
        best_w: float = 0.0
        reason: str = "Initializing"
        
        # PRIORITY 1: Direct Path
        if self.recovery_mode == RecoveryMode.NORMAL:
            path_clear, path_dir, path_clearance = self.check_direct_path(
                obstacles, agv_pos, agv_heading, goal
            )
            
            if path_clear:
                using_direct_path = True
                best_v, best_w = self.go_straight_control(
                    agv_pos, agv_heading, goal, path_clearance
                )
                trajectory = self.predict_trajectory(agv_pos, agv_heading, best_v, best_w)
                reason = "Direct path clear - going straight"
                
        # PRIORITY 2: Gap-Based Navigation
        if not using_direct_path and self.recovery_mode == RecoveryMode.NORMAL:
            detected_gaps = self.detect_gaps(obstacles, agv_pos, agv_heading)
            selected_gap = self.select_best_gap(detected_gaps, goal_dir, agv_heading)
            
            subgoal_angle = selected_gap.center_angle if selected_gap else goal_dir
            apf_force, apf_angle = self.compute_apf_force(obstacles, agv_pos, subgoal_angle)
            
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
                    
        # PRIORITY 3: Recovery Modes
        if self.recovery_mode == RecoveryMode.WALL_FOLLOW:
            best_v, best_w = self.wall_follow_control(
                obstacles, agv_pos, agv_heading, self.recovery_direction
            )
            trajectory = self.predict_trajectory(agv_pos, agv_heading, best_v, best_w)
            reason = f"Recovery: Wall-following ({'left' if self.recovery_direction else 'right'})"
            
        elif self.recovery_mode == RecoveryMode.REVERSE_ESCAPE:
            best_v, best_w = self.reverse_escape_control(obstacles, agv_pos, agv_heading)
            trajectory = self.predict_trajectory(agv_pos, agv_heading, best_v, best_w)
            reason = "Recovery: Reverse escape"
            
        elif self.recovery_mode == RecoveryMode.RANDOM_ESCAPE:
            best_v, best_w = self.random_escape_control(agv_heading)
            trajectory = self.predict_trajectory(agv_pos, agv_heading, best_v, best_w)
            reason = "Recovery: Random rotation"
            
        # Update velocities
        self.current_v = best_v
        self.current_w = best_w
        heading_change = best_w * SIMULATION_DT
        
        # Emergency evasion
        if min_dist < ROBOT_EFFECTIVE_RADIUS * 1.5 and obstacles:
            best_v, evasion_w, heading_change = self.compute_evasion_control(
                obstacles, agv_pos, agv_heading
            )
            best_w = evasion_w
            
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
            
        safety_score = self.compute_safety_score(obstacles, agv_pos)
            
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
