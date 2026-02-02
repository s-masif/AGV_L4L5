# =============================================================================
# L5 Decision - Velocity Obstacles (VO) Algorithm
# =============================================================================
# Standalone algorithm for dynamic obstacle avoidance using:
# - Time-To-Collision (TTC): Predicts when collision will occur
# - Velocity Obstacles (VO): Computes forbidden velocity regions
#
# Algorithm:
# 1. For each dynamic obstacle, calculate TTC
# 2. If TTC < critical threshold, compute VO cone (forbidden velocities)
# 3. Choose velocity outside the cone:
#    - Pass behind (preferred - slow down to let obstacle pass)
#    - Slow down / stop (if obstacle passes in front)
#    - Accelerate (if can pass before obstacle)
#
# This is a STANDALONE algorithm - does NOT use GapNav, APF, or DWA.
# =============================================================================

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..base import BaseDecisionMaker
from ..types import (
    TrackedObstacle,
    NavigationDecision,
    NavigationAction,
    ObstacleState
)

from ..config import (
    ROBOT_RADIUS,
    ROBOT_EFFECTIVE_RADIUS,
    MAX_LINEAR_VELOCITY,
    MIN_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    SIMULATION_DT,
    GOAL_TOLERANCE,
    NAV_CRITICAL_DISTANCE,
    NAV_SAFETY_DISTANCE,
    # VO Parameters
    VO_TIME_HORIZON,
    VO_TTC_CRITICAL,
    VO_TTC_WARNING,
    VO_MIN_RELATIVE_VELOCITY,
    VO_CONE_SAFETY_MARGIN,
    VO_OBSTACLE_SAFETY_RADIUS,
    VO_STATIC_REPULSION_DISTANCE,
    VO_STATIC_REPULSION_STRENGTH,
    VO_DANGER_DISTANCE,
    VO_EMERGENCY_REVERSE_DISTANCE
)


@dataclass
class CollisionThreat:
    """Represents a potential collision with a dynamic obstacle."""
    obstacle_id: int
    obstacle_center: np.ndarray
    obstacle_velocity: np.ndarray
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    time_to_collision: float
    collision_point: np.ndarray
    cone_left_angle: float      # Left edge of VO cone
    cone_right_angle: float     # Right edge of VO cone
    threat_level: str           # 'CRITICAL', 'WARNING', 'MONITOR'
    

class VelocityObstacleCalculator:
    """
    Calculates Velocity Obstacles for dynamic obstacle avoidance.
    
    A Velocity Obstacle (VO) represents the set of velocities that would
    lead to a collision with a moving obstacle within a time horizon.
    """
    
    def __init__(self, 
                 time_horizon: float = VO_TIME_HORIZON,
                 robot_radius: float = ROBOT_EFFECTIVE_RADIUS):
        self.time_horizon = time_horizon
        self.robot_radius = robot_radius
    
    def compute_time_to_collision(self,
                                   rel_pos: np.ndarray,
                                   rel_vel: np.ndarray,
                                   combined_radius: float) -> Tuple[float, np.ndarray]:
        """
        Compute Time-To-Collision (TTC) between AGV and obstacle.
        
        Uses the quadratic formula to find when distance equals combined radius.
        
        Args:
            rel_pos: Relative position (obstacle - AGV)
            rel_vel: Relative velocity (obstacle_vel - AGV_vel)
            combined_radius: Sum of robot and obstacle radii
            
        Returns:
            (ttc, collision_point): Time to collision and predicted collision point
        """
        # Quadratic coefficients: |rel_pos + t * rel_vel|^2 = combined_radius^2
        a = np.dot(rel_vel, rel_vel)
        b = 2 * np.dot(rel_pos, rel_vel)
        c = np.dot(rel_pos, rel_pos) - combined_radius**2
        
        # Already colliding?
        if c < 0:
            return 0.0, rel_pos
        
        # No relative velocity = no collision (or already at rest)
        if a < 1e-6:
            return float('inf'), np.array([0, 0])
        
        discriminant = b**2 - 4*a*c
        
        # No collision (trajectories don't intersect)
        if discriminant < 0:
            return float('inf'), np.array([0, 0])
        
        # Find smallest positive root
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        # Take the smaller positive time
        if t1 > 0:
            ttc = t1
        elif t2 > 0:
            ttc = t2
        else:
            # Collision in the past
            return float('inf'), np.array([0, 0])
        
        # Collision point
        collision_point = rel_pos + ttc * rel_vel
        
        return ttc, collision_point
    
    def compute_velocity_obstacle_cone(self,
                                        rel_pos: np.ndarray,
                                        obstacle_vel: np.ndarray,
                                        combined_radius: float) -> Tuple[float, float, float]:
        """
        Compute the Velocity Obstacle cone.
        
        The VO cone represents all AGV velocities that would lead to
        collision within the time horizon.
        
        Args:
            rel_pos: Relative position (obstacle - AGV)
            obstacle_vel: Obstacle velocity in world frame
            combined_radius: Sum of robot and obstacle radii
            
        Returns:
            (cone_center_angle, cone_half_angle, apex_distance)
        """
        dist = np.linalg.norm(rel_pos)
        
        if dist < combined_radius:
            # Already overlapping - full cone
            return np.arctan2(rel_pos[1], rel_pos[0]), np.pi, 0.0
        
        # Direction to obstacle
        direction_angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        # Half-angle of the cone (based on combined radius)
        # sin(half_angle) = combined_radius / distance
        sin_half = min(combined_radius / dist, 1.0)
        cone_half_angle = np.arcsin(sin_half) + VO_CONE_SAFETY_MARGIN
        
        return direction_angle, cone_half_angle, dist
    
    def analyze_dynamic_obstacles(self,
                                   obstacles: List[TrackedObstacle],
                                   agv_pos: np.ndarray,
                                   agv_vel: np.ndarray) -> List[CollisionThreat]:
        """
        Analyze all dynamic obstacles and identify collision threats.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            agv_vel: AGV velocity
            
        Returns:
            List of CollisionThreat objects sorted by TTC
        """
        threats = []
        
        for obs in obstacles:
            # Only analyze dynamic obstacles
            if obs.state != ObstacleState.DYNAMIC:
                continue
            
            obs_vel = obs.velocity
            obs_vel_magnitude = np.linalg.norm(obs_vel)
            
            # Skip nearly stationary "dynamic" obstacles
            if obs_vel_magnitude < VO_MIN_RELATIVE_VELOCITY:
                continue
            
            # Relative position and velocity
            rel_pos = obs.center - agv_pos
            rel_vel = obs_vel - agv_vel
            
            # === CHECK IF OBSTACLE IS MOVING AWAY ===
            # If the obstacle's velocity has a component away from AGV,
            # and the distance is increasing, it's not a threat
            distance = np.linalg.norm(rel_pos)
            if distance > 0.01:
                # Direction from AGV to obstacle
                dir_to_obs = rel_pos / distance
                # Obstacle velocity component along this direction
                # Positive = moving away from AGV, Negative = approaching
                obs_radial_vel = np.dot(obs_vel, dir_to_obs)
                
                # If obstacle is moving away (positive radial velocity)
                # and is already at a safe distance, skip it
                if obs_radial_vel > 0.1 and distance > NAV_SAFETY_DISTANCE:
                    continue  # Obstacle moving away - not a threat
                
                # If relative velocity is increasing distance, also skip
                radial_closing_speed = -np.dot(rel_vel, dir_to_obs)  # Positive = closing
                if radial_closing_speed < -0.1 and distance > ROBOT_EFFECTIVE_RADIUS + 1.0:
                    continue  # Distance is increasing - not a threat
            
            # Combined radius (AGV + obstacle + safety margin)
            obs_radius = getattr(obs, 'radius', 0.35)
            combined_radius = self.robot_radius + obs_radius + VO_OBSTACLE_SAFETY_RADIUS
            
            # Compute TTC
            ttc, collision_point = self.compute_time_to_collision(
                rel_pos, rel_vel, combined_radius
            )
            
            # Skip if no collision within horizon
            if ttc > self.time_horizon:
                continue
            
            # Compute VO cone
            cone_center, cone_half, _ = self.compute_velocity_obstacle_cone(
                rel_pos, obs_vel, combined_radius
            )
            
            # Determine threat level
            if ttc < VO_TTC_CRITICAL:
                threat_level = 'CRITICAL'
            elif ttc < VO_TTC_WARNING:
                threat_level = 'WARNING'
            else:
                threat_level = 'MONITOR'
            
            threats.append(CollisionThreat(
                obstacle_id=obs.id,
                obstacle_center=obs.center.copy(),
                obstacle_velocity=obs_vel.copy(),
                relative_position=rel_pos,
                relative_velocity=rel_vel,
                time_to_collision=ttc,
                collision_point=collision_point,
                cone_left_angle=cone_center + cone_half,
                cone_right_angle=cone_center - cone_half,
                threat_level=threat_level
            ))
        
        # Sort by TTC (most urgent first)
        threats.sort(key=lambda t: t.time_to_collision)
        
        return threats
    
    def is_velocity_in_vo_cone(self,
                                velocity: np.ndarray,
                                threat: CollisionThreat) -> bool:
        """
        Check if a velocity vector falls within a VO cone.
        
        Args:
            velocity: Proposed AGV velocity
            threat: CollisionThreat to check against
            
        Returns:
            True if velocity would lead to collision
        """
        if np.linalg.norm(velocity) < 0.01:
            # Zero velocity - check if obstacle is approaching
            rel_vel = threat.obstacle_velocity
            approaching = np.dot(threat.relative_position, rel_vel) < 0
            return approaching and threat.time_to_collision < VO_TTC_CRITICAL
        
        # Velocity direction
        vel_angle = np.arctan2(velocity[1], velocity[0])
        
        # Normalize angles to [-π, π]
        def normalize_angle(a):
            while a > np.pi:
                a -= 2*np.pi
            while a < -np.pi:
                a += 2*np.pi
            return a
        
        vel_angle = normalize_angle(vel_angle)
        left = normalize_angle(threat.cone_left_angle)
        right = normalize_angle(threat.cone_right_angle)
        
        # Check if velocity angle is within cone
        if left > right:
            # Normal case
            return right <= vel_angle <= left
        else:
            # Cone wraps around ±π
            return vel_angle >= right or vel_angle <= left
    
    def compute_avoidance_velocity(self,
                                    current_vel: np.ndarray,
                                    desired_vel: np.ndarray,
                                    threats: List[CollisionThreat],
                                    max_speed: float) -> Tuple[np.ndarray, str]:
        """
        Compute safe velocity that avoids all VO cones.
        
        Strategies (in order of preference):
        1. Pass behind the obstacle (slow down to let it pass)
        2. Slow down / stop (if obstacle passes in front)
        3. Pass in front (accelerate if faster than obstacle)
        
        Args:
            current_vel: Current AGV velocity
            desired_vel: Desired velocity toward goal
            threats: List of collision threats
            max_speed: Maximum allowed speed
            
        Returns:
            (safe_velocity, strategy_used)
        """
        if not threats:
            return desired_vel, "no_threats"
        
        # Most critical threat
        critical_threat = threats[0]
        
        # If already safe, return desired velocity
        if not self.is_velocity_in_vo_cone(desired_vel, critical_threat):
            return desired_vel, "desired_safe"
        
        # --- Strategy 1: Pass behind (PREFERRED) ---
        # Velocity perpendicular to obstacle's path, behind it
        obs_dir = np.arctan2(critical_threat.obstacle_velocity[1],
                             critical_threat.obstacle_velocity[0])
        
        # "Behind" means perpendicular, opposite to obstacle direction
        # Check which side is "behind" based on relative position
        cross = (critical_threat.relative_position[0] * critical_threat.obstacle_velocity[1] -
                 critical_threat.relative_position[1] * critical_threat.obstacle_velocity[0])
        
        if cross > 0:
            # Obstacle is to our left, pass behind (go right relative to obstacle)
            behind_angle = obs_dir - np.pi/2
        else:
            # Obstacle is to our right, pass behind (go left relative to obstacle)
            behind_angle = obs_dir + np.pi/2
        
        # Try passing behind at various speeds
        for speed_factor in [0.8, 0.6, 0.4, 1.0]:
            speed = max_speed * speed_factor
            behind_vel = np.array([np.cos(behind_angle), np.sin(behind_angle)]) * speed
            
            if not self.is_velocity_in_vo_cone(behind_vel, critical_threat):
                # Verify against all threats
                safe = True
                for t in threats[1:]:
                    if self.is_velocity_in_vo_cone(behind_vel, t):
                        safe = False
                        break
                if safe:
                    return behind_vel, "pass_behind"
        
        # --- Strategy 2: Slow down / stop ---
        if critical_threat.threat_level == 'CRITICAL':
            # Emergency stop
            return np.array([0.0, 0.0]), "emergency_stop"
        
        # Try reduced speeds in current direction
        desired_dir = desired_vel / (np.linalg.norm(desired_vel) + 0.01)
        for speed_factor in [0.5, 0.3, 0.1]:
            slow_vel = desired_dir * max_speed * speed_factor
            if not self.is_velocity_in_vo_cone(slow_vel, critical_threat):
                return slow_vel, "slow_down"
        
        # --- Strategy 3: Pass in front (risky, only if TTC allows) ---
        if critical_threat.time_to_collision > VO_TTC_WARNING:
            front_angle = obs_dir + (np.pi/2 if cross > 0 else -np.pi/2)
            front_vel = np.array([np.cos(front_angle), np.sin(front_angle)]) * max_speed
            
            if not self.is_velocity_in_vo_cone(front_vel, critical_threat):
                return front_vel, "pass_front"
        
        # Fallback: stop
        return np.array([0.0, 0.0]), "fallback_stop"


class VODecisionMaker(BaseDecisionMaker):
    """
    Velocity Obstacles (VO) Decision Maker.
    
    Standalone algorithm for dynamic obstacle avoidance:
    - Time-To-Collision (TTC) calculation for moving obstacles
    - Velocity Obstacle cones to identify forbidden velocities
    - Intelligent avoidance strategies (pass behind, slow down, stop)
    - Goal-directed navigation with dynamic obstacle awareness
    
    This is a completely independent algorithm.
    """
    
    def __init__(self,
                 max_speed: float = MAX_LINEAR_VELOCITY,
                 min_speed: float = MIN_LINEAR_VELOCITY,
                 max_angular: float = MAX_ANGULAR_VELOCITY,
                 vo_time_horizon: float = VO_TIME_HORIZON):
        """
        Initialize VO-based decision maker.
        
        Args:
            max_speed: Maximum linear velocity (m/s)
            min_speed: Minimum linear velocity (m/s)
            max_angular: Maximum angular velocity (rad/s)
            vo_time_horizon: Time horizon for collision prediction (s)
        """
        super().__init__()
        
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_angular = max_angular
        
        self.vo_calculator = VelocityObstacleCalculator(
            time_horizon=vo_time_horizon,
            robot_radius=ROBOT_EFFECTIVE_RADIUS
        )
        
        # Current velocity state
        self.current_v = 0.0
        self.current_w = 0.0
        
        # Track active threats for visualization
        self.active_threats: List[CollisionThreat] = []
        self.last_avoidance_strategy = "none"
    
    def reset(self):
        """Reset internal state."""
        self.current_v = 0.0
        self.current_w = 0.0
        self.active_threats = []
        self.last_avoidance_strategy = "none"
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def compute_goal_velocity(self, 
                               agv_pos: np.ndarray, 
                               goal_pos: np.ndarray,
                               agv_heading: float) -> np.ndarray:
        """
        Compute desired velocity toward goal.
        
        Args:
            agv_pos: Current AGV position
            goal_pos: Goal position
            agv_heading: Current heading
            
        Returns:
            Desired velocity vector
        """
        to_goal = goal_pos - agv_pos
        dist_to_goal = np.linalg.norm(to_goal)
        
        if dist_to_goal < GOAL_TOLERANCE:
            return np.array([0.0, 0.0])
        
        # Direction to goal
        goal_dir = to_goal / dist_to_goal
        
        # Speed based on distance (slow down near goal)
        if dist_to_goal < 1.0:
            speed = self.max_speed * (dist_to_goal / 1.0)
        else:
            speed = self.max_speed
        
        return goal_dir * speed
    
    def compute_obstacle_repulsion(self,
                                    obstacles: List[TrackedObstacle],
                                    agv_pos: np.ndarray) -> np.ndarray:
        """
        Compute repulsive force from static obstacles.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            
        Returns:
            Repulsion velocity vector
        """
        repulsion = np.array([0.0, 0.0])
        
        for obs in obstacles:
            # Only repel from static obstacles (VO handles dynamic)
            if obs.state == ObstacleState.DYNAMIC:
                continue
                
            to_obs = obs.center - agv_pos
            dist = np.linalg.norm(to_obs)
            
            if dist < VO_STATIC_REPULSION_DISTANCE and dist > 0.01:
                # Inverse-distance repulsion (stronger when closer)
                strength = (VO_STATIC_REPULSION_DISTANCE - dist) / VO_STATIC_REPULSION_DISTANCE
                # Quadratic falloff for stronger close-range repulsion
                strength = strength ** 1.5
                repulsion -= (to_obs / dist) * strength * VO_STATIC_REPULSION_STRENGTH
        
        return repulsion
    
    def decide(self,
               obstacles: List[TrackedObstacle],
               agv_pos: np.ndarray,
               agv_heading: float,
               goal_pos: Optional[np.ndarray] = None,
               current_vel: Optional[Tuple[float, float]] = None) -> NavigationDecision:
        """
        Make navigation decision with Velocity Obstacles.
        
        Algorithm:
        1. Compute desired velocity toward goal
        2. Add static obstacle repulsion
        3. Analyze dynamic obstacles for collision threats (TTC)
        4. If threats detected, compute safe avoidance velocity (VO)
        5. Convert to (speed, heading_change) command
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: Current AGV position
            agv_heading: Current heading (radians)
            goal_pos: Optional goal position
            current_vel: Optional (v, w) tuple
            
        Returns:
            NavigationDecision with speed and heading commands
        """
        # Update current velocity
        if current_vel is not None:
            self.current_v, self.current_w = current_vel
        
        # Current velocity vector
        agv_vel = np.array([
            self.current_v * np.cos(agv_heading),
            self.current_v * np.sin(agv_heading)
        ])
        
        # === PRIORITY CHECK: Minimum distance to ALL obstacles ===
        # Different thresholds for static vs dynamic obstacles
        # Static: can enter WARNING zone, but never DANGER
        # Dynamic: full conservative thresholds apply
        min_edge_distance = float('inf')
        closest_obs_id = None
        closest_obs_center = None
        closest_is_static = True
        obs_type = None  # Ensure obs_type is always defined
        
        for obs in obstacles:
            obs_radius = getattr(obs, 'radius', 0.3)
            edge_dist = np.linalg.norm(obs.center - agv_pos) - ROBOT_RADIUS - obs_radius
            if edge_dist < min_edge_distance:
                min_edge_distance = edge_dist
                closest_obs_id = obs.id
                closest_obs_center = obs.center
                closest_is_static = (obs.state == ObstacleState.STATIC)
        # Set obs_type if an obstacle is found
        if closest_obs_id is not None:
            obs_type = "static" if closest_is_static else "dynamic"
        else:
            obs_type = "unknown"
        
        # Thresholds depend on obstacle type
        # Static obstacles: reduced thresholds (can pass closer)
        # Dynamic obstacles: full conservative thresholds
        if closest_is_static:
            danger_threshold = VO_DANGER_DISTANCE * 0.5  # 0.3m for static
            reverse_threshold = VO_EMERGENCY_REVERSE_DISTANCE * 0.5  # 0.15m for static
        else:
            danger_threshold = VO_DANGER_DISTANCE  # 0.6m for dynamic
            reverse_threshold = VO_EMERGENCY_REVERSE_DISTANCE  # 0.3m for dynamic
        
        # EMERGENCY REVERSE: Too close!
        if min_edge_distance < reverse_threshold and closest_obs_center is not None:
            # Reverse away from obstacle
            away_dir = agv_pos - closest_obs_center
            away_dir = away_dir / (np.linalg.norm(away_dir) + 0.001)
            reverse_heading = np.arctan2(away_dir[1], away_dir[0])
            heading_change = self.normalize_angle(reverse_heading - agv_heading)
            return NavigationDecision(
                action=NavigationAction.STOP,
                target_speed=-0.2,  # Reverse slowly
                target_heading_change=float(np.clip(heading_change, -self.max_angular, self.max_angular)),
                reason=f"VO: EMERGENCY REVERSE! {obs_type} Obs {closest_obs_id} at {min_edge_distance:.2f}m",
                critical_obstacles=[closest_obs_id],
                safety_score=0.05
            )
        
        # DANGER STOP: Approaching danger zone
        if min_edge_distance < danger_threshold and closest_obs_center is not None:
            # Stop and turn away
            away_dir = agv_pos - closest_obs_center
            away_dir = away_dir / (np.linalg.norm(away_dir) + 0.001)
            escape_heading = np.arctan2(away_dir[1], away_dir[0])
            heading_change = self.normalize_angle(escape_heading - agv_heading)
            
            return NavigationDecision(
                action=NavigationAction.STOP,
                target_speed=0.0,
                target_heading_change=float(np.clip(heading_change, -self.max_angular, self.max_angular)),
                reason=f"VO: DANGER STOP! {obs_type} Obs {closest_obs_id} at {min_edge_distance:.2f}m",
                critical_obstacles=[closest_obs_id],
                safety_score=0.1
            )
        
        # Step 1: Compute desired velocity toward goal
        if goal_pos is not None:
            desired_vel = self.compute_goal_velocity(agv_pos, goal_pos, agv_heading)
        else:
            # No goal - maintain current direction
            desired_vel = np.array([
                self.max_speed * 0.5 * np.cos(agv_heading),
                self.max_speed * 0.5 * np.sin(agv_heading)
            ])
        
        # Step 2: Add static obstacle repulsion
        repulsion = self.compute_obstacle_repulsion(obstacles, agv_pos)
        desired_vel = desired_vel + repulsion
        
        # Limit to max speed
        speed = np.linalg.norm(desired_vel)
        if speed > self.max_speed:
            desired_vel = desired_vel / speed * self.max_speed
        
        # Step 3: Analyze dynamic obstacles for collision threats
        self.active_threats = self.vo_calculator.analyze_dynamic_obstacles(
            obstacles, agv_pos, agv_vel
        )
        
        # Step 4: Handle collision threats with VO
        critical_threats = [t for t in self.active_threats 
                          if t.threat_level in ['CRITICAL', 'WARNING']]
        
        if critical_threats:
            # Compute safe avoidance velocity
            safe_vel, strategy = self.vo_calculator.compute_avoidance_velocity(
                agv_vel, desired_vel, critical_threats, self.max_speed
            )
            self.last_avoidance_strategy = strategy
            final_vel = safe_vel
            
            # Determine action and reason
            if strategy in ['emergency_stop', 'fallback_stop']:
                action = NavigationAction.STOP
                reason = f"VO: {strategy} - TTC={critical_threats[0].time_to_collision:.1f}s"
            elif strategy == 'slow_down':
                action = NavigationAction.SLOW_DOWN
                reason = f"VO: slowing - TTC={critical_threats[0].time_to_collision:.1f}s"
            elif strategy in ['pass_behind', 'pass_front']:
                action = NavigationAction.CONTINUE
                reason = f"VO: {strategy} - avoiding Obs {critical_threats[0].obstacle_id}"
            else:
                action = NavigationAction.CONTINUE
                reason = f"VO: {strategy}"
        else:
            # === NO THREATS - GO STRAIGHT TO GOAL ===
            self.last_avoidance_strategy = "direct"
            
            # Check if path to goal is clear (no static obstacles blocking)
            path_clear = True
            blocking_obs_id = None
            
            if goal_pos is not None:
                goal_dir = goal_pos - agv_pos
                goal_dist = np.linalg.norm(goal_dir)
                
                if goal_dist > 0.1:
                    goal_dir_normalized = goal_dir / goal_dist
                    
                    for obs in obstacles:
                        # Project obstacle onto goal path
                        to_obs = obs.center - agv_pos
                        projection = np.dot(to_obs, goal_dir_normalized)
                        
                        # Only check obstacles ahead of us and before goal
                        if projection > 0 and projection < goal_dist:
                            # Perpendicular distance to path
                            perp_dist = np.linalg.norm(to_obs - projection * goal_dir_normalized)
                            obs_radius = getattr(obs, 'radius', 0.3)
                            clearance = perp_dist - ROBOT_EFFECTIVE_RADIUS - obs_radius
                            
                            if clearance < 0.3:  # Path is blocked
                                path_clear = False
                                blocking_obs_id = obs.id
                                break
            
            if path_clear and goal_pos is not None:
                # Direct path to goal!
                goal_dir = goal_pos - agv_pos
                goal_dist = np.linalg.norm(goal_dir)
                if goal_dist > 0.1:
                    final_vel = (goal_dir / goal_dist) * self.max_speed
                    action = NavigationAction.CONTINUE
                    reason = "VO: direct path to goal"
                else:
                    final_vel = np.array([0.0, 0.0])
                    action = NavigationAction.STOP
                    reason = "VO: goal reached"
            else:
                # Use repulsion-adjusted velocity
                final_vel = desired_vel
                action = NavigationAction.CONTINUE
                if blocking_obs_id:
                    reason = f"VO: navigating around Obs {blocking_obs_id}"
                else:
                    reason = "VO: clear path"
        
        # Step 5: Convert velocity to speed and heading change
        final_speed = np.linalg.norm(final_vel)
        
        if final_speed > 0.01:
            target_heading = np.arctan2(final_vel[1], final_vel[0])
            heading_change = self.normalize_angle(target_heading - agv_heading)
        else:
            heading_change = 0.0
        
        # Find closest obstacle distance
        min_dist = float('inf')
        critical_obs = []
        for obs in obstacles:
            dist = np.linalg.norm(obs.center - agv_pos) - getattr(obs, 'radius', 0.35)
            if dist < min_dist:
                min_dist = dist
            if dist < NAV_CRITICAL_DISTANCE:
                critical_obs.append(obs.id)
        
        # Add threat IDs to critical obstacles
        for t in critical_threats:
            if t.obstacle_id not in critical_obs:
                critical_obs.append(t.obstacle_id)
        
        # Compute safety score
        if critical_threats:
            min_ttc = min(t.time_to_collision for t in critical_threats)
            safety_score = max(0.1, min(1.0, min_ttc / VO_TIME_HORIZON))
        else:
            safety_score = min(1.0, min_dist / NAV_SAFETY_DISTANCE) if min_dist < float('inf') else 1.0
        
        return NavigationDecision(
            action=action,
            target_speed=float(np.clip(final_speed, 0, self.max_speed)),
            target_heading_change=float(np.clip(heading_change, -self.max_angular, self.max_angular)),
            reason=reason,
            critical_obstacles=critical_obs,
            safety_score=safety_score
        )
    
    def get_active_threats(self) -> List[CollisionThreat]:
        """Get list of currently tracked collision threats."""
        return self.active_threats
    
    def get_avoidance_strategy(self) -> str:
        """Get the last used avoidance strategy."""
        return self.last_avoidance_strategy
