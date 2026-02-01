# =============================================================================
# L5 Decision - Simple Navigation Algorithm
# =============================================================================
# Basic navigation decision maker using repulsive potential field.
# This is the default algorithm, suitable for simple environments.
# =============================================================================

import numpy as np
from typing import List, Optional, Tuple

from ..base import BaseDecisionMaker
from ..types import (
    TrackedObstacle,
    NavigationDecision,
    NavigationAction,
    ObstacleState
)

# Import configuration
from ..config import (
    NAV_SAFETY_DISTANCE,
    NAV_CRITICAL_DISTANCE,
    NAV_MAX_SPEED,
    NAV_MIN_SPEED,
    NAV_MAX_HEADING_CHANGE,
    NAV_MIN_TURN_THRESHOLD,
    EVASION_MIN_REPULSION,
    EVASION_MIN_DISTANCE,
    SAFETY_SCORE_CRITICAL,
    SAFETY_SCORE_WARNING_BASE,
    SAFETY_SCORE_WARNING_FACTOR,
    SAFETY_SCORE_SAFE_BASE,
    SAFETY_SCORE_SAFE_FACTOR,
    SAFETY_SCORE_SAFE_DISTANCE_FACTOR
)


class SimpleDecisionMaker(BaseDecisionMaker):
    """
    Simple navigation decision maker using repulsive potential field.
    
    This is a straightforward algorithm that:
    1. Checks for obstacles within safety/critical distances
    2. Computes a repulsive force from nearby obstacles
    3. Determines speed based on closest obstacle distance
    4. Decides action (continue, slow down, turn, stop)
    
    Best for: Simple, open environments without complex obstacle configurations.
    """
    
    def __init__(self, 
                 safety_distance: float = NAV_SAFETY_DISTANCE,
                 critical_distance: float = NAV_CRITICAL_DISTANCE,
                 max_speed: float = NAV_MAX_SPEED,
                 min_speed: float = NAV_MIN_SPEED):
        """
        Initialize simple decision maker.
        
        Args:
            safety_distance: Distance to start slowing down (m)
            critical_distance: Distance to stop completely (m)
            max_speed: Maximum travel speed (m/s)
            min_speed: Minimum travel speed (m/s)
        """
        super().__init__(max_speed=max_speed, min_speed=min_speed)
        self.safety_distance = safety_distance
        self.critical_distance = critical_distance
    
    def decide(self,
               obstacles: List[TrackedObstacle],
               agv_pos: np.ndarray,
               agv_heading: float,
               goal_pos: Optional[np.ndarray] = None,
               current_vel: Optional[Tuple[float, float]] = None) -> NavigationDecision:
        """
        Make a navigation decision based on obstacles.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position [x, y]
            agv_heading: Current AGV heading (radians)
            goal_pos: Goal position (not used in simple algorithm)
            current_vel: Current velocities (not used in simple algorithm)
            
        Returns:
            NavigationDecision with action and parameters
        """
        # No obstacles: full speed ahead
        if not obstacles:
            return NavigationDecision(
                action=NavigationAction.CONTINUE,
                target_speed=self.max_speed,
                target_heading_change=0.0,
                reason="No obstacle detected",
                critical_obstacles=[],
                safety_score=1.0
            )
        
        # Find critical obstacles (within safety distance)
        critical_obs = [obs for obs in obstacles if obs.d_eq < self.safety_distance]
        
        if not critical_obs:
            return NavigationDecision(
                action=NavigationAction.CONTINUE,
                target_speed=self.max_speed,
                target_heading_change=0.0,
                reason="Obstacles far away",
                critical_obstacles=[],
                safety_score=0.9
            )
        
        # Find closest obstacle
        closest = min(critical_obs, key=lambda o: o.d_eq)
        
        # Critical distance: STOP
        if closest.d_eq < self.critical_distance:
            return NavigationDecision(
                action=NavigationAction.STOP,
                target_speed=0.0,
                target_heading_change=0.0,
                reason=f"Obstacle {closest.id} too close (d={closest.d_eq:.2f}m)",
                critical_obstacles=[o.id for o in critical_obs],
                safety_score=0.1
            )
        
        # Calculate evasion direction
        evasion_heading = self._compute_evasion(obstacles, agv_pos, agv_heading)
        
        # Determine speed based on distance
        speed_factor = (closest.d_eq - self.critical_distance) / \
                       (self.safety_distance - self.critical_distance)
        speed_factor = np.clip(speed_factor, 0.2, 1.0)
        target_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
        
        # Determine action
        if abs(evasion_heading) > NAV_MIN_TURN_THRESHOLD:
            if evasion_heading > 0:
                action = NavigationAction.TURN_LEFT
            else:
                action = NavigationAction.TURN_RIGHT
            reason = f"Avoiding obstacle {closest.id}"
        else:
            action = NavigationAction.SLOW_DOWN
            reason = f"Slowing down for obstacle {closest.id}"
        
        # Calculate safety score
        safety_score = self._compute_detailed_safety_score(obstacles, agv_pos)
        
        return NavigationDecision(
            action=action,
            target_speed=target_speed,
            target_heading_change=evasion_heading,
            reason=reason,
            critical_obstacles=[o.id for o in critical_obs],
            safety_score=safety_score
        )
    
    def _compute_evasion(self, obstacles: List[TrackedObstacle], 
                         agv_pos: np.ndarray, 
                         agv_heading: float) -> float:
        """
        Calculate optimal evasion direction using repulsive potential field.
        
        Each obstacle exerts a repulsive force inversely proportional to
        distance squared. The total repulsion determines the evasion direction.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            agv_heading: Current heading
            
        Returns:
            Suggested heading change (radians, positive = left)
        """
        if not obstacles:
            return 0.0
        
        # Calculate total repulsion from obstacles
        repulsion = np.zeros(2)
        
        for obs in obstacles:
            diff = agv_pos - obs.center
            dist = np.linalg.norm(diff)
            if dist > EVASION_MIN_DISTANCE:
                # Repulsive force inversely proportional to distance squared
                weight = 1.0 / (dist * dist)
                repulsion += weight * diff / dist
        
        # Check if repulsion is significant
        if np.linalg.norm(repulsion) < EVASION_MIN_REPULSION:
            return 0.0
        
        # Calculate repulsion angle
        repulsion_angle = np.arctan2(repulsion[1], repulsion[0])
        
        # Difference from current heading
        heading_change = self.normalize_angle(repulsion_angle - agv_heading)
        
        # Limit maximum change
        return np.clip(heading_change, -NAV_MAX_HEADING_CHANGE, NAV_MAX_HEADING_CHANGE)
    
    def _compute_detailed_safety_score(self, obstacles: List[TrackedObstacle], 
                                       agv_pos: np.ndarray) -> float:
        """
        Calculate detailed safety score (0-1).
        
        Uses configurable thresholds for critical, warning, and safe zones.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            
        Returns:
            Safety score (1 = very safe, 0 = very dangerous)
        """
        if not obstacles:
            return 1.0
        
        min_distance = min(obs.d_eq for obs in obstacles)
        
        if min_distance < self.critical_distance:
            return SAFETY_SCORE_CRITICAL
        elif min_distance < self.safety_distance:
            progress = (min_distance - self.critical_distance) / \
                       (self.safety_distance - self.critical_distance)
            return SAFETY_SCORE_WARNING_BASE + SAFETY_SCORE_WARNING_FACTOR * progress
        else:
            beyond = min(1.0, (min_distance - self.safety_distance) / SAFETY_SCORE_SAFE_DISTANCE_FACTOR)
            return SAFETY_SCORE_SAFE_BASE + SAFETY_SCORE_SAFE_FACTOR * beyond
