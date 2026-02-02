# =============================================================================
# L5 Decision - Complete Decision Layers
# =============================================================================
# Full decision layer implementations that integrate:
# - L4 Detection Layer for obstacle tracking
# - Navigation algorithms for path planning
#
# This is a convenience layer that combines L4 detection with L5 decision.
# =============================================================================

import numpy as np
from typing import List, Optional, Tuple

from .types import (
    NavigationDecision,
    DWANavigationDecision,
    VFHNavigationDecision,
    GapNavNavigationDecision,
    DetectedGap,
    RecoveryMode
)

# Import detection components from L4
from L4_detection import (
    DetectionLayer as L4DetectionLayer,
    ObstacleTracker,
    LidarProcessor,
    TrackedObstacle,
    ObstacleState
)

from .algorithms import (
    SimpleDecisionMaker,
    DWADecisionMaker,
    VFHDecisionMaker,
    GapNavDecisionMaker,
    VODecisionMaker
)

from .config import NAV_SAFETY_DISTANCE


class DecisionLayer:
    """
    Complete decision layer with Simple navigation algorithm.
    
    This is the default decision layer that integrates:
    - L4 Detection for obstacle tracking and classification
    - Simple repulsive field navigation
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize decision layer.
        
        Args:
            dt: Delta time for simulation
        """
        self.dt = dt
        self.detector = L4DetectionLayer(dt)
        self.navigator = SimpleDecisionMaker()
    
    def process_scan(self, ranges: np.ndarray, angles: np.ndarray,
                     agv_pos: np.ndarray, agv_vel: np.ndarray,
                     agv_heading: float) -> List[TrackedObstacle]:
        """
        Process LiDAR scan and update tracker.
        
        Returns:
            List of tracked obstacles
        """
        return self.detector.process_scan(ranges, angles, agv_pos, agv_vel, agv_heading)
    
    def get_navigation_decision(self, agv_pos: np.ndarray, 
                                 agv_heading: float) -> NavigationDecision:
        """Get navigation decision based on current obstacles."""
        return self.navigator.decide(
            self.detector.get_obstacles(), 
            agv_pos, 
            agv_heading
        )
    
    def get_critical_obstacles(self, 
                               safety_distance: float = NAV_SAFETY_DISTANCE) -> List[TrackedObstacle]:
        """Returns critical obstacles within safety distance."""
        return self.detector.get_critical_obstacles(safety_distance)
    
    def get_dynamic_obstacles(self) -> List[TrackedObstacle]:
        """Returns dynamic obstacles."""
        return self.detector.get_dynamic_obstacles()
    
    def get_all_obstacles(self) -> List[TrackedObstacle]:
        """Returns all tracked obstacles."""
        return self.detector.get_obstacles()
    
    def reset(self):
        """Reset the decision layer."""
        self.detector.reset()
        self.navigator.reset()
    
    def export_state(self) -> List[dict]:
        """Export complete state for analysis/debug."""
        obstacles = self.detector.get_obstacles()
        
        return [{
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
        } for obs in obstacles]
    
    def get_statistics(self) -> dict:
        """Returns system statistics."""
        return self.detector.get_statistics()


class DWADecisionLayer:
    """
    Complete decision layer with DWA navigation algorithm.
    
    Uses Dynamic Window Approach for trajectory-based navigation
    with velocity sampling and trajectory evaluation.
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.detector = L4DetectionLayer(dt)
        self.navigator = DWADecisionMaker()
        self.goal_pos: Optional[np.ndarray] = None
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
        """Process LiDAR scan and update tracker."""
        return self.detector.process_scan(ranges, angles, agv_pos, agv_vel, agv_heading)
    
    def get_navigation_decision(self, agv_pos: np.ndarray,
                                 agv_heading: float) -> DWANavigationDecision:
        """Get DWA-based navigation decision."""
        return self.navigator.decide(
            self.detector.get_obstacles(),
            agv_pos,
            agv_heading,
            self.goal_pos,
            self.current_vel
        )
    
    def get_critical_obstacles(self, safety_distance: float = 2.0) -> List[TrackedObstacle]:
        return self.detector.get_critical_obstacles(safety_distance)
    
    def get_all_obstacles(self) -> List[TrackedObstacle]:
        return self.detector.get_obstacles()
    
    def get_predicted_trajectory(self, agv_pos: np.ndarray, 
                                  agv_heading: float) -> List[Tuple[float, float, float]]:
        """Get predicted trajectory for visualization."""
        decision = self.navigator.decide(
            self.detector.get_obstacles(),
            agv_pos,
            agv_heading,
            self.goal_pos,
            self.current_vel
        )
        return decision.predicted_trajectory
    
    def reset(self):
        """Reset the decision layer."""
        self.detector.reset()
        self.navigator.reset()
        self.current_vel = (0.0, 0.0)
    
    def export_state(self) -> List[dict]:
        """Export complete state for analysis/debug."""
        return [{
            "id": obs.id,
            "pos": obs.center.tolist(),
            "vel": obs.velocity.tolist(),
            "state": obs.state.value,
            "d_eq": float(obs.d_eq),
            "d_dot": float(obs.d_dot),
            "confidence": float(obs.confidence),
            "velocity_magnitude": float(np.linalg.norm(obs.velocity)),
            "last_seen": int(obs.last_seen),
            "num_points": len(obs.points)
        } for obs in self.detector.get_obstacles()]
    
    def get_statistics(self) -> dict:
        stats = self.detector.get_statistics()
        stats["dwa_in_recovery"] = self.navigator.in_recovery
        return stats


class VFHDecisionLayer:
    """
    Complete decision layer with VFH navigation algorithm.
    
    Uses Vector Field Histogram for polar histogram-based
    navigation with sector analysis.
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.detector = L4DetectionLayer(dt)
        self.navigator = VFHDecisionMaker()
        self.goal_pos: Optional[np.ndarray] = None
        
    def set_goal(self, goal_pos: np.ndarray):
        """Set navigation goal position."""
        self.goal_pos = goal_pos.copy()
        
    def process_scan(self, ranges: np.ndarray, angles: np.ndarray,
                     agv_pos: np.ndarray, agv_vel: np.ndarray,
                     agv_heading: float) -> List[TrackedObstacle]:
        """Process LiDAR scan and update tracker."""
        return self.detector.process_scan(ranges, angles, agv_pos, agv_vel, agv_heading)
    
    def get_navigation_decision(self, agv_pos: np.ndarray,
                                 agv_heading: float) -> VFHNavigationDecision:
        """Get VFH-based navigation decision."""
        return self.navigator.decide(
            self.detector.get_obstacles(),
            agv_pos,
            agv_heading,
            self.goal_pos
        )
    
    def get_critical_obstacles(self, safety_distance: float = 2.0) -> List[TrackedObstacle]:
        return self.detector.get_critical_obstacles(safety_distance)
    
    def get_all_obstacles(self) -> List[TrackedObstacle]:
        return self.detector.get_obstacles()
    
    def get_histogram(self, agv_pos: np.ndarray, agv_heading: float) -> np.ndarray:
        """Get current VFH polar histogram for visualization."""
        return self.navigator.build_polar_histogram(
            self.detector.get_obstacles(), agv_pos, agv_heading
        )
    
    def reset(self):
        """Reset the decision layer."""
        self.detector.reset()
        self.navigator.reset()
    
    def export_state(self) -> List[dict]:
        """Export complete state for analysis/debug."""
        return [{
            "id": obs.id,
            "pos": obs.center.tolist(),
            "vel": obs.velocity.tolist(),
            "state": obs.state.value,
            "d_eq": float(obs.d_eq),
            "d_dot": float(obs.d_dot),
            "confidence": float(obs.confidence),
            "velocity_magnitude": float(np.linalg.norm(obs.velocity)),
            "last_seen": int(obs.last_seen),
            "num_points": len(obs.points)
        } for obs in self.detector.get_obstacles()]
    
    def get_statistics(self) -> dict:
        stats = self.detector.get_statistics()
        stats["vfh_in_recovery"] = self.navigator.wall_follow_mode
        return stats


class GapNavDecisionLayer:
    """
    Complete decision layer with GapNav navigation algorithm.
    
    State-of-the-art navigation combining gap detection, APF,
    and enhanced DWA with multi-layer recovery.
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.detector = L4DetectionLayer(dt)
        self.navigator = GapNavDecisionMaker()
        self.goal_pos: Optional[np.ndarray] = None
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
        """Process LiDAR scan and update tracker."""
        return self.detector.process_scan(ranges, angles, agv_pos, agv_vel, agv_heading)
    
    def get_navigation_decision(self, agv_pos: np.ndarray,
                                 agv_heading: float) -> GapNavNavigationDecision:
        """Get GapNav-based navigation decision."""
        return self.navigator.decide(
            self.detector.get_obstacles(),
            agv_pos,
            agv_heading,
            self.goal_pos,
            self.current_vel
        )
    
    def get_critical_obstacles(self, safety_distance: float = 2.0) -> List[TrackedObstacle]:
        return self.detector.get_critical_obstacles(safety_distance)
    
    def get_all_obstacles(self) -> List[TrackedObstacle]:
        return self.detector.get_obstacles()
    
    def get_detected_gaps(self, agv_pos: np.ndarray,
                          agv_heading: float) -> List[DetectedGap]:
        """Get detected gaps for visualization."""
        return self.navigator.detect_gaps(
            self.detector.get_obstacles(), agv_pos, agv_heading
        )
    
    def get_recovery_mode(self) -> RecoveryMode:
        """Get current recovery mode."""
        return self.navigator.recovery_mode
    
    def reset(self):
        """Reset the decision layer."""
        self.detector.reset()
        self.navigator.reset()
        self.current_vel = (0.0, 0.0)
    
    def export_state(self) -> List[dict]:
        """Export complete state for analysis/debug."""
        return [{
            "id": obs.id,
            "pos": obs.center.tolist(),
            "vel": obs.velocity.tolist(),
            "state": obs.state.value,
            "d_eq": float(obs.d_eq),
            "d_dot": float(obs.d_dot),
            "confidence": float(obs.confidence),
            "velocity_magnitude": float(np.linalg.norm(obs.velocity)),
            "last_seen": int(obs.last_seen),
            "num_points": len(obs.points)
        } for obs in self.detector.get_obstacles()]
    
    def get_statistics(self) -> dict:
        stats = self.detector.get_statistics()
        stats["gapnav_recovery_mode"] = self.navigator.recovery_mode.value
        return stats


class VODecisionLayer:
    """
    Complete decision layer with Velocity Obstacles (VO) algorithm.
    
    Standalone algorithm for dynamic obstacle avoidance:
    - Time-To-Collision (TTC) prediction
    - Velocity Obstacles for collision avoidance
    - Intelligent avoidance strategies (pass behind, slow down, stop)
    
    Best for scenarios with dynamic/moving obstacles.
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.detector = L4DetectionLayer(dt)
        self.navigator = VODecisionMaker()
        self.goal_pos: Optional[np.ndarray] = None
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
        """Process LiDAR scan and update tracker."""
        return self.detector.process_scan(ranges, angles, agv_pos, agv_vel, agv_heading)
    
    def get_navigation_decision(self, agv_pos: np.ndarray,
                                 agv_heading: float) -> NavigationDecision:
        """Get VO-based navigation decision."""
        return self.navigator.decide(
            self.detector.get_obstacles(),
            agv_pos,
            agv_heading,
            self.goal_pos,
            self.current_vel
        )
    
    def get_critical_obstacles(self, safety_distance: float = 2.0) -> List[TrackedObstacle]:
        return self.detector.get_critical_obstacles(safety_distance)
    
    def get_all_obstacles(self) -> List[TrackedObstacle]:
        return self.detector.get_obstacles()
    
    def get_active_threats(self):
        """Get current collision threats from VO calculator."""
        return self.navigator.get_active_threats()
    
    def get_avoidance_strategy(self) -> str:
        """Get current avoidance strategy."""
        return self.navigator.get_avoidance_strategy()
    
    def reset(self):
        """Reset the decision layer."""
        self.detector.reset()
        self.navigator.reset()
        self.current_vel = (0.0, 0.0)
    
    def export_state(self) -> List[dict]:
        """Export complete state for analysis/debug."""
        return [{
            "id": obs.id,
            "pos": obs.center.tolist(),
            "vel": obs.velocity.tolist(),
            "state": obs.state.value,
            "d_eq": float(obs.d_eq),
            "d_dot": float(obs.d_dot),
            "confidence": float(obs.confidence),
            "velocity_magnitude": float(np.linalg.norm(obs.velocity)),
            "last_seen": int(obs.last_seen),
            "num_points": len(obs.points)
        } for obs in self.detector.get_obstacles()]
    
    def get_statistics(self) -> dict:
        stats = self.detector.get_statistics()
        stats["avoidance_strategy"] = self.navigator.get_avoidance_strategy()
        stats["active_threats"] = len(self.navigator.get_active_threats())
        return stats
