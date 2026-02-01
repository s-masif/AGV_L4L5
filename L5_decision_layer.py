# =============================================================================
# L5 - DECISION LAYER
# =============================================================================
# This layer decides AGV direction and speed based on obstacles:
# - Obstacle classification (STATIC/DYNAMIC/UNKNOWN)
# - HySDG-ESD calculation (equivalent distance and its derivative)
# - Multi-Object Tracker with history
# - Decision logic for navigation
# =============================================================================

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

# Import dal layer di detection
from L4_detection_layer import (
    DetectedCluster, 
    ExtendedKalmanFilterCV,
    LidarProcessor,
    transform_to_world_frame,
    LidarPoint
)

# Import configurazione
from config_L5 import (
    HYSDG_LAMBDA_ESD,
    HYSDG_STATIC_VELOCITY_THRESHOLD,
    HYSDG_STATIC_DDOT_THRESHOLD,
    CLASSIFIER_MIN_FRAMES,
    CLASSIFIER_ACCURATE_THRESHOLD,
    CLASSIFIER_FAST_DYNAMIC_AVG_THRESHOLD,
    CLASSIFIER_FAST_DYNAMIC_MAX_THRESHOLD,
    CLASSIFIER_FAST_STATIC_AVG_THRESHOLD,
    CLASSIFIER_FAST_STATIC_MAX_THRESHOLD,
    CLASSIFIER_STATIC_VEL_THRESHOLD,
    CLASSIFIER_DYNAMIC_VEL_THRESHOLD,
    CLASSIFIER_STATIC_MAX_VEL,
    CLASSIFIER_STATIC_STD_VEL,
    CLASSIFIER_DYNAMIC_MAX_VEL,
    CLASSIFIER_DYNAMIC_STD_VEL,
    CLASSIFIER_VOTE_MARGIN,
    CLASSIFIER_HISTORY_WINDOW,
    CLASSIFIER_MAX_HISTORY_LENGTH,
    NAV_SAFETY_DISTANCE,
    NAV_CRITICAL_DISTANCE,
    NAV_MAX_SPEED,
    NAV_MIN_SPEED,
    NAV_MAX_HEADING_CHANGE,
    NAV_MIN_TURN_THRESHOLD,
    SAFETY_SCORE_CRITICAL,
    SAFETY_SCORE_WARNING_BASE,
    SAFETY_SCORE_WARNING_FACTOR,
    SAFETY_SCORE_SAFE_BASE,
    SAFETY_SCORE_SAFE_FACTOR,
    SAFETY_SCORE_SAFE_DISTANCE_FACTOR,
    EVASION_MIN_REPULSION,
    EVASION_MIN_DISTANCE,
    CONFIDENCE_INITIAL,
    CONFIDENCE_INCREMENT,
    CONFIDENCE_MAX,
    CONFIDENCE_CRITICAL_THRESHOLD
)

# =============================================================================
# Enumerations and Data Structures
# =============================================================================

class ObstacleState(Enum):
    """Obstacle state (static, dynamic, unknown)."""
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    UNKNOWN = "UNKNOWN"


class NavigationAction(Enum):
    """Available navigation actions."""
    CONTINUE = "CONTINUE"           # Continue straight
    SLOW_DOWN = "SLOW_DOWN"         # Slow down
    TURN_LEFT = "TURN_LEFT"         # Turn left
    TURN_RIGHT = "TURN_RIGHT"       # Turn right
    STOP = "STOP"                   # Stop
    REVERSE = "REVERSE"             # Reverse


@dataclass
class TrackedObstacle:
    """Tracked obstacle with all information for decision-making."""
    id: int
    center: np.ndarray
    velocity: np.ndarray
    points: List[LidarPoint]
    state: ObstacleState
    d_eq: float                     # HySDG equivalent distance
    d_dot: float                    # Equivalent distance derivative
    confidence: float
    last_seen: int
    state_history: List[ObstacleState] = field(default_factory=list)
    velocity_history: List[float] = field(default_factory=list)
    

@dataclass
class NavigationDecision:
    """Navigation decision with all details."""
    action: NavigationAction
    target_speed: float             # Target speed (m/s)
    target_heading_change: float    # Heading change (radians)
    reason: str                     # Decision reason
    critical_obstacles: List[int]   # Critical obstacle IDs
    safety_score: float             # Safety score (0-1)


# =============================================================================
# HySDG-ESD Calculator
# =============================================================================

class HySDGCalculator:
    """
    Calculates HySDG-ESD equivalent distance.
    Takes into account relative velocity between AGV and obstacle.
    """
    
    def __init__(self, lambda_esd: float = HYSDG_LAMBDA_ESD):
        """
        Args:
            lambda_esd: Scaling parameter for velocity component
        """
        self.lambda_esd = lambda_esd
    
    def compute(self, obs_pos: np.ndarray, obs_vel: np.ndarray,
                agv_pos: np.ndarray, agv_vel: np.ndarray,
                prev_d_eq: Optional[float] = None,
                dt: float = 0.1) -> dict:
        """
        Calculates equivalent distance and derivative.
        
        Args:
            obs_pos: Obstacle position in world frame
            obs_vel: Obstacle velocity
            agv_pos: AGV position
            agv_vel: AGV velocity
            prev_d_eq: Previous equivalent distance (for derivative calculation)
            dt: Delta time
            
        Returns:
            Dictionary with d_eq, d_dot and preliminary state
        """
        # Relative position vector
        r_t = obs_pos - agv_pos
        # Relative velocity vector
        u_t = obs_vel - agv_vel
        
        # Euclidean distance
        d = np.linalg.norm(r_t)
        u_norm = np.linalg.norm(u_t)

        # Equivalent distance
        if u_norm < 1e-6:
            d_eq = d
        else:
            d_eq = d - self.lambda_esd * np.dot(r_t, u_t) / u_norm

        # Equivalent distance derivative
        if prev_d_eq is None:
            d_dot = 0.0
        else:
            d_dot = (d_eq - prev_d_eq) / dt

        # Preliminary classification based on velocity
        if u_norm < HYSDG_STATIC_VELOCITY_THRESHOLD and abs(d_dot) < HYSDG_STATIC_DDOT_THRESHOLD:
            state = ObstacleState.STATIC
        else:
            state = ObstacleState.DYNAMIC

        return {
            'd_eq': d_eq, 
            'd_dot': d_dot, 
            'state': state,
            'distance': d,
            'relative_velocity': u_norm
        }


# =============================================================================
# Obstacle Classifier
# =============================================================================

class ObstacleClassifier:
    """
    Classifies obstacles as STATIC, DYNAMIC or UNKNOWN.
    Uses velocity and state history for robust decisions.
    """
    
    def __init__(self, 
                 min_frames_for_decision: int = CLASSIFIER_MIN_FRAMES,
                 static_vel_threshold: float = CLASSIFIER_STATIC_VEL_THRESHOLD,
                 dynamic_vel_threshold: float = CLASSIFIER_DYNAMIC_VEL_THRESHOLD):
        """
        Args:
            min_frames_for_decision: Minimum frames for classification
            static_vel_threshold: Velocity threshold for STATIC
            dynamic_vel_threshold: Velocity threshold for DYNAMIC
        """
        self.min_frames = min_frames_for_decision
        self.static_threshold = static_vel_threshold
        self.dynamic_threshold = dynamic_vel_threshold
    
    def classify(self, velocity_history: List[float], 
                 state_history: List[ObstacleState]) -> ObstacleState:
        """
        Classifies an obstacle based on its history.
        
        Args:
            velocity_history: Velocity history
            state_history: State history
            
        Returns:
            Classified state of the obstacle
        """
        # Insufficient frames for decision
        if len(velocity_history) < self.min_frames:
            return ObstacleState.UNKNOWN
        
        # Fast decision (4-7 frames)
        if len(velocity_history) < CLASSIFIER_ACCURATE_THRESHOLD:
            recent_velocities = velocity_history[-self.min_frames:]
            avg_velocity = np.mean(recent_velocities)
            max_velocity = np.max(recent_velocities)
            
            if avg_velocity > CLASSIFIER_FAST_DYNAMIC_AVG_THRESHOLD or max_velocity > CLASSIFIER_FAST_DYNAMIC_MAX_THRESHOLD:
                return ObstacleState.DYNAMIC
            elif avg_velocity < CLASSIFIER_FAST_STATIC_AVG_THRESHOLD and max_velocity < CLASSIFIER_FAST_STATIC_MAX_THRESHOLD:
                return ObstacleState.STATIC
            else:
                return ObstacleState.UNKNOWN
        
        # Accurate decision (8+ frames)
        recent_velocities = velocity_history[-CLASSIFIER_HISTORY_WINDOW:]
        avg_velocity = np.mean(recent_velocities)
        std_velocity = np.std(recent_velocities)
        max_velocity = np.max(recent_velocities)
        
        # Recent state count
        if len(state_history) >= CLASSIFIER_ACCURATE_THRESHOLD:
            recent_history = state_history[-CLASSIFIER_HISTORY_WINDOW:]
            static_count = sum(1 for s in recent_history if s == ObstacleState.STATIC)
            dynamic_count = sum(1 for s in recent_history if s == ObstacleState.DYNAMIC)
        else:
            static_count = 0
            dynamic_count = 0
        
        # Level 1: Definitely STATIC
        if (avg_velocity < CLASSIFIER_FAST_STATIC_AVG_THRESHOLD and 
            max_velocity < CLASSIFIER_STATIC_MAX_VEL and 
            std_velocity < CLASSIFIER_STATIC_STD_VEL):
            return ObstacleState.STATIC
        
        # Level 2: Definitely DYNAMIC
        elif (avg_velocity > self.dynamic_threshold or 
              max_velocity > CLASSIFIER_DYNAMIC_MAX_VEL or
              (avg_velocity > 0.30 and std_velocity > CLASSIFIER_DYNAMIC_STD_VEL)):
            return ObstacleState.DYNAMIC
        
        # Level 3: Vote based on history
        elif static_count > dynamic_count + CLASSIFIER_VOTE_MARGIN:
            return ObstacleState.STATIC
        elif dynamic_count > static_count + CLASSIFIER_VOTE_MARGIN:
            return ObstacleState.DYNAMIC
        
        # Level 4: Based on average
        elif avg_velocity < self.static_threshold:
            return ObstacleState.STATIC
        else:
            return ObstacleState.DYNAMIC


# Import config L4 per tracker
from config_L4 import (
    TRACKER_MAX_ASSOCIATION_DISTANCE,
    TRACKER_MAX_AGE
)

# =============================================================================
# Multi-Object Tracker with Decision Support
# =============================================================================

class DecisionTracker:
    """
    Multi-object tracker integrated with decision support.
    Manages classification and HySDG calculation for each obstacle.
    """
    
    def __init__(self, dt: float, max_distance: float = TRACKER_MAX_ASSOCIATION_DISTANCE, 
                 max_age: int = TRACKER_MAX_AGE):
        """
        Args:
            dt: Delta time
            max_distance: Maximum distance for association
            max_age: Maximum track age without updates
        """
        self.dt = dt
        self.max_distance = max_distance
        self.max_age = max_age
        
        self.obstacles: List[TrackedObstacle] = []
        self.kalman_filters: dict = {}
        self.next_id = 0
        self.current_time = 0
        
        self.hysdg_calculator = HySDGCalculator()
        self.classifier = ObstacleClassifier()
        
        # LiDAR processor (for clustering)
        self.lidar_processor = LidarProcessor()

    def update(self, clusters: List[List[LidarPoint]], 
               agv_pos: np.ndarray, agv_vel: np.ndarray, 
               agv_heading: float):
        """
        Updates tracker with new clusters.
        
        Args:
            clusters: List of LiDAR point clusters
            agv_pos: AGV position
            agv_vel: AGV velocity
            agv_heading: AGV heading
        """
        self.current_time += 1

        # Prediction for all filters
        for kf in self.kalman_filters.values():
            kf.predict()

        if not clusters:
            self._remove_old_obstacles()
            return

        # Calculate cluster centers in world frame
        cluster_centers_world = []
        for cluster in clusters:
            x_mean = np.mean([p.x for p in cluster])
            y_mean = np.mean([p.y for p in cluster])
            center_agv_frame = np.array([x_mean, y_mean])
            center_world = transform_to_world_frame(center_agv_frame, agv_pos, agv_heading)
            cluster_centers_world.append(center_world)

        # Association with Hungarian algorithm
        if self.obstacles and cluster_centers_world:
            from scipy.optimize import linear_sum_assignment
            
            cost_matrix = np.zeros((len(self.obstacles), len(cluster_centers_world)))
            
            for i, obs in enumerate(self.obstacles):
                for j, center_world in enumerate(cluster_centers_world):
                    dist = np.linalg.norm(obs.center - center_world)
                    cost_matrix[i, j] = dist if dist < self.max_distance else 1e6
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_obs = set()
            matched_clusters = set()
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.max_distance:
                    self._update_obstacle(i, cluster_centers_world[j], clusters[j], 
                                         agv_pos, agv_vel)
                    matched_obs.add(i)
                    matched_clusters.add(j)
        else:
            matched_obs = set()
            matched_clusters = set()

        # Create new obstacles for unassociated clusters
        for j, cluster in enumerate(clusters):
            if j not in matched_clusters:
                self._create_obstacle(cluster_centers_world[j], cluster, agv_pos, agv_vel)

        self._remove_old_obstacles()

    def _update_obstacle(self, idx: int, center_world: np.ndarray, 
                         cluster: List[LidarPoint], 
                         agv_pos: np.ndarray, agv_vel: np.ndarray):
        """Updates an existing obstacle."""
        obs = self.obstacles[idx]

        if obs.id not in self.kalman_filters:
            self.kalman_filters[obs.id] = ExtendedKalmanFilterCV(self.dt)

        self.kalman_filters[obs.id].update(center_world)
        pos_world, vel_world = self.kalman_filters[obs.id].get_state()

        # Calculate HySDG-ESD
        result = self.hysdg_calculator.compute(
            pos_world, vel_world, agv_pos, agv_vel, obs.d_eq, self.dt
        )

        # Update obstacle
        obs.center = pos_world
        obs.velocity = vel_world
        obs.points = cluster
        obs.d_eq = result['d_eq']
        obs.d_dot = result['d_dot']
        
        # Update velocity history
        vel_magnitude = np.linalg.norm(vel_world)
        obs.velocity_history.append(vel_magnitude)
        if len(obs.velocity_history) > CLASSIFIER_MAX_HISTORY_LENGTH:
            obs.velocity_history.pop(0)
        
        # Update state history
        obs.state_history.append(result['state'])
        if len(obs.state_history) > CLASSIFIER_MAX_HISTORY_LENGTH:
            obs.state_history.pop(0)
        
        # Classify obstacle
        obs.state = self.classifier.classify(obs.velocity_history, obs.state_history)
        
        obs.last_seen = self.current_time
        obs.confidence = min(CONFIDENCE_MAX, obs.confidence + CONFIDENCE_INCREMENT)

    def _create_obstacle(self, center_world: np.ndarray, 
                         cluster: List[LidarPoint],
                         agv_pos: np.ndarray, agv_vel: np.ndarray):
        """Creates a new obstacle."""
        kf = ExtendedKalmanFilterCV(self.dt)
        kf.update(center_world)
        pos_world, vel_world = kf.get_state()

        self.kalman_filters[self.next_id] = kf
        
        result = self.hysdg_calculator.compute(
            pos_world, vel_world, agv_pos, agv_vel, None, self.dt
        )

        obstacle = TrackedObstacle(
            id=self.next_id,
            center=pos_world,
            velocity=vel_world,
            points=cluster,
            state=ObstacleState.UNKNOWN,
            d_eq=result['d_eq'],
            d_dot=result['d_dot'],
            confidence=CONFIDENCE_INITIAL,
            last_seen=self.current_time,
            state_history=[result['state']],
            velocity_history=[np.linalg.norm(vel_world)]
        )

        self.obstacles.append(obstacle)
        self.next_id += 1

    def _remove_old_obstacles(self):
        """Removes obstacles not seen recently."""
        self.obstacles = [obs for obs in self.obstacles 
                         if self.current_time - obs.last_seen <= self.max_age]
        active_ids = {obs.id for obs in self.obstacles}
        self.kalman_filters = {k: v for k, v in self.kalman_filters.items() 
                              if k in active_ids}

    def get_obstacles(self) -> List[TrackedObstacle]:
        """Returns all tracked obstacles."""
        return self.obstacles
    
    def get_critical_obstacles(self, safety_distance: float = NAV_SAFETY_DISTANCE) -> List[TrackedObstacle]:
        """Returns obstacles within safety distance."""
        return [obs for obs in self.obstacles 
                if obs.d_eq < safety_distance and obs.confidence > CONFIDENCE_CRITICAL_THRESHOLD]
    
    def get_dynamic_obstacles(self) -> List[TrackedObstacle]:
        """Returns only dynamic obstacles."""
        return [obs for obs in self.obstacles 
                if obs.state == ObstacleState.DYNAMIC]
    
    def get_static_obstacles(self) -> List[TrackedObstacle]:
        """Returns only static obstacles."""
        return [obs for obs in self.obstacles 
                if obs.state == ObstacleState.STATIC]


# =============================================================================
# Navigation Decision Maker
# =============================================================================

class NavigationDecisionMaker:
    """
    Makes navigation decisions based on detected obstacles.
    Determines optimal speed and direction for the AGV.
    """
    
    def __init__(self, 
                 safety_distance: float = NAV_SAFETY_DISTANCE,
                 critical_distance: float = NAV_CRITICAL_DISTANCE,
                 max_speed: float = NAV_MAX_SPEED,
                 min_speed: float = NAV_MIN_SPEED):
        """
        Args:
            safety_distance: Safety distance (m)
            critical_distance: Critical distance for stop (m)
            max_speed: Maximum speed (m/s)
            min_speed: Minimum speed (m/s)
        """
        self.safety_distance = safety_distance
        self.critical_distance = critical_distance
        self.max_speed = max_speed
        self.min_speed = min_speed
    
    def decide(self, obstacles: List[TrackedObstacle], 
               agv_pos: np.ndarray, 
               agv_heading: float) -> NavigationDecision:
        """
        Makes a navigation decision.
        
        Args:
            obstacles: List of tracked obstacles
            agv_pos: AGV position
            agv_heading: Current AGV heading
            
        Returns:
            NavigationDecision with action and parameters
        """
        if not obstacles:
            return NavigationDecision(
                action=NavigationAction.CONTINUE,
                target_speed=self.max_speed,
                target_heading_change=0.0,
                reason="No obstacle detected",
                critical_obstacles=[],
                safety_score=1.0
            )
        
        # Find critical obstacles
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
        speed_factor = (closest.d_eq - self.critical_distance) / (self.safety_distance - self.critical_distance)
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
        safety_score = self._compute_safety_score(obstacles, agv_pos)
        
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
        Calculates optimal evasion direction.
        
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
                # Repulsive force inversely proportional to distance
                weight = 1.0 / (dist * dist)
                repulsion += weight * diff / dist
        
        if np.linalg.norm(repulsion) < EVASION_MIN_REPULSION:
            return 0.0
        
        # Calculate repulsion angle
        repulsion_angle = np.arctan2(repulsion[1], repulsion[0])
        
        # Difference from current heading
        heading_change = repulsion_angle - agv_heading
        
        # Normalize to [-pi, pi]
        while heading_change > np.pi:
            heading_change -= 2 * np.pi
        while heading_change < -np.pi:
            heading_change += 2 * np.pi
        
        # Limit maximum change
        max_change = NAV_MAX_HEADING_CHANGE
        return np.clip(heading_change, -max_change, max_change)
    
    def _compute_safety_score(self, obstacles: List[TrackedObstacle], 
                              agv_pos: np.ndarray) -> float:
        """
        Calculates a safety score (0-1).
        1 = very safe, 0 = very dangerous.
        """
        if not obstacles:
            return 1.0
        
        min_distance = min(obs.d_eq for obs in obstacles)
        
        if min_distance < self.critical_distance:
            return SAFETY_SCORE_CRITICAL
        elif min_distance < self.safety_distance:
            return SAFETY_SCORE_WARNING_BASE + SAFETY_SCORE_WARNING_FACTOR * (min_distance - self.critical_distance) / (self.safety_distance - self.critical_distance)
        else:
            return SAFETY_SCORE_SAFE_BASE + SAFETY_SCORE_SAFE_FACTOR * min(1.0, (min_distance - self.safety_distance) / SAFETY_SCORE_SAFE_DISTANCE_FACTOR)


# =============================================================================
# Decision Layer - Complete System
# =============================================================================

class DecisionLayer:
    """
    Complete decision layer that integrates tracking and navigation.
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Args:
            dt: Delta time
        """
        self.dt = dt
        self.tracker = DecisionTracker(dt)
        self.navigator = NavigationDecisionMaker()
        self.lidar_processor = LidarProcessor()
    
    def process_scan(self, ranges: np.ndarray, angles: np.ndarray,
                     agv_pos: np.ndarray, agv_vel: np.ndarray,
                     agv_heading: float) -> List[TrackedObstacle]:
        """
        Processes LiDAR scan and updates tracker.
        
        Returns:
            List of tracked obstacles
        """
        points = self.lidar_processor.parse_scan(ranges, angles)
        clusters = self.lidar_processor.cluster_points(points)
        self.tracker.update(clusters, agv_pos, agv_vel, agv_heading)
        return self.tracker.get_obstacles()
    
    def get_navigation_decision(self, agv_pos: np.ndarray, 
                                 agv_heading: float) -> NavigationDecision:
        """
        Gets navigation decision based on current obstacles.
        """
        return self.navigator.decide(
            self.tracker.get_obstacles(), 
            agv_pos, 
            agv_heading
        )
    
    def get_critical_obstacles(self, safety_distance: float = NAV_SAFETY_DISTANCE) -> List[TrackedObstacle]:
        """Returns critical obstacles."""
        return self.tracker.get_critical_obstacles(safety_distance)
    
    def get_dynamic_obstacles(self) -> List[TrackedObstacle]:
        """Returns dynamic obstacles."""
        return self.tracker.get_dynamic_obstacles()
    
    def get_all_obstacles(self) -> List[TrackedObstacle]:
        """Returns all obstacles."""
        return self.tracker.get_obstacles()
    
    def export_state(self) -> List[dict]:
        """
        Exports complete state for analysis/debug.
        
        Returns:
            List of dictionaries with info on each obstacle
        """
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
            "active_kalman_filters": len(self.tracker.kalman_filters)
        }
    
    def reset(self):
        """Resets the decision layer."""
        self.tracker = DecisionTracker(self.dt)
