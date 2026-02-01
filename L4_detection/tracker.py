# =============================================================================
# L4 Detection - Multi-Object Tracker
# =============================================================================
# Tracks multiple obstacles over time, integrating:
# - Kalman filtering for state estimation
# - Hungarian algorithm for data association
# - HySDG-ESD calculation
# - Static/Dynamic classification
# =============================================================================

import numpy as np
from typing import List, Optional

from .types import TrackedObstacle, LidarPoint, ObstacleState
from .kalman import ExtendedKalmanFilterCV
from .classifier import HySDGCalculator, ObstacleClassifier
from .lidar import LidarProcessor
from .transforms import transform_to_world_frame

from .config import (
    TRACKER_MAX_ASSOCIATION_DISTANCE,
    TRACKER_MAX_AGE,
    TRACKER_INACTIVE_MAX_AGE,
    TRACKER_REIDENTIFICATION_DISTANCE,
    CLASSIFIER_MAX_HISTORY_LENGTH,
    CONFIDENCE_INITIAL,
    CONFIDENCE_INCREMENT,
    CONFIDENCE_MAX,
    CONFIDENCE_CRITICAL_THRESHOLD
)

# Default safety distance (can be overridden when calling methods)
DEFAULT_SAFETY_DISTANCE = 2.0


class ObstacleTracker:
    """
    Multi-object tracker with classification support.
    
    This is the main L4 component that:
    1. Receives LiDAR clusters
    2. Associates them to existing tracks (Hungarian algorithm)
    3. Updates Kalman filters for each track
    4. Computes HySDG-ESD metrics
    5. Classifies obstacles as Static/Dynamic/Unknown
    
    Output: List of TrackedObstacle ready for L5 decision making.
    """
    
    def __init__(self, dt: float, 
                 max_distance: float = TRACKER_MAX_ASSOCIATION_DISTANCE, 
                 max_age: int = TRACKER_MAX_AGE):
        """
        Initialize the tracker.
        
        Args:
            dt: Delta time between updates
            max_distance: Maximum distance for detection-to-track association
            max_age: Maximum frames without update before track deletion
        """
        self.dt = dt
        self.max_distance = max_distance
        self.max_age = max_age
        
        # Active obstacles (currently visible)
        self.obstacles: List[TrackedObstacle] = []
        self.kalman_filters: dict = {}
        self.next_id = 0
        self.current_time = 0
        
        # Inactive pool: obstacles not seen recently but still remembered
        # Key: obstacle_id, Value: (TrackedObstacle, last_known_position)
        self.inactive_pool: dict = {}
        
        self.hysdg_calculator = HySDGCalculator()
        self.classifier = ObstacleClassifier()

    def update(self, clusters: List[List[LidarPoint]], 
               agv_pos: np.ndarray, agv_vel: np.ndarray, 
               agv_heading: float):
        """
        Update tracker with new cluster detections.
        
        Args:
            clusters: List of LiDAR point clusters (in AGV frame)
            agv_pos: AGV position in world frame
            agv_vel: AGV velocity
            agv_heading: AGV heading (radians)
        """
        self.current_time += 1

        # Prediction step for all filters
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

        # Data association using Hungarian algorithm
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

        # For unmatched clusters, try to re-identify from inactive pool first
        for j, cluster in enumerate(clusters):
            if j not in matched_clusters:
                reidentified = self._try_reidentify_from_pool(
                    cluster_centers_world[j], cluster, agv_pos, agv_vel
                )
                if not reidentified:
                    # Create new track only if not found in pool
                    self._create_obstacle(cluster_centers_world[j], cluster, agv_pos, agv_vel)

        self._manage_obstacle_lifecycle()

    def _update_obstacle(self, idx: int, center_world: np.ndarray, 
                         cluster: List[LidarPoint], 
                         agv_pos: np.ndarray, agv_vel: np.ndarray):
        """Update an existing tracked obstacle."""
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
        """Create a new tracked obstacle."""
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

    def _try_reidentify_from_pool(self, center_world: np.ndarray,
                                   cluster: List[LidarPoint],
                                   agv_pos: np.ndarray, 
                                   agv_vel: np.ndarray) -> bool:
        """
        Try to re-identify an inactive obstacle from the pool.
        
        Args:
            center_world: Cluster center in world frame
            cluster: LiDAR points of the cluster
            agv_pos: AGV position
            agv_vel: AGV velocity
            
        Returns:
            True if obstacle was re-identified, False otherwise
        """
        if not self.inactive_pool:
            return False
        
        # Find closest inactive obstacle
        best_id = None
        best_dist = TRACKER_REIDENTIFICATION_DISTANCE
        
        for obs_id, (obs, last_pos) in self.inactive_pool.items():
            dist = np.linalg.norm(last_pos - center_world)
            if dist < best_dist:
                best_dist = dist
                best_id = obs_id
        
        if best_id is None:
            return False
        
        # Re-activate the obstacle
        obs, _ = self.inactive_pool.pop(best_id)
        
        # Restore or create Kalman filter
        kf = ExtendedKalmanFilterCV(self.dt)
        kf.update(center_world)
        self.kalman_filters[best_id] = kf
        pos_world, vel_world = kf.get_state()
        
        # Compute HySDG-ESD
        result = self.hysdg_calculator.compute(
            pos_world, vel_world, agv_pos, agv_vel, obs.d_eq, self.dt
        )
        
        # Update obstacle with new data
        obs.center = pos_world
        obs.velocity = vel_world
        obs.points = cluster
        obs.d_eq = result['d_eq']
        obs.d_dot = result['d_dot']
        obs.last_seen = self.current_time
        # Keep confidence but add a small boost for re-identification
        obs.confidence = min(CONFIDENCE_MAX, obs.confidence + CONFIDENCE_INCREMENT * 0.5)
        
        # Keep existing history (valuable for classification continuity)
        vel_magnitude = np.linalg.norm(vel_world)
        obs.velocity_history.append(vel_magnitude)
        if len(obs.velocity_history) > CLASSIFIER_MAX_HISTORY_LENGTH:
            obs.velocity_history.pop(0)
        obs.state_history.append(result['state'])
        if len(obs.state_history) > CLASSIFIER_MAX_HISTORY_LENGTH:
            obs.state_history.pop(0)
        
        # Re-classify
        obs.state = self.classifier.classify(obs.velocity_history, obs.state_history)
        
        self.obstacles.append(obs)
        return True

    def _manage_obstacle_lifecycle(self):
        """
        Manage obstacle lifecycle: move stale tracks to inactive pool,
        remove very old inactive obstacles.
        """
        # Separate active and stale obstacles
        active = []
        stale = []
        
        for obs in self.obstacles:
            if self.current_time - obs.last_seen <= self.max_age:
                active.append(obs)
            else:
                stale.append(obs)
        
        self.obstacles = active
        
        # Move stale obstacles to inactive pool
        for obs in stale:
            self.inactive_pool[obs.id] = (obs, obs.center.copy())
            # Remove from active Kalman filters
            if obs.id in self.kalman_filters:
                del self.kalman_filters[obs.id]
        
        # Purge very old inactive obstacles
        expired_ids = [
            obs_id for obs_id, (obs, _) in self.inactive_pool.items()
            if self.current_time - obs.last_seen > TRACKER_INACTIVE_MAX_AGE
        ]
        for obs_id in expired_ids:
            del self.inactive_pool[obs_id]

    def _remove_old_obstacles(self):
        """Remove stale tracks. (Legacy method, use _manage_obstacle_lifecycle)"""
        self._manage_obstacle_lifecycle()

    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_obstacles(self) -> List[TrackedObstacle]:
        """Returns all tracked obstacles."""
        return self.obstacles
    
    def get_critical_obstacles(self, 
                               safety_distance: float = DEFAULT_SAFETY_DISTANCE) -> List[TrackedObstacle]:
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
    
    def get_obstacle_by_id(self, obstacle_id: int) -> Optional[TrackedObstacle]:
        """Get a specific obstacle by ID."""
        for obs in self.obstacles:
            if obs.id == obstacle_id:
                return obs
        return None
    
    def reset(self):
        """Reset the tracker."""
        self.obstacles = []
        self.kalman_filters = {}
        self.inactive_pool = {}
        self.next_id = 0
        self.current_time = 0


# =============================================================================
# Detection Layer - Complete L4 System
# =============================================================================

class DetectionLayer:
    """
    Complete L4 Detection Layer.
    
    Integrates all L4 components:
    - LiDAR processing
    - Clustering
    - Multi-object tracking
    - Classification
    
    Provides TrackedObstacle list for L5 decision layer.
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize detection layer.
        
        Args:
            dt: Delta time for tracking
        """
        self.dt = dt
        self.lidar_processor = LidarProcessor()
        self.tracker = ObstacleTracker(dt)
    
    def process_scan(self, ranges: np.ndarray, angles: np.ndarray,
                     agv_pos: np.ndarray, agv_vel: np.ndarray,
                     agv_heading: float) -> List[TrackedObstacle]:
        """
        Process a LiDAR scan and return tracked obstacles.
        
        Args:
            ranges: LiDAR distance measurements
            angles: LiDAR angle measurements
            agv_pos: AGV position in world frame
            agv_vel: AGV velocity
            agv_heading: AGV heading
            
        Returns:
            List of TrackedObstacle with classification
        """
        points = self.lidar_processor.parse_scan(ranges, angles)
        clusters = self.lidar_processor.cluster_points(points)
        self.tracker.update(clusters, agv_pos, agv_vel, agv_heading)
        return self.tracker.get_obstacles()
    
    def get_obstacles(self) -> List[TrackedObstacle]:
        """Get current tracked obstacles."""
        return self.tracker.get_obstacles()
    
    def get_critical_obstacles(self, safety_distance: float = DEFAULT_SAFETY_DISTANCE) -> List[TrackedObstacle]:
        """Get critical obstacles within safety distance."""
        return self.tracker.get_critical_obstacles(safety_distance)
    
    def get_dynamic_obstacles(self) -> List[TrackedObstacle]:
        """Get dynamic obstacles."""
        return self.tracker.get_dynamic_obstacles()
    
    def get_static_obstacles(self) -> List[TrackedObstacle]:
        """Get static obstacles."""
        return self.tracker.get_static_obstacles()
    
    def reset(self):
        """Reset the detection layer."""
        self.tracker.reset()
    
    def get_statistics(self) -> dict:
        """Get detection statistics."""
        obstacles = self.tracker.get_obstacles()
        return {
            "total_obstacles": len(obstacles),
            "static_count": len([o for o in obstacles if o.state == ObstacleState.STATIC]),
            "dynamic_count": len([o for o in obstacles if o.state == ObstacleState.DYNAMIC]),
            "unknown_count": len([o for o in obstacles if o.state == ObstacleState.UNKNOWN]),
            "average_confidence": np.mean([o.confidence for o in obstacles]) if obstacles else 0.0,
            "tracker_time": self.tracker.current_time,
            "active_filters": len(self.tracker.kalman_filters),
            "inactive_pool_size": len(self.tracker.inactive_pool)
        }
