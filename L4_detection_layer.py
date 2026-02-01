# =============================================================================
# L4 - DETECTION LAYER
# =============================================================================
# This layer detects obstacles in front of the AGV within its field of view:
# - LiDAR data parsing
# - DBSCAN clustering to group points
# - Extended Kalman Filter for tracking
# - Coordinate transformation from AGV frame to world frame
# =============================================================================

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

# Import configurazione
from config_L4 import (
    LIDAR_PROC_MIN_ANGLE,
    LIDAR_PROC_MAX_ANGLE,
    LIDAR_PROC_MAX_RANGE,
    LIDAR_PROC_MIN_POINTS,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    EKF_PROCESS_NOISE,
    EKF_MEASUREMENT_NOISE,
    EKF_INITIAL_COVARIANCE,
    EKF_MAHALANOBIS_THRESHOLD,
    TRACKER_MAX_ASSOCIATION_DISTANCE,
    TRACKER_MAX_AGE,
    EKF_INITIAL_DAMPING,
    EKF_INTERMEDIATE_DAMPING,
    EKF_CONVERGED_DAMPING,
    EKF_INNOVATION_THRESHOLD,
    EKF_POSITION_HISTORY_LENGTH,
    EKF_INNOVATION_HISTORY_LENGTH
)

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LidarPoint:
    """Single point detected by LiDAR."""
    angle: float      # Angle relative to AGV (radians)
    distance: float   # Distance from sensor (meters)
    x: float          # X coordinate in AGV frame
    y: float          # Y coordinate in AGV frame


@dataclass 
class DetectedCluster:
    """Cluster of points detected by LiDAR."""
    center_agv_frame: np.ndarray   # Center in AGV frame
    center_world_frame: np.ndarray  # Center in world frame
    points: List[LidarPoint]        # Points composing the cluster
    num_points: int                 # Number of points
    

# =============================================================================
# Coordinate Transform Utilities
# =============================================================================

def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Creates a 2D rotation matrix.
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        2x2 rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_to_world_frame(pos_agv_frame: np.ndarray, 
                             agv_pos: np.ndarray, 
                             agv_heading: float) -> np.ndarray:
    """
    Transforms a position from AGV frame to world frame.
    
    Args:
        pos_agv_frame: Position in AGV frame [x, y]
        agv_pos: AGV position in world frame [x, y]
        agv_heading: AGV heading in radians
        
    Returns:
        Position in world frame [x, y]
    """
    R = rotation_matrix_2d(agv_heading)
    pos_world = agv_pos + R @ pos_agv_frame
    return pos_world


def transform_to_agv_frame(pos_world_frame: np.ndarray,
                           agv_pos: np.ndarray,
                           agv_heading: float) -> np.ndarray:
    """
    Transforms a position from world frame to AGV frame.
    
    Args:
        pos_world_frame: Position in world frame [x, y]
        agv_pos: AGV position in world frame [x, y]
        agv_heading: AGV heading in radians
        
    Returns:
        Position in AGV frame [x, y]
    """
    R = rotation_matrix_2d(-agv_heading)
    pos_agv = R @ (pos_world_frame - agv_pos)
    return pos_agv


# =============================================================================
# Extended Kalman Filter for Tracking
# =============================================================================

class ExtendedKalmanFilterCV:
    """
    Extended Kalman Filter with Constant Velocity model.
    Used to track obstacle position and velocity.
    """
    
    def __init__(self, dt: float, process_noise: float = EKF_PROCESS_NOISE, 
                 measurement_noise: float = EKF_MEASUREMENT_NOISE):
        """
        Initializes the EKF.
        
        Args:
            dt: Delta time between measurements
            process_noise: Process noise
            measurement_noise: Measurement noise
        """
        self.dt = dt
        
        # State transition matrix (position + velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = process_noise
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ]) * 0.3
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # State covariance
        self.P = np.eye(4) * EKF_INITIAL_COVARIANCE
        
        # State vector [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        
        # Flags and counters
        self.initialized = False
        self.update_count = 0
        
        # History for analysis
        self.innovation_history = []
        self.position_history = []

    def predict(self):
        """Filter prediction step."""
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Filter update step with new measurement.
        
        Args:
            z: Measurement [x, y]
            
        Returns:
            Updated state
        """
        z = np.array(z).reshape(2, 1)
        
        if not self.initialized:
            self.x[0:2] = z
            self.x[2:4] = 0  # Initial velocity zero
            self.initialized = True
            self.update_count = 1
            self.position_history = [z.copy()]
            return self.x.copy()
        
        self.update_count += 1
        
        # Estimate initial velocity from first 2 frames
        if self.update_count == 2:
            delta_pos = z - self.position_history[0]
            estimated_vel = delta_pos / self.dt
            self.x[2:4] = estimated_vel
        
        # Save position for initial frames
        if len(self.position_history) < 3:
            self.position_history.append(z.copy())
        
        # Innovation calculation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        try:
            S_inv = np.linalg.inv(S)
            mahal_dist = np.sqrt(y.T @ S_inv @ y)[0, 0]
        except:
            S_inv = np.eye(2) * 0.1  # Fallback
            mahal_dist = 0
        
        # Gating: discard measurements too far away
        if mahal_dist > EKF_MAHALANOBIS_THRESHOLD:
            return self.x.copy()
        
        # Kalman Gain
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        
        # Update innovation history
        self.innovation_history.append(np.linalg.norm(y))
        if len(self.innovation_history) > EKF_INNOVATION_HISTORY_LENGTH:
            self.innovation_history.pop(0)
        
        # Adaptive velocity damping
        avg_innovation = np.mean(self.innovation_history) if self.innovation_history else 0
        
        if self.update_count <= 3:
            # Initial frames: keep estimated velocity
            damping = EKF_INITIAL_DAMPING
        elif self.update_count < 8:
            # Intermediate frames
            if avg_innovation < EKF_INNOVATION_THRESHOLD:
                damping = EKF_INTERMEDIATE_DAMPING
            else:
                damping = 1.0
        else:
            # After 8 frames: normal behavior
            if avg_innovation < EKF_INNOVATION_THRESHOLD:
                damping = EKF_CONVERGED_DAMPING
            else:
                damping = 1.0
        
        self.x[2:4] *= damping
        
        # Update covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x.copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns estimated position and velocity.
        
        Returns:
            Tuple (position, velocity)
        """
        pos = self.x[0:2].flatten()
        vel = self.x[2:4].flatten()
        return pos, vel
    
    def get_position(self) -> np.ndarray:
        """Returns only the estimated position."""
        return self.x[0:2].flatten()
    
    def get_velocity(self) -> np.ndarray:
        """Returns only the estimated velocity."""
        return self.x[2:4].flatten()
    
    def get_velocity_magnitude(self) -> float:
        """Returns the velocity magnitude."""
        return np.linalg.norm(self.x[2:4])


# =============================================================================
# LiDAR Processor
# =============================================================================

class LidarProcessor:
    """
    Processes raw LiDAR data to extract obstacle clusters.
    """
    
    def __init__(self, min_angle: float = LIDAR_PROC_MIN_ANGLE, 
                 max_angle: float = LIDAR_PROC_MAX_ANGLE,
                 max_range: float = LIDAR_PROC_MAX_RANGE, 
                 min_points: int = LIDAR_PROC_MIN_POINTS, 
                 cluster_distance: float = DBSCAN_EPS):
        """
        Initializes the LiDAR processor.
        
        Args:
            min_angle: Minimum angle to consider (degrees)
            max_angle: Maximum angle to consider (degrees)
            max_range: Maximum distance to consider (meters)
            min_points: Minimum number of points for a valid cluster
            cluster_distance: DBSCAN distance for clustering
        """
        self.min_angle = np.deg2rad(min_angle)
        self.max_angle = np.deg2rad(max_angle)
        self.max_range = max_range
        self.min_points = min_points
        self.cluster_distance = cluster_distance

    def parse_scan(self, ranges: np.ndarray, angles: np.ndarray) -> List[LidarPoint]:
        """
        Converts raw LiDAR data to a list of points.
        Filters out-of-range or angle points.
        
        Args:
            ranges: Array of distances
            angles: Array of angles
            
        Returns:
            List of valid LidarPoints
        """
        points = []
        for angle, distance in zip(angles, ranges):
            # Filter invalid points
            if (distance < 0.1 or distance > self.max_range or
                angle < self.min_angle or angle > self.max_angle):
                continue
            
            # Convert to Cartesian coordinates (AGV frame)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            points.append(LidarPoint(angle, distance, x, y))
        
        return points

    def cluster_points(self, points: List[LidarPoint]) -> List[List[LidarPoint]]:
        """
        Groups points into clusters using DBSCAN.
        
        Args:
            points: List of LiDAR points
            
        Returns:
            List of clusters (each cluster is a list of points)
        """
        if len(points) < 3:
            return []

        # Prepare data for DBSCAN
        X = np.array([[p.x, p.y] for p in points])
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(X)

        labels = db.labels_
        clusters = []

        # Group points by label
        for label in set(labels):
            if label == -1:  # Ignore noise
                continue
            cluster = [points[i] for i in range(len(points)) if labels[i] == label]
            if len(cluster) >= 3:
                clusters.append(cluster)

        return clusters
    
    def get_cluster_center(self, cluster: List[LidarPoint]) -> np.ndarray:
        """
        Calculates the center of a cluster.
        
        Args:
            cluster: List of cluster points
            
        Returns:
            Cluster center [x, y]
        """
        x_mean = np.mean([p.x for p in cluster])
        y_mean = np.mean([p.y for p in cluster])
        return np.array([x_mean, y_mean])


# =============================================================================
# Detection System - Complete detection system
# =============================================================================

class DetectionSystem:
    """
    Complete obstacle detection system.
    Combines LiDAR processing, clustering and tracking.
    """
    
    def __init__(self, dt: float = 0.1, max_range: float = LIDAR_PROC_MAX_RANGE):
        """
        Initializes the detection system.
        
        Args:
            dt: Delta time
            max_range: Maximum detection range
        """
        self.dt = dt
        self.lidar_processor = LidarProcessor(max_range=max_range)
        self.kalman_filters: dict = {}  # id -> EKF
        self.next_id = 0
        self.max_association_distance = TRACKER_MAX_ASSOCIATION_DISTANCE
        self.tracked_obstacles: dict = {}  # id -> info
        self.current_time = 0
        
    def process_scan(self, ranges: np.ndarray, angles: np.ndarray,
                     agv_pos: np.ndarray, agv_heading: float) -> List[DetectedCluster]:
        """
        Processes a LiDAR scan and detects obstacles.
        
        Args:
            ranges: Distances measured by LiDAR
            angles: Measurement angles
            agv_pos: AGV position in world frame
            agv_heading: AGV heading
            
        Returns:
            List of detected clusters with positions in world frame
        """
        self.current_time += 1
        
        # Parse scan and clustering
        points = self.lidar_processor.parse_scan(ranges, angles)
        clusters = self.lidar_processor.cluster_points(points)
        
        if not clusters:
            return []
        
        # Transform clusters to world frame
        detected_clusters = []
        for cluster in clusters:
            center_agv = self.lidar_processor.get_cluster_center(cluster)
            center_world = transform_to_world_frame(center_agv, agv_pos, agv_heading)
            
            detected_clusters.append(DetectedCluster(
                center_agv_frame=center_agv,
                center_world_frame=center_world,
                points=cluster,
                num_points=len(cluster)
            ))
        
        return detected_clusters
    
    def track_clusters(self, detected_clusters: List[DetectedCluster]) -> dict:
        """
        Tracks clusters over time using EKF.
        
        Args:
            detected_clusters: Clusters detected in current frame
            
        Returns:
            Dictionary with tracking info for each cluster
        """
        # Prediction for all existing filters
        for kf in self.kalman_filters.values():
            kf.predict()
        
        if not detected_clusters:
            return self.tracked_obstacles
        
        # Associate clusters to existing trackers
        cluster_centers = [c.center_world_frame for c in detected_clusters]
        
        if self.tracked_obstacles:
            # Create cost matrix
            tracked_ids = list(self.tracked_obstacles.keys())
            cost_matrix = np.zeros((len(tracked_ids), len(cluster_centers)))
            
            for i, obs_id in enumerate(tracked_ids):
                obs_pos = self.tracked_obstacles[obs_id]['position']
                for j, center in enumerate(cluster_centers):
                    dist = np.linalg.norm(obs_pos - center)
                    cost_matrix[i, j] = dist if dist < self.max_association_distance else 1e6
            
            # Hungarian algorithm for optimal association
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_clusters = set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.max_association_distance:
                    obs_id = tracked_ids[i]
                    self._update_track(obs_id, detected_clusters[j])
                    matched_clusters.add(j)
        else:
            matched_clusters = set()
        
        # Create new tracks for unassociated clusters
        for j, cluster in enumerate(detected_clusters):
            if j not in matched_clusters:
                self._create_track(cluster)
        
        # Remove old tracks
        self._remove_old_tracks()
        
        return self.tracked_obstacles
    
    def _create_track(self, cluster: DetectedCluster):
        """Creates a new track for a cluster."""
        obs_id = self.next_id
        self.next_id += 1
        
        # Initialize EKF
        kf = ExtendedKalmanFilterCV(self.dt)
        kf.update(cluster.center_world_frame)
        self.kalman_filters[obs_id] = kf
        
        pos, vel = kf.get_state()
        
        self.tracked_obstacles[obs_id] = {
            'id': obs_id,
            'position': pos,
            'velocity': vel,
            'velocity_magnitude': np.linalg.norm(vel),
            'cluster': cluster,
            'last_seen': self.current_time,
            'age': 1
        }
    
    def _update_track(self, obs_id: int, cluster: DetectedCluster):
        """Updates an existing track."""
        kf = self.kalman_filters[obs_id]
        kf.update(cluster.center_world_frame)
        pos, vel = kf.get_state()
        
        self.tracked_obstacles[obs_id].update({
            'position': pos,
            'velocity': vel,
            'velocity_magnitude': np.linalg.norm(vel),
            'cluster': cluster,
            'last_seen': self.current_time,
            'age': self.tracked_obstacles[obs_id]['age'] + 1
        })
    
    def _remove_old_tracks(self, max_age: int = TRACKER_MAX_AGE):
        """Removes tracks not recently updated."""
        to_remove = []
        for obs_id, info in self.tracked_obstacles.items():
            if self.current_time - info['last_seen'] > max_age:
                to_remove.append(obs_id)
        
        for obs_id in to_remove:
            del self.tracked_obstacles[obs_id]
            if obs_id in self.kalman_filters:
                del self.kalman_filters[obs_id]
    
    def get_tracked_obstacles(self) -> List[dict]:
        """Returns list of tracked obstacles."""
        return list(self.tracked_obstacles.values())
    
    def get_obstacle_by_id(self, obs_id: int) -> Optional[dict]:
        """Returns info about a specific obstacle."""
        return self.tracked_obstacles.get(obs_id)
    
    def get_closest_obstacle(self, reference_pos: np.ndarray) -> Optional[dict]:
        """Returns the closest obstacle to a position."""
        if not self.tracked_obstacles:
            return None
        
        min_dist = float('inf')
        closest = None
        
        for obs in self.tracked_obstacles.values():
            dist = np.linalg.norm(obs['position'] - reference_pos)
            if dist < min_dist:
                min_dist = dist
                closest = obs
        
        return closest
    
    def get_obstacles_in_range(self, reference_pos: np.ndarray, 
                               max_distance: float) -> List[dict]:
        """Returns obstacles within a certain distance."""
        return [
            obs for obs in self.tracked_obstacles.values()
            if np.linalg.norm(obs['position'] - reference_pos) <= max_distance
        ]
    
    def reset(self):
        """Resets the detection system."""
        self.kalman_filters.clear()
        self.tracked_obstacles.clear()
        self.next_id = 0
        self.current_time = 0
