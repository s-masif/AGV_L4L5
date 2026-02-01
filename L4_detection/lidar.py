# =============================================================================
# L4 Detection - LiDAR Processor and Clustering
# =============================================================================
# Processes raw LiDAR data to extract obstacle clusters using DBSCAN.
# =============================================================================

import numpy as np
from typing import List
from sklearn.cluster import DBSCAN

from .types import LidarPoint, DetectedCluster
from .transforms import transform_to_world_frame

from .config import (
    LIDAR_PROC_MIN_ANGLE,
    LIDAR_PROC_MAX_ANGLE,
    LIDAR_PROC_MAX_RANGE,
    LIDAR_PROC_MIN_POINTS,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES
)


class LidarProcessor:
    """
    Processes raw LiDAR data to extract obstacle clusters.
    
    Pipeline:
    1. Parse scan data (filter by angle and range)
    2. Convert polar to Cartesian coordinates
    3. Cluster points using DBSCAN
    """
    
    def __init__(self, 
                 min_angle: float = LIDAR_PROC_MIN_ANGLE, 
                 max_angle: float = LIDAR_PROC_MAX_ANGLE,
                 max_range: float = LIDAR_PROC_MAX_RANGE, 
                 min_points: int = LIDAR_PROC_MIN_POINTS, 
                 cluster_distance: float = DBSCAN_EPS):
        """
        Initialize the LiDAR processor.
        
        Args:
            min_angle: Minimum angle to consider (degrees)
            max_angle: Maximum angle to consider (degrees)
            max_range: Maximum distance to consider (meters)
            min_points: Minimum number of points for a valid cluster
            cluster_distance: DBSCAN epsilon parameter
        """
        self.min_angle = np.deg2rad(min_angle)
        self.max_angle = np.deg2rad(max_angle)
        self.max_range = max_range
        self.min_points = min_points
        self.cluster_distance = cluster_distance

    def parse_scan(self, ranges: np.ndarray, angles: np.ndarray) -> List[LidarPoint]:
        """
        Convert raw LiDAR data to a list of points.
        
        Args:
            ranges: Array of distances
            angles: Array of angles (radians)
            
        Returns:
            List of valid LidarPoints in AGV frame
        """
        points = []
        for angle, distance in zip(angles, ranges):
            # Filter invalid points
            if (distance < 0.1 or distance > self.max_range or
                angle < self.min_angle or angle > self.max_angle):
                continue
            
            # Convert to Cartesian (AGV frame)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            points.append(LidarPoint(angle, distance, x, y))
        
        return points

    def cluster_points(self, points: List[LidarPoint]) -> List[List[LidarPoint]]:
        """
        Group points into clusters using DBSCAN.
        
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
        Calculate the center of a cluster.
        
        Args:
            cluster: List of cluster points
            
        Returns:
            Cluster center [x, y] in AGV frame
        """
        x_mean = np.mean([p.x for p in cluster])
        y_mean = np.mean([p.y for p in cluster])
        return np.array([x_mean, y_mean])
    
    def process_scan_to_clusters(self, 
                                  ranges: np.ndarray, 
                                  angles: np.ndarray,
                                  agv_pos: np.ndarray,
                                  agv_heading: float) -> List[DetectedCluster]:
        """
        Full pipeline: parse scan, cluster, and transform to world frame.
        
        Args:
            ranges: LiDAR distance measurements
            angles: LiDAR angle measurements
            agv_pos: AGV position in world frame
            agv_heading: AGV heading in radians
            
        Returns:
            List of DetectedCluster with world frame positions
        """
        points = self.parse_scan(ranges, angles)
        clusters = self.cluster_points(points)
        
        detected = []
        for cluster in clusters:
            center_agv = self.get_cluster_center(cluster)
            center_world = transform_to_world_frame(center_agv, agv_pos, agv_heading)
            
            detected.append(DetectedCluster(
                center_agv_frame=center_agv,
                center_world_frame=center_world,
                points=cluster,
                num_points=len(cluster)
            ))
        
        return detected
