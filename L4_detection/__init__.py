# =============================================================================
# L4 Detection Package
# =============================================================================
# Obstacle detection, tracking, and classification layer.
#
# Responsibilities:
# - LiDAR data processing and clustering
# - Multi-object tracking with Kalman filters
# - Obstacle classification (Static/Dynamic/Unknown)
# - HySDG-ESD equivalent distance calculation
#
# Usage:
#   from L4_detection import DetectionLayer
#   detector = DetectionLayer(dt=0.1)
#   obstacles = detector.process_scan(ranges, angles, pos, vel, heading)
# =============================================================================

# Types
from .types import (
    ObstacleState,
    LidarPoint,
    DetectedCluster,
    TrackedObstacle
)

# Core components
from .transforms import (
    rotation_matrix_2d,
    transform_to_world_frame,
    transform_to_agv_frame
)
from .kalman import ExtendedKalmanFilterCV
from .lidar import LidarProcessor
from .classifier import HySDGCalculator, ObstacleClassifier
from .tracker import ObstacleTracker, DetectionLayer

__all__ = [
    # Types
    'ObstacleState',
    'LidarPoint',
    'DetectedCluster',
    'TrackedObstacle',
    
    # Transforms
    'rotation_matrix_2d',
    'transform_to_world_frame',
    'transform_to_agv_frame',
    
    # Components
    'ExtendedKalmanFilterCV',
    'LidarProcessor',
    'HySDGCalculator',
    'ObstacleClassifier',
    'ObstacleTracker',
    
    # Complete layer
    'DetectionLayer',
]

__version__ = '2.0.0'
