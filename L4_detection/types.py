# =============================================================================
# L4 Detection - Types and Data Structures
# =============================================================================
# Common data structures for obstacle detection and tracking.
# =============================================================================

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# =============================================================================
# Enumerations
# =============================================================================

class ObstacleState(Enum):
    """Obstacle state classification."""
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# LiDAR Data Structures
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
    center_agv_frame: np.ndarray    # Center in AGV frame
    center_world_frame: np.ndarray  # Center in world frame
    points: List[LidarPoint]        # Points composing the cluster
    num_points: int                 # Number of points


# =============================================================================
# Tracked Obstacle
# =============================================================================

@dataclass
class TrackedObstacle:
    """
    Tracked obstacle with complete information.
    
    This is the main data structure passed from L4 to L5.
    Contains position, velocity, classification, and tracking metadata.
    """
    id: int
    center: np.ndarray                  # Position in world frame
    velocity: np.ndarray                # Estimated velocity
    points: List[LidarPoint]            # LiDAR points
    state: ObstacleState                # Classification (Static/Dynamic/Unknown)
    d_eq: float                         # HySDG equivalent distance
    d_dot: float                        # Equivalent distance derivative
    confidence: float                   # Tracking confidence
    last_seen: int                      # Last frame seen
    state_history: List[ObstacleState] = field(default_factory=list)
    velocity_history: List[float] = field(default_factory=list)
