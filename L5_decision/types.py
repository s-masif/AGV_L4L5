# =============================================================================
# L5 Decision - Types and Data Structures
# =============================================================================
# Navigation-specific enumerations and dataclasses used for decision making.
# Note: Detection types (ObstacleState, TrackedObstacle, LidarPoint) are in L4_detection.
# =============================================================================

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

# Import detection types from L4
from L4_detection import ObstacleState, TrackedObstacle


# =============================================================================
# Navigation Enumerations
# =============================================================================

class NavigationAction(Enum):
    """Available navigation actions."""
    CONTINUE = "CONTINUE"           # Continue straight
    SLOW_DOWN = "SLOW_DOWN"         # Slow down
    TURN_LEFT = "TURN_LEFT"         # Turn left
    TURN_RIGHT = "TURN_RIGHT"       # Turn right
    STOP = "STOP"                   # Stop
    REVERSE = "REVERSE"             # Reverse


class RecoveryMode(Enum):
    """Recovery mode states for hierarchical fallback."""
    NORMAL = 0
    WALL_FOLLOW = 1
    REVERSE_ESCAPE = 2
    RANDOM_ESCAPE = 3


# =============================================================================
# Navigation Decision Structures
# =============================================================================

@dataclass
class NavigationDecision:
    """Base navigation decision with core details."""
    action: NavigationAction
    target_speed: float             # Target speed (m/s)
    target_heading_change: float    # Heading change (radians)
    reason: str                     # Decision reason
    critical_obstacles: List[int]   # Critical obstacle IDs
    safety_score: float             # Safety score (0-1)


@dataclass
class DWANavigationDecision(NavigationDecision):
    """DWA-specific navigation decision with trajectory info."""
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    predicted_trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    in_recovery: bool = False


@dataclass
class VFHNavigationDecision(NavigationDecision):
    """VFH-specific navigation decision with histogram data."""
    histogram: Optional[np.ndarray] = None
    best_sector: int = 0
    in_recovery: bool = False


@dataclass
class DetectedGap:
    """Represents a detected navigable gap."""
    center_angle: float         # Center angle of the gap (radians)
    angular_width: float        # Angular width (radians)
    linear_width: float         # Estimated linear width (meters)
    depth: float                # Average depth (distance to obstacles)
    min_depth: float            # Minimum depth in the gap
    score: float = 0.0          # Computed score for selection


@dataclass
class GapNavNavigationDecision(NavigationDecision):
    """GapNav-specific navigation decision with rich info."""
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    predicted_trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    detected_gaps: List[DetectedGap] = field(default_factory=list)
    selected_gap: Optional[DetectedGap] = None
    apf_force: Optional[np.ndarray] = None
    recovery_mode: RecoveryMode = RecoveryMode.NORMAL
    using_direct_path: bool = False
