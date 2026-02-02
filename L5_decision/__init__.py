# =============================================================================
# L5 Decision Package
# =============================================================================
# Modular decision layer for AGV navigation with multiple algorithm options.
#
# Responsibilities:
# - Navigation algorithm execution (DWA, VFH, GapNav, Simple)
# - Path planning and trajectory selection
# - Obstacle avoidance strategies
# - Recovery behaviors (stuck detection, wall following)
#
# Usage:
#   # Import specific algorithm
#   from L5_decision import DWADecisionLayer
#   layer = DWADecisionLayer(dt=0.1)
#
#   # Or import algorithm maker directly
#   from L5_decision import GapNavDecisionMaker
#   nav = GapNavDecisionMaker()
#
#   # Or use default simple layer
#   from L5_decision import DecisionLayer
#
# Note: Detection/tracking/classification is handled by L4_detection package.
# =============================================================================

# Types and data structures (L5 specific)
from .types import (
    NavigationAction,
    RecoveryMode,
    NavigationDecision,
    DWANavigationDecision,
    VFHNavigationDecision,
    GapNavNavigationDecision,
    DetectedGap
)

# Re-export L4 types for convenience
from L4_detection import (
    ObstacleState,
    TrackedObstacle,
    DetectionLayer
)

# Core components
from .base import BaseDecisionMaker

# Navigation algorithms
from .algorithms import (
    SimpleDecisionMaker,
    DWADecisionMaker,
    VFHDecisionMaker,
    GapNavDecisionMaker,
    VODecisionMaker
)

# Complete decision layers
from .layer import (
    DecisionLayer,
    DWADecisionLayer,
    VFHDecisionLayer,
    GapNavDecisionLayer,
    VODecisionLayer
)

__all__ = [
    # Enums and types (from L4, re-exported)
    'ObstacleState',
    'TrackedObstacle',
    'DetectionLayer',
    
    # Enums and types (L5 specific)
    'NavigationAction',
    'RecoveryMode',
    'NavigationDecision',
    'DWANavigationDecision',
    'VFHNavigationDecision',
    'GapNavNavigationDecision',
    'DetectedGap',
    
    # Base class
    'BaseDecisionMaker',
    
    # Algorithm makers
    'SimpleDecisionMaker',
    'DWADecisionMaker',
    'VFHDecisionMaker',
    'GapNavDecisionMaker',
    'VODecisionMaker',
    
    # Complete layers
    'DecisionLayer',
    'DWADecisionLayer',
    'VFHDecisionLayer',
    'GapNavDecisionLayer',
    'VODecisionLayer',
]

__version__ = '2.0.0'
