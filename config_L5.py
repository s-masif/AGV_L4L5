# =============================================================================
# L5 CONFIGURATION - DECISION LAYER
# =============================================================================
# This file contains all configurable parameters for the decision layer.
# Configure how obstacles are classified and how the AGV decides
# direction and speed here.
# =============================================================================

# =============================================================================
# HySDG-ESD CONFIGURATION
# =============================================================================
# HySDG-ESD (Hybrid State-Dependent Gain - Equivalent Safety Distance)
# Computes an "equivalent distance" that accounts for relative velocity
# between AGV and obstacle.

# Lambda ESD: Scaling parameter for the velocity component
# Controls how much relative velocity influences the equivalent distance.
#
# d_eq = d - λ * (r · u) / |u|
#
# where:
#   d = Euclidean distance
#   r = relative position vector
#   u = relative velocity vector
#   λ = lambda_esd
#
# Typical values:
#   - 0.5: Low velocity influence
#   - 1.0: Standard influence
#   - 2.0: High velocity influence
HYSDG_LAMBDA_ESD = 1.0

# Relative velocity threshold to consider the obstacle static (m/s)
# If |u| < threshold AND |d_dot| < threshold, preliminary state is STATIC.
HYSDG_STATIC_VELOCITY_THRESHOLD = 0.30

# Distance variation threshold to consider the obstacle static (m/s)
# |d_dot| must be below this threshold along with velocity.
HYSDG_STATIC_DDOT_THRESHOLD = 0.25

# =============================================================================
# OBSTACLE CLASSIFIER CONFIGURATION
# =============================================================================
# The classifier decides if an obstacle is STATIC, DYNAMIC, or UNKNOWN
# based on velocity and state history.

# Minimum number of frames before classifying an obstacle
# Before this, the obstacle remains UNKNOWN.
#
# Typical values:
#   - 3: Fast decision but less accurate
#   - 4: Good compromise
#   - 8: Slow decision but more accurate
CLASSIFIER_MIN_FRAMES = 4

# Number of frames to switch to "accurate" classification
# Before this, "fast" classification is used.
CLASSIFIER_ACCURATE_THRESHOLD = 8

# -----------------------------------------------------------------------------
# Fast Classification Thresholds (4-7 frames)
# -----------------------------------------------------------------------------

# If average velocity > threshold OR max velocity > threshold → DYNAMIC
CLASSIFIER_FAST_DYNAMIC_AVG_THRESHOLD = 0.35
CLASSIFIER_FAST_DYNAMIC_MAX_THRESHOLD = 0.50

# If average velocity < threshold AND max velocity < threshold → STATIC
CLASSIFIER_FAST_STATIC_AVG_THRESHOLD = 0.15
CLASSIFIER_FAST_STATIC_MAX_THRESHOLD = 0.25

# -----------------------------------------------------------------------------
# Accurate Classification Thresholds (8+ frames)
# -----------------------------------------------------------------------------

# Average velocity threshold for STATIC (m/s)
# If avg_vel < threshold → probably STATIC
CLASSIFIER_STATIC_VEL_THRESHOLD = 0.25

# Average velocity threshold for DYNAMIC (m/s)
# If avg_vel > threshold → probably DYNAMIC
CLASSIFIER_DYNAMIC_VEL_THRESHOLD = 0.45

# Maximum velocity threshold for STATIC confirmation (m/s)
CLASSIFIER_STATIC_MAX_VEL = 0.25

# Standard deviation threshold for STATIC confirmation (m/s)
CLASSIFIER_STATIC_STD_VEL = 0.12

# Maximum velocity threshold for DYNAMIC confirmation (m/s)
CLASSIFIER_DYNAMIC_MAX_VEL = 0.65

# Standard deviation threshold for DYNAMIC with average velocity (m/s)
CLASSIFIER_DYNAMIC_STD_VEL = 0.15

# Minimum vote difference to decide based on history
# If static_count > dynamic_count + margin → STATIC (and vice versa)
CLASSIFIER_VOTE_MARGIN = 5

# History window length for analysis (in frames)
CLASSIFIER_HISTORY_WINDOW = 15

# Maximum velocity and state history length to maintain
CLASSIFIER_MAX_HISTORY_LENGTH = 20

# =============================================================================
# NAVIGATION CONFIGURATION
# =============================================================================
# The NavigationDecisionMaker decides how the AGV should move
# based on detected obstacles.

# Safety distance (in meters)
# If an obstacle is closer than this, the AGV starts evasive maneuvers.
#
# Typical values:
#   - 1.5m: Aggressive navigation
#   - 2.0m: Standard
#   - 3.0m: Conservative navigation
NAV_SAFETY_DISTANCE = 2.0

# Critical distance (in meters)
# If an obstacle is closer than this, the AGV STOPS.
#
# MUST be < NAV_SAFETY_DISTANCE
#
# Typical values:
#   - 0.5m: Very close to obstacle
#   - 1.0m: Standard
#   - 1.5m: Conservative margin
NAV_CRITICAL_DISTANCE = 1.0

# Maximum AGV speed during navigation (m/s)
NAV_MAX_SPEED = 1.5

# Minimum AGV speed during navigation (m/s)
# The AGV never goes below this speed (except STOP).
NAV_MIN_SPEED = 0.3

# Maximum heading change in a single decision (in radians)
# Limits how much the AGV can turn in a single step.
# π/4 = 45 degrees
NAV_MAX_HEADING_CHANGE = 0.785  # π/4 ≈ 45°

# Minimum heading change threshold to consider a turn
# Below this threshold, the action is SLOW_DOWN instead of TURN
NAV_MIN_TURN_THRESHOLD = 0.1

# =============================================================================
# SAFETY SCORE CONFIGURATION
# =============================================================================
# The safety score is a 0-1 value indicating how safe the situation is.
# 1.0 = very safe, 0.0 = very dangerous

# Score when an obstacle is at critical distance
SAFETY_SCORE_CRITICAL = 0.1

# Base score in the warning zone (between critical and safety distance)
SAFETY_SCORE_WARNING_BASE = 0.3

# Interpolation factor in the warning zone
# Score = WARNING_BASE + WARNING_FACTOR * (normalized distance)
SAFETY_SCORE_WARNING_FACTOR = 0.4

# Base score when obstacles are beyond safety distance
SAFETY_SCORE_SAFE_BASE = 0.7

# Interpolation factor in the safe zone
SAFETY_SCORE_SAFE_FACTOR = 0.3

# Reference distance for score calculation beyond safety distance
SAFETY_SCORE_SAFE_DISTANCE_FACTOR = 5.0

# Minimum score in the safety zone (deprecated, use WARNING_BASE)
SAFETY_SCORE_SAFETY_MIN = 0.3

# Maximum score in the safety zone (deprecated, use SAFE_BASE)
SAFETY_SCORE_SAFETY_MAX = 0.7

# Reference distance for score calculation beyond safety distance (alias)
SAFETY_SCORE_FAR_DISTANCE = 5.0

# =============================================================================
# EVASION CONFIGURATION
# =============================================================================
# Parameters for evasion direction calculation.

# Minimum repulsion force threshold to change direction
# If total repulsion is below this threshold, no evasion occurs.
EVASION_MIN_REPULSION = 0.01

# Minimum distance for repulsion calculation (avoids division by zero)
EVASION_MIN_DISTANCE = 0.1

# =============================================================================
# CONFIDENCE CONFIGURATION
# =============================================================================
# Confidence indicates how certain the system is about the classification.

# Initial confidence for new obstacles
CONFIDENCE_INITIAL = 0.2

# Confidence increment for each update
CONFIDENCE_INCREMENT = 0.1

# Maximum achievable confidence
CONFIDENCE_MAX = 1.0

# Minimum confidence to consider an obstacle "critical"
CONFIDENCE_CRITICAL_THRESHOLD = 0.5

# =============================================================================
# NAVIGATION ACTIONS
# =============================================================================
# Description of available actions:
#
# CONTINUE:   Continue straight at maximum speed
# SLOW_DOWN:  Slow down while maintaining direction
# TURN_LEFT:  Turn left (positive heading)
# TURN_RIGHT: Turn right (negative heading)
# STOP:       Stop completely
# REVERSE:    Reverse (not implemented in base version)
