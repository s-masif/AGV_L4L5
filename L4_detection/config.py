# =============================================================================
# L4 Detection - Configuration
# =============================================================================
# All configurable parameters for the detection and tracking layer.
# =============================================================================

# =============================================================================
# LIDAR PROCESSOR CONFIGURATION
# =============================================================================
# Minimum/Maximum acceptance angles (degrees)
LIDAR_PROC_MIN_ANGLE = -135.0
LIDAR_PROC_MAX_ANGLE = 135.0

# Maximum distance to consider a point valid (meters)
LIDAR_PROC_MAX_RANGE = 10.0

# Minimum number of points for a valid cluster
LIDAR_PROC_MIN_POINTS = 6

# =============================================================================
# DBSCAN CLUSTERING CONFIGURATION
# =============================================================================
# Epsilon: Maximum radius to consider two points "neighbors" (meters)
DBSCAN_EPS = 0.4

# Minimum samples to form a cluster
DBSCAN_MIN_SAMPLES = 3

# =============================================================================
# EXTENDED KALMAN FILTER (EKF) CONFIGURATION
# =============================================================================
# Process noise (motion model uncertainty)
EKF_PROCESS_NOISE = 0.05

# Measurement noise (LiDAR uncertainty)
EKF_MEASUREMENT_NOISE = 0.2

# Initial state covariance
EKF_INITIAL_COVARIANCE = 5.0

# Mahalanobis threshold for gating
EKF_MAHALANOBIS_THRESHOLD = 5.0

# --- EKF Damping Configuration ---
EKF_INITIAL_DAMPING = 0.75
EKF_INTERMEDIATE_DAMPING = 0.65
EKF_CONVERGED_DAMPING = 0.55
EKF_INNOVATION_THRESHOLD = 0.1

# --- EKF History Configuration ---
EKF_POSITION_HISTORY_LENGTH = 30
EKF_INNOVATION_HISTORY_LENGTH = 10

# =============================================================================
# MULTI-OBJECT TRACKING CONFIGURATION
# =============================================================================
# Maximum association distance for cluster-to-track (meters)
TRACKER_MAX_ASSOCIATION_DISTANCE = 2.0

# Maximum age before moving to inactive pool (frames)
TRACKER_MAX_AGE = 10

# Maximum age for inactive obstacles before permanent deletion (frames)
TRACKER_INACTIVE_MAX_AGE = 500

# Maximum distance to re-identify an inactive obstacle (meters)
# Should be larger than MAX_ASSOCIATION_DISTANCE for dynamic obstacles
TRACKER_REIDENTIFICATION_DISTANCE = 3.0

# =============================================================================
# HySDG-ESD CONFIGURATION
# =============================================================================
# Lambda ESD: Scaling parameter for velocity component
# d_eq = d - λ * (r · u) / |u|
HYSDG_LAMBDA_ESD = 1.0

# Velocity threshold to consider obstacle static (m/s)
HYSDG_STATIC_VELOCITY_THRESHOLD = 0.30

# Distance variation threshold for static (m/s)
HYSDG_STATIC_DDOT_THRESHOLD = 0.25

# =============================================================================
# OBSTACLE CLASSIFIER CONFIGURATION
# =============================================================================
# Minimum frames before classification
CLASSIFIER_MIN_FRAMES = 4

# Frames to switch to "accurate" classification
CLASSIFIER_ACCURATE_THRESHOLD = 8

# --- Fast Classification (4-7 frames) ---
CLASSIFIER_FAST_DYNAMIC_AVG_THRESHOLD = 0.35
CLASSIFIER_FAST_DYNAMIC_MAX_THRESHOLD = 0.50
CLASSIFIER_FAST_STATIC_AVG_THRESHOLD = 0.15
CLASSIFIER_FAST_STATIC_MAX_THRESHOLD = 0.25

# --- Accurate Classification (8+ frames) ---
CLASSIFIER_STATIC_VEL_THRESHOLD = 0.25
CLASSIFIER_DYNAMIC_VEL_THRESHOLD = 0.45
CLASSIFIER_STATIC_MAX_VEL = 0.25
CLASSIFIER_STATIC_STD_VEL = 0.12
CLASSIFIER_DYNAMIC_MAX_VEL = 0.65
CLASSIFIER_DYNAMIC_STD_VEL = 0.15
CLASSIFIER_VOTE_MARGIN = 5
CLASSIFIER_HISTORY_WINDOW = 15
CLASSIFIER_MAX_HISTORY_LENGTH = 20

# =============================================================================
# CONFIDENCE CONFIGURATION
# =============================================================================
# Initial confidence for new obstacles
CONFIDENCE_INITIAL = 0.2

# Confidence increment per update
CONFIDENCE_INCREMENT = 0.1

# Maximum achievable confidence
CONFIDENCE_MAX = 1.0

# Minimum confidence to consider "critical"
CONFIDENCE_CRITICAL_THRESHOLD = 0.5
