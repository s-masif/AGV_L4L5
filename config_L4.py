# =============================================================================
# L4 CONFIGURATION - DETECTION LAYER
# =============================================================================
# This file contains all configurable parameters for the detection layer.
# Configure how obstacles are detected and tracked here.
# =============================================================================

# =============================================================================
# LIDAR PROCESSOR CONFIGURATION
# =============================================================================
# The LiDAR Processor converts raw LiDAR data into usable points
# and groups them into clusters representing obstacles.

# Minimum acceptance angle (in degrees)
# Points with lower angles are discarded.
# -135° = also looks slightly behind to the left
LIDAR_PROC_MIN_ANGLE = -135.0

# Maximum acceptance angle (in degrees)
# Points with higher angles are discarded.
# +135° = also looks slightly behind to the right
LIDAR_PROC_MAX_ANGLE = 135.0

# Maximum distance to consider a point valid (in meters)
# Points beyond this distance are ignored.
# 
# NOTE: This is different from LIDAR_MAX_RANGE in L3!
# - L3 simulates the physical range of the sensor
# - L4 filters what is "interesting" for detection
#
# Typical values:
#   - 5m: Only very close obstacles
#   - 10m: Standard range for indoor navigation
#   - 20m: Extended range
LIDAR_PROC_MAX_RANGE = 10.0

# Minimum number of points to consider a cluster valid
# Clusters with fewer points are discarded as noise.
#
# Typical values:
#   - 3: Very sensitive, may capture noise
#   - 6: Good compromise
#   - 10+: Only large/close objects
LIDAR_PROC_MIN_POINTS = 6

# =============================================================================
# DBSCAN CLUSTERING CONFIGURATION
# =============================================================================
# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# groups LiDAR points into clusters based on density.

# Epsilon (eps): Maximum radius to consider two points "neighbors" (in meters)
# Points within this distance can belong to the same cluster.
#
# Typical values:
#   - 0.3m: Tight clusters, can separate close objects
#   - 0.4m: Standard value
#   - 0.6m: Wide clusters, may merge close objects
DBSCAN_EPS = 0.4

# Minimum number of samples to form a cluster
# A point is "core" if it has at least this many neighbors within eps.
#
# Typical values:
#   - 2: Very permissive
#   - 3: Standard
#   - 5: More robust to noise
DBSCAN_MIN_SAMPLES = 3

# =============================================================================
# EXTENDED KALMAN FILTER (EKF) CONFIGURATION
# =============================================================================
# The EKF estimates obstacle position and velocity over time,
# filtering out measurement noise.

# Process noise
# Models uncertainty in the obstacle motion model.
# 
# Higher values: Filter adapts faster to changes,
#                but is more sensitive to noise.
# Lower values: Filter is smoother but reacts slowly.
#
# Typical values:
#   - 0.01: Very predictable motion
#   - 0.05: Standard
#   - 0.20: Unpredictable motion
EKF_PROCESS_NOISE = 0.05

# Measurement noise
# Models uncertainty in LiDAR measurements.
#
# Higher values: Filter trusts measurements less.
# Lower values: Filter follows measurements more closely.
#
# Typical values:
#   - 0.1: Very precise LiDAR
#   - 0.2: Standard
#   - 0.5: Noisy LiDAR
EKF_MEASUREMENT_NOISE = 0.2

# Initial state covariance
# Initial uncertainty on estimated position/velocity.
# Higher values = more initial uncertainty.
EKF_INITIAL_COVARIANCE = 5.0

# Mahalanobis threshold for gating
# If the Mahalanobis distance between prediction and measurement exceeds
# this threshold, the measurement is discarded as an outlier.
#
# Typical values:
#   - 3.0: Tight gating (discards more outliers)
#   - 5.0: Standard
#   - 10.0: Permissive gating
EKF_MAHALANOBIS_THRESHOLD = 5.0

# =============================================================================
# MULTI-OBJECT TRACKING CONFIGURATION
# =============================================================================
# The tracker associates detected clusters with known obstacles,
# managing appearances and disappearances.

# Maximum distance to associate a cluster with an existing obstacle (in meters)
# If a cluster is farther than this from the predicted obstacle,
# it is considered a new obstacle.
#
# Typical values:
#   - 1.0m: Tight association
#   - 2.0m: Standard
#   - 3.0m: Permissive association
TRACKER_MAX_ASSOCIATION_DISTANCE = 2.0

# Maximum age before removing an unmatched obstacle (in frames)
# If an obstacle is not matched for this many frames, it is removed.
#
# Typical values:
#   - 5: Quick removal (may lose temporarily occluded obstacles)
#   - 10: Standard
#   - 20: Keeps obstacles longer (good for occlusions)
TRACKER_MAX_AGE = 10

# =============================================================================
# EKF DAMPING CONFIGURATION
# =============================================================================
# Damping reduces velocity estimate oscillations.
# Different damping levels based on filter convergence.

# Initial damping for new obstacles
# Applied during the first frames when the estimate is uncertain.
EKF_INITIAL_DAMPING = 0.75

# Intermediate damping
# Applied when the filter is partially converged.
EKF_INTERMEDIATE_DAMPING = 0.65

# Converged damping
# Applied when the filter has converged (low innovation).
EKF_CONVERGED_DAMPING = 0.55

# Innovation threshold for damping selection
# If average innovation < threshold, use converged damping.
EKF_INNOVATION_THRESHOLD = 0.1

# =============================================================================
# EKF HISTORY CONFIGURATION
# =============================================================================
# History lengths for position and innovation tracking.

# Maximum position history length (in frames)
# Used to compute velocity statistics.
EKF_POSITION_HISTORY_LENGTH = 30

# Maximum innovation history length (in frames)
# Used to determine filter convergence.
EKF_INNOVATION_HISTORY_LENGTH = 10
