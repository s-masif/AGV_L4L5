# =============================================================================
# L5 Decision - Configuration
# =============================================================================
# All configurable parameters for the decision/navigation layer.
# Includes parameters for all navigation algorithms (Simple, DWA, VFH, GapNav).
# =============================================================================

import numpy as np

# =============================================================================
# COMMON ROBOT PARAMETERS
# =============================================================================
ROBOT_RADIUS = 0.3              # Physical radius (meters)
SAFETY_MARGIN = 0.15            # Additional safety buffer (meters)
ROBOT_EFFECTIVE_RADIUS = ROBOT_RADIUS + SAFETY_MARGIN

# Movement limits
MAX_LINEAR_VELOCITY = 0.8       # Maximum forward speed (m/s)
MIN_LINEAR_VELOCITY = 0.0       # Minimum forward speed (m/s)
MAX_ANGULAR_VELOCITY = np.deg2rad(120)  # Maximum rotation rate (rad/s)

# Acceleration limits
MAX_LINEAR_ACCEL = 1.5          # Maximum linear acceleration (m/s²)
MAX_ANGULAR_ACCEL = np.deg2rad(300)  # Maximum angular acceleration (rad/s²)

# Goal parameters
GOAL_TOLERANCE = 0.5            # Distance to consider goal reached (meters)

# =============================================================================
# NAVIGATION CONFIGURATION (Common)
# =============================================================================
# Safety distance: AGV starts evasive maneuvers (meters)
NAV_SAFETY_DISTANCE = 2.0

# Critical distance: AGV STOPS (meters) - must be < NAV_SAFETY_DISTANCE
NAV_CRITICAL_DISTANCE = 1.0

# Maximum AGV speed during navigation (m/s)
NAV_MAX_SPEED = 1.5

# Minimum AGV speed during navigation (m/s)
NAV_MIN_SPEED = 0.3

# Maximum heading change per decision (radians) - π/4 = 45°
NAV_MAX_HEADING_CHANGE = 0.785

# Minimum turn threshold to trigger TURN action
NAV_MIN_TURN_THRESHOLD = 0.1

# =============================================================================
# SAFETY SCORE CONFIGURATION
# =============================================================================
SAFETY_SCORE_CRITICAL = 0.1
SAFETY_SCORE_WARNING_BASE = 0.3
SAFETY_SCORE_WARNING_FACTOR = 0.4
SAFETY_SCORE_SAFE_BASE = 0.7
SAFETY_SCORE_SAFE_FACTOR = 0.3
SAFETY_SCORE_SAFE_DISTANCE_FACTOR = 5.0
SAFETY_SCORE_SAFETY_MIN = 0.3
SAFETY_SCORE_SAFETY_MAX = 0.7
SAFETY_SCORE_FAR_DISTANCE = 5.0

# =============================================================================
# EVASION CONFIGURATION
# =============================================================================
EVASION_MIN_REPULSION = 0.01
EVASION_MIN_DISTANCE = 0.1

# =============================================================================
# STUCK DETECTION (Common)
# =============================================================================
STUCK_POSITION_THRESHOLD = 0.1  # Movement threshold (meters)
STUCK_COUNTER_THRESHOLD = 20    # Frames without movement

# =============================================================================
# WALL FOLLOWING (Common)
# =============================================================================
WALL_FOLLOW_DISTANCE = 1.0      # Desired distance from wall (meters)
WALL_FOLLOW_GAIN = 0.8          # Proportional gain
WALL_FOLLOW_MAX_ANGULAR_VEL = np.deg2rad(60)

# =============================================================================
# VFH (Vector Field Histogram) PARAMETERS
# =============================================================================
VFH_NUM_SECTORS = 72            # Number of sectors (72 = 5° per sector)
VFH_OBSTACLE_THRESHOLD = 1.5    # Density threshold for blocked sector
VFH_SAFETY_MARGIN = 0.4         # Angular footprint safety margin (meters)

# VFH Wall-following
VFH_WALL_FOLLOW_DISTANCE = 1.0
VFH_STUCK_THRESHOLD = 20
VFH_NO_PROGRESS_THRESHOLD = 50
VFH_RECOVERY_EXIT_STEPS = 80
VFH_RECOVERY_TIMEOUT = 300
VFH_MIN_SAFE_DISTANCE = 0.8

# =============================================================================
# DWA (Dynamic Window Approach) PARAMETERS
# =============================================================================
# Velocity sampling
DWA_VELOCITY_SAMPLES = 7        # Linear velocity samples
DWA_ANGULAR_SAMPLES = 21        # Angular velocity samples

# Trajectory prediction
DWA_PREDICT_TIME = 1.5          # Prediction horizon (seconds)
DWA_PREDICT_DT = 0.1            # Prediction time step (seconds)

# Scoring weights
DWA_WEIGHT_HEADING = 0.30       # Heading towards goal
DWA_WEIGHT_CLEARANCE = 0.40     # Obstacle clearance
DWA_WEIGHT_VELOCITY = 0.10      # Speed preference
DWA_WEIGHT_PROGRESS = 0.20      # Progress towards goal

# DWA Recovery
DWA_WALL_FOLLOW_DISTANCE = 1.2
DWA_MIN_CLEARANCE_THRESHOLD = 0.8
DWA_STUCK_THRESHOLD = 15
DWA_NO_PROGRESS_THRESHOLD = 40

# =============================================================================
# GAPNAV (Gap-Based Navigation + APF) PARAMETERS
# =============================================================================
# Gap detection
GAPNAV_MIN_GAP_WIDTH = 1.0              # Minimum gap width (meters)
GAPNAV_FREE_THRESHOLD_FACTOR = 0.6      # Factor of LIDAR range for "free"
GAPNAV_MAX_FREE_THRESHOLD = 3.0         # Maximum "free" distance

# Artificial Potential Field (APF)
APF_ATTRACT_GAIN = 1.0          # Attractive force towards goal
APF_REPEL_GAIN = 0.8            # Repulsive force from obstacles
APF_REPEL_THRESHOLD = 2.0       # Distance for repulsion (meters)

# Enhanced DWA scoring weights
GAPNAV_WEIGHT_HEADING = 0.20
GAPNAV_WEIGHT_CLEARANCE = 0.30
GAPNAV_WEIGHT_VELOCITY = 0.10
GAPNAV_WEIGHT_PROGRESS = 0.20
GAPNAV_WEIGHT_APF = 0.20

# Velocity sampling
GAPNAV_VELOCITY_SAMPLES = 9
GAPNAV_ANGULAR_SAMPLES = 25
GAPNAV_PREDICT_TIME = 1.5

# Recovery
GAPNAV_WALL_FOLLOW_DISTANCE = 1.2
GAPNAV_REVERSE_DISTANCE = 0.8

# Final approach
GAPNAV_FINAL_APPROACH_DISTANCE = 3.0
GAPNAV_FINAL_APPROACH_TOLERANCE = 0.4

# Recovery timeouts
GAPNAV_WALL_FOLLOW_TIMEOUT = 100
GAPNAV_REVERSE_TIMEOUT = 50
GAPNAV_RANDOM_TIMEOUT = 40

# =============================================================================
# DIRECT PATH CHECK (used by GapNav)
# =============================================================================
DIRECT_PATH_SAFETY_MARGIN = ROBOT_EFFECTIVE_RADIUS
DIRECT_PATH_MIN_CLEARANCE = 1.5
DIRECT_PATH_CORRIDOR_MARGIN = ROBOT_EFFECTIVE_RADIUS

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
SIMULATION_DT = 0.1             # Simulation time step (seconds)
LIDAR_VIRTUAL_RANGE = 5.0       # Virtual LIDAR range for algorithms
LIDAR_VIRTUAL_RAYS = 90         # Virtual LIDAR rays (4° resolution)
