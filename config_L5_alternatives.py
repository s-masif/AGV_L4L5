# =============================================================================
# Configuration for L5 Alternative Decision Layers
# =============================================================================
# This file contains parameters for VFH, DWA, and GapNav decision algorithms.
# These are alternative navigation strategies that can be used instead of
# the default NavigationDecisionMaker in L5_decision_layer.py
# =============================================================================

import numpy as np

# =============================================================================
# Common Robot Parameters
# =============================================================================

ROBOT_RADIUS = 0.3              # Physical radius of the robot (meters)
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
# VFH (Vector Field Histogram) Parameters
# =============================================================================

# Histogram configuration
VFH_NUM_SECTORS = 72            # Number of sectors in polar histogram
                                # 72 sectors = 5° per sector
VFH_OBSTACLE_THRESHOLD = 1.5    # Density threshold for blocked sector
                                # Higher = more permissive
VFH_SAFETY_MARGIN = 0.4         # Angular footprint safety margin (meters)

# Wall-following parameters (for recovery mode)
VFH_WALL_FOLLOW_DISTANCE = 1.0  # Desired distance from wall during following
VFH_STUCK_THRESHOLD = 20        # Steps without movement to trigger recovery
VFH_NO_PROGRESS_THRESHOLD = 50  # Steps without goal approach to trigger recovery
VFH_RECOVERY_EXIT_STEPS = 80    # Steps of progress before exiting recovery
VFH_RECOVERY_TIMEOUT = 300      # Maximum steps in recovery before declaring stuck

# Speed control
VFH_MIN_SAFE_DISTANCE = 0.8     # Minimum distance for dynamic speed adjustment

# =============================================================================
# DWA (Dynamic Window Approach) Parameters
# =============================================================================

# Velocity sampling
DWA_VELOCITY_SAMPLES = 7        # Number of linear velocity samples
DWA_ANGULAR_SAMPLES = 21        # Number of angular velocity samples

# Trajectory prediction
DWA_PREDICT_TIME = 1.5          # Prediction horizon (seconds)
DWA_PREDICT_DT = 0.1            # Prediction time step (seconds)

# Scoring weights
DWA_WEIGHT_HEADING = 0.30       # Heading towards goal
DWA_WEIGHT_CLEARANCE = 0.40     # Obstacle clearance (critical for safety)
DWA_WEIGHT_VELOCITY = 0.10      # Speed preference
DWA_WEIGHT_PROGRESS = 0.20      # Progress towards goal

# Recovery parameters
DWA_WALL_FOLLOW_DISTANCE = 1.2  # Distance to maintain from wall
DWA_MIN_CLEARANCE_THRESHOLD = 0.8  # Trigger caution below this clearance
DWA_STUCK_THRESHOLD = 15        # Steps without movement
DWA_NO_PROGRESS_THRESHOLD = 40  # Steps without progress

# =============================================================================
# GapNav (Gap-Based Navigation + APF) Parameters
# =============================================================================

# Gap detection
GAPNAV_MIN_GAP_WIDTH = 1.0      # Minimum gap width for robot passage (meters)
GAPNAV_FREE_THRESHOLD_FACTOR = 0.6  # Factor of LIDAR range for "free" threshold
GAPNAV_MAX_FREE_THRESHOLD = 3.0     # Maximum distance to consider "free"

# Artificial Potential Field (APF)
APF_ATTRACT_GAIN = 1.0          # Attractive force towards goal
APF_REPEL_GAIN = 0.8            # Repulsive force from obstacles
APF_REPEL_THRESHOLD = 2.0       # Distance within which repulsion acts (meters)

# Enhanced DWA scoring weights (with APF)
GAPNAV_WEIGHT_HEADING = 0.20    # Alignment with goal/subgoal
GAPNAV_WEIGHT_CLEARANCE = 0.30  # Distance from obstacles
GAPNAV_WEIGHT_VELOCITY = 0.10   # Speed preference
GAPNAV_WEIGHT_PROGRESS = 0.20   # Progress towards goal
GAPNAV_WEIGHT_APF = 0.20        # APF alignment score

# Velocity sampling
GAPNAV_VELOCITY_SAMPLES = 9     # Linear velocity samples
GAPNAV_ANGULAR_SAMPLES = 25     # Angular velocity samples
GAPNAV_PREDICT_TIME = 1.5       # Prediction horizon (seconds)

# Recovery parameters
GAPNAV_WALL_FOLLOW_DISTANCE = 1.2
GAPNAV_REVERSE_DISTANCE = 0.8   # Distance to reverse when stuck

# Final approach (when close to goal)
GAPNAV_FINAL_APPROACH_DISTANCE = 3.0  # Activate final approach mode
GAPNAV_FINAL_APPROACH_TOLERANCE = 0.4  # Tighter goal tolerance

# Recovery timeouts
GAPNAV_WALL_FOLLOW_TIMEOUT = 100
GAPNAV_REVERSE_TIMEOUT = 50
GAPNAV_RANDOM_TIMEOUT = 40

# =============================================================================
# Direct Path Check Parameters (used by GapNav)
# =============================================================================

DIRECT_PATH_SAFETY_MARGIN = ROBOT_EFFECTIVE_RADIUS
DIRECT_PATH_MIN_CLEARANCE = 1.5  # Minimum clearance for direct path (meters)
DIRECT_PATH_CORRIDOR_MARGIN = ROBOT_EFFECTIVE_RADIUS  # Side clearance

# =============================================================================
# Common Simulation Parameters
# =============================================================================

SIMULATION_DT = 0.1             # Simulation time step (seconds)
LIDAR_VIRTUAL_RANGE = 5.0       # Virtual LIDAR range for algorithms that need it
LIDAR_VIRTUAL_RAYS = 90         # Virtual LIDAR rays (4° resolution)
