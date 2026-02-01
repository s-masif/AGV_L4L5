# =============================================================================
# L3 CONFIGURATION - WORLD MODEL LAYER
# =============================================================================
# This file contains all configurable parameters for the world model layer.
# Modify these values to customize the simulation environment.
# =============================================================================

# =============================================================================
# SIMULATION WORLD BOUNDARIES
# =============================================================================
# Defines the boundaries of the area where the AGV can move.
# Format: (x_min, x_max, y_min, y_max) in meters
#
# x_min: Left boundary of the world (typical AGV starting point)
# x_max: Right boundary of the world (typical AGV target)
# y_min: Lower boundary of the world
# y_max: Upper boundary of the world
#
# Example: (-2, 30, -6, 6) creates a 32m x 12m area
WORLD_BOUNDS = (-2, 30, -6, 6)

# =============================================================================
# TIME PARAMETERS
# =============================================================================
# Delta time (dt): Time interval between simulation frames.
# Unit: seconds
# 
# Typical values:
#   - 0.1s (10 Hz): Good balance between precision and performance
#   - 0.05s (20 Hz): Higher precision, more computations
#   - 0.2s (5 Hz): Less precise, but faster
#
# NOTE: Smaller values = more precise simulation but slower
DEFAULT_DT = 0.1

# Total number of simulation steps
# With dt=0.1, 600 steps = 60 seconds of simulation
DEFAULT_SIMULATION_STEPS = 600

# =============================================================================
# AGV (Automated Guided Vehicle) CONFIGURATION
# =============================================================================
# AGV movement speed in meters per second (m/s)
#
# Typical values for industrial AGVs:
#   - 0.5 m/s: Slow movement, high precision
#   - 1.5 m/s: Standard speed
#   - 3.0 m/s: Fast movement (requires more braking distance)
DEFAULT_AGV_SPEED = 1.5

# AGV initial position [x, y] in meters
# Default: starts from the left side of the world
DEFAULT_AGV_START_POSITION = [-1.0, 0.0]

# Maximum AGV rotation rate (radians per frame)
# Controls how quickly the AGV can change direction
# 
# Typical values:
#   - 0.02: Very slow and smooth rotation
#   - 0.05: Standard rotation
#   - 0.10: Fast rotation
AGV_MAX_TURN_RATE = 0.05

# Random direction change interval (in frames)
# The AGV randomly changes direction every X frames
# Range: [min, max] frames
AGV_DIRECTION_CHANGE_INTERVAL = (30, 80)

# Maximum random heading change angle (in radians)
# ±60 degrees = ±π/3 radians
AGV_MAX_RANDOM_HEADING_CHANGE = 1.047  # π/3 ≈ 60°

# =============================================================================
# LIDAR CONFIGURATION
# =============================================================================
# The LiDAR simulates a rotating laser sensor that measures distances.

# Field of View in degrees
# How many degrees the LiDAR scan covers
#
# Typical values:
#   - 180°: Front semicircle
#   - 270°: Three-quarter circle (typical for AGV)
#   - 360°: Full circle
LIDAR_FOV = 270.0

# Number of laser rays in the scan
# More rays = higher angular resolution, more computations
#
# Typical values:
#   - 100: Low resolution
#   - 250: Medium resolution (good compromise)
#   - 500: High resolution
#   - 1000+: Very high resolution (expensive LiDARs)
LIDAR_NUM_RAYS = 250

# Maximum LiDAR range in meters
# Maximum distance at which the LiDAR can detect obstacles
#
# Typical values:
#   - 10m: Short-range LiDAR (indoor)
#   - 30m: Medium-range LiDAR
#   - 100m+: Long-range LiDAR (outdoor)
LIDAR_MAX_RANGE = 30.0

# Gaussian noise standard deviation (in meters)
# Simulates measurement inaccuracy of real sensors
#
# Typical values:
#   - 0.01m: Very precise LiDAR
#   - 0.03m: Standard precision
#   - 0.10m: Cheap/noisy LiDAR
LIDAR_NOISE_STD = 0.03

# =============================================================================
# OBSTACLE CONFIGURATION
# =============================================================================

# Static obstacle radius range (min, max) in meters
OBSTACLE_RADIUS_RANGE = (0.4, 0.5)

# Minimum distance between randomly generated obstacles (in meters)
# Prevents overlapping between obstacles
OBSTACLE_MIN_DISTANCE = 2.5

# Default radius for dynamic obstacles (in meters)
DYNAMIC_OBSTACLE_RADIUS = 0.35

# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

# Scenario 1: Static obstacles only
SCENARIO_1_NUM_STATIC_OBSTACLES = 9
SCENARIO_1_X_RANGE = (5, 28)  # X range for obstacle generation
SCENARIO_1_Y_RANGE = (-5, 5)  # Y range for obstacle generation

# Scenario 2: Dynamic obstacles only
# Ranges for initial positions and velocities of dynamic obstacles
SCENARIO_2_OBS1_POS_X_RANGE = (15, 25)
SCENARIO_2_OBS1_POS_Y_RANGE = (-3, 3)
SCENARIO_2_OBS1_VEL_X_RANGE = (-1.5, -0.8)
SCENARIO_2_OBS1_VEL_Y_RANGE = (-0.5, 0.5)

SCENARIO_2_OBS2_POS_X_RANGE = (10, 20)
SCENARIO_2_OBS2_POS_Y_RANGE = (-4, 4)
SCENARIO_2_OBS2_VEL_X_RANGE = (-0.5, 0.5)
SCENARIO_2_OBS2_VEL_Y_RANGE = (-1.0, -0.6)

SCENARIO_2_OBS3_POS_X_RANGE = (8, 18)
SCENARIO_2_OBS3_POS_Y_RANGE = (-3, 3)
SCENARIO_2_OBS3_VEL_X_RANGE = (0.6, 1.2)
SCENARIO_2_OBS3_VEL_Y_RANGE = (0.6, 1.2)

# Scenario 3: Mixed static + dynamic
SCENARIO_3_NUM_STATIC_OBSTACLES = 5
SCENARIO_3_STATIC_X_RANGE = (8, 25)
SCENARIO_3_STATIC_Y_RANGE = (-5, 5)

# =============================================================================
# TRACKING AND VISUALIZATION
# =============================================================================

# Maximum AGV trajectory length to store (in frames)
# Used to visualize the AGV path
AGV_TRAJECTORY_MAX_LENGTH = 100

# Maximum LiDAR history length (in frames)
LIDAR_HISTORY_MAX_LENGTH = 30
