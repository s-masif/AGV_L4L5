# =============================================================================
# L3 World Model - Configuration
# =============================================================================
# All configurable parameters for the world simulation layer.
# =============================================================================

# =============================================================================
# SIMULATION WORLD BOUNDARIES
# =============================================================================
# Defines the boundaries of the area where the AGV can move.
# Format: (x_min, x_max, y_min, y_max) in meters
WORLD_BOUNDS = (-2, 30, -6, 6)

# =============================================================================
# TIME PARAMETERS
# =============================================================================
# Delta time between simulation frames (seconds)
# 0.1s = 10 Hz update rate
DEFAULT_DT = 0.1

# Total number of simulation steps
# With dt=0.1, 600 steps = 60 seconds of simulation
DEFAULT_SIMULATION_STEPS = 600

# =============================================================================
# AGV (Automated Guided Vehicle) CONFIGURATION
# =============================================================================
# AGV movement speed (m/s)
DEFAULT_AGV_SPEED = 1.5

# AGV initial position [x, y] in meters
DEFAULT_AGV_START_POSITION = [-1.0, 0.0]

# Maximum AGV rotation rate (radians per frame)
AGV_MAX_TURN_RATE = 0.05

# Random direction change interval (frames): [min, max]
AGV_DIRECTION_CHANGE_INTERVAL = (30, 80)

# Maximum random heading change angle (radians) - ±60°
AGV_MAX_RANDOM_HEADING_CHANGE = 1.047

# =============================================================================
# LIDAR CONFIGURATION
# =============================================================================
# Field of View (degrees)
LIDAR_FOV = 270.0

# Number of laser rays
LIDAR_NUM_RAYS = 250

# Maximum LiDAR range (meters)
LIDAR_MAX_RANGE = 30.0

# Gaussian noise standard deviation (meters)
LIDAR_NOISE_STD = 0.03

# =============================================================================
# OBSTACLE CONFIGURATION
# =============================================================================
# Static obstacle radius range (min, max) in meters
OBSTACLE_RADIUS_RANGE = (0.4, 0.5)

# Minimum distance between obstacles (meters)
OBSTACLE_MIN_DISTANCE = 2.5

# Default radius for dynamic obstacles (meters)
DYNAMIC_OBSTACLE_RADIUS = 0.35

# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

# --- Scenario 1: Static obstacles only ---
SCENARIO_1_NUM_STATIC_OBSTACLES = 9
SCENARIO_1_X_RANGE = (5, 28)
SCENARIO_1_Y_RANGE = (-5, 5)

# --- Scenario 2: Dynamic obstacles only ---
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

# --- Scenario 3: Mixed static + dynamic ---
SCENARIO_3_NUM_STATIC_OBSTACLES = 5
SCENARIO_3_STATIC_X_RANGE = (8, 25)
SCENARIO_3_STATIC_Y_RANGE = (-5, 5)

# =============================================================================
# TRACKING AND VISUALIZATION
# =============================================================================
# Maximum AGV trajectory length to store (frames)
AGV_TRAJECTORY_MAX_LENGTH = 100

# Maximum LiDAR history length (frames)
LIDAR_HISTORY_MAX_LENGTH = 30
