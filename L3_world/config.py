# =============================================================================
# L3 World Model - Configuration
# =============================================================================
# Merged configuration with all necessary constants
# =============================================================================

import numpy as np

# =============================================================================
# SIMULATION WORLD BOUNDARIES
# =============================================================================
WORLD_BOUNDS = (-2, 30, -6, 6)

# =============================================================================
# TIME PARAMETERS
# =============================================================================
DEFAULT_DT = 0.1
DEFAULT_SIMULATION_STEPS = 600

# =============================================================================
# AGV CONFIGURATION
# =============================================================================
DEFAULT_AGV_SPEED = 1.5
DEFAULT_AGV_START_POSITION = [-1.0, 0.0]
AGV_MAX_TURN_RATE = 0.05
AGV_DIRECTION_CHANGE_INTERVAL = (30, 80)
AGV_MAX_RANDOM_HEADING_CHANGE = 1.047
AGV_MAX_SPEED = 2.0
AGV_MAX_ANGULAR_SPEED = np.pi / 2  # 90 deg/s

# =============================================================================
# ROBOT PARAMETERS
# =============================================================================
ROBOT_RADIUS = 0.6
ROBOT_EFFECTIVE_RADIUS = 1.2

# =============================================================================
# LIDAR CONFIGURATION
# =============================================================================
LIDAR_FOV = 270.0
LIDAR_NUM_RAYS = 250
LIDAR_MAX_RANGE = 30.0
LIDAR_NOISE_STD = 0.03
LIDAR_ANGLE_RANGE = np.pi  # 180 degrees
LIDAR_NUM_RAYS = 180  # Alternative for some imports

# =============================================================================
# OBSTACLE CONFIGURATION
# =============================================================================
OBSTACLE_RADIUS_RANGE = (0.4, 0.5)
OBSTACLE_MIN_DISTANCE = 2.5
DYNAMIC_OBSTACLE_RADIUS = 0.35
OBSTACLE_RADIUS_MIN = 0.2
OBSTACLE_RADIUS_MAX = 0.4
OBSTACLE_MARGIN = 0.3

# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================
# Scenario 1: Static obstacles only
SCENARIO_1_NUM_STATIC_OBSTACLES = 9
SCENARIO_1_X_RANGE = (5, 28)
SCENARIO_1_Y_RANGE = (-5, 5)

# Scenario 2: Dynamic obstacles only
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
SCENARIO_3_NUM_DYNAMIC_OBSTACLES = 4
SCENARIO_3_STATIC_X_RANGE = (8, 25)
SCENARIO_3_STATIC_Y_RANGE = (-5, 5)

# Scenario 4: Empty
SCENARIO_4_NUM_OBSTACLES = 0

# =============================================================================
# DYNAMIC OBSTACLE VELOCITY
# =============================================================================
MIN_VELOCITY = 0.2
MAX_VELOCITY = 1.0
VELOCITY_CHANGE_PROBABILITY = 0.05
VELOCITY_CHANGE_STEP = 0.1

# =============================================================================
# NAVIGATION PARAMETERS
# =============================================================================
NAV_CRITICAL_DISTANCE = 1.5
NAV_SAFETY_DISTANCE = 3.0
GOAL_RADIUS = 1.0

# =============================================================================
# TRACKING AND VISUALIZATION
# =============================================================================
AGV_TRAJECTORY_MAX_LENGTH = 100
LIDAR_HISTORY_MAX_LENGTH = 30