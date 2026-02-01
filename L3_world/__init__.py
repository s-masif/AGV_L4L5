# =============================================================================
# L3 World Model Package
# =============================================================================
# Simulation world layer for AGV navigation.
#
# Responsibilities:
# - LiDAR simulation
# - AGV controllers (Random, Controlled, GoalSeeking)
# - Obstacle generation and management
# - World state management
#
# Usage:
#   from L3_world import WorldModel, ScenarioPresets
#   world = WorldModel(dt=0.1, controlled_mode=True)
#   ScenarioPresets.scenario_mixed(world)
#   state = world.update()
# =============================================================================

from .lidar import LidarSimulator
from .agv import RandomPathAGV, ControlledAGV, GoalSeekingAGV
from .obstacles import ObstacleGenerator
from .world import WorldModel, ScenarioPresets

# Re-export config for convenience
from .config import (
    WORLD_BOUNDS,
    DEFAULT_DT,
    DEFAULT_AGV_SPEED,
    DEFAULT_AGV_START_POSITION,
    DEFAULT_SIMULATION_STEPS
)

__all__ = [
    # LiDAR
    'LidarSimulator',
    
    # AGV Controllers
    'RandomPathAGV',
    'ControlledAGV',
    'GoalSeekingAGV',
    
    # Obstacles
    'ObstacleGenerator',
    
    # World
    'WorldModel',
    'ScenarioPresets',
    
    # Config exports
    'WORLD_BOUNDS',
    'DEFAULT_DT',
    'DEFAULT_AGV_SPEED',
    'DEFAULT_AGV_START_POSITION',
    'DEFAULT_SIMULATION_STEPS',
]

__version__ = '2.0.0'
