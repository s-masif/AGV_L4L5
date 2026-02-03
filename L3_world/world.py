# =============================================================================
# L3 World Model - World Model and Scenario Presets
# =============================================================================

import numpy as np
from typing import List, Tuple
from collections import deque
# Add this import line with the others
from .fusion import SimpleSensorFusion  # New import for sensor fusion
from .config import (
    WORLD_BOUNDS,
    DEFAULT_DT,
    DEFAULT_AGV_SPEED,
    DEFAULT_AGV_START_POSITION,
    AGV_TRAJECTORY_MAX_LENGTH,
    LIDAR_HISTORY_MAX_LENGTH,
    SCENARIO_1_NUM_STATIC_OBSTACLES,
    SCENARIO_1_X_RANGE,
    SCENARIO_1_Y_RANGE,
    SCENARIO_2_OBS1_POS_X_RANGE,
    SCENARIO_2_OBS1_POS_Y_RANGE,
    SCENARIO_2_OBS1_VEL_X_RANGE,
    SCENARIO_2_OBS1_VEL_Y_RANGE,
    SCENARIO_2_OBS2_POS_X_RANGE,
    SCENARIO_2_OBS2_POS_Y_RANGE,
    SCENARIO_2_OBS2_VEL_X_RANGE,
    SCENARIO_2_OBS2_VEL_Y_RANGE,
    SCENARIO_2_OBS3_POS_X_RANGE,
    SCENARIO_2_OBS3_POS_Y_RANGE,
    SCENARIO_2_OBS3_VEL_X_RANGE,
    SCENARIO_2_OBS3_VEL_Y_RANGE,
    SCENARIO_3_NUM_STATIC_OBSTACLES,
    SCENARIO_3_STATIC_X_RANGE,
    SCENARIO_3_STATIC_Y_RANGE
)

from .lidar import LidarSimulator
from .agv import RandomPathAGV, ControlledAGV, GoalSeekingAGV
from .obstacles import ObstacleGenerator


class WorldModel:
    """
    Simulation world model.
    Manages AGV, obstacles, LiDAR and global state.
    """
    
    def __init__(self, dt: float = DEFAULT_DT, 
                    world_bounds: tuple = WORLD_BOUNDS,
                    agv_start_pos: np.ndarray = None,
                    agv_speed: float = DEFAULT_AGV_SPEED,
                    controlled_mode: bool = False,
                    path_mode: str = 'random',
                    goal_pos: np.ndarray = None,
                    enable_fusion: bool = True):  # NEW: Add this parameter
        """
        Initialize the simulation world.
        
        Args:
            dt: Delta time for the simulation
            world_bounds: World limits (x_min, x_max, y_min, y_max)
            agv_start_pos: AGV initial position
            agv_speed: AGV speed
            controlled_mode: If True, AGV is controlled by navigation algorithm
            path_mode: 'random' for random movement, 'straight' for goal-seeking
            goal_pos: Goal position for 'straight' mode
        """
        self.dt = dt
        self.world_bounds = world_bounds
        self.controlled_mode = controlled_mode
        self.path_mode = path_mode
        self.agv_speed = agv_speed
        
        x_min, x_max, y_min, y_max = world_bounds
        if agv_start_pos is None:
            if path_mode == 'straight':
                agv_start_pos = np.array([x_min + 1.0, (y_min + y_max) / 2])
            else:
                agv_start_pos = np.array(DEFAULT_AGV_START_POSITION)
        
        if goal_pos is None and path_mode == 'straight':
            goal_pos = np.array([x_max - 1.0, (y_min + y_max) / 2])
        
        self.agv_start_pos = agv_start_pos.copy()
        self.goal_pos = goal_pos.copy() if goal_pos is not None else None
        self.enable_fusion = enable_fusion  # NEW: Store fusion setting
        
        # Initialize sensor fusion
        if enable_fusion:
            self.sensor_fusion = SimpleSensorFusion()
        else:
            self.sensor_fusion = None
        self.agv = self._create_agv(agv_start_pos, goal_pos)
        self.lidar = LidarSimulator()
        
        self.static_obstacles: List[dict] = []
        self.dynamic_obstacles: List[dict] = []
        
        self.agv_trajectory = deque(maxlen=AGV_TRAJECTORY_MAX_LENGTH)
        self.lidar_history = deque(maxlen=LIDAR_HISTORY_MAX_LENGTH)
        self.fusion_history = deque(maxlen=LIDAR_HISTORY_MAX_LENGTH)
        self.current_time = 0.0
        self.current_frame = 0
    
    def _create_agv(self, start_pos: np.ndarray, goal_pos: np.ndarray = None):
        if self.controlled_mode:
            return ControlledAGV(start_pos=start_pos, speed=self.agv_speed, goal_pos=goal_pos)
        elif self.path_mode == 'straight' and goal_pos is not None:
            return GoalSeekingAGV(start_pos=start_pos, goal_pos=goal_pos, speed=self.agv_speed)
        else:
            return RandomPathAGV(start_pos=start_pos, speed=self.agv_speed)
        
    def add_static_obstacles(self, obstacles: List[dict]):
        self.static_obstacles.extend(obstacles)
    
    def add_dynamic_obstacle(self, obstacle: dict):
        self.dynamic_obstacles.append(obstacle)
    
    def clear_obstacles(self):
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()
        if self.sensor_fusion:  # NEW: Clear fusion tracks too
            self.sensor_fusion.clear_tracks()
    
    def reset(self, agv_start_pos: np.ndarray = None):
        if agv_start_pos is None:
            agv_start_pos = self.agv_start_pos.copy()
        
        self.agv = self._create_agv(agv_start_pos, self.goal_pos)
        
        if self.sensor_fusion:  # NEW: Reset fusion tracks
            self.sensor_fusion.clear_tracks()
        
        self.agv_trajectory.clear()
        self.lidar_history.clear()
        self.fusion_history.clear()  # NEW: Clear fusion history
        self.current_time = 0.0
        self.current_frame = 0
    
    def update(self) -> dict:
        agv_pos, agv_vel, agv_heading = self.agv.update(self.dt, self.world_bounds)
        self.agv_trajectory.append(agv_pos.copy())
        
        x_min, x_max, y_min, y_max = self.world_bounds
        for obs in self.dynamic_obstacles:
            obs['center'] = obs['center'] + obs['velocity'] * self.dt
            radius = obs.get('radius', 0.35)
            
            if obs['center'][0] - radius < x_min:
                obs['center'][0] = x_min + radius
                obs['velocity'][0] = abs(obs['velocity'][0])
            elif obs['center'][0] + radius > x_max:
                obs['center'][0] = x_max - radius
                obs['velocity'][0] = -abs(obs['velocity'][0])
            
            if obs['center'][1] - radius < y_min:
                obs['center'][1] = y_min + radius
                obs['velocity'][1] = abs(obs['velocity'][1])
            elif obs['center'][1] + radius > y_max:
                obs['center'][1] = y_max - radius
                obs['velocity'][1] = -abs(obs['velocity'][1])
        
        all_obstacles = self._get_all_obstacles()
        ranges, angles = self.lidar.scan(all_obstacles, agv_pos, agv_heading)
        self.lidar_history.append((ranges.copy(), angles.copy()))
        
        self.current_time += self.dt
        self.current_frame += 1
        
        # Get goal_reached from AGV state
        agv_state = self.agv.get_state()
        goal_reached = agv_state.get('goal_reached', False)

        fused_obstacles = []
        if self.sensor_fusion:
            fused_obstacles = self.sensor_fusion.update(
                ranges, angles, agv_pos, agv_heading
            )
            self.fusion_history.append(fused_obstacles.copy())
        
        return {
            'agv_pos': agv_pos,
            'agv_vel': agv_vel,
            'agv_heading': agv_heading,
            'lidar_ranges': ranges,
            'lidar_angles': angles,
            'obstacles': all_obstacles,
            'fused_obstacles': fused_obstacles,  # NEW: fused estimates
            'time': self.current_time,
            'frame': self.current_frame,
            'goal_reached': goal_reached
        }
    
    def _get_all_obstacles(self) -> List[dict]:
        all_obs = []
        for obs in self.static_obstacles:
            all_obs.append({
                'center': obs['center'].copy(),
                'radius': obs.get('radius', 0.3),
                'velocity': np.array([0.0, 0.0]),
                'type': 'static'
            })
        for obs in self.dynamic_obstacles:
            all_obs.append({
                'center': obs['center'].copy(),
                'radius': obs.get('radius', 0.35),
                'velocity': obs['velocity'].copy(),
                'type': 'dynamic'
            })
        return all_obs
    
    def get_ground_truth(self) -> dict:
        ground_truth = {}
        idx = 0
        for obs in self.static_obstacles:
            ground_truth[idx] = {
                'position': obs['center'].copy(),
                'velocity': np.array([0.0, 0.0]),
                'type': 'STATIC'
            }
            idx += 1
        for obs in self.dynamic_obstacles:
            ground_truth[idx] = {
                'position': obs['center'].copy(),
                'velocity': obs['velocity'].copy(),
                'type': 'DYNAMIC'
            }
            idx += 1
        return ground_truth
    
    def get_agv_state(self) -> dict:
        return self.agv.get_state()
    
    def set_agv_command(self, target_speed: float, heading_change: float):
        if self.controlled_mode and hasattr(self.agv, 'set_command'):
            self.agv.set_command(target_speed, heading_change)
    
    def get_lidar_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.lidar_history:
            return self.lidar_history[-1]
        return None, None
    
    def get_fused_data(self) -> List[dict]:
        """Get the latest fused obstacle estimates."""
        if self.fusion_history:
            return self.fusion_history[-1]
        return []
    
    def toggle_fusion(self, enable: bool = None):
        """Toggle sensor fusion on/off."""
        if enable is None:
            self.enable_fusion = not self.enable_fusion
        else:
            self.enable_fusion = enable
        
        if self.enable_fusion and self.sensor_fusion is None:
            self.sensor_fusion = SimpleSensorFusion()
        elif not self.enable_fusion:
            self.sensor_fusion = None


class ScenarioPresets:
    """Presets for common test scenarios."""
    
    @staticmethod
    def scenario_static_only(world: WorldModel):
        world.clear_obstacles()
        static_obs = ObstacleGenerator.generate_random_static_obstacles(
            num_obstacles=SCENARIO_1_NUM_STATIC_OBSTACLES,
            x_range=SCENARIO_1_X_RANGE,
            y_range=SCENARIO_1_Y_RANGE
        )
        world.add_static_obstacles(static_obs)
        world.reset()
        return {'type': 'static_only', 'num_obstacles': len(static_obs)}
    
    @staticmethod
    def scenario_dynamic_only(world: WorldModel):
        world.clear_obstacles()
        
        obs1 = ObstacleGenerator.create_dynamic_obstacle(
            position=np.array([
                np.random.uniform(*SCENARIO_2_OBS1_POS_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS1_POS_Y_RANGE)
            ]),
            velocity=np.array([
                np.random.uniform(*SCENARIO_2_OBS1_VEL_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS1_VEL_Y_RANGE)
            ])
        )
        world.add_dynamic_obstacle(obs1)
        
        obs2 = ObstacleGenerator.create_dynamic_obstacle(
            position=np.array([
                np.random.uniform(*SCENARIO_2_OBS2_POS_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS2_POS_Y_RANGE)
            ]),
            velocity=np.array([
                np.random.uniform(*SCENARIO_2_OBS2_VEL_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS2_VEL_Y_RANGE)
            ])
        )
        world.add_dynamic_obstacle(obs2)
        
        obs3 = ObstacleGenerator.create_dynamic_obstacle(
            position=np.array([
                np.random.uniform(*SCENARIO_2_OBS3_POS_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS3_POS_Y_RANGE)
            ]),
            velocity=np.array([
                np.random.uniform(*SCENARIO_2_OBS3_VEL_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS3_VEL_Y_RANGE)
            ])
        )
        world.add_dynamic_obstacle(obs3)
        
        world.reset()
        return {'type': 'dynamic_only', 'num_obstacles': 3}
    
    @staticmethod
    def scenario_mixed(world: WorldModel):
        world.clear_obstacles()
        
        static_obs = ObstacleGenerator.generate_random_static_obstacles(
            num_obstacles=SCENARIO_3_NUM_STATIC_OBSTACLES,
            x_range=SCENARIO_3_STATIC_X_RANGE,
            y_range=SCENARIO_3_STATIC_Y_RANGE
        )
        world.add_static_obstacles(static_obs)
        
        obs1 = ObstacleGenerator.create_dynamic_obstacle(
            position=np.array([
                np.random.uniform(*SCENARIO_2_OBS1_POS_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS1_POS_Y_RANGE)
            ]),
            velocity=np.array([
                np.random.uniform(*SCENARIO_2_OBS1_VEL_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS1_VEL_Y_RANGE)
            ])
        )
        world.add_dynamic_obstacle(obs1)
        
        obs2 = ObstacleGenerator.create_dynamic_obstacle(
            position=np.array([
                np.random.uniform(*SCENARIO_2_OBS2_POS_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS2_POS_Y_RANGE)
            ]),
            velocity=np.array([
                np.random.uniform(*SCENARIO_2_OBS2_VEL_X_RANGE), 
                np.random.uniform(*SCENARIO_2_OBS2_VEL_Y_RANGE)
            ])
        )
        world.add_dynamic_obstacle(obs2)
        
        world.reset()
        return {'type': 'mixed', 'num_static': len(static_obs), 'num_dynamic': 2}

    @staticmethod
    def scenario_empty(world: WorldModel):
        """Empty scenario with no obstacles - for testing goal reaching."""
        world.clear_obstacles()
        world.reset()
        return {'type': 'empty', 'num_obstacles': 0}

    @staticmethod
    def scenario_custom(world: WorldModel, num_static: int = 0, num_dynamic: int = 0):
        """Custom scenario with specified number of obstacles."""
        world.clear_obstacles()
        
        if num_static > 0:
            static_obs = ObstacleGenerator.generate_random_static_obstacles(
                num_obstacles=num_static,
                x_range=SCENARIO_1_X_RANGE,
                y_range=SCENARIO_1_Y_RANGE
            )
            world.add_static_obstacles(static_obs)
        
        world.reset()
        return {'type': 'custom', 'num_static': num_static, 'num_dynamic': num_dynamic}
