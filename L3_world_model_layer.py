# =============================================================================
# L3 - WORLD MODEL LAYER
# =============================================================================
# This layer creates the simulation world where the AGV operates:
# - LiDAR Simulator
# - AGV Controller with movement
# - Obstacle management (static and dynamic)
# - World bounds definition
# =============================================================================

import numpy as np
from typing import List, Tuple
from collections import deque

# Import configurazione
from config_L3 import (
    WORLD_BOUNDS,
    DEFAULT_DT,
    DEFAULT_AGV_SPEED,
    DEFAULT_AGV_START_POSITION,
    AGV_MAX_TURN_RATE,
    AGV_DIRECTION_CHANGE_INTERVAL,
    AGV_MAX_RANDOM_HEADING_CHANGE,
    LIDAR_FOV,
    LIDAR_NUM_RAYS,
    LIDAR_MAX_RANGE,
    LIDAR_NOISE_STD,
    OBSTACLE_RADIUS_RANGE,
    OBSTACLE_MIN_DISTANCE,
    DYNAMIC_OBSTACLE_RADIUS,
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
    SCENARIO_3_STATIC_Y_RANGE,
    AGV_TRAJECTORY_MAX_LENGTH,
    LIDAR_HISTORY_MAX_LENGTH
)

# =============================================================================
# LiDAR Simulator
# =============================================================================
class LidarSimulator:
    """
    LiDAR simulator for obstacle detection.
    Emulates a 2D LiDAR sensor with configurable FOV and noise.
    """
    
    def __init__(self, fov: float = LIDAR_FOV, numrays: int = LIDAR_NUM_RAYS, 
                 maxrange: float = LIDAR_MAX_RANGE, noisestd: float = LIDAR_NOISE_STD):
        """
        Initializes the LiDAR simulator.
        
        Args:
            fov: Field of view in degrees
            numrays: Number of laser rays
            maxrange: Maximum range in meters
            noisestd: Noise standard deviation
        """
        self.fov = np.deg2rad(fov)
        self.numrays = numrays
        self.maxrange = maxrange
        self.noisestd = noisestd
        startangle = -self.fov / 2
        endangle = self.fov / 2
        self.angles = np.linspace(startangle, endangle, numrays)

    def scan(self, obstacles: List[dict], agvpos: np.ndarray, 
             agv_heading: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a LiDAR scan considering the AGV heading.
        
        Args:
            obstacles: List of obstacles with 'center' and 'radius'
            agvpos: AGV position [x, y]
            agv_heading: AGV heading angle in radians
            
        Returns:
            Tuple (ranges, angles): measured distances and corresponding angles
        """
        ranges = np.full(self.numrays, self.maxrange)
        
        for angleidx, angle in enumerate(self.angles):
            # Angolo assoluto = angolo AGV + angolo relativo LiDAR
            absolute_angle = agv_heading + angle
            raydir = np.array([np.cos(absolute_angle), np.sin(absolute_angle)])
            mindist = self.maxrange
            
            for obs in obstacles:
                obscenter = obs['center']
                obsradius = obs.get('radius', 0.3)
                relpos = obscenter - agvpos
                proj = np.dot(relpos, raydir)
                
                if proj > 0:
                    closestpoint = proj * raydir
                    disttocenter = np.linalg.norm(relpos - closestpoint)
                    if disttocenter <= obsradius:
                        dist = proj - np.sqrt(obsradius**2 - disttocenter**2)
                        if 0 < dist < mindist:
                            mindist = dist
            ranges[angleidx] = mindist
        
        # Add Gaussian noise
        ranges += np.random.normal(0, self.noisestd, ranges.shape)
        ranges = np.clip(ranges, 0.1, self.maxrange)
        return ranges, self.angles


# =============================================================================
# AGV Controller with Random Movement
# =============================================================================
class RandomPathAGV:
    """
    AGV Controller with semi-random movement.
    The AGV moves with gradual direction changes and bounces off borders.
    """
    
    def __init__(self, start_pos: np.ndarray, speed: float = DEFAULT_AGV_SPEED):
        """
        Initializes the AGV.
        
        Args:
            start_pos: Initial position [x, y]
            speed: Movement speed in m/s
        """
        self.pos = start_pos.copy()
        self.speed = speed
        self.heading = 0.0  # Movement angle (radians)
        self.vel = np.array([speed, 0.0])
        
        # Random movement parameters
        self.target_heading = 0.0
        self.change_direction_timer = 0
        self.change_direction_interval = np.random.randint(
            AGV_DIRECTION_CHANGE_INTERVAL[0], AGV_DIRECTION_CHANGE_INTERVAL[1]
        )
        
    def update(self, dt: float, world_bounds: tuple) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Updates AGV position with random movement.
        
        Args:
            dt: Delta time
            world_bounds: (x_min, x_max, y_min, y_max)
            
        Returns:
            Tuple (position, velocity, heading)
        """
        self.change_direction_timer += 1
        
        # Periodic random direction change
        if self.change_direction_timer >= self.change_direction_interval:
            self.target_heading = np.random.uniform(
                -AGV_MAX_RANDOM_HEADING_CHANGE, AGV_MAX_RANDOM_HEADING_CHANGE
            )
            self.change_direction_interval = np.random.randint(
                AGV_DIRECTION_CHANGE_INTERVAL[0], AGV_DIRECTION_CHANGE_INTERVAL[1]
            )
            self.change_direction_timer = 0
        
        # Smooth steering
        heading_diff = self.target_heading - self.heading
        if heading_diff > np.pi:
            heading_diff -= 2 * np.pi
        elif heading_diff < -np.pi:
            heading_diff += 2 * np.pi
        
        self.heading += np.clip(heading_diff, -AGV_MAX_TURN_RATE, AGV_MAX_TURN_RATE)
        
        # Calculate velocity based on direction
        self.vel = np.array([
            self.speed * np.cos(self.heading),
            self.speed * np.sin(self.heading)
        ])
        
        # Update position
        new_pos = self.pos + self.vel * dt
        
        # Border collision handling (bounce)
        x_min, x_max, y_min, y_max = world_bounds
        
        if new_pos[0] < x_min or new_pos[0] > x_max:
            self.heading = np.pi - self.heading
            self.target_heading = self.heading
            new_pos[0] = np.clip(new_pos[0], x_min, x_max)
        
        if new_pos[1] < y_min or new_pos[1] > y_max:
            self.heading = -self.heading
            self.target_heading = self.heading
            new_pos[1] = np.clip(new_pos[1], y_min, y_max)
        
        self.pos = new_pos
        return self.pos, self.vel, self.heading
    
    def get_state(self) -> dict:
        """Returns the current AGV state."""
        return {
            'position': self.pos.copy(),
            'velocity': self.vel.copy(),
            'heading': self.heading,
            'speed': self.speed
        }
    
    def set_position(self, pos: np.ndarray):
        """Sets the AGV position."""
        self.pos = pos.copy()
    
    def set_heading(self, heading: float):
        """Sets the AGV heading."""
        self.heading = heading
        self.target_heading = heading


# =============================================================================
# Controlled AGV (for navigation algorithms)
# =============================================================================
class ControlledAGV:
    """
    AGV Controller that can be commanded by navigation algorithms.
    Uses target speed and heading change from decision layer.
    """
    
    def __init__(self, start_pos: np.ndarray, speed: float = DEFAULT_AGV_SPEED):
        """
        Initializes the AGV.
        
        Args:
            start_pos: Initial position [x, y]
            speed: Maximum movement speed in m/s
        """
        self.pos = start_pos.copy()
        self.max_speed = speed
        self.current_speed = 0.0
        self.heading = 0.0
        self.vel = np.array([0.0, 0.0])
        
        # Command inputs (set by decision layer)
        self.target_speed = 0.0
        self.target_heading_change = 0.0
        
    def set_command(self, target_speed: float, heading_change: float):
        """
        Set command from navigation algorithm.
        
        Args:
            target_speed: Desired speed (m/s)
            heading_change: Desired heading change (radians)
        """
        self.target_speed = np.clip(target_speed, 0.0, self.max_speed)
        self.target_heading_change = np.clip(heading_change, -AGV_MAX_TURN_RATE, AGV_MAX_TURN_RATE)
        
    def update(self, dt: float, world_bounds: tuple) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Updates AGV position based on commands.
        
        Args:
            dt: Delta time
            world_bounds: (x_min, x_max, y_min, y_max)
            
        Returns:
            Tuple (position, velocity, heading)
        """
        # Smooth speed transition (acceleration limit: 2.0 m/s²)
        max_accel = 2.0 * dt  # Maximum speed change per step
        speed_diff = self.target_speed - self.current_speed
        self.current_speed += np.clip(speed_diff, -max_accel, max_accel)
        self.current_speed = np.clip(self.current_speed, 0.0, self.max_speed)
        
        # Apply heading change
        self.heading += self.target_heading_change
        
        # Normalize heading to [-pi, pi]
        while self.heading > np.pi:
            self.heading -= 2 * np.pi
        while self.heading < -np.pi:
            self.heading += 2 * np.pi
        
        # Calculate velocity based on direction
        self.vel = np.array([
            self.current_speed * np.cos(self.heading),
            self.current_speed * np.sin(self.heading)
        ])
        
        # Update position
        new_pos = self.pos + self.vel * dt
        
        # Border collision handling (bounce)
        x_min, x_max, y_min, y_max = world_bounds
        
        if new_pos[0] < x_min or new_pos[0] > x_max:
            self.heading = np.pi - self.heading
            new_pos[0] = np.clip(new_pos[0], x_min, x_max)
            self.current_speed *= 0.5  # Slow down on collision
        
        if new_pos[1] < y_min or new_pos[1] > y_max:
            self.heading = -self.heading
            new_pos[1] = np.clip(new_pos[1], y_min, y_max)
            self.current_speed *= 0.5  # Slow down on collision
        
        self.pos = new_pos
        return self.pos, self.vel, self.heading
    
    def get_state(self) -> dict:
        """Returns the current AGV state."""
        return {
            'position': self.pos.copy(),
            'velocity': self.vel.copy(),
            'heading': self.heading,
            'speed': self.current_speed,
            'max_speed': self.max_speed
        }
    
    def set_position(self, pos: np.ndarray):
        """Sets the AGV position."""
        self.pos = pos.copy()
    
    def set_heading(self, heading: float):
        """Sets the AGV heading."""
        self.heading = heading


# =============================================================================
# Goal-Seeking AGV (for straight path with fixed start/goal)
# =============================================================================
class GoalSeekingAGV:
    """
    AGV Controller that navigates toward a fixed goal position.
    Used for straight-line navigation from left to right.
    """
    
    def __init__(self, start_pos: np.ndarray, goal_pos: np.ndarray, 
                 speed: float = DEFAULT_AGV_SPEED):
        """
        Initializes the AGV with start and goal positions.
        
        Args:
            start_pos: Initial position [x, y]
            goal_pos: Target position [x, y]
            speed: Maximum movement speed in m/s
        """
        self.pos = start_pos.copy()
        self.goal_pos = goal_pos.copy()
        self.max_speed = speed
        self.current_speed = speed * 0.5  # Start at half speed
        
        # Calculate initial heading toward goal
        diff = self.goal_pos - self.pos
        self.heading = np.arctan2(diff[1], diff[0])
        self.vel = np.array([
            self.current_speed * np.cos(self.heading),
            self.current_speed * np.sin(self.heading)
        ])
        
        # Goal reached flag
        self.goal_reached = False
        self.goal_tolerance = 0.5  # meters
        
    def update(self, dt: float, world_bounds: tuple) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Updates AGV position, steering toward goal.
        
        Args:
            dt: Delta time
            world_bounds: (x_min, x_max, y_min, y_max)
            
        Returns:
            Tuple (position, velocity, heading)
        """
        # Check if goal reached
        dist_to_goal = np.linalg.norm(self.goal_pos - self.pos)
        if dist_to_goal < self.goal_tolerance:
            self.goal_reached = True
            self.current_speed = 0.0
            self.vel = np.array([0.0, 0.0])
            return self.pos, self.vel, self.heading
        
        # Calculate desired heading toward goal
        diff = self.goal_pos - self.pos
        desired_heading = np.arctan2(diff[1], diff[0])
        
        # Calculate heading difference
        heading_diff = desired_heading - self.heading
        while heading_diff > np.pi:
            heading_diff -= 2 * np.pi
        while heading_diff < -np.pi:
            heading_diff += 2 * np.pi
        
        # Gradual steering
        max_turn = AGV_MAX_TURN_RATE * 2  # Slightly faster steering for goal-seeking
        self.heading += np.clip(heading_diff, -max_turn, max_turn)
        
        # Speed control based on distance
        if dist_to_goal < 2.0:
            # Slow down when approaching goal
            self.current_speed = max(0.1, self.max_speed * (dist_to_goal / 2.0))
        else:
            # Accelerate to max speed
            self.current_speed = min(self.max_speed, self.current_speed + 0.5 * dt)
        
        # Calculate velocity
        self.vel = np.array([
            self.current_speed * np.cos(self.heading),
            self.current_speed * np.sin(self.heading)
        ])
        
        # Update position
        new_pos = self.pos + self.vel * dt
        
        # Border collision (shouldn't happen with proper goal, but safety)
        x_min, x_max, y_min, y_max = world_bounds
        new_pos[0] = np.clip(new_pos[0], x_min, x_max)
        new_pos[1] = np.clip(new_pos[1], y_min, y_max)
        
        self.pos = new_pos
        return self.pos, self.vel, self.heading
    
    def get_state(self) -> dict:
        """Returns the current AGV state."""
        return {
            'position': self.pos.copy(),
            'velocity': self.vel.copy(),
            'heading': self.heading,
            'speed': self.current_speed,
            'max_speed': self.max_speed,
            'goal_pos': self.goal_pos.copy(),
            'goal_reached': self.goal_reached
        }
    
    def set_position(self, pos: np.ndarray):
        """Sets the AGV position."""
        self.pos = pos.copy()
    
    def set_heading(self, heading: float):
        """Sets the AGV heading."""
        self.heading = heading
    
    def set_goal(self, goal_pos: np.ndarray):
        """Sets a new goal position."""
        self.goal_pos = goal_pos.copy()
        self.goal_reached = False


# =============================================================================
# Obstacle Generator
# =============================================================================
class ObstacleGenerator:
    """
    Obstacle generator for the simulation world.
    Supports static and dynamic obstacles.
    """
    
    @staticmethod
    def generate_random_static_obstacles(num_obstacles: int, 
                                         x_range: Tuple[float, float],
                                         y_range: Tuple[float, float],
                                         radius_range: Tuple[float, float] = OBSTACLE_RADIUS_RANGE,
                                         min_dist: float = OBSTACLE_MIN_DISTANCE) -> List[dict]:
        """
        Generates random static obstacles without overlapping.
        
        Args:
            num_obstacles: Number of obstacles to generate
            x_range: X range (min, max)
            y_range: Y range (min, max)
            radius_range: Radius range (min, max)
            min_dist: Minimum distance between obstacles
            
        Returns:
            List of dictionaries with 'center', 'radius', 'velocity'
        """
        obstacles = []
        for _ in range(num_obstacles):
            for attempt in range(100):
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                r = np.random.uniform(radius_range[0], radius_range[1])
                
                valid = True
                for obs in obstacles:
                    dist = np.linalg.norm(np.array([x, y]) - obs['center'])
                    if dist < min_dist:
                        valid = False
                        break
                
                if valid:
                    obstacles.append({
                        'center': np.array([x, y]),
                        'radius': r,
                        'velocity': np.array([0.0, 0.0]),
                        'type': 'static'
                    })
                    break
        return obstacles
    
    @staticmethod
    def create_dynamic_obstacle(position: np.ndarray, 
                                velocity: np.ndarray,
                                radius: float = DYNAMIC_OBSTACLE_RADIUS) -> dict:
        """
        Creates a single dynamic obstacle.
        
        Args:
            position: Initial position [x, y]
            velocity: Velocity [vx, vy]
            radius: Obstacle radius
            
        Returns:
            Dictionary with 'center', 'radius', 'velocity'
        """
        return {
            'center': position.copy(),
            'radius': radius,
            'velocity': velocity.copy(),
            'type': 'dynamic'
        }


# =============================================================================
# World Model - Manages the entire simulation world
# =============================================================================
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
                 goal_pos: np.ndarray = None):
        """
        Initializes the simulation world.
        
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
        
        # Default positions based on world bounds
        x_min, x_max, y_min, y_max = world_bounds
        if agv_start_pos is None:
            if path_mode == 'straight':
                # Start on the left side
                agv_start_pos = np.array([x_min + 1.0, (y_min + y_max) / 2])
            else:
                agv_start_pos = np.array(DEFAULT_AGV_START_POSITION)
        
        if goal_pos is None and path_mode == 'straight':
            # Goal on the right side
            goal_pos = np.array([x_max - 1.0, (y_min + y_max) / 2])
        
        self.agv_start_pos = agv_start_pos.copy()
        self.goal_pos = goal_pos.copy() if goal_pos is not None else None
        
        # Initialize AGV based on mode
        self.agv = self._create_agv(agv_start_pos, goal_pos)
        
        # Initialize LiDAR (uses config parameters)
        self.lidar = LidarSimulator()
        
        # Obstacle list
        self.static_obstacles: List[dict] = []
        self.dynamic_obstacles: List[dict] = []
        
        # Tracking
        self.agv_trajectory = deque(maxlen=AGV_TRAJECTORY_MAX_LENGTH)
        self.lidar_history = deque(maxlen=LIDAR_HISTORY_MAX_LENGTH)
        self.current_time = 0.0
        self.current_frame = 0
    
    def _create_agv(self, start_pos: np.ndarray, goal_pos: np.ndarray = None):
        """Creates the appropriate AGV type based on mode."""
        if self.controlled_mode:
            return ControlledAGV(start_pos=start_pos, speed=self.agv_speed)
        elif self.path_mode == 'straight' and goal_pos is not None:
            return GoalSeekingAGV(start_pos=start_pos, goal_pos=goal_pos, speed=self.agv_speed)
        else:
            return RandomPathAGV(start_pos=start_pos, speed=self.agv_speed)
        
    def add_static_obstacles(self, obstacles: List[dict]):
        """Adds static obstacles to the world."""
        self.static_obstacles.extend(obstacles)
    
    def add_dynamic_obstacle(self, obstacle: dict):
        """Adds a dynamic obstacle to the world."""
        self.dynamic_obstacles.append(obstacle)
    
    def clear_obstacles(self):
        """Removes all obstacles."""
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()
    
    def reset(self, agv_start_pos: np.ndarray = None):
        """Resets the world keeping the configuration."""
        if agv_start_pos is None:
            agv_start_pos = self.agv_start_pos.copy()
        
        # Recreate AGV with appropriate type
        self.agv = self._create_agv(agv_start_pos, self.goal_pos)
        
        self.agv_trajectory.clear()
        self.lidar_history.clear()
        self.current_time = 0.0
        self.current_frame = 0
    
    def update(self) -> dict:
        """
        Updates the world state (one simulation step).
        
        Returns:
            Dictionary with current world state
        """
        # Update AGV
        agv_pos, agv_vel, agv_heading = self.agv.update(self.dt, self.world_bounds)
        self.agv_trajectory.append(agv_pos.copy())
        
        # Update dynamic obstacles with wall bouncing
        x_min, x_max, y_min, y_max = self.world_bounds
        for obs in self.dynamic_obstacles:
            # Move obstacle
            obs['center'] = obs['center'] + obs['velocity'] * self.dt
            radius = obs.get('radius', 0.35)
            
            # Bounce off walls (X axis)
            if obs['center'][0] - radius < x_min:
                obs['center'][0] = x_min + radius
                obs['velocity'][0] = abs(obs['velocity'][0])  # Reverse X velocity
            elif obs['center'][0] + radius > x_max:
                obs['center'][0] = x_max - radius
                obs['velocity'][0] = -abs(obs['velocity'][0])  # Reverse X velocity
            
            # Bounce off walls (Y axis)
            if obs['center'][1] - radius < y_min:
                obs['center'][1] = y_min + radius
                obs['velocity'][1] = abs(obs['velocity'][1])  # Reverse Y velocity
            elif obs['center'][1] + radius > y_max:
                obs['center'][1] = y_max - radius
                obs['velocity'][1] = -abs(obs['velocity'][1])  # Reverse Y velocity
        
        # Build current obstacle list
        all_obstacles = self._get_all_obstacles()
        
        # Perform LiDAR scan
        ranges, angles = self.lidar.scan(all_obstacles, agv_pos, agv_heading)
        self.lidar_history.append((ranges.copy(), angles.copy()))
        
        # Update time
        self.current_time += self.dt
        self.current_frame += 1
        
        return {
            'agv_pos': agv_pos,
            'agv_vel': agv_vel,
            'agv_heading': agv_heading,
            'lidar_ranges': ranges,
            'lidar_angles': angles,
            'obstacles': all_obstacles,
            'time': self.current_time,
            'frame': self.current_frame
        }
    
    def _get_all_obstacles(self) -> List[dict]:
        """Returns all obstacles (static + dynamic)."""
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
        """
        Returns the ground truth for all obstacles.
        Useful for evaluating detection system performance.
        """
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
        """Returns the current AGV state."""
        return self.agv.get_state()
    
    def set_agv_command(self, target_speed: float, heading_change: float):
        """
        Set command for controlled AGV.
        Only works if controlled_mode=True.
        
        Args:
            target_speed: Desired speed (m/s)
            heading_change: Desired heading change (radians)
        """
        if self.controlled_mode and hasattr(self.agv, 'set_command'):
            self.agv.set_command(target_speed, heading_change)
    
    def get_lidar_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the last LiDAR scan."""
        if self.lidar_history:
            return self.lidar_history[-1]
        return None, None


# =============================================================================
# Scenario Presets
# =============================================================================
class ScenarioPresets:
    """
    Presets for common test scenarios.
    """
    
    @staticmethod
    def scenario_static_only(world: WorldModel):
        """Scenario 1: Static obstacles only."""
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
        """Scenario 2: Dynamic obstacles only."""
        world.clear_obstacles()
        
        # Obstacle 1: moves left
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
        
        # Obstacle 2: moves down
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
        
        # Obstacle 3: moves diagonally
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
        """Scenario 3: Mix of static and dynamic obstacles."""
        world.clear_obstacles()
        
        # Static obstacles
        static_obs = ObstacleGenerator.generate_random_static_obstacles(
            num_obstacles=SCENARIO_3_NUM_STATIC_OBSTACLES,
            x_range=SCENARIO_3_STATIC_X_RANGE,
            y_range=SCENARIO_3_STATIC_Y_RANGE
        )
        world.add_static_obstacles(static_obs)
        
        # Dynamic obstacles (uses same ranges as scenario 2)
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
