# =============================================================================
# L3 World Model - Obstacle Generator
# =============================================================================

import numpy as np
from typing import List, Tuple

from .config import (
    OBSTACLE_RADIUS_RANGE,
    OBSTACLE_MIN_DISTANCE,
    DYNAMIC_OBSTACLE_RADIUS
)


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
        Generate random static obstacles without overlapping.
        
        Args:
            num_obstacles: Number of obstacles to generate
            x_range: X range (min, max)
            y_range: Y range (min, max)
            radius_range: Radius range (min, max)
            min_dist: Minimum distance between obstacles
            
        Returns:
            List of obstacle dictionaries
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
        Create a single dynamic obstacle.
        
        Args:
            position: Initial position [x, y]
            velocity: Velocity [vx, vy]
            radius: Obstacle radius
            
        Returns:
            Obstacle dictionary
        """
        return {
            'center': position.copy(),
            'radius': radius,
            'velocity': velocity.copy(),
            'type': 'dynamic'
        }
