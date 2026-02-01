# =============================================================================
# L3 World Model - LiDAR Simulator
# =============================================================================

import numpy as np
from typing import List, Tuple

from .config import (
    LIDAR_FOV,
    LIDAR_NUM_RAYS,
    LIDAR_MAX_RANGE,
    LIDAR_NOISE_STD
)


class LidarSimulator:
    """
    LiDAR simulator for obstacle detection.
    Emulates a 2D LiDAR sensor with configurable FOV and noise.
    """
    
    def __init__(self, fov: float = LIDAR_FOV, numrays: int = LIDAR_NUM_RAYS, 
                 maxrange: float = LIDAR_MAX_RANGE, noisestd: float = LIDAR_NOISE_STD):
        """
        Initialize the LiDAR simulator.
        
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
        Perform a LiDAR scan considering the AGV heading.
        
        Args:
            obstacles: List of obstacles with 'center' and 'radius'
            agvpos: AGV position [x, y]
            agv_heading: AGV heading angle in radians
            
        Returns:
            Tuple (ranges, angles): measured distances and corresponding angles
        """
        ranges = np.full(self.numrays, self.maxrange)
        
        for angleidx, angle in enumerate(self.angles):
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
