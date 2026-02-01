# =============================================================================
# L4 Detection - Coordinate Transforms
# =============================================================================
# Utilities for transforming between AGV and world coordinate frames.
# =============================================================================

import numpy as np


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Creates a 2D rotation matrix.
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        2x2 rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_to_world_frame(pos_agv_frame: np.ndarray, 
                             agv_pos: np.ndarray, 
                             agv_heading: float) -> np.ndarray:
    """
    Transforms a position from AGV frame to world frame.
    
    Args:
        pos_agv_frame: Position in AGV frame [x, y]
        agv_pos: AGV position in world frame [x, y]
        agv_heading: AGV heading in radians
        
    Returns:
        Position in world frame [x, y]
    """
    R = rotation_matrix_2d(agv_heading)
    pos_world = agv_pos + R @ pos_agv_frame
    return pos_world


def transform_to_agv_frame(pos_world_frame: np.ndarray,
                           agv_pos: np.ndarray,
                           agv_heading: float) -> np.ndarray:
    """
    Transforms a position from world frame to AGV frame.
    
    Args:
        pos_world_frame: Position in world frame [x, y]
        agv_pos: AGV position in world frame [x, y]
        agv_heading: AGV heading in radians
        
    Returns:
        Position in AGV frame [x, y]
    """
    R = rotation_matrix_2d(-agv_heading)
    pos_agv = R @ (pos_world_frame - agv_pos)
    return pos_agv
