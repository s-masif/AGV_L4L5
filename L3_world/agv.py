# =============================================================================
# L3 World Model - AGV Controllers
# =============================================================================

import numpy as np
from typing import Tuple

from .config import (
    DEFAULT_AGV_SPEED,
    AGV_MAX_TURN_RATE,
    AGV_DIRECTION_CHANGE_INTERVAL,
    AGV_MAX_RANDOM_HEADING_CHANGE
)


class RandomPathAGV:
    """
    AGV Controller with semi-random movement.
    The AGV moves with gradual direction changes and bounces off borders.
    """
    
    def __init__(self, start_pos: np.ndarray, speed: float = DEFAULT_AGV_SPEED):
        self.pos = start_pos.copy()
        self.speed = speed
        self.heading = 0.0
        self.vel = np.array([speed, 0.0])
        
        self.target_heading = 0.0
        self.change_direction_timer = 0
        self.change_direction_interval = np.random.randint(
            AGV_DIRECTION_CHANGE_INTERVAL[0], AGV_DIRECTION_CHANGE_INTERVAL[1]
        )
        
    def update(self, dt: float, world_bounds: tuple) -> Tuple[np.ndarray, np.ndarray, float]:
        self.change_direction_timer += 1
        
        if self.change_direction_timer >= self.change_direction_interval:
            self.target_heading = np.random.uniform(
                -AGV_MAX_RANDOM_HEADING_CHANGE, AGV_MAX_RANDOM_HEADING_CHANGE
            )
            self.change_direction_interval = np.random.randint(
                AGV_DIRECTION_CHANGE_INTERVAL[0], AGV_DIRECTION_CHANGE_INTERVAL[1]
            )
            self.change_direction_timer = 0
        
        heading_diff = self.target_heading - self.heading
        if heading_diff > np.pi:
            heading_diff -= 2 * np.pi
        elif heading_diff < -np.pi:
            heading_diff += 2 * np.pi
        
        self.heading += np.clip(heading_diff, -AGV_MAX_TURN_RATE, AGV_MAX_TURN_RATE)
        
        self.vel = np.array([
            self.speed * np.cos(self.heading),
            self.speed * np.sin(self.heading)
        ])
        
        new_pos = self.pos + self.vel * dt
        
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
        return {
            'position': self.pos.copy(),
            'velocity': self.vel.copy(),
            'heading': self.heading,
            'speed': self.speed
        }
    
    def set_position(self, pos: np.ndarray):
        self.pos = pos.copy()
    
    def set_heading(self, heading: float):
        self.heading = heading
        self.target_heading = heading


class ControlledAGV:
    """
    AGV Controller that can be commanded by navigation algorithms.
    """
    
    def __init__(self, start_pos: np.ndarray, speed: float = DEFAULT_AGV_SPEED,
                 goal_pos: np.ndarray = None):
        self.pos = start_pos.copy()
        self.max_speed = speed
        self.current_speed = 0.0
        self.heading = 0.0
        self.vel = np.array([0.0, 0.0])
        
        self.target_speed = 0.0
        self.target_heading_change = 0.0
        
        # Goal tracking
        self.goal_pos = goal_pos.copy() if goal_pos is not None else None
        self.goal_reached = False
        self.goal_tolerance = 0.5
        
    def set_command(self, target_speed: float, heading_change: float):
        self.target_speed = np.clip(target_speed, 0.0, self.max_speed)
        self.target_heading_change = np.clip(heading_change, -AGV_MAX_TURN_RATE, AGV_MAX_TURN_RATE)
        
    def update(self, dt: float, world_bounds: tuple) -> Tuple[np.ndarray, np.ndarray, float]:
        # Check if goal reached (before moving further)
        if self.goal_pos is not None and not self.goal_reached:
            dist_to_goal = np.linalg.norm(self.goal_pos - self.pos)
            if dist_to_goal < self.goal_tolerance:
                self.goal_reached = True
                self.current_speed = 0.0
                self.vel = np.array([0.0, 0.0])
                return self.pos, self.vel, self.heading
        
        # If already reached goal, stop
        if self.goal_reached:
            self.current_speed = 0.0
            self.vel = np.array([0.0, 0.0])
            return self.pos, self.vel, self.heading
        
        max_accel = 2.0 * dt
        speed_diff = self.target_speed - self.current_speed
        self.current_speed += np.clip(speed_diff, -max_accel, max_accel)
        self.current_speed = np.clip(self.current_speed, 0.0, self.max_speed)
        
        self.heading += self.target_heading_change
        
        while self.heading > np.pi:
            self.heading -= 2 * np.pi
        while self.heading < -np.pi:
            self.heading += 2 * np.pi
        
        self.vel = np.array([
            self.current_speed * np.cos(self.heading),
            self.current_speed * np.sin(self.heading)
        ])
        
        new_pos = self.pos + self.vel * dt
        
        x_min, x_max, y_min, y_max = world_bounds
        
        if new_pos[0] < x_min or new_pos[0] > x_max:
            self.heading = np.pi - self.heading
            new_pos[0] = np.clip(new_pos[0], x_min, x_max)
            self.current_speed *= 0.5
        
        if new_pos[1] < y_min or new_pos[1] > y_max:
            self.heading = -self.heading
            new_pos[1] = np.clip(new_pos[1], y_min, y_max)
            self.current_speed *= 0.5
        
        self.pos = new_pos
        return self.pos, self.vel, self.heading
    
    def get_state(self) -> dict:
        return {
            'position': self.pos.copy(),
            'velocity': self.vel.copy(),
            'heading': self.heading,
            'speed': self.current_speed,
            'max_speed': self.max_speed,
            'goal_pos': self.goal_pos.copy() if self.goal_pos is not None else None,
            'goal_reached': self.goal_reached
        }
    
    def set_position(self, pos: np.ndarray):
        self.pos = pos.copy()
    
    def set_heading(self, heading: float):
        self.heading = heading
    
    def set_goal(self, goal_pos: np.ndarray):
        """Set a new goal position."""
        self.goal_pos = goal_pos.copy()
        self.goal_reached = False


class GoalSeekingAGV:
    """
    AGV Controller that navigates toward a fixed goal position.
    """
    
    def __init__(self, start_pos: np.ndarray, goal_pos: np.ndarray, 
                 speed: float = DEFAULT_AGV_SPEED):
        self.pos = start_pos.copy()
        self.goal_pos = goal_pos.copy()
        self.max_speed = speed
        self.current_speed = speed * 0.5
        
        diff = self.goal_pos - self.pos
        self.heading = np.arctan2(diff[1], diff[0])
        self.vel = np.array([
            self.current_speed * np.cos(self.heading),
            self.current_speed * np.sin(self.heading)
        ])
        
        self.goal_reached = False
        self.goal_tolerance = 0.5
        
    def update(self, dt: float, world_bounds: tuple) -> Tuple[np.ndarray, np.ndarray, float]:
        dist_to_goal = np.linalg.norm(self.goal_pos - self.pos)
        if dist_to_goal < self.goal_tolerance:
            self.goal_reached = True
            self.current_speed = 0.0
            self.vel = np.array([0.0, 0.0])
            return self.pos, self.vel, self.heading
        
        diff = self.goal_pos - self.pos
        desired_heading = np.arctan2(diff[1], diff[0])
        
        heading_diff = desired_heading - self.heading
        while heading_diff > np.pi:
            heading_diff -= 2 * np.pi
        while heading_diff < -np.pi:
            heading_diff += 2 * np.pi
        
        max_turn = AGV_MAX_TURN_RATE * 2
        self.heading += np.clip(heading_diff, -max_turn, max_turn)
        
        if dist_to_goal < 2.0:
            self.current_speed = max(0.1, self.max_speed * (dist_to_goal / 2.0))
        else:
            self.current_speed = min(self.max_speed, self.current_speed + 0.5 * dt)
        
        self.vel = np.array([
            self.current_speed * np.cos(self.heading),
            self.current_speed * np.sin(self.heading)
        ])
        
        new_pos = self.pos + self.vel * dt
        
        x_min, x_max, y_min, y_max = world_bounds
        new_pos[0] = np.clip(new_pos[0], x_min, x_max)
        new_pos[1] = np.clip(new_pos[1], y_min, y_max)
        
        self.pos = new_pos
        return self.pos, self.vel, self.heading
    
    def get_state(self) -> dict:
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
        self.pos = pos.copy()
    
    def set_heading(self, heading: float):
        self.heading = heading
    
    def set_goal(self, goal_pos: np.ndarray):
        self.goal_pos = goal_pos.copy()
        self.goal_reached = False
