# =============================================================================
# SIMULATION - Layer Coordinator
# =============================================================================
# Starts the simulation coordinating:
# - L3: World Model Layer (world, AGV, obstacles, LiDAR)
# - L4: Detection Layer (obstacle detection)
# - L5: Decision Layer (classification and decisions)
# =============================================================================

import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow
from matplotlib.widgets import Button
from collections import deque
import pandas as pd
import json
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# Import dei Layer
from L3_world import (
    WorldModel, 
    ScenarioPresets, 
    WORLD_BOUNDS,
    DEFAULT_DT
)
from L5_decision import (
    DecisionLayer,
    DWADecisionLayer,
    VFHDecisionLayer,
    GapNavDecisionLayer,
    ObstacleState,
    NavigationAction
)

# L5 Alternative Layers
L5_VARIANTS = {
    'default': 'DecisionLayer',
    'vfh': 'VFHDecisionLayer',
    'dwa': 'DWADecisionLayer',
    'gapnav': 'GapNavDecisionLayer'
}

def get_decision_layer_class(variant: str):
    """Return the decision layer class for the specified variant."""
    if variant == 'default':
        return DecisionLayer
    elif variant == 'vfh':
        return VFHDecisionLayer
    elif variant == 'dwa':
        return DWADecisionLayer
    elif variant == 'gapnav':
        return GapNavDecisionLayer
    else:
        raise ValueError(f"Unknown L5 variant: {variant}")

# =============================================================================
# Scientific Metrics (for evaluation)
# =============================================================================
class ScientificMetrics:
    """Scientific metrics calculation module for system evaluation."""
    
    def __init__(self):
        self.detection_times = {}
        self.classifications = []
        self.velocity_estimates = []
        
    def record_detection(self, obs_id: int, current_time: float, is_first: bool):
        if is_first and obs_id not in self.detection_times:
            self.detection_times[obs_id] = current_time
    
    def record_classification(self, obs_id: int, predicted: str, 
                            actual: str, current_time: float):
        self.classifications.append({
            'time': current_time,
            'obs_id': obs_id,
            'predicted': predicted,
            'actual': actual,
            'correct': predicted == actual
        })
    
    def record_velocity(self, obs_id: int, estimated: float, 
                       actual: float, current_time: float):
        self.velocity_estimates.append({
            'time': current_time,
            'obs_id': obs_id,
            'estimated': estimated,
            'actual': actual,
            'error': abs(estimated - actual)
        })
    
    def compute_metrics(self) -> dict:
        metrics = {}
        
        if self.detection_times:
            latencies = list(self.detection_times.values())
            metrics['detection_latency'] = {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies)
            }
        
        if self.classifications:
            df = pd.DataFrame(self.classifications)
            total = len(df)
            correct = df['correct'].sum()
            metrics['classification_accuracy'] = {
                'overall': correct / total if total > 0 else 0,
                'total_samples': total,
                'correct_classifications': int(correct)
            }
            
            false_static = len(df[(df['predicted'] == 'STATIC') & (df['actual'] == 'DYNAMIC')])
            false_dynamic = len(df[(df['predicted'] == 'DYNAMIC') & (df['actual'] == 'STATIC')])
            metrics['false_classifications'] = {
                'false_static': false_static,
                'false_dynamic': false_dynamic
            }
        
        if self.velocity_estimates:
            df = pd.DataFrame(self.velocity_estimates)
            metrics['velocity_estimation'] = {
                'mean_error': df['error'].mean(),
                'rmse': np.sqrt(np.mean(df['error']**2))
            }
        
        return metrics
    
    def export_to_json(self, filename: str) -> dict:
        metrics = self.compute_metrics()
        output = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        return metrics


# =============================================================================
# Simulation Controller
# =============================================================================
class SimulationController:
    """
    Main controller that coordinates the three layers.
    """
    
    def __init__(self, dt: float = DEFAULT_DT, steps: int = 600, 
                 l5_variant: str = 'default', path_mode: str = 'random'):
        self.dt = dt
        self.steps = steps
        self.current_scenario = 1
        self.l5_variant = l5_variant
        self.path_mode = path_mode
        
        # Use controlled AGV for navigation algorithms (not default)
        use_controlled = (l5_variant != 'default')
        
        # Layer 3: World Model
        self.world = WorldModel(
            dt=dt, 
            world_bounds=WORLD_BOUNDS, 
            controlled_mode=use_controlled,
            path_mode=path_mode
        )
        
        # Layer 5: Decision (includes L4 detection internally)
        DecisionLayerClass = get_decision_layer_class(l5_variant)
        self.decision = DecisionLayerClass(dt=dt)
        
        # Set goal for decision layer if in straight mode
        if path_mode == 'straight' and hasattr(self.decision, 'set_goal') and self.world.goal_pos is not None:
            self.decision.set_goal(self.world.goal_pos)
        
        mode_str = "CONTROLLED" if use_controlled else path_mode.upper()
        print(f"  L5 Variant: {l5_variant.upper()} ({DecisionLayerClass.__name__})")
        print(f"  Path Mode: {path_mode.upper()}")
        if path_mode == 'straight' and self.world.goal_pos is not None:
            print(f"  Goal: ({self.world.goal_pos[0]:.1f}, {self.world.goal_pos[1]:.1f})")
        
        # Tracking
        self.trajectories = {}
        self.agv_trajectory = []  # Full trajectory, no limit
        self.time_data = deque(maxlen=100)
        self.deq_data = {}
        self.detected_ids = set()
        
        # Goal tracking
        self.goal_reached = False
        self.goal_reached_time = None
        self.goal_reached_frame = None
        
        # Metrics
        self.obstacle_log = []
        self.scientific_metrics = ScientificMetrics()
        self.ground_truth_states = {}
        
        # Initialize scenario
        self.reset_scenario(1)
    
    def reset_scenario(self, scenario: int):
        """Resets and configures a scenario."""
        self.current_scenario = scenario
        self.trajectories.clear()
        self.agv_trajectory = []  # Reset full trajectory
        self.time_data.clear()
        self.deq_data.clear()
        self.detected_ids.clear()
        self.obstacle_log = []
        self.scientific_metrics = ScientificMetrics()
        self.ground_truth_states = {}
        
        # Reset goal tracking
        self.goal_reached = False
        self.goal_reached_time = None
        self.goal_reached_frame = None
        
        # Reset decision layer
        self.decision.reset()
        
        # Configure scenario in world model
        if scenario == 1:
            info = ScenarioPresets.scenario_static_only(self.world)
            for i in range(100):
                self.ground_truth_states[i] = 'STATIC'
        elif scenario == 2:
            info = ScenarioPresets.scenario_dynamic_only(self.world)
            for i in range(100):
                self.ground_truth_states[i] = 'DYNAMIC'
        elif scenario == 3:
            info = ScenarioPresets.scenario_mixed(self.world)
            for i in range(50):
                self.ground_truth_states[i] = 'STATIC'
            for i in range(50, 100):
                self.ground_truth_states[i] = 'DYNAMIC'
        elif scenario == 4:
            info = ScenarioPresets.scenario_empty(self.world)
            # No obstacles, no ground truth
        
        print(f"\n{'='*60}")
        print(f"Scenario {scenario} initialized")
        print(f"{'='*60}\n")
    
    def step(self, frame: int) -> dict:
        """
        Executes one simulation step.
        
        Returns:
            Dictionary with all current frame data
        """
        # If goal is already reached, return frozen state immediately
        if self.goal_reached:
            # Get current AGV position (frozen)
            agv_state = self.world.agv.get_state()
            agv_pos = agv_state['position']
            agv_heading = agv_state['heading']
            
            return {
                'frame': self.goal_reached_frame,
                'time': self.goal_reached_time,
                'agv_pos': agv_pos,
                'agv_vel': np.array([0.0, 0.0]),
                'agv_heading': agv_heading,
                'lidar_ranges': np.array([]),
                'lidar_angles': np.array([]),
                'detected_obstacles': self.decision.get_all_obstacles() if hasattr(self.decision, 'get_all_obstacles') else [],
                'nav_decision': self.decision.get_navigation_decision(agv_pos, agv_heading),
                'static_count': 0,
                'dynamic_count': 0,
                'ground_truth_obstacles': [],
                'goal_reached': True
            }
        
        # L3: Update world (only if goal not reached)
        world_state = self.world.update()
        
        agv_pos = world_state['agv_pos']
        agv_vel = world_state['agv_vel']
        agv_heading = world_state['agv_heading']
        ranges = world_state['lidar_ranges']
        angles = world_state['lidar_angles']
        
        # Track goal reached
        if world_state.get('goal_reached', False) and not self.goal_reached:
            self.goal_reached = True
            self.goal_reached_time = frame * self.dt
            self.goal_reached_frame = frame
        
        self.agv_trajectory.append(agv_pos.copy())
        
        # L4+L5: Detection and Decision
        detected = self.decision.process_scan(
            ranges, angles, agv_pos, agv_vel, agv_heading
        )
        
        # Get navigation decision
        nav_decision = self.decision.get_navigation_decision(agv_pos, agv_heading)
        
        # Apply navigation decision to AGV (only in controlled mode)
        if self.l5_variant != 'default':
            self.world.set_agv_command(
                nav_decision.target_speed,
                nav_decision.target_heading_change
            )
        
        current_time = frame * self.dt
        self.time_data.append(current_time)
        
        # Logging and metrics
        static_count = 0
        dynamic_count = 0
        
        for obs in detected:
            obs_id = obs.id
            state_str = obs.state.value
            
            # Trajectory tracking
            if obs_id not in self.trajectories:
                self.trajectories[obs_id] = deque(maxlen=40)
            self.trajectories[obs_id].append(obs.center.copy())
            
            # CSV Log
            self.obstacle_log.append({
                "time": current_time,
                "obs_id": obs.id,
                "x": obs.center[0],
                "y": obs.center[1],
                "vx": obs.velocity[0],
                "vy": obs.velocity[1],
                "v_magnitude": np.linalg.norm(obs.velocity),
                "d_eq": obs.d_eq,
                "d_dot": obs.d_dot,
                "state": state_str,
                "confidence": obs.confidence
            })
            
            # Scientific metrics
            is_new = obs_id not in self.detected_ids
            if is_new:
                self.detected_ids.add(obs_id)
            
            self.scientific_metrics.record_detection(obs_id, current_time, is_new)
            
            actual_state = self.ground_truth_states.get(obs_id, 'UNKNOWN')
            if actual_state != 'UNKNOWN':
                self.scientific_metrics.record_classification(
                    obs_id, state_str, actual_state, current_time
                )
            
            self.scientific_metrics.record_velocity(
                obs_id, np.linalg.norm(obs.velocity), 0.0, current_time
            )
            
            # Counts
            if state_str == 'STATIC':
                static_count += 1
            elif state_str == 'DYNAMIC':
                dynamic_count += 1
            
            # d_eq tracking
            if obs_id not in self.deq_data:
                self.deq_data[obs_id] = deque(maxlen=100)
            self.deq_data[obs_id].append(obs.d_eq)
        
        return {
            'frame': frame,
            'time': current_time,
            'agv_pos': agv_pos,
            'agv_vel': agv_vel,
            'agv_heading': agv_heading,
            'lidar_ranges': ranges,
            'lidar_angles': angles,
            'detected_obstacles': detected,
            'nav_decision': nav_decision,
            'static_count': static_count,
            'dynamic_count': dynamic_count,
            'ground_truth_obstacles': world_state['obstacles'],
            'goal_reached': world_state.get('goal_reached', False)
        }
    
    def save_logs(self):
        """Saves logs and metrics to files in organized subfolders."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create log directories if they don't exist
        base_log_dir = "log"
        obstacle_log_dir = os.path.join(base_log_dir, "obstacle_log")
        metrics_dir = os.path.join(base_log_dir, "scientific_metrics")
        state_dir = os.path.join(base_log_dir, "system_state")
        
        for directory in [obstacle_log_dir, metrics_dir, state_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # CSV - Obstacle Log
        if self.obstacle_log:
            df = pd.DataFrame(self.obstacle_log)
            csv_file = os.path.join(
                obstacle_log_dir, 
                f"obstacle_log_scenario_{self.current_scenario}_{timestamp}.csv"
            )
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"✅ Log saved: {csv_file}")
        
        # JSON - Scientific Metrics
        json_file = os.path.join(
            metrics_dir,
            f"scientific_metrics_scenario_{self.current_scenario}_{timestamp}.json"
        )
        metrics = self.scientific_metrics.export_to_json(json_file)
        print(f"✅ Metrics saved: {json_file}")
        
        # JSON - System State
        state_file = os.path.join(
            state_dir,
            f"system_state_scenario_{self.current_scenario}_{timestamp}.json"
        )
        state = self.decision.export_state()
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ System state saved: {state_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("METRICS SUMMARY")
        print(f"{'='*60}")
        if 'classification_accuracy' in metrics:
            acc = metrics['classification_accuracy']
            print(f"Accuracy: {acc['overall']*100:.2f}%")
        if 'velocity_estimation' in metrics:
            vel = metrics['velocity_estimation']
            print(f"Velocity RMSE: {vel['rmse']:.3f} m/s")
        print(f"{'='*60}\n")


# =============================================================================
# Visualization
# =============================================================================
class SimulationVisualizer:
    """
    Simulation visualization with matplotlib.
    """
    
    def __init__(self, controller: SimulationController):
        self.controller = controller
        self.lidar_history = deque(maxlen=30)
        self.goal_reached_drawn = False  # Track if goal reached state has been drawn
        
        # Setup figure
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25,
                                   left=0.06, right=0.97, top=0.95, bottom=0.08)
        
        self.ax_main = self.fig.add_subplot(gs[:2, :2])
        self.ax_lidar = self.fig.add_subplot(gs[0, 2], projection='polar')
        self.ax_deq = self.fig.add_subplot(gs[1, 2])
        self.ax_info = self.fig.add_subplot(gs[2, :])
        
        # Buttons
        ax_btn1 = plt.axes([0.12, 0.01, 0.15, 0.04])
        ax_btn2 = plt.axes([0.30, 0.01, 0.15, 0.04])
        ax_btn3 = plt.axes([0.48, 0.01, 0.15, 0.04])
        ax_btn4 = plt.axes([0.66, 0.01, 0.15, 0.04])
        
        self.btn1 = Button(ax_btn1, 'Scenario 1: Static', color='lightblue')
        self.btn2 = Button(ax_btn2, 'Scenario 2: Dynamic', color='lightcoral')
        self.btn3 = Button(ax_btn3, 'Scenario 3: Mixed', color='lightgreen')
        self.btn4 = Button(ax_btn4, 'Scenario 4: Empty', color='lightyellow')
        
        self.btn1.on_clicked(lambda e: self._on_scenario(1))
        self.btn2.on_clicked(lambda e: self._on_scenario(2))
        self.btn3.on_clicked(lambda e: self._on_scenario(3))
        self.btn4.on_clicked(lambda e: self._on_scenario(4))
        
        self.fig.canvas.mpl_connect('close_event', self._on_close)
    
    def _on_scenario(self, scenario: int):
        self.controller.reset_scenario(scenario)
        self.goal_reached_drawn = False  # Reset flag when changing scenario
    
    def _on_close(self, event):
        print("\n" + "="*60)
        print("SAVING LOGS AND METRICS...")
        print("="*60)
        self.controller.save_logs()
    
    def animate(self, frame: int):
        """Animation function."""
        # Execute simulation step
        data = self.controller.step(frame)
        
        agv_pos = data['agv_pos']
        agv_vel = data['agv_vel']
        agv_heading = data['agv_heading']
        detected = data['detected_obstacles']
        nav_decision = data['nav_decision']
        current_time = data['time']
        static_count = data['static_count']
        dynamic_count = data['dynamic_count']
        goal_reached = data.get('goal_reached', False)
        
        # Use frozen frame number when goal is reached
        display_frame = data['frame'] if goal_reached else frame
        
        # Only update lidar history if goal not reached
        if not goal_reached:
            self.lidar_history.append((data['lidar_ranges'], data['lidar_angles']))
        
        # === Main View ===
        self.ax_main.clear()
        self.ax_main.set_xlim(WORLD_BOUNDS[0], WORLD_BOUNDS[1])
        self.ax_main.set_ylim(WORLD_BOUNDS[2], WORLD_BOUNDS[3])
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.25, linestyle='--')
        self.ax_main.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        self.ax_main.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
        
        # AGV
        self.ax_main.add_patch(Circle(agv_pos, 0.6, fc='cyan', ec='blue', lw=3, alpha=0.9, zorder=10))
        vel_arrow = FancyArrow(agv_pos[0], agv_pos[1],
                              agv_vel[0] * 1.5, agv_vel[1] * 1.5,
                              width=0.15, head_width=0.4, head_length=0.3,
                              fc='blue', ec='darkblue', lw=2, alpha=0.8, zorder=11)
        self.ax_main.add_patch(vel_arrow)
        
        # Heading indicator
        heading_length = 1.2
        heading_x = agv_pos[0] + heading_length * np.cos(agv_heading)
        heading_y = agv_pos[1] + heading_length * np.sin(agv_heading)
        self.ax_main.plot([agv_pos[0], heading_x], [agv_pos[1], heading_y], 
                         'b-', linewidth=3, alpha=0.7, zorder=9)
        
        # AGV trajectory
        if len(self.controller.agv_trajectory) > 2:
            traj = np.array(self.controller.agv_trajectory)
            self.ax_main.plot(traj[:, 0], traj[:, 1], 'c--', alpha=0.4, linewidth=1.5)
        
        # Start and Goal markers (only in straight path mode)
        if self.controller.path_mode == 'straight':
            # Start position marker (green diamond)
            start_pos = self.controller.world.agv_start_pos
            self.ax_main.plot(start_pos[0], start_pos[1], 'gD', markersize=14, 
                             markeredgecolor='darkgreen', markeredgewidth=2, 
                             alpha=0.9, zorder=5, label='Start')
            self.ax_main.annotate('START', (start_pos[0], start_pos[1] + 0.8),
                                 ha='center', fontsize=8, fontweight='bold',
                                 color='darkgreen', zorder=15)
            
            # Goal position marker (red star or green if reached)
            if self.controller.world.goal_pos is not None:
                goal_pos = self.controller.world.goal_pos
                
                if goal_reached:
                    # Goal reached - show in green with celebration
                    self.ax_main.plot(goal_pos[0], goal_pos[1], 'g*', markersize=25,
                                     markeredgecolor='darkgreen', markeredgewidth=2,
                                     alpha=1.0, zorder=20, label='Goal Reached!')
                    self.ax_main.annotate('GOAL REACHED!', (goal_pos[0], goal_pos[1] + 1.0),
                                         ha='center', fontsize=10, fontweight='bold',
                                         color='darkgreen', zorder=20,
                                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                else:
                    # Goal not yet reached
                    self.ax_main.plot(goal_pos[0], goal_pos[1], 'r*', markersize=20,
                                     markeredgecolor='darkred', markeredgewidth=1.5,
                                     alpha=0.9, zorder=5, label='Goal')
                    self.ax_main.annotate('GOAL', (goal_pos[0], goal_pos[1] + 0.8),
                                         ha='center', fontsize=8, fontweight='bold',
                                         color='darkred', zorder=15)
                
                # Line from AGV to goal (dashed) - only if not reached
                if not goal_reached:
                    self.ax_main.plot([agv_pos[0], goal_pos[0]], [agv_pos[1], goal_pos[1]],
                                     'r:', alpha=0.3, linewidth=1.5, zorder=3)
        
        # Ground truth obstacles
        for obs_gt in data['ground_truth_obstacles']:
            center = obs_gt['center']
            radius = obs_gt.get('radius', 0.3)
            self.ax_main.add_patch(Circle(center, radius, fc='lightgray', 
                                         ec='gray', lw=1, alpha=0.3, zorder=2))
        
        # Detected obstacles
        for obs in detected:
            state_str = obs.state.value
            
            if state_str == 'STATIC':
                color, edge_color = 'limegreen', 'darkgreen'
            elif state_str == 'DYNAMIC':
                color, edge_color = 'gold', 'darkorange'
            else:
                continue
            
            self.ax_main.add_patch(Circle(obs.center, 0.32, fc=color, ec=edge_color, 
                                         lw=2.5, alpha=0.85, zorder=8))
            
            # Velocity arrow per dynamic
            if state_str == 'DYNAMIC' and np.linalg.norm(obs.velocity) > 0.08:
                vel_scale = 2.5
                arrow = FancyArrow(obs.center[0], obs.center[1],
                                  obs.velocity[0] * vel_scale, obs.velocity[1] * vel_scale,
                                  width=0.1, head_width=0.25, head_length=0.2,
                                  fc='purple', ec='indigo', lw=2, alpha=0.8, zorder=9)
                self.ax_main.add_patch(arrow)
            
            # Label
            vel_norm = np.linalg.norm(obs.velocity)
            label = f'ID:{obs.id}\n{state_str[:3]}\nv:{vel_norm:.2f}'
            self.ax_main.text(obs.center[0], obs.center[1] + 0.7, label,
                             ha='center', va='bottom', fontsize=7.5,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      edgecolor=edge_color, alpha=0.95), zorder=12)
        
        # Title con info navigazione
        action_str = nav_decision.action.value
        self.ax_main.set_title(
            f'AGV Simulation | Scenario {self.controller.current_scenario} | Frame {frame}\n'
            f'Time: {current_time:.1f}s | Action: {action_str} | '
            f'Detected: {len(detected)} (S:{static_count} D:{dynamic_count})',
            fontsize=10, fontweight='bold')
        
        # === LiDAR View ===
        if frame % 3 == 0:
            self.ax_lidar.clear()
            self.ax_lidar.set_theta_zero_location('E')
            self.ax_lidar.set_ylim(0, 16)
            self.ax_lidar.grid(True, alpha=0.4)
            self.ax_lidar.set_title('LIDAR POLAR VIEW', fontsize=11, fontweight='bold')
            
            if self.lidar_history:
                ranges, angles = self.lidar_history[-1]
                step = 3
                self.ax_lidar.scatter(angles[::step], ranges[::step],
                                     c=ranges[::step], cmap='RdYlGn_r', s=10, alpha=0.7)
                
                danger_idx = ranges < 2.0
                if np.any(danger_idx):
                    self.ax_lidar.scatter(angles[danger_idx], ranges[danger_idx],
                                         c='red', s=20, marker='o', alpha=0.9, zorder=10)
        
        # === Distance Chart ===
        # Only update if goal not reached OR it's the first draw after goal reached
        should_update_deq = frame % 5 == 0 and (not goal_reached or not self.goal_reached_drawn)
        if should_update_deq:
            if goal_reached:
                self.goal_reached_drawn = True
                
            self.ax_deq.clear()
            self.ax_deq.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
            self.ax_deq.set_ylabel('d_eq (m)', fontsize=9, fontweight='bold')
            
            # Show different title when goal is reached
            if goal_reached:
                self.ax_deq.set_title(f'Distance Metrics (FROZEN at {current_time:.1f}s)', 
                                     fontsize=10, fontweight='bold', color='green')
            else:
                self.ax_deq.set_title('Distance Metrics', fontsize=10, fontweight='bold')
            self.ax_deq.grid(True, alpha=0.35)
            
            times_list = list(self.controller.time_data)
            for obs_id, values_deque in self.controller.deq_data.items():
                values = list(values_deque)
                if values:
                    times = times_list[-len(values):]
                    self.ax_deq.plot(times, values, '-', label=f'Obs {obs_id}', linewidth=1.5)
            
            if self.controller.deq_data:
                self.ax_deq.legend(loc='upper right', fontsize=7, ncol=2)
            
            self.ax_deq.set_xlim(max(0, current_time - 10), max(10, current_time))
            self.ax_deq.axhline(y=2.0, color='r', linestyle='--', linewidth=1.5, alpha=0.6)
        
        # === Info Panel ===
        if frame % 10 == 0:
            self.ax_info.clear()
            self.ax_info.axis('off')
            self.ax_info.set_xlim(0, 1)
            self.ax_info.set_ylim(0, 1)
            
            # Get algorithm and path descriptions
            algo_descriptions = {
                'default': 'DEFAULT (Rule-based)',
                'vfh': 'VFH (Vector Field Histogram)',
                'dwa': 'DWA (Dynamic Window)',
                'gapnav': 'GAPNAV (Gap + APF + DWA)'
            }
            path_descriptions = {
                'random': 'RANDOM',
                'straight': 'STRAIGHT (L→R)'
            }
            algo_desc = algo_descriptions.get(self.controller.l5_variant, 'Unknown')
            path_desc = path_descriptions.get(self.controller.path_mode, 'Unknown')
            
            # Different title if goal reached
            if goal_reached:
                title = f'🏁 GOAL REACHED! | L5: {algo_desc} | Path: {path_desc}'
                title_color = '#4CAF50'
                title_bg = '#E8F5E9'
            else:
                title = f'SYSTEM STATUS | L5: {algo_desc} | Path: {path_desc}'
                title_color = '#C2185B'
                title_bg = '#FCE4EC'
            
            self.ax_info.text(0.5, 0.95, title, 
                             ha='center', va='top',
                             fontsize=9, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor=title_bg,
                                      edgecolor=title_color, linewidth=1.5))
            
            # Stats
            stats = self.controller.decision.get_statistics()
            
            # Goal info for straight mode
            goal_info = ""
            if self.controller.path_mode == 'straight' and self.controller.world.goal_pos is not None:
                goal = self.controller.world.goal_pos
                dist_to_goal = np.linalg.norm(goal - agv_pos)
                if goal_reached:
                    goal_info = f" | ✅ GOAL REACHED in {current_time:.1f}s"
                else:
                    goal_info = f" | Goal: ({goal[0]:.0f},{goal[1]:.0f}) Dist: {dist_to_goal:.1f}m"
            
            info_lines = [
                f"Time: {current_time:.1f}s  |  Frame: {display_frame}/{self.controller.steps}",
                f"AGV: ({agv_pos[0]:.1f}, {agv_pos[1]:.1f}) | V={np.linalg.norm(agv_vel):.2f}m/s{goal_info}",
                f"Heading: {np.rad2deg(agv_heading):.1f}°",
                f"Navigation: {nav_decision.action.value} | Safety: {nav_decision.safety_score:.2f}",
                f"Total: {stats['total_obstacles']} | Static: {stats['static_count']} | Dynamic: {stats['dynamic_count']}",
                f"Reason: {nav_decision.reason}"
            ]
            
            y_pos = 0.78
            for line in info_lines:
                self.ax_info.text(0.05, y_pos, line, ha='left', va='top',
                                 fontsize=7.5, family='monospace')
                y_pos -= 0.12
    
    def run(self):
        """Starts the animation."""
        ani = animation.FuncAnimation(
            self.fig, self.animate, 
            frames=self.controller.steps, 
            interval=100, repeat=True
        )
        plt.show()


# =============================================================================
# Argument Parser
# =============================================================================
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='simulation.py',
        description="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AGV SIMULATION - LAYERED ARCHITECTURE (HySDG-ESD-AGV-Simulator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A multi-layer AGV navigation simulator with obstacle detection,
  tracking, classification, and autonomous decision making.

  LAYERS:
    L3: World Model Layer  - Simulates world, AGV, LiDAR, obstacles
    L4: Detection Layer    - DBSCAN clustering, EKF tracking
    L5: Decision Layer     - Classification and navigation decisions

  L5 NAVIGATION ALGORITHMS (--l5_navigation):

    default  - Rule-based reactive navigation with obstacle repulsion.
               Uses inverse-distance weighting to compute repulsive forces
               from obstacles and determines evasion heading. Simple but
               effective for basic obstacle avoidance.

    vfh      - Vector Field Histogram (VFH). Builds a polar histogram of
               obstacle density around the robot and finds the best clear
               sector toward the goal. Includes wall-following recovery mode.

    dwa      - Dynamic Window Approach. Samples velocities within acceleration
               limits, predicts trajectories, and scores them based on goal
               direction, obstacle clearance, and speed. More computational
               but produces smoother paths.

    gapnav   - GapNav + APF + Enhanced DWA. State-of-the-art hybrid algorithm.
               Detects navigable gaps, uses Artificial Potential Fields for
               smooth obstacle repulsion, and enhanced DWA for trajectory
               optimization. Includes multi-layer recovery (wall-follow,
               reverse, random escape).

  PATH MODES (--l3_path):

    random   - AGV wanders randomly, changing direction every 20-80 steps.
               Bounces off world borders. Used with 'default' L5 to test
               obstacle detection and classification.

    straight - AGV navigates from left side to right side of the world
               with a fixed goal. Start: (-1, 0), Goal: (29, 0).
               Useful for testing navigation algorithms.

  SCENARIOS (--l3_scenario):

    static   - Only static obstacles (default)
    dynamic  - Only dynamic (moving) obstacles
    mixed    - Both static and dynamic obstacles
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python simulation.py                                    # Random path, default L5
  python simulation.py --l3_path straight                 # Straight path, default L5  
  python simulation.py --l5_navigation vfh                # Random path, VFH navigation
  python simulation.py --l5_navigation gapnav --l3_path straight  # GapNav with goal
  python simulation.py --l3_scenario dynamic              # Dynamic obstacles only
  python simulation.py --l5_navigation gapnav --l3_scenario mixed # GapNav + mixed scenario

NOTE: With --l5_navigation vfh/dwa/gapnav, the AGV is controlled by the
      navigation algorithm. With --l3_path straight, the AGV has a fixed
      goal on the right side of the world.

AUTHOR: HySDG-ESD Project
"""
    )
    
    parser.add_argument(
        '--l5_navigation', 
        type=str, 
        choices=['default', 'vfh', 'dwa', 'gapnav'],
        default='default',
        metavar='ALGORITHM',
        help='L5 navigation algorithm: default, vfh, dwa, gapnav (default: default)'
    )
    
    parser.add_argument(
        '--l3_path',
        type=str,
        choices=['random', 'straight'],
        default='random',
        metavar='MODE',
        help='L3 AGV path mode: random (wanders around), straight (left-to-right goal) (default: random)'
    )
    
    parser.add_argument(
        '--l3_scenario',
        type=str,
        choices=['static', 'dynamic', 'mixed'],
        default='static',
        metavar='TYPE',
        help='L3 obstacle scenario: static, dynamic, mixed (default: static)'
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        default=0.1,
        metavar='SEC',
        help='Simulation time step in seconds (default: 0.1)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=600,
        metavar='N',
        help='Maximum simulation steps (default: 600)'
    )
    
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    print("="*60)
    print("AGV SIMULATION - LAYERED ARCHITECTURE")
    print("="*60)
    print("LAYERS:")
    print("  L3: World Model Layer - World, AGV, LiDAR, Obstacles")
    print("  L4: Detection Layer   - DBSCAN, EKF, Tracking")
    print("  L5: Decision Layer    - Classification, Navigation")
    
    # Map scenario names to numbers
    scenario_map = {'static': 1, 'dynamic': 2, 'mixed': 3}
    scenario_num = scenario_map[args.l3_scenario]
    
    # Create controller with selected L5 variant and path mode
    controller = SimulationController(
        dt=args.dt, 
        steps=args.steps, 
        l5_variant=args.l5_navigation,
        path_mode=args.l3_path
    )
    
    # Set initial scenario if specified
    if scenario_num != 1:
        controller.reset_scenario(scenario_num)
    
    print("="*60)
    print("COMMANDS:")
    print("  • Buttons to change scenario")
    print("  • Close window to save logs and metrics")
    print("="*60)
    
    # Create visualizer and start
    visualizer = SimulationVisualizer(controller)
    visualizer.run()


if __name__ == "__main__":
    main()
