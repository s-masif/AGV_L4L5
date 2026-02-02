# HySDG-ESD AGV Simulator

A multi-layer AGV (Autonomous Guided Vehicle) navigation simulator with LiDAR-based obstacle detection, tracking, classification, and autonomous decision making.

---

## ✨ Features

- **Modular Package Architecture**: Clean separation between World Model (`L3_world`), Detection (`L4_detection`), and Decision (`L5_decision`) packages
- **Multiple Navigation Algorithms**: 
  - Simple reactive navigation (repulsive field)
  - VFH (Vector Field Histogram)
  - DWA (Dynamic Window Approach)
  - GapNav + APF (Gap Navigation with Artificial Potential Fields)
  - VO (Velocity Obstacles with TTC for dynamic obstacles, static/dynamic-aware, conservative safety logic)
    - Distinguishes static vs dynamic obstacles for safety margins
    - Emergency reverse and stop logic for critical threats
    - Allows closer approach to static obstacles, more conservative for dynamic
    - Ignores obstacles moving away, goes straight to goal if clear
- **Realistic Visualization**: AGV and obstacles are drawn with true physical radii and safety margins
- **Status Feedback**: Colored status box (safe, warning, danger, collision) in UI info panel (no emojis)
- **Unified Configuration**: All VO and navigation parameters are in `L5_decision/config.py` for easy tuning
- **2D LiDAR Simulation**: Configurable noise, range, and field of view
- **DBSCAN Clustering**: Real-time obstacle clustering from point clouds
- **EKF Tracking**: Extended Kalman Filter for multi-object tracking
- **HySDG-ESD Classification**: Dynamic vs static obstacle classification with ego-motion compensation
- **Unified Configuration**: Each package has its own `config.py` for easy tuning
- **Interactive Visualization**: Real-time plots with scenario switching

---

## 📂 Project Structure

```
HySDG-ESD-AGV-Simulator/
├── simulation.py              # Main entry point and visualization
│
├── L3_world/                  # World Model Package
│   ├── __init__.py            # Package exports
│   ├── config.py              # L3 configuration parameters
│   ├── lidar.py               # LiDAR simulator
│   ├── agv.py                 # AGV controllers (Random, Controlled, GoalSeeking)
│   ├── obstacles.py           # Obstacle generator
│   └── world.py               # WorldModel, ScenarioPresets
│
├── L4_detection/              # Detection & Recognition Package
│   ├── __init__.py            # Package exports
│   ├── config.py              # L4 configuration (EKF, DBSCAN, Tracker, Classifier)
│   ├── types.py               # ObstacleState, LidarPoint, TrackedObstacle
│   ├── transforms.py          # Coordinate transformations
│   ├── kalman.py              # Extended Kalman Filter (EKF-CV)
│   ├── lidar.py               # LiDAR processor with DBSCAN clustering
│   ├── classifier.py          # HySDG-ESD calculator, ObstacleClassifier
│   └── tracker.py             # Multi-object tracker, DetectionLayer
│
├── L5_decision/               # Decision & Navigation Package
│   ├── __init__.py            # Package exports
│   ├── config.py              # L5 configuration (all algorithms unified)
│   ├── types.py               # NavigationAction, NavigationDecision, etc.
│   ├── base.py                # BaseDecisionMaker (shared methods)
│   ├── layer.py               # DecisionLayer, DWADecisionLayer, etc.
│   └── algorithms/            # Navigation algorithms
│       ├── __init__.py
│       ├── simple.py          # Simple repulsive field navigation
│       ├── dwa.py             # Dynamic Window Approach
│       ├── vfh.py             # Vector Field Histogram
│       ├── gapnav.py          # Gap-based + APF + Enhanced DWA
│       └── velocity_obstacles.py  # Velocity Obstacles (TTC-based)
│
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## ▶️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/HySDG-ESD-AGV-Simulator.git
cd HySDG-ESD-AGV-Simulator

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Basic Usage

```bash
# Run with default settings (random path, default navigation)
python simulation.py

# Show all options
python simulation.py --help
```

### Command Line Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--l5_navigation` | `default`, `vfh`, `dwa`, `gapnav`, `vo` | Navigation algorithm |
| `--l3_path` | `random`, `straight` | AGV path mode |
| `--l3_scenario` | `static`, `dynamic`, `mixed` | Obstacle scenario (default: mixed) |
| `--l3_obstacles` | int | Total number of obstacles (overrides scenario defaults). For `mixed`: 1/3 dynamic, 2/3 static |
| `--speed` | `normal`, `fast`, `very_fast` | Animation speed (1x, 10x, 100x) |
| `--dt` | float | Simulation time step (default: 0.1s) |
| `--steps` | int | Maximum simulation steps (default: 600) |

### Examples

```bash
# Random wandering with default navigation (mixed obstacles by default)
python simulation.py

# VFH navigation with dynamic obstacles only
python simulation.py --l5_navigation vfh --l3_scenario dynamic --l3_path straight

# DWA navigation with static obstacles only
python simulation.py --l5_navigation dwa --l3_path straight --l3_scenario static

# Navigate from left to right with GapNav (mixed obstacles by default)
python simulation.py --l5_navigation gapnav --l3_path straight

# VO (Velocity Obstacles) - best for dynamic obstacle scenarios
python simulation.py --l5_navigation vo --l3_scenario dynamic --l3_path straight

# Empty scenario (no obstacles) with fast animation
python simulation.py --l3_path straight --l5_navigation vo --l3_obstacles 0 --speed fast

# Custom 15 obstacles with mixed scenario (5 dynamic + 10 static)
python simulation.py --l3_path straight --l5_navigation vo --l3_obstacles 15

# Custom 12 dynamic obstacles only
python simulation.py --l5_navigation vo --l3_path straight --l3_scenario dynamic --l3_obstacles 12
```

---

## 🎮 Scenarios

| Scenario | Description |
|----------|-------------|
| `static` | Only static obstacles |
| `dynamic` | Only moving obstacles that bounce off walls |
| `mixed` | Both static and dynamic obstacles (default). With `--l3_obstacles N`: 1/3 dynamic (ceiling), 2/3 static |

---

## 🧠 Navigation Algorithms

### Simple (Default)
Rule-based reactive navigation with inverse-distance obstacle repulsion. Simple but effective for basic obstacle avoidance.

### VFH (Vector Field Histogram)
Builds a polar histogram of obstacle density and finds the best clear sector toward the goal. Includes wall-following recovery mode.

### DWA (Dynamic Window Approach)
Samples velocities within acceleration limits, predicts trajectories, and scores them based on goal direction, obstacle clearance, and speed.

### GapNav + APF
State-of-the-art hybrid algorithm. Detects navigable gaps, uses Artificial Potential Fields for smooth obstacle repulsion, and enhanced DWA for trajectory optimization. Includes multi-layer recovery (wall-follow, reverse, random escape).

### Velocity Obstacles (VO)
Standalone algorithm specialized for **dynamic obstacle avoidance**. Uses:
- **Time-To-Collision (TTC)**: Predicts when collision will occur with each moving obstacle
- **Velocity Obstacles**: Computes forbidden velocity regions (cones)
- **Avoidance strategies** (in order of preference):
  1. **Pass behind**: Slow down to let obstacle pass (preferred)
  2. **Slow down**: Reduce speed when approaching collision zone
  3. **Emergency stop**: Immediate stop for critical threats (TTC < 2s)
  4. **Pass front**: Accelerate past slow-moving obstacles (if safe)

#### 2026+ Improvements
- **Static vs Dynamic**: Safety thresholds are reduced for static obstacles, allowing closer approach; dynamic obstacles use full conservative margins
- **Emergency Reverse**: If too close, AGV reverses away from obstacle
- **Warning/Danger Zones**: UI shows colored box for safe, warning, danger, or collision state
- **Ignores obstacles moving away**: AGV proceeds directly to goal if no collision threat
- **All parameters in config.py**: Easy to tune VO and safety logic
- **No emojis in UI**: Status is professional and clear

Best for scenarios with dynamic/moving obstacles. Use with `--l3_scenario dynamic`.

---

## ⚙️ Configuration

Each package has its own `config.py` file for easy parameter tuning:

| File | Contents |
|------|----------|
| `L3_world/config.py` | World bounds, AGV parameters, LiDAR settings, scenario configs |
| `L4_detection/config.py` | DBSCAN, EKF, tracker, classifier, HySDG-ESD parameters |
| `L5_decision/config.py` | Robot limits, navigation, VFH, DWA, GapNav, VO parameters (unified) |

---

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    L5: Decision Layer                       │
│  Navigation Algorithms (Simple, DWA, VFH, GapNav, VO)      │
│  Path Planning, Obstacle Avoidance, Recovery Behaviors     │
├─────────────────────────────────────────────────────────────┤
│                    L4: Detection Layer                      │
│  DBSCAN Clustering, EKF Tracking, HySDG-ESD Classification │
│  Obstacle Recognition (Static/Dynamic/Unknown)             │
├─────────────────────────────────────────────────────────────┤
│                  L3: World Model Layer                      │
│  World Simulation, AGV Controllers, LiDAR, Obstacles       │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Package | Responsibility |
|-------|---------|----------------|
| **L3** | `L3_world` | World simulation, AGV physics, LiDAR emulation, obstacle management |
| **L4** | `L4_detection` | Obstacle detection, tracking, classification (DBSCAN, EKF, HySDG-ESD) |
| **L5** | `L5_decision` | Navigation decisions, path planning, algorithm execution |

### Python API

```python
# Import packages
from L3_world import WorldModel, ScenarioPresets
from L4_detection import DetectionLayer, ObstacleState
from L5_decision import DWADecisionLayer, GapNavDecisionLayer

# Create world
world = WorldModel(dt=0.1, controlled_mode=True, path_mode='straight')
ScenarioPresets.scenario_mixed(world)

# Create decision layer
decision_layer = GapNavDecisionLayer(dt=0.1)
decision_layer.set_goal(np.array([29.0, 0.0]))

# Simulation loop
state = world.update()
obstacles = decision_layer.process_scan(
    state['lidar_ranges'], state['lidar_angles'],
    state['agv_pos'], state['agv_vel'], state['agv_heading']
)
decision = decision_layer.get_navigation_decision(state['agv_pos'], state['agv_heading'])
```

---

## 📊 Output

The simulator generates logs in the `log/` directory:
- `obstacle_log_*.csv` - Obstacle detection history
- `scientific_metrics_*.json` - Classification accuracy metrics
- `system_state_*.json` - Complete system state snapshots

---

## 🆕 2026+ Major Updates

- **VO logic**: Now distinguishes static/dynamic obstacles, with conservative safety, emergency reverse, and warning/danger logic
- **Visualization**: AGV and obstacles drawn with real radii and safety margins
- **Status UI**: Colored box for safe/warning/danger/collision, no emojis
- **Config**: All VO/navigation parameters in `L5_decision/config.py`

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Authors

- **MILAD JAFARI BARANI** - PhD Researcher, Explainable AI & Intelligent Systems
- **Contributors** - HySDG-ESD Project Team
