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
│       └── gapnav.py          # Gap-based + APF + Enhanced DWA
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
| `--l5_navigation` | `default`, `vfh`, `dwa`, `gapnav` | Navigation algorithm |
| `--l3_path` | `random`, `straight` | AGV path mode |
| `--l3_scenario` | `static`, `dynamic`, `mixed` | Obstacle scenario |
| `--dt` | float | Simulation time step (default: 0.1s) |
| `--steps` | int | Maximum simulation steps (default: 600) |

### Examples

```bash
# Random wandering with default navigation
python simulation.py

# Navigate from left to right with GapNav
python simulation.py --l5_navigation gapnav --l3_path straight

# VFH navigation with dynamic obstacles
python simulation.py --l5_navigation vfh --l3_scenario dynamic --l3_path straight

# DWA navigation with mixed obstacles and straight path
python simulation.py --l5_navigation dwa --l3_path straight --l3_scenario mixed
```

---

## 🎮 Scenarios

| Scenario | Description |
|----------|-------------|
| `static` | Only static obstacles (default) |
| `dynamic` | Only moving obstacles that bounce off walls |
| `mixed` | Both static and dynamic obstacles |

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

---

## ⚙️ Configuration

Each package has its own `config.py` file for easy parameter tuning:

| File | Contents |
|------|----------|
| `L3_world/config.py` | World bounds, AGV parameters, LiDAR settings, scenario configs |
| `L4_detection/config.py` | DBSCAN, EKF, tracker, classifier, HySDG-ESD parameters |
| `L5_decision/config.py` | Robot limits, navigation, VFH, DWA, GapNav parameters (unified) |

---

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    L5: Decision Layer                       │
│  Navigation Algorithms (Simple, DWA, VFH, GapNav)          │
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

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Authors

- **MILAD JAFARI BARANI** - PhD Researcher, Explainable AI & Intelligent Systems
- **Contributors** - HySDG-ESD Project Team
