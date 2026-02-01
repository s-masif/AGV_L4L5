# HySDG-ESD AGV Simulator

A multi-layer AGV (Autonomous Guided Vehicle) navigation simulator with LiDAR-based obstacle detection, tracking, classification, and autonomous decision making.

---

## ✨ Features

- **Layered Architecture**: Clean separation between World Model (L3), Detection (L4), and Decision (L5) layers
- **Multiple Navigation Algorithms**: 
  - Default reactive navigation
  - VFH (Vector Field Histogram)
  - DWA (Dynamic Window Approach)
  - GapNav + APF (Gap Navigation with Artificial Potential Fields)
- **2D LiDAR Simulation**: Configurable noise, range, and field of view
- **DBSCAN Clustering**: Real-time obstacle clustering from point clouds
- **EKF Tracking**: Extended Kalman Filter for multi-object tracking
- **HySDG-ESD Classification**: Dynamic vs static obstacle classification with ego-motion compensation
- **Interactive Visualization**: Real-time plots with scenario switching

---

## 📂 Project Structure

```
HySDG-ESD-AGV-Simulator/
├── simulation.py              # Main entry point and visualization
├── L3_world_model_layer.py    # World, AGV, LiDAR, obstacles simulation
├── L4_detection_layer.py      # DBSCAN clustering, EKF tracking
├── L5_decision_layer.py       # Default navigation and classification
├── L5_decision_vfh.py         # VFH navigation algorithm
├── L5_decision_dwa.py         # DWA navigation algorithm
├── L5_decision_gapnav.py      # GapNav + APF navigation algorithm
├── config_L3.py               # L3 configuration parameters
├── config_L4.py               # L4 configuration parameters
├── config_L5.py               # L5 configuration parameters
├── config_L5_alternatives.py  # Alternative algorithms configuration
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

### Default
Rule-based reactive navigation with inverse-distance obstacle repulsion. Simple but effective for basic obstacle avoidance.

### VFH (Vector Field Histogram)
Builds a polar histogram of obstacle density and finds the best clear sector toward the goal. Includes wall-following recovery mode.

### DWA (Dynamic Window Approach)
Samples velocities within acceleration limits, predicts trajectories, and scores them based on goal direction, obstacle clearance, and speed.

### GapNav + APF
State-of-the-art hybrid algorithm. Detects navigable gaps, uses Artificial Potential Fields for smooth obstacle repulsion, and enhanced DWA for trajectory optimization. Includes multi-layer recovery (wall-follow, reverse, random escape).

---

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    L5: Decision Layer                       │
│  Classification, Navigation Decisions, Obstacle Avoidance  │
├─────────────────────────────────────────────────────────────┤
│                    L4: Detection Layer                      │
│      DBSCAN Clustering, EKF Tracking, State Estimation     │
├─────────────────────────────────────────────────────────────┤
│                  L3: World Model Layer                      │
│        World Simulation, AGV, LiDAR, Obstacles              │
└─────────────────────────────────────────────────────────────┘
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
