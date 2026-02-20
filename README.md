# Waypoint Navigation — ROS2 / TurtleBot3

A ROS2 Python package implementing **cubic spline path smoothing**, **time-parameterized trajectory generation**, and a **Pure Pursuit tracking controller** for a TurtleBot3 Burger differential drive robot in Gazebo.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Algorithm Design Choices](#algorithm-design-choices)
3. [Setup & Installation](#setup--installation)
4. [Running the System](#running-the-system)
5. [Testing](#testing)
6. [Performance Analysis](#performance-analysis)
7. [Extending to a Real Robot](#extending-to-a-real-robot)
8. [Extra Credit: Obstacle Avoidance](#extra-credit-obstacle-avoidance)
9. [AI Tools Used](#ai-tools-used)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        waypoint_navigation                        │
│                                                                  │
│   ┌─────────────────┐   smoothed_path    ┌───────────────────┐  │
│   │  path_smoother  │ ──────────────────► │ pure_pursuit_     │  │
│   │                 │   (nav_msgs/Path)   │ controller        │  │
│   │  • Cubic Spline │                    │                   │  │
│   │  • Time param.  │   waypoint_markers  │  • State machine  │  │
│   │  • 1 Hz publish │ ──────────────────► │  • Adaptive LA    │  │
│   └─────────────────┘   (MarkerArray)    │  • Vel ramp       │  │
│                                          └────────┬──────────┘  │
│                                                   │ /cmd_vel     │
│   ┌─────────────────┐   /odom                     ▼             │
│   │  monitor_node   │ ◄──────── TurtleBot3 ◄──────────────────  │
│   │  (diagnostic)   │   /scan                                   │
│   └─────────────────┘                                           │
└──────────────────────────────────────────────────────────────────┘
```

**Node summary:**

| Node | File | Publishes | Subscribes |
|---|---|---|---|
| `path_smoother` | `path_smoother.py` | `smoothed_path`, `waypoint_markers` | — |
| `pure_pursuit_controller` | `pure_pursuit_controller.py` | `/cmd_vel` | `smoothed_path`, `/odom`, `/scan` |
| `monitor_node` | `monitor_node.py` | — | `/odom`, `/cmd_vel` |

---

## Algorithm Design Choices

### Task 1 — Path Smoothing: Cubic Spline

**Algorithm:** `scipy.interpolate.CubicSpline`

A cubic spline fits a piecewise-cubic polynomial through every waypoint, guaranteeing **C2 continuity** — meaning position, velocity, *and* acceleration are all continuous. This is important for robot motion because:
- Discontinuous acceleration causes jerky motion and wheel slip.
- The controller receives smooth curvature inputs, reducing oscillation.

**Why not Bézier curves?** Bézier curves offer shape control but don't pass through intermediate control points. Our waypoints come from a global planner and must be exactly visited — cubic splines enforce this by construction.

**Why not B-splines?** B-splines approximate rather than interpolate. For this task, exact interpolation is required.

**Parameterisation:** A uniform parameter `t ∈ [0, 1]` with one knot per waypoint is used. Chord-length parameterisation would be marginally more accurate for highly non-uniform waypoint spacing but is unnecessary here.

---

### Task 2 — Trajectory Generation: Constant-Velocity Time Parameterisation

Each sampled path point is assigned a timestamp using:

```
t_i = Σ(d_k) / v_target   for k = 1..i
```

where `d_k` is the arc-length of each segment and `v_target = 0.15 m/s`.

**Why constant velocity?** Simple, predictable, and sufficient for a TurtleBot3 in an uncluttered environment. Each `PoseStamped.header.stamp` in the published `Path` encodes the time the robot *should* arrive at that point — fulfilling the `[(x, y, t)]` trajectory format.

**Upgrade path for real robots:** Replace constant velocity with a **trapezoidal velocity profile**:
- Ramp up during the first N metres (respects max acceleration).
- Cruise at max speed.
- Ramp down before the goal (prevents overshoot).

---

### Task 3 — Trajectory Tracking: Pure Pursuit Controller

**Algorithm:** Pure Pursuit with adaptive lookahead + velocity ramping.

Pure Pursuit is a classic geometric tracking algorithm:

1. Project a 'lookahead point' (the *carrot*) at distance `L` ahead on the path.
2. Compute the curvature κ of the circular arc connecting the robot to the carrot:
   ```
   κ = 2 * local_y / (local_x² + local_y²)
   ```
3. Set angular velocity: `ω = κ · v`

**State machine:**

```
ALIGNING ──(angle error < 5.7°)──► TRACKING ──(dist < 5 cm)──► REACHED
```

The robot first rotates in-place (`ALIGNING`) to avoid the controller issuing large forward commands while severely misaligned, which would cause wide arcs away from the path.

**Adaptive lookahead:** `L = min(L_default, dist_to_goal)`. Prevents the carrot from jumping past the goal on the final approach — without this, the robot circles near the endpoint.

**Velocity ramp:** Linear decrease from cruise speed to `MIN_VEL` within `0.8 m` of the goal. Prevents overshoot from inertia.

**Why Pure Pursuit and not PID?** PID on heading error becomes unstable at high curvatures (sharp turns). Pure Pursuit uses geometry rather than error feedback, making it inherently stable on curved paths and requiring only one tuning parameter (lookahead distance).

---

## Setup & Installation

### Prerequisites

| Dependency | Version |
|---|---|
| Ubuntu | 22.04 |
| ROS2 | Humble |
| TurtleBot3 packages | `ros-humble-turtlebot3*` |
| Python | 3.10+ |
| scipy | `pip install scipy` |

### Install

```bash
# 1. Clone into your ROS2 workspace
cd ~/ros2_ws/src
git clone <repo-url> waypoint_navigation

# 2. Install Python dependencies
pip install scipy numpy

# 3. Set TurtleBot3 model
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc

# 4. Build
cd ~/ros2_ws
colcon build --packages-select waypoint_navigation
source install/setup.bash
```

---

## Running the System

### Full launch (Gazebo + RViz + all nodes)

```bash
ros2 launch waypoint_navigation navigation_launch.py
```

This starts:
- Gazebo with an empty world and TurtleBot3 Burger
- RViz2 with the pre-configured `navigation.rviz` layout
- `path_smoother` node
- `pure_pursuit_controller` node

### Optional: diagnostic terminal monitor

```bash
# In a separate terminal
source ~/ros2_ws/install/setup.bash
ros2 run waypoint_navigation monitor_node
```

### Optional: performance plots (no ROS2 needed)

```bash
python3 plot_performance.py
# Saves: navigation_performance.png
```

### RViz topics to add

| Topic | Type | Purpose |
|---|---|---|
| `/smoothed_path` | `nav_msgs/Path` | Visualise the reference path |
| `/waypoint_markers` | `visualization_msgs/MarkerArray` | Red spheres at each waypoint |
| `/odom` | `nav_msgs/Odometry` | Robot pose |

---

## Testing

Tests require only `pytest`, `numpy`, and `scipy` — no ROS2 needed.

```bash
pip install pytest numpy scipy
pytest test_waypoint_navigation.py -v
```

### Test coverage

| Test Class | What is tested |
|---|---|
| `TestPathSmoothing` (7 tests) | Spline endpoints, continuity, interpolation, edge cases |
| `TestTrajectoryGeneration` (7 tests) | Time monotonicity, arc-length correctness, error handling |
| `TestAngleHandling` (4 tests) | Angle wrapping at ±π, quaternion yaw extraction |
| `TestPurePursuitCurvature` (6 tests) | κ formula, sign conventions, division-by-zero safety |
| `TestVelocityRamping` (5 tests) | Ramp profile shape, min-velocity floor, linear interpolation |
| `TestObstacleDetection` (8 tests) | Arc parsing, range filtering, front/back discrimination |
| `TestGoalDetection` (4 tests) | Threshold boundary conditions |
| `TestFullPipeline` (2 tests) | End-to-end: waypoints → curvature command |

**43 tests total.**

---

## Performance Analysis

Run `plot_performance.py` to generate `navigation_performance.png`, which shows:

- **2D path overlay** — reference spline vs simulated robot trajectory
- **Cross-Track Error (CTE)** — lateral deviation from path over time
- **Linear velocity profile** — shows ramp-down near goal
- **Angular velocity profile** — shows turning commands at each curve
- **Distance to goal** — convergence curve

Typical results (simulated, 0.2 m/s cruise):

| Metric | Value |
|---|---|
| Mean CTE | ~6 cm |
| Max CTE | ~23 cm (at sharp curve transitions) |
| Final distance to goal | < 5 cm |
| Total mission time | ~65 s |

---

## Extending to a Real Robot

Deploying this system on a physical TurtleBot3 requires the following changes:

**1. Sensor calibration**
- The laser scan's angular offset and min/max range must match the physical sensor's datasheet.
- Odometry drift accumulates over long distances — integrate a wheel encoder + IMU fusion node (e.g. `robot_localization` with an EKF).

**2. Velocity profile**
- Replace constant velocity with a trapezoidal profile that respects the robot's measured acceleration limits (typically ~0.1 m/s² for TurtleBot3 Burger on carpet).
- Add a current-sensing safety cutoff to stop if wheel stall is detected.

**3. Coordinate frames**
- In simulation, `odom` and `map` frames are identical. On a real robot, add SLAM (e.g. Nav2 + SLAM Toolbox) to maintain a `map → odom → base_link` TF tree.

**4. Lookahead tuning**
- The 0.5 m default lookahead was tuned in simulation. On real hardware, re-tune on the actual surface — carpet vs tile changes effective wheel radius and thus slip characteristics.

**5. Obstacle avoidance**
- Upgrade the reactive nudge to a Dynamic Window Approach (DWA) or a potential field planner for reliable real-world avoidance.

**6. Communication latency**
- Add a watchdog timer that stops the robot if no `/odom` message is received within 200 ms (handles dropped USB connections, etc.).

---

## Extra Credit: Obstacle Avoidance

The `PurePursuitController` includes a reactive obstacle avoidance layer:

**Detection:** The `_scan_callback` monitors a ±30° front arc. Any laser return within 0.7 m sets `obstacle_detected = True`.

**Response:** When an obstacle is detected, a lateral nudge (`+0.4 m`) is added to `local_y` before the curvature calculation. This biases the controller to steer left, around the obstacle.

**Limitation:** This is a simple reactive strategy. It works for single obstacles in open space but will fail in narrow corridors. A production system would use:
- Dynamic Window Approach (DWA) for kinodynamic feasibility
- Potential fields for smooth repulsion
- Nav2's local costmap for full collision avoidance

---

## AI Tools Used

**Claude (Anthropic)** was used throughout development for:
- Initial code scaffolding of the ROS2 node structure
- Debugging subscription/publisher mismatches
- Generating docstrings and inline comments
- Writing unit tests and the performance plot script
- Drafting this README

**Workflow:** All AI-generated code was reviewed, understood, and tested before inclusion. The core algorithm choices (cubic spline, pure pursuit, adaptive lookahead) were made independently and then implemented with AI assistance for boilerplate.

---

## File Structure

```
waypoint_navigation/
├── waypoint_navigation/
│   ├── __init__.py
│   ├── path_smoother.py           # Task 1 & 2: spline + trajectory
│   ├── pure_pursuit_controller.py # Task 3: tracking controller
│   └── monitor_node.py            # Diagnostic logger
├── launch/
│   └── navigation_launch.py       # Full system launch
├── rviz/
│   └── navigation.rviz            # Pre-configured RViz layout
├── test/
│   └── test_waypoint_navigation.py # 43 unit tests (no ROS2 needed)
├── plot_performance.py            # Offline performance analysis
├── package.xml
├── setup.py
└── README.md
```
