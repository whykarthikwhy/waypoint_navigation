"""
plot_performance.py
===================
Offline performance analysis for the waypoint_navigation package.

This script:
  1. Regenerates the exact smoothed path that PathSmoother publishes.
  2. Simulates a realistic robot trajectory using a simple kinematic
     model with the Pure Pursuit curvature law.
  3. Computes and plots:
       a) Cross-Track Error (CTE) over time — lateral deviation from path
       b) Robot path vs reference path in 2D
       c) Linear and angular velocity profiles over time
       d) Distance-to-goal over time

Usage (no ROS2 required):
    python3 plot_performance.py

Output:
    navigation_performance.png — saved in the current directory

If you recorded a rosbag, replace the simulated trajectory with
real odometry data by parsing the bag file.

Author: karburettor
"""

import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


# ─────────────────────────────────────────────────────────────
# Configuration — matches node constants
# ─────────────────────────────────────────────────────────────

WAYPOINTS = np.array([
    [0.0, 0.0],
    [2.0, 1.0],
    [4.0, -1.0],
    [6.0, 1.0],
    [8.0, 0.0],
])

SPLINE_SAMPLES = 100
TARGET_VELOCITY = 0.15    # m/s — path_smoother.py
LOOKAHEAD = 0.5           # m   — pure_pursuit_controller.py
CRUISE = 0.2              # m/s
SLOWDOWN_R = 0.8          # m
MIN_VEL = 0.05            # m/s
GOAL_THRESHOLD = 0.05     # m
DT = 0.033                # s  (~30 Hz, matches TurtleBot3 odom rate)


# ─────────────────────────────────────────────────────────────
# Step 1: Generate reference path (mirrors PathSmoother)
# ─────────────────────────────────────────────────────────────

def generate_reference_path():
    """Reproduce the cubic spline from path_smoother.py."""
    t_knots = np.linspace(0, 1, len(WAYPOINTS))
    cs_x = CubicSpline(t_knots, WAYPOINTS[:, 0])
    cs_y = CubicSpline(t_knots, WAYPOINTS[:, 1])
    t_fine = np.linspace(0, 1, SPLINE_SAMPLES)
    return cs_x(t_fine), cs_y(t_fine)


# ─────────────────────────────────────────────────────────────
# Step 2: Simulate robot (kinematic bicycle model + Pure Pursuit)
# ─────────────────────────────────────────────────────────────

def get_lookahead_point(rx, ry, path_xs, path_ys, dist_to_goal):
    """Find the carrot point — mirrors PurePursuitController._get_lookahead_point()."""
    current_la = max(0.1, min(LOOKAHEAD, dist_to_goal))
    best_pt = None
    min_diff = float('inf')
    for px, py in zip(path_xs, path_ys):
        d = math.sqrt((px - rx)**2 + (py - ry)**2)
        diff = abs(d - current_la)
        if diff < min_diff:
            min_diff = diff
            best_pt = (px, py)
    return best_pt or (path_xs[-1], path_ys[-1])


def velocity_ramp(dist):
    """Mirrors PurePursuitController._compute_velocity_ramp()."""
    if dist < SLOWDOWN_R:
        return max(MIN_VEL, CRUISE * (dist / SLOWDOWN_R))
    return CRUISE


def simulate_robot(path_xs, path_ys):
    """
    Simulate the robot following the path using a discrete kinematic model.

    State: (x, y, yaw)
    Update:
        x   += v * cos(yaw) * dt
        y   += v * sin(yaw) * dt
        yaw += omega * dt
    """
    # Initial state
    rx, ry, ryaw = 0.0, 0.0, 0.0

    sim_xs, sim_ys, sim_ts = [rx], [ry], [0.0]
    linears, angulars = [0.0], [0.0]
    t = 0.0

    goal_x, goal_y = path_xs[-1], path_ys[-1]
    max_steps = 5000  # safety limit

    for _ in range(max_steps):
        dist_to_goal = math.sqrt((goal_x - rx)**2 + (goal_y - ry)**2)
        if dist_to_goal < GOAL_THRESHOLD:
            break

        tx, ty = get_lookahead_point(rx, ry, path_xs, path_ys, dist_to_goal)
        dx, dy = tx - rx, ty - ry

        # Transform to robot frame
        lx = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        ly = -dx * math.sin(ryaw) + dy * math.cos(ryaw)

        L_sq = lx**2 + ly**2
        kappa = (2.0 * ly) / L_sq if L_sq > 0.001 else 0.0

        v = velocity_ramp(dist_to_goal)
        omega = kappa * v

        # Kinematic update
        rx += v * math.cos(ryaw) * DT
        ry += v * math.sin(ryaw) * DT
        ryaw += omega * DT
        ryaw = math.atan2(math.sin(ryaw), math.cos(ryaw))  # wrap
        t += DT

        sim_xs.append(rx)
        sim_ys.append(ry)
        sim_ts.append(t)
        linears.append(v)
        angulars.append(omega)

    return (np.array(sim_xs), np.array(sim_ys), np.array(sim_ts),
            np.array(linears), np.array(angulars))


# ─────────────────────────────────────────────────────────────
# Step 3: Compute cross-track error
# ─────────────────────────────────────────────────────────────

def compute_cte(sim_xs, sim_ys, path_xs, path_ys):
    """
    For each simulated robot position, find the closest point on the
    reference path and return the perpendicular distance (CTE).
    """
    ctes = []
    for rx, ry in zip(sim_xs, sim_ys):
        dists = [math.sqrt((rx - px)**2 + (ry - py)**2)
                 for px, py in zip(path_xs, path_ys)]
        ctes.append(min(dists))
    return np.array(ctes)


# ─────────────────────────────────────────────────────────────
# Step 4: Plot everything
# ─────────────────────────────────────────────────────────────

def plot_results(path_xs, path_ys, sim_xs, sim_ys, sim_ts,
                 linears, angulars, ctes):

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Waypoint Navigation — Performance Analysis\n'
        'Pure Pursuit Controller + Cubic Spline Path Smoother',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Plot 1: 2D Path Comparison ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(path_xs, path_ys, 'b--', linewidth=2, label='Reference (spline)', alpha=0.7)
    ax1.plot(sim_xs, sim_ys, 'r-', linewidth=2, label='Robot trajectory', alpha=0.9)
    ax1.scatter(WAYPOINTS[:, 0], WAYPOINTS[:, 1], c='orange', s=100, zorder=5,
                label='Raw waypoints', edgecolors='black')
    ax1.scatter([sim_xs[0]], [sim_ys[0]], c='green', s=120, zorder=6,
                label='Start', marker='^', edgecolors='black')
    ax1.scatter([sim_xs[-1]], [sim_ys[-1]], c='red', s=120, zorder=6,
                label='End', marker='*', edgecolors='black')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('2D Path: Reference vs Robot Trajectory')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # ── Plot 2: Cross-Track Error ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(sim_ts, ctes * 100, 'purple', linewidth=1.5)
    ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.6, label='5 cm threshold')
    ax2.fill_between(sim_ts, ctes * 100, alpha=0.15, color='purple')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('CTE (cm)')
    ax2.set_title('Cross-Track Error over Time')
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    mean_cte = np.mean(ctes) * 100
    max_cte = np.max(ctes) * 100
    ax2.text(0.98, 0.95, f'Mean: {mean_cte:.1f} cm\nMax: {max_cte:.1f} cm',
             transform=ax2.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    # ── Plot 3: Linear Velocity Profile ──────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sim_ts, linears, 'green', linewidth=1.5)
    ax3.axhline(y=CRUISE, color='gray', linestyle='--', alpha=0.5, label=f'Cruise ({CRUISE} m/s)')
    ax3.axhline(y=MIN_VEL, color='orange', linestyle=':', alpha=0.5, label=f'Min ({MIN_VEL} m/s)')
    ax3.fill_between(sim_ts, linears, alpha=0.15, color='green')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Linear Velocity (m/s)')
    ax3.set_title('Linear Velocity Profile')
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Angular Velocity Profile ─────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(sim_ts, np.degrees(angulars), 'darkorange', linewidth=1.5)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax4.fill_between(sim_ts, np.degrees(angulars), alpha=0.15, color='darkorange')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Velocity (°/s)')
    ax4.set_title('Angular Velocity Profile')
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Distance-to-Goal ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    goal_x, goal_y = path_xs[-1], path_ys[-1]
    dist_to_goal = [math.sqrt((goal_x - x)**2 + (goal_y - y)**2)
                    for x, y in zip(sim_xs, sim_ys)]
    ax5.plot(sim_ts, dist_to_goal, 'steelblue', linewidth=1.5)
    ax5.axhline(y=GOAL_THRESHOLD, color='red', linestyle='--', alpha=0.6,
                label=f'Goal threshold ({GOAL_THRESHOLD*100:.0f} cm)')
    ax5.fill_between(sim_ts, dist_to_goal, alpha=0.15, color='steelblue')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Distance to Goal (m)')
    ax5.set_title('Distance to Goal over Time')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.savefig('navigation_performance.png',
                dpi=150, bbox_inches='tight')
    print('Saved: navigation_performance.png')

    # Print summary stats
    print('\n── Performance Summary ─────────────────────────')
    print(f'  Total simulation time : {sim_ts[-1]:.1f} s')
    print(f'  Mean cross-track error: {np.mean(ctes)*100:.2f} cm')
    print(f'  Max  cross-track error: {np.max(ctes)*100:.2f} cm')
    print(f'  Final dist to goal    : {dist_to_goal[-1]*100:.2f} cm')
    print(f'  Path points simulated : {len(sim_xs)}')
    print('────────────────────────────────────────────────')


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating reference path...')
    path_xs, path_ys = generate_reference_path()

    print('Simulating robot...')
    sim_xs, sim_ys, sim_ts, linears, angulars = simulate_robot(path_xs, path_ys)

    print('Computing cross-track error...')
    ctes = compute_cte(sim_xs, sim_ys, path_xs, path_ys)

    print('Plotting...')
    plot_results(path_xs, path_ys, sim_xs, sim_ys, sim_ts, linears, angulars, ctes)
