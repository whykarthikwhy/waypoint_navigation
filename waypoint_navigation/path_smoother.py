"""
path_smoother.py
================
ROS2 Node: PathSmoother

Responsibilities:
  1. Accept a hardcoded list of 2D waypoints (Task 1).
  2. Fit a Cubic Spline through them to produce a smooth, continuous path.
  3. Time-parameterize the path at a constant velocity to yield a
     trajectory of the form [(x, y, t), ...] (Task 2).
  4. Publish the trajectory as a nav_msgs/Path on 'smoothed_path'.
  5. Publish sphere markers for each raw waypoint on 'waypoint_markers'
     for RViz visualisation.

Design Choices:
  - Cubic Spline (scipy.interpolate.CubicSpline): Guarantees C2 continuity
    (continuous position, velocity, and acceleration), which is ideal for
    smooth robot motion. Alternative: Bezier curves are simpler but don't
    guarantee passage through all waypoints.
  - Constant-velocity time parameterisation: Simple and sufficient for this
    task. Can be upgraded to trapezoidal or S-curve profiles for real robots
    (see comments in time_parameterize()).
  - The node republishes on a 1-second timer so the controller can subscribe
    at any time without missing the path.

Topics:
  Publishers:
    /smoothed_path      (nav_msgs/Path)       — time-stamped trajectory
    /waypoint_markers   (visualization_msgs/MarkerArray) — RViz spheres

Author: karburettor
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from visualization_msgs.msg import Marker, MarkerArray


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

#: Number of points sampled along the fitted spline.
SPLINE_SAMPLE_COUNT: int = 100

#: Radius (m) of the RViz sphere markers for raw waypoints.
MARKER_SCALE: float = 0.15

#: RGBA colour of waypoint markers (red, fully opaque).
MARKER_COLOR: tuple = (1.0, 0.0, 0.0, 1.0)  # (r, g, b, a)


class PathSmoother(Node):
    """
    ROS2 node that smooths a set of 2D waypoints into a time-parameterized
    trajectory and publishes it for the PurePursuitController to follow.
    """

    def __init__(self) -> None:
        super().__init__('path_smoother')

        # ── Publishers ────────────────────────────────────────
        self.path_pub = self.create_publisher(Path, 'smoothed_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoint_markers', 10)

        # ── Task 1: Define raw waypoints ──────────────────────
        # Format: [[x0, y0], [x1, y1], ...]
        # These simulate the output of a global planner.
        self.waypoints = np.array([
            [0.0, 0.0],
            [2.0, 1.0],
            [4.0, -1.0],
            [6.0, 1.0],
            [8.0, 0.0],
        ])

        # ── Velocity for time parameterisation (m/s) ─────────
        # Kept low (0.15 m/s) to match the physical limits of TurtleBot3 Burger.
        self.target_velocity: float = 0.15

        # ── Timer: republish at 1 Hz ─────────────────────────
        self.timer = self.create_timer(1.0, self._timer_callback)

        self.get_logger().info(
            f'PathSmoother started. {len(self.waypoints)} waypoints loaded.'
        )

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    def _fit_spline(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit a Cubic Spline through the raw waypoints and sample it uniformly.

        A uniform parameter t ∈ [0, 1] is used — one value per waypoint.
        The spline is then evaluated at SPLINE_SAMPLE_COUNT evenly-spaced
        t values to yield a dense, smooth set of (x, y) coordinates.

        Returns
        -------
        xs, ys : np.ndarray
            Arrays of sampled x and y coordinates (length = SPLINE_SAMPLE_COUNT).
        """
        # One parameter value per waypoint (chord-length parameterisation
        # would be more accurate but uniform is sufficient here).
        t_knots = np.linspace(0, 1, len(self.waypoints))

        cs_x = CubicSpline(t_knots, self.waypoints[:, 0])
        cs_y = CubicSpline(t_knots, self.waypoints[:, 1])

        t_fine = np.linspace(0, 1, SPLINE_SAMPLE_COUNT)
        return cs_x(t_fine), cs_y(t_fine)

    def _time_parameterize(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> list[float]:
        """
        Assign a timestamp to each sampled point using constant-velocity
        integration along arc length.

        Formula: t_i = Σ(d_i) / v_target
        where d_i is the Euclidean distance between consecutive points.

        Note for real-robot extension:
            Replace constant velocity with a trapezoidal profile:
              - Ramp up over the first N points (acceleration phase).
              - Hold at cruise speed.
              - Ramp down near the goal (deceleration phase).
            This respects actuator torque limits and prevents wheel slip.

        Returns
        -------
        list[float]
            Elapsed time (seconds) for each point.
        """
        times: list[float] = [0.0]
        accumulated_dist = 0.0

        for i in range(1, len(xs)):
            segment_dist = math.sqrt(
                (xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2
            )
            accumulated_dist += segment_dist
            times.append(accumulated_dist / self.target_velocity)

        return times

    def _build_path_msg(
        self, xs: np.ndarray, ys: np.ndarray, times: list[float]
    ) -> Path:
        """
        Construct a nav_msgs/Path from sampled coordinates and timestamps.

        Each PoseStamped.header.stamp encodes the time at which the robot
        should reach that point — fulfilling the Task 2 trajectory format
        [(x, y, t), ...].

        Parameters
        ----------
        xs, ys : np.ndarray
            Sampled path coordinates.
        times : list[float]
            Elapsed time in seconds for each point.

        Returns
        -------
        nav_msgs/Path
        """
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()

        start_time = self.get_clock().now()

        for x, y, t_offset in zip(xs, ys, times):
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = (
                start_time + rclpy.duration.Duration(seconds=t_offset)
            ).to_msg()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            msg.poses.append(pose)

        return msg

    def _build_marker_array(self) -> MarkerArray:
        """
        Build a MarkerArray of red spheres at each raw waypoint for RViz.

        Returns
        -------
        visualization_msgs/MarkerArray
        """
        marker_array = MarkerArray()
        r, g, b, a = MARKER_COLOR

        for i, (wx, wy) in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(wx)
            marker.pose.position.y = float(wy)
            marker.scale.x = MARKER_SCALE
            marker.scale.y = MARKER_SCALE
            marker.scale.z = MARKER_SCALE
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = a
            marker_array.markers.append(marker)

        return marker_array

    # ─────────────────────────────────────────────────────────
    # Timer callback (main loop)
    # ─────────────────────────────────────────────────────────

    def _timer_callback(self) -> None:
        """
        Called at 1 Hz. Generates and publishes the smoothed trajectory
        and waypoint markers.
        """
        xs, ys = self._fit_spline()
        times = self._time_parameterize(xs, ys)

        path_msg = self._build_path_msg(xs, ys, times)
        marker_msg = self._build_marker_array()

        self.path_pub.publish(path_msg)
        self.marker_pub.publish(marker_msg)

        self.get_logger().debug(
            f'Published smoothed path ({len(path_msg.poses)} poses, '
            f'total time={times[-1]:.1f}s)'
        )


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathSmoother()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
