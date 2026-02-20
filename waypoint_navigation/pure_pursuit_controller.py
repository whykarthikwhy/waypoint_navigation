"""
pure_pursuit_controller.py
==========================
ROS2 Node: PurePursuitController

Responsibilities:
  1. Subscribe to the smoothed trajectory from PathSmoother.
  2. Subscribe to odometry to obtain the robot's current pose.
  3. Subscribe to LaserScan for basic obstacle detection.
  4. Run the Pure Pursuit algorithm at odometry rate to compute and
     publish velocity commands (TwistStamped) on /cmd_vel.

Algorithm — Pure Pursuit:
  Pure Pursuit is a geometric path-tracking controller. It works by:
    a) Projecting a 'lookahead point' (the 'carrot') ahead on the path
       at a fixed distance from the robot.
    b) Computing the arc curvature κ needed to reach that point.
    c) Setting angular velocity ω = κ · v, where v is the desired speed.

  Adaptive lookahead: The lookahead distance shrinks as the robot
  approaches the goal, preventing the controller from 'cutting corners'
  near the endpoint.

State Machine:
  ALIGNING  → Robot rotates in-place to face the first lookahead point.
  TRACKING  → Pure Pursuit runs; robot follows the curved path.
  REACHED   → Robot stops; mission complete.

Extra Credit — Obstacle Avoidance:
  A LaserScan subscriber monitors a ±30° front arc. If anything is
  detected within 0.7m, a lateral nudge is added to local_y before
  the curvature calculation, steering the robot away from the obstacle.

Topics:
  Subscribers:
    /smoothed_path   (nav_msgs/Path)      — from PathSmoother
    /odom            (nav_msgs/Odometry)  — robot pose & velocity
    /scan            (sensor_msgs/LaserScan) — obstacle detection
  Publishers:
    /cmd_vel         (geometry_msgs/TwistStamped) — velocity commands

Author: karburettor
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


# ─────────────────────────────────────────────────────────────
# Tunable parameters
# ─────────────────────────────────────────────────────────────

#: Default lookahead distance (m). Larger = smoother but less accurate tracking.
LOOKAHEAD_DISTANCE: float = 0.5

#: Cruise linear velocity (m/s). TurtleBot3 Burger max is ~0.22 m/s.
CRUISE_VELOCITY: float = 0.2

#: Robot must be within this distance (m) to declare goal reached.
GOAL_THRESHOLD: float = 0.05   # 5 cm — precise stop

#: Distance (m) at which velocity ramp-down begins.
SLOWDOWN_RADIUS: float = 0.8

#: Minimum velocity (m/s) during ramp-down, to avoid stalling.
MIN_VELOCITY: float = 0.05

#: Angular speed (rad/s) used during the ALIGNING phase.
ALIGN_ANGULAR_SPEED: float = 0.3

#: Angle error (rad) below which alignment is considered complete.
ALIGN_THRESHOLD: float = 0.1   # ~5.7°

#: Minimum obstacle range (m) in the front arc to trigger avoidance.
OBSTACLE_RANGE: float = 0.7

#: Half-width of the front arc (degrees) used for obstacle detection.
OBSTACLE_ARC_DEG: int = 30

#: Lateral nudge magnitude (m) added to local_y when obstacle detected.
OBSTACLE_NUDGE: float = 0.4

#: Minimum valid laser return (m); filters out sensor noise.
MIN_LASER_RANGE: float = 0.05


class PurePursuitController(Node):
    """
    ROS2 node implementing Pure Pursuit trajectory tracking with
    adaptive lookahead, velocity ramping, and reactive obstacle avoidance.
    """

    def __init__(self) -> None:
        super().__init__('pure_pursuit_controller')

        # ── Subscribers ───────────────────────────────────────
        self.path_sub = self.create_subscription(
            Path, 'smoothed_path', self._path_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self._scan_callback, 10
        )

        # ── Publisher ─────────────────────────────────────────
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # ── Internal state ────────────────────────────────────
        self.path = None                    # nav_msgs/Path received from smoother
        self.current_pose = None            # geometry_msgs/Pose from odometry
        self.obstacle_detected: bool = False

        # State machine: ALIGNING → TRACKING → REACHED
        self.state: str = 'ALIGNING'

        self.get_logger().info('PurePursuitController started. Waiting for path...')

    # ─────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ─────────────────────────────────────────────────────────

    def _path_callback(self, msg: Path) -> None:
        """Cache the latest smoothed path from PathSmoother."""
        self.path = msg

    def _scan_callback(self, msg: LaserScan) -> None:
        """
        Detect obstacles in the robot's front arc.

        The LaserScan from TurtleBot3 is a 360-degree scan indexed 0–359.
        The front-facing arc is the union of [0, ARC] and [360-ARC, 359].
        Any valid return closer than OBSTACLE_RANGE sets the flag.
        """
        arc = OBSTACLE_ARC_DEG
        front_arc = msg.ranges[:arc] + msg.ranges[360 - arc:]
        valid = [r for r in front_arc if r > MIN_LASER_RANGE]

        if valid:
            self.obstacle_detected = min(valid) < OBSTACLE_RANGE
        else:
            self.obstacle_detected = False

    def _odom_callback(self, msg: Odometry) -> None:
        """
        Update current pose from odometry and trigger the control loop.

        Odometry drives the control loop rather than a timer, so the
        controller runs at the same rate as odometry (~30 Hz for TurtleBot3).
        """
        self.current_pose = msg.pose.pose
        if self.path is not None:
            self._control_loop()

    # ─────────────────────────────────────────────────────────
    # Core algorithm
    # ─────────────────────────────────────────────────────────

    def _get_lookahead_point(self, dist_to_goal: float):
        """
        Find the 'carrot' — the point on the path closest to the
        adaptive lookahead distance from the robot.

        Adaptive lookahead: current_lookahead = min(LOOKAHEAD_DISTANCE, dist_to_goal)
        This prevents the robot from looking past the goal during the
        final approach, ensuring a precise stop.

        Parameters
        ----------
        dist_to_goal : float
            Euclidean distance from robot to the final path point (m).

        Returns
        -------
        geometry_msgs/Point
            The target point the robot should steer toward.
        """
        # Shrink lookahead as we approach the goal (minimum 10 cm)
        current_lookahead = max(0.1, min(LOOKAHEAD_DISTANCE, dist_to_goal))

        best_point = None
        min_diff = float('inf')

        for pose in self.path.poses:
            dx = pose.pose.position.x - self.current_pose.position.x
            dy = pose.pose.position.y - self.current_pose.position.y
            dist = math.sqrt(dx ** 2 + dy ** 2)

            # Pick the path point whose distance from the robot is
            # closest to the desired lookahead distance.
            diff = abs(dist - current_lookahead)
            if diff < min_diff:
                min_diff = diff
                best_point = pose.pose.position

        # Fallback: use the last path point if nothing was found
        return best_point if best_point else self.path.poses[-1].pose.position

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        """
        Extract yaw (rotation about Z) from a geometry_msgs/Quaternion.

        Uses the ZYX Euler angle extraction formula. Only yaw is needed
        for a planar (2D) differential drive robot.

        Parameters
        ----------
        q : geometry_msgs/Quaternion

        Returns
        -------
        float
            Yaw angle in radians ∈ (-π, π].
        """
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """
        Wrap an angle to the range [-π, π].

        Using atan2(sin, cos) is numerically stable and avoids
        discontinuities at ±π.
        """
        return math.atan2(math.sin(angle), math.cos(angle))

    def _compute_velocity_ramp(self, dist_to_goal: float) -> float:
        """
        Scale linear velocity down smoothly as the robot nears the goal.

        Profile:
          - dist > SLOWDOWN_RADIUS  →  cruise at CRUISE_VELOCITY
          - dist ≤ SLOWDOWN_RADIUS  →  linearly interpolate down to MIN_VELOCITY
          - dist < GOAL_THRESHOLD   →  handled externally (stop_robot)

        This prevents the robot from overshooting the goal due to inertia.
        On a real robot this would be a trapezoidal profile that also
        accounts for acceleration limits.

        Parameters
        ----------
        dist_to_goal : float

        Returns
        -------
        float
            Target linear velocity (m/s).
        """
        if dist_to_goal < SLOWDOWN_RADIUS:
            return max(MIN_VELOCITY, CRUISE_VELOCITY * (dist_to_goal / SLOWDOWN_RADIUS))
        return CRUISE_VELOCITY

    def _control_loop(self) -> None:
        """
        Main control loop — called every time a new odometry message arrives.

        Flow:
          1. Check if goal is reached → stop.
          2. Find the lookahead 'carrot' point.
          3. Compute angle error and transform target to robot-local frame.
          4. Apply state machine logic (ALIGNING or TRACKING).
          5. Publish TwistStamped command.
        """
        if self.path is None or self.current_pose is None or self.state == 'REACHED':
            return

        # ── Step 1: Goal check ────────────────────────────────
        final_pt = self.path.poses[-1].pose.position
        dist_to_goal = math.sqrt(
            (final_pt.x - self.current_pose.position.x) ** 2
            + (final_pt.y - self.current_pose.position.y) ** 2
        )

        if dist_to_goal < GOAL_THRESHOLD:
            self._stop_robot()
            self.state = 'REACHED'
            self.get_logger().info('✓ MISSION SUCCESS: goal reached.')
            return

        # ── Step 2: Lookahead point ───────────────────────────
        target = self._get_lookahead_point(dist_to_goal)
        dx = target.x - self.current_pose.position.x
        dy = target.y - self.current_pose.position.y

        # ── Step 3: Robot heading & angle to target ───────────
        yaw = self._yaw_from_quaternion(self.current_pose.orientation)
        angle_to_target = self._wrap_angle(math.atan2(dy, dx) - yaw)

        # ── Step 4: State machine ─────────────────────────────
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        if self.state == 'ALIGNING':
            # Rotate in-place until roughly facing the first target
            if abs(angle_to_target) > ALIGN_THRESHOLD:
                msg.twist.angular.z = (
                    ALIGN_ANGULAR_SPEED if angle_to_target > 0 else -ALIGN_ANGULAR_SPEED
                )
            else:
                self.state = 'TRACKING'
                self.get_logger().info('Alignment done — switching to TRACKING.')

        elif self.state == 'TRACKING':
            # Transform target vector into robot-local frame
            local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
            local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)

            # ── Extra Credit: Reactive Obstacle Avoidance ────
            # A positive lateral nudge steers the robot left (away from
            # obstacles detected in the front arc). This is a simple
            # reactive strategy; a full implementation would use a
            # potential field or dynamic window approach.
            if self.obstacle_detected:
                self.get_logger().warn('Obstacle detected — applying lateral nudge.')
                local_y += OBSTACLE_NUDGE

            # ── Pure Pursuit curvature ────────────────────────
            # κ = 2y / L²  (standard pure pursuit formula)
            # where L = distance to the lookahead point, y = lateral offset.
            L_sq = local_x ** 2 + local_y ** 2
            kappa = (2.0 * local_y) / L_sq if L_sq > 0.001 else 0.0

            # ── Velocity ramp ─────────────────────────────────
            v = self._compute_velocity_ramp(dist_to_goal)

            msg.twist.linear.x = float(v)
            # ω = κ · v  (relates curvature to angular rate)
            msg.twist.angular.z = float(kappa * v)

        self.cmd_pub.publish(msg)

    # ─────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────

    def _stop_robot(self) -> None:
        """Publish a zero-velocity command to halt the robot immediately."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        # All twist fields default to 0.0 — robot stops.
        self.cmd_pub.publish(msg)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()
