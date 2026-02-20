"""
monitor_node.py
===============
ROS2 Node: MonitorNode

A lightweight diagnostic node that logs the robot's position and
velocity commands to the terminal at a throttled rate (1 Hz).

Use this node alongside the main launch file to verify:
  - The robot is moving (position changes)
  - Commands are being published (linear and angular non-zero)
  - The robot stops at the goal (commands return to zero)

Topics Subscribed:
  /odom     (nav_msgs/Odometry)          — robot position
  /cmd_vel  (geometry_msgs/TwistStamped) — velocity commands

Usage:
  ros2 run waypoint_navigation monitor_node

Author: karburettor
"""

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class MonitorNode(Node):
    """
    Diagnostic node: prints robot pose and velocity commands at 1 Hz.
    Useful for quick terminal inspection without opening RViz.
    """

    def __init__(self) -> None:
        super().__init__('monitor_node')

        self.create_subscription(Odometry, 'odom', self._odom_callback, 10)
        self.create_subscription(TwistStamped, '/cmd_vel', self._cmd_callback, 10)

        self.get_logger().info('MonitorNode active — logging at 1 Hz.')

    def _odom_callback(self, msg: Odometry) -> None:
        """Log the robot's (x, y) position from odometry."""
        pos = msg.pose.pose.position
        self.get_logger().info(
            f'[POSE]  x={pos.x:.2f} m  y={pos.y:.2f} m',
            throttle_duration_sec=1.0,
        )

    def _cmd_callback(self, msg: TwistStamped) -> None:
        """Log the latest velocity command sent to the robot."""
        lin = msg.twist.linear.x
        ang = msg.twist.angular.z
        self.get_logger().info(
            f'[CMD]   linear={lin:.2f} m/s  angular={ang:.2f} rad/s',
            throttle_duration_sec=1.0,
        )


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
