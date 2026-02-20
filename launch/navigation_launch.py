import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('waypoint_navigation')
    tb3_gazebo_path = get_package_share_directory('turtlebot3_gazebo')

    # 1. Include Gazebo (Empty World)
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_path, 'launch', 'empty_world.launch.py')
        )
    )

    # 2. RViz with pre-saved config
    rviz_config_path = os.path.join(pkg_path, 'rviz', 'navigation.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    # 3. Your Nodes
    path_smoother = Node(
        package='waypoint_navigation',
        executable='path_smoother',
        name='path_smoother',
        parameters=[{'use_sim_time': True}]
    )

    controller = Node(
        package='waypoint_navigation',
        executable='pure_pursuit_controller',
        name='pure_pursuit_controller',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        gazebo_launch,
        rviz_node,
        path_smoother,
        controller
    ])