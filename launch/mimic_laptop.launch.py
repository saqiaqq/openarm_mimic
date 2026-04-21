"""
Mimic 笔记本端 Launch
===================
职责：可视化 + 启停控制
  1. rviz2          —— 显示双臂三维姿态（通过 /robot_description + /tf + /joint_states）
  2. mimic_viewer   —— 显示板端 /mimic/vision/annotated/compressed 里的骨架叠加视频，
                       并用鼠标/键盘把 /mimic/enable 发回板子

前提：
  - 与开发板 ROS_DOMAIN_ID / RMW / 局域网一致，且已能互通话题
  - 笔记本已编译 openarm_mimic 与 openarm_description
  - 开发板已启动 mimic_board.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Whether to launch rviz2')
    use_viewer_arg = DeclareLaunchArgument(
        'use_viewer', default_value='true',
        description='Whether to launch mimic_viewer (camera preview + enable toggle)')
    annotated_topic_arg = DeclareLaunchArgument(
        'annotated_topic', default_value='/mimic/vision/annotated/compressed')

    rviz_config = PathJoinSubstitution(
        [FindPackageShare('openarm_description'), 'rviz', 'bimanual.rviz']
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    viewer_node = Node(
        package='openarm_mimic',
        executable='mimic_viewer',
        name='mimic_viewer',
        output='screen',
        parameters=[{
            'annotated_topic': LaunchConfiguration('annotated_topic'),
            'enable_topic': '/mimic/enable',
            'window_name': 'Mimic Viewer',
        }],
        condition=IfCondition(LaunchConfiguration('use_viewer')),
    )

    return LaunchDescription([
        use_rviz_arg,
        use_viewer_arg,
        annotated_topic_arg,
        rviz_node,
        viewer_node,
    ])
