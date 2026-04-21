"""
Mimic 笔记本端 Launch（只做 RViz 可视化）

运行位置: 笔记本 (Ubuntu 22 + ROS 2 Humble)

前提：
  - 与开发板在同一 ROS_DOMAIN_ID / RMW / 局域网
  - 笔记本已编译 openarm_description (RViz 需要 mesh 文件)
  - 开发板已经启动 mimic_board.launch.py（其 robot_state_publisher
    会把 URDF 以 transient_local QoS 发到 /robot_description，
    bimanual.rviz 里的 RobotModel 就是从该话题读取的）

启停控制（任选其一）：
  # 启动
  ros2 topic pub --once /mimic/enable std_msgs/msg/Bool "data: true"
  # 停止
  ros2 topic pub --once /mimic/enable std_msgs/msg/Bool "data: false"
或用 rqt → "Message Publisher" 面板。
"""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_config = PathJoinSubstitution(
        [FindPackageShare('openarm_description'), 'rviz', 'bimanual.rviz']
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([rviz_node])
