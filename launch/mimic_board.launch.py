"""
Mimic 板端 Launch（只做数据采集与计算，无 RViz、无 cv2 窗口）

运行位置: 开发板 (RDK S100P)

包含节点：
  - xacro 生成 URDF → /tmp/openarm_mimic.urdf
  - robot_state_publisher (把 URDF 以 /robot_description 话题广播给笔记本 RViz)
  - orbbec_camera (Gemini 330)
  - mimic_vision (headless, show_window=false; 通过 /mimic/enable 远程启停)
  - mimic_control left / right (真机/仿真由 use_sim 控制)

笔记本端另起 mimic_laptop.launch.py 做 RViz 可视化和启停控制。
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    left_can_arg = DeclareLaunchArgument('left_can', default_value='can0')
    right_can_arg = DeclareLaunchArgument('right_can', default_value='can1')
    camera_device_arg = DeclareLaunchArgument('camera_device', default_value='2')
    use_sim_arg = DeclareLaunchArgument(
        'use_sim', default_value='false',
        description='Enable simulation mode (no CAN, just publish joint_states)'
    )
    use_joint_mapping_arg = DeclareLaunchArgument(
        'use_joint_mapping', default_value='false',
        description='Direct joint mapping vs Cartesian IK'
    )
    start_enabled_arg = DeclareLaunchArgument(
        'start_enabled', default_value='false',
        description='If true, mimic_vision starts activated; otherwise wait for /mimic/enable=true'
    )

    left_can = LaunchConfiguration('left_can')
    right_can = LaunchConfiguration('right_can')
    camera_device = LaunchConfiguration('camera_device')
    use_sim = LaunchConfiguration('use_sim')

    # ---- URDF 生成 ----
    pkg_share = FindPackageShare('openarm_description')
    xacro_file = PathJoinSubstitution([pkg_share, 'urdf', 'robot', 'v10.urdf.xacro'])
    urdf_path = "/tmp/openarm_mimic.urdf"
    if os.name == 'nt':
        urdf_path = os.path.join(os.environ['TEMP'], 'openarm_mimic.urdf')

    gen_urdf = ExecuteProcess(
        cmd=['ros2', 'run', 'xacro', 'xacro', xacro_file, 'bimanual:=true', '-o', urdf_path],
        output='screen'
    )

    # ---- robot_state_publisher（板端发布，笔记本 RViz 订阅 /robot_description + /tf）----
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        arguments=[urdf_path]
    )
    rsp_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=gen_urdf,
            on_exit=[robot_state_publisher]
        )
    )

    # ---- Orbbec 相机 ----
    orbbec_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('orbbec_camera'), 'launch', 'gemini_330_series.launch.py'])
        ),
        launch_arguments={
            'camera_name': 'camera',
            'depth_registration': 'false',
            'enable_point_cloud': 'false',
            'enable_colored_point_cloud': 'false',
            'color_width': '640', 'color_height': '480', 'color_fps': '30',
            'depth_width': '640', 'depth_height': '400', 'depth_fps': '30',
            'left_ir_width': '640', 'left_ir_height': '400', 'left_ir_fps': '30',
            'right_ir_width': '640', 'right_ir_height': '400', 'right_ir_fps': '30',
        }.items()
    )

    # ---- 视觉节点（板端 headless + 发布压缩骨架图给笔记本）----
    vision_node = Node(
        package='openarm_mimic',
        executable='mimic_vision',
        name='mimic_vision',
        output='screen',
        parameters=[{
            'use_ros_driver': True,
            'device_id': camera_device,
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/depth/image_raw',
            'use_joint_mapping': LaunchConfiguration('use_joint_mapping'),
            'show_window': False,
            'start_enabled': LaunchConfiguration('start_enabled'),
            'publish_annotated': True,
            'annotated_topic': '/mimic/vision/annotated/compressed',
            'annotated_jpeg_quality': 70,
            'annotated_stride': 1,  # 如需省流量可改 2(≈15fps) 或 3(≈10fps)
        }]
    )

    # ---- 左右臂控制节点 ----
    left_node = Node(
        package='openarm_mimic',
        executable='mimic_control',
        name='mimic_control_left',
        output='screen',
        parameters=[{
            'arm_side': 'left_arm',
            'can_interface': left_can,
            'urdf_path': urdf_path,
            'kp': 30.0, 'kd': 1.0, 'smoothing_factor': 0.05,
            'use_sim': use_sim,
        }]
    )
    right_node = Node(
        package='openarm_mimic',
        executable='mimic_control',
        name='mimic_control_right',
        output='screen',
        parameters=[{
            'arm_side': 'right_arm',
            'can_interface': right_can,
            'urdf_path': urdf_path,
            'kp': 30.0, 'kd': 1.0, 'smoothing_factor': 0.05,
            'use_sim': use_sim,
        }]
    )

    return LaunchDescription([
        left_can_arg, right_can_arg, camera_device_arg,
        use_sim_arg, use_joint_mapping_arg, start_enabled_arg,
        gen_urdf,
        rsp_handler,
        orbbec_launch,
        vision_node,
        left_node,
        right_node,
    ])
