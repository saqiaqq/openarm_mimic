import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Arguments
    left_can_arg = DeclareLaunchArgument('left_can', default_value='can1')
    right_can_arg = DeclareLaunchArgument('right_can', default_value='can0')
    camera_device_arg = DeclareLaunchArgument('camera_device', default_value='2')
    
    left_can = LaunchConfiguration('left_can')
    right_can = LaunchConfiguration('right_can')
    camera_device = LaunchConfiguration('camera_device')
    
    # Generate URDF
    pkg_share = FindPackageShare('openarm_description')
    xacro_file = PathJoinSubstitution([pkg_share, 'urdf', 'robot', 'v10.urdf.xacro'])
    
    urdf_path = "/tmp/openarm_mimic.urdf"
    
    # Note: On Windows /tmp might not exist.
    if os.name == 'nt':
        urdf_path = os.path.join(os.environ['TEMP'], 'openarm_mimic.urdf')
    
    # Command to generate URDF
    gen_urdf = ExecuteProcess(
        cmd=['ros2', 'run', 'xacro', 'xacro', xacro_file, 'bimanual:=true', '-o', urdf_path],
        output='screen'
    )

    # Orbbec Camera Launch
    orbbec_camera_pkg = FindPackageShare('orbbec_camera')
    orbbec_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([orbbec_camera_pkg, 'launch', 'gemini_330_series.launch.py'])
        ),
        launch_arguments={
            'camera_name': 'camera',
            'depth_registration': 'false',
            'enable_point_cloud': 'false',
            'enable_colored_point_cloud': 'false',
        }.items()
    )

    # Vision Node
    vision_node = Node(
        package='openarm_mimic',
        executable='mimic_vision',
        name='mimic_vision',
        output='screen',
        parameters=[{
            'use_ros_driver': True,
            'device_id': camera_device,
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/depth/image_raw'
        }]
    )
    
    # Left Arm Node
    left_node = Node(
        package='openarm_mimic',
        executable='mimic_control',
        name='mimic_control_left',
        output='screen',
        parameters=[{
            'arm_side': 'left_arm',
            'can_interface': left_can,
            'urdf_path': urdf_path,
            'kp': 30.0,
            'kd': 1.0,
            'smoothing_factor': 0.05
        }]
    )

    # Right Arm Node
    right_node = Node(
        package='openarm_mimic',
        executable='mimic_control',
        name='mimic_control_right',
        output='screen',
        parameters=[{
            'arm_side': 'right_arm',
            'can_interface': right_can,
            'urdf_path': urdf_path,
            'kp': 30.0,
            'kd': 1.0,
            'smoothing_factor': 0.05
        }]
    )
    
    return LaunchDescription([
        left_can_arg,
        right_can_arg,
        camera_device_arg,
        gen_urdf,
        orbbec_launch,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gen_urdf,
                on_exit=[vision_node, left_node, right_node]
            )
        )
    ])
