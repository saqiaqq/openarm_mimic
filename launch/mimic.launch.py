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
    use_sim_arg = DeclareLaunchArgument('use_sim', default_value='false', description='Enable simulation mode (RViz only, no hardware)')
    
    left_can = LaunchConfiguration('left_can')
    right_can = LaunchConfiguration('right_can')
    camera_device = LaunchConfiguration('camera_device')
    use_sim = LaunchConfiguration('use_sim')
    
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

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        arguments=[urdf_path]
    )

    # Note: We need to delay robot_state_publisher until URDF is generated.
    # The current ExecuteProcess for xacro runs in background.
    # So we should probably use RegisterEventHandler to launch RSP after xacro exits.
    
    rsp_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=gen_urdf,
            on_exit=[robot_state_publisher]
        )
    )

    # RViz
    rviz_config = PathJoinSubstitution([FindPackageShare('openarm_description'), 'rviz', 'bimanual.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )
    # Launch RViz if use_sim is true? Or always. Let's launch always for visualization.
    
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
            'color_width': '640',
            'color_height': '480',
            'color_fps': '30',
            'depth_width': '640',
            'depth_height': '400',
            'depth_fps': '30',
            'left_ir_width': '640',
            'left_ir_height': '400',
            'left_ir_fps': '30',
            'right_ir_width': '640',
            'right_ir_height': '400',
            'right_ir_fps': '30',
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
            'smoothing_factor': 0.05,
            'use_sim': use_sim
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
            'smoothing_factor': 0.05,
            'use_sim': use_sim
        }]
    )

    return LaunchDescription([
        left_can_arg,
        right_can_arg,
        camera_device_arg,
        use_sim_arg,
        gen_urdf,
        rsp_handler, # Delayed launch of RSP
        rviz_node,   # RViz
        orbbec_launch,
        vision_node,
        left_node,
        right_node
    ])
