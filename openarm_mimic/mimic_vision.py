#!/usr/bin/python3
"""
Mimic Vision Node
=================
此模块实现了OpenArm Mimic系统的视觉处理核心节点。
主要功能：
1. 订阅RGB和深度摄像头图像话题。
2. 使用MediaPipe进行人体姿态估计和手势识别。
3. 将人体动作映射为机械臂的控制指令。
4. 发布目标位姿给控制器。
5. 提供可视化调试界面。

Author: OpenArm Team
Date: 2026-02-21
"""

import sys
# Remove conda paths from sys.path to avoid conflicts
# 移除Conda环境路径，防止与ROS2系统环境冲突
sys.path = [p for p in sys.path if "miniconda" not in p and "anaconda" not in p]

import rclpy
from rclpy.node import Node
from rclpy.time import Time
import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

# Import Modules
from openarm_mimic.mimic_motion_capture import MimicMotionCapture
from openarm_mimic.mimic_coordinate_mapping import MimicCoordinateMapping
from openarm_mimic.mimic_command_publisher import MimicCommandPublisher

class MimicVisionNode(Node):
    """
    视觉处理节点类，负责协调图像接收、算法处理和指令发布。
    """
    def __init__(self):
        """
        初始化节点，设置参数、模块、订阅者和定时器。
        """
        super().__init__('mimic_vision_node')
        
        # Parameters
        # 声明并获取ROS参数
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw') 
        self.declare_parameter('mirror', True)
        self.declare_parameter('web_debug', True) # Enable web debug by default
        
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.mirror = self.get_parameter('mirror').value
        
        # Initialize Modules
        # 初始化各个功能模块：动作捕捉、坐标映射、指令发布
        self.mocap = MimicMotionCapture()
        self.mapper = MimicCoordinateMapping()
        self.publisher = MimicCommandPublisher(self)
        
        self.bridge = CvBridge()
        self.current_frame = None
        self.current_depth = None
        
        # Interaction State
        self.active = False
        self.need_reset = False # Flag to trigger reset on next frame
        self.cv_window_name = "Mimic Vision"
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.cv_window_name, self.mouse_callback)

        # Subscribers
        self.get_logger().info("Waiting for camera topics...")
        # 创建图像订阅者，使用传感器QoS以确保最佳实时性
        self.sub_color = self.create_subscription(Image, self.color_topic, self.color_callback, qos_profile=qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile=qos_profile_sensor_data)
        
        # FPS calculation variables
        self.frame_count = 0
        self.last_fps_time = self.get_clock().now()
        
        # Timer for processing loop (30 FPS)
        # 创建30Hz的处理定时器，与摄像头帧率匹配
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info(f"Mimic Vision Node Started. Subscribing to {self.color_topic} and {self.depth_topic}")

    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数，用于切换系统激活状态。
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.active:
                self.need_reset = True
                self.get_logger().info("System Activating... Waiting for pose...")
            else:
                self.active = False
                self.get_logger().info("System Deactivated by mouse click.")

    def color_callback(self, msg):
        """
        彩色图像回调函数。
        
        功能：
        1. 接收最新的彩色图像消息。
        2. 计算传输延迟并记录日志。
        3. 将ROS消息转换为OpenCV格式。
        4. 统计并打印实际接收帧率。
        
        Args:
            msg (sensor_msgs/Image): 输入的彩色图像消息
        """
        try:
            # Log latency for color frame
            now = self.get_clock().now()
            color_time = Time.from_msg(msg.header.stamp)
            latency = (now - color_time).nanoseconds / 1e9
            # Only log every 2 seconds to avoid spamming
            if (now.nanoseconds % 2000000000) < 50000000: 
                 self.get_logger().info(f"Color Latency: {latency:.3f}s")
            
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Update FPS counter on color frames
            self.frame_count += 1
            dt = (now - self.last_fps_time).nanoseconds / 1e9
            if dt > 2.0:
                fps = self.frame_count / dt
                self.get_logger().info(f"Subscription FPS: {fps:.2f}")
                self.frame_count = 0
                self.last_fps_time = now
                
        except Exception as e:
            self.get_logger().error(f"Color CV Bridge Error: {e}")

    def depth_callback(self, msg):
        """
        深度图像回调函数。
        
        功能：
        1. 接收最新的深度图像消息。
        2. 将ROS消息转换为OpenCV格式（passthrough）。
        
        Args:
            msg (sensor_msgs/Image): 输入的深度图像消息
        """
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth CV Bridge Error: {e}")

    def timer_callback(self):
        """
        主处理循环回调函数（30Hz）。
        
        功能：
        1. 获取当前最新的彩色和深度帧。
        2. 执行人体姿态估计 (MediaPipe)。
        3. 计算目标位姿和手爪开合度。
        4. 发布控制指令。
        5. 可视化处理结果。
        6. 监控处理耗时。
        """
        # Measure processing time
        start_time = self.get_clock().now()

        # 1. Acquire Frame
        frame = self.current_frame
        depth = self.current_depth

        # 2. Visualization & Processing
        if frame is None:
            return # Don't show anything if no topic data yet
        else:
            try:
                # Resize for performance if too large (improves FPS significantly)
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (640, new_h))

                # Mirror if needed
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                
                # --- Motion Capture ---
                # 执行姿态估计
                results = self.mocap.process_frame(frame)
                self.mocap.draw_landmarks(frame, results)
                
                # Activate logic
                if self.need_reset and results.pose_world_landmarks:
                    if self.mapper.reset_origin(results.pose_world_landmarks.landmark):
                        self.active = True
                        self.need_reset = False
                        self.get_logger().info("System Activated! Driving OpenArm.")
                
                # --- Coordinate Mapping ---
                # 计算目标位姿
                if results.pose_world_landmarks and self.active:
                    # Pass self.mirror to apply Y-axis correction (prevent crossing) without swapping indices
                    target_l, target_r, orient_l, orient_r, left_valid, right_valid = \
                        self.mapper.compute_target_pose(results.pose_world_landmarks.landmark, mirror=self.mirror)
                    
                    # 获取手爪开合度
                    gripper_l = self.mapper.get_gripper_ratio(results.left_hand_landmarks)
                    gripper_r = self.mapper.get_gripper_ratio(results.right_hand_landmarks)
                    
                    # Do not swap gripper either (MediaPipe correctly identifies left/right even in mirror)
                    # if self.mirror:
                    #    gripper_l, gripper_r = gripper_r, gripper_l

                    self.publisher.publish_frame(target_l, target_r, orient_l, orient_r, left_valid, right_valid,
                                                  gripper_l, gripper_r)
                
                # Draw status overlay
                status_color = (0, 255, 0) if self.active else (0, 0, 255)
                status_text = "ACTIVE" if self.active else "PAUSED (Click/Press 'S' to Start)"
                cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                # --- Visualization ---
                # 图像可视化处理
                if depth is not None:
                    # Normalize depth for display
                    if self.mirror:
                         depth = cv2.flip(depth, 1)
                    
                    if depth.dtype == np.uint16:
                        depth_disp = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_disp = cv2.applyColorMap(depth_disp, cv2.COLORMAP_JET)
                    else:
                        depth_disp = depth
                        if len(depth_disp.shape) == 2:
                            depth_disp = cv2.cvtColor(depth_disp, cv2.COLOR_GRAY2BGR)
                    
                    h_rgb, w_rgb = frame.shape[:2]
                    h_d, w_d = depth_disp.shape[:2]
                    
                    if h_d != h_rgb:
                        scale = h_rgb / h_d
                        new_w = int(w_d * scale)
                        depth_disp = cv2.resize(depth_disp, (new_w, h_rgb))
                    
                    combined = np.hstack((frame, depth_disp))
                    # cv2.imshow("Mimic Vision (RGB + Depth)", combined)
                    # cv2.imshow("Mimic Vision (Depth)", depth_disp)
                    cv2.imshow("Mimic Vision (RGB)", frame)
                else:
                    cv2.imshow("Mimic Vision", frame)
                
                # Log processing time if it exceeds 30ms (warning level)
                end_time = self.get_clock().now()
                proc_duration = (end_time - start_time).nanoseconds / 1e6 # ms
                if proc_duration > 30.0:
                    self.get_logger().warn(f"Processing time: {proc_duration:.2f} ms")

            except Exception as e:
                self.get_logger().error(f"Processing Error: {e}")

        # 3. Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()
        elif key == ord('s'):
            self.active = not self.active
            status = "Activated" if self.active else "Deactivated"
            self.get_logger().info(f"System {status} by keyboard.")

def main(args=None):
    """
    程序入口函数。
    """
    rclpy.init(args=args)
    node = MimicVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
