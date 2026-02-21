"""
Mimic Command Publisher Module
==============================
此模块负责将计算得到的目标位姿打包成ROS消息并发布。

主要功能：
1. 封装ROS发布者。
2. 构造`MimicFrame`自定义消息。
3. 填充位姿数据（位置和姿态）。
"""

from openarm_mimic.msg import MimicFrame
from geometry_msgs.msg import Pose

class MimicCommandPublisher:
    """
    指令发布类，处理ROS消息的构造和发布。
    """
    def __init__(self, node):
        """
        初始化发布者。
        
        Args:
            node (rclpy.node.Node): ROS节点实例，用于创建发布者。
        """
        self.node = node
        self.pub = node.create_publisher(MimicFrame, '/mimic/target_frame', 10)

    def publish_frame(self, target_l, target_r, left_valid, right_valid, left_gripper, right_gripper):
        """
        构造并发布MimicFrame消息。
        
        Args:
            target_l (list/array): 左臂目标位置 [x, y, z]
            target_r (list/array): 右臂目标位置 [x, y, z]
            left_valid (bool): 左臂数据是否有效
            right_valid (bool): 右臂数据是否有效
            left_gripper (float): 左手爪开合度 [0.0, 1.0]
            right_gripper (float): 右手爪开合度 [0.0, 1.0]
        """
        if target_l is None or target_r is None:
            return

        frame_msg = MimicFrame()
        frame_msg.header.stamp = self.node.get_clock().now().to_msg()
        frame_msg.header.frame_id = "openarm_body_link0"
        
        frame_msg.left_track_valid = left_valid
        frame_msg.right_track_valid = right_valid
        
        # Fill Arm Poses
        self._fill_pose(frame_msg.left_arm_pose, target_l)
        self._fill_pose(frame_msg.right_arm_pose, target_r)
        
        # Fill Gripper Ratios
        frame_msg.left_gripper_ratio = left_gripper
        frame_msg.right_gripper_ratio = right_gripper

        self.pub.publish(frame_msg)

    def _fill_pose(self, pose_msg, target_pos):
        """
        辅助函数：填充Pose消息。
        
        Args:
            pose_msg (geometry_msgs.msg.Pose): 要填充的Pose消息对象。
            target_pos (list/array): 目标位置 [x, y, z]。
        """
        pose_msg.position.x = target_pos[0]
        pose_msg.position.y = target_pos[1]
        pose_msg.position.z = target_pos[2]
        # Default Orientation (Forward/Inwards)
        # 设置默认姿态（此处简化为固定方向，实际应用中可能需要根据手腕角度计算）
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = 0.0
        pose_msg.orientation.w = 1.0 
