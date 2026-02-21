"""
Mimic Coordinate Mapping Module
===============================
此模块负责将MediaPipe检测到的3D关键点坐标映射到机器人的坐标系中。

主要功能：
1. 坐标系转换（MediaPipe -> Robot）。
2. 计算机械臂末端执行器的目标位置。
3. 计算手爪的开合比例。
"""

import numpy as np

class MimicCoordinateMapping:
    """
    坐标映射类，处理空间坐标变换和目标位姿计算。
    """
    def __init__(self, scale_factor=1.0):
        """
        初始化映射参数。
        
        Args:
            scale_factor (float): 动作缩放因子，用于调整动作幅度。
        """
        # Calibration / Offsets (Meters)
        # 机器人左右肩部在基座坐标系下的位置偏移
        self.robot_shoulder_left = np.array([0.0, 0.20, 0.35]) 
        self.robot_shoulder_right = np.array([0.0, -0.20, 0.35])
        self.scale_factor = scale_factor

    def transform_vec(self, v):
        """
        将向量从MediaPipe坐标系转换到机器人坐标系。
        
        Args:
            v (list/array): MediaPipe坐标系下的向量 [x, y, z]
            
        Returns:
            numpy.ndarray: 机器人坐标系下的向量 [x, y, z]
        
        Mapping Logic:
        - Robot X (Forward) = - MP Z (MediaPipe Z is depth, positive away from camera)
        - Robot Y (Left) = - MP X (MediaPipe X is right)
        - Robot Z (Up) = - MP Y (MediaPipe Y is down)
        """
        return np.array([-v[2], -v[0], -v[1]])

    def compute_target_pose(self, landmarks):
        """
        根据关键点计算左右臂的目标位置。
        
        Args:
            landmarks: MediaPipe的pose_world_landmarks.landmark列表。
            
        Returns:
            tuple: (target_l, target_r, left_valid, right_valid)
                   - target_l/r: 左右臂目标位置 [x, y, z]
                   - left/right_valid: 检测是否有效
        """
        if not landmarks:
            return None, None, False, False

        # Helper to get np array
        def get_vec(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

        # Extract Keypoints
        # 11: left_shoulder, 12: right_shoulder
        # 15: left_wrist, 16: right_wrist
        ls = get_vec(11)
        rs = get_vec(12)
        lw = get_vec(15)
        rw = get_vec(16)
        
        # Visibility check
        left_valid = landmarks[15].visibility > 0.5
        right_valid = landmarks[16].visibility > 0.5
        
        # Calculate Relative Vectors (Wrist relative to Shoulder)
        # 计算手腕相对于肩膀的向量
        vec_l = lw - ls
        vec_r = rw - rs
        
        # Apply transformation
        # 应用坐标变换和缩放
        robot_vec_l = self.transform_vec(vec_l) * self.scale_factor
        robot_vec_r = self.transform_vec(vec_r) * self.scale_factor
        
        # Absolute Target Position
        # 叠加到机器人肩部坐标上，得到绝对目标位置
        target_l = self.robot_shoulder_left + robot_vec_l
        target_r = self.robot_shoulder_right + robot_vec_r

        return target_l, target_r, left_valid, right_valid

    def get_gripper_ratio(self, hand_landmarks):
        """
        根据手部关键点计算手爪开合比例 (0.0 到 1.0)。
        
        Args:
            hand_landmarks: MediaPipe的hand_landmarks对象。
            
        Returns:
            float: 开合比例，0.0表示完全张开，1.0表示完全闭合。
        """
        if not hand_landmarks:
            return 0.0
            
        # Index Tip: 8, Thumb Tip: 4
        # 获取拇指指尖(4)和食指指尖(8)的位置
        thumb = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z])
        index = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z])
        
        # 计算拇指和食指的距离
        dist = np.linalg.norm(thumb - index)
        
        # Map distance to 0-1. Heuristic: 0.02 (closed) to 0.15 (open)
        # 将距离映射到0-1之间。经验值：0.02m视为闭合，0.15m视为张开
        # ratio 1.0 means closed, 0.0 means open
        ratio = np.clip((0.15 - dist) / (0.15 - 0.02), 0.0, 1.0)
        return float(ratio)
