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
# from scipy.spatial.transform import Rotation as R

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
        # Updated to match OpenArm V10 URDF defaults
        # Left: 0.0 0.031 0.698
        # Right: 0.0 -0.031 0.698
        self.robot_shoulder_left = np.array([0.0, 0.031, 0.698]) 
        self.robot_shoulder_right = np.array([0.0, -0.031, 0.698])
        self.scale_factor = scale_factor
        
        # Workspace Limits (Based on OpenArm URDF approximate dimensions)
        # 工作空间限制
        self.max_reach = 0.70  # Maximum reach radius from shoulder (meters) - URDF limit approx
        self.min_z = 0.05      # Minimum Z height (meters) to avoid hitting table
        
        # Initial human pose offsets (for relative motion)
        self.initial_human_pose = None
        self.is_calibrated = False

    def reset_origin(self, landmarks):
        """
        重置原点，以当前人体姿态作为初始状态。
        """
        if not landmarks:
            return False
            
        def get_vec(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

        # Store initial wrist positions relative to shoulders
        ls = get_vec(11) # Left Shoulder
        rs = get_vec(12) # Right Shoulder
        lw = get_vec(15) # Left Wrist
        rw = get_vec(16) # Right Wrist
        
        self.initial_l_vec = lw - ls
        self.initial_r_vec = rw - rs
        self.is_calibrated = True
        return True

    def _limit_target(self, target, shoulder_pos, arm_name="arm"):
        """
        限制目标位置在安全工作空间内。
        
        Args:
            target: 目标位置 [x, y, z]
            shoulder_pos: 肩部位置 [x, y, z]
            arm_name: 机械臂名称（用于日志）
            
        Returns:
            clamped_target: 限制后的目标位置
        """
        # 1. Check Max Reach
        rel_pos = target - shoulder_pos
        dist = np.linalg.norm(rel_pos)
        
        if dist > self.max_reach:
            # Clamp to max reach sphere
            scale = self.max_reach / dist
            rel_pos = rel_pos * scale
            target = shoulder_pos + rel_pos
            print(f"[WARN] {arm_name} target exceeds reach ({dist:.3f}m > {self.max_reach}m). Clamping.")
            
        # 2. Check Min Z
        if target[2] < self.min_z:
            target[2] = self.min_z
            # print(f"[WARN] {arm_name} target too low ({target[2]:.3f}m < {self.min_z}m). Clamping.")
            
        return target

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

    def _matrix_to_quaternion(self, R):
        """
        将3x3旋转矩阵转换为四元数 [x, y, z, w]。
        """
        # Manual implementation to avoid scipy dependency
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif (R[1,1] > R[2,2]):
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw])

    def _compute_orientation(self, wrist, index, pinky):
        """
        根据手腕、食指和尾指关键点计算手部朝向（四元数）。
        
        Args:
            wrist, index, pinky: [x, y, z] numpy arrays
            
        Returns:
            numpy.ndarray: [x, y, z, w] 四元数
        """
        # 1. 构建手部局部坐标系 (MediaPipe Frame)
        # Z轴: 手腕 -> 指尖 (前进方向)
        v_z = (index + pinky) / 2.0 - wrist
        v_z /= np.linalg.norm(v_z)
        
        # Y轴: 尾指 -> 食指 (大致指向拇指侧)
        v_y = index - pinky
        v_y /= np.linalg.norm(v_y)
        
        # X轴: 掌心法线 (Z x Y)
        v_x = np.cross(v_z, v_y)
        v_x /= np.linalg.norm(v_x)
        
        # 重新正交化 Y轴 (X x Z)
        v_y = np.cross(v_x, v_z)
        v_y /= np.linalg.norm(v_y)
        
        # 旋转矩阵 (列向量为基向量)
        # R_hand_mp = [v_x, v_y, v_z]
        R_hand_mp = np.column_stack((v_x, v_y, v_z))
        
        # 2. 转换到机器人坐标系
        # R_mp_to_robot
        # Robot X (Forward) = - MP Z
        # Robot Y (Left) = - MP X
        # Robot Z (Up) = - MP Y
        R_mp_to_robot = np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ])
        
        R_hand_robot = R_mp_to_robot @ R_hand_mp
        
        # 3. 转换为四元数 [x, y, z, w]
        # r = R.from_matrix(R_hand_robot)
        # return r.as_quat()
        return self._matrix_to_quaternion(R_hand_robot)

    def compute_target_pose(self, landmarks, mirror=False):
        """
        根据关键点计算左右臂的目标位置和姿态。
        
        Args:
            landmarks: MediaPipe的pose_world_landmarks.landmark列表。
            mirror (bool): 是否为镜像模式。如果为True，将修正左右方向的映射，防止动作交叉。
            
        Returns:
            tuple: (target_l, target_r, orient_l, orient_r, left_valid, right_valid)
        """
        if not landmarks:
            return None, None, None, None, False, False

        # Helper to get np array
        def get_vec(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

        # Define indices
        # 11: left_shoulder, 12: right_shoulder
        # 15: left_wrist, 16: right_wrist
        # 17: left_pinky, 19: left_index
        # 18: right_pinky, 20: right_index
        
        idx_l_s, idx_r_s = 11, 12
        idx_l_w, idx_r_w = 15, 16
        idx_l_p, idx_r_p = 17, 18
        idx_l_i, idx_r_i = 19, 20
        
        # Note: We do NOT swap indices even in mirror mode, because MediaPipe is smart enough 
        # to identify "Left" correctly even if it appears on the right side of the image.
        # Swapping indices caused the "Left controls Right" issue.

        # Extract Keypoints
        ls = get_vec(idx_l_s)
        rs = get_vec(idx_r_s)
        lw = get_vec(idx_l_w)
        rw = get_vec(idx_r_w)
        
        l_pinky = get_vec(idx_l_p)
        l_index = get_vec(idx_l_i)
        r_pinky = get_vec(idx_r_p)
        r_index = get_vec(idx_r_i)
        
        # Visibility check
        left_valid = landmarks[idx_l_w].visibility > 0.5
        right_valid = landmarks[idx_r_w].visibility > 0.5
        
        # Calculate Relative Vectors (Wrist relative to Shoulder)
        # 计算手腕相对于肩膀的向量
        vec_l = lw - ls
        vec_r = rw - rs
        
        # Apply transformation
        # 应用坐标变换和缩放
        robot_vec_l = self.transform_vec(vec_l) * self.scale_factor
        robot_vec_r = self.transform_vec(vec_r) * self.scale_factor
        
        if mirror:
            # Correction for mirroring:
            # In standard mapping (transform_vec), Robot Y = -MP X.
            # In mirror mode, User Left moves to Image Right (MP X+).
            # -MP X becomes Negative (Robot Right).
            # But we want Robot Left (Positive Y).
            # So we must invert the Y axis.
            robot_vec_l[1] = -robot_vec_l[1]
            robot_vec_r[1] = -robot_vec_r[1]

        # Absolute Target Position
        # 叠加到机器人肩部坐标上，得到绝对目标位置
        target_l = self.robot_shoulder_left + robot_vec_l
        target_r = self.robot_shoulder_right + robot_vec_r
        
        # Apply Workspace Limits
        # 应用工作空间限制
        target_l = self._limit_target(target_l, self.robot_shoulder_left, "Left Arm")
        target_r = self._limit_target(target_r, self.robot_shoulder_right, "Right Arm")
        
        # Calculate Orientation
        # 计算手部朝向
        orient_l = self._compute_orientation(lw, l_index, l_pinky)
        orient_r = self._compute_orientation(rw, r_index, r_pinky)

        return target_l, target_r, orient_l, orient_r, left_valid, right_valid

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
            
        # Index Tip: 8, Middle Tip: 12, Thumb Tip: 4
        # 获取拇指指尖(4)和中指指尖(12)的位置
        thumb = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z])
        middle = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z])
        
        # 计算拇指和中指的距离
        dist = np.linalg.norm(thumb - middle)
        
        # Map distance to 0-1. Heuristic: 0.02 (closed) to 0.15 (open)
        # 将距离映射到0-1之间。经验值：0.02m视为闭合，0.15m视为张开
        # ratio 1.0 means closed, 0.0 means open
        ratio = np.clip((0.15 - dist) / (0.15 - 0.02), 0.0, 1.0)
        return float(ratio)
