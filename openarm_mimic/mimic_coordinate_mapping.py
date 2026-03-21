"""
Mimic Coordinate Mapping Module
===============================
此模块负责将MediaPipe检测到的3D关键点坐标映射到机器人的坐标系中。

主要功能：
1. 坐标系转换（MediaPipe -> Robot）。
2. 计算机械臂末端执行器的目标位置。
3. 计算手爪的开合比例。


映射算法提示词：

现在针对MediaPipe出来的3维空间的姿态模式数据我有一个想法（里面还有一些是todo或者第三点还有问题，你可以帮忙优化一下），如图1的姿态图序号，图2为手部特征图序号，以右手臂为例：
一、以11号点为基座标点，经过点11、12和23的平面为基准面a，经过点11并且垂直于线段11-12的平面为基准面b,经过线段11-12并且垂直于基准面b和a的平面为基准面c，那么基准面a、b和c在3维空中是两两垂直的平面，那么设基准面a和b的交线为AB，基准面a和c的交线为AC，基准面b和c的交线为BC。
    1、现在把线段11-13投影到基准面b的如果是线段B11-13（点11向13方向），那么它和射线AB11（起点为11，和直线AB平行，方向为点11指向23的大方向）的夹角A1映射到右手臂的1号电机的转角；
    2、如果把线段11-13投影到基准面b的是圆点，。。。todo
    3、把线段11-13投影到基准面a的如果是线段A11-13（点11向13方向），那么它和射线AB11（起点为11，和直线AB平行，方向11指向23的大方向）的夹角A2映射到右手臂的2号电机转角；
    4、如果把线段11-13投影到基准面a的是圆点，。。。todo
二、以通过点11并且垂直于线段11-13的平面称为基准面d，基准面d上经过点11，平行直线BC的射线BC11,方向为右手法则（四指从线段11-12向线段11-23方向握拳，大拇指竖起），大拇指的方向，经过点11，并且垂直于基准面d和平行直线BC（并且经过直线BC）的平面为基准面e，基准面d和基准面e的交线为DE。
    1、如果线段13-15在平面d上的投影是一根线段D13-15，那么线段D13-15和射线BC11的夹角A3即为右手臂的3号电机的转角；
    2、如果线段13-15在平面d上的投影是一个圆点，那么右手臂的3号电机的转角就按照逻辑0（下面会介绍）来计算；
    3、线段13-15（13向15方向）和线段11-13（11向13方向）的夹角A4为右手臂4号电机的转角；
三、以通过点15并且垂直于线段15-13的平面称为基准面f，如果11、13和15三点不共线，那么由11、13和15三点组成的平面为基准面j，如果11、13和15三点共线，那么。。。。(todo)为基准面j，过点15并且分别垂直于基准面f和基准面j的为基准面h，基准面f和基准面j的交线为FJ，基准面f和基准面h的交线为FH，基准面J和基准面H的交线为JH。
    1、线段17-19在基准面f上面的投影F17-19（点17向19方向）和射线FJ15（起点为15，和直线FJ平行，方向15指向13的大方向）的夹角A5即为右手臂的5号电机的转角；
    2、线段17-19在基准面j上面的投影J17-19（点17向19方向）和射线JH15（起点为15，和直线JH平行，方向15指向13的大方向）的夹角A7即为右手臂的7号电机的转角；
    3、线段17-19在基准面h上面的投影H17-19（点17向19方向）和射线FH15（起点为15，和直线FH平行，方向15指向13的大方向）的夹角A6即为右手臂的6号电机的转角；
四、补充逻辑0：线段17-19在基准面e上面的投影E17-19（点17向19方向）和射线DE11（起点为15，和直线DE平行，方向15指向13的大方向）的夹角A8，那么A8-A5的夹角为右手臂的3号电机的转角。
解读一下这个mimic_coordinate_mapping.py文件映射逻辑，和我这里的逻辑有什么差异，如果不通，那么两个逻辑各有什么优缺点

"""

import numpy as np
# from scipy.spatial.transform import Rotation as R

class MimicCoordinateMapping:
    """
    坐标映射类，处理空间坐标变换和目标位姿计算。
    """
    def __init__(self, scale_factor=1.2):
        """
        初始化映射参数。
        
        Args:
            scale_factor (float): 动作缩放因子，用于调整动作幅度。
                                  默认为1.2，以确保用户手臂伸直时能映射到机器人的最大行程（配合clamping）。
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
        self.max_reach = 0.68  # Maximum reach radius from shoulder (meters) - Slightly reduced for safety
        self.min_z = 0.05      # Minimum Z height (meters) to avoid hitting table
        
        # Initial human pose offsets (for relative motion)
        self.initial_human_pose = None
        self.is_calibrated = False

    def reset_origin(self, landmarks):
        """
        重置原点，以当前人体姿态作为初始状态。
        同时根据用户手臂长度自动计算最佳 scale_factor。
        """
        if not landmarks:
            return False
            
        def get_vec(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

        # Extract Keypoints for calibration
        ls = get_vec(11) # Left Shoulder
        rs = get_vec(12) # Right Shoulder
        le = get_vec(13) # Left Elbow
        re = get_vec(14) # Right Elbow
        lw = get_vec(15) # Left Wrist
        rw = get_vec(16) # Right Wrist
        
        # Calculate User Arm Lengths (Shoulder -> Elbow -> Wrist)
        # 计算用户手臂长度
        len_l = np.linalg.norm(le - ls) + np.linalg.norm(lw - le)
        len_r = np.linalg.norm(re - rs) + np.linalg.norm(rw - re)
        avg_user_arm_len = (len_l + len_r) / 2.0
        
        # Calculate Optimal Scale Factor
        # 目标：Scale * User_Arm = Robot_Reach
        # => Scale = Robot_Reach / User_Arm
        if avg_user_arm_len > 0.1: # Prevent division by zero or noise
            # 使用 max_reach (0.68m) 作为机器人参考臂长
            calculated_scale = self.max_reach / avg_user_arm_len
            
            # Sanity Check: Ensure scale factor is within reasonable bounds (0.8 ~ 2.0)
            # 如果计算出的比例太离谱（例如用户离得太远导致测量很小，或者误检测），则保持默认值
            # 默认值通常在 1.2 左右
            if 0.8 <= calculated_scale <= 2.0:
                print(f"[Calibration] User Arm Length: {avg_user_arm_len:.3f}m. Robot Reach: {self.max_reach}m.")
                print(f"[Calibration] Auto-tuning scale_factor to {calculated_scale:.3f} (was {self.scale_factor})")
                self.scale_factor = calculated_scale
            else:
                print(f"[Calibration] Calculated scale {calculated_scale:.3f} out of reasonable range (0.8-2.0). Keeping default {self.scale_factor}.")
        
        # Store initial wrist positions relative to shoulders
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
        idx_l_e, idx_r_e = 13, 14
        idx_l_w, idx_r_w = 15, 16
        idx_l_p, idx_r_p = 17, 18
        idx_l_i, idx_r_i = 19, 20
        
        # Note: We do NOT swap indices even in mirror mode, because MediaPipe is smart enough 
        # to identify "Left" correctly even if it appears on the right side of the image.
        # Swapping indices caused the "Left controls Right" issue.

        # Extract Keypoints
        ls = get_vec(idx_l_s)
        rs = get_vec(idx_r_s)
        le = get_vec(idx_l_e)
        re = get_vec(idx_r_e)
        lw = get_vec(idx_l_w)
        rw = get_vec(idx_r_w)
        
        l_pinky = get_vec(idx_l_p)
        l_index = get_vec(idx_l_i)
        r_pinky = get_vec(idx_r_p)
        r_index = get_vec(idx_r_i)
        
        # Visibility check
        left_valid = landmarks[idx_l_w].visibility > 0.5
        right_valid = landmarks[idx_r_w].visibility > 0.5
        
        # Dynamic Scaling (Extension Boosting)
        # Calculate extension ratio: current_dist / total_arm_length
        # Left Arm
        l_arm_len = np.linalg.norm(le - ls) + np.linalg.norm(lw - le)
        l_curr_dist = np.linalg.norm(lw - ls)
        l_ratio = l_curr_dist / (l_arm_len + 1e-6)
        
        scale_l = self.scale_factor
        if l_ratio > 0.85:
            # Boost scale when arm is nearly straight to ensure full reach
            # Boost up to 1.3x when fully extended
            scale_l = self.scale_factor * (1.0 + 0.8 * (l_ratio - 0.85))
            
        # Right Arm
        r_arm_len = np.linalg.norm(re - rs) + np.linalg.norm(rw - re)
        r_curr_dist = np.linalg.norm(rw - rs)
        r_ratio = r_curr_dist / (r_arm_len + 1e-6)
        
        scale_r = self.scale_factor
        if r_ratio > 0.85:
            scale_r = self.scale_factor * (1.0 + 0.8 * (r_ratio - 0.85))

        # Calculate Relative Vectors (Wrist relative to Shoulder)
        # 计算手腕相对于肩膀的向量
        vec_l = lw - ls
        vec_r = rw - rs
        
        # Apply transformation
        # 应用坐标变换和缩放
        robot_vec_l = self.transform_vec(vec_l) * scale_l
        robot_vec_r = self.transform_vec(vec_r) * scale_r
        
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
