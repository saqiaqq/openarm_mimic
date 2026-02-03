import numpy as np

class MimicCoordinateMapping:
    def __init__(self, scale_factor=1.0):
        # Calibration / Offsets (Meters)
        self.robot_shoulder_left = np.array([0.0, 0.20, 0.35]) 
        self.robot_shoulder_right = np.array([0.0, -0.20, 0.35])
        self.scale_factor = scale_factor

    def transform_vec(self, v):
        """
        Transform vector from MediaPipe frame to Robot frame.
        v is [x, y, z] in MediaPipe
        returns [x, y, z] in Robot
        
        Mapping Logic:
        - Robot X (Forward) = - MP Z
        - Robot Y (Left) = - MP X (So User Right (+X) becomes Robot Right (-Y))
        - Robot Z (Up) = - MP Y (So User Up (-Y) becomes Robot Up (+Z))
        """
        return np.array([-v[2], -v[0], -v[1]])

    def compute_target_pose(self, landmarks):
        """
        Compute target positions for left and right arms based on landmarks.
        Returns (target_l, target_r, left_valid, right_valid)
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
        vec_l = lw - ls
        vec_r = rw - rs
        
        # Apply transformation
        robot_vec_l = self.transform_vec(vec_l) * self.scale_factor
        robot_vec_r = self.transform_vec(vec_r) * self.scale_factor
        
        # Absolute Target Position
        target_l = self.robot_shoulder_left + robot_vec_l
        target_r = self.robot_shoulder_right + robot_vec_r

        return target_l, target_r, left_valid, right_valid

    def get_gripper_ratio(self, hand_landmarks):
        """
        Calculate gripper open/close ratio (0.0 to 1.0) based on hand landmarks.
        """
        if not hand_landmarks:
            return 0.0
            
        # Index Tip: 8, Thumb Tip: 4
        thumb = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z])
        index = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z])
        
        dist = np.linalg.norm(thumb - index)
        
        # Map distance to 0-1. Heuristic: 0.02 (closed) to 0.15 (open)
        # ratio 1.0 means closed, 0.0 means open
        ratio = np.clip((0.15 - dist) / (0.15 - 0.02), 0.0, 1.0)
        return float(ratio)
