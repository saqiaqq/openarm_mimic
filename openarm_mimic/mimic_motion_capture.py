"""
Mimic Motion Capture Module
===========================
此模块封装了MediaPipe Holistic模型，用于进行人体姿态估计。

主要功能：
1. 初始化MediaPipe Holistic模型。
2. 处理图像帧，提取人体骨骼关键点。
3. 在图像上绘制检测到的关键点。
"""

import mediapipe as mp
import cv2
import numpy as np

class MimicMotionCapture:
    """
    动作捕捉类，使用MediaPipe Holistic进行全身姿态估计。
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0):
        """
        初始化MediaPipe Holistic模型。
        
        Args:
            min_detection_confidence (float): 最小检测置信度阈值 [0.0, 1.0]。
            min_tracking_confidence (float): 最小跟踪置信度阈值 [0.0, 1.0]。
            model_complexity (int): 模型复杂度 (0, 1, 2)。0最快，2最准。
        """
        # MediaPipe Holistic initialization
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity)
        
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        处理单帧图像并返回MediaPipe结果。
        
        Args:
            frame (numpy.ndarray): BGR格式的输入图像。
            
        Returns:
            results: MediaPipe处理结果对象，包含pose_landmarks等。
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        return results

    def draw_landmarks(self, image, results):
        """
        在图像上绘制检测到的关键点（原地修改）。
        
        Args:
            image (numpy.ndarray): 要绘制的图像。
            results: MediaPipe处理结果对象。
        """
        if results.pose_landmarks:
             self.mp_drawing.draw_landmarks(
                 image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
             self.mp_drawing.draw_landmarks(
                 image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
             self.mp_drawing.draw_landmarks(
                 image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
