import mediapipe as mp
import cv2
import numpy as np

class MimicMotionCapture:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1):
        # MediaPipe Holistic initialization
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity)
        
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Process a BGR frame and return the MediaPipe results.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        return results

    def draw_landmarks(self, image, results):
        """
        Draw landmarks on the image in-place.
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
