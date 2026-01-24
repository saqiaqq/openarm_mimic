#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from openarm_mimic.msg import MimicFrame
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np

class MimicVisionNode(Node):
    def __init__(self):
        super().__init__('mimic_vision_node')
        
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('mirror', True)
        self.declare_parameter('device_id', 2) # Default to 2 for external USB camera
        
        self.image_topic = self.get_parameter('image_topic').value
        self.mirror = self.get_parameter('mirror').value
        self.device_id = self.get_parameter('device_id').value
        
        self.sub = None 
        self.pub = self.create_publisher(MimicFrame, '/mimic/target_frame', 10)
        
        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1)
            
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Calibration / Offsets (Meters)
        self.robot_shoulder_left = np.array([0.0, 0.20, 0.35]) 
        self.robot_shoulder_right = np.array([0.0, -0.20, 0.35])
        
        self.scale_factor = 1.0

        # Camera source
        self.current_device_index = 0
        # Try to open initial camera (prefer 0, usually webcam or primary)
        # If Gemini is plugged in, it might be 0, 1, or /dev/video*
        # We will try to find the FIRST working one.
        if not self.open_camera(0):
            self.get_logger().info("Camera 0 failed. Use 'n' to switch.")
            # self.find_next_working_camera(0, 1) # Removed blocking call
        
        self.timer = self.create_timer(0.033, self.timer_callback) # 30 FPS

        self.get_logger().info("Mimic Vision Node Started")

    def open_camera(self, index):
        """Try to open camera with V4L2 and MJPG to ensure RGB stream"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Update index immediately so UI shows what we are trying
        self.current_device_index = index
        self.get_logger().info(f"Attempting to open Camera Index: {index}")
        
        # Prioritize V4L2 backend for Linux/ROS
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        if not cap.isOpened():
             # Fallback to default
             self.get_logger().warn(f"Failed to open camera {index} with V4L2. Trying default backend...")
             cap = cv2.VideoCapture(index)

        if cap.isOpened():
            # Force MJPG format - this is CRITICAL for many UVC cameras (like Gemini) to give RGB instead of Raw
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Warmup / Check
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self.get_logger().info(f"Camera {index} opened. Resolution: {w}x{h}")
                self.cap = cap
                return True
            else:
                 self.get_logger().warn(f"Camera {index} opened but returned empty frame. Closing.")
                 cap.release()
        else:
            self.get_logger().error(f"Failed to open camera {index}")
            
        return False

    def switch_camera(self, direction=1):
        """Switch to next/prev camera index without blocking loop"""
        next_index = self.current_device_index + direction
        if next_index < 0:
            next_index = 0
        # No upper limit check, just let it grow or user can go back
        
        self.open_camera(next_index)

    def timer_callback(self):
        # 1. Handle Input (Always allow switching)
        # We need ONE main loop structure for waitKey
        
        frame = None
        has_frame = False

        # Try to read frame if camera is open
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            ret, img = self.cap.read()
            if ret:
                frame = img
                has_frame = True
        
        # If no frame, create blank
        if not has_frame:
            frame = np.zeros((480, 640, 3), np.uint8)
            msg = "No Signal" if (not hasattr(self, 'cap') or self.cap is None) else "Read Failed"
            cv2.putText(frame, f"Cam {self.current_device_index}: {msg}", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'n'/'p' to switch", (50, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
            # Display Info on valid frame
            cv2.putText(frame, f"Cam: {self.current_device_index} (n/p to switch)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # SHOW WINDOW FIRST
        cv2.imshow("Mimic Vision", frame)
        
        # THEN WAIT KEY (Essential for GUI events)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            self.get_logger().info("Key 'n' pressed: Switching to next camera")
            self.switch_camera(1)
            # Don't process this frame if we just switched
            return
        elif key == ord('p'):
            self.get_logger().info("Key 'p' pressed: Switching to prev camera")
            self.switch_camera(-1)
            return

        # ONLY PROCESS IF WE HAVE A VALID FRAME
        if has_frame:
            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image_rgb)
                self.process_results(results, frame) # This handles drawing landmarks
                # Re-show to update with landmarks
                cv2.imshow("Mimic Vision", frame) 
                cv2.waitKey(1) # Small delay for update
            except Exception as e:
                self.get_logger().error(f"Processing Error: {e}")

    def process_results(self, results, image):
        # Draw Logic
        if results.pose_landmarks:
             self.mp_drawing.draw_landmarks(
                 image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
             self.mp_drawing.draw_landmarks(
                 image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
             self.mp_drawing.draw_landmarks(
                 image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        # Update Window
        # cv2.imshow("Mimic Vision", image) # Moved to timer_callback for single imshow point
        # cv2.waitKey(1) # Moved to timer_callback


        # ... Existing logic extraction ...
        # frame_msg = MimicFrame() # Removed duplicate initialization
        # frame_msg.header.stamp = self.get_clock().now().to_msg()
        # ...

        frame_msg = MimicFrame()
        frame_msg.header.stamp = self.get_clock().now().to_msg()
        frame_msg.header.frame_id = "openarm_body_link0"
        
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # Extract Keypoints
            # 11: left_shoulder, 12: right_shoulder
            # 15: left_wrist, 16: right_wrist
            # 19: left_index, 20: right_index (tips? No, 19/20 are index_1)
            # Actually MP Pose: 15=L_Wrist, 16=R_Wrist.
            # Hands details are better with MP Hands, but Pose has basics.
            
            # Helper to get np array
            def get_vec(idx):
                return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

            ls = get_vec(11)
            rs = get_vec(12)
            lw = get_vec(15)
            rw = get_vec(16)
            
            # Visibility check
            frame_msg.left_track_valid = landmarks[15].visibility > 0.5
            frame_msg.right_track_valid = landmarks[16].visibility > 0.5
            
            # Calculate Relative Vectors (Wrist relative to Shoulder)
            # MP Coordinates: +X Left, +Y Down, +Z Back (World) - Check doc
            # Let's assume standard MP World: Origin Hip.
            # We use vectors, so origin doesn't matter.
            
            vec_l = lw - ls
            vec_r = rw - rs
            
            # Mapping to Robot Frame
            # Robot: +X Forward, +Y Left, +Z Up
            # Camera facing User:
            # User moves Hand Right (+X in MP? No, MP Left is +X).
            # Let's calibrate directions:
            # User Right Hand moves Right (User's Right).

            # Simple Mapping (Scale Factor)
            # MP Y (Down) -> Robot -Z (Down)
            # MP X (Left) -> Robot +Y (Left)
            # MP Z (Back?) -> Robot -X (Back?) - Needs tuning
            
            # Left Arm
            # Robot Target = Robot Shoulder + Rotated/Scaled Vector
            
            # Mapping MP (x,y,z) to Robot (x,y,z)
            # MP x (Left) -> Robot y (Left)
            # MP y (Down) -> Robot -z (Down)
            # MP z (Mid-Hips center?) -> Robot -x (Back)
            
            # Let's try:
            # Robot X (Forward) = - MP Z (Forward in MP is -Z)
            # Robot Y (Left) = MP X
            # Robot Z (Up) = - MP Y
            
            scale = self.scale_factor
            
            l_target = self.robot_shoulder_left + np.array([-vec_l[2], vec_l[0], -vec_l[1]]) * scale
            r_target = self.robot_shoulder_right + np.array([-vec_r[2], vec_r[0], -vec_r[1]]) * scale
            
            frame_msg.left_arm_pose.position.x = l_target[0]
            frame_msg.left_arm_pose.position.y = l_target[1]
            frame_msg.left_arm_pose.position.z = l_target[2]
            
            # Orientation - Keep simple for now (Forward/Down)
            # Or identity? Let's use fixed orientation for now.
            # Ideally calculate from Elbow-Wrist vector
            frame_msg.left_arm_pose.orientation.w = 1.0
            
            frame_msg.right_arm_pose.position.x = r_target[0]
            frame_msg.right_arm_pose.position.y = r_target[1]
            frame_msg.right_arm_pose.position.z = r_target[2]
            frame_msg.right_arm_pose.orientation.w = 1.0

            # Gripper Logic (Thumb-Index Distance)
            # Left Hand
            if results.left_hand_landmarks:
                lh = results.left_hand_landmarks.landmark
                # 4: Thumb Tip, 8: Index Tip
                dist = np.linalg.norm(
                    np.array([lh[4].x, lh[4].y, lh[4].z]) - 
                    np.array([lh[8].x, lh[8].y, lh[8].z]))
                # Map 0.02 (Closed) - 0.15 (Open) to 1.0 - 0.0
                ratio = np.clip((0.15 - dist) / (0.15 - 0.02), 0.0, 1.0)
                frame_msg.left_gripper_ratio = float(ratio)
                
            # Right Hand
            if results.right_hand_landmarks:
                rh = results.right_hand_landmarks.landmark
                dist = np.linalg.norm(
                    np.array([rh[4].x, rh[4].y, rh[4].z]) - 
                    np.array([rh[8].x, rh[8].y, rh[8].z]))
                ratio = np.clip((0.15 - dist) / (0.15 - 0.02), 0.0, 1.0)
                frame_msg.right_gripper_ratio = float(ratio)

            self.pub.publish(frame_msg)
            # In image (mirrored), hand moves Right.
            # MP X coordinate increases? 
            # MP: "x coordinates are negative when the coordinate is to the left of the user"
            # So User Right is +X.
            # Robot Right is -Y.
            # So Robot_Y ~ -MP_X.
            
            # User moves Hand Up.
            # MP Y decreases (negative is up).
            # Robot Z increases.
            # So Robot_Z ~ -MP_Y.
            
            # User moves Hand Forward (to camera).
            # MP Z decreases (negative is front).
            # Robot X increases.
            # So Robot_X ~ -MP_Z.
            
            # Apply transformation
            def transform_vec(v):
                # v is [x, y, z] in MP
                # returns [x, y, z] in Robot
                return np.array([-v[2], -v[0], -v[1]])
            
            robot_vec_l = transform_vec(vec_l) * self.scale_factor
            robot_vec_r = transform_vec(vec_r) * self.scale_factor
            
            # Absolute Target Position
            target_l = self.robot_shoulder_left + robot_vec_l
            target_r = self.robot_shoulder_right + robot_vec_r
            
            # Fill Message
            def fill_pose(pose_msg, target_pos):
                pose_msg.position.x = target_pos[0]
                pose_msg.position.y = target_pos[1]
                pose_msg.position.z = target_pos[2]
                # Default Orientation (Forward/Inwards)
                # Quaternion for identity?
                pose_msg.orientation.x = 0.0
                pose_msg.orientation.y = 0.0
                pose_msg.orientation.z = 0.0
                pose_msg.orientation.w = 1.0 
            
            fill_pose(frame_msg.left_arm_pose, target_l)
            fill_pose(frame_msg.right_arm_pose, target_r)
            
            # Gripper Logic using Hand Landmarks
            # Index Tip: 8, Thumb Tip: 4
            def get_gripper_ratio(hand_landmarks):
                if not hand_landmarks:
                    return 0.0
                thumb = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z])
                index = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z])
                dist = np.linalg.norm(thumb - index)
                # Map distance to 0-1. Heuristic: 0.02 (closed) to 0.15 (open)
                # Note: These are normalized coordinates relative to hand bounding box usually? 
                # Holistic hand landmarks are in world coordinates? No, they are normalized to image or hand ROI.
                # Actually for holistic, right_hand_landmarks are normalized.
                # Let's use simple heuristic.
                ratio = np.clip((0.15 - dist) / (0.15 - 0.02), 0.0, 1.0)
                # If dist is large (0.15), ratio is 0 (Open).
                # If dist is small (0.02), ratio is 1 (Closed).
                # Wait, user said: "close -> gripper close". 
                # So dist small -> ratio should be 1.0 (Closed).
                return ratio

            frame_msg.left_gripper_pos = get_gripper_ratio(results.left_hand_landmarks)
            frame_msg.right_gripper_pos = get_gripper_ratio(results.right_hand_landmarks)

        self.pub.publish(frame_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MimicVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
