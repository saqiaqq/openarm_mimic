import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

class MimicCameraController:
    def __init__(self, node, use_ros_driver=False, device_id=2, color_topic='/camera/color/image_raw', depth_topic='/camera/depth/image_raw'):
        self.node = node
        self.use_ros_driver = use_ros_driver
        self.device_id = device_id
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        
        self.bridge = CvBridge()
        
        # State
        self.current_frame = None
        self.current_depth = None
        self.cap = None
        self.cap_depth = None
        self.current_device_index = device_id
        
        # Subscribers placeholders
        self.sub_color = None
        self.sub_depth = None
        self.ts = None

        # Initialize based on mode
        if self.use_ros_driver:
            self.setup_subscribers()
        else:
            self.setup_usb_camera()

    def setup_subscribers(self):
        self.node.get_logger().info("Waiting for camera topics...")
        # Synchronize Color and Depth
        # Note: message_filters.Subscriber takes the node as the first argument
        self.sub_color = message_filters.Subscriber(self.node, Image, self.color_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(self.node, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        
        # Use ApproximateTimeSynchronizer for loose syncing
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_color, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.topic_callback)

    def topic_callback(self, color_msg, depth_msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.current_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            self.node.get_logger().error(f"CV Bridge Error: {e}")

    def setup_usb_camera(self):
        self.current_device_index = self.device_id
        if not self.open_camera(self.device_id):
            self.node.get_logger().info(f"Camera {self.device_id} failed. Auto-searching...")
            self.find_best_camera()

    def find_best_camera(self):
        """Iterate through indices to find a working RGB camera"""
        candidates = []
        self.node.get_logger().info("Starting Camera Scan (0-20)...")
        
        for i in range(20):
            if i == self.device_id: continue 
            
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    
                    self.node.get_logger().info(f"Cam {i}: {w}x{h}")
                    
                    if h >= 400.0:
                        candidates.append((i, w, h))
                cap.release()
        
        self.node.get_logger().info(f"Scan complete. Candidates: {candidates}")
        
        # Select best candidate (prefer external > internal)
        selected_idx = -1
        external_candidates = [c for c in candidates if c[0] > 0]
        
        if external_candidates:
            selected_idx = external_candidates[0][0]
        elif candidates:
            selected_idx = candidates[0][0]
            
        if selected_idx >= 0:
            self.node.get_logger().info(f"Auto-selecting Camera {selected_idx}")
            self.open_camera(selected_idx)
        else:
            self.node.get_logger().warn("No suitable camera found.")

    def open_camera(self, index):
        """Try to open camera (RGB) and attempt to find associated Depth stream"""
        if self.cap is not None:
            self.cap.release()
        
        self.current_device_index = index
        self.node.get_logger().info(f"Opening RGB Camera Index: {index}")
        
        # RGB Stream
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
             cap = cv2.VideoCapture(index) # Fallback

        if cap.isOpened():
            # Set RGB Resolution
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            ret, frame = cap.read()
            if ret:    
                self.cap = cap
                self.node.get_logger().info(f"RGB Camera {index} ready.")
                
                # Attempt to open Depth
                self.node.get_logger().info("Attempting to find Depth stream (Software Match)...")
                self.try_open_depth(index)
                return True
        
        return False

    def try_open_depth(self, rgb_index):
        possible_indices = [rgb_index + 1, rgb_index + 2, rgb_index - 1]
        for idx in possible_indices:
            if idx < 0 or idx == rgb_index: continue
            
            cap_d = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap_d.isOpened():
                cap_d.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_d.set(cv2.CAP_PROP_FRAME_HEIGHT, 400) 
                
                ret, frame = cap_d.read()
                if ret:
                    h, w = frame.shape[:2]
                    if h == 400: # Found typical Gemini IR resolution
                        self.node.get_logger().info(f"Depth/IR Camera found at {idx} ({w}x{h})")
                        self.cap_depth = cap_d
                        return
                cap_d.release()
        self.node.get_logger().warn("Could not auto-detect Depth stream via USB.")

    def switch_camera(self, direction=1):
        if self.use_ros_driver:
            self.node.get_logger().warn("Cannot switch camera in ROS Driver mode.")
            return
        
        next_index = self.current_device_index + direction
        if next_index < 0: next_index = 0
        self.open_camera(next_index)

    def get_frames(self):
        """
        Returns (frame, depth).
        If frames are not available, returns (None, None) or partial data.
        """
        frame = None
        depth = None
        
        if self.use_ros_driver:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
            if self.current_depth is not None:
                depth = self.current_depth.copy()
        else:
            if self.cap is not None and self.cap.isOpened():
                ret, img = self.cap.read()
                if ret: frame = img
            
            if self.cap_depth is not None and self.cap_depth.isOpened():
                ret, d_img = self.cap_depth.read()
                if ret: depth = d_img
                
        return frame, depth

    def release(self):
        if self.cap: self.cap.release()
        if self.cap_depth: self.cap_depth.release()
