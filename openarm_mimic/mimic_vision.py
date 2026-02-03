#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np

# Import Modules
from openarm_mimic.mimic_motion_capture import MimicMotionCapture
from openarm_mimic.mimic_coordinate_mapping import MimicCoordinateMapping
from openarm_mimic.mimic_command_publisher import MimicCommandPublisher
from openarm_mimic.mimic_camera_controller import MimicCameraController

class MimicVisionNode(Node):
    def __init__(self):
        super().__init__('mimic_vision_node')
        
        # Parameters
        self.declare_parameter('use_ros_driver', False) 
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw') 
        self.declare_parameter('mirror', True)
        self.declare_parameter('device_id', 2) # Default to 2 for external USB camera
        
        self.use_ros_driver = self.get_parameter('use_ros_driver').value
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.mirror = self.get_parameter('mirror').value
        self.device_id = self.get_parameter('device_id').value
        
        # Initialize Modules
        self.mocap = MimicMotionCapture()
        self.mapper = MimicCoordinateMapping()
        self.publisher = MimicCommandPublisher(self)
        
        # Initialize Camera Controller
        self.camera = MimicCameraController(
            self, 
            use_ros_driver=self.use_ros_driver,
            device_id=self.device_id,
            color_topic=self.color_topic,
            depth_topic=self.depth_topic
        )
        
        # Timer for processing loop (30 FPS)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info(f"Mimic Vision Node Started. Mode: {'ROS Driver' if self.use_ros_driver else 'USB Direct'}")

    def timer_callback(self):
        # 1. Acquire Frame
        frame, depth = self.camera.get_frames()

        # 2. Visualization & Processing
        if frame is None:
            if not self.use_ros_driver:
                # Show waiting screen
                blank = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank, "Waiting for Camera...", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Mimic Vision", blank)
            else:
                return # Don't show anything if no topic data yet
        else:
            try:
                # Mirror if needed
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                
                # --- Motion Capture ---
                results = self.mocap.process_frame(frame)
                self.mocap.draw_landmarks(frame, results)
                
                # --- Coordinate Mapping ---
                if results.pose_world_landmarks:
                    target_l, target_r, left_valid, right_valid = \
                        self.mapper.compute_target_pose(results.pose_world_landmarks.landmark)
                    
                    left_gripper = self.mapper.get_gripper_ratio(results.left_hand_landmarks)
                    right_gripper = self.mapper.get_gripper_ratio(results.right_hand_landmarks)
                    
                    # --- Command Publishing ---
                    self.publisher.publish_frame(target_l, target_r, left_valid, right_valid, left_gripper, right_gripper)

                # --- Visualization ---
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
                    cv2.imshow("Mimic Vision (RGB + Depth)", combined)
                else:
                    cv2.imshow("Mimic Vision", frame)
                    
            except Exception as e:
                self.get_logger().error(f"Processing Error: {e}")

        # 3. Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            self.camera.switch_camera(1)
        elif key == ord('p'):
            self.camera.switch_camera(-1)
        elif key == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MimicVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.camera.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
