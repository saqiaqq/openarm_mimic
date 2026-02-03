from openarm_mimic.msg import MimicFrame
from geometry_msgs.msg import Pose

class MimicCommandPublisher:
    def __init__(self, node):
        self.node = node
        self.pub = node.create_publisher(MimicFrame, '/mimic/target_frame', 10)

    def publish_frame(self, target_l, target_r, left_valid, right_valid, left_gripper, right_gripper):
        """
        Construct and publish the MimicFrame message.
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
        pose_msg.position.x = target_pos[0]
        pose_msg.position.y = target_pos[1]
        pose_msg.position.z = target_pos[2]
        # Default Orientation (Forward/Inwards)
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = 0.0
        pose_msg.orientation.w = 1.0 
