std_msgs/Header header

geometry_msgs/Pose left_arm_pose
geometry_msgs/Pose right_arm_pose

float64[] left_joint_angles
float64[] right_joint_angles
bool use_joint_mapping

float32 left_gripper_ratio   # 0.0 (open) to 1.0 (closed)
float32 right_gripper_ratio  # 0.0 (open) to 1.0 (closed)

bool left_track_valid
bool right_track_valid
