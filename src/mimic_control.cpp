#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include "openarm_mimic/msg/mimic_frame.hpp"
#include "openarm_mimic/robot_kdl.hpp"
#include <openarm/can/socket/openarm.hpp>
#include <openarm_port/openarm_init.hpp>
#include <controller/dynamics.hpp>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std::chrono_literals;

class MimicArmNode : public rclcpp::Node {
public:
    MimicArmNode() : Node("mimic_arm_node") {
        // Parameters
        this->declare_parameter("arm_side", "right_arm");
        this->declare_parameter("can_interface", "can0");
        this->declare_parameter("urdf_path", "/tmp/openarm_mimic.urdf");
        this->declare_parameter("kp", 30.0);
        this->declare_parameter("kd", 1.0);
        this->declare_parameter("smoothing_factor", 0.1); // 0.0 - 1.0

        arm_side_ = this->get_parameter("arm_side").as_string();
        can_interface_ = this->get_parameter("can_interface").as_string();
        urdf_path_ = this->get_parameter("urdf_path").as_string();
        kp_ = this->get_parameter("kp").as_double();
        kd_ = this->get_parameter("kd").as_double();
        alpha_ = this->get_parameter("smoothing_factor").as_double();

        RCLCPP_INFO(this->get_logger(), "Starting Mimic Node for %s on %s", arm_side_.c_str(), can_interface_.c_str());

        // Wait for URDF file
        while (!std::filesystem::exists(urdf_path_) && rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "Waiting for URDF file at %s...", urdf_path_.c_str());
            std::this_thread::sleep_for(1s);
        }

        // Initialize KDL & Dynamics
        std::string root_link = "openarm_body_link0";
        std::string leaf_link = (arm_side_ == "left_arm") ? "openarm_left_hand" : "openarm_right_hand";

        // Read URDF content for RobotKDL
        std::ifstream urdf_file(urdf_path_);
        std::stringstream buffer;
        buffer << urdf_file.rdbuf();
        std::string urdf_content = buffer.str();

        robot_kdl_ = std::make_shared<RobotKDL>(urdf_content, root_link, leaf_link);
        if (!robot_kdl_->init()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to init RobotKDL");
            throw std::runtime_error("KDL Init Failed");
        }

        dynamics_ = std::make_shared<Dynamics>(urdf_path_, root_link, leaf_link);
        if (!dynamics_->Init()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to init Dynamics");
            throw std::runtime_error("Dynamics Init Failed");
        }

        // Initialize Hardware
        try {
            openarm_ = std::shared_ptr<openarm::can::socket::OpenArm>(
                openarm_init::OpenArmInitializer::initialize_openarm(can_interface_, true)
            );
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to init Hardware: %s", e.what());
            throw;
        }

        num_joints_ = robot_kdl_->getNumJoints();
        
        // Subscription
        mimic_sub_ = this->create_subscription<openarm_mimic::msg::MimicFrame>(
            "/mimic/target_frame", 10,
            std::bind(&MimicArmNode::mimic_callback, this, std::placeholders::_1)
        );

        // Timer for Control Loop (100Hz)
        timer_ = this->create_wall_timer(
            10ms, std::bind(&MimicArmNode::control_loop, this)
        );

        RCLCPP_INFO(this->get_logger(), "Mimic Node Initialized. Waiting for targets...");
    }

    ~MimicArmNode() {
        if (openarm_) {
            openarm_->disable_all();
            openarm_->recv_all();
        }
    }

private:
    void mimic_callback(const openarm_mimic::msg::MimicFrame::SharedPtr msg) {
        last_frame_time_ = this->now();
        
        bool valid = (arm_side_ == "left_arm") ? msg->left_track_valid : msg->right_track_valid;
        if (!valid) return;

        geometry_msgs::msg::Pose pose = (arm_side_ == "left_arm") ? msg->left_arm_pose : msg->right_arm_pose;
        float gripper = (arm_side_ == "left_arm") ? msg->left_gripper_ratio : msg->right_gripper_ratio;

        // Update target
        target_pose_.resize(7);
        target_pose_[0] = pose.position.x;
        target_pose_[1] = pose.position.y;
        target_pose_[2] = pose.position.z;
        target_pose_[3] = pose.orientation.x;
        target_pose_[4] = pose.orientation.y;
        target_pose_[5] = pose.orientation.z;
        target_pose_[6] = pose.orientation.w;

        target_gripper_ = gripper;
        has_target_ = true;
    }

    void control_loop() {
        // 1. Read State
        openarm_->recv_all();
        auto motors = openarm_->get_arm().get_motors();
        
        std::vector<double> q_curr(num_joints_);
        std::vector<double> dq_curr(num_joints_);
        for(int i=0; i<num_joints_; ++i) {
            q_curr[i] = motors[i].get_position();
            dq_curr[i] = motors[i].get_velocity();
        }

        // 2. Compute Gravity
        std::vector<double> tau_g(num_joints_);
        dynamics_->GetGravity(q_curr.data(), tau_g.data());

        // 3. Determine Control Mode
        // Safety: if no target received for 500ms, fallback to gravity comp
        if (has_target_ && (this->now() - last_frame_time_).seconds() > 0.5) {
            has_target_ = false;
            RCLCPP_WARN(this->get_logger(), "Target lost, switching to gravity comp");
        }

        std::vector<double> q_cmd = q_curr; // Default to hold current or gravity comp
        double kp_run = 0.0;
        double kd_run = 0.0; // Use small damping in gravity comp?

        if (has_target_) {
            // Smooth Target Pose
            if (!initialized_pose_) {
                // Initialize current_pose_ from FK of current joint angles to prevent jump
                if (robot_kdl_->solveFK(q_curr, current_pose_)) {
                    initialized_pose_ = true;
                    RCLCPP_INFO(this->get_logger(), "Pose Initialized from FK");
                } else {
                    // Fallback if FK fails (unlikely)
                    current_pose_ = target_pose_;
                    initialized_pose_ = true;
                }
            } else {
                // Apply max velocity limit (safety)
                double max_step = 0.005; // 0.5 cm per 10ms = 0.5 m/s
                for(int i=0; i<3; ++i) {
                    double diff = target_pose_[i] - current_pose_[i];
                    // Exponential smoothing
                    double smooth_diff = diff * alpha_;
                    // Velocity clamp
                    if (std::abs(smooth_diff) > max_step) {
                         smooth_diff = (smooth_diff > 0 ? max_step : -max_step);
                    }
                    current_pose_[i] += smooth_diff;
                }
                
                // Orientation: Fixed for now or smooth slerp if implemented
                current_pose_[3] = target_pose_[3];
                current_pose_[4] = target_pose_[4];
                current_pose_[5] = target_pose_[5];
                current_pose_[6] = target_pose_[6];
            }

            // IK Solver
            std::vector<double> q_ik_out;
            if (robot_kdl_->solveIK(q_curr, current_pose_, q_ik_out)) {
                q_cmd = q_ik_out;
                kp_run = kp_;
                kd_run = kd_;
            } else {
                // IK Failed, keep current?
                // RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "IK Failed");
            }
        }

        // 4. Send Command
        std::vector<openarm::damiao_motor::MITParam> cmds;
        cmds.reserve(num_joints_);
        
        for(int i=0; i<num_joints_; ++i) {
            cmds.push_back({
                (float)kp_run, 
                (float)kd_run, 
                (float)q_cmd[i], 
                0.0f, // Velocity target 0
                (float)tau_g[i]
            });
        }
        openarm_->get_arm().mit_control_all(cmds);

        // Gripper Control
        // Mapping: 0.0 (Open) -> Min Pos, 1.0 (Closed) -> Max Pos
        // For V10 Gripper: 0 is Open? Need to check.
        // Usually 0 is Open, Positive is Closed? Or Vice versa.
        // Assuming: 0.0 is Open, 1.0 is Closed.
        // Target: map 0.0-1.0 to motor position.
        // Let's assume motor range 0 to 5.0 (radians approx) for full close?
        // Or use MIT control for gripper too.
        if (openarm_->get_gripper().get_motors().size() > 0) {
             float gripper_target = target_gripper_ * 3.0f; // Approx range check needed
             openarm_->get_gripper().mit_control_one(0, {
                 10.0f, 0.1f, gripper_target, 0.0f, 0.0f
             });
        }
    }

    std::string arm_side_;
    std::string can_interface_;
    std::string urdf_path_;
    double kp_, kd_, alpha_;
    
    std::shared_ptr<RobotKDL> robot_kdl_;
    std::shared_ptr<Dynamics> dynamics_;
    std::shared_ptr<openarm::can::socket::OpenArm> openarm_;
    int num_joints_;

    rclcpp::Subscription<openarm_mimic::msg::MimicFrame>::SharedPtr mimic_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    bool has_target_ = false;
    rclcpp::Time last_frame_time_;
    std::vector<double> target_pose_;
    std::vector<double> current_pose_;
    float target_gripper_;
    bool initialized_pose_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MimicArmNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
