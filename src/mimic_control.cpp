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
#include <thread>
#include <cmath>
#include <algorithm>

using namespace std::chrono_literals;

/**
 * @class MimicArmNode
 * @brief 机械臂模仿控制节点
 * 
 * 此节点负责接收来自视觉模块的目标位姿指令，
 * 利用 KDL 进行逆运动学解算，并结合动力学前馈（重力补偿），
 * 通过 CAN 总线控制 OpenArm 机械臂运动。
 */
class MimicArmNode : public rclcpp::Node {
public:
    /**
     * @brief 构造函数
     * 初始化 ROS 参数、KDL 求解器、动力学模型和硬件接口。
     */
    MimicArmNode() : Node("mimic_arm_node") {
        // Parameters
        // 声明参数
        this->declare_parameter("arm_side", "right_arm");       // 机械臂侧别：left_arm 或 right_arm
        this->declare_parameter("can_interface", "can0");       // CAN 接口名称
        this->declare_parameter("urdf_path", "/tmp/openarm_mimic.urdf"); // URDF 文件路径
        this->declare_parameter("kp", 30.0);                    // 位置控制增益
        this->declare_parameter("kd", 1.0);                     // 速度/阻尼控制增益
        this->declare_parameter("smoothing_factor", 0.1);       // 轨迹平滑因子 (0.0 - 1.0)

        // 获取参数值
        arm_side_ = this->get_parameter("arm_side").as_string();
        can_interface_ = this->get_parameter("can_interface").as_string();
        urdf_path_ = this->get_parameter("urdf_path").as_string();
        kp_ = this->get_parameter("kp").as_double();
        kd_ = this->get_parameter("kd").as_double();
        alpha_ = this->get_parameter("smoothing_factor").as_double();

        RCLCPP_INFO(this->get_logger(), "Starting Mimic Node for %s on %s", arm_side_.c_str(), can_interface_.c_str());

        // Wait for URDF file
        // 等待 URDF 文件生成（通常由 launch 文件中的 robot_state_publisher 生成）
        while (!std::filesystem::exists(urdf_path_) && rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "Waiting for URDF file at %s...", urdf_path_.c_str());
            std::this_thread::sleep_for(1s);
        }

        // Initialize KDL & Dynamics
        // 初始化 KDL 和动力学库
        std::string root_link = "openarm_body_link0";
        std::string leaf_link = (arm_side_ == "left_arm") ? "openarm_left_hand" : "openarm_right_hand";

        // Read URDF content for RobotKDL
        // 读取 URDF 内容
        std::ifstream urdf_file(urdf_path_);
        std::stringstream buffer;
        buffer << urdf_file.rdbuf();
        std::string urdf_content = buffer.str();

        // 初始化运动学求解器
        robot_kdl_ = std::make_shared<RobotKDL>(urdf_content, root_link, leaf_link);
        if (!robot_kdl_->init()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to init RobotKDL");
            throw std::runtime_error("KDL Init Failed");
        }

        // 初始化动力学求解器（用于重力补偿）
        dynamics_ = std::make_shared<Dynamics>(urdf_path_, root_link, leaf_link);
        if (!dynamics_->Init()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to init Dynamics");
            throw std::runtime_error("Dynamics Init Failed");
        }

        // Initialize Hardware
        // 初始化 OpenArm 硬件通信
        try {
            openarm_ = std::shared_ptr<openarm::can::socket::OpenArm>(
                openarm_init::OpenArmInitializer::initialize_openarm(can_interface_, true)
            );
            
            // Enable motors
            RCLCPP_INFO(this->get_logger(), "Enabling motors...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            openarm_->get_arm().enable_all();
            RCLCPP_INFO(this->get_logger(), "Motors Enabled.");
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to init Hardware: %s", e.what());
            throw;
        }

        num_joints_ = robot_kdl_->getNumJoints();
        
        // Subscription
        // 订阅视觉节点发布的目标位姿
        mimic_sub_ = this->create_subscription<openarm_mimic::msg::MimicFrame>(
            "/mimic/target_frame", 10,
            std::bind(&MimicArmNode::frame_callback, this, std::placeholders::_1)
        );

        // Timer for Control Loop (500Hz)
        // 创建 500Hz 的高频控制定时器
        timer_ = this->create_wall_timer(
            2ms, std::bind(&MimicArmNode::control_loop, this)
        );

        RCLCPP_INFO(this->get_logger(), "Mimic Node Initialized. Waiting for targets...");
    }

    /**
     * @brief 析构函数
     * 节点关闭时，禁用所有电机并停止通信。
     */
    ~MimicArmNode() {
        if (openarm_) {
            openarm_->disable_all();
            openarm_->recv_all();
        }
    }

private:
    /**
     * @brief 目标位姿回调函数
     * 接收并处理视觉系统发送的 MimicFrame 消息。
     * @param msg MimicFrame 消息指针
     */
    void frame_callback(const openarm_mimic::msg::MimicFrame::SharedPtr msg) {
        last_frame_time_ = this->now();
        
        // 检查对应侧别的数据是否有效
        bool valid = (arm_side_ == "left_arm") ? msg->left_track_valid : msg->right_track_valid;
        if (!valid) return;

        // 提取位姿和手爪数据
        geometry_msgs::msg::Pose pose = (arm_side_ == "left_arm") ? msg->left_arm_pose : msg->right_arm_pose;
        float gripper = (arm_side_ == "left_arm") ? msg->left_gripper_ratio : msg->right_gripper_ratio;

        // Update target
        // 更新内部目标变量
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

    /**
     * @brief 主控制循环 (500Hz)
     * 执行状态读取、重力补偿计算、IK解算和电机指令发送。
     */
    void control_loop() {
        try {
            if (first_loop_) {
                std::cout << "[MimicControl] Entering Control Loop (Standard Output)" << std::endl;
                RCLCPP_INFO(this->get_logger(), "Entering Control Loop");
                first_loop_ = false;
            }

            // 1. Read State (From internal memory, updated by previous recv_all)
            // 读取电机当前状态（位置和速度）
            auto motors = openarm_->get_arm().get_motors();
        
            std::vector<double> q_curr(num_joints_);
            std::vector<double> dq_curr(num_joints_);
            for(int i=0; i<num_joints_; ++i) {
                q_curr[i] = motors[i].get_position();
                dq_curr[i] = motors[i].get_velocity();
            }

            // 2. Compute Gravity
            // 计算重力补偿力矩
            std::vector<double> tau_g(num_joints_);
            dynamics_->GetGravity(q_curr.data(), tau_g.data());

            // 3. Determine Control Mode
            // 安全检查：如果超过 500ms 未收到目标，回退到纯重力补偿模式
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
                // 目标位姿平滑处理
                if (!initialized_pose_) {
                    // Initialize current_pose_ from FK of current joint angles to prevent jump
                    // 初始化：使用当前机械臂姿态作为起点，防止突变
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
                    // 速度限制：限制每周期最大位移量
                    double max_step = 0.005; // 0.5 cm per 10ms = 0.5 m/s
                    for(int i=0; i<3; ++i) {
                        double diff = target_pose_[i] - current_pose_[i];
                        // Exponential smoothing
                        // 指数平滑
                        double smooth_diff = diff * alpha_;
                        // Velocity clamp
                        if (std::abs(smooth_diff) > max_step) {
                             smooth_diff = (smooth_diff > 0 ? max_step : -max_step);
                        }
                        current_pose_[i] += smooth_diff;
                    }
                    
                    // Orientation: Fixed for now or smooth slerp if implemented
                    // 姿态目前直接赋值（建议后续增加球面插值 Slerp）
                    current_pose_[3] = target_pose_[3];
                    current_pose_[4] = target_pose_[4];
                    current_pose_[5] = target_pose_[5];
                    current_pose_[6] = target_pose_[6];
                }

                // IK Solver
                // 执行逆运动学解算
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
            // 构造并发送电机控制指令
            std::vector<openarm::damiao_motor::MITParam> cmds;
            cmds.reserve(num_joints_);
            
            for(int i=0; i<num_joints_; ++i) {
                cmds.push_back({
                    (float)kp_run, 
                    (float)kd_run, 
                    (float)q_cmd[i], 
                    0.0f, // Velocity target 0
                    (float)tau_g[i] // 前馈重力补偿
                });
            }
            openarm_->get_arm().mit_control_all(cmds);

            // Gripper Control
            // 手爪控制
            if (openarm_->get_gripper().get_motors().size() > 0) {
                 float gripper_target = target_gripper_ * 3.0f; // Approx range check needed
                 openarm_->get_gripper().mit_control_one(0, {
                     10.0f, 0.1f, gripper_target, 0.0f, 0.0f
                 });
            }

            // 5. Receive State (For NEXT loop)
            // Motors reply to the commands sent above.
            // 接收电机反馈（作为下一次循环的状态输入）
            openarm_->recv_all();

        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Control Loop Error: %s", e.what());
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
    float target_gripper_ = 0.0f;
    bool initialized_pose_ = false;
    bool first_loop_ = true;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MimicArmNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
