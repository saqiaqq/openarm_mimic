#include "openarm_mimic/robot_kdl.hpp"

/**
 * @file robot_kdl.cpp
 * @brief 机器人运动学库实现文件
 * 
 * 实现了 RobotKDL 类中定义的运动学求解功能。
 */

/**
 * @brief 构造函数实现
 */
RobotKDL::RobotKDL(const std::string& urdf_content, const std::string& base_link, const std::string& tip_link)
    : urdf_content_(urdf_content), base_link_(base_link), tip_link_(tip_link), num_joints_(0) {}

/**
 * @brief 初始化 KDL 求解器实现
 */
bool RobotKDL::init() {
    // 从 URDF 字符串构建 KDL Tree
    if (!kdl_parser::treeFromString(urdf_content_, tree_)) {
        std::cerr << "Failed to construct KDL tree from URDF string" << std::endl;
        return false;
    }

    // 从 Tree 中提取指定基座到末端的 Chain
    if (!tree_.getChain(base_link_, tip_link_, chain_)) {
        std::cerr << "Failed to get KDL chain from " << base_link_ << " to " << tip_link_ << std::endl;
        return false;
    }

    num_joints_ = chain_.getNrOfJoints();
    
    // 初始化正运动学求解器
    fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
    
    // 初始化逆运动学求解器 (LMA: Levenberg-Marquardt Algorithm)
    // LMA solver parameters: epsilon=1e-5, max_iter=500, eps_joints=1e-15
    Eigen::Matrix<double, 6, 1> L;
    L.fill(1.0); // 权重设置：x,y,z,rx,ry,rz 权重相等
    ik_solver_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(chain_, L);

    return true;
}

/**
 * @brief 逆向运动学求解实现
 */
bool RobotKDL::solveIK(const std::vector<double>& q_init, 
                       const std::vector<double>& target_pose, 
                       std::vector<double>& q_out) {
    if (q_init.size() != (size_t)num_joints_) return false;
    
    // 将 std::vector 转换为 KDL::JntArray
    KDL::JntArray q_in_kdl(num_joints_);
    for(int i=0; i<num_joints_; ++i) q_in_kdl(i) = q_init[i];

    // 构建目标 KDL::Frame
    KDL::Frame target_frame;
    target_frame.p = KDL::Vector(target_pose[0], target_pose[1], target_pose[2]);
    target_frame.M = KDL::Rotation::Quaternion(target_pose[3], target_pose[4], target_pose[5], target_pose[6]);

    KDL::JntArray q_out_kdl(num_joints_);
    
    // 执行 IK 计算
    int ret = ik_solver_->CartToJnt(q_in_kdl, target_frame, q_out_kdl);

    if (ret >= 0) {
        // 转换结果回 std::vector
        q_out.resize(num_joints_);
        for(int i=0; i<num_joints_; ++i) q_out[i] = q_out_kdl(i);
        return true;
    } else {
        // std::cerr << "IK Solver failed with error: " << ret << std::endl;
        return false;
    }
}

/**
 * @brief 正向运动学求解实现
 */
bool RobotKDL::solveFK(const std::vector<double>& q_in, std::vector<double>& pose_out) {
    if (q_in.size() != (size_t)num_joints_) return false;

    // 将 std::vector 转换为 KDL::JntArray
    KDL::JntArray q_kdl(num_joints_);
    for(int i=0; i<num_joints_; ++i) q_kdl(i) = q_in[i];

    KDL::Frame p_out;
    // 执行 FK 计算
    if (fk_solver_->JntToCart(q_kdl, p_out) < 0) return false;

    // 填充输出位姿 [x, y, z, qx, qy, qz, qw]
    pose_out.resize(7);
    pose_out[0] = p_out.p.x();
    pose_out[1] = p_out.p.y();
    pose_out[2] = p_out.p.z();
    
    double x, y, z, w;
    p_out.M.GetQuaternion(x, y, z, w);
    pose_out[3] = x;
    pose_out[4] = y;
    pose_out[5] = z;
    pose_out[6] = w;

    return true;
}
