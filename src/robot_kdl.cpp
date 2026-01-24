#include "openarm_mimic/robot_kdl.hpp"

RobotKDL::RobotKDL(const std::string& urdf_content, const std::string& base_link, const std::string& tip_link)
    : urdf_content_(urdf_content), base_link_(base_link), tip_link_(tip_link), num_joints_(0) {}

bool RobotKDL::init() {
    if (!kdl_parser::treeFromString(urdf_content_, tree_)) {
        std::cerr << "Failed to construct KDL tree from URDF string" << std::endl;
        return false;
    }

    if (!tree_.getChain(base_link_, tip_link_, chain_)) {
        std::cerr << "Failed to get KDL chain from " << base_link_ << " to " << tip_link_ << std::endl;
        return false;
    }

    num_joints_ = chain_.getNrOfJoints();
    
    fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
    
    // LMA solver parameters: epsilon=1e-5, max_iter=500, eps_joints=1e-15
    Eigen::Matrix<double, 6, 1> L;
    L.fill(1.0); // Weights for x,y,z,rx,ry,rz
    ik_solver_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(chain_, L);

    return true;
}

bool RobotKDL::solveIK(const std::vector<double>& q_init, 
                       const std::vector<double>& target_pose, 
                       std::vector<double>& q_out) {
    if (q_init.size() != (size_t)num_joints_) return false;
    
    KDL::JntArray q_in_kdl(num_joints_);
    for(int i=0; i<num_joints_; ++i) q_in_kdl(i) = q_init[i];

    KDL::Frame target_frame;
    target_frame.p = KDL::Vector(target_pose[0], target_pose[1], target_pose[2]);
    target_frame.M = KDL::Rotation::Quaternion(target_pose[3], target_pose[4], target_pose[5], target_pose[6]);

    KDL::JntArray q_out_kdl(num_joints_);
    
    // Calculate IK
    int ret = ik_solver_->CartToJnt(q_in_kdl, target_frame, q_out_kdl);

    if (ret >= 0) {
        q_out.resize(num_joints_);
        for(int i=0; i<num_joints_; ++i) q_out[i] = q_out_kdl(i);
        return true;
    } else {
        // std::cerr << "IK Solver failed with error: " << ret << std::endl;
        return false;
    }
}

bool RobotKDL::solveFK(const std::vector<double>& q_in, std::vector<double>& pose_out) {
    if (q_in.size() != (size_t)num_joints_) return false;

    KDL::JntArray q_kdl(num_joints_);
    for(int i=0; i<num_joints_; ++i) q_kdl(i) = q_in[i];

    KDL::Frame p_out;
    if (fk_solver_->JntToCart(q_kdl, p_out) < 0) return false;

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
